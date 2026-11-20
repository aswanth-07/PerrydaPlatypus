package ai.echogesture.app

import android.app.Application
import android.net.Uri
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import ai.echogesture.app.data.Progress
import ai.echogesture.app.data.ProgressRepository
import ai.echogesture.app.data.Settings
import ai.echogesture.app.data.SettingsRepository
import ai.echogesture.app.engine.ClipResult
import ai.echogesture.app.engine.SignEngine
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class ClipProgress(val fraction: Float, val stage: String, val error: String? = null)

class MainViewModel(app: Application) : AndroidViewModel(app) {
    val engine = SignEngine(app)
    private val repo = SettingsRepository(app)
    private val progressRepo = ProgressRepository(app)

    val settings: StateFlow<Settings> = repo.settings.stateIn(viewModelScope, SharingStarted.Eagerly, Settings())
    /** False until the saved settings are read, so the first screen is not chosen from the defaults. */
    val settingsLoaded: StateFlow<Boolean> = repo.settings.map { true }.stateIn(viewModelScope, SharingStarted.Eagerly, false)
    /** Signs learned and days practised, kept across restarts. */
    val progress: StateFlow<Progress> = progressRepo.progress.stateIn(viewModelScope, SharingStarted.Eagerly, Progress())

    fun markLearned(word: String) {
        viewModelScope.launch { progressRepo.markLearned(word) }
    }

    fun resetProgress() {
        viewModelScope.launch { progressRepo.reset() }
    }
    val live = engine.liveState
    val overlay = engine.overlayState
    val transcript = engine.transcript

    private val _clips = MutableStateFlow<List<ClipResult>>(emptyList())
    val clips: StateFlow<List<ClipResult>> = _clips.asStateFlow()
    private val _clipProgress = MutableStateFlow<ClipProgress?>(null)
    val clipProgress: StateFlow<ClipProgress?> = _clipProgress.asStateFlow()

    init {
        viewModelScope.launch {
            var prev: Settings? = null
            // Collect persisted DataStore values directly so default Settings() never prunes history or churns trackers.
            repo.settings.collect { s ->
                val p = prev
                if (p == null || s.keepDays != p.keepDays) {
                    engine.setRetention(s.keepDays.takeIf { it != Settings.KEEP_FOREVER })
                }
                if (p == null) {
                    engine.load()
                }
                if (p == null || s.threshold != p.threshold || s.agree != p.agree || s.windowSec != p.windowSec ||
                    s.gpu != p.gpu || s.continuous != p.continuous || s.caution != p.caution
                ) {
                    engine.configure(s.threshold, s.agree, (s.windowSec * 1000).toLong(), s.gpu, s.continuous, s.caution)
                }
                if (p == null || s.speak != p.speak || s.speechRate != p.speechRate) {
                    engine.setSpeech(s.speak, s.speechRate)
                }
                if (p == null || s.teachThreshold != p.teachThreshold) {
                    engine.setTeachThreshold(s.teachThreshold)
                }
                prev = s
            }
        }
    }

    fun update(change: (Settings) -> Settings) {
        viewModelScope.launch { repo.update(change) }
    }

    fun recognise(uri: Uri) {
        if (_clipProgress.value?.error == null && _clipProgress.value != null) return
        val name = displayName(uri)
        _clipProgress.value = ClipProgress(0f, "Opening $name")
        viewModelScope.launch {
            try {
                engine.load()
                val r = engine.recogniseClip(uri, name) { f, stage -> _clipProgress.value = ClipProgress(f, stage) }
                _clips.update { listOf(r) + it }
                _clipProgress.value = null
            } catch (e: Exception) {
                _clipProgress.value = ClipProgress(0f, "Failed", e.message ?: e.javaClass.simpleName)
            }
        }
    }

    fun dismissClipError() { _clipProgress.value = null }

    fun clearClips() = _clips.update { emptyList() }

    private fun displayName(uri: Uri): String =
        getApplication<Application>().contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        } ?: "video"

    override fun onCleared() {
        engine.close()
    }
}
