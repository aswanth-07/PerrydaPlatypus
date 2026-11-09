package ai.echogesture.app.data

import android.content.Context
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

data class Settings(
    val threshold: Float = 0.45f,      // a word must reach this confidence
    val agree: Int = 2,                // ... and win this many windows in a row
    val windowSec: Float = 2.0f,       // seconds of signing looked at each time
    val showLandmarks: Boolean = true,
    val backCamera: Boolean = true,    // exocentric: the camera faces the signer
    val gpu: Boolean = true,           // MediaPipe on the GPU delegate
    val vocabulary: String = "core",   // "core" 100 words or "full" 2,000 words
)

private val Context.store by preferencesDataStore(name = "settings")

class SettingsRepository(private val context: Context) {
    private object K {
        val threshold = floatPreferencesKey("threshold")
        val agree = intPreferencesKey("agree")
        val window = floatPreferencesKey("window_sec")
        val landmarks = booleanPreferencesKey("show_landmarks")
        val back = booleanPreferencesKey("back_camera")
        val gpu = booleanPreferencesKey("gpu")
        val vocabulary = stringPreferencesKey("vocabulary")
    }

    private fun read(p: Preferences) = Settings(
        threshold = p[K.threshold] ?: Settings().threshold,
        agree = p[K.agree] ?: Settings().agree,
        windowSec = p[K.window] ?: Settings().windowSec,
        showLandmarks = p[K.landmarks] ?: Settings().showLandmarks,
        backCamera = p[K.back] ?: Settings().backCamera,
        gpu = p[K.gpu] ?: Settings().gpu,
        vocabulary = p[K.vocabulary] ?: Settings().vocabulary,
    )

    val settings: Flow<Settings> = context.store.data.map(::read)

    suspend fun update(change: (Settings) -> Settings) {
        context.store.edit { p ->
            val s = change(read(p))
            p[K.threshold] = s.threshold
            p[K.agree] = s.agree
            p[K.window] = s.windowSec
            p[K.landmarks] = s.showLandmarks
            p[K.back] = s.backCamera
            p[K.gpu] = s.gpu
            p[K.vocabulary] = s.vocabulary
        }
    }
}
