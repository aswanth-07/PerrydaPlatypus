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
    val windowSec: Float = 1.5f,       // seconds of signing looked at each time (1.5 s won on val streams for both vocabularies)
    val showLandmarks: Boolean = true,
    val backCamera: Boolean = true,    // exocentric: the camera faces the signer
    val gpu: Boolean = false,          // MediaPipe on the GPU delegate; CPU lanes are faster (bench, 13 Nov)
    val vocabulary: String = "full",   // "full" 2,000 words (default), "core" 100 words, "union" 3,075
    val speak: Boolean = false,        // say each caption word aloud (on-device voice)
    val speechRate: Float = 1.15f,     // a little faster than the engine's default keeps up with signing
    val teachThreshold: Float = 0.55f, // cosine a window needs to a taught sign's prototype
    val showDetails: Boolean = false,  // best guesses and timings over the live video (for demos and debugging)
)

private val Context.store by preferencesDataStore(name = "settings")

class SettingsRepository(private val context: Context) {
    private object K {
        val threshold = floatPreferencesKey("threshold")
        val agree = intPreferencesKey("agree")
        val window = floatPreferencesKey("window_sec")
        val landmarks = booleanPreferencesKey("show_landmarks")
        val back = booleanPreferencesKey("back_camera")
        val gpu = booleanPreferencesKey("gpu_tracking")   // new key: the default changed from GPU to CPU
        val vocabulary = stringPreferencesKey("vocabulary")
        val speak = booleanPreferencesKey("speak")
        val speechRate = floatPreferencesKey("speech_rate")
        val teachThreshold = floatPreferencesKey("teach_threshold")
        val details = booleanPreferencesKey("show_details")
    }

    private fun read(p: Preferences) = Settings(
        threshold = p[K.threshold] ?: Settings().threshold,
        agree = p[K.agree] ?: Settings().agree,
        windowSec = p[K.window] ?: Settings().windowSec,
        showLandmarks = p[K.landmarks] ?: Settings().showLandmarks,
        backCamera = p[K.back] ?: Settings().backCamera,
        gpu = p[K.gpu] ?: Settings().gpu,
        vocabulary = p[K.vocabulary] ?: Settings().vocabulary,
        speak = p[K.speak] ?: Settings().speak,
        speechRate = p[K.speechRate] ?: Settings().speechRate,
        teachThreshold = p[K.teachThreshold] ?: Settings().teachThreshold,
        showDetails = p[K.details] ?: Settings().showDetails,
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
            p[K.speak] = s.speak
            p[K.speechRate] = s.speechRate
            p[K.teachThreshold] = s.teachThreshold
            p[K.details] = s.showDetails
        }
    }
}
