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
    val continuous: Boolean = true,    // caption while the hands stay up; off: one answer per sign as the hands rest
    val caution: Int = 1,              // continuous captions: 0 eager, 1 balanced (the measured setting), 2 careful
    val threshold: Float = 0.15f,      // captions at rest: a sign's word must reach this confidence (0.15 won on val streams)
    val agree: Int = 2,                // captions at rest, hands up for over 4 s: a word must win this many windows in a row
    val windowSec: Float = 1.5f,       // ... each covering this many seconds (1.5 s won on val streams)
    val showLandmarks: Boolean = true,
    val backCamera: Boolean = true,    // exocentric: the camera faces the signer
    val gpu: Boolean = false,          // MediaPipe on the GPU delegate; CPU lanes are faster (bench, 13 Nov)
    val speak: Boolean = false,        // say each caption word aloud (on-device voice)
    val speechRate: Float = 1.15f,     // a little faster than the engine's default keeps up with signing
    val teachThreshold: Float = 0.55f, // cosine a window needs to a taught sign's prototype
    val showDetails: Boolean = false,  // best guesses and timings over the live video (for demos and debugging)
    val onboarded: Boolean = false,    // first-launch setup finished
    val developer: Boolean = false,    // developer options unlocked (tap the version seven times)
    val keepDays: Int = 30,            // conversations older than this are deleted; KEEP_FOREVER keeps them
    val glossCaps: Boolean = false,    // captions as glosses in capitals (HELLO) instead of words (hello)
    val vibrate: Boolean = true,       // short haptics: a word appears, a correction, recording, a sign learned
    val skeleton: Boolean = false,     // developer: draw the tracked hand skeleton, not just the glow
) {
    companion object { const val KEEP_FOREVER = 0 }
}

private val Context.store by preferencesDataStore(name = "settings")

class SettingsRepository(private val context: Context) {
    private object K {
        val threshold = floatPreferencesKey("caption_confidence")   // new key: the old one held the sliding window's 0.45
        val continuous = booleanPreferencesKey("caption_continuous")
        val caution = intPreferencesKey("caption_caution")
        val agree = intPreferencesKey("agree")
        val window = floatPreferencesKey("window_sec")
        val landmarks = booleanPreferencesKey("show_landmarks")
        val back = booleanPreferencesKey("back_camera")
        val gpu = booleanPreferencesKey("gpu_tracking")   // new key: the default changed from GPU to CPU
        val speak = booleanPreferencesKey("speak")
        val speechRate = floatPreferencesKey("speech_rate")
        val teachThreshold = floatPreferencesKey("teach_threshold")
        val details = booleanPreferencesKey("show_details")
        val onboarded = booleanPreferencesKey("onboarded")
        val developer = booleanPreferencesKey("developer")
        val keepDays = intPreferencesKey("keep_days")
        val glossCaps = booleanPreferencesKey("gloss_caps")
        val vibrate = booleanPreferencesKey("vibrate")
        val skeleton = booleanPreferencesKey("skeleton")
    }

    private fun read(p: Preferences) = Settings(
        threshold = p[K.threshold] ?: Settings().threshold,
        continuous = p[K.continuous] ?: Settings().continuous,
        caution = p[K.caution] ?: Settings().caution,
        agree = p[K.agree] ?: Settings().agree,
        windowSec = p[K.window] ?: Settings().windowSec,
        showLandmarks = p[K.landmarks] ?: Settings().showLandmarks,
        backCamera = p[K.back] ?: Settings().backCamera,
        gpu = p[K.gpu] ?: Settings().gpu,
        speak = p[K.speak] ?: Settings().speak,
        speechRate = p[K.speechRate] ?: Settings().speechRate,
        teachThreshold = p[K.teachThreshold] ?: Settings().teachThreshold,
        showDetails = p[K.details] ?: Settings().showDetails,
        onboarded = p[K.onboarded] ?: Settings().onboarded,
        developer = p[K.developer] ?: Settings().developer,
        keepDays = p[K.keepDays] ?: Settings().keepDays,
        glossCaps = p[K.glossCaps] ?: Settings().glossCaps,
        vibrate = p[K.vibrate] ?: Settings().vibrate,
        skeleton = p[K.skeleton] ?: Settings().skeleton,
    )

    val settings: Flow<Settings> = context.store.data.map(::read)

    suspend fun update(change: (Settings) -> Settings) {
        context.store.edit { p ->
            val s = change(read(p))
            p[K.threshold] = s.threshold
            p[K.continuous] = s.continuous
            p[K.caution] = s.caution
            p[K.agree] = s.agree
            p[K.window] = s.windowSec
            p[K.landmarks] = s.showLandmarks
            p[K.back] = s.backCamera
            p[K.gpu] = s.gpu
            p[K.speak] = s.speak
            p[K.speechRate] = s.speechRate
            p[K.teachThreshold] = s.teachThreshold
            p[K.details] = s.showDetails
            p[K.onboarded] = s.onboarded
            p[K.developer] = s.developer
            p[K.keepDays] = s.keepDays
            p[K.glossCaps] = s.glossCaps
            p[K.vibrate] = s.vibrate
            p[K.skeleton] = s.skeleton
        }
    }
}
