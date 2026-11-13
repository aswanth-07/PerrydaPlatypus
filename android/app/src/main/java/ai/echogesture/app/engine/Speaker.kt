package ai.echogesture.app.engine

import android.content.Context
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.speech.tts.Voice
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

data class VoiceState(
    val ready: Boolean = false,
    val onDevice: Boolean = false,     // the chosen voice needs no network
    val voice: String? = null,
    val lastMs: Double = 0.0,          // camera frame that settled the word -> first sound
    val medianMs: Double = 0.0,
    val spoken: Int = 0,
    val error: String? = null,
)

/**
 * Speaks each caption word with the phone's own text-to-speech engine, using a voice that is installed on the
 * device, so sound never needs the network (the app has no internet permission either).
 *
 * Built for delay, not for prose: the engine is warmed up when it starts, the fastest installed voice is chosen,
 * and when two words are already waiting a new word replaces them, so speech never trails the signer. Each word
 * is timed from the camera frame that settled it to the moment the engine starts to sound.
 */
class Speaker(context: Context) : AutoCloseable {
    private val _state = MutableStateFlow(VoiceState())
    val state: StateFlow<VoiceState> = _state.asStateFlow()

    @Volatile var enabled = false
    @Volatile var hold = false     // true while the microphone listens, so the phone does not hear itself
    @Volatile private var ready = false
    private val settledAt = ConcurrentHashMap<String, Long>()   // utterance id -> uptime of the frame that settled the word
    private val waiting = AtomicInteger(0)
    private val delays = ArrayDeque<Double>()
    private var next = 0L
    private val tts: TextToSpeech = TextToSpeech(context.applicationContext) { status -> onInit(status) }

    private fun onInit(status: Int) {
        if (status != TextToSpeech.SUCCESS) {
            _state.update { it.copy(error = "No text-to-speech engine on this phone") }
            return
        }
        tts.language = Locale.US
        // An installed English voice, US first, then the one the engine says answers fastest.
        val voice = runCatching { tts.voices }.getOrNull().orEmpty()
            .filter { !it.isNetworkConnectionRequired && it.locale.language == "en" && NOT_INSTALLED !in it.features }
            .sortedWith(compareBy<Voice>({ if (it.locale.country == "US") 0 else 1 }, { it.latency }, { -it.quality }))
            .firstOrNull()
        voice?.let { tts.voice = it }
        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String) {
                settledAt.remove(utteranceId)?.let { record((SystemClock.uptimeMillis() - it).toDouble()) }
            }

            override fun onDone(utteranceId: String) = left(utteranceId)
            override fun onStop(utteranceId: String, interrupted: Boolean) = left(utteranceId)
            @Deprecated("Deprecated in the platform; still called by older engines")
            override fun onError(utteranceId: String) = left(utteranceId)
            override fun onError(utteranceId: String, errorCode: Int) = left(utteranceId)
        })
        tts.playSilentUtterance(1, TextToSpeech.QUEUE_ADD, "warm")   // the first real word should not pay the engine's start-up
        ready = true
        _state.update { it.copy(ready = true, onDevice = voice != null, voice = (voice ?: tts.voice)?.name) }
    }

    private fun left(id: String) {
        settledAt.remove(id)
        if (id != "warm") waiting.updateAndGet { maxOf(it - 1, 0) }
    }

    private fun record(ms: Double) {
        synchronized(delays) {
            delays.addLast(ms)
            while (delays.size > 25) delays.removeFirst()
            val median = delays.sorted()[delays.size / 2]
            _state.update { it.copy(lastMs = ms, medianMs = median, spoken = it.spoken + 1) }
        }
    }

    fun setRate(rate: Float) {
        if (ready) tts.setSpeechRate(rate)
    }

    /** [frameAtMs]: SystemClock.uptimeMillis() of the camera frame whose result settled this word. */
    fun speak(word: String, frameAtMs: Long = SystemClock.uptimeMillis()) {
        if (!ready || !enabled || hold) return
        val id = "w${next++}"
        val mode = if (waiting.get() >= 2) {
            waiting.set(0)
            settledAt.clear()
            TextToSpeech.QUEUE_FLUSH
        } else TextToSpeech.QUEUE_ADD
        settledAt[id] = frameAtMs
        waiting.incrementAndGet()
        tts.speak(sayable(word), mode, null, id)
    }

    fun silence() {
        if (ready) tts.stop()
        waiting.set(0)
        settledAt.clear()
    }

    override fun close() {
        runCatching { tts.stop(); tts.shutdown() }
    }

    companion object {
        private const val NOT_INSTALLED = "notInstalled"   // TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED

        /** A label as a person would say it: "THANK YOU" -> "thank you", "BAT3" -> "bat", "DOCTOR · PHYSICIAN" -> "doctor". */
        fun sayable(label: String): String =
            label.substringBefore('·').trim().lowercase(Locale.US).replace('_', ' ').replace(Regex("\\d+$"), "").ifBlank { label }
    }
}
