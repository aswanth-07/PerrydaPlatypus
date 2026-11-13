package ai.echogesture.app.engine

import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

/** The hearing person's side of the conversation: what the microphone is doing and what it heard. */
data class ReplyState(
    val available: Boolean = false,
    val listening: Boolean = false,
    val partial: String = "",          // words heard so far, while listening
    val last: String? = null,          // the last finished reply, shown large for the signer
    val lastAt: Long = 0,
    val note: String? = null,          // why replies are unavailable, or what went wrong
)

/**
 * Two-way reply: the hearing person speaks, the signer reads. Uses Android's on-device speech recogniser only
 * (SpeechRecognizer.createOnDeviceSpeechRecognizer, Android 12+), so audio never leaves the phone; the app has no
 * internet permission either. All calls must come from the main thread, as SpeechRecognizer requires.
 */
class Listener(
    private val context: Context,
    private val onListening: (Boolean) -> Unit,
    private val onReply: (String) -> Unit,
) : AutoCloseable {
    private val _state = MutableStateFlow(ReplyState())
    val state: StateFlow<ReplyState> = _state.asStateFlow()
    private var recognizer: SpeechRecognizer? = null
    private var lastIntent: Intent? = null
    private var askedDownload = false

    init {
        refresh()
    }

    fun refresh() {
        val ok = Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && SpeechRecognizer.isOnDeviceRecognitionAvailable(context)
        _state.update {
            it.copy(
                available = ok,
                note = when {
                    ok -> null
                    Build.VERSION.SDK_INT < Build.VERSION_CODES.S -> "Spoken replies need Android 12 or later (on-device speech recognition)."
                    else -> "This phone has no on-device speech recogniser. Install or update Google's speech services to turn replies on."
                },
            )
        }
    }

    fun start() {
        if (!_state.value.available || _state.value.listening) return
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) return
        val r = recognizer ?: SpeechRecognizer.createOnDeviceSpeechRecognizer(context).also {
            it.setRecognitionListener(callbacks)
            recognizer = it
        }
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
            .putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            .putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
            .putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            .putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
        lastIntent = intent
        _state.update { it.copy(listening = true, partial = "", note = null) }
        onListening(true)
        r.startListening(intent)
    }

    /** Stop listening and keep what was heard. */
    fun stop() {
        recognizer?.stopListening()
    }

    fun dismiss() = _state.update { it.copy(last = null) }

    private val callbacks = object : RecognitionListener {
        override fun onReadyForSpeech(params: Bundle?) {}
        override fun onBeginningOfSpeech() {}
        override fun onRmsChanged(rmsdB: Float) {}
        override fun onBufferReceived(buffer: ByteArray?) {}
        override fun onEndOfSpeech() {}
        override fun onEvent(eventType: Int, params: Bundle?) {}

        override fun onPartialResults(partialResults: Bundle?) {
            val text = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull().orEmpty()
            if (text.isNotBlank()) _state.update { it.copy(partial = text) }
        }

        override fun onResults(results: Bundle?) {
            val text = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.trim().orEmpty()
            onListening(false)
            _state.update {
                if (text.isEmpty()) it.copy(listening = false, partial = "", note = "Didn't catch that. Tap the microphone and speak again.")
                else it.copy(listening = false, partial = "", last = text, lastAt = System.currentTimeMillis())
            }
            if (text.isNotEmpty()) onReply(text)
        }

        override fun onError(error: Int) {
            val msg = when (error) {
                SpeechRecognizer.ERROR_NO_MATCH, SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Didn't catch that. Tap the microphone and speak again."
                SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Allow the microphone to hear spoken replies."
                SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED, SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE ->
                    if (requestDownload()) "Getting English speech recognition for offline use. Try Reply again in a minute."
                    else "English speech recognition is not downloaded on this phone. Add it in the phone's speech settings."
                SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "The speech recogniser is busy. Try again in a moment."
                else -> "Couldn't listen (error $error). Try again."
            }
            onListening(false)
            _state.update { it.copy(listening = false, partial = "", note = msg) }
        }
    }

    /** Android 13+ can fetch the on-device language pack on request; the system downloads it, not this app. */
    private fun requestDownload(): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU || askedDownload) return false
        val r = recognizer ?: return false
        val i = lastIntent ?: return false
        askedDownload = true
        return runCatching { r.triggerModelDownload(i) }.isSuccess
    }

    override fun close() {
        recognizer?.destroy()
        recognizer = null
    }
}
