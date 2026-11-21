package ai.echogesture.app.engine

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import ai.echogesture.app.data.ConversationStore
import ai.echogesture.app.data.Conversations
import ai.echogesture.app.data.TaughtSign
import ai.echogesture.app.data.TaughtStore
import ai.echogesture.signcore.Clip
import ai.echogesture.signcore.Cores
import ai.echogesture.signcore.Embeddings
import ai.echogesture.signcore.Layout
import ai.echogesture.signcore.Normaliser
import ai.echogesture.signcore.Prediction
import ai.echogesture.signcore.RawFrame
import ai.echogesture.signcore.Resampler
import ai.echogesture.signcore.SameSignSets
import ai.echogesture.signcore.SignModel
import ai.echogesture.signcore.Spotter
import ai.echogesture.signcore.TaughtMatcher
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarker
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.FlowPreview
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.debounce
import kotlinx.coroutines.flow.drop
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit

data class ModelInfo(
    val words: Int,
    val testTop1: Double,
    val testTop5: Double,
    val testClips: Int,
    val ensemble: Int,
    val classifierMb: Double,
)

/**
 * High-frequency hand overlay and live timings, separated from [LiveState] so 30 Hz camera updates only
 * recompose the hand Canvas (and Developer details when visible), not the whole Talk/Learn/Teach screen.
 */
data class OverlayState(
    val hands: List<FloatArray?> = listOf(null, null),
    val imageWidth: Int = 0,
    val imageHeight: Int = 0,
    val fps: Double = 0.0,
    val landmarksMs: Double = 0.0,
    val modelMs: Double = 0.0,
)

data class LiveState(
    val ready: Boolean = false,
    val running: Boolean = false,
    val paused: Boolean = false,
    val handsVisible: Boolean = false,
    val signing: Boolean = false,               // a sign is under way: the hands are up, its word comes when they rest
    val bodyVisible: Boolean = false,           // Holistic finds hands from the body: no body, no hands
    val top: List<Pair<String, Float>> = emptyList(),
    val caption: List<String> = emptyList(),
    val pending: String? = null,                // the sign under way, not yet settled (continuous captions)
    val pendingProgress: Float = 0f,            // 0..1 progress of the forming sign toward locking in
    val taughtMatch: Pair<String, Float>? = null, // live closest taught sign + cosine score (for Teach Try-it meter)
    val attempt: Attempt? = null,               // Learn mode: the last sign made while a word was being practised
    val lastWord: String? = null,
    val lastWordAt: Long = 0,
    val lastAlso: List<String> = emptyList(),   // other names of the same sign (DOCTOR · PHYSICIAN)
    val lastTaught: Boolean = false,            // the last word is a sign taught on this phone
    val lastAlternatives: List<String> = emptyList(),   // other signs it could have been; a tap swaps one in
    val recording: Float? = null,               // 0..1 while an example for a taught sign is being recorded
    val gpu: Boolean = true,
    val lanes: Int = 1,                         // live landmarkers taking turns on frames
    val error: String? = null,
    val undo: UndoOffer? = null,                // the person just removed a word or cleared the captions
)

/** What the person can take back: the word they removed from the caption, or all the captions they cleared. */
data class UndoOffer(val kind: Kind, val word: String?, val at: Long) {
    enum class Kind { REMOVED_WORD, CLEARED }
}

/**
 * Learn mode: one sign the learner made while practising [target]. [share] is how much of the sign's evidence was
 * for the target (1 when the target was the caption); [correct] when at least [SignEngine.PRACTICE_PASS] of it.
 */
data class Attempt(val target: String, val seen: String, val share: Float, val correct: Boolean, val at: Long)

/** Who said a line of the conversation: the signer (recognised signs) or the hearing person (spoken reply). */
enum class Who { SIGNER, VOICE }

data class TranscriptEntry(val word: String, val confidence: Float, val atMillis: Long, val who: Who = Who.SIGNER)

/** Teach-a-sign state: whether the loaded model can do it, and the signs taught so far. */
data class TeachState(val supported: Boolean = false, val signs: List<TaughtSign> = emptyList(), val note: String? = null)

/** One recorded example: the model input (64 steps), the share of frames with a hand, and its embedding. */
class TeachExample(val clip: Clip, val handFraction: Float, val embedding: FloatArray?)

data class ClipResult(
    val name: String,
    val top5: List<Pair<String, Float>>,
    val frames: Int,
    val landmarksMs: Double,          // hand tracking, summed over the frames
    val modelMs: Double,
    val totalMs: Double,
    val decodeMs: Double = 0.0,       // reading the frames (runs alongside tracking)
    val atMillis: Long = System.currentTimeMillis(),
    val also: List<String> = emptyList(),
    val taught: Boolean = false,
) {
    val word: String get() = top5.first().first
    val confidence: Float get() = top5.first().second
}

/**
 * Everything that runs on the device: MediaPipe Holistic, the signcore normaliser and the INT8 classifier.
 * Live frames come from CameraX; clips come from a video the user picks. Nothing is sent anywhere.
 */
class SignEngine(private val context: Context) : AutoCloseable {
    @Volatile private var model: SignModel? = null
    private val retired = java.util.concurrent.CopyOnWriteArrayList<SignModel>()
    private var spotter: Spotter? = null
    /**
     * One live Holistic landmarker. Holistic costs about 65 ms a frame with a signer in view, so one landmarker
     * tops out near 15 fps; several lanes take turns on camera frames, run at the same time, and their results
     * are put back in frame order before the spotter sees them. Each lane still tracks (its own previous frame
     * guides its next crop), and the landmarks are the same model's, so the classifier sees what it was trained on.
     */
    private class Lane(val gpu: Boolean) {
        lateinit var landmarker: HolisticLandmarker
        val lock = Any()               // guards detectAsync against concurrent close() in stopLive()
        @Volatile var closed = false
        @Volatile var startedNs = 0L   // 0: idle
        @Volatile var inFlightTs = 0L  // timestamp currently owned by this lane
    }

    private fun newLane(gpu: Boolean): Lane = Lane(gpu).also { lane ->
        lane.landmarker = createLandmarker(
            RunningMode.LIVE_STREAM,
            gpu,
            onResult = { r, img -> onLaneResult(lane, r, img) },
            onError = { onLaneError(lane) },
        )
    }
    @Volatile private var lanes: List<Lane> = emptyList()
    private val live: Any? get() = lanes.firstOrNull()
    private val worker = Executors.newSingleThreadExecutor()   // the spotter is used from this thread only
    private class Tracked(val frame: RawFrame, val w: Int, val h: Int, val landmarksMs: Double)
    private val order = Any()                                  // guards the four fields below
    private val inFlight = java.util.TreeMap<Long, Long>()     // frame timestamp -> nanoTime it was sent
    private val ready = java.util.TreeMap<Long, Tracked>()     // results waiting for an earlier frame
    private var lastTs = 0L
    private var lastResultTs = 0L
    private var nextLane = 0
    @Volatile private var useGpu = false   // as Settings' default, so the first start builds the right trackers
    @Volatile private var laneCount = DEFAULT_LANES
    @Volatile private var benchLanes: List<Boolean>? = null    // benchmark override: one delegate per lane

    /** Spoken captions: every word the spotter settles is said aloud by the phone's own installed voice. */
    val speaker = Speaker(context)

    /** Spoken replies from the hearing person, recognised on the device and shown as text for the signer. */
    val listener = Listener(
        context,
        onListening = { on -> speaker.hold = on; if (on) speaker.silence() },
        onReply = { text -> _transcript.update { it + TranscriptEntry(text, 1f, System.currentTimeMillis(), Who.VOICE) } },
    )

    fun setSpeech(on: Boolean, rate: Float) {
        speaker.enabled = on
        speaker.setRate(rate)
        if (!on) speaker.silence()
    }

    // ---------- benchmark hooks (engine/Bench.kt) ----------

    /** Latency samples collected while a benchmark runs; null otherwise. */
    class BenchStats {
        val landmarks = ArrayList<Double>()
        val model = ArrayList<Double>()
        var results = 0L   // since the last report
        var total = 0L     // since the start
    }
    @Volatile var bench: BenchStats? = null

    /** Benchmark only: the delegate the Videos tab's landmarker uses. */
    fun clipDelegateForBench(gpu: Boolean) { benchClipGpu = gpu }
    @Volatile private var benchClipGpu: Boolean? = null

    /** Benchmark only: the delegate of each live lane (true: GPU), fixed before the live landmarkers start. */
    fun lanesForBench(plan: List<Boolean>) { benchLanes = plan }

    /** Benchmark only: the spotter's caption mode, whatever Settings says (true: one answer per sign at rest). */
    fun boundaryForBench(on: Boolean) {
        benchBoundary = on
        worker.submit { spotter?.boundary = on }.get()
    }
    @Volatile private var benchBoundary: Boolean? = null

    /** Benchmark only: continuous captions (one per sign while the hands stay up) on or off, whatever Settings says. */
    fun continuousForBench(on: Boolean) {
        benchContinuous = on
        worker.submit { spotter?.continuous = on }.get()
    }
    @Volatile private var benchContinuous: Boolean? = null

    // ---------- Learn mode: the app shows a sign, the learner copies it, the recogniser checks it ----------

    /** The word being practised; null outside Learn mode. Captions made meanwhile are not spoken or kept in Chat. */
    @Volatile var practiceTarget: String? = null
        set(value) {
            field = value?.let(TaughtStore::clean)
            _live.update { it.copy(attempt = null) }
        }

    /** True while Teach's "Try it" step tests a new sign: updates liveState without speaking or logging to Chat. */
    @Volatile var sandboxMode: Boolean = false

    // ---------- continuous signing: the live model and its measured decoder settings ----------

    /** The live-caption model, trained on windows of continuous signing (assets/echogesture_live_int8.onnx), if packed. */
    @Volatile private var liveModel: SignModel? = null
    @Volatile private var liveMatcher: TaughtMatcher? = null
    /** The evidence a caption needs, as measured for the loaded live model (live.json); Settings' caution scales both. */
    @Volatile private var liveEmin = 2.5f
    @Volatile private var liveCommit = 6f
    @Volatile private var continuousOn = true
    @Volatile private var caution = 1

    /** Decoder settings measured for the live model on held-out streams (scripts/live_cont.py). */
    private fun applyLive(s: Spotter, cfg: JSONObject?) {
        cfg ?: return
        s.liveWindowMs = cfg.optLong("window_ms", s.liveWindowMs)
        s.liveStrideMs = cfg.optLong("stride_ms", s.liveStrideMs)
        s.lo = cfg.optDouble("lo", s.lo.toDouble()).toFloat()
        s.gap = cfg.optInt("gap", s.gap)
        s.tailMs = cfg.optLong("tail_ms", s.tailMs)
        liveEmin = cfg.optDouble("emin", liveEmin.toDouble()).toFloat()
        liveCommit = cfg.optDouble("commit", liveCommit.toDouble()).toFloat()
    }

    /** Eager captions need less evidence (sooner, more wrong words); careful ones more. */
    private fun applyCaution(s: Spotter) {
        val k = CAUTION_SCALE[caution.coerceIn(0, 2)]
        s.emin = liveEmin * k
        s.commit = liveCommit * k
    }

    /** The loaded classifier's words, in label order; empty until a model is loaded. */
    fun labels(): List<String> = model?.labels.orEmpty()

    /** Same-sign synonym map (word -> other names of the same sign), for offline Text-to-Sign lookup. */
    fun synonymMap(): Map<String, List<String>> {
        val m = model ?: return emptyMap()
        val sets = sameSign ?: return emptyMap()
        val out = HashMap<String, List<String>>()
        for (i in m.labels.indices) {
            val others = sets.others(i)
            if (others.isNotEmpty()) out[m.labels[i].uppercase()] = others.map { m.labels[it].uppercase() }
        }
        return out
    }

    /** Speak a phrase or word immediately on user tap, even when automatic speaking is turned off. */
    fun speakNow(text: String) = speaker.speakNow(text)

    /** Speak the current live caption sentence aloud on demand. */
    fun speakCaption() {
        val words = _live.value.caption
        if (words.isNotEmpty()) speaker.speakNow(words.joinToString(" ") { Speaker.sayable(it) })
    }

    /** The loaded classifier's bytes and labels, for the latency benchmark. */
    fun modelAsset(): Pair<ByteArray, List<String>> {
        val labels = JSONArray(context.assets.open("labels.json").bufferedReader().readText()).let { a -> (0 until a.length()).map { a.getString(it) } }
        return context.assets.open(MODEL_ASSET).readBytes() to labels
    }

    // ---------- same-sign names and signs taught on the phone ----------

    @Volatile private var sameSign: SameSignSets? = null
    @Volatile private var labelIndex: Map<String, Int> = emptyMap()
    private val store = TaughtStore(context)
    @Volatile private var matcher: TaughtMatcher? = null
    @Volatile private var teachThreshold = TaughtMatcher.DEFAULT_THRESHOLD
    private val _teach = MutableStateFlow(TeachState())
    val teach: StateFlow<TeachState> = _teach.asStateFlow()

    private class Recording(val durationMs: Long, val done: CompletableDeferred<TeachExample>) {
        var startTs = -1L
        val frames = ArrayList<RawFrame>()
        var w = 0
        var h = 0
    }
    @Volatile private var recording: Recording? = null

    private fun also(p: Prediction, word: String): List<String> {
        val i = labelIndex[word] ?: return emptyList()
        return sameSign?.others(i)?.map { p.labels[it] }.orEmpty()
    }

    /**
     * Other signs the caption offers beside [word]: the best word of each further sign in the model's ranking
     * (SameSignSets.alternatives). Offered only while the model's confidence in the shown sign is under ALT_BELOW,
     * so a caption it is sure of stays clean. On the WLASL2000 test clips the sign that was made is then on screen
     * for 89.9% of clips against 76.8% for the top word alone, with words offered under 67% of captions
     * (scripts/alternatives_eval.py).
     */
    private fun alternatives(p: Prediction, word: String): List<String> {
        val i = labelIndex[word] ?: return emptyList()
        if (i !in p.setProbs.indices || p.setProbs[i] >= ALT_BELOW) return emptyList()
        val sets = sameSign ?: SameSignSets(IntArray(p.labels.size) { it })   // no sets: every word its own sign
        return sets.alternatives(p.order, i, ALTERNATIVES).map { p.labels[it] }
    }

    private val _live = MutableStateFlow(LiveState())
    val liveState: StateFlow<LiveState> = _live.asStateFlow()
    private val _overlay = MutableStateFlow(OverlayState())
    val overlayState: StateFlow<OverlayState> = _overlay.asStateFlow()
    private val _transcript = MutableStateFlow<List<TranscriptEntry>>(emptyList())
    val transcript: StateFlow<List<TranscriptEntry>> = _transcript.asStateFlow()
    var info: ModelInfo? = null
        private set

    // ---------- the conversation, kept on the phone across restarts ----------

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val conversations = ConversationStore(context.filesDir)
    @Volatile private var keepDays: Int? = null

    init {
        scope.launch {
            val saved = conversations.load()
            _transcript.update { now -> Conversations.prune((saved + now).sortedBy { it.atMillis }, keepDays, System.currentTimeMillis()) }
            @OptIn(FlowPreview::class)
            _transcript.drop(1).debounce(400).collect { conversations.save(it) }
        }
    }

    /** How long conversations are kept, in days; null until the person deletes them. Older ones go at once. */
    fun setRetention(days: Int?) {
        keepDays = days
        _transcript.update { Conversations.prune(it, days, System.currentTimeMillis()) }
    }

    /** History: delete the one conversation that started at [start]. */
    fun deleteConversation(start: Long) = _transcript.update { Conversations.without(it, start) }

    private val loading = Mutex()
    /** Loads the EchoGesture model (2,000 ASL words) and the live-caption model once; later calls return at once. */
    suspend fun load() = loading.withLock { loadOnce() }

    private suspend fun loadOnce() = withContext(Dispatchers.IO) {
        if (model != null) return@withContext
        _live.update { it.copy(ready = false) }
        val labels = JSONArray(context.assets.open("labels.json").bufferedReader().readText()).let { a -> (0 until a.length()).map { a.getString(it) } }
        // One thread, workers sleeping between calls: as fast as two when paced like the spotter (6 a second), and
        // no second core kept spinning (bench_model_spin*.jsonl, power_phase_*.jsonl).
        val m = SignModel(context.assets.open(MODEL_ASSET).readBytes(), labels, threads = 1, spin = false)
        // The live model answers windows of continuous signing; a build packed without it falls back on the
        // clip model for them (measured: about level with captions at rest, but no waiting for the hands to drop).
        val lm = runCatching { context.assets.open(LIVE_MODEL_ASSET).readBytes() }.getOrNull()
            ?.let { SignModel(it, labels, threads = 1, spin = false) }
        val liveCfg = runCatching { JSONObject(context.assets.open("live.json").bufferedReader().readText()) }.getOrNull()
        val old = model
        val oldLive = liveModel
        val s = Spotter(m)
        spotter?.let { o ->
            s.threshold = o.threshold; s.slideThreshold = o.slideThreshold; s.agree = o.agree; s.windowMs = o.windowMs
            s.boundary = o.boundary
        }
        // sets.json: one set id per label; only exports made with the same-sign rule have it.
        val setIds = runCatching {
            val a = JSONArray(context.assets.open("sets.json").bufferedReader().readText())
            IntArray(a.length()) { a.getInt(it) }.takeIf { it.size == labels.size }
        }.getOrNull()
        val sets = setIds?.let(::SameSignSets)
        liveEmin = 2.5f
        liveCommit = 6f
        applyLive(s, liveCfg)
        s.liveModel = lm
        s.setOf = setIds
        applyCaution(s)
        s.continuous = benchContinuous ?: continuousOn
        val idxMap = HashMap<String, Int>(labels.size * 2).apply { labels.forEachIndexed { i, l -> put(l, i) } }
        // Swap on the worker thread so a live frame never meets a half-replaced model.
        worker.submit { model = m; liveModel = lm; spotter = s; sameSign = sets; labelIndex = idxMap }.get()
        old?.let(retired::add)   // a clip may still be using it; closed with the engine
        oldLive?.let(retired::add)
        info = runCatching {
            val e = JSONObject(context.assets.open("export.json").bufferedReader().readText())
            val int8 = e.getJSONObject("int8")
            ModelInfo(labels.size, int8.getJSONObject("test").getDouble("top1"), int8.getJSONObject("test").getDouble("top5"),
                e.getInt("test_clips"), e.getJSONArray("ckpts").length(), int8.getDouble("size_mb"))
        }.getOrNull()
        rebuildTaught(m)
        _live.update { it.copy(ready = true, caption = emptyList(), top = emptyList(), lastWord = null, lastAlso = emptyList(), lastTaught = false) }
    }

    /** Prototypes are rebuilt from the stored examples with the loaded model: each model has its own embedding. */
    private fun rebuildTaught(m: SignModel) {
        val zero = Clip(m.tIn, FloatArray(m.tIn * Layout.P * 3), FloatArray(m.tIn * 2 * 21 * 3), FloatArray(m.tIn * 4))
        val supported = runCatching { m.classify(zero).embedding != null }.getOrDefault(false)
        val signs = store.list()
        val protos = if (!supported) emptyList() else signs.mapNotNull { sign ->
            val embs = store.examples(sign.id).mapNotNull { runCatching { m.classify(it).embedding }.getOrNull() }
            if (embs.isEmpty()) null else sign.word to Embeddings.prototype(embs)
        }
        val tm = TaughtMatcher(protos, teachThreshold).takeIf { protos.isNotEmpty() }
        // The live model has its own embedding space, so it gets its own prototypes from the same stored examples.
        // An example is a whole recording, rest to rest; a live window is shorter and centred on the sign, so each
        // example is cut to the part such a window would hold (Cores.window) before the live model embeds it.
        val lm = liveModel ?: m
        val share = (spotter?.liveWindowMs ?: 1200L) / TEACH_MS.toFloat()
        val liveProtos = if (!supported) emptyList() else signs.mapNotNull { sign ->
            val embs = store.examples(sign.id).flatMap { ex ->
                Cores.windows(ex, share).mapNotNull { win -> runCatching { lm.classify(win, needRaw = false).embedding }.getOrNull() }
            }
            if (embs.isEmpty()) null else sign.word to Embeddings.prototype(embs)
        }
        val ltm = TaughtMatcher(liveProtos, teachThreshold).takeIf { liveProtos.isNotEmpty() }
        worker.submit { matcher = tm; liveMatcher = ltm; spotter?.taught = tm; spotter?.liveTaught = ltm }.get()
        _teach.value = TeachState(
            supported, signs,
            if (supported) null else "This model has no embedding output, so signs cannot be taught.",
        )
    }

    fun setTeachThreshold(t: Float) {
        teachThreshold = t
        matcher?.threshold = t
        liveMatcher?.threshold = t
    }

    /** Records [durationMs] of live tracked frames as one example; the camera must be running. */
    suspend fun recordExample(durationMs: Long = TEACH_MS): TeachExample {
        check(live != null && model != null) { "The camera is not running" }
        val r = Recording(durationMs, CompletableDeferred())
        recording = r
        try {
            return withTimeout(durationMs + 4000) { r.done.await() }
        } finally {
            if (recording === r) recording = null
            _live.update { it.copy(recording = null) }
        }
    }

    private fun finishRecording(r: Recording) {
        val m = model
        if (m == null || r.frames.size < 8) {
            r.done.completeExceptionally(IllegalStateException("Too few camera frames. Try again."))
            return
        }
        val hands = r.frames.count { it.hands[0] != null || it.hands[1] != null }.toFloat() / r.frames.size
        val clip = Resampler.resample(Normaliser.normalise(r.frames, r.w, r.h), m.tIn)
        val emb = runCatching { m.classify(clip).embedding }.getOrNull()
        r.done.complete(TeachExample(clip, hands, emb))
    }

    suspend fun saveTaught(word: String, clips: List<Clip>) = withContext(Dispatchers.IO) {
        store.add(word, clips)
        loading.withLock { model?.let(::rebuildTaught) }
    }

    suspend fun renameTaught(id: String, word: String) = withContext(Dispatchers.IO) {
        store.rename(id, word)
        loading.withLock { model?.let(::rebuildTaught) }
    }

    suspend fun deleteTaught(id: String) = withContext(Dispatchers.IO) {
        store.delete(id)
        loading.withLock { model?.let(::rebuildTaught) }
    }

    /**
     * [continuous]: caption each sign while the hands stay up, the default; off: one answer per sign as the
     * hands come to rest, with [threshold], [agree] and [windowMs] as before. [caution] (0 eager, 1 balanced,
     * 2 careful) scales the evidence a continuous caption needs around the value measured for the loaded model.
     */
    fun configure(threshold: Float, agree: Int, windowMs: Long, gpu: Boolean, continuous: Boolean = true, caution: Int = 1) {
        continuousOn = continuous
        this.caution = caution
        worker.execute {
            spotter?.let { s ->
                // One slider: a sign's own answer needs [threshold]; a sliding window, which sees part of a sign,
                // never less than the 0.35 that won on val streams.
                s.threshold = threshold; s.slideThreshold = maxOf(threshold, SLIDE_THRESHOLD)
                s.agree = agree; s.windowMs = windowMs; s.boundary = benchBoundary ?: true
                applyCaution(s)
                s.continuous = benchContinuous ?: continuous
            }
        }
        if (gpu != useGpu) {
            useGpu = gpu
            liveControl.execute { if (live != null) { stopLive(); startLive() } }
        }
    }

    /** Lanes the live tracker runs (1-3); see [Lane]. */
    fun setLanes(n: Int) {
        val k = n.coerceIn(1, 3)
        if (k == laneCount) return
        laneCount = k
        liveControl.execute { if (live != null) { stopLive(); startLive() } }
    }

    // ---------- live camera ----------

    // Live and Teach both use the camera; navigating between them may start the new screen before the old one
    // stops, so the landmarker is counted rather than toggled.
    private var liveUsers = 0   // used on liveControl's thread only

    /**
     * Starting and stopping the live trackers, one at a time and in the order asked, off the main thread: building
     * a Holistic graph takes about 1.5 s on an emulator, and doing it on the main thread froze whichever screen
     * asked for the camera (and the splash screen at launch). Frames that arrive before a tracker exists are skipped.
     */
    private val liveControl = Executors.newSingleThreadExecutor()

    /** A screen needs the live trackers. The returned future completes once they run (benchmarks wait for it). */
    fun acquireLive(): Future<*> = liveControl.submit {
        liveUsers++
        startLive()
    }

    fun releaseLive() {
        liveControl.execute {
            liveUsers = maxOf(0, liveUsers - 1)
            if (liveUsers == 0) stopLive()
        }
    }

    private fun startLive() {
        if (lanes.isNotEmpty()) return
        val plan = benchLanes ?: List(laneCount) { useGpu }
        val made = ArrayList<Lane>()
        for (gpu in plan) {
            val lane = runCatching { newLane(gpu) }
                .recoverCatching { Log.w(TAG, "GPU landmarker failed, using CPU", it); newLane(false) }
                .onFailure { e -> if (made.isEmpty()) _live.update { it.copy(error = "Could not start hand tracking: ${e.message}") } }
                .getOrNull() ?: break   // a phone short of memory runs fewer lanes
            made += lane
        }
        synchronized(order) { inFlight.clear(); ready.clear(); lastResultTs = 0L }
        lanes = made
        _live.update {
            it.copy(running = made.isNotEmpty(), gpu = made.any { l -> l.gpu }, lanes = made.size,
                error = if (made.isEmpty()) it.error else null)
        }
    }

    /** Hand tracking failed to start: try again with the current settings, for the screens that need it now. */
    fun retryLive() {
        liveControl.execute {
            _live.update { it.copy(error = null) }
            if (liveUsers > 0) { stopLive(); startLive() }
        }
    }

    private fun stopLive() {
        val old = lanes
        lanes = emptyList()
        old.forEach { lane ->
            synchronized(lane.lock) {
                lane.closed = true
                lane.startedNs = 0L
                lane.inFlightTs = 0L
                runCatching { lane.landmarker.close() }
            }
        }
        synchronized(order) { inFlight.clear(); ready.clear() }
        _live.update { it.copy(running = false) }
    }

    fun setPaused(paused: Boolean) = _live.update { it.copy(paused = paused, top = if (paused) emptyList() else it.top) }

    fun clearCaption() {
        worker.execute { spotter?.reset() }
        _live.update {
            it.copy(caption = emptyList(), lastWord = null, top = emptyList(), lastAlso = emptyList(), lastTaught = false,
                lastAlternatives = emptyList(), undo = null)
        }
    }

    private var undoAction: (() -> Unit)? = null   // used on the spotter's thread only

    /**
     * The person removed the newest caption word (it was wrong): it leaves the screen and the conversation, and can
     * be put back with [undo]. Runs after the frames already queued and does nothing if the caption is empty.
     */
    fun removeLastWord() {
        worker.execute {
            val before = _live.value
            val word = before.caption.lastOrNull() ?: return@execute
            if (spotter?.removeLast(word) != true) return@execute
            var removed: TranscriptEntry? = null
            _transcript.update { t ->
                val k = t.indexOfLast { it.who == Who.SIGNER && it.word == word }
                if (k < 0) t else { removed = t[k]; t.toMutableList().also { m -> m.removeAt(k) } }
            }
            val after = before.caption.dropLast(1)
            _live.update {
                it.copy(caption = after, lastWord = after.lastOrNull(), lastAlso = emptyList(), lastTaught = false,
                    lastAlternatives = emptyList(), undo = UndoOffer(UndoOffer.Kind.REMOVED_WORD, word, SystemClock.uptimeMillis()))
            }
            undoAction = {
                if (_live.value.caption == after) {   // nothing new was captioned meanwhile
                    spotter?.restoreCaption(before.caption)
                    removed?.let { e -> _transcript.update { t -> (t + e).sortedBy { it.atMillis } } }
                    _live.update {
                        it.copy(caption = before.caption, lastWord = before.lastWord, lastAlso = before.lastAlso,
                            lastTaught = before.lastTaught)
                    }
                }
            }
        }
    }

    /** The person cleared all the captions on screen (the conversation keeps them); [undo] brings them back. */
    fun clearCaptionByUser() {
        worker.execute {
            val before = _live.value
            if (before.caption.isEmpty()) return@execute
            spotter?.reset()
            _live.update {
                it.copy(caption = emptyList(), lastWord = null, top = emptyList(), lastAlso = emptyList(), lastTaught = false,
                    lastAlternatives = emptyList(), undo = UndoOffer(UndoOffer.Kind.CLEARED, null, SystemClock.uptimeMillis()))
            }
            undoAction = {
                if (_live.value.caption.isEmpty()) {
                    spotter?.restoreCaption(before.caption)
                    _live.update { it.copy(caption = before.caption, lastWord = before.lastWord, lastAlso = before.lastAlso, lastTaught = before.lastTaught) }
                }
            }
        }
    }

    fun undo() {
        worker.execute {
            undoAction?.invoke()
            undoAction = null
            _live.update { it.copy(undo = null) }
        }
    }

    fun dismissUndo() {
        worker.execute {
            undoAction = null
            _live.update { it.copy(undo = null) }
        }
    }

    /**
     * The person tapped one of the offered words: it replaces the newest caption word on screen and in the
     * conversation, and is spoken, so the hearing person hears the correction. Runs on the spotter's thread, after
     * the frames already queued, and does nothing if the caption has moved on by then.
     */
    fun correctLast(to: String) {
        worker.execute {
            val from = _live.value.lastWord ?: return@execute
            if (spotter?.replaceLast(from, to) != true) return@execute
            val labels = model?.labels.orEmpty()
            val names = labelIndex[to]?.let { sameSign?.others(it) }?.map { labels[it] }.orEmpty()
            _live.update {
                it.copy(caption = it.caption.dropLast(1) + to, lastWord = to, lastWordAt = SystemClock.uptimeMillis(),
                    lastAlso = names, lastTaught = false, lastAlternatives = emptyList())
            }
            _transcript.update { t ->
                val k = t.indexOfLast { it.who == Who.SIGNER && it.word == from }
                if (k < 0) t else t.toMutableList().also { m -> m[k] = m[k].copy(word = to) }
            }
            speaker.speak(to)
        }
    }

    /** A benchmark replays a video through the live screen (--ez ui true): camera frames are ignored meanwhile. */
    @Volatile var replayOnly = false

    fun clearTranscript() = _transcript.update { emptyList() }

    private val frameConverter = CameraFrameConverter(maxSide = 480, slots = 4)
    private val idleGrid = ByteArray(192)
    private var hasIdleGrid = false

    /** Fast 16x12 coarse luminance grid diff (~5 us, zero allocs) to wake immediately from deep idle on motion. */
    private fun hasIdleMotion(proxy: ImageProxy): Boolean {
        val planes = proxy.planes
        if (planes.isEmpty()) return true
        val plane = planes[0]
        val buf = plane.buffer
        val w = proxy.width
        val h = proxy.height
        val rowStride = plane.rowStride
        val pixStride = plane.pixelStride
        if (w < 16 || h < 12 || pixStride <= 0) return true
        val chOff = if (pixStride >= 3) 1 else 0   // green channel for RGBA, Y for YUV
        val limit = buf.limit()
        var diffSum = 0
        var idx = 0
        for (gy in 0 until 12) {
            val y = (gy * h + h / 2) / 12
            val rowBase = y * rowStride
            for (gx in 0 until 16) {
                val x = (gx * w + w / 2) / 16
                val off = rowBase + x * pixStride + chOff
                val v = if (off in 0 until limit) buf.get(off).toInt() and 0xFF else 0
                if (hasIdleGrid) {
                    val prev = idleGrid[idx].toInt() and 0xFF
                    diffSum += kotlin.math.abs(v - prev)
                }
                idleGrid[idx++] = v.toByte()
            }
        }
        if (!hasIdleGrid) {
            hasIdleGrid = true
            return false
        }
        return diffSum >= 192 * 7   // mean abs delta >= 7/255 across the 16x12 grid
    }

    /** CameraX analyzer: upright, resized frame into the live landmarker (zero per-frame Bitmap allocations). */
    fun analyze(proxy: ImageProxy) {
        // MediaPipe drops frames that arrive while it is busy; dropping them here, before the RGBA copy and the
        // resize, saves that work too.
        val now = SystemClock.uptimeMillis()
        if (replayOnly || lanes.isEmpty() || model == null || _live.value.paused || busy()) {
            proxy.close()
            return
        }
        if (gateIdle && recording == null && now - lastHandsAt > IDLE_AFTER_MS) {
            if (hasIdleMotion(proxy)) {
                lastHandsAt = now
                lastSigningAt = now
                lastFedAt = now
                hasIdleGrid = false
            } else if (now - lastFedAt < STILL_IDLE_PERIOD_MS) {
                proxy.close()
                return
            } else {
                lastFedAt = now
            }
        } else {
            hasIdleGrid = false
            if (!wantFrame(now)) {
                proxy.close()
                return
            }
        }
        val bitmap = try { frameConverter.convert(proxy) } finally { proxy.close() }
        analyzeBitmap(bitmap)
    }

    // 3-tier signing-gated rate (after MobileASL's variable frame rate):
    //   - active signing / recording: full 30 fps
    //   - hands visible but resting below the rest line for > REST_IDLE_AFTER_MS: 15 fps (halves Holistic power)
    //   - no hands in view for > IDLE_AFTER_MS: ~8.3 fps
    @Volatile private var lastHandsAt = 0L
    @Volatile private var lastSigningAt = 0L
    @Volatile private var lastFedAt = 0L
    @Volatile var gateIdle = true

    /** Whether the camera frame arriving at [now] (uptime ms) should be tracked. */
    fun wantFrame(now: Long): Boolean {
        if (gateIdle && recording == null) {
            if (now - lastHandsAt > IDLE_AFTER_MS) {
                if (now - lastFedAt < IDLE_PERIOD_MS) return false
            } else if (now - lastSigningAt > REST_IDLE_AFTER_MS) {
                if (now - lastFedAt < REST_PERIOD_MS) return false
            }
        }
        lastFedAt = now
        return true
    }

    private fun idle(lane: Lane, now: Long) = !lane.closed && (lane.startedNs == 0L || now - lane.startedNs > STALE_NS)

    /** Every lane is tracking a frame; a frame with no result after a second is forgotten, so tracking never stalls. */
    fun busy(): Boolean {
        val now = System.nanoTime()
        return lanes.none { idle(it, now) }
    }

    /**
     * One upright frame into the next idle lane (camera frames, or a replayed video in the benchmark).
     * Returns false, and drops the frame, when every lane is busy.
     */
    fun analyzeBitmap(bitmap: Bitmap): Boolean {
        val all = lanes
        if (all.isEmpty()) return false
        val lane: Lane
        val ts: Long
        synchronized(order) {
            val now = System.nanoTime()
            var picked = -1
            for (i in all.indices) {
                val idx = (nextLane + i) % all.size
                if (idle(all[idx], now)) { picked = idx; break }
            }
            if (picked < 0) return false
            nextLane = (picked + 1) % all.size
            lane = all[picked]
            var t = SystemClock.uptimeMillis()
            if (t <= lastTs) t = lastTs + 1   // MediaPipe needs strictly increasing timestamps
            lastTs = t
            ts = t
            lane.startedNs = now
            lane.inFlightTs = ts
            inFlight[t] = now
        }
        val ok = synchronized(lane.lock) {
            if (lane.closed) {
                synchronized(order) { inFlight.remove(ts) }
                lane.startedNs = 0L
                lane.inFlightTs = 0L
                return false
            }
            runCatching { lane.landmarker.detectAsync(BitmapImageBuilder(bitmap).build(), ts) }
                .onFailure {
                    Log.w(TAG, "detectAsync", it)
                    synchronized(order) {
                        inFlight.remove(ts)
                        if (lane.inFlightTs == ts) { lane.startedNs = 0L; lane.inFlightTs = 0L }
                    }
                }.isSuccess
        }
        return ok
    }

    /** Recover immediately if a lane reports an async error instead of waiting STALE_NS (1 s). */
    private fun onLaneError(lane: Lane) {
        synchronized(order) {
            val ts = lane.inFlightTs
            if (ts != 0L) inFlight.remove(ts)
            lane.startedNs = 0L
            lane.inFlightTs = 0L
            while (ready.isNotEmpty()) {
                val head = ready.firstKey()
                if (inFlight.isNotEmpty() && inFlight.firstKey() < head) break
                deliver(head, ready.remove(head)!!)
            }
        }
    }

    /** A lane finished a frame: hold it until every earlier frame is back, then pass them on in order. */
    private fun onLaneResult(lane: Lane, result: HolisticLandmarkerResult, image: MPImage) {
        val ts = result.timestampMs()
        val frame = result.toRawFrame()   // on this lane's own thread, so lanes convert in parallel
        val doneNs = System.nanoTime()
        synchronized(order) {
            if (lane.inFlightTs == ts) {
                lane.startedNs = 0L
                lane.inFlightTs = 0L
            }
            val sent = inFlight.remove(ts) ?: return   // forgotten as stale, or the lanes were restarted
            ready[ts] = Tracked(frame, image.width, image.height, (doneNs - sent) / 1e6)
            inFlight.entries.removeIf { doneNs - it.value > STALE_NS }
            while (ready.isNotEmpty()) {
                val head = ready.firstKey()
                if (inFlight.isNotEmpty() && inFlight.firstKey() < head) break
                deliver(head, ready.remove(head)!!)   // queues on the worker, so frames reach the spotter in order
            }
        }
    }

    private fun deliver(ts: Long, t: Tracked) {
        val frame = t.frame
        val w = t.w
        val h = t.h
        val landmarksMs = t.landmarksMs
        bench?.let { b -> synchronized(b) { b.landmarks += landmarksMs; b.results++; b.total++ } }
        val hasHands = frame.hands[0] != null || frame.hands[1] != null
        val nowMs = SystemClock.uptimeMillis()
        if (hasHands) lastHandsAt = nowMs
        val dt = if (lastResultTs > 0) (ts - lastResultTs).coerceAtLeast(1) else 0
        lastResultTs = ts
        worker.execute {
            val s = spotter ?: return@execute
            // While an example is being recorded the frames go to it, not to the captions.
            val rec = recording
            var progress: Float? = null
            if (rec != null) {
                if (rec.startTs < 0) rec.startTs = ts
                rec.frames += frame
                rec.w = w
                rec.h = h
                progress = ((ts - rec.startTs).toFloat() / rec.durationMs).coerceIn(0f, 1f)
                if (ts - rec.startTs >= rec.durationMs) {
                    recording = null
                    progress = null
                    finishRecording(rec)
                }
            }
            val target = practiceTarget
            s.watch = target?.let { labelIndex[it] }
            val event = if (rec == null) s.push(ts, frame, w, h) else null
            if (s.signing || rec != null) lastSigningAt = nowMs
            event?.prediction?.let { pr -> bench?.let { b -> synchronized(b) { b.model += pr.latencyMs } } }
            val word = event?.word
            val hit = event?.taught
            val taught = hit != null
            val alsoNames = if (word != null && !taught) event.prediction?.let { also(it, word) }.orEmpty() else emptyList()
            val others = if (word != null && !taught) event.prediction?.let { alternatives(it, word) }.orEmpty() else emptyList()
            var attempt: Attempt? = null
            val sandbox = sandboxMode
            for (wrd in event?.words.orEmpty()) {
                val conf = hit?.cosine ?: event?.prediction?.confidence ?: 0f
                if (target != null) {   // practising: judge the sign against the word asked for, say and keep nothing
                    val share = if (wrd == target) 1f else event?.watchShare ?: 0f
                    attempt = Attempt(target, wrd, share, share >= PRACTICE_PASS, SystemClock.uptimeMillis())
                    if (bench != null) Log.i(Bench.TAG, "attempt " + target + " seen " + wrd + " share " + share + " correct " + (share >= PRACTICE_PASS))
                    continue
                }
                if (sandbox) continue   // Teach "Try it": caption shows on screen without speaking or polluting Chat
                speaker.speak(wrd, ts)   // ts is the uptime of the camera frame that settled the word
                _transcript.update { it + TranscriptEntry(wrd, conf, System.currentTimeMillis()) }
                if (bench != null) Log.i(Bench.TAG, "caption " + wrd + " conf " + conf + " at " + ts + (if (taught) " taught" else ""))
            }
            _overlay.update { ov ->
                val fps = if (dt > 0) (if (ov.fps == 0.0) 1000.0 / dt else 0.9 * ov.fps + 0.1 * 1000.0 / dt) else ov.fps
                ov.copy(
                    hands = frame.hands.toList(),
                    imageWidth = w,
                    imageHeight = h,
                    fps = fps,
                    landmarksMs = if (ov.landmarksMs == 0.0) landmarksMs else 0.8 * ov.landmarksMs + 0.2 * landmarksMs,
                    modelMs = event?.prediction?.latencyMs ?: ov.modelMs,
                )
            }
            _live.update { st ->
                val newTop = when {
                    rec != null -> emptyList()
                    event == null -> st.top
                    hit != null -> listOf(hit.word to hit.cosine) + (event.prediction?.topK(2) ?: emptyList())
                    event.prediction != null -> event.prediction!!.topK(3)
                    (s.continuous || s.boundary) && !event.signing -> st.top   // at rest: the last sign's guesses stay until the next
                    else -> emptyList()
                }
                val newCaption = if (word != null || st.caption.size != s.caption.size) s.caption.toList() else st.caption
                val newPending = if (rec == null && s.continuous) s.tentative else null
                val newPendingProgress = if (rec == null && s.continuous) s.tentativeProgress else 0f
                val newTaughtBest = if (rec == null && sandbox) s.liveTaughtBest?.let { it.word to it.cosine } else null
                st.copy(
                    handsVisible = hasHands,
                    signing = s.signing,
                    bodyVisible = frame.pose != null,
                    top = newTop,
                    caption = newCaption,
                    pending = newPending,
                    pendingProgress = newPendingProgress,
                    taughtMatch = newTaughtBest,
                    lastWord = word ?: st.lastWord,
                    lastWordAt = if (word != null) SystemClock.uptimeMillis() else st.lastWordAt,
                    lastAlso = if (word != null) alsoNames else st.lastAlso,
                    lastTaught = if (word != null) taught else st.lastTaught,
                    lastAlternatives = if (word != null) others else st.lastAlternatives,
                    recording = progress,
                    attempt = attempt ?: st.attempt,
                )
            }
        }
    }

    // ---------- video clips: the same pipeline as the scored predict(video) ----------

    // VIDEO-mode landmarkers kept across clips (as in src/main.py): rebuilt when the frame size or delegate changes,
    // and fed a blank frame before each clip so tracking starts afresh. CLIP_LANES of them take the clip's frames in
    // turn at the same time (src/main.py ECHO_LANES does the same, checked on 600 test clips against one lane).
    private val clipLock = Mutex()
    private val clipLandmarkers = arrayOfNulls<HolisticLandmarker>(CLIP_LANES)
    private val clipKeys = arrayOfNulls<Triple<Int, Int, Boolean>>(CLIP_LANES)
    private var clipTs = 0L

    private fun clipLandmarkerFor(lane: Int, w: Int, h: Int): HolisticLandmarker {
        val gpu = benchClipGpu ?: useGpu
        val key = Triple(w, h, gpu)
        clipLandmarkers[lane]?.let { if (clipKeys[lane] == key) return it; it.close() }
        val lm = runCatching { createLandmarker(RunningMode.VIDEO, gpu) }.getOrElse { createLandmarker(RunningMode.VIDEO, false) }
        clipLandmarkers[lane] = lm
        clipKeys[lane] = key
        return lm
    }

    suspend fun recogniseClip(uri: Uri, name: String, progress: (Float, String) -> Unit): ClipResult = withContext(Dispatchers.Default) {
        val m = model ?: error("Model is still loading")
        val t0 = System.nanoTime()
        val tracked = arrayOfNulls<RawFrame>(MAX_CLIP_FRAMES)
        val done = java.util.concurrent.atomic.AtomicInteger(0)
        var w = 0
        var h = 0
        var trackFirstNs = Long.MAX_VALUE
        var trackLastNs = 0L
        var maxTs = 0L
        var decodeMs = 0.0
        val timing = Any()
        val bitmapsToRecycle = java.util.Collections.newSetFromMap(java.util.IdentityHashMap<Bitmap, Boolean>())
        clipLock.withLock {
            // Every lane counts timestamps from one base, later than anything used before; a lane's own frames
            // arrive in order, so its timestamps strictly increase as MediaPipe requires.
            val base = clipTs + 1000
            val start = base + 1000
            class LaneState(val lane: Int) { var lm: HolisticLandmarker? = null; var lastTs = -1L }
            val laneStates = Array(CLIP_LANES) { LaneState(it) }
            fun track(st: LaneState, f: PickedFrame, total: Int) {
                val bmp = f.bitmap
                val t = System.nanoTime()
                if (st.lm == null) {
                    st.lm = clipLandmarkerFor(st.lane, bmp.width, bmp.height)
                    val blank = Bitmap.createBitmap(bmp.width, bmp.height, Bitmap.Config.ARGB_8888)
                    try { st.lm!!.detectForVideo(BitmapImageBuilder(blank).build(), base) } finally { blank.recycle() }
                }
                val ts = maxOf(start + f.timeMs, st.lastTs + 1)
                st.lastTs = ts
                val frame = st.lm!!.detectForVideo(BitmapImageBuilder(bmp).build(), ts).toRawFrame()
                val e = System.nanoTime()
                synchronized(timing) {
                    tracked[f.k] = frame
                    w = bmp.width
                    h = bmp.height
                    trackFirstNs = minOf(trackFirstNs, t)
                    trackLastNs = maxOf(trackLastNs, e)
                    maxTs = maxOf(maxTs, ts)
                    bitmapsToRecycle.add(bmp)
                }
                val n = done.incrementAndGet()
                val denom = maxOf(total, 1)
                progress((0.05f + 0.9f * n / denom).coerceAtMost(0.95f), "Tracking hands and body · frame $n of $denom")
            }

            // One sequential decode on its own thread; round-robin feeds all active lanes immediately so every
            // lane overlaps with decoding from frame 0. On GPU cap to 2 lanes to avoid OpenCL/EGL contention.
            val activeLanes = if (benchClipGpu ?: useGpu) minOf(CLIP_LANES, 2) else CLIP_LANES
            var decoded = false
            var expectedTotal = MAX_CLIP_FRAMES
            try {
                coroutineScope {
                    val chs = Array(activeLanes) { Channel<PickedFrame>(capacity = MAX_CLIP_FRAMES) }
                    launch(Dispatchers.IO) {
                        val td = System.nanoTime()
                        try {
                            decoded = VideoFrames.decode(context, uri, MAX_CLIP_FRAMES) { f ->
                                expectedTotal = f.total
                                chs[f.k % activeLanes].trySendBlocking(f)
                            }
                        } finally {
                            decodeMs = (System.nanoTime() - td) / 1e6
                            chs.forEach { it.close() }
                        }
                    }
                    for (k in 0 until activeLanes) launch(Dispatchers.Default) { for (f in chs[k]) track(laneStates[k], f, f.total) }
                }
            } finally {
                clipTs = maxOf(start, maxTs) + 1000   // later than anything used so far, even if this clip failed
                bitmapsToRecycle.forEach { if (!it.isRecycled) it.recycle() }
                bitmapsToRecycle.clear()
            }

            // Fallback for videos the decoder path cannot read (or decoded too few frames): retriever, one seek per frame.
            val gotCount = tracked.count { it != null }
            if (!decoded || gotCount < minOf(expectedTotal, 4)) {
                java.util.Arrays.fill(tracked, null)
                done.set(0)
                trackFirstNs = Long.MAX_VALUE
                trackLastNs = 0L
                val mmr = MediaMetadataRetriever()
                try {
                    val td = System.nanoTime()
                    mmr.setDataSource(context, uri)
                    val count = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull() ?: 0
                    val durationMs = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
                    require(count > 0) { "This video has no readable frames" }
                    val fps = if (durationMs > 0) count * 1000.0 / durationMs else 25.0
                    val n = minOf(count, MAX_CLIP_FRAMES)
                    // One lane, timestamps after the failed pass's: a fresh blank frame starts tracking again.
                    val st = LaneState(0)
                    val shift = clipTs - start + 1000
                    for (k in 0 until n) {
                        val i = if (n == 1) 0 else Math.round(k * (count - 1).toDouble() / (n - 1)).toInt()
                        val rawBmp = mmr.getFrameAtIndex(i) ?: continue
                        val bmp = rawBmp.prepared()
                        if (bmp !== rawBmp) rawBmp.recycle()
                        track(st, PickedFrame(k, (i * 1000.0 / fps).toLong() + shift, bmp, n), n)
                    }
                    clipTs = maxOf(clipTs, maxTs) + 1000
                    decodeMs = (System.nanoTime() - td) / 1e6 - (trackLastNs - trackFirstNs) / 1e6
                } finally {
                    mmr.release()
                    bitmapsToRecycle.forEach { if (!it.isRecycled) it.recycle() }
                    bitmapsToRecycle.clear()
                }
            }
        }
        val frames = tracked.filterNotNull()
        val trackNs = if (trackLastNs > trackFirstNs) trackLastNs - trackFirstNs else 0L   // wall time; lanes overlap
        require(frames.isNotEmpty()) { "Could not decode frames from this video" }
        progress(0.97f, "Recognising")
        val pred = m.classifyFrames(frames, w, h, MAX_CLIP_FRAMES)
        val hit = matcher?.match(pred.embedding)
        val top5 = if (hit != null) listOf(hit.word to hit.cosine) + pred.topK(4) else pred.topK(5)
        speaker.speak(top5.first().first)
        ClipResult(name, top5, frames.size, trackNs / 1e6, pred.latencyMs, (System.nanoTime() - t0) / 1e6,
            decodeMs = decodeMs, also = if (hit == null) also(pred, pred.top) else emptyList(), taught = hit != null)
    }

    private fun createLandmarker(
        mode: RunningMode, gpu: Boolean,
        onResult: ((HolisticLandmarkerResult, MPImage) -> Unit)? = null,
        onError: (() -> Unit)? = null,
    ): HolisticLandmarker {
        val baseBuilder = BaseOptions.builder()
            .setModelAssetPath("holistic_landmarker.task")
            .setDelegate(if (gpu) Delegate.GPU else Delegate.CPU)
        if (gpu) {
            runCatching {
                val cacheDir = File(context.codeCacheDir, "mp_gpu_cache").apply { mkdirs() }.absolutePath
                baseBuilder.setDelegateOptions(
                    BaseOptions.DelegateOptions.GpuOptions.builder()
                        .setCachedKernelPath(cacheDir)
                        .setSerializedModelDir(cacheDir)
                        .setModelToken("holistic_v1")
                        .build(),
                )
            }
        }
        // Thresholds match src/holistic_kp.py, so the phone sees what the model was trained on.
        val b = HolisticLandmarker.HolisticLandmarkerOptions.builder()
            .setBaseOptions(baseBuilder.build())
            .setRunningMode(mode)
            .setOutputFaceBlendshapes(false)
            .setOutputPoseSegmentationMasks(false)
            .setMinPoseDetectionConfidence(0.3f)
            .setMinPosePresenceConfidence(0.3f)
            .setMinHandLandmarksConfidence(0.3f)
            .setMinFaceDetectionConfidence(0.3f)
            .setMinFacePresenceConfidence(0.3f)
        if (mode == RunningMode.LIVE_STREAM) {
            b.setResultListener { result: HolisticLandmarkerResult, image: MPImage -> onResult?.invoke(result, image) }
            b.setErrorListener { e -> Log.e(TAG, "holistic", e); onError?.invoke() }
        }
        // MediaPipe's native Graph.startRunningGraph is not thread-safe across concurrent landmarker creations.
        return synchronized(landmarkerInitLock) { HolisticLandmarker.createFromOptions(context, b.build()) }
    }

    override fun close() {
        conversations.save(_transcript.value)
        scope.cancel()
        liveControl.shutdown()
        runCatching { liveControl.awaitTermination(3, TimeUnit.SECONDS) }   // a start under way finishes first
        stopLive()
        speaker.close()
        listener.close()
        clipLandmarkers.forEach { it?.close() }
        worker.shutdown()
        runCatching { worker.awaitTermination(3, TimeUnit.SECONDS) }   // in-flight Spotter/SignModel calls finish before session.close()
        model?.close()
        liveModel?.close()
        retired.forEach { it.close() }
    }

    companion object {
        private const val TAG = "EchoGesture"
        private val landmarkerInitLock = Any()
        /** The EchoGesture model (2,000 ASL words) and its live-caption model, copied into assets by the build. */
        const val MODEL_ASSET = "echogesture_int8.onnx"
        const val LIVE_MODEL_ASSET = "echogesture_live_int8.onnx"
        // 24 evenly spaced frames: fast picked-clip recognition on the phone (~0.46 s median on 4 CPU lanes),
        // while the scored predict() tracks up to 96 frames for maximum accuracy.
        const val MAX_CLIP_FRAMES = 24
        /** Words offered beside the caption's newest word, and the confidence in the shown sign under which they are. */
        const val ALTERNATIVES = 2
        const val ALT_BELOW = 0.8f
        /** Live lanes by default; see [Lane]. */
        const val DEFAULT_LANES = 2
        /** Video landmarkers tracking contiguous chunks of a picked clip in parallel. */
        const val CLIP_LANES = 4
        private const val STALE_NS = 1_000_000_000L
        private const val REST_IDLE_AFTER_MS = 1_500L
        private const val REST_PERIOD_MS = 66L    // ~15 fps when hands are visible but resting below the rest line
        private const val IDLE_AFTER_MS = 2_000L
        private const val IDLE_PERIOD_MS = 120L       // ~8.3 fps fallback when no hands are in view
        private const val STILL_IDLE_PERIOD_MS = 250L // ~4.0 fps when idle AND scene is visually still (wakes on motion)
        private const val SLIDE_THRESHOLD = 0.35f
        /** Learn mode: a sign passes when the practised word holds at least this share of its evidence. */
        const val PRACTICE_PASS = 0.5f
        /** How long an example of a taught sign is recorded for. */
        const val TEACH_MS = 2500L
        /** Eager, balanced, careful: the measured evidence thresholds times this. */
        private val CAUTION_SCALE = floatArrayOf(0.65f, 1f, 1.4f)
    }
}
