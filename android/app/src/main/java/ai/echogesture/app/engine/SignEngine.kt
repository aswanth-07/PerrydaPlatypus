package ai.echogesture.app.engine

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import ai.echogesture.signcore.RawFrame
import ai.echogesture.signcore.SignModel
import ai.echogesture.signcore.Spotter
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarker
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors

data class ModelInfo(
    val words: Int,
    val testTop1: Double,
    val testTop5: Double,
    val testClips: Int,
    val ensemble: Int,
    val classifierMb: Double,
)

data class LiveState(
    val ready: Boolean = false,
    val running: Boolean = false,
    val paused: Boolean = false,
    val handsVisible: Boolean = false,
    val top: List<Pair<String, Float>> = emptyList(),
    val caption: List<String> = emptyList(),
    val lastWord: String? = null,
    val lastWordAt: Long = 0,
    val fps: Double = 0.0,
    val landmarksMs: Double = 0.0,
    val modelMs: Double = 0.0,
    val hands: List<FloatArray?> = listOf(null, null),
    val imageWidth: Int = 0,
    val imageHeight: Int = 0,
    val gpu: Boolean = true,
    val error: String? = null,
)

data class TranscriptEntry(val word: String, val confidence: Float, val atMillis: Long)

data class ClipResult(
    val name: String,
    val top5: List<Pair<String, Float>>,
    val frames: Int,
    val landmarksMs: Double,
    val modelMs: Double,
    val totalMs: Double,
    val atMillis: Long = System.currentTimeMillis(),
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
    private var live: HolisticLandmarker? = null
    private val worker = Executors.newSingleThreadExecutor()   // the spotter is used from this thread only
    private val started = ConcurrentHashMap<Long, Long>()
    private var lastTs = 0L
    private var lastResultTs = 0L
    @Volatile private var useGpu = true

    private val _live = MutableStateFlow(LiveState())
    val liveState: StateFlow<LiveState> = _live.asStateFlow()
    private val _transcript = MutableStateFlow<List<TranscriptEntry>>(emptyList())
    val transcript: StateFlow<List<TranscriptEntry>> = _transcript.asStateFlow()
    var info: ModelInfo? = null
        private set

    private val loading = Mutex()
    var vocabulary = "core"   // "core": 100 words, most accurate; "full": all 2,000 WLASL words; "union": 3,075 signs
        private set

    /** Vocabulary folders packed into this build, in display order. */
    val vocabularies: List<String> by lazy {
        val packed = context.assets.list("").orEmpty().toSet()
        listOf("core", "full", "union").filter { it in packed }
    }

    suspend fun load(vocab: String = vocabulary) = loading.withLock { loadOnce(vocab) }

    private suspend fun loadOnce(vocab: String) = withContext(Dispatchers.IO) {
        if (model != null && vocab == vocabulary) return@withContext
        _live.update { it.copy(ready = false) }
        val labels = JSONArray(context.assets.open("$vocab/labels.json").bufferedReader().readText()).let { a -> (0 until a.length()).map { a.getString(it) } }
        val m = SignModel(context.assets.open("$vocab/sign_int8.onnx").readBytes(), labels, threads = 2)
        val old = model
        val s = Spotter(m)
        spotter?.let { o -> s.threshold = o.threshold; s.agree = o.agree; s.windowMs = o.windowMs }
        // Swap on the worker thread so a live frame never meets a half-replaced model.
        worker.submit { model = m; spotter = s }.get()
        old?.let(retired::add)   // a clip may still be using it; closed with the engine
        vocabulary = vocab
        info = runCatching {
            val e = JSONObject(context.assets.open("$vocab/export.json").bufferedReader().readText())
            val int8 = e.getJSONObject("int8")
            ModelInfo(labels.size, int8.getJSONObject("test").getDouble("top1"), int8.getJSONObject("test").getDouble("top5"),
                e.getInt("test_clips"), e.getJSONArray("ckpts").length(), int8.getDouble("size_mb"))
        }.getOrNull()
        _live.update { it.copy(ready = true, caption = emptyList(), top = emptyList(), lastWord = null) }
    }

    fun configure(threshold: Float, agree: Int, windowMs: Long, gpu: Boolean) {
        worker.execute {
            spotter?.let { s -> s.threshold = threshold; s.agree = agree; s.windowMs = windowMs }
        }
        if (gpu != useGpu) {
            useGpu = gpu
            if (live != null) { stopLive(); startLive() }
        }
    }

    // ---------- live camera ----------

    fun startLive() {
        if (live != null) return
        var onGpu = useGpu
        live = runCatching { createLandmarker(RunningMode.LIVE_STREAM, useGpu) }
            .recoverCatching { Log.w(TAG, "GPU landmarker failed, using CPU", it); onGpu = false; createLandmarker(RunningMode.LIVE_STREAM, false) }
            .onFailure { e -> _live.update { it.copy(error = "Could not start hand tracking: ${e.message}") } }
            .getOrNull()
        _live.update { it.copy(running = live != null, gpu = onGpu) }
    }

    fun stopLive() {
        live?.close()
        live = null
        _live.update { it.copy(running = false) }
    }

    fun setPaused(paused: Boolean) = _live.update { it.copy(paused = paused, top = if (paused) emptyList() else it.top) }

    fun clearCaption() {
        worker.execute { spotter?.reset() }
        _live.update { it.copy(caption = emptyList(), lastWord = null, top = emptyList()) }
    }

    fun clearTranscript() = _transcript.update { emptyList() }

    /** CameraX analyzer: upright, resized frame into the live landmarker. */
    fun analyze(proxy: ImageProxy) {
        val landmarker = live
        if (landmarker == null || model == null || _live.value.paused) {
            proxy.close()
            return
        }
        val bitmap = try { proxy.toBitmap().prepared(proxy.imageInfo.rotationDegrees) } finally { proxy.close() }
        var ts = SystemClock.uptimeMillis()
        if (ts <= lastTs) ts = lastTs + 1   // MediaPipe needs strictly increasing timestamps
        lastTs = ts
        started[ts] = System.nanoTime()
        landmarker.detectAsync(BitmapImageBuilder(bitmap).build(), ts)
    }

    private fun onLiveResult(result: HolisticLandmarkerResult, image: MPImage) {
        val ts = result.timestampMs()
        val landmarksMs = started.remove(ts)?.let { (System.nanoTime() - it) / 1e6 } ?: 0.0
        val frame = result.toRawFrame()
        val w = image.width
        val h = image.height
        val dt = if (lastResultTs > 0) (ts - lastResultTs).coerceAtLeast(1) else 0
        lastResultTs = ts
        worker.execute {
            val s = spotter ?: return@execute
            val event = s.push(ts, frame, w, h)
            val word = event?.word
            if (word != null) {
                _transcript.update { it + TranscriptEntry(word, event.prediction?.confidence ?: 0f, System.currentTimeMillis()) }
            }
            _live.update { st ->
                val fps = if (dt > 0) (if (st.fps == 0.0) 1000.0 / dt else 0.9 * st.fps + 0.1 * 1000.0 / dt) else st.fps
                st.copy(
                    handsVisible = frame.hands.any { it != null },
                    hands = frame.hands.toList(),
                    imageWidth = w, imageHeight = h,
                    fps = fps,
                    landmarksMs = if (st.landmarksMs == 0.0) landmarksMs else 0.8 * st.landmarksMs + 0.2 * landmarksMs,
                    modelMs = event?.prediction?.latencyMs ?: st.modelMs,
                    top = when {
                        event == null -> st.top
                        !event.active -> emptyList()
                        else -> event.prediction?.topK(3) ?: st.top
                    },
                    caption = s.caption.toList(),
                    lastWord = word ?: st.lastWord,
                    lastWordAt = if (word != null) SystemClock.uptimeMillis() else st.lastWordAt,
                )
            }
        }
    }

    // ---------- video clips: the same pipeline as the scored predict(video) ----------

    // One VIDEO-mode landmarker kept across clips (as in src/main.py): rebuilt when the frame size or delegate
    // changes, and fed a blank frame before each clip so tracking starts afresh.
    private val clipLock = Mutex()
    private var clipLandmarker: HolisticLandmarker? = null
    private var clipKey: Triple<Int, Int, Boolean>? = null
    private var clipTs = 0L

    private fun clipLandmarkerFor(w: Int, h: Int): HolisticLandmarker {
        val key = Triple(w, h, useGpu)
        clipLandmarker?.let { if (clipKey == key) return it; it.close() }
        val lm = runCatching { createLandmarker(RunningMode.VIDEO, useGpu) }.getOrElse { createLandmarker(RunningMode.VIDEO, false) }
        clipLandmarker = lm
        clipKey = key
        return lm
    }

    suspend fun recogniseClip(uri: Uri, name: String, progress: (Float, String) -> Unit): ClipResult = withContext(Dispatchers.Default) {
        val m = model ?: error("Model is still loading")
        val t0 = System.nanoTime()
        val mmr = MediaMetadataRetriever()
        try {
            mmr.setDataSource(context, uri)
            val count = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull() ?: 0
            val durationMs = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            require(count > 0) { "This video has no readable frames" }
            val fps = if (durationMs > 0) count * 1000.0 / durationMs else 25.0
            val n = minOf(count, MAX_CLIP_FRAMES)
            val picks = (0 until n).map { if (n == 1) 0 else Math.round(it * (count - 1).toDouble() / (n - 1)).toInt() }
            val frames = ArrayList<RawFrame>(n)
            var w = 0
            var h = 0
            val tLandmarks = System.nanoTime()
            clipLock.withLock {
                var landmarker: HolisticLandmarker? = null
                var start = 0L
                for ((k, i) in picks.withIndex()) {
                    val bmp = mmr.getFrameAtIndex(i)?.prepared() ?: continue
                    if (landmarker == null) {
                        landmarker = clipLandmarkerFor(bmp.width, bmp.height)
                        val blank = Bitmap.createBitmap(bmp.width, bmp.height, Bitmap.Config.ARGB_8888)
                        landmarker.detectForVideo(BitmapImageBuilder(blank).build(), clipTs)
                        start = clipTs + 1000
                        clipTs = start
                    }
                    w = bmp.width
                    h = bmp.height
                    val ts = start + (k * 1000.0 / fps).toLong()
                    clipTs = ts + 1000   // later than anything used so far, even if this clip stops part-way
                    frames += landmarker.detectForVideo(BitmapImageBuilder(bmp).build(), ts).toRawFrame()
                    progress(0.05f + 0.9f * (k + 1) / n, "Tracking hands and body · frame ${k + 1} of $n")
                }
            }
            require(frames.isNotEmpty()) { "Could not decode frames from this video" }
            val landmarksMs = (System.nanoTime() - tLandmarks) / 1e6
            progress(0.97f, "Recognising")
            val pred = m.classifyFrames(frames, w, h, MAX_CLIP_FRAMES)
            ClipResult(name, pred.topK(5), frames.size, landmarksMs, pred.latencyMs, (System.nanoTime() - t0) / 1e6)
        } finally {
            mmr.release()
        }
    }

    private fun createLandmarker(mode: RunningMode, gpu: Boolean): HolisticLandmarker {
        val base = BaseOptions.builder()
            .setModelAssetPath("holistic_landmarker.task")
            .setDelegate(if (gpu) Delegate.GPU else Delegate.CPU)
            .build()
        // Thresholds match scripts/holistic_kp.py, so the phone sees what the model was trained on.
        val b = HolisticLandmarker.HolisticLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(mode)
            .setMinPoseDetectionConfidence(0.3f)
            .setMinPosePresenceConfidence(0.3f)
            .setMinHandLandmarksConfidence(0.3f)
            .setMinFaceDetectionConfidence(0.3f)
            .setMinFacePresenceConfidence(0.3f)
        if (mode == RunningMode.LIVE_STREAM) {
            b.setResultListener { result: HolisticLandmarkerResult, image: MPImage -> onLiveResult(result, image) }
            b.setErrorListener { e -> Log.e(TAG, "holistic", e) }
        }
        return HolisticLandmarker.createFromOptions(context, b.build())
    }

    override fun close() {
        stopLive()
        clipLandmarker?.close()
        worker.shutdown()
        model?.close()
        retired.forEach { it.close() }
    }

    companion object {
        private const val TAG = "EchoGesture"
        // 24 evenly spaced frames scored the same as all frames on the WLASL100 test split (84.5 vs 84.5).
        const val MAX_CLIP_FRAMES = 24
    }
}
