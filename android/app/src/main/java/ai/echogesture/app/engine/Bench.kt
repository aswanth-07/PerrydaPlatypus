package ai.echogesture.app.engine

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.net.Uri
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import ai.echogesture.app.data.TaughtStore
import ai.echogesture.signcore.Clip
import ai.echogesture.signcore.Embeddings
import ai.echogesture.signcore.Layout
import ai.echogesture.signcore.SignModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.util.Random

/**
 * On-device benchmarks for the brief's metrics, started from adb so they are repeatable:
 *
 *   adb shell am start -n ai.echogesture.app/.MainActivity --es bench model --ei runs 400
 *       classifier latency of the EchoGesture model for 1, 2 and 4 threads (warm-up, then [runs] timed calls)
 *   adb shell am start -n ai.echogesture.app/.MainActivity --es bench live --ei seconds 600 --ei fps 30 --es lanes gpu,gpu
 *       a signer video (files/bench.mp4 in the app's external files dir) replayed like a camera at [fps] through the
 *       live pipeline: Holistic lanes (gpu or cpu each), the spotter, the classifier; every 10 s a line with
 *       throughput, latency
 *       percentiles, process CPU, battery current and voltage (power), battery temperature and thermal headroom;
 *       --es file NAME replays another video from that folder, --ez boundary false captions with the sliding
 *       window alone; every caption word goes to logcat ("caption WORD conf C at UPTIME_MS")
 *
 *   adb shell am start -n ai.echogesture.app/.MainActivity --es bench clip --ei runs 10 --ez gpu false
 *       the Videos tab's path on files/bench.mp4 (or --es file NAME): total, tracking, decode and model time
 *
 *   adb shell am start -n ai.echogesture.app/.MainActivity --es bench power
 *       on battery (unplug after starting): signing on 2 CPU lanes 150 s, nobody in view gated 90 s and ungated
 *       90 s, signing on 1 GPU lane 90 s; files power_live_*.jsonl, power_mw averaged from 1 s current samples
 *
 *   adb shell am start -n ai.echogesture.app/.MainActivity --es bench powerall [--ei wait 60]
 *       on battery (unplug within [wait] s): screen-on baseline 90 s, the classifier alone at 6 windows a second on
 *       1 thread 90 s, signing on 2 CPU lanes 150 s, nobody in view at the signing-gated rate 90 s
 *
 *   adb shell am start -n ai.echogesture.app/.MainActivity --es bench powerpair [--ei wait 60] [--ei rounds 4] [--ei seconds 40]
 *       on battery, best in airplane mode: the classifier's own power as paired differences, baseline and classifier
 *       phases alternating (files power_phase_pair<N>_baseline / _classifier), then the live pipeline 150 s
 *
 *   adb shell am start -n ai.echogesture.app/.MainActivity --es bench teach --es words AXE,TWINS --ei examples 3 --es file teach_test.mp4
 *       Teach a sign end to end without a camera: for each word, files/teach_<word>_<n>.mp4 (one example each) are
 *       replayed like a camera while the app records an example exactly as the Teach tab does; the sign is saved;
 *       then the test video is replayed through the live captions. A taught caption is logged with " taught".
 *       The signs taught by the run are deleted afterwards unless --ez keep true.
 *
 * Results go to logcat (tag EchoBench) and to bench_<mode>.jsonl next to bench.mp4. The screen stays on at the
 * brightness it had; the speaker is silent unless a live run is started with --ez voice true, which speaks the
 * captions as the app does and times each word from the frame that settled it to the first sound.
 */
class Bench(private val context: Context, private val engine: SignEngine) {
    private val _status = MutableStateFlow("Starting benchmark")
    val status: StateFlow<String> = _status.asStateFlow()
    private val dir: File = context.getExternalFilesDir(null) ?: context.filesDir

    private fun emit(out: File, o: JSONObject) {
        Log.i(TAG, o.toString())
        out.appendText(o.toString() + "\n")
    }

    suspend fun model(runs: Int) = withContext(Dispatchers.Default) {
        val out = File(dir, "bench_model.jsonl").apply { writeText("") }
        val (bytes, labels) = engine.modelAsset()
        val rng = Random(7)
        val t = 64
        // A realistic input: random but finite points, both hands seen; latency does not depend on the values.
        val clip = Clip(t, FloatArray(t * Layout.P * 3) { rng.nextGaussian().toFloat() * 0.5f },
            FloatArray(t * 2 * 21 * 3) { rng.nextGaussian().toFloat() * 0.05f }, FloatArray(t * 4) { 1f })
        for (spin in listOf(false, true)) for (threads in listOf(1, 2, 4)) {
            _status.value = "Classifier latency, $threads thread(s), spinning ${if (spin) "on" else "off"}"
            SignModel(bytes, labels, threads = threads, spin = spin).use { m ->
                repeat(30) { m.classify(clip) }
                // Back to back, and paced like the live spotter (6 a second), where idle workers either spin or sleep.
                val ms = DoubleArray(runs) { m.classify(clip).latencyMs }
                val paced = DoubleArray(minOf(runs, 120)) { Thread.sleep(160); m.classify(clip).latencyMs }
                emit(out, JSONObject().put("bench", "model").put("words", engine.labels().size).put("threads", threads).put("spin", spin)
                    .put("runs", runs).put("p50_ms", pct(ms, 50.0)).put("p95_ms", pct(ms, 95.0)).put("mean_ms", ms.average())
                    .put("paced_p50_ms", pct(paced, 50.0)).put("paced_p95_ms", pct(paced, 95.0)))
            }
        }
        _status.value = "Classifier latency done"
    }

    /**
     * Recognising a picked video, as the Videos tab does: decode, track 24 frames, classify. [file] is in the app's
     * external files dir (default bench.mp4); the first run is a warm-up and is not counted.
     */
    suspend fun clip(runs: Int, gpu: Boolean, file: String) {
        val out = File(dir, "bench_clip_${if (gpu) "gpu" else "cpu"}.jsonl").apply { writeText("") }
        engine.clipDelegateForBench(gpu)
        val video = File(dir, file)
        require(video.exists()) { "Push a video to ${video.absolutePath}" }
        val total = ArrayList<Double>(); val track = ArrayList<Double>(); val decode = ArrayList<Double>(); val model = ArrayList<Double>()
        var top = ""
        for (i in 0..runs) {
            _status.value = "Video recognition ${if (gpu) "GPU" else "CPU"}: run $i of $runs"
            val r = engine.recogniseClip(Uri.fromFile(video), video.name) { _, _ -> }
            top = "${r.top5.first().first} ${"%.2f".format(r.top5.first().second)}"
            if (i == 0) continue
            total += r.totalMs; track += r.landmarksMs; decode += r.decodeMs; model += r.modelMs
        }
        fun a(x: List<Double>) = x.toDoubleArray()
        emit(out, JSONObject().put("bench", "clip").put("gpu", gpu).put("file", file).put("runs", runs).put("top1", top)
            .put("total_p50_ms", pct(a(total), 50.0)).put("total_p95_ms", pct(a(total), 95.0))
            .put("tracking_p50_ms", pct(a(track), 50.0)).put("decode_p50_ms", pct(a(decode), 50.0))
            .put("model_p50_ms", pct(a(model), 50.0)).put("words", engine.labels().size))
        _status.value = "Video recognition done: $top"
    }

    private suspend fun decode(name: String): List<Bitmap> {
        val video = File(dir, name)
        require(video.exists()) { "Push ${video.absolutePath}" }
        val frames = ArrayList<Bitmap>()
        withContext(Dispatchers.IO) { VideoFrames.decode(context, Uri.fromFile(video), 1000) { f -> frames += f.bitmap } }
        require(frames.isNotEmpty()) { "Could not decode $name" }
        return frames
    }

    /** Teach-a-sign end to end: record [examples] examples of each word from replayed videos, save, caption [test]. */
    suspend fun teach(words: List<String>, examples: Int, test: String, fps: Int, seconds: Int, keep: Boolean) = coroutineScope {
        val lanes = listOf(false, false)
        engine.lanesForBench(lanes)
        engine.speaker.enabled = false
        val period = 1000L / fps
        engine.acquireLive().get()
        try {
            for (w in words) {
                val clips = ArrayList<Clip>()
                val embs = ArrayList<FloatArray>()
                for (n in 1..examples) {
                    _status.value = "Teaching $w: example $n of $examples"
                    val frames = decode("teach_${w.lowercase()}_$n.mp4")
                    val rec = async { engine.recordExample() }
                    val t0 = SystemClock.elapsedRealtime()
                    var next = t0
                    while (!rec.isCompleted) {   // the video once, then its last frame held, as a signer back at rest
                        val now = SystemClock.elapsedRealtime()
                        if (now >= next) {
                            val k = ((now - t0) / period).toInt()
                            engine.analyzeBitmap(frames[minOf(k, frames.size - 1)])
                            next = t0 + (k + 1) * period
                        } else delay(1)
                    }
                    val ex = rec.await()
                    val match = ex.embedding?.takeIf { embs.isNotEmpty() }?.let { Embeddings.cosine(it, Embeddings.prototype(embs)) }
                    Log.i(TAG, "teach example $w $n hands ${"%.2f".format(ex.handFraction)} match_to_earlier ${match?.let { "%.2f".format(it) } ?: "-"}")
                    clips += ex.clip
                    ex.embedding?.let { embs += it }
                }
                engine.saveTaught(w, clips)
                Log.i(TAG, "teach saved $w with ${clips.size} examples; taught now: ${engine.teach.value.signs.joinToString { it.word }}")
            }
        } finally {
            engine.releaseLive()
        }
        try {
            live(seconds, fps, lanes, prefix = "teach", file = test)
        } finally {
            if (!keep) engine.teach.value.signs.filter { it.word in words.map(TaughtStore::clean) }.forEach { engine.deleteTaught(it.id) }
            Log.i(TAG, "teach done; taught left: ${engine.teach.value.signs.joinToString { it.word }}")
        }
    }

    /** [lanes]: one delegate per live landmarker, true for GPU (SignEngine.Lane). */
    suspend fun live(seconds: Int, fps: Int, lanes: List<Boolean>, blank: Boolean = false, gate: Boolean = false, prefix: String = "bench",
                     file: String = "bench.mp4", voice: Boolean = false) {
        val tag = lanes.joinToString("+") { if (it) "gpu" else "cpu" } + (if (blank) "_blank" else "") + (if (gate) "_gated" else "")
        val out = File(dir, "${prefix}_live_${fps}fps_$tag.jsonl").apply { writeText("") }
        engine.lanesForBench(lanes)
        val video = File(dir, file)
        require(video.exists()) { "Push a signer video to ${video.absolutePath}" }
        _status.value = "Decoding the replay video"
        val frames = ArrayList<Bitmap>()
        withContext(Dispatchers.IO) { VideoFrames.decode(context, Uri.fromFile(video), 1000) { f -> frames += f.bitmap } }
        require(frames.isNotEmpty()) { "Could not decode ${video.name}" }
        if (blank) {   // nobody in view: an empty grey room, for the idle cost
            val g = Bitmap.createBitmap(frames[0].width, frames[0].height, Bitmap.Config.ARGB_8888).apply { eraseColor(0xFF6E6A64.toInt()) }
            frames.clear(); frames += g
        }
        engine.gateIdle = gate

        // voice: the captions are spoken as in the app, and each word is timed from the frame that settled it
        // to the moment the text-to-speech engine starts to sound.
        val voiceMs = ArrayList<Double>()
        engine.speaker.onDelay = if (voice) { ms -> synchronized(voiceMs) { voiceMs += ms }; Log.i(TAG, "voice " + ms + " ms after the frame") } else null
        engine.speaker.enabled = voice
        engine.acquireLive().get()
        val stats = SignEngine.BenchStats()
        engine.bench = stats
        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val period = 1000L / fps
        val t0 = SystemClock.elapsedRealtime()
        var sent = 0L
        var lastCpu = cpuTicks()
        var lastT = t0
        var nextReport = t0 + 10_000
        var nextSample = t0 + 1_000
        var mwSum = 0.0
        var uaSum = 0.0
        var samples = 0
        var nextFrame = t0
        emit(out, JSONObject().put("bench", "live").put("event", "start").put("fps_target", fps).put("seconds", seconds)
            .put("gpu", lanes.any { it }).put("lanes", tag).put("blank", blank).put("gate", gate).put("replay_frames", frames.size).put("frame_w", frames[0].width).put("frame_h", frames[0].height)
            .put("words", engine.labels().size).put("device", Build.MODEL).put("soc", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE).put("android", Build.VERSION.RELEASE))
        try {
            while (true) {
                val now = SystemClock.elapsedRealtime()
                if (now - t0 >= seconds * 1000L) break
                // A camera: frame k arrives at t0 + k / fps and is dropped if every lane is busy, as CameraX's
                // keep-only-latest analyzer does; the video plays at its real speed either way.
                if (now >= nextFrame) {
                    val k = ((now - t0) / period).toInt()
                    if ((!gate || engine.wantFrame(now)) && engine.analyzeBitmap(frames[k % frames.size])) sent++
                    nextFrame = t0 + (k + 1) * period
                } else {
                    delay(1)
                }
                if (now >= nextSample) {   // battery current once a second, averaged per report
                    val ua = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW).toDouble()
                    val mv = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
                        ?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0
                    uaSum += ua
                    mwSum += Math.abs(ua) / 1e6 * mv   // uA x mV / 1e6 = mW
                    samples++
                    nextSample += 1_000
                }
                if (now >= nextReport) {
                    val cpu = cpuTicks()
                    val (lm, md, res) = synchronized(stats) {
                        Triple(stats.landmarks.toDoubleArray(), stats.model.toDoubleArray(), stats.results).also {
                            stats.landmarks.clear(); stats.model.clear(); stats.results = 0
                        }
                    }
                    val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
                    val mv = battery?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0
                    val ua = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
                    val secs = (now - lastT) / 1000.0
                    val mem = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }
                    emit(out, JSONObject().put("bench", "live").put("t_s", (now - t0) / 1000.0)
                        .put("fps", res / secs)
                        .put("landmarks_p50_ms", pct(lm, 50.0)).put("landmarks_p95_ms", pct(lm, 95.0))
                        .put("model_p50_ms", pct(md, 50.0)).put("model_p95_ms", pct(md, 95.0)).put("windows", md.size)
                        .put("cpu_pct_of_one_core", 100.0 * (cpu - lastCpu) / CLK_TCK / secs)
                        .put("current_now", ua).put("voltage_mv", mv)
                        .put("current_ua_mean", if (samples > 0) uaSum / samples else 0.0)
                        .put("power_mw", if (samples > 0) mwSum / samples else 0.0)
                        .put("plugged", battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1) ?: -1)
                        .put("battery_c", (battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10.0)
                        .put("level", battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1)
                        .put("thermal_status", if (Build.VERSION.SDK_INT >= 29) pm.currentThermalStatus else -1)
                        .put("thermal_headroom_10s", if (Build.VERSION.SDK_INT >= 30) pm.getThermalHeadroom(10).toDouble() else -1.0)
                        .put("pss_mb", mem.totalPss / 1024.0))
                    _status.value = "Live replay at $fps fps: %.0f of %d s, %.1f fps, tracking p95 %.0f ms"
                        .format((now - t0) / 1000.0, seconds, res / secs, pct(lm, 95.0))
                    lastCpu = cpu
                    lastT = now
                    mwSum = 0.0; uaSum = 0.0; samples = 0
                    nextReport += 10_000
                }
            }
        } finally {
            engine.gateIdle = true
            engine.bench = null
            engine.releaseLive()
            if (voice) {
                delay(1500)   // the last word may still be on its way to the speaker
                engine.speaker.enabled = false
                engine.speaker.onDelay = null
                val v = synchronized(voiceMs) { voiceMs.toDoubleArray() }
                emit(out, JSONObject().put("bench", "live").put("event", "voice").put("words_spoken", v.size)
                    .put("voice_p50_ms", pct(v, 50.0)).put("voice_p90_ms", pct(v, 90.0)).put("voice_max_ms", v.maxOrNull() ?: 0.0)
                    .put("voice", engine.speaker.state.value.voice ?: "").put("on_device", engine.speaker.state.value.onDevice))
            }
            emit(out, JSONObject().put("bench", "live").put("event", "end"))
            _status.value = "Live replay done"
        }
    }

    /**
     * Power of one piece on its own, on battery: [perSecond] calls of [work] a second (none: the screen and the
     * idle app only, the baseline), with battery current sampled every second; a row every 10 s.
     */
    suspend fun powerPhase(name: String, seconds: Int, perSecond: Int, work: (() -> Unit)?) = withContext(Dispatchers.Default) {
        val out = File(dir, "power_phase_$name.jsonl").apply { writeText("") }
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val t0 = SystemClock.elapsedRealtime()
        var nextSample = t0 + 1_000
        var nextReport = t0 + 10_000
        var nextWork = t0
        var mwSum = 0.0
        var samples = 0
        var calls = 0
        var lastCpu = cpuTicks()
        var lastT = t0
        emit(out, JSONObject().put("bench", "power_phase").put("event", "start").put("phase", name).put("per_second", perSecond))
        while (true) {
            val now = SystemClock.elapsedRealtime()
            if (now - t0 >= seconds * 1000L) break
            _status.value = "Power: $name, %.0f of %d s".format((now - t0) / 1000.0, seconds)
            if (work != null && perSecond > 0 && now >= nextWork) { work(); calls++; nextWork += 1000L / perSecond }
            if (now >= nextSample) {
                val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
                val ua = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW).toDouble()
                mwSum += Math.abs(ua) / 1e6 * (battery?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0)
                samples++
                nextSample += 1_000
            }
            if (now >= nextReport) {
                val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
                val cpu = cpuTicks()
                emit(out, JSONObject().put("bench", "power_phase").put("phase", name).put("t_s", (now - t0) / 1000.0)
                    .put("power_mw", if (samples > 0) mwSum / samples else 0.0).put("calls", calls)
                    .put("cpu_pct_of_one_core", 100.0 * (cpu - lastCpu) / CLK_TCK / ((now - lastT) / 1000.0))
                    .put("plugged", battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1) ?: -1))
                mwSum = 0.0; samples = 0; calls = 0; lastCpu = cpu; lastT = now
                nextReport += 10_000
            }
            delay(5)
        }
        emit(out, JSONObject().put("bench", "power_phase").put("event", "end"))
    }

    /** Baseline (screen on, nothing running), then the classifier alone at the live caption rate. */
    suspend fun powerSplit(seconds: Int, threads: Int = 2, spin: Boolean = false) {
        engine.speaker.enabled = false
        powerPhase("baseline", seconds, 0, null)
        val (bytes, labels) = engine.modelAsset()
        val rng = Random(7)
        val t = 64
        val clip = Clip(t, FloatArray(t * Layout.P * 3) { rng.nextGaussian().toFloat() * 0.5f },
            FloatArray(t * 2 * 21 * 3) { rng.nextGaussian().toFloat() * 0.05f }, FloatArray(t * 4) { 1f })
        // The spotter classifies every 5 frames: 6 windows a second at 30 fps.
        val tag = "classifier_6hz_${threads}t" + if (spin) "_spin" else ""
        SignModel(bytes, labels, threads = threads, spin = spin).use { m -> powerPhase(tag, seconds, 6) { m.classify(clip) } }
        _status.value = "Power split done"
    }

    /**
     * The classifier's own power, paired: [rounds] times a screen-on baseline phase then the classifier alone at the
     * live rate (6 windows a second, 1 thread, not spinning), [seconds] each. Alternating cancels the slow drift of
     * the phone's other consumers; each round's classifier phase minus its baseline is one paired difference.
     */
    suspend fun powerPaired(rounds: Int, seconds: Int) {
        engine.speaker.enabled = false
        val (bytes, labels) = engine.modelAsset()
        val rng = Random(7)
        val t = 64
        val clip = Clip(t, FloatArray(t * Layout.P * 3) { rng.nextGaussian().toFloat() * 0.5f },
            FloatArray(t * 2 * 21 * 3) { rng.nextGaussian().toFloat() * 0.05f }, FloatArray(t * 4) { 1f })
        SignModel(bytes, labels, threads = 1, spin = false).use { m ->
            for (r in 1..rounds) {
                powerPhase("pair${r}_baseline", seconds, 0, null)
                powerPhase("pair${r}_classifier", seconds, 6) { m.classify(clip) }
            }
        }
        _status.value = "Paired power done"
    }

    companion object {
        const val TAG = "EchoBench"
        private const val CLK_TCK = 100.0

        /** utime + stime of this process, in clock ticks. */
        private fun cpuTicks(): Long = runCatching {
            val f = File("/proc/self/stat").readText().substringAfterLast(") ").split(" ")
            f[11].toLong() + f[12].toLong()   // fields 14 and 15 of /proc/<pid>/stat
        }.getOrDefault(0L)

        fun pct(a: DoubleArray, q: Double): Double {
            if (a.isEmpty()) return 0.0
            val s = a.sorted()
            return s[((q / 100.0) * (s.size - 1)).toInt().coerceIn(0, s.size - 1)]
        }
    }
}

