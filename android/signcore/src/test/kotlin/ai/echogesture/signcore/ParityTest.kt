package ai.echogesture.signcore

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Feeds the raw Holistic arrays that Python saved (scripts/make_parity_fixture.py) through the Kotlin
 * normaliser and the exported INT8 model, and checks that the points and probabilities match Python's.
 */
class ParityTest {
    companion object {
        init {
            // Desktop test only: the JDK ships msvcp140.dll 14.36, older than ONNX Runtime needs, and Windows
            // finds it first (java.exe's folder). Loading the system copy first makes ONNX Runtime bind to it.
            val sys = File(System.getenv("SystemRoot") ?: "C:\\Windows", "System32\\msvcp140.dll")
            if (System.getProperty("os.name").startsWith("Windows") && sys.exists()) System.load(sys.absolutePath)
        }
    }

    private val dir = File(System.getProperty("parity.dir") ?: "parity")

    private fun floats(a: Any?): FloatArray? {
        if (a == null || a == JSONObject.NULL) return null
        val arr = a as JSONArray
        return FloatArray(arr.length()) { arr.getDouble(it).toFloat() }
    }

    private fun frames(clip: JSONObject): List<RawFrame> {
        val fr = clip.getJSONArray("frames")
        return (0 until fr.length()).map { k ->
            val f = fr.getJSONObject(k)
            val hands = f.getJSONArray("hands")
            val world = f.getJSONArray("world")
            RawFrame(floats(f.opt("pose")), arrayOf(floats(hands.opt(0)), floats(hands.opt(1))),
                arrayOf(floats(world.opt(0)), floats(world.opt(1))), floats(f.opt("face")))
        }
    }

    @Test
    fun normaliserMatchesPython() {
        val fixture = JSONObject(File(dir, "fixture.json").readText())
        val clips = fixture.getJSONArray("clips")
        var worst = 0.0
        for (i in 0 until clips.length()) {
            val c = clips.getJSONObject(i)
            val norm = Resampler.resample(Normaliser.normalise(frames(c), c.getInt("width"), c.getInt("height")), fixture.getInt("t_in"))
            val ref = floats(c.get("points"))!!
            val refSeen = floats(c.get("seen"))!!
            val refWorld = floats(c.get("world_in"))!!
            for (k in ref.indices) worst = maxOf(worst, abs(ref[k] - norm.points[k]).toDouble())
            for (k in refWorld.indices) worst = maxOf(worst, abs(refWorld[k] - norm.world[k]).toDouble())
            for (k in refSeen.indices) worst = maxOf(worst, abs(refSeen[k] - norm.seen[k]).toDouble())
        }
        println("normaliser: worst abs difference from Python over ${clips.length()} clips = $worst")
        assertTrue(worst < 1e-3, "normalised inputs differ from Python by $worst")
    }

    @Test
    fun int8ModelMatchesPython() {
        val fixture = JSONObject(File(dir, "fixture.json").readText())
        val labels = JSONArray(File(dir, "labels.json").readText()).let { a -> (0 until a.length()).map { a.getString(it) } }
        val clips = fixture.getJSONArray("clips")
        // Dynamic INT8 quantizes activations from each input's own range, so the model is sensitive to tiny input
        // changes by itself: in Python, noise of +-7e-6 (the normaliser's worst difference) on the present points
        // moves the shipped 2,000-word model's probabilities by up to 0.037 while its FP32 export moves by 0.0
        // (2026-11-13). The bound sits above that; agreement on the answer is the real test.
        val tolerance = 5e-2
        var worst = 0.0
        var worstRaw = 0.0
        var rawChecked = 0
        var agree = 0
        var correct = 0
        val flips = mutableListOf<String>()
        SignModel(File(dir, "sign_int8.onnx").readBytes(), labels, fixture.getInt("t_in")).use { model ->
            for (i in 0 until clips.length()) {
                val c = clips.getJSONObject(i)
                val pred = model.classify(Normaliser.normalise(frames(c), c.getInt("width"), c.getInt("height")))
                val ref = floats(c.get("probs"))!!
                for (k in ref.indices) worst = maxOf(worst, abs(ref[k] - pred.probs[k]).toDouble())
                val refRaw = floats(c.opt("raw_probs"))
                val raw = pred.rawProbs
                if (refRaw != null && raw != null) {
                    rawChecked++
                    for (k in refRaw.indices) worstRaw = maxOf(worstRaw, abs(refRaw[k] - raw[k]).toDouble())
                }
                val sorted = ref.indices.sortedByDescending { ref[it] }
                if (sorted[0] == pred.order[0]) agree++
                else {
                    // Dynamic INT8 rescales activations from each input's own range, so the Kotlin normaliser's
                    // ~1e-5 differences can reorder two classes Python itself scores as a tie. Such a flip is only
                    // acceptable where Python's top-1 leads by less than the probability tolerance.
                    val margin = (ref[sorted[0]] - ref[sorted[1]]).toDouble()
                    flips.add("clip $i (Python margin $margin)")
                    assertTrue(margin < tolerance, "top-1 differs from Python on clip $i, where Python leads by $margin")
                }
                if (labels[pred.order[0]] == c.getString("label")) correct++
            }
        }
        println("int8 model: worst prob difference $worst (network's own, before the rule: $worstRaw on $rawChecked clips); " +
            "top-1 agrees with Python on $agree/${clips.length()}; correct $correct; near-tie flips: $flips")
        assertTrue(worst < tolerance, "probabilities differ from Python by $worst")
        assertTrue(worstRaw < tolerance, "the network's own probabilities differ from Python by $worstRaw")
    }

    @Test
    fun slotsFollowPoseWrists() {
        val pose = FloatArray(33 * 4)
        pose[Layout.POSE_WRIST_L * 4] = 0.7f; pose[Layout.POSE_WRIST_L * 4 + 1] = 0.5f
        pose[Layout.POSE_WRIST_R * 4] = 0.3f; pose[Layout.POSE_WRIST_R * 4 + 1] = 0.5f
        val nearRight = Hand(FloatArray(63).also { it[0] = 0.31f; it[1] = 0.52f }, null)
        val nearLeft = Hand(FloatArray(63).also { it[0] = 0.69f; it[1] = 0.48f }, null)
        val slots = Slots.assign(listOf(nearRight, nearLeft), pose)
        assertTrue(slots[0] === nearLeft && slots[1] === nearRight)
        val one = Slots.assign(listOf(nearRight), pose)
        assertTrue(one[0] == null && one[1] === nearRight)
    }
}
