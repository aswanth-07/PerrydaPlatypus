package ai.echogesture.signcore

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

/**
 * Ranked output of one classification.
 *
 * setProbs gives, per word, the probability of its whole same-sign set (words that are one sign under several
 * names, such as DOCTOR and PHYSICIAN); it equals probs for a model exported without sets. embedding is the
 * clip's pooled features, used for signs taught on the device; null for a model exported without it. rawProbs is
 * the network's own answer before the same-sign rule; null for a model exported without the rule.
 */
class Prediction(
    val labels: List<String>, val probs: FloatArray, val latencyMs: Double,
    val setProbs: FloatArray = probs, val embedding: FloatArray? = null, val rawProbs: FloatArray? = null,
) {
    val order: List<Int> = probs.indices.sortedByDescending { probs[it] }
    val top: String get() = labels[order[0]]
    val confidence: Float get() = probs[order[0]]
    /** How sure the model is of the sign itself, whichever of its names is shown. */
    val setConfidence: Float get() = setProbs[order[0]]
    fun topK(k: Int): List<Pair<String, Float>> = order.take(k).map { labels[it] to probs[it] }
}

/**
 * The exported INT8 classifier (scripts/export_v2.py). It takes a clip already resampled to tIn steps and
 * returns class probabilities; dominant-hand flip, features, the ensemble average and (when exported with it)
 * the same-sign answer rule are inside the graph. Newer exports also return set_probs and embedding by name.
 */
class SignModel(
    modelBytes: ByteArray, val labels: List<String>, val tIn: Int = 64, threads: Int = 1,
    spin: Boolean = false,   // ONNX Runtime's workers busy-wait between calls by default; off saves power on a phone
) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(threads)
        opts.setInterOpNumThreads(1)
        opts.addConfigEntry("session.intra_op.allow_spinning", if (spin) "1" else "0")
        opts.addConfigEntry("session.inter_op.allow_spinning", if (spin) "1" else "0")
        session = env.createSession(modelBytes, opts)
    }

    fun classify(clip: Clip): Prediction {
        val c = if (clip.t == tIn) clip else Resampler.resample(clip, tIn)
        val start = System.nanoTime()
        val inputs = mapOf(
            "points" to OnnxTensor.createTensor(env, FloatBuffer.wrap(c.points), longArrayOf(1, tIn.toLong(), Layout.P.toLong(), 3)),
            "world" to OnnxTensor.createTensor(env, FloatBuffer.wrap(c.world), longArrayOf(1, tIn.toLong(), 2, 21, 3)),
            "seen" to OnnxTensor.createTensor(env, FloatBuffer.wrap(c.seen), longArrayOf(1, tIn.toLong(), 4)),
        )
        val prediction = session.run(inputs).use { result ->
            @Suppress("UNCHECKED_CAST")
            fun row(name: String): FloatArray? = result.get(name).orElse(null)?.let { (it.value as Array<FloatArray>)[0] }
            @Suppress("UNCHECKED_CAST")
            val probs = (result[0].value as Array<FloatArray>)[0]
            Prediction(labels, probs, (System.nanoTime() - start) / 1e6, row("set_probs") ?: probs, row("embedding"), row("raw_probs"))
        }
        inputs.values.forEach { it.close() }
        return prediction
    }

    /** Raw frames from a whole clip, as predict(video) sees them: at most maxFrames evenly spaced. */
    fun classifyFrames(frames: List<RawFrame>, width: Int, height: Int, maxFrames: Int = 96): Prediction {
        val picked = if (frames.size <= maxFrames) frames else
            (0 until maxFrames).map { frames[Math.round(it * (frames.size - 1).toDouble() / (maxFrames - 1)).toInt()] }
        return classify(Normaliser.normalise(picked, width, height))
    }

    override fun close() = session.close()
}
