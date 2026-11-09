package ai.echogesture.signcore

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

/** Ranked output of one classification. */
class Prediction(val labels: List<String>, val probs: FloatArray, val latencyMs: Double) {
    val order: List<Int> = probs.indices.sortedByDescending { probs[it] }
    val top: String get() = labels[order[0]]
    val confidence: Float get() = probs[order[0]]
    fun topK(k: Int): List<Pair<String, Float>> = order.take(k).map { labels[it] to probs[it] }
}

/**
 * The exported INT8 classifier (scripts/export_v2.py). It takes a clip already resampled to tIn steps and
 * returns class probabilities; dominant-hand flip, features and the ensemble average are inside the graph.
 */
class SignModel(modelBytes: ByteArray, val labels: List<String>, val tIn: Int = 64, threads: Int = 1) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(threads)
        opts.setInterOpNumThreads(1)
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
        val probs = session.run(inputs).use { result ->
            @Suppress("UNCHECKED_CAST")
            (result[0].value as Array<FloatArray>)[0]
        }
        inputs.values.forEach { it.close() }
        return Prediction(labels, probs, (System.nanoTime() - start) / 1e6)
    }

    /** Raw frames from a whole clip, as predict(video) sees them: at most maxFrames evenly spaced. */
    fun classifyFrames(frames: List<RawFrame>, width: Int, height: Int, maxFrames: Int = 96): Prediction {
        val picked = if (frames.size <= maxFrames) frames else
            (0 until maxFrames).map { frames[Math.round(it * (frames.size - 1).toDouble() / (maxFrames - 1)).toInt()] }
        return classify(Normaliser.normalise(picked, width, height))
    }

    override fun close() = session.close()
}
