package ai.echogesture.signcore

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Ranked output of one classification.
 *
 * setProbs gives, per word, the probability of its whole same-sign set (words that are one sign under several
 * names, such as DOCTOR and PHYSICIAN); it equals probs for a model exported without sets. embedding is the
 * clip's pooled features, used for signs taught on the device; null for a model exported without it. rawProbs is
 * the network's own answer before the same-sign rule; null for a model exported without the rule. none is the
 * probability that the window holds no sign at its centre (the movement between two signs); 0 for a model
 * without that class. With it, probs sums to 1 - none.
 */
class Prediction(
    val labels: List<String>, val probs: FloatArray, val latencyMs: Double,
    val setProbs: FloatArray = probs, val embedding: FloatArray? = null, val rawProbs: FloatArray? = null,
    val none: Float = 0f,
) {
    /** Full ranking across all labels, computed lazily only when needed (e.g. SameSignSets.alternatives). */
    val order: List<Int> by lazy { probs.indices.sortedByDescending { probs[it] } }
    private val top5Idx: IntArray = selectTopK(probs, 5)
    val topIndex: Int get() = if (top5Idx.isNotEmpty()) top5Idx[0] else 0
    val top: String get() = labels[topIndex]
    val confidence: Float get() = probs[topIndex]
    /** How sure the model is of the sign itself, whichever of its names is shown. */
    val setConfidence: Float get() = setProbs[topIndex]
    /** Zero-boxing top-k label indices, descending by probability with ties broken to the lower index. */
    fun topIndices(k: Int): IntArray = when {
        k <= 0 || probs.isEmpty() -> IntArray(0)
        k == top5Idx.size -> top5Idx
        k < top5Idx.size -> top5Idx.copyOf(k)
        k <= 16 -> selectTopK(probs, k)
        else -> order.take(k).toIntArray()
    }
    fun topK(k: Int): List<Pair<String, Float>> {
        val idx = topIndices(k)
        return List(idx.size) { i -> val j = idx[i]; labels[j] to probs[j] }
    }

    companion object {
        private fun selectTopK(probs: FloatArray, k: Int): IntArray {
            val n = minOf(k, probs.size)
            if (n <= 0) return IntArray(0)
            val idx = IntArray(n) { -1 }
            val vals = FloatArray(n) { Float.NEGATIVE_INFINITY }
            var filled = 0
            for (i in probs.indices) {
                val p = probs[i]
                if (filled == n && p <= vals[n - 1]) continue
                var pos = if (filled < n) filled++ else n - 1
                // Strict > preserves stable lower-index tie breaking (matching sortedByDescending).
                while (pos > 0 && p > vals[pos - 1]) {
                    vals[pos] = vals[pos - 1]
                    idx[pos] = idx[pos - 1]
                    pos--
                }
                vals[pos] = p
                idx[pos] = i
            }
            return idx
        }
    }
}

/**
 * The exported INT8 classifier (scripts/export.py). It takes a clip already resampled to tIn steps and
 * returns class probabilities; dominant-hand flip, features, the ensemble average and (when exported with it)
 * the same-sign answer rule are inside the graph. Newer exports also return set_probs and embedding by name.
 */
class SignModel(
    modelBytes: ByteArray, val labels: List<String>, val tIn: Int = 64, threads: Int = 1,
    spin: Boolean = false,   // ONNX Runtime's workers busy-wait between calls by default; off saves power on a phone
) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val outputNames: Set<String>
    private val primaryOutput: String
    private val ptsShape = longArrayOf(1, tIn.toLong(), Layout.P.toLong(), 3)
    private val wldShape = longArrayOf(1, tIn.toLong(), 2, 21, 3)
    private val seenShape = longArrayOf(1, tIn.toLong(), 4)

    // Reusable direct buffers per thread eliminate 108 KB heap-to-native copies per call with zero lock contention.
    private class Scratch(tIn: Int) {
        val pts: FloatBuffer = ByteBuffer.allocateDirect(tIn * Layout.P * 3 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        val wld: FloatBuffer = ByteBuffer.allocateDirect(tIn * 2 * 21 * 3 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        val seen: FloatBuffer = ByteBuffer.allocateDirect(tIn * 4 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    }
    private val scratch = ThreadLocal.withInitial { Scratch(tIn) }

    init {
        session = OrtSession.SessionOptions().use { opts ->
            opts.setIntraOpNumThreads(threads)
            opts.setInterOpNumThreads(1)
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            opts.setMemoryPatternOptimization(true)
            opts.addConfigEntry("session.intra_op.allow_spinning", if (spin) "1" else "0")
            opts.addConfigEntry("session.inter_op.allow_spinning", if (spin) "1" else "0")
            opts.addConfigEntry("session.set_denormal_as_zero", "1")
            env.createSession(modelBytes, opts)
        }
        outputNames = session.outputNames
        primaryOutput = if ("probs" in outputNames) "probs" else outputNames.first()
    }

    fun classify(clip: Clip, needEmbedding: Boolean = true, needRaw: Boolean = true): Prediction {
        val c = if (clip.t == tIn) clip else Resampler.resample(clip, tIn)
        val start = System.nanoTime()
        val buf = scratch.get()!!
        buf.pts.clear(); buf.pts.put(c.points); buf.pts.flip()
        buf.wld.clear(); buf.wld.put(c.world); buf.wld.flip()
        buf.seen.clear(); buf.seen.put(c.seen); buf.seen.flip()
        var tPts: OnnxTensor? = null
        var tWld: OnnxTensor? = null
        var tSeen: OnnxTensor? = null
        try {
            tPts = OnnxTensor.createTensor(env, buf.pts, ptsShape)
            tWld = OnnxTensor.createTensor(env, buf.wld, wldShape)
            tSeen = OnnxTensor.createTensor(env, buf.seen, seenShape)
            val inputs = HashMap<String, OnnxTensor>(4).apply {
                put("points", tPts)
                put("world", tWld)
                put("seen", tSeen)
            }
            val want = HashSet<String>(6).apply {
                add(primaryOutput)
                if ("set_probs" in outputNames) add("set_probs")
                if (needEmbedding && "embedding" in outputNames) add("embedding")
                if (needRaw && "raw_probs" in outputNames) add("raw_probs")
                if ("none" in outputNames) add("none")
            }
            return session.run(inputs, want).use { result ->
                fun readFloats(name: String): FloatArray? {
                    val opt = result.get(name)
                    if (!opt.isPresent) return null
                    val fb = (opt.get() as OnnxTensor).floatBuffer
                    return FloatArray(fb.remaining()).also { fb.get(it) }
                }
                val probs = readFloats(primaryOutput)!!
                Prediction(
                    labels, probs, (System.nanoTime() - start) / 1e6,
                    readFloats("set_probs") ?: probs,
                    if (needEmbedding) readFloats("embedding") else null,
                    if (needRaw) readFloats("raw_probs") else null,
                    readFloats("none")?.get(0) ?: 0f,
                )
            }
        } finally {
            tPts?.close()
            tWld?.close()
            tSeen?.close()
        }
    }

    /** Raw frames from a whole clip, as predict(video) sees them: at most maxFrames evenly spaced. */
    fun classifyFrames(frames: List<RawFrame>, width: Int, height: Int, maxFrames: Int = 96): Prediction {
        val picked = if (frames.size <= maxFrames) frames else
            (0 until maxFrames).map { frames[Math.round(it * (frames.size - 1).toDouble() / (maxFrames - 1)).toInt()] }
        return classify(Normaliser.normalise(picked, width, height), needRaw = false)
    }

    override fun close() = session.close()
}
