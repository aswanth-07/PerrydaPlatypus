package ai.echogesture.signcore

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.InputStream
import java.io.OutputStream
import kotlin.math.sqrt

/**
 * Words that are one sign under several names (DOCTOR and PHYSICIAN), from the export's sets.json: one set id per
 * label, and labels that share an id are the same sign. Built from WLASL's train and validation listings only.
 */
class SameSignSets(private val setOf: IntArray) {
    private val members: Map<Int, List<Int>> = setOf.indices.groupBy { setOf[it] }

    /** The other names of label i, in label order; empty when the sign has one name. */
    fun others(i: Int): List<Int> = if (i in setOf.indices) members[setOf[i]].orEmpty().filter { it != i } else emptyList()

    /**
     * What a caption can offer beside the word it shows: walking the model's ranking `order` (best first), the best
     * label of each further sign, `k` of them. Names of the shown word's own sign are skipped (the caption already
     * lists them as "same sign"), and so is every later name of a sign already offered. scripts/alternatives_eval.py
     * measures the same rule.
     */
    fun alternatives(order: List<Int>, shown: Int, k: Int): List<Int> {
        if (shown !in setOf.indices || k <= 0) return emptyList()
        val seen = hashSetOf(setOf[shown])
        val out = ArrayList<Int>(k)
        for (j in order) {
            if (j !in setOf.indices || !seen.add(setOf[j])) continue
            out += j
            if (out.size == k) break
        }
        return out
    }
}

/** Pooled clip embeddings (the model's `embedding` output) compared by cosine. */
object Embeddings {
    fun unit(v: FloatArray): FloatArray {
        var s = 0.0
        for (x in v) s += x.toDouble() * x
        val n = sqrt(s).toFloat()
        return if (n > 0f) FloatArray(v.size) { v[it] / n } else v.copyOf()
    }

    fun cosine(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "embedding sizes differ: ${a.size} vs ${b.size}" }
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        for (i in a.indices) {
            dot += a[i].toDouble() * b[i]
            na += a[i].toDouble() * a[i]
            nb += b[i].toDouble() * b[i]
        }
        return if (na == 0.0 || nb == 0.0) 0f else (dot / sqrt(na * nb)).toFloat()
    }

    /** A taught sign's prototype: the unit mean of its examples' unit embeddings (as scripts/fewshot_eval.py). */
    fun prototype(examples: List<FloatArray>): FloatArray {
        require(examples.isNotEmpty()) { "a taught sign needs at least one example" }
        val sum = FloatArray(examples[0].size)
        for (e in examples) {
            val u = unit(e)
            for (i in sum.indices) sum[i] += u[i]
        }
        return unit(sum)
    }
}

/**
 * Signs the user taught on the device. A window or clip is the taught word when its embedding is at least
 * `threshold` cosine from that word's prototype (the closest one wins); otherwise the classifier answers.
 * Offline, 20 new signs with 5 examples each from other signers: 92.5% right; at 0.55, 2% of known-word clips
 * trigger a taught sign and 80.5% of taught-sign clips are caught (scripts/fewshot_eval.py).
 */
class TaughtMatcher(val signs: List<Pair<String, FloatArray>>, @Volatile var threshold: Float = DEFAULT_THRESHOLD) {
    class Match(val word: String, val cosine: Float)

    /** The closest taught sign whatever its cosine, or null when nothing is taught or there is no embedding. */
    fun best(embedding: FloatArray?): Match? {
        if (embedding == null || signs.isEmpty()) return null
        var bestWord: String? = null
        var bestCos = -2f
        for ((word, proto) in signs) {
            if (proto.size != embedding.size) continue
            val c = Embeddings.cosine(embedding, proto)
            if (c > bestCos) { bestCos = c; bestWord = word }
        }
        return bestWord?.let { Match(it, bestCos) }
    }

    fun match(embedding: FloatArray?): Match? = best(embedding)?.takeIf { it.cosine >= threshold }

    companion object {
        const val DEFAULT_THRESHOLD = 0.55f
    }
}

/** A clip saved as raw floats: the model input of one taught example (no video is ever stored). */
object ClipCodec {
    private const val MAGIC = 0x45474331   // "EGC1"

    fun write(clip: Clip, out: OutputStream) {
        val d = DataOutputStream(out.buffered())
        d.writeInt(MAGIC)
        d.writeInt(clip.t)
        for (a in listOf(clip.points, clip.world, clip.seen)) {
            d.writeInt(a.size)
            for (x in a) d.writeFloat(x)
        }
        d.flush()
    }

    fun read(input: InputStream): Clip {
        val d = DataInputStream(input.buffered())
        require(d.readInt() == MAGIC) { "not a saved clip" }
        val t = d.readInt()
        fun arr(): FloatArray = FloatArray(d.readInt()) { d.readFloat() }
        val points = arr()
        val world = arr()
        val seen = arr()
        require(points.size == t * Layout.P * 3 && world.size == t * 2 * 21 * 3 && seen.size == t * 4) { "saved clip has the wrong shape" }
        return Clip(t, points, world, seen)
    }
}
