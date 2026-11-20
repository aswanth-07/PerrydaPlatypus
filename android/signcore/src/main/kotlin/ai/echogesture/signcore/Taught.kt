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

    private val unitSigns: List<Pair<String, FloatArray>> = signs.map { (w, p) -> w to Embeddings.unit(p) }

    /** The closest taught sign whatever its cosine, or null when nothing is taught or there is no embedding. */
    fun best(embedding: FloatArray?): Match? {
        if (embedding == null || unitSigns.isEmpty()) return null
        val q = Embeddings.unit(embedding)
        var qNorm = 0.0
        for (x in q) qNorm += x.toDouble() * x
        val hasQuery = qNorm > 0.0
        var bestWord: String? = null
        var bestCos = -2f
        for ((word, proto) in unitSigns) {
            if (proto.size != q.size) continue
            val c = if (!hasQuery) 0f else {
                var dot = 0.0
                for (i in q.indices) dot += q[i].toDouble() * proto[i]
                dot.toFloat()
            }
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

/**
 * The core of a signed clip: the stretch between the hands' rise and their drop, found from the wrists' height
 * alone. A port of scripts/contsign.py `core_span`, used so that a taught sign's examples (recorded from rest to
 * rest) can be shown to the live model the way a live window shows a sign: centred on the sign itself.
 */
object Cores {
    const val REST_H = -1.0f
    const val FRAC = 0.65f

    /** (s0, a, b, s1): the stretch above the rest line is [s0, s1) and the core [a, b). h: the higher wrist's height per step. */
    fun span(h: FloatArray, restH: Float = REST_H, frac: Float = FRAC, minLen: Int = 5): IntArray {
        val t = h.size
        var s0 = -1
        var s1 = -1
        for (i in 0 until t) if (h[i] > restH) { if (s0 < 0) s0 = i; s1 = i + 1 }
        if (s0 < 0) return intArrayOf(0, 0, t, t)
        val sorted = h.copyOfRange(s0, s1).also { it.sort() }
        val pos = 0.9f * (sorted.size - 1)
        val lo = pos.toInt()
        val peak = sorted[lo] + (sorted[minOf(lo + 1, sorted.size - 1)] - sorted[lo]) * (pos - lo)
        val lo0 = minOf(h[s0], peak)
        val lo1 = minOf(h[s1 - 1], peak)
        var a = s0
        while (a < s1 - 1 && h[a] < lo0 + frac * (peak - lo0)) a++
        var b = s1
        while (b > s0 + 1 && h[b - 1] < lo1 + frac * (peak - lo1)) b--
        if (b - a < maxOf(minLen.toFloat(), 0.35f * (s1 - s0))) { a = s0; b = s1 }
        return intArrayOf(s0, a, b, s1)
    }

    /** The higher wrist's height above the shoulder line at every step of a normalised clip, in shoulder widths. */
    fun heights(clip: Clip): FloatArray {
        val p = Layout.P
        fun y(t: Int, k: Int) = clip.points[(t * p + k) * 3 + 1]
        val wl = Layout.POSE_KEEP.indexOf(Layout.POSE_WRIST_L)
        val wr = Layout.POSE_KEEP.indexOf(Layout.POSE_WRIST_R)
        return FloatArray(clip.t) { t ->
            val mid = (y(t, Layout.SHOULDER_L) + y(t, Layout.SHOULDER_R)) / 2
            maxOf(mid - y(t, wl), mid - y(t, wr))
        }
    }

    /** The steps of [clip] a live window `share` of its length would hold when centred on the sign's core. */
    fun window(clip: Clip, share: Float): Clip = windowShifted(clip, share, 0f)

    /**
     * Multi-offset crops around the sign's core (centre, slight lead-in, slight lead-out) so a taught sign's
     * live prototype tolerates real-world timing shifts in continuous signing.
     */
    fun windows(clip: Clip, share: Float): List<Clip> {
        val w = maxOf(8, Math.round(share * clip.t))
        if (w >= clip.t) return listOf(clip)
        val s = span(heights(clip))
        val centre = (s[1] + s[2] - 1) / 2f
        val shift = maxOf(2f, w * 0.14f)
        val starts = linkedSetOf(
            Math.round(centre - (w - 1) / 2f).coerceIn(0, clip.t - w),
            Math.round(centre - shift - (w - 1) / 2f).coerceIn(0, clip.t - w),
            Math.round(centre + shift - (w - 1) / 2f).coerceIn(0, clip.t - w),
        )
        return starts.map { lo -> slice(clip, lo, w) }
    }

    private fun windowShifted(clip: Clip, share: Float, offsetFrac: Float): Clip {
        val w = maxOf(8, Math.round(share * clip.t))
        if (w >= clip.t) return clip
        val s = span(heights(clip))
        val lo = Math.round((s[1] + s[2] - 1) / 2f + offsetFrac * w - (w - 1) / 2f).coerceIn(0, clip.t - w)
        return slice(clip, lo, w)
    }

    private fun slice(clip: Clip, lo: Int, w: Int): Clip {
        val p3 = Layout.P * 3
        return Clip(w, clip.points.copyOfRange(lo * p3, (lo + w) * p3), clip.world.copyOfRange(lo * 126, (lo + w) * 126),
            clip.seen.copyOfRange(lo * 4, (lo + w) * 4))
    }
}
