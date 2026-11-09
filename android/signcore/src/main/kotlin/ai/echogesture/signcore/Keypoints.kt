package ai.echogesture.signcore

/**
 * Point layout and normalisation, a line-for-line port of scripts/data_v2.py and holistic_kp.py.
 * The exported model expects exactly these numbers; ParityTest checks them against Python.
 *
 * Layout per frame (P = 119 points, x y z):
 *   0-14    pose points POSE_KEEP
 *   15-35   person's left hand, 36-56 person's right hand
 *   57-118  face: 40 lips, 4 nose, 18 eye and brow points
 */
object Layout {
    val POSE_KEEP = intArrayOf(0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24)
    val FACE_MESH = intArrayOf(
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        1, 2, 98, 327,
        33, 133, 160, 158, 144, 153, 362, 263, 385, 387, 380, 373, 70, 105, 107, 336, 334, 300,
    )
    const val NPOSE = 15
    const val LH = 15
    const val RH = 36
    const val FACE = 57
    const val P = 119
    const val NFACE = 62
    const val CLAMP = 6.0
    const val WORLD_SCALE = 10.0
    const val SHOULDER_L = 7   // POSE_KEEP index of pose 11
    const val SHOULDER_R = 8   // POSE_KEEP index of pose 12
    const val POSE_WRIST_L = 15
    const val POSE_WRIST_R = 16
    val GROUPS = arrayOf(intArrayOf(0, NPOSE), intArrayOf(LH, LH + 21), intArrayOf(RH, RH + 21), intArrayOf(FACE, FACE + NFACE))
}

/** One frame from the Holistic landmarker, normalised image coordinates; null means not seen. */
class RawFrame(
    val pose: FloatArray?,              // 33 x 4: x, y, z, visibility
    val hands: Array<FloatArray?>,      // [left, right], each 21 x 3
    val handsWorld: Array<FloatArray?>, // [left, right], each 21 x 3, metres
    val face: FloatArray?,              // 62 x 3, already reduced to Layout.FACE_MESH
)

/** A detected hand before it is given a slot. */
class Hand(val image: FloatArray, val world: FloatArray?)

object Slots {
    /** Put detected hands in slot 0 (person's left) or 1 (right) by distance to the pose wrists, not by handedness. */
    fun assign(found: List<Hand>, pose: FloatArray?): Array<Hand?> {
        val out = arrayOfNulls<Hand>(2)
        if (found.isEmpty()) return out
        if (pose == null) {
            found.take(2).forEachIndexed { k, h -> out[k] = h }
            return out
        }
        fun cost(h: Hand, wrist: Int): Double {
            val dx = (h.image[0] - pose[wrist * 4]).toDouble()
            val dy = (h.image[1] - pose[wrist * 4 + 1]).toDouble()
            return Math.sqrt(dx * dx + dy * dy)
        }
        val wrists = intArrayOf(Layout.POSE_WRIST_L, Layout.POSE_WRIST_R)
        if (found.size == 1) {
            val h = found[0]
            out[if (cost(h, wrists[0]) <= cost(h, wrists[1])) 0 else 1] = h
            return out
        }
        val a = found[0]
        val b = found[1]
        val straight = cost(a, wrists[0]) + cost(b, wrists[1])
        val crossed = cost(a, wrists[1]) + cost(b, wrists[0])
        if (straight <= crossed) { out[0] = a; out[1] = b } else { out[0] = b; out[1] = a }
        return out
    }
}

/** Normalised clip: points T x P x 3, world T x 2 x 21 x 3, seen T x 4, all flat row-major. */
class Clip(val t: Int, val points: FloatArray, val world: FloatArray, val seen: FloatArray)

object Normaliser {
    fun normalise(frames: List<RawFrame>, width: Int, height: Int): Clip {
        val t = frames.size
        val p = Layout.P
        val side = maxOf(width, height).toDouble()
        val aspect = doubleArrayOf(width / side, height / side, width / side)
        val pts = DoubleArray(t * p * 3) { Double.NaN }
        val seen = Array(t) { BooleanArray(4) }
        for ((f, fr) in frames.withIndex()) {
            fr.pose?.let { pose ->
                for ((k, id) in Layout.POSE_KEEP.withIndex()) for (c in 0 until 3) pts[(f * p + k) * 3 + c] = pose[id * 4 + c] * aspect[c]
                seen[f][0] = true
            }
            for (s in 0 until 2) fr.hands[s]?.let { h ->
                val base = if (s == 0) Layout.LH else Layout.RH
                for (k in 0 until 21) for (c in 0 until 3) pts[(f * p + base + k) * 3 + c] = h[k * 3 + c] * aspect[c]
                seen[f][s + 1] = true
            }
            fr.face?.let { face ->
                for (k in 0 until Layout.NFACE) for (c in 0 until 3) pts[(f * p + Layout.FACE + k) * 3 + c] = face[k * 3 + c] * aspect[c]
                seen[f][3] = true
            }
        }

        // Centre on the mean shoulder midpoint, scale by the median shoulder width.
        val mids = ArrayList<DoubleArray>()
        val widths = ArrayList<Double>()
        for (f in 0 until t) {
            val a = (f * p + Layout.SHOULDER_L) * 3
            val b = (f * p + Layout.SHOULDER_R) * 3
            if (!pts[a].isNaN() && !pts[b].isNaN()) {
                mids.add(DoubleArray(3) { (pts[a + it] + pts[b + it]) / 2 })
                val dx = pts[a] - pts[b]
                val dy = pts[a + 1] - pts[b + 1]
                widths.add(Math.sqrt(dx * dx + dy * dy))
            }
        }
        val centre: DoubleArray
        val scale: Double
        if (mids.isNotEmpty()) {
            centre = DoubleArray(3) { c -> mids.sumOf { it[c] } / mids.size }
            scale = median(widths)
        } else {
            centre = DoubleArray(3) { c ->
                val v = (0 until t * p).map { pts[it * 3 + c] }.filter { !it.isNaN() }
                if (v.isEmpty()) 0.0 else median(v)
            }
            scale = 0.25
        }
        val div = maxOf(scale, 1e-2)
        for (i in 0 until t * p) for (c in 0 until 3) pts[i * 3 + c] = (pts[i * 3 + c] - centre[c]) / div

        for ((g, range) in Layout.GROUPS.withIndex()) fillGroup(pts, t, p, range[0], range[1], BooleanArray(t) { seen[it][g] })
        val out = FloatArray(t * p * 3) { i -> clampNan(pts[i]) }

        val world = FloatArray(t * 2 * 21 * 3)
        for (s in 0 until 2) {
            val w = DoubleArray(t * 21 * 3)
            val ok = BooleanArray(t)
            for ((f, fr) in frames.withIndex()) {
                val hw = fr.handsWorld[s]
                if (fr.hands[s] != null && hw != null) {
                    ok[f] = true
                    // Holistic gives hand world points in the body's frame; make them wrist-relative.
                    for (k in 0 until 63) w[f * 63 + k] = (hw[k] - hw[k % 3]) * Layout.WORLD_SCALE
                }
            }
            fillGroup(w, t, 21, 0, 21, ok)
            for (f in 0 until t) for (k in 0 until 63) world[((f * 2 + s) * 21) * 3 + k] = clampNan(w[f * 63 + k])
        }
        val seenOut = FloatArray(t * 4) { i -> if (seen[i / 4][i % 4]) 1f else 0f }
        return Clip(t, out, world, seenOut)
    }

    /** Linear interpolation over time for points [lo, hi) of a T x n x 3 array, seen as a whole per frame. */
    fun fillGroup(a: DoubleArray, t: Int, n: Int, lo: Int, hi: Int, ok: BooleanArray) {
        if (ok.all { it }) return
        if (ok.none { it }) {
            for (f in 0 until t) for (k in lo until hi) for (c in 0 until 3) a[(f * n + k) * 3 + c] = 0.0
            return
        }
        val prev = IntArray(t)
        val next = IntArray(t)
        var last = -1
        for (f in 0 until t) { if (ok[f]) last = f; prev[f] = last }
        last = t
        for (f in t - 1 downTo 0) { if (ok[f]) last = f; next[f] = last }
        val src = a.copyOf()
        for (f in 0 until t) {
            val pp = if (prev[f] >= 0) prev[f] else next[f]
            val nn = if (next[f] < t) next[f] else prev[f]
            val w = if (nn != pp) (f - pp).toDouble() / maxOf(nn - pp, 1) else 0.0
            for (k in lo until hi) for (c in 0 until 3) {
                val x0 = src[(pp * n + k) * 3 + c]
                val x1 = src[(nn * n + k) * 3 + c]
                a[(f * n + k) * 3 + c] = x0 * (1 - w) + x1 * w
            }
        }
    }

    private fun clampNan(v: Double): Float = if (v.isNaN()) 0f else v.coerceIn(-Layout.CLAMP, Layout.CLAMP).toFloat()

    private fun median(v: List<Double>): Double {
        val s = v.sorted()
        val m = s.size / 2
        return if (s.size % 2 == 1) s[m] else (s[m - 1] + s[m]) / 2
    }
}

object Resampler {
    /** Linear resampling of a clip to tIn steps, as train_v2.sample_time does at test time. */
    fun resample(clip: Clip, tIn: Int): Clip {
        val t = clip.t
        fun lerp(src: FloatArray, stride: Int): FloatArray {
            val out = FloatArray(tIn * stride)
            for (i in 0 until tIn) {
                val pos = if (tIn == 1) 0.0 else i.toDouble() / (tIn - 1) * maxOf(t - 1, 0)
                val lo = Math.floor(pos).toInt()
                val hi = minOf(lo + 1, maxOf(t - 1, 0))
                val a = (pos - lo).toFloat()
                for (k in 0 until stride) {
                    val z0 = src[lo * stride + k]
                    val z1 = src[hi * stride + k]
                    out[i * stride + k] = z0 + (z1 - z0) * a
                }
            }
            return out
        }
        return Clip(tIn, lerp(clip.points, Layout.P * 3), lerp(clip.world, 2 * 21 * 3), lerp(clip.seen, 4))
    }
}
