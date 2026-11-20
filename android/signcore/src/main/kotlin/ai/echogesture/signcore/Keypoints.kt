package ai.echogesture.signcore

/**
 * Point layout and normalisation, a line-for-line port of src/normalise.py and holistic_kp.py.
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

/** A detected hand before it is given a slot. [preferredSlot] is MediaPipe's native side (0 left, 1 right, -1 unknown). */
class Hand(val image: FloatArray, val world: FloatArray?, val preferredSlot: Int = -1)

object Slots {
    /** Put detected hands in slot 0 (person's left) or 1 (right) by distance to the pose wrists, not by handedness. */
    fun assign(found: List<Hand>, pose: FloatArray?): Array<Hand?> {
        val out = arrayOfNulls<Hand>(2)
        if (found.isEmpty()) return out
        val wl = Layout.POSE_WRIST_L * 4
        val wr = Layout.POSE_WRIST_R * 4
        val validWrists = pose != null && pose.size > wr + 1 &&
            pose[wl].isFinite() && pose[wl + 1].isFinite() && pose[wr].isFinite() && pose[wr + 1].isFinite()
        if (!validWrists) {
            val rest = ArrayList<Hand>(2)
            for (i in 0 until minOf(2, found.size)) {
                val h = found[i]
                if (h.preferredSlot in 0..1 && out[h.preferredSlot] == null) out[h.preferredSlot] = h else rest.add(h)
            }
            for (h in rest) {
                if (out[0] == null) out[0] = h else if (out[1] == null) out[1] = h
            }
            return out
        }
        val p = pose
        fun cost(h: Hand, wrist: Int): Double {
            val dx = (h.image[0] - p[wrist * 4]).toDouble()
            val dy = (h.image[1] - p[wrist * 4 + 1]).toDouble()
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
        val ax = width / side
        val ay = height / side
        val az = ax
        val pts = DoubleArray(t * p * 3) { Double.NaN }
        val seenByGroup = Array(4) { BooleanArray(t) }
        val poseKeep = Layout.POSE_KEEP
        for (f in 0 until t) {
            val fr = frames[f]
            val fBase = f * p
            fr.pose?.let { pose ->
                for (k in poseKeep.indices) {
                    val src = poseKeep[k] * 4
                    val dst = (fBase + k) * 3
                    pts[dst] = pose[src] * ax
                    pts[dst + 1] = pose[src + 1] * ay
                    pts[dst + 2] = pose[src + 2] * az
                }
                seenByGroup[0][f] = true
            }
            for (s in 0 until 2) fr.hands[s]?.let { h ->
                val base = fBase + (if (s == 0) Layout.LH else Layout.RH)
                for (k in 0 until 21) {
                    val src = k * 3
                    val dst = (base + k) * 3
                    pts[dst] = h[src] * ax
                    pts[dst + 1] = h[src + 1] * ay
                    pts[dst + 2] = h[src + 2] * az
                }
                seenByGroup[s + 1][f] = true
            }
            fr.face?.let { face ->
                val base = fBase + Layout.FACE
                for (k in 0 until Layout.NFACE) {
                    val src = k * 3
                    val dst = (base + k) * 3
                    pts[dst] = face[src] * ax
                    pts[dst + 1] = face[src + 1] * ay
                    pts[dst + 2] = face[src + 2] * az
                }
                seenByGroup[3][f] = true
            }
        }

        // Centre on the mean shoulder midpoint, scale by the median shoulder width.
        var mid0 = 0.0
        var mid1 = 0.0
        var mid2 = 0.0
        var midCount = 0
        val widths = DoubleArray(t)
        for (f in 0 until t) {
            val a = (f * p + Layout.SHOULDER_L) * 3
            val b = (f * p + Layout.SHOULDER_R) * 3
            val ax0 = pts[a]
            val bx0 = pts[b]
            if (!ax0.isNaN() && !bx0.isNaN()) {
                val ay0 = pts[a + 1]
                val by0 = pts[b + 1]
                mid0 += (ax0 + bx0) / 2.0
                mid1 += (ay0 + by0) / 2.0
                mid2 += (pts[a + 2] + pts[b + 2]) / 2.0
                val dx = ax0 - bx0
                val dy = ay0 - by0
                widths[midCount++] = Math.sqrt(dx * dx + dy * dy)
            }
        }
        val c0: Double
        val c1: Double
        val c2: Double
        val scale: Double
        if (midCount > 0) {
            c0 = mid0 / midCount
            c1 = mid1 / midCount
            c2 = mid2 / midCount
            scale = medianInPlace(widths, midCount)
        } else {
            val buf = DoubleArray(t * p)
            fun colMedian(c: Int): Double {
                var cnt = 0
                for (i in 0 until t * p) {
                    val v = pts[i * 3 + c]
                    if (!v.isNaN()) buf[cnt++] = v
                }
                return if (cnt == 0) 0.0 else medianInPlace(buf, cnt)
            }
            c0 = colMedian(0)
            c1 = colMedian(1)
            c2 = colMedian(2)
            scale = 0.25
        }
        val div = maxOf(scale, 1e-2)
        val nPts = t * p
        var idx = 0
        for (i in 0 until nPts) {
            pts[idx] = (pts[idx] - c0) / div
            pts[idx + 1] = (pts[idx + 1] - c1) / div
            pts[idx + 2] = (pts[idx + 2] - c2) / div
            idx += 3
        }

        for (g in Layout.GROUPS.indices) {
            val range = Layout.GROUPS[g]
            fillGroup(pts, t, p, range[0], range[1], seenByGroup[g])
        }
        val out = FloatArray(pts.size) { i -> clampNan(pts[i]) }

        val world = FloatArray(t * 2 * 21 * 3)
        val w = DoubleArray(t * 63)
        val ok = BooleanArray(t)
        for (s in 0 until 2) {
            java.util.Arrays.fill(w, 0.0)
            java.util.Arrays.fill(ok, false)
            for (f in 0 until t) {
                val fr = frames[f]
                val hw = fr.handsWorld[s]
                if (fr.hands[s] != null && hw != null) {
                    ok[f] = true
                    // Holistic gives hand world points in the body's frame; make them wrist-relative.
                    val w0 = hw[0].toDouble()
                    val w1 = hw[1].toDouble()
                    val w2 = hw[2].toDouble()
                    val base = f * 63
                    var k = 0
                    while (k < 63) {
                        w[base + k] = (hw[k] - w0) * Layout.WORLD_SCALE
                        w[base + k + 1] = (hw[k + 1] - w1) * Layout.WORLD_SCALE
                        w[base + k + 2] = (hw[k + 2] - w2) * Layout.WORLD_SCALE
                        k += 3
                    }
                }
            }
            fillGroup(w, t, 21, 0, 21, ok)
            for (f in 0 until t) {
                val srcBase = f * 63
                val dstBase = (f * 2 + s) * 63
                for (k in 0 until 63) world[dstBase + k] = clampNan(w[srcBase + k])
            }
        }
        val seenOut = FloatArray(t * 4)
        for (f in 0 until t) {
            val base = f * 4
            seenOut[base] = if (seenByGroup[0][f]) 1f else 0f
            seenOut[base + 1] = if (seenByGroup[1][f]) 1f else 0f
            seenOut[base + 2] = if (seenByGroup[2][f]) 1f else 0f
            seenOut[base + 3] = if (seenByGroup[3][f]) 1f else 0f
        }
        return Clip(t, out, world, seenOut)
    }

    /** Linear interpolation over time for points [lo, hi) of a T x n x 3 array, seen as a whole per frame. */
    fun fillGroup(a: DoubleArray, t: Int, n: Int, lo: Int, hi: Int, ok: BooleanArray) {
        var anySeen = false
        var allSeen = true
        for (f in 0 until t) {
            if (ok[f]) anySeen = true else allSeen = false
        }
        if (allSeen) return
        val stride = n * 3
        val lo3 = lo * 3
        val hi3 = hi * 3
        if (!anySeen) {
            for (f in 0 until t) {
                val base = f * stride
                for (k in lo3 until hi3) a[base + k] = 0.0
            }
            return
        }
        val prev = IntArray(t)
        val next = IntArray(t)
        var last = -1
        for (f in 0 until t) { if (ok[f]) last = f; prev[f] = last }
        last = t
        for (f in t - 1 downTo 0) { if (ok[f]) last = f; next[f] = last }
        // Zero-copy in place: only unseen frames (!ok[f]) are written, and pp/nn are always seen frames (ok == true).
        for (f in 0 until t) {
            if (ok[f]) continue
            val pp = if (prev[f] >= 0) prev[f] else next[f]
            val nn = if (next[f] < t) next[f] else prev[f]
            val fBase = f * stride
            val pBase = pp * stride
            if (nn == pp) {
                System.arraycopy(a, pBase + lo3, a, fBase + lo3, hi3 - lo3)
            } else {
                val w = (f - pp).toDouble() / (nn - pp)
                val oneMinusW = 1.0 - w
                val nBase = nn * stride
                for (k in lo3 until hi3) {
                    a[fBase + k] = a[pBase + k] * oneMinusW + a[nBase + k] * w
                }
            }
        }
    }

    private fun clampNan(v: Double): Float = if (v.isNaN()) 0f else v.coerceIn(-Layout.CLAMP, Layout.CLAMP).toFloat()

    private fun medianInPlace(v: DoubleArray, size: Int): Double {
        java.util.Arrays.sort(v, 0, size)
        val m = size / 2
        return if (size % 2 == 1) v[m] else (v[m - 1] + v[m]) / 2.0
    }
}

object Resampler {
    /** Linear resampling of a clip to tIn steps, as train.sample_time does at test time. */
    fun resample(clip: Clip, tIn: Int): Clip {
        val t = clip.t
        val maxIdx = maxOf(t - 1, 0)
        val loArr = IntArray(tIn)
        val hiArr = IntArray(tIn)
        val aArr = FloatArray(tIn)
        val denom = if (tIn == 1) 1.0 else (tIn - 1).toDouble()
        for (i in 0 until tIn) {
            val pos = if (tIn == 1) 0.0 else i / denom * maxIdx
            val lo = Math.floor(pos).toInt()
            loArr[i] = lo
            hiArr[i] = minOf(lo + 1, maxIdx)
            aArr[i] = (pos - lo).toFloat()
        }
        fun lerp(src: FloatArray, stride: Int): FloatArray {
            val out = FloatArray(tIn * stride)
            for (i in 0 until tIn) {
                val lo = loArr[i]
                val hi = hiArr[i]
                val dstBase = i * stride
                val loBase = lo * stride
                val a = aArr[i]
                if (lo == hi || a == 0f) {
                    System.arraycopy(src, loBase, out, dstBase, stride)
                } else {
                    val hiBase = hi * stride
                    for (k in 0 until stride) {
                        val z0 = src[loBase + k]
                        out[dstBase + k] = z0 + (src[hiBase + k] - z0) * a
                    }
                }
            }
            return out
        }
        return Clip(tIn, lerp(clip.points, Layout.P * 3), lerp(clip.world, 2 * 21 * 3), lerp(clip.seen, 4))
    }
}
