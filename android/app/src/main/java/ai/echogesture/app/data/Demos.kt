package ai.echogesture.app.data

import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * One sign shown as a moving stick figure: [frames] steps of [points] 2D points (nose, shoulders, elbows, wrists,
 * hips, then 21 points of each hand) in a unit square, x then y per point as bytes 0..254; 255 = not shown.
 */
class Demo(val word: String, val durationMs: Int, val frames: Int, val points: Int, private val data: ByteArray) {
    /** Point [k] at the fractional step [t] (linear between steps); null when the point is not shown. */
    fun at(t: Float, k: Int): Pair<Float, Float>? {
        val f = t.coerceIn(0f, frames - 1f)
        val a = f.toInt()
        val b = minOf(a + 1, frames - 1)
        val w = f - a
        fun v(step: Int, c: Int) = data[(step * points + k) * 2 + c].toInt() and 0xFF
        val xa = v(a, 0); val ya = v(a, 1); val xb = v(b, 0); val yb = v(b, 1)
        if (xa == 255 || xb == 255) return null
        return ((xa + (xb - xa) * w) / 254f) to ((ya + (yb - ya) * w) / 254f)
    }

    /** The box every shown point stays inside over the whole sign, as (x0, y0, x1, y1) of the unit square. */
    val bounds: FloatArray by lazy { boundsOf(0 until points) }

    /** The same box for the hands alone: "Zoom on hands" fills the card with it. */
    val handBounds: FloatArray by lazy { boundsOf(HAND_L until points).takeIf { it[2] > it[0] } ?: bounds }

    private fun boundsOf(range: IntRange): FloatArray {
        var x0 = 1f; var y0 = 1f; var x1 = 0f; var y1 = 0f
        for (step in 0 until frames) for (k in range) {
            val i = step * points + k
            val x = data[i * 2].toInt() and 0xFF
            val y = data[i * 2 + 1].toInt() and 0xFF
            if (x == 255) continue
            x0 = minOf(x0, x / 254f); x1 = maxOf(x1, x / 254f)
            y0 = minOf(y0, y / 254f); y1 = maxOf(y1, y / 254f)
        }
        return if (x1 < x0) floatArrayOf(0f, 0f, 1f, 1f) else floatArrayOf(x0, y0, x1, y1)
    }

    companion object {
        const val NOSE = 0; const val SHOULDER_L = 1; const val SHOULDER_R = 2; const val ELBOW_L = 3; const val ELBOW_R = 4
        const val WRIST_L = 5; const val WRIST_R = 6; const val HIP_L = 7; const val HIP_R = 8
        const val HAND_L = 9; const val HAND_R = 30
    }
}

/**
 * The demonstrations packed with the app (assets/demos.bin, written by scripts/make_demos.py): keypoints of
 * one training clip per word, never video. Loaded once, on first use.
 */
object Demos {
    @Volatile private var cached: Map<String, Demo>? = null

    fun load(context: Context): Map<String, Demo> = cached ?: synchronized(this) {
        cached ?: runCatching { parse(context.assets.open("demos.bin").readBytes()) }.getOrDefault(emptyMap()).also { cached = it }
    }

    fun parse(bytes: ByteArray): Map<String, Demo> {
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(4).also { b.get(it) }
        require(String(magic, Charsets.US_ASCII) == "EGD1") { "not a demonstrations file" }
        val count = b.short.toInt() and 0xFFFF
        val frames = b.get().toInt() and 0xFF
        val points = b.get().toInt() and 0xFF
        val out = LinkedHashMap<String, Demo>(count)
        repeat(count) {
            val name = ByteArray(b.get().toInt() and 0xFF).also { b.get(it) }.toString(Charsets.UTF_8)
            val ms = b.short.toInt() and 0xFFFF
            val data = ByteArray(frames * points * 2).also { b.get(it) }
            out[name] = Demo(name, ms, frames, points, data)
        }
        return out
    }
}
