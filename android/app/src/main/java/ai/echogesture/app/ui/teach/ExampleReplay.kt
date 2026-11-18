package ai.echogesture.app.ui.teach

import ai.echogesture.app.engine.HAND_EDGES
import ai.echogesture.app.ui.theme.Palette
import ai.echogesture.signcore.Clip
import ai.echogesture.signcore.Layout
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.keyframes
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.foundation.Canvas
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import kotlin.math.roundToInt

/**
 * One recorded example played back from its keypoints (the only thing kept; there is no video): shoulders,
 * arms, head and both hands, looping. [mirror] matches the front camera's preview, so it moves the way the person
 * saw themselves while recording.
 */
@Composable
fun ExampleReplay(clip: Clip, mirror: Boolean, modifier: Modifier = Modifier, playMs: Int = 2500) {
    val steps = clip.t
    val step by rememberInfiniteTransition(label = "replay").animateFloat(
        0f, (steps - 1).coerceAtLeast(1).toFloat(),
        infiniteRepeatable(keyframes {
            durationMillis = playMs + 500
            0f at 0 using LinearEasing
            (steps - 1f) at playMs using LinearEasing
            (steps - 1f) at playMs + 500
        }, RepeatMode.Restart),
        label = "step",
    )
    val bounds = remember(clip) { bounds(clip) }
    val body = Palette.TextMuted
    val right = MaterialTheme.colorScheme.primary
    Canvas(modifier.clipToBounds()) {
        val (x0, y0, x1, y1) = bounds
        val margin = 0.12f * maxOf(x1 - x0, y1 - y0)
        val scale = minOf(size.width / (x1 - x0 + 2 * margin), size.height / (y1 - y0 + 2 * margin))
        val cx = (x0 + x1) / 2; val cy = (y0 + y1) / 2
        val f = step.roundToInt().coerceIn(0, steps - 1)
        fun pt(k: Int): Offset? {
            if (!seen(clip, f, k)) return null
            val i = (f * Layout.P + k) * 3
            val dx = (clip.points[i] - cx) * scale
            return Offset(size.width / 2 + if (mirror) -dx else dx, size.height / 2 + (clip.points[i + 1] - cy) * scale)
        }
        val limb = 4.dp.toPx()
        fun bone(a: Int, b: Int, colour: Color, w: Float) {
            val p = pt(a) ?: return
            val q = pt(b) ?: return
            drawLine(colour, p, q, w, StrokeCap.Round)
        }
        val sl = pt(Layout.SHOULDER_L); val sr = pt(Layout.SHOULDER_R)
        if (sl != null && sr != null) {
            drawLine(body.copy(alpha = 0.7f), sl, sr, limb, StrokeCap.Round)
            pt(0)?.let { nose -> drawCircle(body.copy(alpha = 0.7f), (sl - sr).getDistance() * 0.3f, nose, style = Stroke(2.dp.toPx())) }
        }
        for ((a, b) in ARMS) bone(a, b, body.copy(alpha = 0.85f), limb)
        for ((base, colour) in listOf(Layout.LH to Palette.FirstHand, Layout.RH to right)) {
            val palm = listOf(0, 1, 5, 9, 13, 17).map { pt(base + it) }
            if (palm.all { it != null }) {
                val path = Path().apply {
                    moveTo(palm[0]!!.x, palm[0]!!.y)
                    for (o in palm.drop(1)) lineTo(o!!.x, o.y)
                    close()
                }
                drawPath(path, colour.copy(alpha = 0.5f))
            }
            for ((a, b) in HAND_EDGES) bone(base + a, base + b, colour, 2.5.dp.toPx())
        }
    }
}

/** Shoulder, elbow and wrist pairs in [Layout]'s pose order (points 11-16 of MediaPipe's pose). */
private val ARMS = listOf(7 to 9, 9 to 11, 8 to 10, 10 to 12)

/** Whether point [k] of step [f] was seen (pose, either hand), rather than filled in by the normaliser. */
private fun seen(clip: Clip, f: Int, k: Int): Boolean {
    val group = when {
        k < Layout.NPOSE -> 0
        k < Layout.RH -> 1
        k < Layout.FACE -> 2
        else -> 3
    }
    return clip.seen[f * 4 + group] > 0.5f
}

/** The box the arms and hands stay inside over the whole example, as x0, y0, x1, y1. */
private fun bounds(clip: Clip): List<Float> {
    var x0 = Float.MAX_VALUE; var y0 = Float.MAX_VALUE; var x1 = -Float.MAX_VALUE; var y1 = -Float.MAX_VALUE
    val drawn = (0 until Layout.FACE).filter { it == 0 || it >= Layout.SHOULDER_L && it <= 12 || it >= Layout.LH }
    for (f in 0 until clip.t) for (k in drawn) {
        if (!seen(clip, f, k)) continue
        val i = (f * Layout.P + k) * 3
        x0 = minOf(x0, clip.points[i]); x1 = maxOf(x1, clip.points[i])
        y0 = minOf(y0, clip.points[i + 1]); y1 = maxOf(y1, clip.points[i + 1])
    }
    if (x1 < x0) return listOf(-1f, -1f, 1f, 1f)
    // Never zoom in past the shoulders' width, so a still example does not fill the frame with one hand.
    val w = maxOf(x1 - x0, 1.2f); val h = maxOf(y1 - y0, 1.2f)
    val cx = (x0 + x1) / 2; val cy = (y0 + y1) / 2
    return listOf(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
}
