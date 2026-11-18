package ai.echogesture.app.ui.components

import ai.echogesture.app.ui.theme.Motion
import ai.echogesture.app.ui.theme.Palette
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathMeasure
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/** An empty screen: a small line drawing, a title and one sentence on how it fills up. */
@Composable
fun EmptyState(art: @Composable () -> Unit, title: String, body: String, modifier: Modifier = Modifier) {
    Column(modifier.padding(32.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
        art()
        Spacer(Modifier.height(20.dp))
        Text(title, style = MaterialTheme.typography.titleMedium, textAlign = TextAlign.Center)
        Spacer(Modifier.height(6.dp))
        Text(body, Modifier.widthIn(max = 360.dp), style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center)
    }
}

private fun DrawScope.line(width: Float) = Stroke(width = width, cap = StrokeCap.Round, join = StrokeJoin.Round)

/** Two speech bubbles, the signer's (cyan, left) and the reply (amber, right): History with nothing in it yet. */
@Composable
fun ConversationArt(modifier: Modifier = Modifier.size(width = 148.dp, height = 112.dp)) {
    val cyan = MaterialTheme.colorScheme.primary
    Canvas(modifier.clearAndSetSemantics { }) {
        val u = size.width / 148f
        val w = 2.5.dp.toPx()
        fun bubble(x: Float, y: Float, bw: Float, bh: Float, colour: Color, tailLeft: Boolean, lines: Int) {
            drawRoundRect(colour, Offset(x * u, y * u), Size(bw * u, bh * u), CornerRadius(14 * u), style = line(w))
            val tail = Path().apply {
                val tx = if (tailLeft) (x + 18) * u else (x + bw - 18) * u
                moveTo(tx - 6 * u, (y + bh) * u); lineTo(tx + if (tailLeft) -10 * u else 10 * u, (y + bh + 12) * u); lineTo(tx + 6 * u, (y + bh) * u)
            }
            drawPath(tail, colour, style = line(w))
            for (i in 0 until lines) {
                val ly = (y + 16 + i * 12) * u
                val len = if (i == lines - 1) bw * 0.45f else bw * 0.7f
                drawLine(colour.copy(alpha = 0.6f), Offset((x + 14) * u, ly), Offset((x + 14) * u + len * u - 14 * u, ly), w, StrokeCap.Round)
            }
        }
        bubble(4f, 6f, 88f, 46f, cyan, tailLeft = true, lines = 2)
        bubble(56f, 58f, 86f, 36f, Palette.Amber, tailLeft = false, lines = 1)
    }
}

/** A video frame with a play mark and an open hand: no video has been identified yet. */
@Composable
fun VideoArt(modifier: Modifier = Modifier.size(width = 148.dp, height = 112.dp)) {
    val cyan = MaterialTheme.colorScheme.primary
    Canvas(modifier.clearAndSetSemantics { }) {
        val u = size.width / 148f
        val w = 2.5.dp.toPx()
        drawRoundRect(cyan, Offset(10 * u, 10 * u), Size(128 * u, 88 * u), CornerRadius(14 * u), style = line(w))
        for (x in listOf(24f, 44f, 64f, 84f, 104f, 124f)) {
            drawLine(cyan.copy(alpha = 0.4f), Offset(x * u, 18 * u), Offset((x + 8) * u, 18 * u), w, StrokeCap.Round)
        }
        val play = Path().apply { moveTo(40 * u, 40 * u); lineTo(40 * u, 72 * u); lineTo(64 * u, 56 * u); close() }
        drawPath(play, cyan, style = line(w))
        // An open hand: palm and four fingers, in amber.
        val a = Palette.Amber
        drawRoundRect(a, Offset(88 * u, 56 * u), Size(26 * u, 26 * u), CornerRadius(9 * u), style = line(w))
        for ((i, h) in listOf(16f, 20f, 18f, 13f).withIndex()) {
            val x = (91 + i * 7) * u
            drawLine(a, Offset(x, 56 * u), Offset(x, (56 - h) * u), w, StrokeCap.Round)
        }
        drawLine(a, Offset(88 * u, 68 * u), Offset(80 * u, 58 * u), w, StrokeCap.Round)
    }
}

/** A tick in a circle that draws itself on once, for a sign just learned. */
@Composable
fun DrawnCheck(key: Any?, size: Dp, colour: Color = Palette.Green, modifier: Modifier = Modifier) {
    val progress = remember(key) { Animatable(0f) }
    LaunchedEffect(key) { progress.animateTo(1f, tween(Motion.LONG, easing = Motion.Standard)) }
    Canvas(modifier.size(size)) {
        val w = 3.dp.toPx()
        val p = progress.value
        drawArc(colour, -90f, 360f * minOf(1f, p * 1.6f), false, Offset(w / 2, w / 2),
            Size(this.size.width - w, this.size.height - w), style = line(w))
        val tick = Path().apply {
            moveTo(this@Canvas.size.width * 0.28f, this@Canvas.size.height * 0.52f)
            lineTo(this@Canvas.size.width * 0.44f, this@Canvas.size.height * 0.67f)
            lineTo(this@Canvas.size.width * 0.73f, this@Canvas.size.height * 0.36f)
        }
        val t = ((p - 0.4f) / 0.6f).coerceIn(0f, 1f)
        if (t > 0f) {
            val measure = PathMeasure().apply { setPath(tick, false) }
            val part = Path()
            measure.getSegment(0f, measure.length * t, part, true)
            drawPath(part, colour, style = line(w))
        }
    }
}
