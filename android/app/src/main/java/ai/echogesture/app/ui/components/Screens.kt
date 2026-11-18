package ai.echogesture.app.ui.components

import android.Manifest
import ai.echogesture.app.R
import ai.echogesture.app.ui.theme.Palette
import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp

/** A screen reached from a tab (Settings, a conversation, Teach…): a back arrow, a title and optional actions. */
@Composable
fun SubScreen(
    title: String,
    onBack: () -> Unit,
    actions: @Composable RowScope.() -> Unit = {},
    content: @Composable ColumnScope.() -> Unit,
) {
    Column(Modifier.fillMaxSize()) {
        Row(
            Modifier.fillMaxWidth().statusBarsPadding().heightIn(min = 64.dp).padding(start = 4.dp, end = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") }
            Spacer(Modifier.width(4.dp))
            Text(
                title, Modifier.weight(1f).semantics { heading() }, style = MaterialTheme.typography.titleLarge,
                maxLines = 1, overflow = TextOverflow.Ellipsis,
            )
            actions()
        }
        content()
    }
}

/** The title row of a tab, with the way to Settings at its end. */
@Composable
fun TabHeader(title: String, onSettings: () -> Unit, actions: @Composable RowScope.() -> Unit = {}) {
    Row(
        Modifier.fillMaxWidth().statusBarsPadding().padding(start = 20.dp, end = 4.dp, top = 8.dp, bottom = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(title, Modifier.weight(1f).semantics { heading() }, style = MaterialTheme.typography.headlineMedium)
        actions()
        IconButton(onClick = onSettings) { Icon(Icons.Filled.Settings, "Settings") }
    }
}

/**
 * Where a signer's head and shoulders should be in the picture. Holistic finds the hands from the body, so a signer
 * too close to the camera gets no captions; the outline shows how far back to stand. Dashed until a body is seen,
 * then solid green.
 */
@Composable
fun FramingGuide(ok: Boolean, modifier: Modifier = Modifier) {
    val colour by animateColorAsState(if (ok) Palette.Green else Color.White.copy(alpha = 0.75f), label = "framing")
    Canvas(modifier) {
        val w = size.width
        val h = size.height
        val unit = minOf(w, h * 0.7f)
        val cx = w / 2
        val headR = unit * 0.12f
        val headCy = h * 0.30f
        val stroke = Stroke(
            width = 3.dp.toPx(),
            pathEffect = if (ok) null else PathEffect.dashPathEffect(floatArrayOf(14.dp.toPx(), 10.dp.toPx())),
        )
        drawOval(colour, Offset(cx - headR * 0.85f, headCy - headR), Size(headR * 1.7f, headR * 2f), style = stroke)
        val top = headCy + headR + unit * 0.06f
        val span = unit * 0.78f
        drawArc(colour, 180f, 180f, false, Offset(cx - span / 2, top), Size(span, unit * 0.7f), style = stroke)
    }
}

/**
 * Shows [content] once the camera is allowed, and before that a short explanation written for this screen. When
 * Android will no longer show its dialog, the button opens the app's settings page instead of doing nothing.
 */
@Composable
fun CameraGate(
    icon: ImageVector, title: String, body: String, points: List<Pair<ImageVector, String>>,
    content: @Composable () -> Unit,
) {
    val camera = rememberPermission(Manifest.permission.CAMERA)
    if (camera.granted) {
        content()
        return
    }
    Box(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), contentAlignment = Alignment.Center) {
        Column(Modifier.widthIn(max = 420.dp).padding(28.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Box(
                Modifier.size(88.dp).clip(CircleShape).background(MaterialTheme.colorScheme.primaryContainer),
                contentAlignment = Alignment.Center,
            ) { Icon(icon, null, Modifier.size(40.dp), tint = MaterialTheme.colorScheme.primary) }
            Spacer(Modifier.height(20.dp))
            Text(title, style = MaterialTheme.typography.headlineMedium, textAlign = TextAlign.Center)
            Spacer(Modifier.height(10.dp))
            Text(body, style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center)
            Spacer(Modifier.height(20.dp))
            points.forEach { (i, t) -> Point(i, t) }
            Spacer(Modifier.height(24.dp))
            Button(onClick = camera::request, modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp)) {
                Text(stringResource(if (camera.blocked) R.string.action_open_settings else R.string.gate_allow_camera))
            }
            val note = when {
                camera.blocked -> R.string.gate_camera_blocked
                camera.denied -> R.string.gate_camera_denied
                else -> null
            }
            if (note != null) {
                Spacer(Modifier.height(12.dp))
                Text(stringResource(note), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.secondary,
                    textAlign = TextAlign.Center)
            }
        }
    }
}

@Composable
fun Point(icon: ImageVector, text: String) {
    Row(Modifier.fillMaxWidth().padding(vertical = 6.dp), verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Start) {
        Icon(icon, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(20.dp))
        Spacer(Modifier.width(12.dp))
        Text(text, style = MaterialTheme.typography.bodyMedium)
    }
}
