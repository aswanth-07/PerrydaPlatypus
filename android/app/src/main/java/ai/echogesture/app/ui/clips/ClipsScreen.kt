package ai.echogesture.app.ui.clips

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.VideoLibrary
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.ClipProgress
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.engine.ClipResult
import ai.echogesture.app.ui.components.ConfidenceRow
import ai.echogesture.app.ui.components.EmptyState
import ai.echogesture.app.ui.components.Metric
import ai.echogesture.app.ui.theme.Palette

/** Identify the sign in a saved video. [showTimings] (developer options) adds how long each step took. */
@Composable
fun ClipsScreen(vm: MainViewModel, showTimings: Boolean) {
    val clips by vm.clips.collectAsStateWithLifecycle()
    val progress by vm.clipProgress.collectAsStateWithLifecycle()
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri -> uri?.let(vm::recognise) }
    val busy = progress != null && progress?.error == null

    LazyColumn(Modifier.fillMaxSize(), contentPadding = androidx.compose.foundation.layout.PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)) {
        item {
            Text(
                "Pick a short video of one sign from your gallery. EchoGesture names the sign on this phone; the video is " +
                    "never uploaded.",
                style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(16.dp))
            Button(
                onClick = { pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)) },
                enabled = !busy,
                modifier = Modifier.fillMaxWidth().height(52.dp),
            ) {
                Icon(Icons.Filled.VideoLibrary, null)
                Spacer(Modifier.width(10.dp))
                Text(if (busy) "Working…" else "Choose a video")
            }
        }
        progress?.let { p -> item { ProgressCard(p, onDismiss = vm::dismissClipError) } }
        if (clips.isEmpty() && progress == null) {
            item {
                EmptyState(
                    Icons.Filled.VideoLibrary, "No videos yet",
                    "Each result shows the sign and the other signs it was closest to.",
                    Modifier.fillMaxWidth(),
                )
            }
        }
        items(clips, key = { it.atMillis }) { ResultCard(it, showTimings) }
        if (clips.isNotEmpty()) item {
            TextButton(onClick = vm::clearClips, modifier = Modifier.fillMaxWidth()) { Text("Clear results") }
        }
    }
}

@Composable
private fun ProgressCard(p: ClipProgress, onDismiss: () -> Unit) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerHigh)) {
        Column(Modifier.padding(18.dp)) {
            if (p.error != null) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Filled.ErrorOutline, null, tint = MaterialTheme.colorScheme.error)
                    Spacer(Modifier.width(10.dp))
                    Text("Could not recognise this clip", style = MaterialTheme.typography.titleMedium)
                }
                Spacer(Modifier.height(6.dp))
                Text(p.error, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                TextButton(onClick = onDismiss) { Text("Dismiss") }
            } else {
                Text(p.stage, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Spacer(Modifier.height(10.dp))
                LinearProgressIndicator(progress = { p.fraction }, modifier = Modifier.fillMaxWidth())
            }
        }
    }
}

@Composable
private fun ResultCard(r: ClipResult, showTimings: Boolean) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer)) {
        Column(Modifier.padding(20.dp)) {
            Text(if (showTimings) "${r.name} · ${r.frames} frames" else r.name, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1, overflow = TextOverflow.Ellipsis)
            Spacer(Modifier.height(4.dp))
            Row(verticalAlignment = Alignment.Bottom) {
                Text(r.word, style = MaterialTheme.typography.displaySmall, color = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.width(12.dp))
                Text(if (r.taught) "taught sign · match %.2f".format(r.confidence) else "${(100 * r.confidence).toInt()}% sure",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(bottom = 6.dp))
            }
            if (r.also.isNotEmpty()) {
                Spacer(Modifier.height(4.dp))
                Text("Same sign: " + r.also.joinToString(" · "), style = MaterialTheme.typography.titleSmall,
                    color = Palette.Amber)
            }
            Spacer(Modifier.height(14.dp))
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                r.top5.forEachIndexed { i, (w, p) -> ConfidenceRow(w, p, highlight = i == 0) }
            }
            if (showTimings) {
                Spacer(Modifier.height(16.dp))
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Metric("reading", "%.0f ms".format(r.decodeMs))
                    Metric("tracking", "%.0f ms".format(r.landmarksMs))
                    Metric("model", "%.1f ms".format(r.modelMs))
                    Metric("total", "%.2f s".format(r.totalMs / 1000))
                }
            }
        }
    }
}
