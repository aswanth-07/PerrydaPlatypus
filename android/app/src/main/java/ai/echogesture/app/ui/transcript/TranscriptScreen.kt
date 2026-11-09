package ai.echogesture.app.ui.transcript

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.engine.TranscriptEntry
import ai.echogesture.app.ui.components.EmptyState
import java.text.DateFormat
import java.util.Date

/** Words from the live screen, grouped into lines wherever the signer paused for more than 4 seconds. */
@Composable
fun TranscriptScreen(vm: MainViewModel) {
    val entries by vm.transcript.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val lines = remember(entries) { group(entries) }
    val text = lines.joinToString("\n") { line -> line.joinToString(" ") { it.word } }

    Column(Modifier.fillMaxSize()) {
        Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 8.dp, top = 20.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("Transcript", style = MaterialTheme.typography.headlineMedium, modifier = Modifier.weight(1f))
            IconButton(onClick = { copy(context, text) }, enabled = entries.isNotEmpty()) { Icon(Icons.Filled.ContentCopy, "Copy") }
            IconButton(onClick = { share(context, text) }, enabled = entries.isNotEmpty()) { Icon(Icons.Filled.Share, "Share") }
            IconButton(onClick = vm.engine::clearTranscript, enabled = entries.isNotEmpty()) { Icon(Icons.Filled.DeleteOutline, "Clear") }
        }
        if (entries.isEmpty()) {
            EmptyState(Icons.Filled.Description, "Nothing captioned yet",
                "Words recognised on the Live screen collect here so you can copy or share them.",
                Modifier.fillMaxSize())
        } else {
            LazyColumn(contentPadding = PaddingValues(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                items(lines.reversed()) { line ->
                    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer)) {
                        Column(Modifier.padding(18.dp)) {
                            Text(DateFormat.getTimeInstance(DateFormat.SHORT).format(Date(line.first().atMillis)),
                                style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Spacer(Modifier.height(6.dp))
                            Text(line.joinToString(" ") { it.word }, style = MaterialTheme.typography.headlineSmall)
                            Spacer(Modifier.height(4.dp))
                            Text("average confidence %d%%".format((100 * line.map { it.confidence }.average()).toInt()),
                                style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }
        }
    }
}

private fun group(entries: List<TranscriptEntry>): List<List<TranscriptEntry>> {
    val out = ArrayList<MutableList<TranscriptEntry>>()
    for (e in entries) {
        val last = out.lastOrNull()
        if (last == null || e.atMillis - last.last().atMillis > 4000) out.add(mutableListOf(e)) else last.add(e)
    }
    return out
}

private fun copy(context: Context, text: String) {
    val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    cm.setPrimaryClip(ClipData.newPlainText("EchoGesture transcript", text))
}

private fun share(context: Context, text: String) {
    val send = Intent(Intent.ACTION_SEND).apply { type = "text/plain"; putExtra(Intent.EXTRA_TEXT, text) }
    context.startActivity(Intent.createChooser(send, "Share transcript"))
}
