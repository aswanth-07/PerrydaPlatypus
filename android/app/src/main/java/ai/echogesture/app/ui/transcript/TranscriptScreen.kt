package ai.echogesture.app.ui.transcript

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.R
import androidx.compose.ui.res.stringResource
import ai.echogesture.app.engine.TranscriptEntry
import ai.echogesture.app.engine.Who
import ai.echogesture.app.ui.components.EmptyState
import ai.echogesture.app.ui.theme.Palette
import java.text.DateFormat
import java.util.Date

/** One bubble: signed words run together until the signer pauses for 4 s; each spoken reply is its own bubble. */
private class Line(val who: Who, val entries: List<TranscriptEntry>) {
    val text: String get() = entries.joinToString(" ") { it.word }
    val at: Long get() = entries.first().atMillis
}

/** Both sides of the conversation: what was signed (recognised on Live) and what was said back (Reply). */
@Composable
fun TranscriptScreen(vm: MainViewModel) {
    val entries by vm.transcript.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val lines = remember(entries) { group(entries) }
    val text = lines.joinToString("\n") { (if (it.who == Who.SIGNER) "Signed: " else "Said: ") + it.text }
    var confirmClear by remember { mutableStateOf(false) }
    val list = rememberLazyListState()
    LaunchedEffect(lines.size) { if (lines.isNotEmpty()) list.animateScrollToItem(lines.lastIndex) }

    Column(Modifier.fillMaxSize().statusBarsPadding()) {
        Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 8.dp, top = 12.dp, bottom = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("Conversation", style = MaterialTheme.typography.headlineMedium, modifier = Modifier.weight(1f))
            IconButton(onClick = { copy(context, text) }, enabled = entries.isNotEmpty()) { Icon(Icons.Filled.ContentCopy, "Copy conversation") }
            IconButton(onClick = { share(context, text) }, enabled = entries.isNotEmpty()) { Icon(Icons.Filled.Share, "Share conversation") }
            IconButton(onClick = { confirmClear = true }, enabled = entries.isNotEmpty()) { Icon(Icons.Filled.DeleteOutline, "Clear conversation") }
        }
        if (entries.isEmpty()) {
            EmptyState(
                Icons.AutoMirrored.Filled.Chat, "No conversation yet",
                "Signs recognised on the Live tab and spoken replies collect here, so you can copy or share them.",
                Modifier.fillMaxSize(),
            )
        } else {
            LazyColumn(state = list, contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                items(lines, key = { it.at.toString() + it.who }) { Bubble(it) }
            }
        }
    }

    if (confirmClear) {
        AlertDialog(
            onDismissRequest = { confirmClear = false },
            title = { Text(stringResource(R.string.conversation_clear_title)) },
            text = { Text(stringResource(R.string.conversation_clear_body)) },
            confirmButton = { TextButton(onClick = { confirmClear = false; vm.engine.clearTranscript() }) { Text("Clear") } },
            dismissButton = { TextButton(onClick = { confirmClear = false }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun Bubble(line: Line) {
    val signer = line.who == Who.SIGNER
    Box(Modifier.fillMaxWidth(), contentAlignment = if (signer) Alignment.CenterStart else Alignment.CenterEnd) {
        Column(
            Modifier.widthIn(max = 300.dp)
                .background(
                    if (signer) MaterialTheme.colorScheme.surfaceContainerHigh else Palette.ReplyDeep,
                    RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp, bottomStart = if (signer) 6.dp else 20.dp, bottomEnd = if (signer) 20.dp else 6.dp),
                )
                .padding(horizontal = 16.dp, vertical = 12.dp),
        ) {
            Text(
                (if (signer) "Signed" else "Said") + "  ·  " + DateFormat.getTimeInstance(DateFormat.SHORT).format(Date(line.at)),
                style = MaterialTheme.typography.labelMedium,
                color = if (signer) MaterialTheme.colorScheme.primary else Palette.Reply,
            )
            Spacer(Modifier.height(4.dp))
            Text(line.text, style = MaterialTheme.typography.titleLarge, color = if (signer) Palette.Text else Palette.Reply)
        }
    }
}

private fun group(entries: List<TranscriptEntry>): List<Line> {
    val out = ArrayList<Line>()
    var run = ArrayList<TranscriptEntry>()
    fun flush() { if (run.isNotEmpty()) { out += Line(Who.SIGNER, run); run = ArrayList() } }
    for (e in entries) {
        if (e.who == Who.VOICE) { flush(); out += Line(Who.VOICE, listOf(e)); continue }
        if (run.isNotEmpty() && e.atMillis - run.last().atMillis > 4000) flush()
        run += e
    }
    flush()
    return out
}

private fun copy(context: Context, text: String) {
    val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    cm.setPrimaryClip(ClipData.newPlainText("EchoGesture conversation", text))
}

private fun share(context: Context, text: String) {
    val send = Intent(Intent.ACTION_SEND).apply { type = "text/plain"; putExtra(Intent.EXTRA_TEXT, text) }
    context.startActivity(Intent.createChooser(send, "Share conversation"))
}
