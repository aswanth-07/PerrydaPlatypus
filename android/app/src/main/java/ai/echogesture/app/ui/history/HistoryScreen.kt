package ai.echogesture.app.ui.history

import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.foundation.layout.calculateStartPadding
import ai.echogesture.app.ui.components.LocalTabBarInset
import androidx.compose.material3.ripple
import ai.echogesture.app.ui.components.rememberPress
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.text.format.DateUtils
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ContentCopy
import androidx.compose.material.icons.rounded.DeleteOutline
import androidx.compose.material.icons.rounded.History
import androidx.compose.material.icons.rounded.Share
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.R
import ai.echogesture.app.data.Conversation
import ai.echogesture.app.data.Conversations
import ai.echogesture.app.data.Settings
import ai.echogesture.app.engine.TranscriptEntry
import ai.echogesture.app.engine.Who
import ai.echogesture.app.ui.components.ConversationArt
import ai.echogesture.app.ui.components.EmptyState
import ai.echogesture.app.ui.components.LocalCaption
import ai.echogesture.app.ui.theme.TranscriptStyle
import ai.echogesture.app.ui.components.SubScreen
import ai.echogesture.app.ui.components.TabHeader
import ai.echogesture.app.ui.theme.Palette
import java.text.DateFormat
import java.util.Date

/** Past conversations, newest first, opened from Talk or Settings. A pause of over 30 minutes starts a new one. */
@Composable
fun HistoryScreen(vm: MainViewModel, onOpen: (Long) -> Unit, onBack: () -> Unit) {
    val entries by vm.transcript.collectAsStateWithLifecycle()
    val settings by vm.settings.collectAsStateWithLifecycle()
    val conversations = remember(entries) { Conversations.group(entries).asReversed() }

    SubScreen(stringResource(R.string.history_title), onBack) {
        if (conversations.isEmpty()) {
            EmptyState(
                { ConversationArt() }, stringResource(R.string.history_empty_title), stringResource(R.string.history_empty_body),
                Modifier.fillMaxSize(),
            )
            return@SubScreen
        }
        LazyColumn(
            contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 8.dp, bottom = 24.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            items(conversations, key = { it.start }) { c -> Box(Modifier.animateItem()) { ConversationCard(c) { onOpen(c.start) } } }
            item {
                Text(
                    retentionNote(settings.keepDays),
                    Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 12.dp),
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun retentionNote(keepDays: Int): String = when (keepDays) {
    Settings.KEEP_FOREVER -> stringResource(R.string.history_kept_forever)
    else -> pluralStringResource(R.plurals.history_kept_days, keepDays, keepDays)
}

@Composable
private fun ConversationCard(c: Conversation, onClick: () -> Unit) {
    val context = LocalContext.current
    val press = rememberPress()
    Column(
        press.scale.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(MaterialTheme.colorScheme.surfaceContainer)
            .clickable(interactionSource = press.interaction, indication = ripple(), onClickLabel = stringResource(R.string.history_open), onClick = onClick)
            .padding(horizontal = 18.dp, vertical = 14.dp),
    ) {
        Text(
            dayLabel(context, c.start) + "  ·  " + DateUtils.formatDateRange(context, c.start, c.end, DateUtils.FORMAT_SHOW_TIME),
            style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary,
        )
        Spacer(Modifier.height(4.dp))
        Text(preview(c.entries, LocalCaption.current), style = MaterialTheme.typography.bodyLarge, maxLines = 2, overflow = TextOverflow.Ellipsis)
        Spacer(Modifier.height(6.dp))
        Text(
            listOfNotNull(
                c.signed.takeIf { it > 0 }?.let { pluralStringResource(R.plurals.history_signs, it, it) },
                c.said.takeIf { it > 0 }?.let { pluralStringResource(R.plurals.history_replies, it, it) },
            ).joinToString("  ·  "),
            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

/** One conversation in full, with copy, share and delete. Leaves when the conversation no longer exists. */
@Composable
fun ConversationScreen(vm: MainViewModel, start: Long, onBack: () -> Unit) {
    val entries by vm.transcript.collectAsStateWithLifecycle()
    val conversation = remember(entries, start) { Conversations.group(entries).firstOrNull { it.start == start } }
    val context = LocalContext.current
    var confirmDelete by remember { mutableStateOf(false) }
    LaunchedEffect(conversation == null) { if (conversation == null) onBack() }
    conversation ?: return
    val text = conversationText(conversation.entries, LocalCaption.current,
        stringResource(R.string.history_line_signed), stringResource(R.string.history_line_said))

    SubScreen(
        dayLabel(context, start) + ", " + DateFormat.getTimeInstance(DateFormat.SHORT).format(Date(start)),
        onBack,
        actions = {
            IconButton(onClick = { copyText(context, text) }) { Icon(Icons.Rounded.ContentCopy, stringResource(R.string.conversation_copy)) }
            IconButton(onClick = { shareText(context, text) }) { Icon(Icons.Rounded.Share, stringResource(R.string.conversation_share)) }
            IconButton(onClick = { confirmDelete = true }) { Icon(Icons.Rounded.DeleteOutline, stringResource(R.string.conversation_delete)) }
        },
    ) {
        val list = rememberLazyListState()
        ConversationBubbles(conversation.entries, list, Modifier.weight(1f))
    }

    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text(stringResource(R.string.conversation_delete_title)) },
            text = { Text(stringResource(R.string.conversation_delete_body)) },
            confirmButton = {
                TextButton(onClick = { confirmDelete = false; vm.engine.deleteConversation(start) }) { Text(stringResource(R.string.action_delete)) }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text(stringResource(R.string.action_cancel)) } },
        )
    }
}

/** One bubble: signed words run together until the signer pauses for 4 s; each spoken reply is its own bubble. */
private class Line(val who: Who, val entries: List<TranscriptEntry>) {
    /** Signed words are glosses, written by [show]; a spoken reply is already ordinary text. */
    fun text(show: (String) -> String): String =
        if (who == Who.VOICE) entries.joinToString(" ") { it.word } else entries.joinToString(" ") { show(it.word) }
    val at: Long get() = entries.first().atMillis
}

/** The conversation as chat bubbles, scrolled to the newest line as lines arrive. */
@Composable
fun ConversationBubbles(entries: List<TranscriptEntry>, list: LazyListState, modifier: Modifier = Modifier) {
    val lines = remember(entries) { lines(entries) }
    LaunchedEffect(lines.size) { if (lines.isNotEmpty()) list.animateScrollToItem(lines.lastIndex) }
    LazyColumn(modifier, state = list, contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        items(lines, key = { it.at.toString() + it.who }) { Bubble(it) }
    }
}

@Composable
private fun Bubble(line: Line) {
    val signer = line.who == Who.SIGNER
    val who = stringResource(if (signer) R.string.conversation_signed else R.string.conversation_said)
    Box(Modifier.fillMaxWidth(), contentAlignment = if (signer) Alignment.CenterStart else Alignment.CenterEnd) {
        Column(
            Modifier.widthIn(max = 300.dp)
                .background(
                    if (signer) MaterialTheme.colorScheme.surfaceContainerHigh else Palette.ReplyDeep,
                    RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp, bottomStart = if (signer) 6.dp else 20.dp, bottomEnd = if (signer) 20.dp else 6.dp),
                )
                .semantics(mergeDescendants = true) {}   // read as one line: "Signed, 8:01 PM, HELLO HOW"
                .padding(horizontal = 16.dp, vertical = 12.dp),
        ) {
            Text(
                who + "  ·  " + DateFormat.getTimeInstance(DateFormat.SHORT).format(Date(line.at)),
                style = MaterialTheme.typography.labelMedium,
                color = if (signer) MaterialTheme.colorScheme.primary else Palette.Reply,
            )
            Spacer(Modifier.height(4.dp))
            Text(line.text(LocalCaption.current), style = TranscriptStyle, color = if (signer) Palette.Text else Palette.Reply)
        }
    }
}

private fun lines(entries: List<TranscriptEntry>): List<Line> {
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

private fun preview(entries: List<TranscriptEntry>, show: (String) -> String): String = lines(entries).joinToString("  ·  ") { it.text(show) }

/** "Today", "Yesterday", or the date. */
fun dayLabel(context: Context, at: Long): String = when {
    DateUtils.isToday(at) -> context.getString(R.string.day_today)
    DateUtils.isToday(at + DateUtils.DAY_IN_MILLIS) -> context.getString(R.string.day_yesterday)
    else -> DateUtils.formatDateTime(context, at, DateUtils.FORMAT_SHOW_DATE or DateUtils.FORMAT_SHOW_WEEKDAY or DateUtils.FORMAT_ABBREV_ALL)
}

/** Plain text for copying or sharing: one line per bubble, "Signed: …" ([signed]) or "Said: …" ([said]). */
fun conversationText(entries: List<TranscriptEntry>, show: (String) -> String, signed: String, said: String): String =
    lines(entries).joinToString("\n") { (if (it.who == Who.SIGNER) signed else said).format(it.text(show)) }

fun copyText(context: Context, text: String) {
    val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    cm.setPrimaryClip(ClipData.newPlainText("EchoGesture conversation", text))
}

fun shareText(context: Context, text: String) {
    val send = Intent(Intent.ACTION_SEND).apply { type = "text/plain"; putExtra(Intent.EXTRA_TEXT, text) }
    context.startActivity(Intent.createChooser(send, context.getString(R.string.conversation_share)))
}
