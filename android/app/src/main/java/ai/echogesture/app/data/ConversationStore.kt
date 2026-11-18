package ai.echogesture.app.data

import android.util.AtomicFile
import ai.echogesture.app.engine.TranscriptEntry
import ai.echogesture.app.engine.Who
import java.io.File

/**
 * The conversation on the phone's private storage (files/conversation.tsv), so it survives Android ending the
 * process. Nothing leaves the phone: the app has no network access and backups are off. One line per entry:
 * who, time, confidence, words. Kept until the person deletes it, or for as long as Settings says.
 */
class ConversationStore(dir: File) {
    private val file = AtomicFile(File(dir, "conversation.tsv"))

    @Synchronized
    fun load(): List<TranscriptEntry> = runCatching { Conversations.decode(String(file.readFully(), Charsets.UTF_8)) }
        .getOrDefault(emptyList())

    @Synchronized
    fun save(entries: List<TranscriptEntry>) {
        if (entries.isEmpty()) { file.delete(); return }
        val out = file.startWrite()
        try {
            out.write(Conversations.encode(entries).toByteArray(Charsets.UTF_8))
            file.finishWrite(out)
        } catch (e: Exception) {
            file.failWrite(out)
        }
    }
}

/** One conversation: the entries between two pauses of more than [Conversations.GAP_MS]. */
data class Conversation(val entries: List<TranscriptEntry>) {
    val start: Long get() = entries.first().atMillis
    val end: Long get() = entries.last().atMillis
    val signed: Int get() = entries.count { it.who == Who.SIGNER }
    val said: Int get() = entries.count { it.who == Who.VOICE }
}

/** Pure functions over the stored conversation, kept free of Android so they are unit-tested on the JVM. */
object Conversations {
    /** A pause this long starts a new conversation in History. */
    const val GAP_MS = 30 * 60 * 1000L
    private const val DAY_MS = 24 * 60 * 60 * 1000L

    fun encode(entries: List<TranscriptEntry>): String = buildString {
        for (e in entries) {
            append(if (e.who == Who.VOICE) 'V' else 'S').append('\t')
            append(e.atMillis).append('\t')
            append(e.confidence).append('\t')
            append(e.word.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')).append('\n')
        }
    }

    fun decode(text: String): List<TranscriptEntry> = text.lineSequence().mapNotNull { line ->
        val f = line.split('\t', limit = 4)
        if (f.size < 4 || f[3].isBlank()) return@mapNotNull null
        val at = f[1].toLongOrNull() ?: return@mapNotNull null
        TranscriptEntry(f[3], f[2].toFloatOrNull() ?: 0f, at, if (f[0] == "V") Who.VOICE else Who.SIGNER)
    }.sortedBy { it.atMillis }.toList()

    /** Entries newer than [keepDays] days before [now]; null keeps everything. */
    fun prune(entries: List<TranscriptEntry>, keepDays: Int?, now: Long): List<TranscriptEntry> =
        if (keepDays == null) entries else entries.filter { now - it.atMillis < keepDays * DAY_MS }

    /** Entries grouped into conversations, oldest first. */
    fun group(entries: List<TranscriptEntry>, gapMs: Long = GAP_MS): List<Conversation> {
        val out = ArrayList<Conversation>()
        var run = ArrayList<TranscriptEntry>()
        for (e in entries.sortedBy { it.atMillis }) {
            if (run.isNotEmpty() && e.atMillis - run.last().atMillis > gapMs) { out += Conversation(run); run = ArrayList() }
            run += e
        }
        if (run.isNotEmpty()) out += Conversation(run)
        return out
    }

    /** The entries without the conversation that started at [start]. */
    fun without(entries: List<TranscriptEntry>, start: Long, gapMs: Long = GAP_MS): List<TranscriptEntry> {
        val drop = group(entries, gapMs).firstOrNull { it.start == start }?.entries?.toSet() ?: return entries
        return entries.filterNot { it in drop }
    }
}
