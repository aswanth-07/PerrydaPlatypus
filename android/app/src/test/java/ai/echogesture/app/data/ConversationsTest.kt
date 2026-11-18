package ai.echogesture.app.data

import ai.echogesture.app.engine.TranscriptEntry
import ai.echogesture.app.engine.Who
import kotlin.test.Test
import kotlin.test.assertEquals

class ConversationsTest {
    private val min = 60_000L
    private val day = 24 * 60 * min

    private fun s(word: String, at: Long) = TranscriptEntry(word, 0.5f, at, Who.SIGNER)
    private fun v(text: String, at: Long) = TranscriptEntry(text, 1f, at, Who.VOICE)

    @Test
    fun encodeDecodeRoundTripsAndCleansSeparators() {
        val entries = listOf(s("HELLO", 1_000), v("How are you?\tToday", 2_000), s("THANK YOU", 3_000))
        val back = Conversations.decode(Conversations.encode(entries))
        assertEquals(listOf("HELLO", "How are you? Today", "THANK YOU"), back.map { it.word })
        assertEquals(listOf(Who.SIGNER, Who.VOICE, Who.SIGNER), back.map { it.who })
        assertEquals(listOf(1_000L, 2_000L, 3_000L), back.map { it.atMillis })
    }

    @Test
    fun decodeSkipsBrokenLinesAndSortsByTime() {
        val text = "S\t3000\t0.5\tLATER\ngarbage\nV\tnot-a-time\t1\tx\nS\t1000\t0.5\tFIRST\nS\t2000\t0.5\t\n"
        assertEquals(listOf("FIRST", "LATER"), Conversations.decode(text).map { it.word })
    }

    @Test
    fun groupSplitsAtPausesLongerThanThirtyMinutes() {
        val e = listOf(s("A", 0), s("B", 10 * min), s("C", 41 * min), v("reply", 42 * min))
        val groups = Conversations.group(e)
        assertEquals(2, groups.size)
        assertEquals(listOf("A", "B"), groups[0].entries.map { it.word })
        assertEquals(1, groups[1].signed)
        assertEquals(1, groups[1].said)
    }

    @Test
    fun pruneKeepsOnlyRecentEntriesAndNullKeepsAll() {
        val now = 100 * day
        val e = listOf(s("OLD", now - 31 * day), s("WEEK", now - 7 * day), s("NOW", now - min))
        assertEquals(listOf("WEEK", "NOW"), Conversations.prune(e, 30, now).map { it.word })
        assertEquals(listOf("NOW"), Conversations.prune(e, 1, now).map { it.word })
        assertEquals(e, Conversations.prune(e, null, now))
    }

    @Test
    fun withoutDeletesOneConversationOnly() {
        val e = listOf(s("A", 0), s("B", min), s("C", 2 * 60 * min))
        assertEquals(listOf("C"), Conversations.without(e, start = 0).map { it.word })
        assertEquals(e, Conversations.without(e, start = 12345))
    }
}
