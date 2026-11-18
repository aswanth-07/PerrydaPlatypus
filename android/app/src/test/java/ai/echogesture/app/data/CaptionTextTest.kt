package ai.echogesture.app.data

import kotlin.test.Test
import kotlin.test.assertEquals

class CaptionTextTest {
    private val none = emptySet<String>()

    @Test
    fun glossesBecomeLowerCaseWords() {
        assertEquals("thank you", CaptionText.show("THANK YOU", none, caps = false))
        assertEquals("don't", CaptionText.show("DON'T", none, caps = false))
    }

    @Test
    fun theWordIStaysCapital() {
        assertEquals("I", CaptionText.show("I", none, caps = false))
        assertEquals("I'm fine", CaptionText.show("I'M FINE", none, caps = false))
        assertEquals("in", CaptionText.show("IN", none, caps = false))
    }

    @Test
    fun taughtSignsAreTitleCaseAsNames() {
        val taught = setOf("PRIYA", "NEW YORK", "MARY-JANE")
        assertEquals("Priya", CaptionText.show("PRIYA", taught, caps = false))
        assertEquals("New York", CaptionText.show("NEW YORK", taught, caps = false))
        assertEquals("Mary-Jane", CaptionText.show("MARY-JANE", taught, caps = false))
    }

    @Test
    fun capsKeepsTheGloss() {
        assertEquals("THANK YOU", CaptionText.show("THANK YOU", none, caps = true))
        assertEquals("PRIYA", CaptionText.show("PRIYA", setOf("PRIYA"), caps = true))
    }
}
