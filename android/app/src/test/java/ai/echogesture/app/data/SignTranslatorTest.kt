package ai.echogesture.app.data

import kotlin.test.Test
import kotlin.test.assertEquals

class SignTranslatorTest {
    private val pack = setOf(
        "YES", "NO", "PLEASE", "HELP", "FINE", "GOOD", "WATER", "COFFEE", "DRINK",
        "WAIT", "AGAIN", "DOCTOR", "FAMILY", "HAPPY", "FRIEND", "RUN", "EAT", "STUDY",
        "CHILDREN", "MOTHER", "FATHER", "BATHROOM", "WANT", "YOU", "WHERE", "SIGN",
    )

    @Test
    fun translatesExactWordsAndPunctuation() {
        val tokens = SignTranslator.translate("Yes, please help!", pack)
        assertEquals(listOf("YES", "PLEASE", "HELP"), tokens.mapNotNull { it.gloss })
        assertEquals(3, tokens.size)
    }

    @Test
    fun translatesAliasesInflectionsAndTwoWordPhrases() {
        val tokens = SignTranslator.translate("Hey mom, hold on! Drinking waters and running with friends.", pack)
        assertEquals(
            listOf("MOTHER", "WAIT", "DRINK", "WATER", "RUN", "FRIEND"),
            tokens.mapNotNull { it.gloss },
        )
    }

    @Test
    fun usesSameSignSynonymsFromModel() {
        val synonyms = mapOf("PHYSICIAN" to listOf("DOCTOR"), "MEDIC" to listOf("DOCTOR"))
        assertEquals(listOf("WHERE", "DOCTOR"), SignTranslator.glosses("Where is the medic?", pack, synonyms))
    }

    @Test
    fun preservesUnmatchedTokensInOrder() {
        val tokens = SignTranslator.translate("Do you want water?", pack)
        assertEquals(listOf(null, "YOU", "WANT", "WATER"), tokens.map { it.gloss })
    }

    @Test
    fun searchRanksPrefixBeforeSubstring() {
        val words = listOf("BATHROOM", "AWAKE", "WATER", "WAIT")
        assertEquals(listOf("WATER", "WAIT", "AWAKE"), SignTranslator.search("wa", words))
    }
}
