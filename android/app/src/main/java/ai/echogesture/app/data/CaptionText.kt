package ai.echogesture.app.data

/**
 * How a recognised sign is written on screen. The model's labels are glosses in capitals (THANK YOU), which read
 * as shouting to most people, so captions are ordinary lower-case words by default ("thank you"), with "I" kept
 * capital. A sign taught on this phone is usually a name, so it is written in title case ("Priya"). [caps] keeps
 * the gloss as it is, the way sign language dictionaries write it.
 */
object CaptionText {
    fun show(gloss: String, taught: Set<String>, caps: Boolean): String {
        if (caps) return gloss
        if (gloss.uppercase() in taught) return gloss.split(' ').joinToString(" ") { titleCase(it) }
        return gloss.lowercase().split(' ').joinToString(" ") { w ->
            if (w == "i" || w.startsWith("i'")) "I" + w.substring(1) else w
        }
    }

    private fun titleCase(word: String): String =
        word.lowercase().split('-').joinToString("-") { part -> part.replaceFirstChar { it.uppercaseChar() } }
}
