package ai.echogesture.app.data

import java.util.Locale

/**
 * Offline English-to-ASL gloss matcher powering two features using zero network access and zero extra model bytes:
 *   1. "Reply in Sign" on the Talk tab: turns a hearing person's spoken or typed reply into a playable sequence of
 *      real-signer ASL demonstrations from the bundled 300-sign pack (demos.bin).
 *   2. Offline Sign Dictionary & Text-to-Sign translator on the Learn tab.
 *
 * Pure Kotlin with no Android dependencies so every matching rule is unit-tested on the JVM.
 */
data class SignToken(
    val spoken: String,       // original phrase/word as spoken or typed
    val gloss: String?,       // matched ASL demo key (e.g. "WATER", "THANK YOU"), or null if filler/not in pack
)

object SignTranslator {
    /** Common everyday words and spoken contractions mapped to ASL glosses present in the demonstration pack. */
    private val ALIASES = mapOf(
        "HI" to "HELLO",
        "HEY" to "HELLO",
        "GREETINGS" to "HELLO",
        "YEAH" to "YES",
        "YEP" to "YES",
        "YUP" to "YES",
        "SURE" to "YES",
        "NOPE" to "NO",
        "NAH" to "NO",
        "OK" to "FINE",
        "OKAY" to "FINE",
        "ALRIGHT" to "FINE",
        "THANKS" to "THANK YOU",
        "THX" to "THANK YOU",
        "TY" to "THANK YOU",
        "MOM" to "MOTHER",
        "MUM" to "MOTHER",
        "MOMMY" to "MOTHER",
        "DAD" to "FATHER",
        "DADDY" to "FATHER",
        "KID" to "CHILDREN",
        "KIDS" to "CHILDREN",
        "CHILD" to "CHILDREN",
        "PHYSICIAN" to "DOCTOR",
        "DOC" to "DOCTOR",
        "ILL" to "SICK",
        "UNWELL" to "SICK",
        "QUICK" to "HURRY",
        "FAST" to "HURRY",
        "QUICKLY" to "HURRY",
        "HOLD ON" to "WAIT",
        "ONE MOMENT" to "WAIT",
        "A MOMENT" to "WAIT",
        "REPEAT" to "AGAIN",
        "ONCE MORE" to "AGAIN",
        "SPEAK" to "TALK",
        "TALKING" to "CHAT",
        "CONVERSATION" to "CHAT",
        "HOUSE" to "HOUSE",
        "FOOD" to "EAT",
        "HUNGRY" to "EAT",
        "THIRSTY" to "DRINK",
        "RESTROOM" to "BATHROOM",
        "TOILET" to "BATHROOM",
        "WASHROOM" to "BATHROOM",
        "PILL" to "MEDICINE",
        "PILLS" to "MEDICINE",
        "MEDS" to "MEDICINE",
        "DONE" to "FINISH",
        "FINISHED" to "FINISH",
        "OVER" to "FINISH",
        "LOVE" to "LIKE",
        "GLAD" to "HAPPY",
        "GREAT" to "GOOD",
        "NICE" to "GOOD",
        "WELL" to "GOOD",
        "BETTER" to "GOOD",
        "BEST" to "GOOD",
        "SORRY" to "PLEASE",
        "EXCUSE" to "PLEASE",
        "BIG" to "TALL",
        "LITTLE" to "SMALL",
        "TINY" to "SMALL",
        "JOB" to "WORK",
        "PHONE" to "CALL",
        "TEXT" to "WRITE",
        "CAR" to "DRIVE",
        "BUS" to "RIDE",
        "COP" to "POLICE",
        "MONEY" to "MONEY",
        "CASH" to "MONEY",
        "PAY" to "BUY",
        "PURCHASE" to "BUY",
        "WATCH" to "SEE",
        "LOOK" to "SEE",
        "LISTEN" to "HEAR",
        "UNDERSTAND" to "KNOW",
        "GOT IT" to "KNOW",
        "I'M" to "ME",
        "IM" to "ME",
        "YOU'RE" to "YOU",
        "YOURE" to "YOU",
        "DON'T" to "NO",
        "DONT" to "NO",
        "CAN'T" to "NO",
        "CANT" to "NO",
        "WON'T" to "NO",
        "WONT" to "NO",
        "NOT" to "NO",
    )

    /** Everyday replies that work offline in one tap, curated so almost every word animates in demos.bin. */
    val QUICK_REPLIES = listOf(
        "Yes, please",
        "No, fine",
        "Please wait",
        "Can I help you?",
        "Do you want water or coffee?",
        "Please sign again",
        "Where is the bathroom?",
        "Do you need a doctor?",
        "Good, I know",
        "See you later",
    )

    /** Sample sentences for the Sign Dictionary / Text-to-Sign screen. */
    val SAMPLE_SENTENCES = listOf(
        "Please help me",
        "Do you want water or coffee?",
        "My family is happy",
        "Where is the doctor?",
        "I want to learn sign language",
        "Wait, sign again please",
    )

    /**
     * Translate English [text] into tokens, matching multi-word glosses first, then exact words, aliases,
     * model same-sign synonyms ([synonyms]), and regular English inflections (plurals / -ing / -ed / -s).
     */
    fun translate(
        text: String,
        available: Set<String>,
        synonyms: Map<String, List<String>> = emptyMap(),
    ): List<SignToken> {
        if (text.isBlank()) return emptyList()
        val upperAvail = available.mapTo(HashSet(available.size * 2)) { it.uppercase(Locale.US) }
        val rawWords = text.trim().split(Regex("\\s+")).filter { it.isNotBlank() }
        val cleaned = rawWords.map { cleanWord(it) }
        val out = ArrayList<SignToken>(rawWords.size)
        var i = 0
        while (i < rawWords.size) {
            // Try 2-word phrases first (e.g. "THANK YOU", "HOLD ON", "ONCE MORE", "GOT IT").
            if (i + 1 < rawWords.size && cleaned[i].isNotEmpty() && cleaned[i + 1].isNotEmpty()) {
                val two = "${cleaned[i]} ${cleaned[i + 1]}"
                val matchTwo = resolve(two, upperAvail, synonyms)
                if (matchTwo != null) {
                    out += SignToken("${rawWords[i]} ${rawWords[i + 1]}", matchTwo)
                    i += 2
                    continue
                }
            }
            val c = cleaned[i]
            val match = if (c.isNotEmpty()) resolve(c, upperAvail, synonyms) else null
            out += SignToken(rawWords[i], match)
            i++
        }
        return out
    }

    /** Extract only the matched ASL gloss sequence (in order) from [text]. */
    fun glosses(
        text: String,
        available: Set<String>,
        synonyms: Map<String, List<String>> = emptyMap(),
    ): List<String> = translate(text, available, synonyms).mapNotNull { it.gloss }

    /** Search demonstrated signs: prefix matches first, then substring matches, keeping topic order within ties. */
    fun search(query: String, words: List<String>, show: (String) -> String = { it }): List<String> {
        val q = query.trim().lowercase(Locale.US)
        if (q.isEmpty()) return words
        val prefix = ArrayList<String>()
        val contains = ArrayList<String>()
        for (w in words) {
            val s = show(w).lowercase(Locale.US)
            val raw = w.lowercase(Locale.US)
            when {
                s.startsWith(q) || raw.startsWith(q) -> prefix += w
                s.contains(q) || raw.contains(q) -> contains += w
            }
        }
        return prefix + contains
    }

    private fun cleanWord(w: String): String =
        w.uppercase(Locale.US).replace(Regex("^[^A-Z0-9']+|[^A-Z0-9']+$"), "")

    private fun resolve(term: String, available: Set<String>, synonyms: Map<String, List<String>>): String? {
        direct(term, available, synonyms)?.let { return it }
        ALIASES[term]?.let { alias -> direct(alias, available, synonyms)?.let { return it } }
        for (stem in stems(term)) {
            direct(stem, available, synonyms)?.let { return it }
            ALIASES[stem]?.let { alias -> direct(alias, available, synonyms)?.let { return it } }
        }
        return null
    }

    private fun direct(term: String, available: Set<String>, synonyms: Map<String, List<String>>): String? {
        if (term in available) return term
        synonyms[term]?.firstOrNull { it in available }?.let { return it }
        return null
    }

    private fun stems(w: String): List<String> {
        if (w.length <= 3) return emptyList()
        val out = ArrayList<String>(6)
        if (w.endsWith("'S") && w.length > 3) out += w.dropLast(2)
        if (w.endsWith("IES") && w.length > 4) out += w.dropLast(3) + "Y"
        if (w.endsWith("ES") && w.length > 4) out += w.dropLast(2)
        if (w.endsWith("S") && !w.endsWith("SS") && w.length > 3) out += w.dropLast(1)
        if (w.endsWith("ING") && w.length > 5) {
            val base = w.dropLast(3)
            out += base
            out += base + "E"
            if (base.length >= 2 && base.last() == base[base.length - 2]) out += base.dropLast(1)
        }
        if (w.endsWith("ED") && w.length > 4) {
            val base = w.dropLast(2)
            out += base
            out += base + "E"
            if (base.length >= 2 && base.last() == base[base.length - 2]) out += base.dropLast(1)
        }
        if (w.endsWith("LY") && w.length > 4) out += w.dropLast(2)
        return out
    }
}
