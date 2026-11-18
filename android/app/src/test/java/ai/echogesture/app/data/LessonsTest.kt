package ai.echogesture.app.data

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class LessonsTest {
    /**
     * The words with a demonstration, from the record the build packs next to the app's demonstrations
     * (android/models/demos.json: "clips": {"WORD": "clip id", ...}). Read with a regex: org.json is only a stub in
     * JVM unit tests.
     */
    private val demoWords: List<String> by lazy {
        val file = listOf("../models/demos.json", "models/demos.json", "android/models/demos.json").map(::File).first { it.exists() }
        val text = file.readText()
        val clips = text.substring(text.indexOf("\"clips\""))
        Regex("\"([^\"]+)\"\\s*:\\s*\"").findAll(clips.substring(clips.indexOf('{'), clips.indexOf('}'))).map { it.groupValues[1] }.toList()
    }

    @Test
    fun splitIsAsEvenAsPossibleAroundFive() {
        assertEquals(listOf(5), Lessons.split(5))
        assertEquals(listOf(4, 4, 4), Lessons.split(12))
        assertEquals(listOf(4, 4, 3), Lessons.split(11))
        assertEquals(listOf(5, 5, 5, 5), Lessons.split(20))
        assertEquals(listOf(5, 4, 4, 4, 4), Lessons.split(21))
        assertEquals(emptyList(), Lessons.split(0))
    }

    @Test
    fun everyDemonstrationIsInExactlyOneLesson() {
        val topics = Lessons.build(demoWords)
        val inLessons = topics.flatMap { t -> t.lessons.flatMap { it.words } }
        assertEquals(demoWords.size, inLessons.size, "no sign twice")
        assertEquals(demoWords.toSet(), inLessons.toSet(), "no sign missing")
        assertTrue(topics.none { it.key == "more" }, "every demonstration has a topic: " + topics.find { it.key == "more" }?.lessons?.flatMap { it.words })
        assertTrue(topics.flatMap { it.lessons }.all { it.words.size in 3..5 }, "lessons of three to five signs")
    }

    @Test
    fun unknownWordsGoToMoreSignsAndMissingOnesAreSkipped() {
        val topics = Lessons.build(listOf("YES", "NO", "ZEBRA"))
        assertEquals(listOf("first", "more"), topics.map { it.key })
        assertEquals(listOf("YES", "NO"), topics[0].lessons.single().words)
        assertEquals(listOf("ZEBRA"), topics[1].lessons.single().words)
    }

    @Test
    fun nextLessonIsTheFirstWithASignNotLearned() {
        val topics = Lessons.build(demoWords)
        val first = topics.first().lessons.first()
        assertEquals(first.id, Lessons.next(topics, emptyList())?.id)
        assertEquals("first-2", Lessons.next(topics, first.words)?.id)
        assertNull(Lessons.next(topics, demoWords))
        assertEquals("first-2", Lessons.after(topics, "first-1")?.id)
        assertEquals("questions-1", Lessons.after(topics, "first-4")?.id)
        assertEquals(first, Lessons.find(topics, "first-1"))
    }
}
