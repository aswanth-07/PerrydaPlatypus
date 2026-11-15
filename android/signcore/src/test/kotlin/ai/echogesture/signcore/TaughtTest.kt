package ai.echogesture.signcore

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class TaughtTest {
    @Test
    fun sameSignSetsListTheOtherNames() {
        // labels 0 and 3 are one sign, 1 is alone, 2 and 4 and 5 are one sign
        val sets = SameSignSets(intArrayOf(0, 1, 2, 0, 2, 2))
        assertEquals(listOf(3), sets.others(0))
        assertEquals(emptyList(), sets.others(1))
        assertEquals(listOf(2, 5), sets.others(4))
        assertEquals(emptyList(), sets.others(99))
    }

    @Test
    fun alternativesAreTheBestWordOfEachOtherSign() {
        // labels 0 and 3 are one sign, 1 is alone, 2 and 4 and 5 are one sign
        val sets = SameSignSets(intArrayOf(0, 1, 2, 0, 2, 2))
        // the caption shows 0: its twin 3 is skipped, then 4 stands for its sign, so 2 and 5 are skipped, then 1
        assertEquals(listOf(4, 1), sets.alternatives(listOf(0, 3, 4, 2, 1, 5), shown = 0, k = 2))
        assertEquals(listOf(4), sets.alternatives(listOf(0, 3, 4, 2, 1, 5), shown = 0, k = 1))
        // the caption shows a word that is not the model's first choice: the first choice is offered
        assertEquals(listOf(0, 4), sets.alternatives(listOf(0, 3, 4, 2, 1, 5), shown = 1, k = 2))
        assertEquals(emptyList(), sets.alternatives(listOf(0, 3), shown = 0, k = 2), "only names of the shown sign: nothing to offer")
        assertEquals(emptyList(), sets.alternatives(listOf(0, 1), shown = 99, k = 2))
        // every label its own sign (a vocabulary without sets): the next best words
        assertEquals(listOf(2, 0), SameSignSets(intArrayOf(0, 1, 2)).alternatives(listOf(1, 2, 0), shown = 1, k = 2))
    }

    @Test
    fun prototypeIsTheUnitMeanOfUnitExamples() {
        // scale must not matter: a long and a short vector pointing the same way count equally
        val p = Embeddings.prototype(listOf(floatArrayOf(10f, 0f), floatArrayOf(0f, 0.1f)))
        assertEquals(p[0], p[1], 1e-6f)
        assertEquals(1f, p[0] * p[0] + p[1] * p[1], 1e-5f)
    }

    @Test
    fun matcherPicksTheClosestAboveThreshold() {
        val m = TaughtMatcher(listOf("WAVE" to floatArrayOf(1f, 0f, 0f), "NOD" to floatArrayOf(0f, 1f, 0f)), threshold = 0.55f)
        assertEquals("WAVE", m.match(floatArrayOf(0.9f, 0.2f, 0.1f))?.word)
        assertEquals("NOD", m.match(floatArrayOf(0.1f, 2f, 0f))?.word)
        assertNull(m.match(floatArrayOf(0f, 0f, 1f)), "far from every prototype: the classifier answers")
        assertNull(m.match(null), "a model without an embedding output never matches")
        assertNull(TaughtMatcher(emptyList()).match(floatArrayOf(1f)))
        assertTrue(m.best(floatArrayOf(0f, 0f, 1f))!!.cosine < 0.55f)
    }

    @Test
    fun clipRoundTrips() {
        val t = 3
        val clip = Clip(t, FloatArray(t * Layout.P * 3) { it * 0.5f }, FloatArray(t * 2 * 21 * 3) { -it.toFloat() },
            FloatArray(t * 4) { (it % 2).toFloat() })
        val out = ByteArrayOutputStream()
        ClipCodec.write(clip, out)
        val back = ClipCodec.read(ByteArrayInputStream(out.toByteArray()))
        assertEquals(t, back.t)
        assertContentEquals(clip.points, back.points)
        assertContentEquals(clip.world, back.world)
        assertContentEquals(clip.seen, back.seen)
    }
}
