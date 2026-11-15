package ai.echogesture.signcore

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The live spotter against Python: the fixture's clips are chained into streams and captioned frame by frame by
 * Spotter (boundary mode) with the shipped INT8 model; scripts/make_stream_fixture.py did the same with
 * live_eval.run_stream_seg and saved the words and the frames they came at.
 */
class SpotterTest {
    companion object {
        init {
            // As in ParityTest: bind ONNX Runtime to the system C++ runtime on Windows.
            val sys = File(System.getenv("SystemRoot") ?: "C:\\Windows", "System32\\msvcp140.dll")
            if (System.getProperty("os.name").startsWith("Windows") && sys.exists()) System.load(sys.absolutePath)
        }
    }

    private val dir = File(System.getProperty("parity.dir") ?: "parity")

    private fun floats(a: Any?): FloatArray? {
        if (a == null || a == JSONObject.NULL) return null
        val arr = a as JSONArray
        return FloatArray(arr.length()) { arr.getDouble(it).toFloat() }
    }

    private fun frames(clip: JSONObject): List<RawFrame> {
        val fr = clip.getJSONArray("frames")
        return (0 until fr.length()).map { k ->
            val f = fr.getJSONObject(k)
            val hands = f.getJSONArray("hands")
            val world = f.getJSONArray("world")
            RawFrame(floats(f.opt("pose")), arrayOf(floats(hands.opt(0)), floats(hands.opt(1))),
                arrayOf(floats(world.opt(0)), floats(world.opt(1))), floats(f.opt("face")))
        }
    }

    @Test
    fun boundarySpotterMatchesPython() {
        val expect = File(dir, "streams.json")
        if (!expect.exists()) { println("no streams.json: run scripts/make_stream_fixture.py"); return }
        val spec = JSONObject(expect.readText())
        val stepMs = 1000.0 / spec.getDouble("fps")
        val clips = JSONObject(File(dir, "fixture.json").readText()).getJSONArray("clips")
        val la = JSONArray(File(dir, "labels.json").readText())
        val labels = (0 until la.length()).map { la.getString(it).uppercase() }
        val model = SignModel(File(dir, "sign_int8.onnx").readBytes(), labels)
        val streams = spec.getJSONArray("streams")
        var words = 0
        for (n in 0 until streams.length()) {
            val s = streams.getJSONObject(n)
            val ids = s.getJSONArray("clips")
            val first = clips.getJSONObject(ids.getInt(0))
            val w = first.getInt("width")
            val h = first.getInt("height")
            val stream = ArrayList<RawFrame>()
            for (k in 0 until ids.length()) stream += frames(clips.getJSONObject(ids.getInt(k)))
            repeat(s.getInt("rest_frames")) { stream += RawFrame(null, arrayOfNulls(2), arrayOfNulls(2), null) }   // nobody in view
            val spotter = Spotter(model)
            val got = ArrayList<Pair<String, Int>>()
            for ((t, f) in stream.withIndex()) {
                spotter.push(Math.round(t * stepMs), f, w, h)?.words?.forEach { got += it to t }
            }
            val want = s.getJSONArray("caption").let { c -> (0 until c.length()).map { c.getJSONArray(it).getString(0) to c.getJSONArray(it).getInt(1) } }
            println("stream $n: ${got.joinToString(" ") { "${it.first}@${it.second}" }}")
            assertEquals(want, got, "stream $n: Kotlin's caption differs from Python's")
            words += got.size
        }
        assertTrue(words > 0, "no word was captioned in any stream")
        model.close()
    }

    /** A wrist that never comes down is captioned by the sliding window; switching modes forgets the open stretch. */
    @Test
    fun modeSwitchStartsAfresh() {
        val clips = JSONObject(File(dir, "fixture.json").readText()).getJSONArray("clips")
        val la = JSONArray(File(dir, "labels.json").readText())
        val model = SignModel(File(dir, "sign_int8.onnx").readBytes(), (0 until la.length()).map { la.getString(it).uppercase() })
        val c = clips.getJSONObject(0)
        val spotter = Spotter(model)
        val fr = frames(c)
        for ((t, f) in fr.take(fr.size / 2).withIndex()) spotter.push(t * 40L, f, c.getInt("width"), c.getInt("height"))
        spotter.boundary = false
        assertTrue(!spotter.signing, "an open stretch survived the switch to the sliding window")
        model.close()
    }

    /** A tap on an offered word rewrites the newest caption word, and only while it still is the newest. */
    @Test
    fun correctionReplacesOnlyTheNewestWord() {
        val la = JSONArray(File(dir, "labels.json").readText())
        val model = SignModel(File(dir, "sign_int8.onnx").readBytes(), (0 until la.length()).map { la.getString(it).uppercase() })
        val spotter = Spotter(model)
        assertTrue(!spotter.replaceLast("BOOK", "BIBLE"), "nothing captioned yet")
        spotter.caption.addLast("DRINK")
        spotter.caption.addLast("BOOK")
        assertTrue(spotter.replaceLast("BOOK", "BIBLE"))
        assertEquals(listOf("DRINK", "BIBLE"), spotter.caption.toList())
        assertTrue(!spotter.replaceLast("BOOK", "WALLET"), "the caption moved on: a late tap changes nothing")
        assertEquals(listOf("DRINK", "BIBLE"), spotter.caption.toList())
        model.close()
    }
}
