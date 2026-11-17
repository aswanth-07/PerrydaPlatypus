package ai.echogesture.app.data

import android.content.Context
import ai.echogesture.signcore.Clip
import ai.echogesture.signcore.ClipCodec
import org.json.JSONObject
import java.io.File
import java.util.UUID

/** A sign taught on this phone: its word and how many examples it has. */
data class TaughtSign(val id: String, val word: String, val created: Long, val examples: Int)

/**
 * Taught signs on the phone's private storage: one folder per sign under files/taught/<id>/ with meta.json and
 * each example's model input (the clip resampled to the model's 64 steps) as raw floats. Only keypoints are
 * stored, never video. Examples are kept rather than prototypes because the clip model and the live-caption
 * model each have their own embedding, so prototypes are rebuilt with each model when it loads.
 */
class TaughtStore(context: Context) {
    private val root = File(context.filesDir, "taught").apply { mkdirs() }

    @Synchronized
    fun list(): List<TaughtSign> = root.listFiles().orEmpty().filter { it.isDirectory }.mapNotNull { dir ->
        runCatching {
            val meta = JSONObject(File(dir, META).readText())
            TaughtSign(dir.name, meta.getString("word"), meta.getLong("created"), exampleFiles(dir).size)
        }.getOrNull()
    }.sortedBy { it.created }

    @Synchronized
    fun examples(id: String): List<Clip> = exampleFiles(File(root, id)).mapNotNull { f -> runCatching { f.inputStream().use(ClipCodec::read) }.getOrNull() }

    @Synchronized
    fun add(word: String, clips: List<Clip>): TaughtSign {
        require(clips.isNotEmpty()) { "record at least one example" }
        val id = UUID.randomUUID().toString().take(8)
        val dir = File(root, id).apply { mkdirs() }
        clips.forEachIndexed { k, c -> File(dir, "ex_$k.bin").outputStream().use { ClipCodec.write(c, it) } }
        val created = System.currentTimeMillis()
        File(dir, META).writeText(JSONObject().put("word", clean(word)).put("created", created).put("note", "keypoints only, no video").toString())
        return TaughtSign(id, clean(word), created, clips.size)
    }

    @Synchronized
    fun rename(id: String, word: String) {
        val f = File(File(root, id), META)
        if (!f.exists()) return
        f.writeText(JSONObject(f.readText()).put("word", clean(word)).toString())
    }

    @Synchronized
    fun delete(id: String) {
        File(root, id).takeIf { it.parentFile == root }?.deleteRecursively()
    }

    private fun exampleFiles(dir: File) = dir.listFiles { f -> f.name.startsWith("ex_") && f.name.endsWith(".bin") }.orEmpty().sortedBy { it.name }

    companion object {
        private const val META = "meta.json"
        /** Captions are upper case like the model's own words. */
        fun clean(word: String) = word.trim().replace(Regex("\\s+"), " ").uppercase()
    }
}
