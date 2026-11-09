package ai.echogesture.signcore

/**
 * Live word captions from a camera stream. Frames go into a ring buffer; every `stride` frames the last
 * `windowMs` are classified. A word is emitted when the hands are active, the same word wins `agree`
 * windows in a row above `threshold`, and it differs from the last word or the hands rested in between.
 *
 * This is the word-level spotter; a learned "no sign" class from How2Sign continuous signing replaces the
 * activity gate later.
 */
class Spotter(
    private val model: SignModel,
    @Volatile var windowMs: Long = 2000,
    @Volatile var stride: Int = 5,
    @Volatile var threshold: Float = 0.45f,
    @Volatile var agree: Int = 2,
    @Volatile var minHandFraction: Float = 0.5f,
) {
    class Event(val word: String?, val prediction: Prediction?, val active: Boolean)

    private val frames = ArrayDeque<Pair<Long, RawFrame>>()
    private var sinceLast = 0
    private var streakWord: String? = null
    private var streak = 0
    private var lastEmitted: String? = null
    private var restedSinceEmit = true
    val caption = ArrayDeque<String>()

    fun push(timestampMs: Long, frame: RawFrame, width: Int, height: Int): Event? {
        frames.addLast(timestampMs to frame)
        while (frames.isNotEmpty() && timestampMs - frames.first().first > windowMs) frames.removeFirst()
        if (++sinceLast < stride || frames.size < 8) return null
        sinceLast = 0

        val window = frames.map { it.second }
        val handFrames = window.count { it.hands[0] != null || it.hands[1] != null }
        val active = handFrames >= minHandFraction * window.size
        if (!active) {
            restedSinceEmit = true
            streak = 0
            streakWord = null
            return Event(null, null, false)
        }
        val pred = model.classify(Normaliser.normalise(window, width, height))
        val word = pred.top
        if (pred.confidence >= threshold && word == streakWord) streak++ else {
            streakWord = if (pred.confidence >= threshold) word else null
            streak = if (streakWord != null) 1 else 0
        }
        var emitted: String? = null
        if (streak >= agree && (word != lastEmitted || restedSinceEmit)) {
            emitted = word
            lastEmitted = word
            restedSinceEmit = false
            caption.addLast(word)
            while (caption.size > 8) caption.removeFirst()
        }
        return Event(emitted, pred, true)
    }

    fun reset() {
        frames.clear()
        caption.clear()
        streak = 0
        streakWord = null
        lastEmitted = null
        restedSinceEmit = true
    }
}
