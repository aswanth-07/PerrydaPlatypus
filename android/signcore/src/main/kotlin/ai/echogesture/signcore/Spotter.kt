package ai.echogesture.signcore

/**
 * Live word captions from a camera stream.
 *
 * Boundary mode (the default): a sign is the stretch between two rests. A stretch opens when a wrist rises above
 * the rest line (a height under the shoulders, in shoulder widths) and closes once both wrists have been back
 * below it for `restMs`; the whole stretch, with `preMs` of lead-in, is then classified once, the way a clip is,
 * and its word is emitted if it reaches `threshold`. The classifier was trained on whole signs, so it is asked
 * about a whole sign rather than about windows that straddle two.
 *
 * Signing that never rests (a stretch longer than `maxOneMs`) is captioned by the sliding window instead: every
 * `stride` frames the last `windowMs` (never reaching back past the stretch's start) are classified, and a word
 * is emitted when it wins `agree` windows in a row above `slideThreshold`. With `preview` on, the same windows
 * run from the start of every stretch; their words are held back, and are used only when the stretch closes
 * below `threshold` or outlives `maxOneMs`.
 *
 * With `boundary` off, the sliding window alone captions, as before 15 Nov 2026.
 *
 * scripts/live_eval.py (`run_stream_seg`) is the same logic in Python; it measured this spotter on chained
 * held-out clips. Keep the two identical.
 */
class Spotter(
    private val model: SignModel,
    @Volatile var windowMs: Long = 1500,
    @Volatile var stride: Int = 5,
    @Volatile var threshold: Float = 0.15f,        // a closed stretch's word needs this probability
    @Volatile var agree: Int = 2,
    @Volatile var minHandFraction: Float = 0.5f,
    boundary: Boolean = true,
    @Volatile var slideThreshold: Float = 0.35f,   // a sliding window's word needs this probability
    @Volatile var restHeight: Float = -1.0f,       // rest line: the higher wrist, in shoulder widths above the shoulders
    @Volatile var restMs: Long = 200,
    @Volatile var preMs: Long = 300,
    @Volatile var minMs: Long = 400,
    @Volatile var minStretchHands: Float = 0.3f,
    @Volatile var maxOneMs: Long = 4000,
    @Volatile var preview: Boolean = true,
) {
    /**
     * `word` is the newest word emitted at this frame (null: none) and `words` all of them in order (a stretch
     * that closes unsure can release more than one held-back word). `taught` is set when the classified word came
     * from a sign taught on the device. `signing` is true while a stretch is open.
     */
    class Event(
        val word: String?, val prediction: Prediction?, val active: Boolean, val taught: TaughtMatcher.Match? = null,
        val words: List<String> = listOfNotNull(word), val signing: Boolean = false,
    )

    /** Signs taught on the device; a window that matches one is that word, whatever the classifier says. */
    @Volatile var taught: TaughtMatcher? = null

    /** One answer per stretch between two rests (true) or the sliding window alone; switching starts afresh. */
    var boundary: Boolean = boundary
        set(value) {
            if (value != field) { field = value; frames.clear(); idle() }
        }

    private val frames = ArrayDeque<Pair<Long, RawFrame>>()
    private var sinceLast = 0
    private var streakWord: String? = null
    private var streak = 0
    private var lastEmitted: String? = null
    private var restedSinceEmit = true
    val caption = ArrayDeque<String>()

    // Boundary mode.
    private var shoulder = 0f            // running shoulder width, image units
    private var openAt = -1L             // timestamp the open stretch starts at (lead-in included); -1: at rest
    private var lowSince = -1L           // first frame of the current run below the rest line; -1: a wrist is up
    private var continuous = false       // the open stretch outlived maxOneMs: the sliding window captions it
    private val pending = ArrayList<String>()
    private var lastInStretch: String? = null

    /** Whether a stretch is open: the signer's hands are up. */
    val signing: Boolean get() = openAt >= 0

    fun push(timestampMs: Long, frame: RawFrame, width: Int, height: Int): Event? {
        frames.addLast(timestampMs to frame)
        val keepMs = if (boundary) maxOf(windowMs, maxOneMs + preMs + restMs + 1000) else windowMs
        while (frames.isNotEmpty() && timestampMs - frames.first().first > keepMs) frames.removeFirst()
        return if (boundary) pushBoundary(timestampMs, frame, width, height) else pushSliding(timestampMs, width, height)
    }

    // ---------- the sliding window alone ----------

    private fun pushSliding(timestampMs: Long, width: Int, height: Int): Event? {
        if (++sinceLast < stride || frames.size < 8) return null
        sinceLast = 0
        val window = frames.map { it.second }   // the buffer holds exactly the last windowMs in this mode
        if (handFraction(window) < minHandFraction) {
            restedSinceEmit = true
            streak = 0
            streakWord = null
            return Event(null, null, false)
        }
        val pred = model.classify(Normaliser.normalise(window, width, height))
        val hit = taught?.match(pred.embedding)
        val word = hit?.word ?: pred.top
        val sure = hit != null || pred.confidence >= slideThreshold   // a taught match has passed its own threshold
        if (sure && word == streakWord) streak++ else {
            streakWord = if (sure) word else null
            streak = if (streakWord != null) 1 else 0
        }
        var emitted: String? = null
        if (streak >= agree && (word != lastEmitted || restedSinceEmit)) {
            emitted = word
            lastEmitted = word
            restedSinceEmit = false
            emit(word)
        }
        return Event(emitted, pred, true, hit)
    }

    // ---------- one answer per stretch between two rests ----------

    /** Height of the higher wrist above the shoulder line in shoulder widths (negative: below); null without a body. */
    private fun wristHeight(f: RawFrame, width: Int, height: Int): Float? {
        val p = f.pose ?: return null
        val side = maxOf(width, height).toFloat()
        val ax = width / side
        val ay = height / side
        val dx = (p[11 * 4] - p[12 * 4]) * ax
        val dy = (p[11 * 4 + 1] - p[12 * 4 + 1]) * ay
        val cur = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat()
        shoulder = if (shoulder == 0f) cur else 0.9f * shoulder + 0.1f * cur
        val mid = (p[11 * 4 + 1] + p[12 * 4 + 1]) / 2
        val scale = ay / maxOf(shoulder, 1e-3f)
        return maxOf((mid - p[15 * 4 + 1]) * scale, (mid - p[16 * 4 + 1]) * scale)
    }

    private fun pushBoundary(ts: Long, frame: RawFrame, width: Int, height: Int): Event? {
        val h = wristHeight(frame, width, height)
        val up = h != null && h > restHeight + (if (openAt >= 0) 0f else 0.1f)   // a little hysteresis to open
        if (openAt < 0) {
            if (!up) return if (++sinceLast >= stride) { sinceLast = 0; Event(null, null, false) } else null
            openAt = ts - preMs
            lowSince = -1
            continuous = false
            pending.clear()
            streakWord = null
            streak = 0
            sinceLast = 0
            lastInStretch = null
            return Event(null, null, true, signing = true)
        }
        if (up) lowSince = -1 else if (lowSince < 0) lowSince = ts
        if (lowSince >= 0 && ts - lowSince >= restMs) return close(ts, width, height)

        // The stretch is open: slide a window over it, for continuous signing and (with preview) as a fallback.
        var out: Event? = null
        if (++sinceLast >= stride && (preview || continuous) && frames.count { it.first >= openAt } >= 8) {
            sinceLast = 0
            val window = frames.filter { it.first >= openAt && ts - it.first < windowMs }.map { it.second }
            if (handFraction(window) >= minHandFraction) {
                val pred = model.classify(Normaliser.normalise(window, width, height))
                val hit = taught?.match(pred.embedding)
                val word = hit?.word ?: pred.top
                val sure = hit != null || pred.confidence >= slideThreshold
                if (sure && word == streakWord) streak++ else {
                    streakWord = if (sure) word else null
                    streak = if (streakWord != null) 1 else 0
                }
                var emitted: String? = null
                if (streak >= agree && word != lastInStretch) {
                    lastInStretch = word
                    if (continuous) { emitted = word; emit(word) } else pending += word
                }
                out = Event(emitted, pred, true, hit, signing = true)
            }
        }
        if (!continuous && ts - openAt >= maxOneMs) {   // no rest for this long: several signs, caption as they come
            continuous = true
            if (pending.isNotEmpty()) {
                val words = pending.toList()
                pending.clear()
                words.forEach(::emit)
                out = Event(words.last(), out?.prediction, true, out?.taught, words, signing = true)
            }
        }
        return out
    }

    /** The hands are back at rest: answer for the stretch that just ended. */
    private fun close(ts: Long, width: Int, height: Int): Event {
        val from = openAt
        val wasContinuous = continuous
        val upMs = lowSince - from
        openAt = -1
        lowSince = -1
        continuous = false
        sinceLast = 0
        if (wasContinuous) {
            // The sliding window captioned this stretch; give its last sign one more look, without the agreement.
            val tail = frames.filter { it.first >= from && ts - it.first < windowMs }.map { it.second }
            if (handFraction(tail) >= minHandFraction) {
                val pred = model.classify(Normaliser.normalise(tail, width, height))
                val hit = taught?.match(pred.embedding)
                val word = hit?.word ?: pred.top
                if ((hit != null || pred.confidence >= slideThreshold) && word != lastInStretch) {
                    emit(word)
                    return Event(word, pred, false, hit)
                }
            }
            return Event(null, null, false)
        }
        val stretch = frames.filter { it.first >= from }.map { it.second }
        if (upMs < minMs || handFraction(stretch) < minStretchHands) {
            pending.clear()
            return Event(null, null, false)
        }
        val pred = model.classify(Normaliser.normalise(stretch, width, height))
        val hit = taught?.match(pred.embedding)
        if (hit != null || pred.confidence >= threshold) {
            val word = hit?.word ?: pred.top
            pending.clear()
            emit(word)
            return Event(word, pred, false, hit)
        }
        val words = pending.toList()   // unsure about the whole stretch: fall back on what the windows settled
        pending.clear()
        words.forEach(::emit)
        return Event(words.lastOrNull(), pred, false, null, words)
    }

    private fun handFraction(window: List<RawFrame>): Float =
        window.count { it.hands[0] != null || it.hands[1] != null }.toFloat() / maxOf(window.size, 1)

    private fun emit(word: String) {
        caption.addLast(word)
        while (caption.size > 8) caption.removeFirst()
    }

    private fun idle() {
        streak = 0
        streakWord = null
        lastEmitted = null
        restedSinceEmit = true
        openAt = -1
        lowSince = -1
        continuous = false
        pending.clear()
        lastInStretch = null
        sinceLast = 0
    }

    /**
     * The person corrected the newest caption word: `from` becomes `to`. False when the caption has moved on (its
     * last word is no longer `from`), so a late tap never rewrites a different word.
     */
    fun replaceLast(from: String, to: String): Boolean {
        if (caption.lastOrNull() != from) return false
        caption.removeLast()
        caption.addLast(to)
        return true
    }

    fun reset() {
        frames.clear()
        caption.clear()
        idle()
    }
}
