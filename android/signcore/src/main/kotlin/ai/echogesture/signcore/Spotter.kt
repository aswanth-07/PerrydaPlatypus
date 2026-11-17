package ai.echogesture.signcore

/**
 * Live word captions from a camera stream.
 *
 * Continuous mode (`continuous`, what the app uses): signs follow one another with the hands kept up. While a
 * wrist is above the rest line, and for `tailMs` after both have dropped, the last `liveWindowMs` are classified
 * every `liveStride` frames by `liveModel`, a model trained on such windows: it names the sign the window is
 * centred on, or says "no sign" when the window is centred on the movement between two. A run of windows that
 * name a sign is one sign: while it lasts, every window adds evidence to the signs it names, and when it ends
 * (`gap` windows of "no sign", or the hands at rest) the sign with the most evidence is captioned, if it reached
 * `emin`. A sign held for long is captioned as soon as its evidence reaches `commit`. Until then the leading word
 * is `tentative`: shown, not yet settled, the way live speech captions settle a word. Nothing waits for the hands
 * to drop. Answering as soon as a few windows agreed was measured first and captioned half-seen signs.
 * scripts/contsign.py (`Decoder`, `RestGate`) and scripts/live_cont.py (`run_cont`) are the same logic in
 * Python, measured on held-out clips joined without rests. Keep them identical.
 *
 * Boundary mode: a sign is the stretch between two rests. A stretch opens when a wrist rises above
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
        /** Continuous mode with a `watch` word: its share of the settled sign's evidence, 1 when it is the sign. */
        val watchShare: Float? = null,
    )

    /** Signs taught on the device; a window that matches one is that word, whatever the classifier says. */
    @Volatile var taught: TaughtMatcher? = null

    /** One answer per stretch between two rests (true) or the sliding window alone; switching starts afresh. */
    var boundary: Boolean = boundary
        set(value) {
            if (value != field) { field = value; frames.clear(); idle() }
        }

    // ---------- continuous signing: settings ----------

    /** One caption per run of sign windows while the hands are up; takes precedence over `boundary`. Switching starts afresh. */
    var continuous: Boolean = false
        set(value) {
            if (value != field) { field = value; frames.clear(); idle() }
        }

    /** The model trained on windows of continuous signing; null: the clip model answers the windows. */
    @Volatile var liveModel: SignModel? = null
    /** Signs taught on the device, as `liveModel` embeds them. */
    @Volatile var liveTaught: TaughtMatcher? = null
    /** Same-sign set of each label (words that are one sign under two names gather evidence together); null: none. */
    @Volatile var setOf: IntArray? = null
    @Volatile var liveWindowMs: Long = 1200
    /** A window is classified every this many ms, rounded to whole frames (80 ms: every second frame at 25 or 30 fps). */
    @Volatile var liveStrideMs: Long = 80
    /** A window names a sign when its top word's set reaches this and beats "no sign". */
    @Volatile var lo: Float = 0.1f
    /** This many windows in a row without a sign end the sign. */
    @Volatile var gap: Int = 1
    /** The evidence a sign needs to be captioned when it ends (the sum of its probability over its windows). */
    @Volatile var emin: Float = 2.5f
    /** The evidence at which a sign is captioned without waiting for its end; 0: never. */
    @Volatile var commit: Float = 6f
    /** The evidence at which the leading word is offered as `tentative`. */
    @Volatile var show: Float = 1f
    /**
     * A word the signer is trying to make (the app's Learn mode), as a label index; null: none. Every settled
     * caption then reports how much of the sign's evidence was for this word's set: with 2,000 words open the
     * top answer is often a near neighbour, and "was that the sign I asked for" is an easier question than
     * "which sign was that".
     */
    @Volatile var watch: Int? = null
    @Volatile var tailMs: Long = 600
    @Volatile var minHandFrames: Int = 6

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
    private var overlong = false         // the open stretch outlived maxOneMs: the sliding window captions it
    private val pending = ArrayList<String>()
    private var lastInStretch: String? = null

    /** Whether a stretch is open: the signer's hands are up. */
    val signing: Boolean get() = if (continuous) gateOpen else openAt >= 0

    fun push(timestampMs: Long, frame: RawFrame, width: Int, height: Int): Event? {
        frames.addLast(timestampMs to frame)
        val keepMs = if (continuous) liveWindowMs else if (boundary) maxOf(windowMs, maxOneMs + preMs + restMs + 1000) else windowMs
        while (frames.isNotEmpty() && timestampMs - frames.first().first > keepMs) frames.removeFirst()
        return when {
            continuous -> pushContinuous(timestampMs, frame, width, height)
            boundary -> pushBoundary(timestampMs, frame, width, height)
            else -> pushSliding(timestampMs, width, height)
        }
    }

    // ---------- continuous signing: windows voted on ----------

    /** What a sign's caption is shown with: the newest window that named it. */
    private class Named(val pred: Prediction?, val taught: TaughtMatcher.Match?)
    private class Settled(val word: String, val by: Named?, val watchShare: Float?)

    private val ev = LinkedHashMap<String, Double>()     // evidence per sign of the open run: a same-sign set or a taught word
    private val wsum = LinkedHashMap<Int, Double>()      // summed probability per word of the open run
    private val namedBy = HashMap<String, Named>()
    private var gapRun = 0
    private var done: String? = null                     // the sign already captioned in this run (commit)
    private var gateOpen = false
    private var closedAt = -1L
    private var liveActive = false
    private var classifiedAt = 0L
    private var prevTs = -1L
    private var framePeriod = 0f     // running time between frames, ms

    /** The open run's leading word, once its evidence reaches `show`: what the caption will most likely be. */
    @Volatile var tentative: String? = null
        private set

    private fun pushContinuous(ts: Long, frame: RawFrame, width: Int, height: Int): Event? {
        if (prevTs >= 0) framePeriod = if (framePeriod == 0f) (ts - prevTs).toFloat() else 0.9f * framePeriod + 0.1f * (ts - prevTs)
        prevTs = ts
        // The gate: open from the frame a wrist rises above the rest line until both have been below it for restMs;
        // windows are still classified for tailMs after that, while their centre is on the last sign.
        val h = wristHeight(frame, width, height)
        val up = h != null && h > restHeight + (if (gateOpen) 0f else 0.1f)
        if (gateOpen) {
            if (up) lowSince = -1 else if (lowSince < 0) lowSince = ts
            if (lowSince >= 0 && ts - lowSince >= restMs) { gateOpen = false; closedAt = ts }
        } else if (up) {
            gateOpen = true
            lowSince = -1
        }
        if (!(gateOpen || (closedAt >= 0 && ts - closedAt <= tailMs))) {
            if (!liveActive) return null
            liveActive = false
            val last = flush()       // the hands came to rest: the open sign is settled
            if (last != null) emit(last.word)
            return Event(last?.word, last?.by?.pred, false, last?.by?.taught, watchShare = last?.watchShare)
        }
        // The first active frame classifies at once; after that every liveStrideMs, to the nearest whole frame.
        if (liveActive && ts - classifiedAt < liveStrideMs - framePeriod / 2) return null
        liveActive = true
        classifiedAt = ts

        val window = frames.filter { ts - it.first < liveWindowMs }.map { it.second }
        var pred: Prediction? = null
        var hit: TaughtMatcher.Match? = null
        var settled: Settled? = null
        var sign = false
        if (window.size >= 8 && window.count { it.hands[0] != null || it.hands[1] != null } >= minHandFrames) {
            val live = liveModel
            pred = (live ?: model).classify(Normaliser.normalise(window, width, height))
            hit = (if (live != null) liveTaught else taught)?.match(pred.embedding)
            val conf = pred.setProbs[pred.order[0]]
            sign = hit != null || (conf >= lo && conf > pred.none)   // a taught match passed its own threshold
        }
        if (!sign) {                 // the movement between two signs, or no hands to read
            if (++gapRun >= gap) settled = flush()
        } else {
            gapRun = 0
            val p = pred!!
            val named = Named(p, hit)
            if (hit != null) {
                // A window that matches a taught sign is that sign, whatever the classifier says of it.
                val key = "t:" + hit.word
                ev[key] = (ev[key] ?: 0.0) + 1.0
                namedBy[key] = named
            } else {
                val seen = HashSet<String>()
                for (n in 0 until minOf(TOP_K, p.order.size)) {
                    val k = p.order[n]
                    val key = setKey(k)
                    wsum[k] = (wsum[k] ?: 0.0) + p.probs[k].toDouble()
                    if (seen.add(key)) ev[key] = (ev[key] ?: 0.0) + p.setProbs[k].toDouble()
                    if (n == 0) namedBy[key] = named
                }
            }
            if (commit > 0 && done == null) {
                val (key, word, e) = leader()
                if (e >= commit) { done = key; settled = Settled(word, namedBy[key], share(e)) }
            }
        }
        tentative = if (ev.isEmpty()) null else leader().let { (key, word, e) -> if (e >= show && key != done) word else null }
        if (settled != null) emit(settled.word)
        return Event(settled?.word, settled?.by?.pred ?: pred, true, if (settled != null) settled.by?.taught else hit, signing = gateOpen,
            watchShare = settled?.watchShare)
    }

    private fun setKey(k: Int): String = "s:" + (setOf?.getOrNull(k) ?: k)

    /** The open run's leading sign, the word to show for it, and its evidence. Ties go to whichever came first. */
    private fun leader(): Triple<String, String, Double> {
        val top = ev.entries.maxByOrNull { it.value }!!
        val word = if (top.key.startsWith("t:")) top.key.substring(2)
        else model.labels[wsum.entries.filter { setKey(it.key) == top.key }.maxByOrNull { it.value }!!.key]
        return Triple(top.key, word, top.value)
    }

    /** The watched word's evidence in the open run as a share of the leader's [top] evidence; null: nothing watched. */
    private fun share(top: Double): Float? = watch?.let { ((ev[setKey(it)] ?: 0.0) / top).toFloat().coerceIn(0f, 1f) }

    /** Close the open run (a gap, or the hands at rest): its caption, or null. */
    private fun flush(): Settled? {
        var out: Settled? = null
        if (ev.isNotEmpty() && done == null) {
            val (key, word, e) = leader()
            if (e >= emin) out = Settled(word, namedBy[key], share(e))
        }
        clearRun()
        return out
    }

    private fun clearRun() {
        ev.clear()
        wsum.clear()
        namedBy.clear()
        gapRun = 0
        done = null
        tentative = null
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
            overlong = false
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
        if (++sinceLast >= stride && (preview || overlong) && frames.count { it.first >= openAt } >= 8) {
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
                    if (overlong) { emitted = word; emit(word) } else pending += word
                }
                out = Event(emitted, pred, true, hit, signing = true)
            }
        }
        if (!overlong && ts - openAt >= maxOneMs) {   // no rest for this long: several signs, caption as they come
            overlong = true
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
        val wasContinuous = overlong
        val upMs = lowSince - from
        openAt = -1
        lowSince = -1
        overlong = false
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

    private companion object {
        /** A window's this many most probable words add evidence (contsign.TOPK). */
        const val TOP_K = 5
    }

    private fun idle() {
        clearRun()
        gateOpen = false
        closedAt = -1
        liveActive = false
        prevTs = -1
        framePeriod = 0f
        streak = 0
        streakWord = null
        lastEmitted = null
        restedSinceEmit = true
        openAt = -1
        lowSince = -1
        overlong = false
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
