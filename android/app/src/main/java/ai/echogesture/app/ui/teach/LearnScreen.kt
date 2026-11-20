package ai.echogesture.app.ui.teach

import ai.echogesture.app.R
import androidx.compose.ui.res.stringResource
import android.app.Activity
import android.view.WindowManager
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.keyframes
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.VolumeUp
import androidx.compose.material.icons.rounded.Cameraswitch
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.Pause
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Search
import androidx.compose.material.icons.rounded.SkipNext
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.data.Demo
import ai.echogesture.app.data.Demos
import ai.echogesture.app.engine.HAND_EDGES
import ai.echogesture.app.ui.components.DrawnCheck
import ai.echogesture.app.ui.components.FramingGuide
import ai.echogesture.app.ui.components.LocalCaption
import ai.echogesture.app.ui.components.LocalHaptics
import ai.echogesture.app.ui.components.StatusPill
import ai.echogesture.app.ui.theme.Motion
import ai.echogesture.app.ui.theme.glass
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeJoin
import ai.echogesture.app.data.Lesson
import ai.echogesture.app.data.Lessons
import ai.echogesture.app.data.Topic
import ai.echogesture.app.ui.components.TrackingError
import ai.echogesture.app.ui.components.description
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.material.icons.rounded.Flag
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import ai.echogesture.app.ui.live.CameraPreview
import ai.echogesture.app.ui.live.HandOverlay
import ai.echogesture.app.ui.theme.CaptionStyle
import ai.echogesture.app.ui.theme.Palette

/** Signs that have a demonstration and that the loaded model can recognise, in the demonstrations' order. */
@Composable
fun rememberPracticeWords(vm: MainViewModel): Pair<Map<String, Demo>, List<String>> {
    val live by vm.live.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val demos = remember { Demos.load(context) }
    val known = remember(live.ready) { vm.engine.labels().toSet() }
    val words = remember(demos, known) { demos.keys.filter { it in known } }
    return demos to words
}

/** The lessons for the signs above; empty until the model has loaded. */
@Composable
fun rememberTopics(vm: MainViewModel): List<Topic> {
    val (_, words) = rememberPracticeWords(vm)
    return remember(words) { Lessons.build(words) }
}

/**
 * The app teaches a sign: a stick figure shows it (keypoints of one training clip, no video), the learner copies
 * it in front of the camera, and the live recogniser says whether that was the sign. The check asks only "was
 * that the word asked for" (SignEngine.practiceTarget), which is easier than naming one sign of 2,000.
 *
 * With a [lesson], only its signs are practised, in order, and the last one's button ends the lesson
 * ([onLessonEnd]); without one, any sign can be chosen. Side by side in landscape.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun LearnScreen(vm: MainViewModel, lesson: Lesson? = null, initialWord: String? = null, onLessonEnd: () -> Unit = {}) {
    val live by vm.live.collectAsStateWithLifecycle()
    val settings by vm.settings.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val (demos, all) = rememberPracticeWords(vm)
    val words = remember(all, lesson) { if (lesson == null) all else lesson.words.filter { it in all.toSet() } }

    var index by rememberSaveable(lesson?.id, initialWord) {
        val initIdx = initialWord?.uppercase()?.let { w -> words.indexOf(w).takeIf { it >= 0 } } ?: 0
        mutableIntStateOf(initIdx)
    }
    LaunchedEffect(words, initialWord) {
        val target = initialWord?.uppercase() ?: return@LaunchedEffect
        val idx = words.indexOf(target)
        if (idx >= 0 && index == 0) index = idx
    }
    var slow by rememberSaveable { mutableStateOf(false) }
    var zoom by rememberSaveable { mutableStateOf(false) }
    var pausedDemo by rememberSaveable { mutableStateOf(false) }
    val show = LocalCaption.current
    val haptics = LocalHaptics.current
    val progress by vm.progress.collectAsStateWithLifecycle()   // kept on the phone, not just for this session
    val learned = progress.learned
    var choosing by remember { mutableStateOf(false) }
    // Practice faces the learner, so Learn starts on the front camera whatever the Talk tab uses.
    var front by rememberSaveable { mutableStateOf(true) }
    val word = words.getOrNull(index.coerceIn(0, maxOf(words.size - 1, 0)))
    val attempt = live.attempt?.takeIf { it.target == word }

    DisposableEffect(Unit) {
        vm.engine.setPaused(false)
        vm.engine.acquireLive()
        val window = (context as? Activity)?.window
        window?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onDispose {
            vm.engine.practiceTarget = null
            vm.engine.clearCaption()
            vm.engine.releaseLive()
            window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }
    LaunchedEffect(word) { vm.engine.practiceTarget = word }
    LaunchedEffect(attempt?.at) {
        val a = attempt ?: return@LaunchedEffect
        if (a.correct) { vm.markLearned(a.target); haptics.confirm() }
    }

    if (word == null) {
        Box(Modifier.fillMaxSize().padding(28.dp), contentAlignment = Alignment.Center) {
            Text(
                if (demos.isEmpty()) stringResource(R.string.practice_this_build_has_no_sign) else stringResource(R.string.practice_loading_the_model),
                style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        return
    }
    val demo = demos.getValue(word)
    val last = lesson != null && index >= words.size - 1

    val watch: @Composable (Modifier) -> Unit = { m ->
        Box(m.clip(RoundedCornerShape(24.dp)).background(Palette.Panel)
            .border(1.dp, Palette.Hairline, RoundedCornerShape(24.dp))
            .description(stringResource(R.string.practice_demonstration_of_the_sign, show(word)))) {
            DemoFigure(demo, slow, mirror = front, zoom = zoom, Modifier.fillMaxSize().padding(top = 44.dp, bottom = 52.dp), paused = pausedDemo)
            Tag(stringResource(R.string.practice_watch), Modifier.align(Alignment.TopStart).padding(10.dp))
            FilledTonalIconButton(
                onClick = { pausedDemo = !pausedDemo },
                modifier = Modifier.align(Alignment.TopEnd).padding(10.dp).size(40.dp).glass(CircleShape),
                colors = IconButtonDefaults.filledTonalIconButtonColors(containerColor = Color.Transparent, contentColor = Palette.Text),
            ) {
                Icon(
                    if (pausedDemo) Icons.Rounded.PlayArrow else Icons.Rounded.Pause,
                    contentDescription = stringResource(if (pausedDemo) R.string.talk_resume else R.string.talk_pause),
                    modifier = Modifier.size(20.dp),
                )
            }
            // The chips wrap onto two lines when the text is large, rather than breaking a word.
            FlowRow(Modifier.align(Alignment.BottomStart).padding(start = 10.dp, end = 10.dp, bottom = 4.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                FilterChip(selected = slow, onClick = { slow = !slow }, label = { Text(stringResource(R.string.practice_slow), maxLines = 1, softWrap = false) })
                FilterChip(selected = zoom, onClick = { zoom = !zoom }, label = { Text(stringResource(R.string.practice_hands), maxLines = 1, softWrap = false) },
                    modifier = Modifier.description(stringResource(R.string.practice_zoom_on_hands)))
            }
        }
    }
    val camera: @Composable (Modifier) -> Unit = { m ->
        Box(m.clip(RoundedCornerShape(24.dp)).background(Palette.Onyx2)
            .border(1.dp, Palette.Hairline, RoundedCornerShape(24.dp))) {
            CameraPreview(backCamera = !front, analyzer = vm.engine::analyze, modifier = Modifier.fillMaxSize())
            if (settings.showLandmarks) HandOverlay(vm.overlay, mirror = front, skeleton = settings.developer && settings.skeleton, modifier = Modifier.fillMaxSize())
            if (live.ready && live.running && live.error == null && !live.bodyVisible) {
                FramingGuide(ok = false, modifier = Modifier.fillMaxSize().padding(horizontal = 18.dp, vertical = 56.dp))
            }
            StatusPill(
                when {
                    live.error != null -> stringResource(R.string.talk_hand_tracking_unavailable)
                    live.signing -> stringResource(R.string.talk_signing)
                    live.handsVisible -> stringResource(R.string.practice_ready)
                    live.bodyVisible -> stringResource(R.string.practice_raise_your_hands)
                    else -> stringResource(R.string.talk_step_back)
                },
                when {
                    live.error != null -> Palette.Red
                    live.signing || live.handsVisible -> Palette.Green
                    else -> Palette.Amber
                },
                Modifier.align(Alignment.TopStart).padding(10.dp),
            )
            FilledTonalIconButton(
                onClick = { front = !front },
                modifier = Modifier.align(Alignment.BottomEnd).padding(8.dp).size(48.dp).glass(CircleShape),
                colors = IconButtonDefaults.filledTonalIconButtonColors(containerColor = Color.Transparent, contentColor = Palette.Text),
            ) { Icon(Icons.Rounded.Cameraswitch, stringResource(R.string.practice_switch_camera)) }
            TrackingError(live, settings.gpu, onRetry = vm.engine::retryLive, onSwitch = { vm.update { it.copy(gpu = !it.gpu) } },
                modifier = Modifier.align(Alignment.Center).padding(12.dp))
        }
    }
    val panel: @Composable (Modifier) -> Unit = { m ->
        Column(m.verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 6.dp)) {
            if (lesson != null) LessonSteps(words, index, learned)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(stringResource(R.string.practice_sign_this), style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Text(show(word), style = CaptionStyle, color = MaterialTheme.colorScheme.primary,
                        maxLines = 2, overflow = TextOverflow.Ellipsis)
                }
                FilledTonalIconButton(
                    onClick = { vm.engine.speakNow(ai.echogesture.app.engine.Speaker.sayable(word)) },
                    modifier = Modifier.size(44.dp),
                    colors = IconButtonDefaults.filledTonalIconButtonColors(
                        containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
                        contentColor = Palette.Text,
                    ),
                ) { Icon(Icons.AutoMirrored.Rounded.VolumeUp, stringResource(R.string.dict_hear_aloud)) }
                if (word in learned) {
                    Spacer(Modifier.width(8.dp))
                    DrawnCheck(word, 32.dp, modifier = Modifier.description(stringResource(R.string.practice_learned)))
                }
            }
            Spacer(Modifier.height(10.dp))
            // What the recogniser made of the last try, plus a visual match-closeness meter.
            val (text, colour) = when {
                attempt == null && live.pending != null -> stringResource(R.string.practice_looking) to Palette.TextMuted
                attempt == null -> stringResource(R.string.practice_watch_the_figure_then_make) to Palette.TextMuted
                attempt.correct -> stringResource(R.string.practice_yes_that_was, show(word)) to Palette.Green
                attempt.share >= 0.2f -> stringResource(R.string.practice_close_it_looked_a_little, show(attempt.seen)) to Palette.Amber
                else -> stringResource(R.string.practice_that_looked_like_watch_once, show(attempt.seen)) to Palette.Amber
            }
            Column(
                Modifier.fillMaxWidth().heightIn(min = 64.dp).clip(RoundedCornerShape(16.dp))
                    .background(MaterialTheme.colorScheme.surfaceContainer)
                    .border(
                        1.dp,
                        when {
                            attempt?.correct == true -> Palette.Green.copy(alpha = 0.55f)
                            attempt != null -> Palette.Amber.copy(alpha = 0.45f)
                            else -> Palette.Hairline
                        },
                        RoundedCornerShape(16.dp),
                    )
                    .semantics { liveRegion = LiveRegionMode.Polite }
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                verticalArrangement = Arrangement.Center,
            ) {
                Text(text, style = MaterialTheme.typography.titleMedium, color = colour)
                if (attempt != null) {
                    Spacer(Modifier.height(8.dp))
                    val barFill by animateFloatAsState(attempt.share.coerceIn(0.05f, 1f), tween(Motion.MEDIUM), label = "matchBar")
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Box(Modifier.weight(1f).height(6.dp).clip(RoundedCornerShape(3.dp)).background(Color.White.copy(alpha = 0.10f))) {
                            Box(Modifier.fillMaxWidth(barFill).fillMaxHeight().clip(RoundedCornerShape(3.dp))
                                .background(if (attempt.correct) Palette.Green else Palette.Amber))
                        }
                        Text(
                            stringResource(R.string.practice_match_pct, (attempt.share * 100).toInt()),
                            style = MaterialTheme.typography.labelMedium,
                            color = colour,
                        )
                    }
                }
            }
            Spacer(Modifier.height(12.dp))
            FlowRow(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                val advancePrimary = last || attempt?.correct == true
                if (advancePrimary) {
                    Button(
                        onClick = { if (last) onLessonEnd() else index = (index + 1) % words.size },
                        modifier = Modifier.heightIn(min = 48.dp),
                    ) {
                        Icon(if (last) Icons.Rounded.Flag else Icons.Rounded.SkipNext, null)
                        Spacer(Modifier.width(6.dp))
                        Text(if (last) stringResource(R.string.practice_finish_lesson) else stringResource(R.string.practice_next_sign))
                    }
                } else {
                    OutlinedButton(
                        onClick = { index = (index + 1) % words.size },
                        modifier = Modifier.heightIn(min = 48.dp),
                    ) {
                        Icon(Icons.Rounded.SkipNext, null)
                        Spacer(Modifier.width(6.dp))
                        Text(stringResource(R.string.practice_skip))
                    }
                }
                if (lesson == null) {
                    OutlinedButton(onClick = { choosing = true }, modifier = Modifier.heightIn(min = 48.dp)) {
                        Icon(Icons.Rounded.Search, null)
                        Spacer(Modifier.width(6.dp))
                        Text(stringResource(R.string.practice_choose))
                    }
                } else if (index > 0) {
                    OutlinedButton(onClick = { index-- }, modifier = Modifier.heightIn(min = 48.dp)) { Text(stringResource(R.string.practice_previous)) }
                }
            }
            Spacer(Modifier.height(12.dp))
            Text(
                when {
                    lesson != null || learned.isEmpty() ->
                        stringResource(if (front) R.string.practice_figure_note_mirrored else R.string.practice_figure_note)
                    else -> stringResource(R.string.practice_learned_2, learned.size, learned.takeLast(6).joinToString(", ", transform = show))
                },
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }

    BoxWithConstraints(Modifier.fillMaxSize()) {
        if (maxWidth > maxHeight) {
            // Landscape: watch, yourself, then the instructions, side by side.
            Row(Modifier.fillMaxSize().padding(horizontal = 12.dp, vertical = 6.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                watch(Modifier.weight(1f).fillMaxHeight())
                camera(Modifier.weight(1f).fillMaxHeight())
                panel(Modifier.weight(1.2f).fillMaxHeight())
            }
        } else {
            Column(Modifier.fillMaxSize()) {
                // Watch on the left, yourself on the right.
                Row(Modifier.fillMaxWidth().weight(0.58f).padding(horizontal = 12.dp, vertical = 6.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    watch(Modifier.weight(1f).fillMaxSize())
                    camera(Modifier.weight(1f).fillMaxSize())
                }
                panel(Modifier.fillMaxWidth().weight(0.42f))
            }
        }
    }

    if (choosing) {
        var query by remember { mutableStateOf("") }
        val shown = remember(query, words) { words.filter { show(it).contains(query.trim(), ignoreCase = true) } }
        AlertDialog(
            onDismissRequest = { choosing = false },
            title = { Text(stringResource(R.string.practice_choose_a_sign)) },
            text = {
                Column {
                    OutlinedTextField(query, { query = it.take(24) }, singleLine = true, label = { Text(stringResource(R.string.practice_search)) }, modifier = Modifier.fillMaxWidth())
                    Spacer(Modifier.height(8.dp))
                    LazyColumn(Modifier.fillMaxWidth().heightIn(max = 320.dp)) {
                        items(shown) { w ->
                            Row(Modifier.fillMaxWidth().clickable { index = words.indexOf(w); choosing = false }.heightIn(min = 48.dp)
                                .padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                                Text(show(w), Modifier.weight(1f), style = MaterialTheme.typography.titleMedium)
                                if (w in learned) Icon(Icons.Rounded.CheckCircle, stringResource(R.string.practice_learned), Modifier.size(20.dp), tint = Palette.Green)
                            }
                        }
                    }
                    if (shown.isEmpty()) Text(stringResource(R.string.practice_no_sign_with_a_demonstration), style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            },
            confirmButton = { TextButton(onClick = { choosing = false }) { Text(stringResource(R.string.practice_close)) } },
        )
    }
}

/** "Sign 2 of 5" and one mark per sign of the lesson: green when learned, a ring on the one being practised. */
@Composable
private fun LessonSteps(words: List<String>, index: Int, learned: List<String>) {
    Row(Modifier.padding(bottom = 8.dp).semantics(mergeDescendants = true) {}, verticalAlignment = Alignment.CenterVertically) {
        Text(stringResource(R.string.practice_sign_of, index + 1, words.size), style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.width(12.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            words.forEachIndexed { i, w ->
                val dot = Modifier.size(if (i == index) 12.dp else 10.dp).clip(CircleShape)
                Box(
                    when {
                        w in learned -> dot.background(Palette.Green)
                        i == index -> dot.border(2.dp, MaterialTheme.colorScheme.primary, CircleShape)
                        else -> dot.border(1.5.dp, Palette.Outline, CircleShape)
                    }
                )
            }
        }
    }
}

@Composable
private fun Tag(text: String, modifier: Modifier) {
    Text(text, modifier.glass(RoundedCornerShape(50)).padding(horizontal = 12.dp, vertical = 6.dp),
        style = MaterialTheme.typography.labelLarge, color = Palette.Text)
}

/**
 * The demonstration, looping: the sign at its own speed (half speed when [slow]), then a short hold. Scaled so the
 * sign's own extent fills the card, or only the hands' when [zoom]; [mirror] flips it left to right so it moves like
 * the learner's front-camera preview beside it (the learner copies a reflection, as in a mirror). The body is
 * shaded and the palms filled so the figure reads as a person, and a fading trail behind each wrist shows the
 * path of the movement.
 */
@Composable
internal fun DemoFigure(
    demo: Demo,
    slow: Boolean,
    mirror: Boolean,
    zoom: Boolean,
    modifier: Modifier,
    paused: Boolean = false,
    stepOverride: Float? = null,
) {
    val play = (demo.durationMs.coerceIn(600, 4000) * (if (slow) 2.2f else 1f)).toInt()
    val hold = 700
    var frozenStep by remember(demo.word) { mutableStateOf(0f) }
    val step = if (stepOverride != null) {
        stepOverride
    } else if (paused) {
        frozenStep
    } else {
        val transition = rememberInfiniteTransition(label = "demo")
        val animated by transition.animateFloat(
            0f, demo.frames - 1f,
            infiniteRepeatable(keyframes {
                durationMillis = play + hold
                0f at 0 using LinearEasing
                (demo.frames - 1f) at play using LinearEasing
                (demo.frames - 1f) at play + hold
            }, RepeatMode.Restart),
            label = "step",
        )
        frozenStep = animated
        animated
    }
    val body = Palette.TextMuted
    val left = Palette.FirstHand
    val right = MaterialTheme.colorScheme.primary
    // The box to fill: the whole sign with room for the head drawn around the nose, or just the hands.
    val target = remember(demo, zoom) { if (zoom) demo.handBounds.copyOf() else demo.bounds.copyOf().also { it[1] -= 0.07f } }
    val ease = tween<Float>(Motion.MEDIUM, easing = Motion.Standard)
    // Zoomed on the hands, the body steps back so the hands carry the picture.
    val dim by animateFloatAsState(if (zoom) 0.4f else 1f, ease, label = "dim")
    val bx0 by animateFloatAsState(target[0], ease, label = "x0")
    val by0 by animateFloatAsState(target[1], ease, label = "y0")
    val bx1 by animateFloatAsState(target[2], ease, label = "x1")
    val by1 by animateFloatAsState(target[3], ease, label = "y1")
    Canvas(modifier.clipToBounds()) {
        val cx = (bx0 + bx1) / 2; val cy = (by0 + by1) / 2
        val margin = if (zoom) 0.03f else 0.05f
        val least = if (zoom) 0.14f else 0.30f
        val scale = minOf(size.width / (maxOf(bx1 - bx0, least) + 2 * margin), size.height / (maxOf(by1 - by0, least) + 2 * margin))
        fun at(t: Float, k: Int): Offset? = demo.at(t, k)?.let { (x, y) ->
            val dx = (x - cx) * scale
            Offset(size.width / 2 + if (mirror) -dx else dx, size.height / 2 + (y - cy) * scale)
        }
        fun pt(k: Int) = at(step, k)
        val sl = pt(Demo.SHOULDER_L)
        val sr = pt(Demo.SHOULDER_R)
        // Line widths follow the figure's size, so zooming in gives fuller limbs, not hairlines.
        val shoulders = if (sl != null && sr != null) (sl - sr).getDistance() else size.width * 0.3f
        val limb = (shoulders * 0.10f).coerceIn(4.dp.toPx(), 22.dp.toPx())
        val finger = (limb * 0.42f).coerceIn(2.5.dp.toPx(), 9.dp.toPx())
        fun bone(a: Int, b: Int, colour: Color, width: Float) {
            val p = pt(a) ?: return
            val q = pt(b) ?: return
            drawLine(colour, p, q, strokeWidth = width, cap = StrokeCap.Round)
        }
        fun polygon(points: List<Offset?>): Path? {
            if (points.any { it == null }) return null
            return Path().apply {
                moveTo(points[0]!!.x, points[0]!!.y)
                for (o in points.drop(1)) lineTo(o!!.x, o.y)
                close()
            }
        }
        // Torso: filled between shoulders and hips, with a soft outline.
        polygon(listOf(sl, sr, pt(Demo.HIP_R), pt(Demo.HIP_L)))?.let { torso ->
            drawPath(torso, body.copy(alpha = 0.22f * dim))
            drawPath(torso, body.copy(alpha = 0.55f * dim), style = Stroke(2.dp.toPx(), join = StrokeJoin.Round))
        }
        // Head on the nose, as wide as two thirds of the shoulders, with a neck to the shoulders' middle.
        val nose = pt(Demo.NOSE)
        if (sl != null && sr != null && nose != null) {
            val r = shoulders * 0.33f
            val neck = Offset((sl.x + sr.x) / 2, (sl.y + sr.y) / 2)
            drawLine(body.copy(alpha = 0.7f * dim), neck, Offset(nose.x, nose.y + r * 0.9f), strokeWidth = limb, cap = StrokeCap.Round)
            drawCircle(body.copy(alpha = 0.22f * dim), radius = r, center = nose)
            drawCircle(body.copy(alpha = 0.7f * dim), radius = r, center = nose, style = Stroke(2.dp.toPx()))
        }
        // Arms.
        val arm = body.copy(alpha = 0.85f * dim)
        bone(Demo.SHOULDER_L, Demo.ELBOW_L, arm, limb)
        bone(Demo.ELBOW_L, Demo.WRIST_L, arm, limb)
        bone(Demo.SHOULDER_R, Demo.ELBOW_R, arm, limb)
        bone(Demo.ELBOW_R, Demo.WRIST_R, arm, limb)
        // Where each wrist has just been: a short trail of fading dots.
        for ((wrist, colour) in listOf(Demo.WRIST_L to left, Demo.WRIST_R to right)) {
            for (i in 1..TRAIL) {
                val t = step - i * 0.6f
                if (t < 0f) break
                val o = at(t, wrist) ?: continue
                val fade = 1f - i / (TRAIL + 1f)
                drawCircle(colour.copy(alpha = 0.30f * fade), radius = limb * 0.5f * (0.5f + 0.5f * fade), center = o)
            }
        }
        // Hands: a filled palm and round fingers, each hand in its own colour (as on the camera).
        for ((base, colour) in listOf(Demo.HAND_L to left, Demo.HAND_R to right)) {
            polygon(listOf(0, 1, 5, 9, 13, 17).map { pt(base + it) })?.let { palm ->
                drawPath(palm, colour.copy(alpha = 0.55f))
                drawPath(palm, colour, style = Stroke(finger * 0.6f, join = StrokeJoin.Round))
            }
            for (chain in FINGERS) for ((a, c) in chain.zipWithNext()) bone(base + a, base + c, colour, finger)
            for (k in 4 until 21 step 4) pt(base + k)?.let { drawCircle(Color.White, radius = finger * 0.45f, center = it) }
        }
    }
}

private const val TRAIL = 8
private val FINGERS = listOf(listOf(1, 2, 3, 4), listOf(5, 6, 7, 8), listOf(9, 10, 11, 12), listOf(13, 14, 15, 16), listOf(17, 18, 19, 20))
