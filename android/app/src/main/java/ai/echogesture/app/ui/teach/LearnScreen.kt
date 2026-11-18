package ai.echogesture.app.ui.teach

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
import androidx.compose.material.icons.filled.Cameraswitch
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.SkipNext
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
import ai.echogesture.app.ui.components.StatusPill
import ai.echogesture.app.ui.live.CameraPreview
import ai.echogesture.app.ui.live.HandOverlay
import ai.echogesture.app.ui.theme.CaptionStyle
import ai.echogesture.app.ui.theme.Palette

/**
 * The app teaches a sign: a stick figure shows it (keypoints of one training clip, no video), the learner copies
 * it in front of the camera, and the live recogniser says whether that was the sign. The check asks only "was
 * that the word asked for" (SignEngine.practiceTarget), which is easier than naming one sign of 2,000.
 */
@Composable
fun LearnScreen(vm: MainViewModel) {
    val live by vm.live.collectAsStateWithLifecycle()
    val settings by vm.settings.collectAsStateWithLifecycle()
    val context = LocalContext.current

    // Signs that have a demonstration and that the model can recognise.
    val demos = remember { Demos.load(context) }
    val known = remember(live.ready) { vm.engine.labels().toSet() }
    val words = remember(demos, known) { demos.keys.filter { it in known } }

    var index by rememberSaveable { mutableIntStateOf(0) }
    var slow by rememberSaveable { mutableStateOf(false) }
    val progress by vm.progress.collectAsStateWithLifecycle()   // kept on the phone, not just for this session
    val learned = progress.learned
    var choosing by remember { mutableStateOf(false) }
    // Practice faces the learner, so Learn starts on the front camera whatever the Live tab uses.
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
            vm.engine.releaseLive()
            window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }
    LaunchedEffect(word) { vm.engine.practiceTarget = word }
    LaunchedEffect(attempt?.at) {
        val a = attempt ?: return@LaunchedEffect
        if (a.correct) vm.markLearned(a.target)
    }

    if (word == null) {
        Box(Modifier.fillMaxSize().padding(28.dp), contentAlignment = Alignment.Center) {
            Text(
                if (demos.isEmpty()) "This build has no sign demonstrations packed." else "Loading the model…",
                style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        return
    }
    val demo = demos.getValue(word)

    Column(Modifier.fillMaxSize()) {
        // Watch on the left, yourself on the right.
        Row(Modifier.fillMaxWidth().weight(0.58f).padding(horizontal = 12.dp, vertical = 6.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Box(Modifier.weight(1f).fillMaxSize().clip(RoundedCornerShape(24.dp)).background(Palette.Panel)
                .semantics { contentDescription = "Demonstration of the sign $word" }) {
                DemoFigure(demo, slow, mirror = front, Modifier.fillMaxSize().padding(top = 44.dp, bottom = 40.dp))
                Tag("Watch", Modifier.align(Alignment.TopStart).padding(10.dp))
                FilterChip(selected = slow, onClick = { slow = !slow }, label = { Text("Slow") },
                    modifier = Modifier.align(Alignment.BottomStart).padding(start = 10.dp, bottom = 4.dp))
            }
            Box(Modifier.weight(1f).fillMaxSize().clip(RoundedCornerShape(24.dp)).background(Color.Black)) {
                CameraPreview(backCamera = !front, analyzer = vm.engine::analyze, modifier = Modifier.fillMaxSize())
                if (settings.showLandmarks) HandOverlay(live, mirror = front, modifier = Modifier.fillMaxSize())
                StatusPill(
                    when {
                        live.signing -> "Signing"
                        live.handsVisible -> "Ready"
                        live.bodyVisible -> "Raise your hands"
                        else -> "Step back"
                    },
                    if (live.signing || live.handsVisible) Palette.Green else Palette.Amber,
                    Modifier.align(Alignment.TopStart).padding(10.dp),
                )
                FilledTonalIconButton(
                    onClick = { front = !front },
                    modifier = Modifier.align(Alignment.BottomEnd).padding(8.dp).size(48.dp),
                    colors = IconButtonDefaults.filledTonalIconButtonColors(containerColor = Palette.Glass, contentColor = Palette.Text),
                ) { Icon(Icons.Filled.Cameraswitch, "Switch camera") }
            }
        }

        Column(Modifier.fillMaxWidth().weight(0.42f).verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 6.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Sign this", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Text(word, style = CaptionStyle, color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.ExtraBold,
                        maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
                if (word in learned) Icon(Icons.Filled.CheckCircle, "Learned", Modifier.size(32.dp), tint = Palette.Green)
            }
            Spacer(Modifier.height(10.dp))
            // What the recogniser made of the last try.
            val (text, colour) = when {
                attempt == null && live.pending != null -> "Looking…" to Palette.TextMuted
                attempt == null -> "Watch the figure, then make the sign in front of the camera." to Palette.TextMuted
                attempt.correct -> "Yes, that was $word." to Palette.Green
                attempt.share >= 0.2f -> "Close. It looked a little more like ${attempt.seen}. Watch once more and try again." to Palette.Amber
                else -> "That looked like ${attempt.seen}. Watch once more and try again." to Palette.Amber
            }
            Box(Modifier.fillMaxWidth().heightIn(min = 64.dp).clip(RoundedCornerShape(16.dp)).background(MaterialTheme.colorScheme.surfaceContainer)
                .padding(horizontal = 16.dp, vertical = 12.dp), contentAlignment = Alignment.CenterStart) {
                Text(text, style = MaterialTheme.typography.titleMedium, color = colour)
            }
            Spacer(Modifier.height(12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                Button(onClick = { index = (index + 1) % words.size }, modifier = Modifier.height(48.dp)) {
                    Icon(Icons.Filled.SkipNext, null)
                    Spacer(Modifier.width(6.dp))
                    Text(if (attempt?.correct == true) "Next sign" else "Skip")
                }
                OutlinedButton(onClick = { choosing = true }, modifier = Modifier.height(48.dp)) {
                    Icon(Icons.Filled.Search, null)
                    Spacer(Modifier.width(6.dp))
                    Text("Choose")
                }
            }
            Spacer(Modifier.height(12.dp))
            Text(
                if (learned.isEmpty()) "${words.size} signs to learn. The figure is a real signer's keypoints${if (front) ", mirrored like your camera" else ""}; no video is stored."
                else "${learned.size} learned: " + learned.takeLast(6).joinToString(", "),
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }

    if (choosing) {
        var query by remember { mutableStateOf("") }
        val shown = remember(query, words) { words.filter { it.contains(query.trim(), ignoreCase = true) } }
        AlertDialog(
            onDismissRequest = { choosing = false },
            title = { Text("Choose a sign") },
            text = {
                Column {
                    OutlinedTextField(query, { query = it.take(24) }, singleLine = true, label = { Text("Search") },
                        keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Characters), modifier = Modifier.fillMaxWidth())
                    Spacer(Modifier.height(8.dp))
                    LazyColumn(Modifier.fillMaxWidth().heightIn(max = 320.dp)) {
                        items(shown) { w ->
                            Row(Modifier.fillMaxWidth().clickable { index = words.indexOf(w); choosing = false }.heightIn(min = 48.dp)
                                .padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                                Text(w, Modifier.weight(1f), style = MaterialTheme.typography.titleMedium)
                                if (w in learned) Icon(Icons.Filled.CheckCircle, "Learned", Modifier.size(20.dp), tint = Palette.Green)
                            }
                        }
                    }
                    if (shown.isEmpty()) Text("No sign with a demonstration matches.", style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            },
            confirmButton = { TextButton(onClick = { choosing = false }) { Text("Close") } },
        )
    }
}

@Composable
private fun Tag(text: String, modifier: Modifier) {
    Text(text, modifier.clip(RoundedCornerShape(50)).background(Palette.Glass).padding(horizontal = 12.dp, vertical = 6.dp),
        style = MaterialTheme.typography.labelLarge, color = Palette.Text)
}

/**
 * The demonstration, looping: the sign at its own speed (half speed when [slow]), then a short hold. Scaled so the
 * sign's own extent fills the card; [mirror] flips it left to right so it moves like the learner's front-camera
 * preview beside it (the learner copies a reflection, as in a mirror).
 */
@Composable
private fun DemoFigure(demo: Demo, slow: Boolean, mirror: Boolean, modifier: Modifier) {
    val play = (demo.durationMs.coerceIn(600, 4000) * (if (slow) 2.2f else 1f)).toInt()
    val hold = 700
    val transition = rememberInfiniteTransition(label = "demo")
    val step by transition.animateFloat(
        0f, demo.frames - 1f,
        infiniteRepeatable(keyframes {
            durationMillis = play + hold
            0f at 0 using LinearEasing
            (demo.frames - 1f) at play using LinearEasing
            (demo.frames - 1f) at play + hold
        }, RepeatMode.Restart),
        label = "step",
    )
    val body = Palette.TextMuted
    val left = Palette.Amber
    val right = MaterialTheme.colorScheme.primary
    // The figure fills the card: the box the sign's points stay inside, with room for the head drawn around the nose.
    val b = remember(demo) { demo.bounds.copyOf().also { it[1] -= 0.07f } }
    Canvas(modifier) {
        val cx = (b[0] + b[2]) / 2; val cy = (b[1] + b[3]) / 2
        val margin = 0.05f
        val scale = minOf(size.width / (maxOf(b[2] - b[0], 0.30f) + 2 * margin), size.height / (maxOf(b[3] - b[1], 0.30f) + 2 * margin))
        fun pt(k: Int): Offset? = demo.at(step, k)?.let { (x, y) ->
            val dx = (x - cx) * scale
            Offset(size.width / 2 + if (mirror) -dx else dx, size.height / 2 + (y - cy) * scale)
        }
        fun bone(a: Int, b: Int, colour: Color, width: Float) {
            val p = pt(a) ?: return
            val q = pt(b) ?: return
            drawLine(colour, p, q, strokeWidth = width, cap = StrokeCap.Round)
        }
        val thick = 5.dp.toPx()
        // Trunk and arms.
        bone(Demo.SHOULDER_L, Demo.SHOULDER_R, body, thick)
        bone(Demo.SHOULDER_L, Demo.HIP_L, body, thick)
        bone(Demo.SHOULDER_R, Demo.HIP_R, body, thick)
        bone(Demo.HIP_L, Demo.HIP_R, body, thick)
        bone(Demo.SHOULDER_L, Demo.ELBOW_L, body, thick)
        bone(Demo.ELBOW_L, Demo.WRIST_L, body, thick)
        bone(Demo.SHOULDER_R, Demo.ELBOW_R, body, thick)
        bone(Demo.ELBOW_R, Demo.WRIST_R, body, thick)
        // Head: a circle on the nose, as wide as two thirds of the shoulders.
        val sl = pt(Demo.SHOULDER_L)
        val sr = pt(Demo.SHOULDER_R)
        val nose = pt(Demo.NOSE)
        if (sl != null && sr != null && nose != null) {
            drawCircle(body, radius = (sl - sr).getDistance() * 0.33f, center = nose, style = Stroke(thick))
        }
        // Hands, each in its own colour (the same colours as the tracking overlay on the camera).
        for ((base, colour) in listOf(Demo.HAND_L to left, Demo.HAND_R to right)) {
            for ((a, c) in HAND_EDGES) bone(base + a, base + c, colour, 3.5.dp.toPx())
            for (k in 4 until 21 step 4) pt(base + k)?.let { drawCircle(Color.White, radius = 2.5.dp.toPx(), center = it) }
        }
    }
}
