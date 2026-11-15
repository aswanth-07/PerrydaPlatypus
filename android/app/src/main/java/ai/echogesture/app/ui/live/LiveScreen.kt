package ai.echogesture.app.ui.live

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.view.WindowManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
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
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cameraswitch
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.automirrored.filled.VolumeOff
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material.icons.automirrored.outlined.Backspace
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.engine.HAND_EDGES
import ai.echogesture.app.engine.LiveState
import ai.echogesture.app.engine.ReplyState
import ai.echogesture.app.engine.VoiceState
import ai.echogesture.app.ui.components.ConfidenceRow
import ai.echogesture.app.ui.components.StatusPill
import ai.echogesture.app.ui.theme.CaptionStyle
import ai.echogesture.app.ui.theme.Palette
import ai.echogesture.app.ui.theme.ReplyStyle
import kotlinx.coroutines.delay
import java.util.concurrent.Executors

private val VOCAB_SHORT = mapOf("core" to "100 signs", "full" to "2,000 signs", "union" to "3,075 signs")

/**
 * The conversation screen: the camera fills the screen, the signer's words appear in the caption card at the
 * bottom (and are spoken when Voice is on), and the hearing person taps Reply to speak back; their words appear
 * large at the top for the signer to read. Timings and best guesses appear only when "Show details" is on.
 */
@Composable
fun LiveScreen(vm: MainViewModel) {
    val state by vm.live.collectAsStateWithLifecycle()
    val settings by vm.settings.collectAsStateWithLifecycle()
    val voice by vm.engine.speaker.state.collectAsStateWithLifecycle()
    val reply by vm.engine.listener.state.collectAsStateWithLifecycle()
    val context = LocalContext.current

    DisposableEffect(Unit) {
        vm.engine.acquireLive()
        val window = (context as? Activity)?.window
        window?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onDispose {
            vm.engine.releaseLive()
            window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }

    // Reply: ask for the microphone the first time, then listen; tapping again stops and keeps what was heard.
    var note by remember { mutableStateOf<String?>(null) }
    val askMic = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) vm.engine.listener.start() else note = "Allow the microphone to hear spoken replies."
    }
    fun onReply() {
        when {
            reply.listening -> vm.engine.listener.stop()
            !reply.available -> note = reply.note
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED ->
                vm.engine.listener.start()
            else -> askMic.launch(Manifest.permission.RECORD_AUDIO)
        }
    }
    // Recogniser errors after an attempt; "unavailable" is explained only when Reply is tapped.
    LaunchedEffect(reply.note) { if (reply.available) reply.note?.let { note = it } }
    LaunchedEffect(note) { if (note != null) { delay(4500); note = null } }
    LaunchedEffect(reply.lastAt) { if (reply.last != null) { delay(15_000); vm.engine.listener.dismiss() } }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        // A benchmark replaying a video through this screen (--ez ui true) needs no camera.
        if (!vm.engine.replayOnly) {
            CameraPreview(backCamera = settings.backCamera, analyzer = vm.engine::analyze, modifier = Modifier.fillMaxSize())
        }
        if (settings.showLandmarks) HandOverlay(state, mirror = !settings.backCamera, modifier = Modifier.fillMaxSize())

        // Scrims keep text legible over any background without boxing the whole video in.
        Box(Modifier.fillMaxWidth().height(160.dp).background(Brush.verticalGradient(listOf(Color.Black.copy(alpha = 0.65f), Color.Transparent))))
        Box(
            Modifier.align(Alignment.BottomCenter).fillMaxWidth().height(320.dp)
                .background(Brush.verticalGradient(listOf(Color.Transparent, Color.Black.copy(alpha = 0.75f))))
        )

        Column(Modifier.fillMaxWidth().statusBarsPadding().padding(horizontal = 16.dp, vertical = 8.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                val (text, colour) = status(state)
                StatusPill(text + "  ·  " + (VOCAB_SHORT[settings.vocabulary] ?: settings.vocabulary), colour, Modifier.weight(1f, fill = false))
                Spacer(Modifier.width(8.dp))
                GlassIconButton(Icons.Filled.Cameraswitch, "Switch camera") {
                    vm.update { it.copy(backCamera = !it.backCamera) }
                    vm.engine.clearCaption()
                }
            }
            if (settings.showDetails) {
                Spacer(Modifier.height(8.dp))
                Details(state, if (settings.speak) voice else null)
            }
            Spacer(Modifier.height(12.dp))
            ReplyArea(reply, onDismiss = vm.engine.listener::dismiss)
        }

        Column(Modifier.align(Alignment.BottomCenter).fillMaxWidth().padding(horizontal = 12.dp, vertical = 12.dp)) {
            AnimatedVisibility(note != null, enter = fadeIn(), exit = fadeOut()) {
                Text(
                    note.orEmpty(),
                    Modifier.fillMaxWidth().padding(bottom = 10.dp).clip(RoundedCornerShape(14.dp)).background(Palette.Glass)
                        .padding(horizontal = 16.dp, vertical = 10.dp),
                    style = MaterialTheme.typography.bodyMedium, color = Palette.Text,
                )
            }
            CaptionCard(state, atRest = settings.boundary, onClear = vm.engine::clearCaption, onCorrect = vm.engine::correctLast)
            Spacer(Modifier.height(14.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly, verticalAlignment = Alignment.Bottom) {
                LabelledControl(
                    if (settings.speak) Icons.AutoMirrored.Filled.VolumeUp else Icons.AutoMirrored.Filled.VolumeOff,
                    if (settings.speak) "Voice on" else "Voice off", active = settings.speak,
                ) { vm.update { it.copy(speak = !it.speak) } }
                MicControl(reply, onClick = ::onReply)
                LabelledControl(
                    if (state.paused) Icons.Filled.PlayArrow else Icons.Filled.Pause,
                    if (state.paused) "Resume" else "Pause", active = state.paused,
                ) { vm.engine.setPaused(!state.paused) }
            }
        }

        if (!state.ready) {
            Column(Modifier.align(Alignment.Center), horizontalAlignment = Alignment.CenterHorizontally) {
                CircularProgressIndicator()
                Spacer(Modifier.height(12.dp))
                Text("Loading on-device models", color = Palette.Text)
            }
        }
    }
}

private fun status(state: LiveState): Pair<String, Color> = when {
    state.error != null -> "Hand tracking unavailable" to Palette.Red
    !state.ready -> "Loading" to Palette.TextMuted
    state.paused -> "Paused" to Palette.TextMuted
    state.recording != null -> "Recording" to Palette.Red
    state.signing -> "Signing" to Palette.Green
    state.handsVisible -> "Watching" to Palette.Green
    state.bodyVisible -> "Show your hands" to Palette.Amber
    else -> "Step back" to Palette.Amber
}

@Composable
internal fun CameraPreview(backCamera: Boolean, analyzer: (ImageProxy) -> Unit, modifier: Modifier) {
    val context = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER   // full-bleed; the overlay maps landmarks with the same crop
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    val executor = remember { Executors.newSingleThreadExecutor() }
    DisposableEffect(Unit) { onDispose { executor.shutdown() } }

    DisposableEffect(backCamera, owner) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val provider = future.get()
            val fourThree = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY).build()
            val preview = Preview.Builder().setResolutionSelector(fourThree).build()
            preview.setSurfaceProvider(previewView.surfaceProvider)
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(fourThree)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(executor) { proxy -> analyzer(proxy) }
            val lens = if (backCamera) CameraSelector.LENS_FACING_BACK else CameraSelector.LENS_FACING_FRONT
            provider.unbindAll()
            runCatching { provider.bindToLifecycle(owner, CameraSelector.Builder().requireLensFacing(lens).build(), preview, analysis) }
        }, ContextCompat.getMainExecutor(context))
        onDispose { future.addListener({ future.get().unbindAll() }, ContextCompat.getMainExecutor(context)) }
    }
    androidx.compose.ui.viewinterop.AndroidView(factory = { previewView }, modifier = modifier)
}

/** Hand skeletons drawn with the same centre crop the full-bleed preview uses. */
@Composable
internal fun HandOverlay(state: LiveState, mirror: Boolean, modifier: Modifier) {
    val primary = MaterialTheme.colorScheme.primary
    Canvas(modifier) {
        if (state.imageWidth == 0 || state.imageHeight == 0) return@Canvas
        val scale = maxOf(size.width / state.imageWidth, size.height / state.imageHeight)
        val w = state.imageWidth * scale
        val h = state.imageHeight * scale
        val left = (size.width - w) / 2
        val top = (size.height - h) / 2
        state.hands.forEachIndexed { slot, hand ->
            hand ?: return@forEachIndexed
            fun pt(k: Int): Offset {
                val x = hand[k * 3].let { if (mirror) 1 - it else it }
                return Offset(left + x * w, top + hand[k * 3 + 1] * h)
            }
            val colour = if (slot == 1) primary else Palette.Amber
            for ((a, b) in HAND_EDGES) drawLine(colour.copy(alpha = 0.85f), pt(a), pt(b), strokeWidth = 3.dp.toPx(), cap = StrokeCap.Round)
            for (k in 0 until 21) drawCircle(Color.White, radius = (if (k % 4 == 0) 4.5f else 3f).dp.toPx(), center = pt(k))
        }
    }
}

/** Timings, voice delay and the model's best guesses: for demos and debugging, off by default. */
@Composable
private fun Details(state: LiveState, voice: VoiceState?) {
    Column(
        Modifier.widthIn(max = 320.dp).clip(RoundedCornerShape(16.dp)).background(Palette.Glass).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            "%.0f fps · tracking %.0f ms · model %.1f ms · %d %s tracker%s".format(state.fps, state.landmarksMs, state.modelMs, state.lanes, if (state.gpu) "GPU" else "CPU", if (state.lanes == 1) "" else "s"),
            style = MaterialTheme.typography.labelMedium, color = Palette.TextMuted,
        )
        if (voice != null && voice.spoken > 0) {
            Text("voice %.0f ms after the frame · median %.0f ms".format(voice.lastMs, voice.medianMs),
                style = MaterialTheme.typography.labelMedium, color = Palette.TextMuted)
        }
        state.top.forEachIndexed { i, (word, p) -> ConfidenceRow(word, p, highlight = i == 0) }
    }
}

/** The hearing person's side: "Listening…" with the words so far, then the finished reply, large. */
@Composable
private fun ReplyArea(reply: ReplyState, onDismiss: () -> Unit) {
    AnimatedVisibility(
        visible = reply.listening || reply.last != null,
        enter = fadeIn() + slideInVertically { -it / 3 },
        exit = fadeOut() + slideOutVertically { -it / 3 },
    ) {
        val listening = reply.listening
        Column(
            Modifier.fillMaxWidth().clip(RoundedCornerShape(24.dp)).background(Palette.ReplyDeep.copy(alpha = 0.92f))
                .clickable(enabled = !listening, onClickLabel = "Dismiss reply", onClick = onDismiss)
                .padding(horizontal = 20.dp, vertical = 16.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(if (listening) "Listening…" else "They said", style = MaterialTheme.typography.labelLarge, color = Palette.Reply,
                    modifier = Modifier.weight(1f))
                if (!listening) Icon(Icons.Filled.Close, "Dismiss", tint = Palette.Reply.copy(alpha = 0.7f), modifier = Modifier.size(18.dp))
            }
            Spacer(Modifier.height(6.dp))
            val text = if (listening) reply.partial.ifEmpty { "Speak now" } else reply.last.orEmpty()
            Text(text, style = ReplyStyle, color = if (listening && reply.partial.isEmpty()) Palette.Reply.copy(alpha = 0.6f) else Palette.Reply,
                maxLines = 4, overflow = TextOverflow.Ellipsis)
        }
    }
}

@Composable
private fun CaptionCard(state: LiveState, atRest: Boolean, onClear: () -> Unit, onCorrect: (String) -> Unit) {
    Column(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(24.dp)).background(Palette.Glass)
            .padding(start = 20.dp, end = 8.dp, top = 14.dp, bottom = 16.dp),
    ) {
        if (state.caption.isEmpty()) {
            Text(
                when {
                    state.paused -> "Paused"
                    !state.bodyVisible -> "Hands are found from the body: keep your head, shoulders and both hands in view."
                    atRest -> "Sign, then lower your hands. The word appears as they come to rest."
                    else -> "Sign in front of the camera. Words appear here."
                },
                Modifier.padding(end = 12.dp, top = 4.dp),
                style = MaterialTheme.typography.titleMedium, color = Palette.TextMuted,
            )
            return@Column
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            val older = state.caption.dropLast(1).takeLast(4).joinToString(" ")
            Row(Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically) {
                if (older.isNotEmpty()) {
                    Text(older, style = CaptionStyle, color = Palette.Text.copy(alpha = 0.55f), maxLines = 1,
                        overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f, fill = false))
                    Spacer(Modifier.width(10.dp))
                }
                AnimatedContent(
                    targetState = state.caption.last() to state.lastWordAt,
                    transitionSpec = { (slideInVertically { it / 2 } + fadeIn()) togetherWith fadeOut() },
                    label = "word",
                ) { (word, _) ->
                    Text(word, style = CaptionStyle, color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.ExtraBold, maxLines = 1)
                }
            }
            IconButton(onClick = onClear) { Icon(Icons.AutoMirrored.Outlined.Backspace, "Clear captions", tint = Palette.TextMuted) }
        }
        // One sign, several names: WLASL lists the same signing under DOCTOR and PHYSICIAN, so show both.
        val sub = when {
            state.lastTaught -> "taught on this phone"
            state.lastAlso.isNotEmpty() -> "same sign: " + state.lastAlso.joinToString(" · ")
            else -> null
        }
        if (sub != null) Text(sub, style = MaterialTheme.typography.titleSmall, color = Palette.Amber, maxLines = 1, overflow = TextOverflow.Ellipsis)
        // Other signs it could have been, offered when the model was not sure: one tap swaps the word, here, in the
        // conversation and aloud. They leave with the next word, or after a few seconds.
        var offer by remember { mutableStateOf(true) }
        LaunchedEffect(state.lastWordAt) { offer = true; delay(ALTERNATIVES_SHOWN_MS); offer = false }
        AnimatedVisibility(offer && state.lastAlternatives.isNotEmpty(), enter = fadeIn(), exit = fadeOut()) {
            Row(Modifier.padding(top = 8.dp, end = 12.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("or", style = MaterialTheme.typography.titleSmall, color = Palette.TextMuted)
                for (w in state.lastAlternatives) {
                    Box(
                        Modifier.weight(1f, fill = false).clip(RoundedCornerShape(50)).background(Palette.PanelHigh)
                            .clickable(onClickLabel = "Change the last word to $w") { onCorrect(w) }
                            .heightIn(min = 44.dp).padding(horizontal = 16.dp),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(w, style = MaterialTheme.typography.titleSmall, color = Palette.Text, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                }
            }
        }
    }
}

private const val ALTERNATIVES_SHOWN_MS = 8_000L

@Composable
private fun LabelledControl(icon: ImageVector, label: String, active: Boolean, onClick: () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        FilledIconButton(
            onClick = onClick,
            modifier = Modifier.size(56.dp).semantics { contentDescription = label },
            colors = IconButtonDefaults.filledIconButtonColors(
                containerColor = if (active) MaterialTheme.colorScheme.secondaryContainer else Palette.Glass,
                contentColor = if (active) MaterialTheme.colorScheme.onSecondaryContainer else Palette.Text,
            ),
        ) { Icon(icon, contentDescription = null) }
        Spacer(Modifier.height(6.dp))
        Text(label, style = MaterialTheme.typography.labelMedium, color = Palette.Text)
    }
}

/** The main control: tap to let the hearing person reply; it pulses while listening. */
@Composable
private fun MicControl(reply: ReplyState, onClick: () -> Unit) {
    val pulse by rememberInfiniteTransition(label = "mic").animateFloat(
        initialValue = 1f, targetValue = 1.08f,
        animationSpec = infiniteRepeatable(tween(700), RepeatMode.Reverse), label = "pulse",
    )
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        FilledIconButton(
            onClick = onClick,
            modifier = Modifier.size(72.dp).scale(if (reply.listening) pulse else 1f)
                .semantics { contentDescription = if (reply.listening) "Stop listening" else "Reply by speaking" },
            colors = IconButtonDefaults.filledIconButtonColors(
                containerColor = if (reply.listening) Palette.Reply else MaterialTheme.colorScheme.primary,
                contentColor = if (reply.listening) Palette.ReplyDeep else MaterialTheme.colorScheme.onPrimary,
            ),
        ) { Icon(if (reply.listening) Icons.Filled.Stop else Icons.Filled.Mic, contentDescription = null, modifier = Modifier.size(32.dp)) }
        Spacer(Modifier.height(6.dp))
        Text(if (reply.listening) "Stop" else "Reply", style = MaterialTheme.typography.labelMedium, color = Palette.Text)
    }
}

@Composable
internal fun GlassIconButton(icon: ImageVector, label: String, size: Dp = 48.dp, onClick: () -> Unit) {
    FilledIconButton(
        onClick = onClick, modifier = Modifier.size(size),
        colors = IconButtonDefaults.filledIconButtonColors(containerColor = Palette.Glass, contentColor = Palette.Text),
    ) { Icon(icon, contentDescription = label) }
}
