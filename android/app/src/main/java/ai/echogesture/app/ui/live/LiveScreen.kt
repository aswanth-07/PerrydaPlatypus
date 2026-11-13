package ai.echogesture.app.ui.live

import android.app.Activity
import android.view.WindowManager
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
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cameraswitch
import androidx.compose.material.icons.filled.ClearAll
import androidx.compose.material.icons.filled.FrontHand
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.RecordVoiceOver
import androidx.compose.material.icons.filled.VoiceOverOff
import androidx.compose.material.icons.outlined.FrontHand
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.engine.HAND_EDGES
import ai.echogesture.app.engine.LiveState
import ai.echogesture.app.engine.VoiceState
import ai.echogesture.app.ui.components.ConfidenceRow
import ai.echogesture.app.ui.components.StatusPill
import ai.echogesture.app.ui.theme.CaptionStyle
import ai.echogesture.app.ui.theme.Palette
import java.util.concurrent.Executors

@Composable
fun LiveScreen(vm: MainViewModel) {
    val state by vm.live.collectAsStateWithLifecycle()
    val settings by vm.settings.collectAsStateWithLifecycle()
    val voice by vm.engine.speaker.state.collectAsStateWithLifecycle()
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

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        CameraPreview(backCamera = settings.backCamera, analyzer = vm.engine::analyze, modifier = Modifier.fillMaxSize())
        if (settings.showLandmarks) HandOverlay(state, mirror = !settings.backCamera, modifier = Modifier.fillMaxSize())

        StatusBar(state, if (settings.speak) voice else null, Modifier.align(Alignment.TopStart).padding(16.dp))
        AnimatedVisibility(
            visible = state.top.isNotEmpty() && !state.paused,
            modifier = Modifier.align(Alignment.TopEnd).padding(top = 64.dp, end = 16.dp),
            enter = fadeIn(), exit = fadeOut(),
        ) { Candidates(state.top) }

        Column(Modifier.align(Alignment.BottomCenter).fillMaxWidth()) {
            Controls(
                paused = state.paused,
                showLandmarks = settings.showLandmarks,
                speak = settings.speak,
                onSpeak = { vm.update { it.copy(speak = !it.speak) } },
                onFlip = { vm.update { it.copy(backCamera = !it.backCamera) }; vm.engine.clearCaption() },
                onPause = { vm.engine.setPaused(!state.paused) },
                onClear = vm.engine::clearCaption,
                onLandmarks = { vm.update { it.copy(showLandmarks = !it.showLandmarks) } },
                modifier = Modifier.align(Alignment.End).padding(end = 16.dp, bottom = 12.dp),
            )
            CaptionBand(state)
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

@Composable
internal fun CameraPreview(backCamera: Boolean, analyzer: (ImageProxy) -> Unit, modifier: Modifier) {
    val context = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER   // the overlay maps landmarks with the same fit
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
            provider.bindToLifecycle(owner, CameraSelector.Builder().requireLensFacing(lens).build(), preview, analysis)
        }, ContextCompat.getMainExecutor(context))
        onDispose { future.addListener({ future.get().unbindAll() }, ContextCompat.getMainExecutor(context)) }
    }
    AndroidView(factory = { previewView }, modifier = modifier)
}

/** Hand skeletons drawn in the same letterboxed rectangle the preview uses. */
@Composable
internal fun HandOverlay(state: LiveState, mirror: Boolean, modifier: Modifier) {
    val primary = MaterialTheme.colorScheme.primary
    Canvas(modifier) {
        if (state.imageWidth == 0 || state.imageHeight == 0) return@Canvas
        val scale = minOf(size.width / state.imageWidth, size.height / state.imageHeight)
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

@Composable
private fun StatusBar(state: LiveState, voice: VoiceState?, modifier: Modifier) {
    Column(modifier) {
        val (text, colour) = when {
            state.error != null -> "Hand tracking unavailable" to Palette.Red
            !state.ready -> "Loading" to Palette.TextMuted
            state.paused -> "Paused" to Palette.TextMuted
            state.handsVisible -> "Listening" to Palette.Green
            else -> "Show your hands" to Palette.Amber
        }
        StatusPill(text, colour)
        Spacer(Modifier.height(8.dp))
        Text(
            "%.0f fps  ·  landmarks %.0f ms  ·  model %.1f ms%s".format(state.fps, state.landmarksMs, state.modelMs, if (state.gpu) "  ·  GPU" else ""),
            Modifier.clip(RoundedCornerShape(8.dp)).background(Palette.Glass).padding(horizontal = 10.dp, vertical = 4.dp),
            style = MaterialTheme.typography.labelSmall, color = Palette.TextMuted,
        )
        if (voice != null) {
            Spacer(Modifier.height(6.dp))
            // Frame-to-sound: from the camera frame that settled the word to the first sound, median of the last words.
            val text = when {
                voice.error != null -> voice.error
                !voice.ready -> "Voice starting"
                voice.spoken == 0 -> "Voice on" + if (voice.onDevice) "  ·  on this phone" else ""
                else -> "Voice %.0f ms after the frame  ·  median %.0f ms".format(voice.lastMs, voice.medianMs)
            }
            Text(
                text,
                Modifier.clip(RoundedCornerShape(8.dp)).background(Palette.Glass).padding(horizontal = 10.dp, vertical = 4.dp),
                style = MaterialTheme.typography.labelSmall, color = if (voice.error != null) Palette.Red else Palette.TextMuted,
            )
        }
    }
}

@Composable
private fun Candidates(top: List<Pair<String, Float>>) {
    Column(
        Modifier.widthIn(max = 280.dp).clip(RoundedCornerShape(20.dp)).background(Palette.Glass).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("Best guesses", style = MaterialTheme.typography.labelMedium, color = Palette.TextMuted)
        top.forEachIndexed { i, (word, p) -> ConfidenceRow(word, p, highlight = i == 0) }
    }
}

@Composable
private fun CaptionBand(state: LiveState) {
    Box(
        Modifier.fillMaxWidth()
            .background(Palette.Glass, RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp))
            .padding(horizontal = 24.dp, vertical = 18.dp),
    ) {
        if (state.caption.isEmpty()) {
            Text(
                if (state.paused) "Paused" else "Sign in front of the camera — captions appear here",
                style = MaterialTheme.typography.titleMedium, color = Palette.TextMuted,
            )
        } else Column {
            Row(verticalAlignment = Alignment.CenterVertically) {
                val older = state.caption.dropLast(1).takeLast(5).joinToString(" ")
                if (older.isNotEmpty()) {
                    Text(older, style = CaptionStyle, color = Palette.Text.copy(alpha = 0.6f), maxLines = 1, overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f, fill = false))
                    Spacer(Modifier.width(12.dp))
                }
                AnimatedContent(
                    targetState = state.caption.last() to state.lastWordAt,
                    transitionSpec = { (slideInVertically { it / 2 } + fadeIn()) togetherWith fadeOut() },
                    label = "word",
                ) { (word, _) ->
                    Text(word, style = CaptionStyle, color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.ExtraBold)
                }
            }
            // One sign, several names: WLASL lists the same signing under DOCTOR and PHYSICIAN, so show both.
            val note = when {
                state.lastTaught -> "taught on this phone"
                state.lastAlso.isNotEmpty() -> "same sign: " + state.lastAlso.joinToString(" · ")
                else -> null
            }
            if (note != null) {
                Text(note, style = MaterialTheme.typography.titleSmall, color = Palette.Amber, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
        }
    }
}

@Composable
private fun Controls(
    paused: Boolean, showLandmarks: Boolean, speak: Boolean,
    onSpeak: () -> Unit, onFlip: () -> Unit, onPause: () -> Unit, onClear: () -> Unit, onLandmarks: () -> Unit,
    modifier: Modifier,
) {
    Row(modifier, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        ControlButton(if (speak) Icons.Filled.RecordVoiceOver else Icons.Filled.VoiceOverOff,
            if (speak) "Stop speaking captions" else "Speak captions aloud", onSpeak, active = speak)
        ControlButton(if (showLandmarks) Icons.Filled.FrontHand else Icons.Outlined.FrontHand, "Show hand tracking", onLandmarks)
        ControlButton(Icons.Filled.ClearAll, "Clear captions", onClear)
        ControlButton(if (paused) Icons.Filled.PlayArrow else Icons.Filled.Pause, if (paused) "Resume" else "Pause", onPause)
        ControlButton(Icons.Filled.Cameraswitch, "Switch camera", onFlip)
    }
}

@Composable
private fun ControlButton(icon: ImageVector, label: String, onClick: () -> Unit, active: Boolean = false) {
    FilledTonalIconButton(
        onClick = onClick,
        modifier = Modifier.size(52.dp),
        colors = IconButtonDefaults.filledTonalIconButtonColors(
            containerColor = if (active) MaterialTheme.colorScheme.primary else Palette.Glass,
            contentColor = if (active) MaterialTheme.colorScheme.onPrimary else Palette.Text,
        ),
    ) { Icon(icon, contentDescription = label) }
}
