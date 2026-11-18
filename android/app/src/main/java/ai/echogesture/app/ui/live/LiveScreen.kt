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
import androidx.compose.material.icons.rounded.Cameraswitch
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Mic
import androidx.compose.material.icons.rounded.Pause
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Stop
import androidx.compose.material.icons.automirrored.rounded.VolumeOff
import androidx.compose.material.icons.automirrored.rounded.VolumeUp
import androidx.compose.material.icons.automirrored.rounded.Backspace
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
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
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
import ai.echogesture.app.R
import ai.echogesture.app.engine.HAND_EDGES
import ai.echogesture.app.engine.UndoOffer
import ai.echogesture.app.ui.components.rememberPermission
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.combinedClickable
import androidx.compose.material3.TextButton
import androidx.compose.ui.res.stringResource
import ai.echogesture.app.engine.LiveState
import ai.echogesture.app.engine.ReplyState
import ai.echogesture.app.engine.VoiceState
import ai.echogesture.app.engine.TranscriptEntry
import ai.echogesture.app.ui.components.ConfidenceRow
import ai.echogesture.app.ui.components.LocalCaption
import ai.echogesture.app.ui.components.TrackingError
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.ColumnScope
import ai.echogesture.app.ui.components.LocalHaptics
import ai.echogesture.app.ui.theme.Motion
import ai.echogesture.app.ui.theme.glass
import ai.echogesture.app.ui.components.FramingGuide
import ai.echogesture.app.ui.history.ConversationBubbles
import ai.echogesture.app.ui.history.conversationText
import ai.echogesture.app.ui.history.copyText
import ai.echogesture.app.data.Conversations
import ai.echogesture.app.ui.history.shareText
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.rounded.ContentCopy
import androidx.compose.material.icons.rounded.Forum
import androidx.compose.material.icons.rounded.Fullscreen
import androidx.compose.material.icons.rounded.ScreenRotation
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material.icons.rounded.Share
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.material3.OutlinedButton
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import ai.echogesture.app.ui.components.StatusPill
import ai.echogesture.app.ui.theme.CaptionStyle
import ai.echogesture.app.ui.theme.Palette
import ai.echogesture.app.ui.theme.ReplyStyle
import kotlinx.coroutines.delay
import java.util.concurrent.Executors

/**
 * The Talk tab: the camera fills the screen, the signer's words appear in the caption card at the bottom (and are
 * spoken when Speak words is on), and the hearing person taps Reply to speak back; their words appear large at the
 * top, and full screen for the signer with Show to signer. The conversation so far pulls up from the top bar.
 * Timings and best guesses appear only with developer options' "Show details".
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LiveScreen(vm: MainViewModel, onSettings: () -> Unit, onHistory: () -> Unit) {
    val state by vm.live.collectAsStateWithLifecycle()
    val settings by vm.settings.collectAsStateWithLifecycle()
    val voice by vm.engine.speaker.state.collectAsStateWithLifecycle()
    val reply by vm.engine.listener.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val haptics = LocalHaptics.current
    val skeleton = settings.developer && settings.skeleton

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
    // When Android will no longer ask, the note offers the app's settings page instead of failing silently.
    var note by remember { mutableStateOf<String?>(null) }
    var noteOffersSettings by remember { mutableStateOf(false) }
    val micDenied = stringResource(R.string.mic_denied)
    val micBlocked = stringResource(R.string.mic_blocked)
    val mic = rememberPermission(Manifest.permission.RECORD_AUDIO) { ok, blocked ->
        if (ok) vm.engine.listener.start() else { note = if (blocked) micBlocked else micDenied; noteOffersSettings = blocked }
    }
    fun onReply() {
        when {
            reply.listening -> vm.engine.listener.stop()
            !reply.available -> { note = reply.note; noteOffersSettings = false }
            mic.granted -> vm.engine.listener.start()
            mic.blocked -> { note = micBlocked; noteOffersSettings = true }
            else -> mic.request()
        }
    }
    // Recogniser errors after an attempt; "unavailable" is explained only when Reply is tapped.
    LaunchedEffect(reply.note) { if (reply.available) reply.note?.let { note = it; noteOffersSettings = false } }
    LaunchedEffect(note) { if (note != null) { delay(if (noteOffersSettings) 9000 else 4500); note = null } }
    LaunchedEffect(state.undo?.at) { if (state.undo != null) { delay(UNDO_SHOWN_MS); vm.engine.dismissUndo() } }
    // A reply shown full screen stays until it is closed; otherwise it leaves after 15 s.
    var showReply by remember { mutableStateOf(false) }
    LaunchedEffect(reply.lastAt, showReply) { if (reply.last != null && !showReply) { delay(15_000); vm.engine.listener.dismiss() } }
    var conversationOpen by remember { mutableStateOf(false) }
    val back = settings.backCamera
    val firstWordAt = remember { state.lastWordAt }
    LaunchedEffect(state.lastWordAt) { if (state.lastWordAt != firstWordAt && state.caption.isNotEmpty()) haptics.tick() }

    // The video and everything drawn over it.
    val stage: @Composable BoxScope.() -> Unit = {
        // A benchmark replaying a video through this screen (--ez ui true) needs no camera.
        if (!vm.engine.replayOnly) {
            CameraPreview(backCamera = settings.backCamera, analyzer = vm.engine::analyze, modifier = Modifier.fillMaxSize())
        }
        if (settings.showLandmarks) HandOverlay(state, mirror = !settings.backCamera, skeleton = skeleton, modifier = Modifier.fillMaxSize())
        // No body in view means no hands and no captions: show where the signer's head and shoulders should be.
        AnimatedVisibility(
            state.ready && state.running && !state.paused && state.error == null && !state.bodyVisible,
            enter = fadeIn(), exit = fadeOut(),
        ) { FramingGuide(ok = false, modifier = Modifier.fillMaxSize().padding(horizontal = 40.dp, vertical = 120.dp)) }

        // Scrims keep text legible over any background without boxing the whole video in.
        Box(Modifier.fillMaxWidth().height(160.dp).background(Brush.verticalGradient(listOf(Color.Black.copy(alpha = 0.65f), Color.Transparent))))

        Column(Modifier.fillMaxWidth().statusBarsPadding().padding(horizontal = 16.dp, vertical = 8.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                val (text, colour) = status(state, back)
                StatusPill(text, colour, Modifier.weight(1f, fill = false))
                Spacer(Modifier.width(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    GlassIconButton(Icons.Rounded.Forum, "Conversation so far") { conversationOpen = true }
                    GlassIconButton(Icons.Rounded.Cameraswitch, if (back) "Switch to the front camera" else "Switch to the back camera") {
                        vm.update { it.copy(backCamera = !it.backCamera) }
                        vm.engine.clearCaption()
                    }
                    GlassIconButton(Icons.Rounded.Settings, "Settings", onClick = onSettings)
                }
            }
            if (settings.showDetails) {
                Spacer(Modifier.height(8.dp))
                Details(state, if (settings.speak) voice else null)
            }
            Spacer(Modifier.height(12.dp))
            ReplyArea(reply, onDismiss = vm.engine.listener::dismiss, onShow = { showReply = true })
        }

        TrackingError(state, settings.gpu, onRetry = vm.engine::retryLive, onSwitch = { vm.update { it.copy(gpu = !it.gpu) } },
            modifier = Modifier.align(Alignment.Center).padding(16.dp))
        if (!state.ready) {
            Column(Modifier.align(Alignment.Center), horizontalAlignment = Alignment.CenterHorizontally) {
                CircularProgressIndicator()
                Spacer(Modifier.height(12.dp))
                Text("Loading on-device models", color = Palette.Text)
            }
        }
    }
    // Notes, the undo offer, the captions and the three controls.
    val panel: @Composable ColumnScope.() -> Unit = {
        AnimatedVisibility(note != null, enter = fadeIn(), exit = fadeOut()) {
            Row(
                Modifier.fillMaxWidth().padding(bottom = 10.dp).glass(RoundedCornerShape(14.dp))
                    .padding(start = 16.dp, end = 4.dp, top = 4.dp, bottom = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(note.orEmpty(), Modifier.weight(1f).padding(vertical = 6.dp), style = MaterialTheme.typography.bodyMedium, color = Palette.Text)
                if (noteOffersSettings) TextButton(onClick = { mic.openSettings(); note = null }) { Text(stringResource(R.string.action_open_settings)) }
            }
        }
        AnimatedVisibility(state.undo != null, enter = fadeIn(), exit = fadeOut()) {
            val u = state.undo
            Row(
                Modifier.fillMaxWidth().padding(bottom = 10.dp).glass(RoundedCornerShape(14.dp))
                    .padding(start = 16.dp, end = 4.dp, top = 4.dp, bottom = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    if (u?.kind == UndoOffer.Kind.REMOVED_WORD) stringResource(R.string.undo_removed_word, u.word.orEmpty()) else stringResource(R.string.undo_cleared),
                    Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium, color = Palette.Text,
                )
                TextButton(onClick = vm.engine::undo) { Text(stringResource(R.string.action_undo)) }
            }
        }
        CaptionCard(state, back = back, atRest = !settings.continuous, onRemoveLast = vm.engine::removeLastWord,
            onClearAll = vm.engine::clearCaptionByUser, onCorrect = { w -> haptics.confirm(); vm.engine.correctLast(w) })
        Spacer(Modifier.height(14.dp))
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly, verticalAlignment = Alignment.Bottom) {
            // A toggle named for its action; the fill shows whether it is on.
            LabelledControl(
                if (settings.speak) Icons.AutoMirrored.Rounded.VolumeUp else Icons.AutoMirrored.Rounded.VolumeOff,
                "Speak words", active = settings.speak, toggle = true,
            ) { vm.update { it.copy(speak = !it.speak) } }
            MicControl(reply, onClick = ::onReply)
            LabelledControl(
                if (state.paused) Icons.Rounded.PlayArrow else Icons.Rounded.Pause,
                if (state.paused) "Resume" else "Pause", active = state.paused,
            ) { vm.engine.setPaused(!state.paused) }
        }
    }

    BoxWithConstraints(Modifier.fillMaxSize().background(Color.Black)) {
        if (maxWidth > maxHeight) {
            // Landscape: the video on the left, captions and controls in a panel beside it, so neither covers the other.
            Row(Modifier.fillMaxSize()) {
                Box(Modifier.weight(1f).fillMaxHeight()) { stage() }
                Column(
                    Modifier.width(380.dp).fillMaxHeight().background(Palette.Ink).statusBarsPadding()
                        .verticalScroll(rememberScrollState()).padding(12.dp),
                    verticalArrangement = Arrangement.Bottom,
                ) { panel() }
            }
        } else {
            Box(Modifier.fillMaxSize()) {
                stage()
                Box(
                    Modifier.align(Alignment.BottomCenter).fillMaxWidth().height(320.dp)
                        .background(Brush.verticalGradient(listOf(Color.Transparent, Color.Black.copy(alpha = 0.75f))))
                )
                Column(Modifier.align(Alignment.BottomCenter).fillMaxWidth().padding(horizontal = 12.dp, vertical = 12.dp)) { panel() }
            }
        }
    }

    if (showReply && reply.last != null) {
        ReplyForSigner(reply.last.orEmpty(), onClose = { showReply = false; vm.engine.listener.dismiss() })
    }
    if (conversationOpen) {
        val entries by vm.transcript.collectAsStateWithLifecycle()
        ModalBottomSheet(
            onDismissRequest = { conversationOpen = false },
            sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
            containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
        ) {
            ConversationSheet(entries, onAll = { conversationOpen = false; onHistory() })
        }
    }
}

/** The conversation under way, pulled up over the camera: copy or share it, or open History for older ones. */
@Composable
private fun ConversationSheet(entries: List<TranscriptEntry>, onAll: () -> Unit) {
    val context = LocalContext.current
    val current = remember(entries) { Conversations.current(entries, System.currentTimeMillis()) }
    Column(Modifier.fillMaxWidth().fillMaxHeight(0.75f)) {
        Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("This conversation", Modifier.weight(1f), style = MaterialTheme.typography.titleLarge)
            val text = conversationText(current, LocalCaption.current)
            IconButton(onClick = { copyText(context, text) }, enabled = current.isNotEmpty()) { Icon(Icons.Rounded.ContentCopy, "Copy conversation") }
            IconButton(onClick = { shareText(context, text) }, enabled = current.isNotEmpty()) { Icon(Icons.Rounded.Share, "Share conversation") }
        }
        if (current.isEmpty()) {
            Text(
                "Signed words and spoken replies appear here as you talk. A pause of more than 30 minutes starts a new conversation.",
                Modifier.padding(horizontal = 20.dp, vertical = 16.dp),
                style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.weight(1f))
        } else {
            ConversationBubbles(current, rememberLazyListState(), Modifier.weight(1f))
        }
        TextButton(onClick = onAll, modifier = Modifier.align(Alignment.CenterHorizontally).padding(bottom = 12.dp)) {
            Text("All conversations")
        }
    }
}

/**
 * The reply full screen, for the signer to read. Rotate turns it upside down, for a signer across a table from
 * the person holding the phone.
 */
@Composable
private fun ReplyForSigner(text: String, onClose: () -> Unit) {
    var flipped by remember { mutableStateOf(false) }
    Dialog(onDismissRequest = onClose, properties = DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false)) {
        Box(Modifier.fillMaxSize().background(Palette.ReplyDeep).statusBarsPadding().navigationBarsPadding()) {
            Box(Modifier.fillMaxSize().padding(bottom = 80.dp).rotate(if (flipped) 180f else 0f), contentAlignment = Alignment.Center) {
                Text(
                    text, Modifier.padding(28.dp).verticalScroll(rememberScrollState()),
                    style = ReplyStyle.copy(fontSize = 44.sp, lineHeight = 54.sp), color = Palette.Reply,
                )
            }
            Row(Modifier.align(Alignment.BottomCenter).fillMaxWidth().padding(16.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                OutlinedButton(onClick = { flipped = !flipped }, modifier = Modifier.heightIn(min = 48.dp)) {
                    Icon(Icons.Rounded.ScreenRotation, null, tint = Palette.Reply)
                    Spacer(Modifier.width(8.dp))
                    Text("Rotate", color = Palette.Reply)
                }
                Button(onClick = onClose, modifier = Modifier.heightIn(min = 48.dp)) { Text("Done") }
            }
        }
    }
}

/** The status pill, worded for who is in front of the camera: someone else (back camera) or the person holding it. */
private fun status(state: LiveState, back: Boolean): Pair<String, Color> = when {
    state.error != null -> "Hand tracking unavailable" to Palette.Red
    !state.ready -> "Loading" to Palette.TextMuted
    state.paused -> "Paused" to Palette.TextMuted
    state.recording != null -> "Recording" to Palette.Red
    state.signing -> "Signing" to Palette.Green
    state.handsVisible -> "Watching" to Palette.Green
    state.bodyVisible -> (if (back) "Their hands aren't in view" else "Show your hands") to Palette.Amber
    else -> (if (back) "Ask them to step back" else "Step back") to Palette.Amber
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
    // The app handles rotation itself, so the use cases are bound again with the new display rotation; otherwise
    // analysis frames reach the tracker sideways in landscape and no body is found.
    LocalConfiguration.current
    val rotation = LocalView.current.display?.rotation ?: android.view.Surface.ROTATION_0

    DisposableEffect(backCamera, owner, rotation) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val provider = future.get()
            val fourThree = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY).build()
            val preview = Preview.Builder().setResolutionSelector(fourThree).setTargetRotation(rotation).build()
            preview.setSurfaceProvider(previewView.surfaceProvider)
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(fourThree)
                .setTargetRotation(rotation)
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
    // Clipped: the preview's TextureView is scaled to fill (centre crop) and otherwise draws past its own box,
    // over the tab rail beside it in landscape.
    androidx.compose.ui.viewinterop.AndroidView(factory = { previewView }, modifier = modifier.clipToBounds())
}

/**
 * The tracked hands, drawn with the same centre crop the full-bleed preview uses: a soft glow over each hand
 * (amber for the first, cyan for the second, as in the Learn figure) with a dot on each fingertip, so the person
 * sees the app is following the hands without a wireframe over them. [skeleton] (developer options) adds the
 * bones and joints.
 */
@Composable
internal fun HandOverlay(state: LiveState, mirror: Boolean, skeleton: Boolean, modifier: Modifier) {
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
            // The palm's centre (wrist and the four knuckles) and the hand's reach from it.
            val palm = listOf(0, 5, 9, 13, 17).map(::pt)
            val centre = Offset(palm.map { it.x }.average().toFloat(), palm.map { it.y }.average().toFloat())
            val reach = (0 until 21).maxOf { (pt(it) - centre).getDistance() }.coerceAtLeast(12.dp.toPx())
            val radius = reach * 1.35f
            drawCircle(
                Brush.radialGradient(
                    0f to colour.copy(alpha = 0.40f), 0.55f to colour.copy(alpha = 0.16f), 1f to Color.Transparent,
                    center = centre, radius = radius,
                ),
                radius = radius, center = centre,
            )
            for (k in listOf(4, 8, 12, 16, 20)) drawCircle(colour, radius = 3.5.dp.toPx(), center = pt(k))
            if (skeleton) {
                for ((a, b) in HAND_EDGES) drawLine(colour.copy(alpha = 0.85f), pt(a), pt(b), strokeWidth = 2.dp.toPx(), cap = StrokeCap.Round)
                for (k in 0 until 21) drawCircle(Color.White, radius = 2.5.dp.toPx(), center = pt(k))
            }
        }
    }
}

/** Timings, voice delay and the model's best guesses: for demos and debugging, off by default. */
@Composable
private fun Details(state: LiveState, voice: VoiceState?) {
    Column(
        Modifier.widthIn(max = 320.dp).glass(RoundedCornerShape(16.dp)).padding(12.dp),
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
private fun ReplyArea(reply: ReplyState, onDismiss: () -> Unit, onShow: () -> Unit) {
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
                if (!listening) Icon(Icons.Rounded.Close, "Dismiss", tint = Palette.Reply.copy(alpha = 0.7f), modifier = Modifier.size(18.dp))
            }
            Spacer(Modifier.height(6.dp))
            val text = if (listening) reply.partial.ifEmpty { "Speak now" } else reply.last.orEmpty()
            Text(text, style = ReplyStyle, color = if (listening && reply.partial.isEmpty()) Palette.Reply.copy(alpha = 0.6f) else Palette.Reply,
                maxLines = 4, overflow = TextOverflow.Ellipsis)
            if (!listening) {
                Spacer(Modifier.height(4.dp))
                TextButton(onClick = onShow, modifier = Modifier.offset(x = (-12).dp)) {
                    Icon(Icons.Rounded.Fullscreen, null, tint = Palette.Reply)
                    Spacer(Modifier.width(6.dp))
                    Text("Show to signer", color = Palette.Reply)
                }
            }
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun CaptionCard(state: LiveState, back: Boolean, atRest: Boolean, onRemoveLast: () -> Unit, onClearAll: () -> Unit, onCorrect: (String) -> Unit) {
    var reopen by remember { mutableIntStateOf(0) }
    val show = LocalCaption.current
    Column(
        Modifier.fillMaxWidth().glass(RoundedCornerShape(24.dp))
            .semantics { liveRegion = LiveRegionMode.Polite }
            .padding(start = 20.dp, end = 8.dp, top = 14.dp, bottom = 16.dp),
    ) {
        if (state.caption.isEmpty() && state.pending != null) {
            // The first sign is under way: its word so far, not yet settled.
            Text(show(state.pending) + "…", Modifier.padding(end = 12.dp, top = 4.dp), style = CaptionStyle, color = Palette.TextMuted, maxLines = 1)
            return@Column
        }
        if (state.caption.isEmpty()) {
            Text(
                when {
                    state.paused -> "Paused"
                    !state.bodyVisible && back -> "Hold the phone so their head, shoulders and both hands are in view."
                    !state.bodyVisible -> "Keep your head, shoulders and both hands in view."
                    atRest && back -> "Each word appears when their hands come to rest after a sign."
                    atRest -> "Sign, then lower your hands. The word appears as they come to rest."
                    back -> "Captions appear here as they sign, one word at a time."
                    else -> "Sign one word after another. Each appears as you make it; keep your hands up."
                },
                Modifier.padding(end = 12.dp, top = 4.dp),
                style = MaterialTheme.typography.titleMedium, color = Palette.TextMuted,
            )
            return@Column
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            val older = state.caption.dropLast(1).takeLast(4).joinToString(" ", transform = show)
            Row(Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically) {
                if (older.isNotEmpty()) {
                    Text(older, style = CaptionStyle, color = Palette.Text.copy(alpha = 0.72f), maxLines = 1,
                        overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f, fill = false))
                    Spacer(Modifier.width(10.dp))
                }
                // Tapping the newest word brings back the other signs it could have been.
                val canOffer = state.lastAlternatives.isNotEmpty()
                AnimatedContent(
                    targetState = state.caption.last() to state.lastWordAt,
                    transitionSpec = {
                        (slideInVertically(Motion.wordSpring) { it / 2 } + fadeIn(tween(Motion.SHORT))) togetherWith fadeOut(tween(Motion.SHORT))
                    },
                    label = "word",
                    modifier = if (canOffer) Modifier.clip(RoundedCornerShape(10.dp)).clickable(onClickLabel = "Show other signs it could be") { reopen++ } else Modifier,
                ) { (word, _) ->
                    Text(show(word), style = CaptionStyle, color = MaterialTheme.colorScheme.primary, maxLines = 1)
                }
                // The sign under way: shown muted until it settles, as live speech captions do.
                state.pending?.let {
                    Spacer(Modifier.width(10.dp))
                    Text("${show(it)}…", style = CaptionStyle, color = Palette.TextMuted, maxLines = 1, overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f, fill = false))
                }
            }
            // Tap: remove the newest word (it was wrong). Long press: clear every caption. Both can be undone.
            val removeLabel = stringResource(R.string.caption_remove_last)
            val clearLabel = stringResource(R.string.caption_clear_all)
            Box(
                Modifier.size(48.dp).clip(CircleShape)
                    .combinedClickable(onClickLabel = removeLabel, onLongClickLabel = clearLabel, onLongClick = onClearAll, onClick = onRemoveLast)
                    .semantics { contentDescription = removeLabel },
                contentAlignment = Alignment.Center,
            ) { Icon(Icons.AutoMirrored.Rounded.Backspace, null, tint = Palette.TextMuted) }
        }
        // One sign, several names: WLASL lists the same signing under DOCTOR and PHYSICIAN, so show both.
        val sub = when {
            state.lastTaught -> "taught on this phone"
            state.lastAlso.isNotEmpty() -> "same sign: " + state.lastAlso.joinToString(" · ", transform = show)
            else -> null
        }
        if (sub != null) Text(sub, style = MaterialTheme.typography.titleSmall, color = Palette.Amber, maxLines = 1, overflow = TextOverflow.Ellipsis)
        // Other signs it could have been, offered when the model was not sure: one tap swaps the word, here, in the
        // conversation and aloud. They leave with the next word, or after a few seconds.
        var offer by remember { mutableStateOf(true) }
        LaunchedEffect(state.lastWordAt, reopen) { offer = true; delay(ALTERNATIVES_SHOWN_MS); offer = false }
        AnimatedVisibility(offer && state.lastAlternatives.isNotEmpty(), enter = fadeIn(), exit = fadeOut()) {
            Row(Modifier.padding(top = 8.dp, end = 12.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Not right?", style = MaterialTheme.typography.titleSmall, color = Palette.TextMuted)
                for (w in state.lastAlternatives) {
                    Box(
                        Modifier.weight(1f, fill = false).clip(RoundedCornerShape(50)).background(Palette.PanelHigh)
                            .clickable(onClickLabel = "Change the last word to ${show(w)}") { onCorrect(w) }
                            .heightIn(min = 44.dp).padding(horizontal = 16.dp),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(show(w), style = MaterialTheme.typography.titleSmall, color = Palette.Text, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                }
            }
        }
    }
}

private const val ALTERNATIVES_SHOWN_MS = 8_000L
private const val UNDO_SHOWN_MS = 5_000L

@Composable
private fun LabelledControl(icon: ImageVector, label: String, active: Boolean, toggle: Boolean = false, onClick: () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        FilledIconButton(
            onClick = onClick,
            modifier = Modifier.size(56.dp).then(if (active) Modifier else Modifier.glass(CircleShape)).semantics {
                contentDescription = label
                if (toggle) stateDescription = if (active) "On" else "Off"
            },
            colors = IconButtonDefaults.filledIconButtonColors(
                containerColor = if (active) MaterialTheme.colorScheme.secondaryContainer else Color.Transparent,
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
        ) { Icon(if (reply.listening) Icons.Rounded.Stop else Icons.Rounded.Mic, contentDescription = null, modifier = Modifier.size(32.dp)) }
        Spacer(Modifier.height(6.dp))
        Text(if (reply.listening) "Stop" else "Reply", style = MaterialTheme.typography.labelMedium, color = Palette.Text)
    }
}

@Composable
internal fun GlassIconButton(icon: ImageVector, label: String, size: Dp = 48.dp, onClick: () -> Unit) {
    FilledIconButton(
        onClick = onClick, modifier = Modifier.size(size).glass(CircleShape),
        colors = IconButtonDefaults.filledIconButtonColors(containerColor = Color.Transparent, contentColor = Palette.Text),
    ) { Icon(icon, contentDescription = label) }
}
