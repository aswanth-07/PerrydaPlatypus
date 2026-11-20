package ai.echogesture.app.ui.live

import android.Manifest
import android.app.Activity
import android.util.Size
import android.view.WindowManager
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.calculateStartPadding
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.Backspace
import androidx.compose.material.icons.automirrored.rounded.Send
import androidx.compose.material.icons.automirrored.rounded.VolumeOff
import androidx.compose.material.icons.automirrored.rounded.VolumeUp
import androidx.compose.material.icons.rounded.Cameraswitch
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.DeleteSweep
import androidx.compose.material.icons.rounded.Forum
import androidx.compose.material.icons.rounded.Fullscreen
import androidx.compose.material.icons.rounded.Keyboard
import androidx.compose.material.icons.rounded.KeyboardArrowDown
import androidx.compose.material.icons.rounded.Mic
import androidx.compose.material.icons.rounded.Pause
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.RecordVoiceOver
import androidx.compose.material.icons.rounded.ScreenRotation
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material.icons.rounded.Share
import androidx.compose.material.icons.rounded.Stop
import androidx.compose.material.icons.rounded.Translate
import androidx.compose.material.icons.rounded.Tune
import androidx.compose.material.icons.rounded.ContentCopy
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.R
import ai.echogesture.app.data.Conversations
import ai.echogesture.app.data.Demo
import ai.echogesture.app.data.Demos
import ai.echogesture.app.data.Settings
import ai.echogesture.app.data.SignToken
import ai.echogesture.app.data.SignTranslator
import ai.echogesture.app.engine.HAND_EDGES
import ai.echogesture.app.engine.LiveState
import ai.echogesture.app.engine.OverlayState
import ai.echogesture.app.engine.ReplyState
import ai.echogesture.app.engine.TranscriptEntry
import ai.echogesture.app.engine.UndoOffer
import ai.echogesture.app.engine.VoiceState
import ai.echogesture.app.ui.components.ConfidenceRow
import ai.echogesture.app.ui.components.ConversationArt
import ai.echogesture.app.ui.components.FramingGuide
import ai.echogesture.app.ui.components.LocalCaption
import ai.echogesture.app.ui.components.LocalHaptics
import ai.echogesture.app.ui.components.LocalTabBarInset
import ai.echogesture.app.ui.components.StatusPill
import ai.echogesture.app.ui.components.TrackingError
import ai.echogesture.app.ui.components.description
import ai.echogesture.app.ui.components.rememberPermission
import ai.echogesture.app.ui.components.rememberPress
import ai.echogesture.app.ui.history.ConversationBubbles
import ai.echogesture.app.ui.history.conversationText
import ai.echogesture.app.ui.history.copyText
import ai.echogesture.app.ui.history.shareText
import ai.echogesture.app.ui.teach.SignSequencePlayer
import ai.echogesture.app.ui.theme.CaptionStyle
import ai.echogesture.app.ui.theme.Motion
import ai.echogesture.app.ui.theme.Palette
import ai.echogesture.app.ui.theme.ReplyStyle
import ai.echogesture.app.ui.theme.glass
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.StateFlow
import java.util.concurrent.Executors

/**
 * The Talk tab: live continuous sign captions over the camera with smooth lock-in feedback, tap-to-correct
 * alternatives, on-demand phrase speech, quick caption-mode tuning, and true two-way replies (offline voice,
 * offline typed/quick-phrases, and animated real-signer ASL "Reply in Sign").
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
    val demos = remember { Demos.load(context) }
    val synonyms = remember(state.ready) { vm.engine.synonymMap() }

    DisposableEffect(Unit) {
        vm.engine.acquireLive()
        val window = (context as? Activity)?.window
        window?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onDispose {
            vm.engine.listener.stop()
            vm.engine.releaseLive()
            window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }

    var note by remember { mutableStateOf<String?>(null) }
    var noteOffersSettings by remember { mutableStateOf(false) }
    var typeReplyOpen by remember { mutableStateOf(false) }
    var quickTuneOpen by remember { mutableStateOf(false) }
    var showReply by remember { mutableStateOf(false) }
    var showReplyInSign by remember { mutableStateOf(false) }
    var conversationOpen by remember { mutableStateOf(false) }

    val micDenied = stringResource(R.string.mic_denied)
    val micBlocked = stringResource(R.string.mic_blocked)
    val mic = rememberPermission(Manifest.permission.RECORD_AUDIO) { ok, blocked ->
        if (ok) vm.engine.listener.start() else { note = if (blocked) micBlocked else micDenied; noteOffersSettings = blocked }
    }
    fun onReply() {
        when {
            reply.listening -> vm.engine.listener.stop()
            // If offline STT is unavailable on this phone, open offline Type & Quick-Reply seamlessly instead of dead-ending!
            !reply.available -> { typeReplyOpen = true }
            mic.granted -> vm.engine.listener.start()
            mic.blocked -> { note = micBlocked; noteOffersSettings = true }
            else -> mic.request()
        }
    }

    val replyTokens = remember(reply.last, demos, synonyms) {
        reply.last?.let { SignTranslator.translate(it, demos.keys, synonyms) }.orEmpty()
    }
    val replySignCount = remember(replyTokens, demos) { replyTokens.count { it.gloss != null && it.gloss in demos } }

    LaunchedEffect(reply.note) { if (reply.available) reply.note?.let { note = it; noteOffersSettings = false } }
    LaunchedEffect(note) { if (note != null) { delay(if (noteOffersSettings) 9000 else 4500); note = null } }
    LaunchedEffect(state.undo?.at) { if (state.undo != null) { delay(UNDO_SHOWN_MS); vm.engine.dismissUndo() } }
    LaunchedEffect(reply.lastAt, showReply) { if (reply.last != null && !showReply) { delay(18_000); vm.engine.listener.dismiss() } }

    val back = settings.backCamera
    val firstWordAt = remember { state.lastWordAt }
    LaunchedEffect(state.lastWordAt) { if (state.lastWordAt != firstWordAt && state.caption.isNotEmpty()) haptics.tick() }

    val stage: @Composable BoxScope.() -> Unit = {
        if (!vm.engine.replayOnly) {
            CameraPreview(backCamera = settings.backCamera, analyzer = vm.engine::analyze, modifier = Modifier.fillMaxSize())
        }
        if (settings.showLandmarks) HandOverlay(vm.overlay, mirror = !settings.backCamera, skeleton = skeleton, modifier = Modifier.fillMaxSize())
        AnimatedVisibility(
            state.ready && state.running && !state.paused && state.error == null && !state.bodyVisible && !reply.listening && reply.last == null,
            enter = fadeIn(), exit = fadeOut(),
        ) { FramingGuide(ok = false, modifier = Modifier.fillMaxSize().padding(horizontal = 40.dp, vertical = 120.dp)) }

        Box(Modifier.fillMaxWidth().height(172.dp).background(Brush.verticalGradient(listOf(Color.Black.copy(alpha = 0.70f), Color.Transparent))))

        Column(
            Modifier.fillMaxWidth().statusBarsPadding()
                .padding(start = LocalTabBarInset.current.calculateStartPadding(LayoutDirection.Ltr))
                .padding(horizontal = 16.dp, vertical = 8.dp),
        ) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                val (text, colour) = status(state, back)
                StatusPill(text, colour, Modifier.weight(1f, fill = false))
                Spacer(Modifier.width(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    GlassIconButton(Icons.Rounded.Forum, stringResource(R.string.talk_conversation_so_far)) { conversationOpen = true }
                    GlassIconButton(Icons.Rounded.Cameraswitch, if (back) stringResource(R.string.talk_switch_to_the_front_camera) else stringResource(R.string.talk_switch_to_the_back_camera)) {
                        vm.update { it.copy(backCamera = !it.backCamera) }
                        vm.engine.clearCaption()
                    }
                    GlassIconButton(Icons.Rounded.Settings, stringResource(R.string.common_settings), onClick = onSettings)
                }
            }
            Spacer(Modifier.height(8.dp))
            // 1-tap Live Caption Mode pill: lets the user switch Continuous / After-each-sign and sensitivity without leaving Talk.
            ModePill(settings = settings, onClick = { quickTuneOpen = true })

            if (settings.showDetails) {
                Spacer(Modifier.height(8.dp))
                Details(state, vm.overlay, if (settings.speak) voice else null)
            }
            Spacer(Modifier.height(10.dp))
            ReplyArea(
                reply = reply,
                signCount = replySignCount,
                onDismiss = vm.engine.listener::dismiss,
                onShowText = { showReplyInSign = false; showReply = true },
                onShowSign = { showReplyInSign = true; showReply = true },
            )
        }

        TrackingError(state, settings.gpu, onRetry = vm.engine::retryLive, onSwitch = { vm.update { it.copy(gpu = !it.gpu) } },
            modifier = Modifier.align(Alignment.Center).padding(16.dp))
        if (!state.ready) {
            Column(Modifier.align(Alignment.Center), horizontalAlignment = Alignment.CenterHorizontally) {
                CircularProgressIndicator()
                Spacer(Modifier.height(12.dp))
                Text(stringResource(R.string.talk_loading_on_device_models), color = Palette.Text)
            }
        }
    }

    val panel: @Composable ColumnScope.() -> Unit = {
        AnimatedVisibility(note != null, enter = fadeIn(), exit = fadeOut()) {
            Row(
                Modifier.fillMaxWidth().padding(bottom = 10.dp).glass(RoundedCornerShape(16.dp))
                    .padding(start = 16.dp, end = 6.dp, top = 4.dp, bottom = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(note.orEmpty(), Modifier.weight(1f).padding(vertical = 6.dp), style = MaterialTheme.typography.bodyMedium, color = Palette.Text)
                if (noteOffersSettings) TextButton(onClick = { mic.openSettings(); note = null }) { Text(stringResource(R.string.action_open_settings)) }
                TextButton(onClick = { note = null; typeReplyOpen = true }) { Text(stringResource(R.string.talk_type_reply)) }
            }
        }
        AnimatedVisibility(state.undo != null, enter = fadeIn(), exit = fadeOut()) {
            val u = state.undo
            Row(
                Modifier.fillMaxWidth().padding(bottom = 10.dp).glass(RoundedCornerShape(16.dp))
                    .padding(start = 16.dp, end = 6.dp, top = 4.dp, bottom = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    if (u?.kind == UndoOffer.Kind.REMOVED_WORD) stringResource(R.string.undo_removed_word, u.word.orEmpty()) else stringResource(R.string.undo_cleared),
                    Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium, color = Palette.Text,
                )
                TextButton(onClick = vm.engine::undo) { Text(stringResource(R.string.action_undo)) }
            }
        }
        CaptionCard(
            state = state,
            back = back,
            atRest = !settings.continuous,
            onSpeakSentence = { haptics.tick(); vm.engine.speakCaption() },
            onRemoveLast = vm.engine::removeLastWord,
            onClearAll = vm.engine::clearCaptionByUser,
            onCorrect = { w -> haptics.confirm(); vm.engine.correctLast(w) },
        )
        Spacer(Modifier.height(10.dp))
        // Frosted high-contrast control dock keeps every icon and label crisp over any camera scene.
        Row(
            Modifier
                .fillMaxWidth()
                .glass(RoundedCornerShape(28.dp), Palette.Glass)
                .padding(horizontal = 10.dp, vertical = 10.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.Bottom,
        ) {
            DockControl(
                icon = if (settings.speak) Icons.AutoMirrored.Rounded.VolumeUp else Icons.AutoMirrored.Rounded.VolumeOff,
                label = stringResource(R.string.talk_speak_short),
                contentDesc = stringResource(R.string.talk_speak_words),
                active = settings.speak,
                toggle = true,
            ) { vm.update { it.copy(speak = !it.speak) } }

            MicControl(reply = reply, onClick = ::onReply)

            DockControl(
                icon = Icons.Rounded.Keyboard,
                label = stringResource(R.string.talk_type_short),
                contentDesc = stringResource(R.string.talk_type_or_sign_reply),
                active = typeReplyOpen,
            ) { typeReplyOpen = true }

            DockControl(
                icon = if (state.paused) Icons.Rounded.PlayArrow else Icons.Rounded.Pause,
                label = if (state.paused) stringResource(R.string.talk_resume) else stringResource(R.string.talk_pause),
                contentDesc = if (state.paused) stringResource(R.string.talk_resume) else stringResource(R.string.talk_pause),
                active = state.paused,
            ) { vm.engine.setPaused(!state.paused) }
        }
    }

    BoxWithConstraints(Modifier.fillMaxSize().background(Color.Black)) {
        if (maxWidth > maxHeight) {
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
                    Modifier.align(Alignment.BottomCenter).fillMaxWidth().height(340.dp)
                        .background(Brush.verticalGradient(listOf(Color.Transparent, Color.Black.copy(alpha = 0.78f))))
                )
                Column(
                    Modifier.align(Alignment.BottomCenter).fillMaxWidth().padding(LocalTabBarInset.current)
                        .padding(horizontal = 12.dp, vertical = 10.dp),
                ) { panel() }
            }
        }
    }

    if (showReply && reply.last != null) {
        ReplyForSigner(
            text = reply.last.orEmpty(),
            tokens = replyTokens,
            demos = demos,
            startInSign = showReplyInSign && replySignCount > 0,
            onClose = { showReply = false; vm.engine.listener.dismiss() },
        )
    }

    if (typeReplyOpen) {
        ModalBottomSheet(
            onDismissRequest = { typeReplyOpen = false },
            sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
            containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
        ) {
            TypeReplySheet(
                demos = demos,
                synonyms = synonyms,
                onSend = { text, openInSign ->
                    vm.engine.listener.submitTyped(text)
                    typeReplyOpen = false
                    if (openInSign) {
                        showReplyInSign = true
                        showReply = true
                    }
                },
            )
        }
    }

    if (quickTuneOpen) {
        ModalBottomSheet(
            onDismissRequest = { quickTuneOpen = false },
            sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
            containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
        ) {
            QuickTuneSheet(
                settings = settings,
                onUpdate = vm::update,
                onAllSettings = { quickTuneOpen = false; onSettings() },
            )
        }
    }

    if (conversationOpen) {
        val entries by vm.transcript.collectAsStateWithLifecycle()
        ModalBottomSheet(
            onDismissRequest = { conversationOpen = false },
            sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
            containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
        ) {
            ConversationSheet(
                entries = entries,
                onTypeReply = { conversationOpen = false; typeReplyOpen = true },
                onAll = { conversationOpen = false; onHistory() },
            )
        }
    }
}

@Composable
private fun ModePill(settings: Settings, onClick: () -> Unit) {
    val modeLabel = if (settings.continuous) {
        val pace = when (settings.caution) {
            0 -> stringResource(R.string.settings_more_words)
            2 -> stringResource(R.string.settings_fewer_surer)
            else -> stringResource(R.string.settings_balanced)
        }
        stringResource(R.string.talk_mode_continuous_pill, pace)
    } else {
        stringResource(R.string.talk_mode_rest_pill)
    }
    Row(
        Modifier
            .glass(CircleShape, Palette.Glass)
            .clickable(role = Role.Button, onClickLabel = stringResource(R.string.talk_tune_captions), onClick = onClick)
            .padding(start = 12.dp, end = 8.dp, top = 6.dp, bottom = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        Icon(Icons.Rounded.Tune, null, Modifier.size(14.dp), tint = MaterialTheme.colorScheme.primary)
        Text(modeLabel, style = MaterialTheme.typography.labelMedium, color = Palette.LightAzure, maxLines = 1)
        Icon(Icons.Rounded.KeyboardArrowDown, null, Modifier.size(16.dp), tint = Palette.LightAzure.copy(alpha = 0.8f))
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun QuickTuneSheet(settings: Settings, onUpdate: ((Settings) -> Settings) -> Unit, onAllSettings: () -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp)
            .padding(bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(stringResource(R.string.talk_quick_tune_title), style = MaterialTheme.typography.titleLarge, modifier = Modifier.weight(1f))
            TextButton(onClick = onAllSettings) { Text(stringResource(R.string.talk_all_settings)) }
        }

        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(stringResource(R.string.settings_captions_appear), style = MaterialTheme.typography.titleSmall)
            val modes = listOf(stringResource(R.string.settings_as_they_sign), stringResource(R.string.settings_after_each_sign))
            SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                modes.forEachIndexed { i, label ->
                    SegmentedButton(
                        selected = (i == 0) == settings.continuous,
                        onClick = { onUpdate { it.copy(continuous = i == 0) } },
                        shape = SegmentedButtonDefaults.itemShape(i, modes.size),
                    ) { Text(label, maxLines = 1) }
                }
            }
            Text(
                if (settings.continuous) stringResource(R.string.settings_each_word_shows_while_the)
                else stringResource(R.string.settings_one_word_per_sign_shown),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        if (settings.continuous) {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(stringResource(R.string.settings_caption_style), style = MaterialTheme.typography.titleSmall)
                val styles = listOf(
                    stringResource(R.string.settings_more_words),
                    stringResource(R.string.settings_balanced),
                    stringResource(R.string.settings_fewer_surer),
                )
                SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                    styles.forEachIndexed { i, label ->
                        SegmentedButton(
                            selected = settings.caution == i,
                            onClick = { onUpdate { it.copy(caution = i) } },
                            shape = SegmentedButtonDefaults.itemShape(i, styles.size),
                        ) { Text(label, maxLines = 1) }
                    }
                }
                Text(
                    stringResource(R.string.settings_more_words_shows_more_of),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
        Row(
            Modifier.fillMaxWidth().clickable(role = Role.Switch) { onUpdate { it.copy(showLandmarks = !it.showLandmarks) } }.padding(vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text(stringResource(R.string.settings_highlight_tracked_hands), style = MaterialTheme.typography.bodyLarge)
                Text(stringResource(R.string.settings_draws_the_hands_the_app), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Spacer(Modifier.width(12.dp))
            Switch(checked = settings.showLandmarks, onCheckedChange = null)
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun TypeReplySheet(
    demos: Map<String, Demo>,
    synonyms: Map<String, List<String>>,
    onSend: (text: String, openInSign: Boolean) -> Unit,
) {
    var draft by remember { mutableStateOf("") }
    val previewTokens = remember(draft, demos, synonyms) { SignTranslator.translate(draft, demos.keys, synonyms) }
    val signCount = remember(previewTokens, demos) { previewTokens.count { it.gloss != null && it.gloss in demos } }

    Column(
        Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp)
            .padding(bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(stringResource(R.string.talk_type_sheet_title), style = MaterialTheme.typography.titleLarge)
        Text(
            stringResource(R.string.talk_type_sheet_subtitle),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        OutlinedTextField(
            value = draft,
            onValueChange = { draft = it.take(180) },
            placeholder = { Text(stringResource(R.string.talk_type_placeholder)) },
            singleLine = false,
            maxLines = 3,
            shape = RoundedCornerShape(16.dp),
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
            keyboardActions = KeyboardActions(onSend = { if (draft.isNotBlank()) onSend(draft, false) }),
            modifier = Modifier.fillMaxWidth(),
        )

        if (draft.isNotBlank()) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Button(
                    onClick = { onSend(draft, false) },
                    modifier = Modifier.weight(1f).heightIn(min = 48.dp),
                ) {
                    Icon(Icons.AutoMirrored.Rounded.Send, null, Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text(stringResource(R.string.talk_send_text), maxLines = 1)
                }
                if (signCount > 0) {
                    OutlinedButton(
                        onClick = { onSend(draft, true) },
                        modifier = Modifier.weight(1f).heightIn(min = 48.dp),
                    ) {
                        Icon(Icons.Rounded.Translate, null, Modifier.size(18.dp))
                        Spacer(Modifier.width(6.dp))
                        Text(stringResource(R.string.talk_show_in_sign_n, signCount), maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                }
            }
        }

        Text(stringResource(R.string.talk_quick_phrases), style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            SignTranslator.QUICK_REPLIES.forEach { phrase ->
                AssistChip(
                    onClick = { onSend(phrase, false) },
                    label = { Text(phrase) },
                )
            }
        }
    }
}

/** The conversation under way, pulled up over the camera: copy, share, start a typed reply, or open History. */
@Composable
private fun ConversationSheet(entries: List<TranscriptEntry>, onTypeReply: () -> Unit, onAll: () -> Unit) {
    val context = LocalContext.current
    val current = remember(entries) { Conversations.current(entries, System.currentTimeMillis()) }
    Column(Modifier.fillMaxWidth().fillMaxHeight(0.75f)) {
        Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(stringResource(R.string.talk_this_conversation), Modifier.weight(1f), style = MaterialTheme.typography.titleLarge)
            val text = conversationText(current, LocalCaption.current,
                stringResource(R.string.history_line_signed), stringResource(R.string.history_line_said))
            IconButton(onClick = { copyText(context, text) }, enabled = current.isNotEmpty()) { Icon(Icons.Rounded.ContentCopy, stringResource(R.string.conversation_copy)) }
            IconButton(onClick = { shareText(context, text) }, enabled = current.isNotEmpty()) { Icon(Icons.Rounded.Share, stringResource(R.string.conversation_share)) }
        }
        if (current.isEmpty()) {
            Column(
                Modifier.weight(1f).fillMaxWidth().padding(horizontal = 28.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center,
            ) {
                ConversationArt()
                Spacer(Modifier.height(16.dp))
                Text(stringResource(R.string.talk_no_messages_yet), style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(6.dp))
                Text(
                    stringResource(R.string.talk_signed_words_and_spoken_replies),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                )
                Spacer(Modifier.height(16.dp))
                OutlinedButton(onClick = onTypeReply) {
                    Icon(Icons.Rounded.Keyboard, null, Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text(stringResource(R.string.talk_type_or_sign_reply))
                }
            }
        } else {
            ConversationBubbles(current, rememberLazyListState(), Modifier.weight(1f))
        }
        TextButton(onClick = onAll, modifier = Modifier.align(Alignment.CenterHorizontally).padding(bottom = 12.dp)) {
            Text(stringResource(R.string.talk_all_conversations))
        }
    }
}

/**
 * Full-screen reply for the signer: shows large legible text AND (when matched in demos.bin) animated real-signer
 * ASL playback! Rotate turns the whole display 180° for across-the-table conversations.
 */
@Composable
private fun ReplyForSigner(
    text: String,
    tokens: List<SignToken>,
    demos: Map<String, Demo>,
    startInSign: Boolean,
    onClose: () -> Unit,
) {
    val hasSigns = remember(tokens, demos) { tokens.any { it.gloss != null && it.gloss in demos } }
    var flipped by remember { mutableStateOf(false) }
    var showSigns by remember { mutableStateOf(startInSign && hasSigns) }

    Dialog(onDismissRequest = onClose, properties = DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false)) {
        Box(Modifier.fillMaxSize().background(Palette.ReplyDeep).statusBarsPadding().navigationBarsPadding()) {
            Column(
                Modifier
                    .fillMaxSize()
                    .padding(bottom = 86.dp)
                    .rotate(if (flipped) 180f else 0f)
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 24.dp, vertical = 20.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text(
                    text,
                    style = if (showSigns) ReplyStyle.copy(fontSize = 30.sp, lineHeight = 38.sp) else ReplyStyle.copy(fontSize = 44.sp, lineHeight = 54.sp),
                    color = Palette.Reply,
                    modifier = Modifier.fillMaxWidth(),
                )
                if (showSigns && hasSigns) {
                    Spacer(Modifier.height(18.dp))
                    SignSequencePlayer(
                        tokens = tokens,
                        demos = demos,
                        figureHeight = 260.dp,
                        darkContainer = Palette.Onyx.copy(alpha = 0.72f),
                    )
                }
            }
            Row(
                Modifier.align(Alignment.BottomCenter).fillMaxWidth().padding(horizontal = 16.dp, vertical = 14.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = { flipped = !flipped }, modifier = Modifier.heightIn(min = 48.dp)) {
                        Icon(Icons.Rounded.ScreenRotation, null, tint = Palette.Reply)
                        Spacer(Modifier.width(6.dp))
                        Text(stringResource(R.string.talk_rotate), color = Palette.Reply)
                    }
                    if (hasSigns) {
                        Row(
                            Modifier
                                .heightIn(min = 48.dp)
                                .clip(RoundedCornerShape(50))
                                .background(if (showSigns) Palette.LightAzure else Color.Black.copy(alpha = 0.25f))
                                .border(1.dp, if (showSigns) Palette.LightAzure else Palette.Reply.copy(alpha = 0.65f), RoundedCornerShape(50))
                                .clickable(role = Role.Button) { showSigns = !showSigns }
                                .padding(horizontal = 16.dp, vertical = 10.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Icon(
                                Icons.Rounded.Translate,
                                null,
                                Modifier.size(18.dp),
                                tint = if (showSigns) Palette.CyanDeep else Palette.Reply,
                            )
                            Spacer(Modifier.width(6.dp))
                            Text(
                                stringResource(R.string.talk_sign_toggle),
                                style = MaterialTheme.typography.labelLarge,
                                fontWeight = FontWeight.SemiBold,
                                color = if (showSigns) Palette.CyanDeep else Palette.Reply,
                            )
                        }
                    }
                }
                Button(onClick = onClose, modifier = Modifier.heightIn(min = 48.dp)) { Text(stringResource(R.string.talk_done)) }
            }
        }
    }
}

/** The status pill, worded for who is in front of the camera: someone else (back camera) or the person holding it. */
@Composable
private fun status(state: LiveState, back: Boolean): Pair<String, Color> = when {
    state.error != null -> stringResource(R.string.talk_hand_tracking_unavailable) to Palette.Red
    !state.ready -> stringResource(R.string.talk_loading) to Palette.TextMuted
    state.paused -> stringResource(R.string.talk_paused) to Palette.TextMuted
    state.recording != null -> stringResource(R.string.talk_recording) to Palette.Red
    state.signing -> stringResource(R.string.talk_signing) to Palette.Green
    state.handsVisible -> stringResource(R.string.talk_watching) to Palette.Green
    state.bodyVisible -> (if (back) stringResource(R.string.talk_their_hands_aren_t_in) else stringResource(R.string.talk_show_your_hands)) to Palette.Amber
    else -> (if (back) stringResource(R.string.talk_ask_them_to_step_back) else stringResource(R.string.talk_step_back)) to Palette.Amber
}

@Composable
internal fun CameraPreview(backCamera: Boolean, analyzer: (ImageProxy) -> Unit, modifier: Modifier) {
    val context = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    val executor = remember { Executors.newSingleThreadExecutor() }
    DisposableEffect(Unit) { onDispose { executor.shutdown() } }
    LocalConfiguration.current
    val rotation = LocalView.current.display?.rotation ?: android.view.Surface.ROTATION_0

    DisposableEffect(backCamera, owner, rotation) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val provider = future.get()
            val fourThree = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY).build()
            val analysisSelector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(ResolutionStrategy(Size(480, 360), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build()
            val preview = Preview.Builder().setResolutionSelector(fourThree).setTargetRotation(rotation).build()
            preview.setSurfaceProvider(previewView.surfaceProvider)
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(analysisSelector)
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
    androidx.compose.ui.viewinterop.AndroidView(factory = { previewView }, modifier = modifier.clipToBounds())
}

private val PALM_INDICES = intArrayOf(0, 5, 9, 13, 17)
private val TIP_INDICES = intArrayOf(4, 8, 12, 16, 20)

@Composable
internal fun HandOverlay(overlayFlow: StateFlow<OverlayState>, mirror: Boolean, skeleton: Boolean, modifier: Modifier) {
    val overlay by overlayFlow.collectAsStateWithLifecycle()
    val primary = MaterialTheme.colorScheme.primary
    Canvas(modifier) {
        if (overlay.imageWidth == 0 || overlay.imageHeight == 0) return@Canvas
        val scale = maxOf(size.width / overlay.imageWidth, size.height / overlay.imageHeight)
        val w = overlay.imageWidth * scale
        val h = overlay.imageHeight * scale
        val left = (size.width - w) / 2
        val top = (size.height - h) / 2
        val minReach = 12.dp.toPx()
        val tipRadius = 3.5.dp.toPx()
        val boneWidth = 2.dp.toPx()
        val jointRadius = 2.5.dp.toPx()
        for (slot in overlay.hands.indices) {
            val hand = overlay.hands[slot] ?: continue
            fun px(k: Int): Float {
                val x = hand[k * 3]
                return left + (if (mirror) 1f - x else x) * w
            }
            fun py(k: Int): Float = top + hand[k * 3 + 1] * h
            val colour = if (slot == 1) primary else Palette.FirstHand
            var cx = 0f
            var cy = 0f
            for (k in PALM_INDICES) { cx += px(k); cy += py(k) }
            cx /= PALM_INDICES.size
            cy /= PALM_INDICES.size
            var maxDistSq = minReach * minReach
            for (k in 0 until 21) {
                val dx = px(k) - cx
                val dy = py(k) - cy
                val dSq = dx * dx + dy * dy
                if (dSq > maxDistSq) maxDistSq = dSq
            }
            val centre = Offset(cx, cy)
            val radius = kotlin.math.sqrt(maxDistSq) * 1.35f
            drawCircle(
                Brush.radialGradient(
                    0f to colour.copy(alpha = 0.40f), 0.55f to colour.copy(alpha = 0.16f), 1f to Color.Transparent,
                    center = centre, radius = radius,
                ),
                radius = radius, center = centre,
            )
            for (k in TIP_INDICES) drawCircle(colour, radius = tipRadius, center = Offset(px(k), py(k)))
            if (skeleton) {
                val boneColour = colour.copy(alpha = 0.85f)
                for ((a, b) in HAND_EDGES) drawLine(boneColour, Offset(px(a), py(a)), Offset(px(b), py(b)), strokeWidth = boneWidth, cap = StrokeCap.Round)
                for (k in 0 until 21) drawCircle(Color.White, radius = jointRadius, center = Offset(px(k), py(k)))
            }
        }
    }
}

@Composable
private fun Details(state: LiveState, overlayFlow: StateFlow<OverlayState>, voice: VoiceState?) {
    val overlay by overlayFlow.collectAsStateWithLifecycle()
    Column(
        Modifier.widthIn(max = 320.dp).glass(RoundedCornerShape(16.dp)).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            "%.0f fps · tracking %.0f ms · model %.1f ms · %d %s tracker%s".format(overlay.fps, overlay.landmarksMs, overlay.modelMs, state.lanes, if (state.gpu) stringResource(R.string.talk_gpu) else stringResource(R.string.talk_cpu), if (state.lanes == 1) "" else "s"),
            style = MaterialTheme.typography.labelMedium, color = Palette.TextMuted,
        )
        if (voice != null && voice.spoken > 0) {
            Text(stringResource(R.string.talk_voice_0f_ms_after_the).format(voice.lastMs, voice.medianMs),
                style = MaterialTheme.typography.labelMedium, color = Palette.TextMuted)
        }
        state.top.forEachIndexed { i, (word, p) -> ConfidenceRow(word, p, highlight = i == 0) }
    }
}

/** The hearing person's reply banner: shows spoken or typed words, plus 1-tap "Show to signer" and "Show in Sign". */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ReplyArea(
    reply: ReplyState,
    signCount: Int,
    onDismiss: () -> Unit,
    onShowText: () -> Unit,
    onShowSign: () -> Unit,
) {
    AnimatedVisibility(
        visible = reply.listening || reply.last != null,
        enter = fadeIn() + slideInVertically { -it / 3 },
        exit = fadeOut() + slideOutVertically { -it / 3 },
    ) {
        val listening = reply.listening
        Column(
            Modifier.fillMaxWidth().animateContentSize(spring(dampingRatio = 0.8f, stiffness = Spring.StiffnessMediumLow))
                .clip(RoundedCornerShape(24.dp)).background(Palette.ReplyDeep.copy(alpha = 0.95f))
                .border(1.dp, Palette.LightAzure.copy(alpha = 0.35f), RoundedCornerShape(24.dp))
                .clickable(enabled = !listening, onClickLabel = stringResource(R.string.talk_dismiss_reply), onClick = onDismiss)
                .padding(horizontal = 20.dp, vertical = 14.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    if (listening) stringResource(R.string.talk_listening) else stringResource(R.string.talk_they_said),
                    style = MaterialTheme.typography.labelLarge,
                    color = Palette.LightAzure,
                    modifier = Modifier.weight(1f),
                )
                if (!listening) Icon(Icons.Rounded.Close, stringResource(R.string.video_dismiss), tint = Palette.Reply.copy(alpha = 0.75f), modifier = Modifier.size(18.dp))
            }
            Spacer(Modifier.height(4.dp))
            val text = if (listening) reply.partial.ifEmpty { stringResource(R.string.talk_speak_now) } else reply.last.orEmpty()
            Text(
                text,
                style = ReplyStyle,
                color = if (listening && reply.partial.isEmpty()) Palette.Reply.copy(alpha = 0.6f) else Palette.Reply,
                maxLines = 4,
                overflow = TextOverflow.Ellipsis,
            )
            if (!listening) {
                Spacer(Modifier.height(8.dp))
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Row(
                        Modifier
                            .clip(RoundedCornerShape(50))
                            .background(Color.Black.copy(alpha = 0.28f))
                            .clickable(role = Role.Button, onClick = onShowText)
                            .padding(horizontal = 12.dp, vertical = 7.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(Icons.Rounded.Fullscreen, null, Modifier.size(16.dp), tint = Palette.Reply)
                        Spacer(Modifier.width(6.dp))
                        Text(stringResource(R.string.talk_show_to_signer), style = MaterialTheme.typography.labelLarge, color = Palette.Reply)
                    }
                    if (signCount > 0) {
                        Row(
                            Modifier
                                .clip(RoundedCornerShape(50))
                                .background(Palette.LightAzure)
                                .clickable(role = Role.Button, onClick = onShowSign)
                                .padding(horizontal = 12.dp, vertical = 7.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Icon(Icons.Rounded.Translate, null, Modifier.size(16.dp), tint = Palette.CyanDeep)
                            Spacer(Modifier.width(6.dp))
                            Text(
                                pluralStringResource(R.plurals.talk_watch_in_sign, signCount, signCount),
                                style = MaterialTheme.typography.labelLarge,
                                fontWeight = FontWeight.Bold,
                                color = Palette.CyanDeep,
                            )
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalFoundationApi::class, ExperimentalLayoutApi::class)
@Composable
private fun CaptionCard(
    state: LiveState,
    back: Boolean,
    atRest: Boolean,
    onSpeakSentence: () -> Unit,
    onRemoveLast: () -> Unit,
    onClearAll: () -> Unit,
    onCorrect: (String) -> Unit,
) {
    var reopen by remember { mutableIntStateOf(0) }
    val show = LocalCaption.current
    val smoothProgress by animateFloatAsState(
        targetValue = if (state.pending != null || (state.signing && state.pendingProgress > 0.05f)) state.pendingProgress else 0f,
        animationSpec = tween(Motion.SHORT),
        label = "lockIn",
    )

    Column(
        Modifier
            .fillMaxWidth()
            .animateContentSize(spring(dampingRatio = 0.82f, stiffness = Spring.StiffnessMediumLow))
            .glass(RoundedCornerShape(24.dp))
            .semantics { liveRegion = LiveRegionMode.Polite },
    ) {
        // Sleek 3 dp live lock-in hairline across the top edge while a continuous sign is forming.
        Box(Modifier.fillMaxWidth().height(3.dp).background(Color.Transparent)) {
            if (smoothProgress > 0.01f) {
                Box(
                    Modifier
                        .fillMaxWidth(smoothProgress.coerceIn(0f, 1f))
                        .fillMaxHeight()
                        .background(MaterialTheme.colorScheme.primary),
                )
            }
        }
        Column(Modifier.fillMaxWidth().padding(start = 18.dp, end = 14.dp, top = 12.dp, bottom = 14.dp)) {
            if (state.caption.isEmpty() && state.pending != null) {
                Text(show(state.pending) + "…", Modifier.padding(top = 2.dp), style = CaptionStyle, color = Palette.LightAzure.copy(alpha = 0.85f), maxLines = 1)
                return@Column
            }
            if (state.caption.isEmpty()) {
                Text(
                    when {
                        state.paused -> stringResource(R.string.talk_paused)
                        !state.bodyVisible && back -> stringResource(R.string.talk_hold_the_phone_so_their)
                        !state.bodyVisible -> stringResource(R.string.talk_keep_your_head_shoulders_and)
                        atRest && back -> stringResource(R.string.talk_each_word_appears_when_their)
                        atRest -> stringResource(R.string.talk_sign_then_lower_your_hands)
                        back -> stringResource(R.string.talk_captions_appear_here_as_they)
                        else -> stringResource(R.string.talk_sign_one_word_after_another)
                    },
                    Modifier.padding(top = 2.dp),
                    style = MaterialTheme.typography.titleMedium,
                    color = Palette.Text.copy(alpha = 0.84f),
                )
                return@Column
            }
            // Smooth auto-scrolling live caption ribbon: newest committed word + live tentative word never get clipped or squeezed.
            val older = state.caption.dropLast(1).takeLast(8).joinToString(" ", transform = show)
            val ribbonScroll = rememberScrollState()
            LaunchedEffect(ribbonScroll.maxValue) {
                ribbonScroll.animateScrollTo(ribbonScroll.maxValue)
            }
            Row(
                Modifier
                    .fillMaxWidth()
                    .graphicsLayer(compositingStrategy = CompositingStrategy.Offscreen)
                    .drawWithContent {
                        drawContent()
                        if (ribbonScroll.value > 0) {
                            val fadePx = 28.dp.toPx().coerceAtMost(size.width * 0.25f)
                            drawRect(
                                brush = Brush.horizontalGradient(
                                    colors = listOf(Color.Transparent, Color.Black),
                                    startX = 0f,
                                    endX = fadePx,
                                ),
                                size = androidx.compose.ui.geometry.Size(fadePx, size.height),
                                blendMode = BlendMode.DstIn,
                            )
                        }
                    }
                    .horizontalScroll(ribbonScroll),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                if (older.isNotEmpty()) {
                    Text(
                        older,
                        style = CaptionStyle,
                        color = Palette.Text.copy(alpha = 0.72f),
                        maxLines = 1,
                        softWrap = false,
                    )
                    Spacer(Modifier.width(10.dp))
                }
                val canOffer = state.lastAlternatives.isNotEmpty()
                AnimatedContent(
                    targetState = state.caption.last() to state.lastWordAt,
                    transitionSpec = {
                        (slideInVertically(Motion.wordSpring) { it / 2 } + fadeIn(tween(Motion.SHORT))) togetherWith fadeOut(tween(Motion.SHORT))
                    },
                    label = "word",
                    modifier = if (canOffer) Modifier.clip(RoundedCornerShape(10.dp)).clickable(onClickLabel = stringResource(R.string.talk_show_other_signs_it_could)) { reopen++ } else Modifier,
                ) { (word, _) ->
                    Text(show(word), style = CaptionStyle, color = MaterialTheme.colorScheme.primary, maxLines = 1, softWrap = false)
                }
                state.pending?.let {
                    Spacer(Modifier.width(10.dp))
                    Text(
                        "${show(it)}…",
                        style = CaptionStyle,
                        color = Palette.TextMuted,
                        maxLines = 1,
                        softWrap = false,
                    )
                }
            }

            val sub = when {
                state.lastTaught -> stringResource(R.string.talk_taught_on_this_phone)
                state.lastAlso.isNotEmpty() -> stringResource(R.string.talk_same_sign, state.lastAlso.joinToString(" · ", transform = show))
                else -> null
            }
            if (sub != null) {
                Spacer(Modifier.height(2.dp))
                Text(sub, style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.secondary, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }

            var offer by remember { mutableStateOf(true) }
            LaunchedEffect(state.lastWordAt, reopen) { offer = true; delay(ALTERNATIVES_SHOWN_MS); offer = false }
            AnimatedVisibility(offer && state.lastAlternatives.isNotEmpty(), enter = fadeIn(), exit = fadeOut()) {
                Row(Modifier.padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(stringResource(R.string.talk_not_right), style = MaterialTheme.typography.labelLarge, color = Palette.TextMuted)
                    for (w in state.lastAlternatives) {
                        Box(
                            Modifier
                                .weight(1f, fill = false)
                                .clip(RoundedCornerShape(50))
                                .background(Palette.PanelHigh)
                                .border(1.dp, Palette.Hairline, RoundedCornerShape(50))
                                .clickable(onClickLabel = stringResource(R.string.talk_change_the_last_word_to, show(w))) { onCorrect(w) }
                                .heightIn(min = 40.dp)
                                .padding(horizontal = 14.dp),
                            contentAlignment = Alignment.Center,
                        ) {
                            Text(show(w), style = MaterialTheme.typography.titleSmall, color = Palette.Text, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        }
                    }
                }
            }

            // Compact, clear caption action row: Speak phrase, Undo last word, Clear all (wraps cleanly at large font sizes).
            Spacer(Modifier.height(8.dp))
            val removeLabel = stringResource(R.string.caption_remove_last)
            val clearLabel = stringResource(R.string.caption_clear_all)
            FlowRow(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                CaptionActionChip(
                    icon = Icons.Rounded.RecordVoiceOver,
                    label = stringResource(R.string.talk_speak_short),
                    contentDesc = stringResource(R.string.talk_speak_sentence),
                    tint = Palette.LightAzure,
                    onClick = onSpeakSentence,
                )
                Box(
                    Modifier
                        .clip(RoundedCornerShape(50))
                        .background(Palette.PanelHigh.copy(alpha = 0.75f))
                        .border(1.dp, Palette.Hairline, RoundedCornerShape(50))
                        .combinedClickable(onClickLabel = removeLabel, onLongClickLabel = clearLabel, onLongClick = onClearAll, onClick = onRemoveLast)
                        .heightIn(min = 34.dp)
                        .padding(horizontal = 11.dp, vertical = 5.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                        Icon(Icons.AutoMirrored.Rounded.Backspace, null, Modifier.size(15.dp), tint = Palette.TextMuted)
                        Text(stringResource(R.string.action_undo), style = MaterialTheme.typography.labelMedium, color = Palette.TextMuted, maxLines = 1)
                    }
                }
                if (state.caption.size >= 2) {
                    CaptionActionChip(
                        icon = Icons.Rounded.DeleteSweep,
                        label = stringResource(R.string.dict_clear),
                        contentDesc = clearLabel,
                        tint = Palette.TextMuted,
                        onClick = onClearAll,
                    )
                }
            }
        }
    }
}

private const val ALTERNATIVES_SHOWN_MS = 9_000L
private const val UNDO_SHOWN_MS = 5_000L

@Composable
private fun CaptionActionChip(icon: ImageVector, label: String, contentDesc: String, tint: Color, onClick: () -> Unit) {
    Row(
        Modifier
            .clip(RoundedCornerShape(50))
            .background(Palette.PanelHigh.copy(alpha = 0.75f))
            .border(1.dp, Palette.Hairline, RoundedCornerShape(50))
            .clickable(role = Role.Button, onClickLabel = contentDesc, onClick = onClick)
            .heightIn(min = 34.dp)
            .padding(horizontal = 11.dp, vertical = 5.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        Icon(icon, null, Modifier.size(15.dp), tint = tint)
        Text(label, style = MaterialTheme.typography.labelMedium, color = tint)
    }
}

@Composable
private fun DockControl(
    icon: ImageVector,
    label: String,
    contentDesc: String,
    active: Boolean,
    toggle: Boolean = false,
    onClick: () -> Unit,
) {
    val onOff = stringResource(if (active) R.string.talk_on else R.string.talk_off)
    val press = rememberPress(0.9f)
    val fill by animateColorAsState(if (active) MaterialTheme.colorScheme.primaryContainer else Palette.PanelHigh, tween(Motion.MEDIUM), label = "dock")
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.widthIn(min = 60.dp)) {
        FilledIconButton(
            onClick = onClick,
            interactionSource = press.interaction,
            modifier = press.scale.size(48.dp).semantics {
                contentDescription = contentDesc
                if (toggle) stateDescription = onOff
            },
            colors = IconButtonDefaults.filledIconButtonColors(
                containerColor = fill,
                contentColor = if (active) MaterialTheme.colorScheme.onPrimaryContainer else Palette.Text,
            ),
        ) { Icon(icon, contentDescription = null, modifier = Modifier.size(22.dp)) }
        Spacer(Modifier.height(4.dp))
        Text(
            label,
            style = MaterialTheme.typography.labelSmall,
            color = if (active) Palette.LightAzure else Palette.Text,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

/** The main control: tap to reply by speaking (or seamlessly opens offline Type & Quick-Reply if STT is unavailable). */
@Composable
private fun MicControl(reply: ReplyState, onClick: () -> Unit) {
    val press = rememberPress(0.9f)
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.widthIn(min = 68.dp)) {
        Box(contentAlignment = Alignment.Center) {
            val pulseScale = if (reply.listening) {
                val anim = rememberInfiniteTransition(label = "mic")
                val pulse by anim.animateFloat(
                    initialValue = 1f, targetValue = 1.08f,
                    animationSpec = infiniteRepeatable(tween(700), RepeatMode.Reverse), label = "pulse",
                )
                val ring by anim.animateFloat(
                    0f, 1f, infiniteRepeatable(tween(1400, easing = LinearEasing), RepeatMode.Restart), label = "ring",
                )
                Canvas(Modifier.size(62.dp)) {
                    for (k in 0 until 2) {
                        val t = (ring + k / 2f) % 1f
                        drawCircle(Palette.LightAzure.copy(alpha = 0.45f * (1f - t)), radius = size.minDimension / 2 * (1f + 0.55f * t),
                            style = Stroke(2.dp.toPx()))
                    }
                }
                pulse
            } else 1f
            FilledIconButton(
                onClick = onClick, interactionSource = press.interaction,
                modifier = press.scale.size(60.dp).scale(pulseScale)
                    .description(stringResource(if (reply.listening) R.string.talk_stop_listening else R.string.talk_reply_by_speaking)),
                colors = IconButtonDefaults.filledIconButtonColors(
                    containerColor = if (reply.listening) Palette.Reply else MaterialTheme.colorScheme.primary,
                    contentColor = if (reply.listening) Palette.ReplyDeep else MaterialTheme.colorScheme.onPrimary,
                ),
            ) { Icon(if (reply.listening) Icons.Rounded.Stop else Icons.Rounded.Mic, contentDescription = null, modifier = Modifier.size(28.dp)) }
        }
        Spacer(Modifier.height(4.dp))
        Text(
            if (reply.listening) stringResource(R.string.talk_stop) else stringResource(R.string.talk_reply),
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.SemiBold,
            color = Palette.Text,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

@Composable
internal fun GlassIconButton(icon: ImageVector, label: String, size: Dp = 48.dp, onClick: () -> Unit) {
    FilledIconButton(
        onClick = onClick, modifier = Modifier.size(size).glass(CircleShape),
        colors = IconButtonDefaults.filledIconButtonColors(containerColor = Color.Transparent, contentColor = Palette.Text),
    ) { Icon(icon, contentDescription = label) }
}
