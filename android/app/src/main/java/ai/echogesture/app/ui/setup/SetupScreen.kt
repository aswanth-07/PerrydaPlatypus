package ai.echogesture.app.ui.setup

import ai.echogesture.app.ui.theme.Motion
import androidx.compose.animation.core.tween
import androidx.compose.animation.animateColorAsState
import androidx.compose.material3.ripple
import ai.echogesture.app.ui.components.rememberPress
import ai.echogesture.app.R
import androidx.compose.ui.res.stringResource
import android.Manifest
import android.content.Context
import android.content.Intent
import android.provider.Settings
import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.FrontHand
import androidx.compose.material.icons.rounded.Lock
import androidx.compose.material.icons.rounded.PhotoCamera
import androidx.compose.material.icons.rounded.RecordVoiceOver
import androidx.compose.material.icons.rounded.Videocam
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.ui.components.FramingGuide
import ai.echogesture.app.ui.components.description
import ai.echogesture.app.ui.components.rememberPermission
import ai.echogesture.app.ui.live.CameraPreview
import ai.echogesture.app.ui.theme.Palette

private const val STEPS = 3

/**
 * First launch (and Settings > Set up again): who holds the phone, how far back the signer should be, and whether
 * the phone can speak and listen offline. Finishing it opens the Talk tab.
 */
@Composable
fun SetupScreen(vm: MainViewModel, onDone: () -> Unit) {
    val settings by vm.settings.collectAsStateWithLifecycle()
    var step by rememberSaveable { mutableIntStateOf(0) }
    var back by rememberSaveable { mutableStateOf(settings.backCamera) }
    val camera = rememberPermission(Manifest.permission.CAMERA)
    BackHandler(enabled = step > 0) { step-- }

    Column(Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding()) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 16.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(stringResource(R.string.setup_step_of, step + 1, STEPS), style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.weight(1f))
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                repeat(STEPS) { i ->
                    Box(Modifier.size(width = if (i == step) 24.dp else 8.dp, height = 8.dp).clip(CircleShape)
                        .background(if (i <= step) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceContainerHighest))
                }
            }
        }
        AnimatedContent(step, Modifier.weight(1f), transitionSpec = { fadeIn() togetherWith fadeOut() }, label = "step") { s ->
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 24.dp)) {
                when (s) {
                    0 -> UseStep(back) { back = it }
                    1 -> FramingStep(vm, back, camera.granted, camera.blocked, camera::request)
                    else -> VoiceStep(vm)
                }
            }
        }
        Row(Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 16.dp), verticalAlignment = Alignment.CenterVertically) {
            if (step > 0) TextButton(onClick = { step-- }) { Text(stringResource(R.string.common_back)) }
            Spacer(Modifier.weight(1f))
            Button(
                onClick = {
                    when (step) {
                        0 -> { vm.update { it.copy(backCamera = back) }; step++ }
                        STEPS - 1 -> { vm.update { it.copy(onboarded = true, backCamera = back) }; onDone() }
                        else -> step++
                    }
                },
                modifier = Modifier.heightIn(min = 48.dp),
            ) {
                Text(
                    when {
                        step == STEPS - 1 -> stringResource(R.string.setup_start)
                        step == 1 && !camera.granted -> stringResource(R.string.setup_skip_for_now)
                        else -> stringResource(R.string.setup_continue)
                    }
                )
            }
        }
    }
}

@Composable
private fun StepTitle(title: String, body: String) {
    Text(title, Modifier.semantics { heading() }, style = MaterialTheme.typography.headlineMedium)
    Spacer(Modifier.height(8.dp))
    Text(body, style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
    Spacer(Modifier.height(24.dp))
}

@Composable
private fun UseStep(back: Boolean, onChoose: (Boolean) -> Unit) {
    StepTitle(stringResource(R.string.setup_how_will_you_use_echogesture), stringResource(R.string.setup_you_can_change_this_later))
    UseChoice(
        Icons.Rounded.Videocam, stringResource(R.string.setup_i_m_talking_with_someone),
        stringResource(R.string.setup_hold_the_phone_with_its),
        selected = back,
    ) { onChoose(true) }
    Spacer(Modifier.height(12.dp))
    UseChoice(
        Icons.Rounded.FrontHand, stringResource(R.string.setup_i_sign_and_the_phone),
        stringResource(R.string.setup_prop_the_phone_up_facing),
        selected = !back,
    ) { onChoose(false) }
}

@Composable
private fun UseChoice(icon: ImageVector, title: String, body: String, selected: Boolean, onClick: () -> Unit) {
    val press = rememberPress()
    val fill by animateColorAsState(
        if (selected) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceContainer, tween(Motion.MEDIUM), label = "choice",
    )
    val edge by animateColorAsState(if (selected) MaterialTheme.colorScheme.primary else Color.Transparent, tween(Motion.MEDIUM), label = "edge")
    Row(
        press.scale.fillMaxWidth().clip(RoundedCornerShape(20.dp))
            .background(fill)
            .border(2.dp, edge, RoundedCornerShape(20.dp))
            .selectable(selected = selected, interactionSource = press.interaction, indication = ripple(), role = Role.RadioButton, onClick = onClick)
            .padding(18.dp),
        verticalAlignment = Alignment.Top,
    ) {
        Icon(icon, null, Modifier.size(28.dp), tint = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.width(14.dp))
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(4.dp))
            Text(body, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        if (selected) {
            Spacer(Modifier.width(8.dp))
            Icon(Icons.Rounded.CheckCircle, null, tint = MaterialTheme.colorScheme.primary)
        }
    }
}

@Composable
private fun FramingStep(vm: MainViewModel, back: Boolean, granted: Boolean, blocked: Boolean, request: () -> Unit) {
    StepTitle(
        if (back) stringResource(R.string.setup_frame_the_signer) else stringResource(R.string.setup_frame_yourself),
        if (back) stringResource(R.string.setup_hands_are_found_from_the)
        else stringResource(R.string.setup_hands_are_found_from_the_2),
    )
    if (!granted) {
        Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(MaterialTheme.colorScheme.surfaceContainer).padding(20.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Rounded.PhotoCamera, null, tint = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.width(12.dp))
                Text(stringResource(R.string.setup_echogesture_needs_the_camera_to), style = MaterialTheme.typography.titleMedium)
            }
            Spacer(Modifier.height(8.dp))
            Text(stringResource(R.string.setup_video_is_processed_on_this), style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.height(16.dp))
            Button(onClick = request, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                Text(if (blocked) stringResource(R.string.action_open_settings) else stringResource(R.string.gate_allow_camera))
            }
            if (blocked) {
                Spacer(Modifier.height(8.dp))
                Text(stringResource(R.string.setup_camera_access_is_turned_off),
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        return
    }
    val live by vm.live.collectAsStateWithLifecycle()
    DisposableEffect(Unit) {
        vm.engine.setPaused(false)
        vm.engine.acquireLive()
        onDispose { vm.engine.releaseLive() }
    }
    val (text, colour) = when {
        !live.ready -> stringResource(R.string.setup_loading_on_device_models) to Palette.TextMuted
        live.bodyVisible && live.handsVisible -> (if (back) stringResource(R.string.setup_their_hands_are_in_view) else stringResource(R.string.setup_your_hands_are_in_view)) to Palette.Green
        live.bodyVisible -> (if (back) stringResource(R.string.setup_good_distance_their_hands_are) else stringResource(R.string.setup_good_distance_raise_your_hands)) to Palette.Green
        else -> (if (back) stringResource(R.string.setup_step_back_until_their_head) else stringResource(R.string.setup_move_back_until_your_head)) to Palette.Amber
    }
    Box(
        Modifier.fillMaxWidth().widthIn(max = 420.dp).aspectRatio(3f / 4f).clip(RoundedCornerShape(24.dp)).background(Color.Black)
            .description(stringResource(R.string.setup_camera_preview, text)),
    ) {
        CameraPreview(backCamera = back, analyzer = vm.engine::analyze, modifier = Modifier.fillMaxSize())
        FramingGuide(ok = live.bodyVisible, modifier = Modifier.fillMaxSize().padding(24.dp))
    }
    Spacer(Modifier.height(14.dp))
    Text(text, style = MaterialTheme.typography.titleMedium, color = colour)
}

@Composable
private fun VoiceStep(vm: MainViewModel) {
    val settings by vm.settings.collectAsStateWithLifecycle()
    val voice by vm.engine.speaker.state.collectAsStateWithLifecycle()
    val reply by vm.engine.listener.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    StepTitle(stringResource(R.string.setup_voice_and_replies), stringResource(R.string.setup_both_use_what_is_installed))

    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(MaterialTheme.colorScheme.surfaceContainer).padding(18.dp)) {
        Row(
            Modifier.fillMaxWidth().toggleable(value = settings.speak, role = Role.Switch) { v -> vm.update { it.copy(speak = v) } },
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text(stringResource(R.string.settings_speak_signed_words_aloud), style = MaterialTheme.typography.titleMedium)
                Text(
                    when {
                        voice.error != null -> voice.error!!
                        !voice.ready -> stringResource(R.string.settings_starting_the_phone_s_text)
                        voice.onDevice -> stringResource(R.string.setup_an_installed_voice_is_ready)
                        else -> stringResource(R.string.setup_no_installed_english_voice_was)
                    },
                    style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Spacer(Modifier.width(12.dp))
            Switch(checked = settings.speak, onCheckedChange = null)
        }
        if (voice.ready && !voice.onDevice || voice.error != null) {
            OutlinedButton(onClick = { context.openSystem("com.android.settings.TTS_SETTINGS") }, modifier = Modifier.padding(top = 10.dp)) {
                Text(stringResource(R.string.setup_text_to_speech_settings))
            }
        }
    }
    Spacer(Modifier.height(12.dp))
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(MaterialTheme.colorScheme.surfaceContainer).padding(18.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Rounded.RecordVoiceOver, null, tint = if (reply.available) Palette.Green else Palette.Amber)
            Spacer(Modifier.width(12.dp))
            Text(stringResource(R.string.settings_spoken_replies), style = MaterialTheme.typography.titleMedium)
        }
        Spacer(Modifier.height(4.dp))
        Text(
            if (reply.available) stringResource(R.string.setup_ready_tap_reply_on_the)
            else reply.note ?: stringResource(R.string.settings_checking_for_on_device_speech),
            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        if (!reply.available && reply.note != null) {
            OutlinedButton(onClick = { context.openSystem(Settings.ACTION_VOICE_INPUT_SETTINGS) }, modifier = Modifier.padding(top = 10.dp)) {
                Text(stringResource(R.string.setup_speech_recognition_settings))
            }
        }
    }
    Spacer(Modifier.height(16.dp))
    Row(verticalAlignment = Alignment.CenterVertically) {
        Icon(Icons.Rounded.Lock, null, Modifier.size(16.dp), tint = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.width(8.dp))
        Text(stringResource(R.string.setup_echogesture_has_no_internet_access),
            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

private fun Context.openSystem(action: String) {
    runCatching { startActivity(Intent(action).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) }
        .recoverCatching { startActivity(Intent(Settings.ACTION_SETTINGS).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) }
}
