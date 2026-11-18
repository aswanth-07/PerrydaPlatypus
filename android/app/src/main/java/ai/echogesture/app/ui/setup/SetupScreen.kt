package ai.echogesture.app.ui.setup

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
            Text("Step ${step + 1} of $STEPS", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant,
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
            if (step > 0) TextButton(onClick = { step-- }) { Text("Back") }
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
                        step == STEPS - 1 -> "Start"
                        step == 1 && !camera.granted -> "Skip for now"
                        else -> "Continue"
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
    StepTitle("How will you use EchoGesture?", "You can change this later on the Talk tab or in Settings.")
    UseChoice(
        Icons.Rounded.Videocam, "I'm talking with someone who signs",
        "Hold the phone with its back camera toward them. Their signs appear as captions, and you can reply by speaking.",
        selected = back,
    ) { onChoose(true) }
    Spacer(Modifier.height(12.dp))
    UseChoice(
        Icons.Rounded.FrontHand, "I sign, and the phone faces me",
        "Prop the phone up facing you. Your signs appear as captions and can be spoken aloud.",
        selected = !back,
    ) { onChoose(false) }
}

@Composable
private fun UseChoice(icon: ImageVector, title: String, body: String, selected: Boolean, onClick: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp))
            .background(if (selected) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceContainer)
            .border(2.dp, if (selected) MaterialTheme.colorScheme.primary else Color.Transparent, RoundedCornerShape(20.dp))
            .selectable(selected = selected, role = Role.RadioButton, onClick = onClick)
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
        if (back) "Frame the signer" else "Frame yourself",
        if (back) "Hands are found from the body, so the app needs to see their head, shoulders and both hands."
        else "Hands are found from the body, so the app needs to see your head, shoulders and both hands.",
    )
    if (!granted) {
        Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(MaterialTheme.colorScheme.surfaceContainer).padding(20.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Rounded.PhotoCamera, null, tint = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.width(12.dp))
                Text("EchoGesture needs the camera to see signs.", style = MaterialTheme.typography.titleMedium)
            }
            Spacer(Modifier.height(8.dp))
            Text("Video is processed on this phone and never uploaded or stored.", style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.height(16.dp))
            Button(onClick = request, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                Text(if (blocked) "Open settings" else "Allow camera")
            }
            if (blocked) {
                Spacer(Modifier.height(8.dp))
                Text("Camera access is turned off for EchoGesture. In settings, open Permissions, Camera, and choose Allow.",
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
        !live.ready -> "Loading on-device models…" to Palette.TextMuted
        live.bodyVisible && live.handsVisible -> (if (back) "Their hands are in view. You're set." else "Your hands are in view. You're set.") to Palette.Green
        live.bodyVisible -> (if (back) "Good distance. Their hands are tracked once they're raised." else "Good distance. Raise your hands to sign.") to Palette.Green
        else -> (if (back) "Step back until their head and shoulders fit the outline." else "Move back until your head and shoulders fit the outline.") to Palette.Amber
    }
    Box(
        Modifier.fillMaxWidth().widthIn(max = 420.dp).aspectRatio(3f / 4f).clip(RoundedCornerShape(24.dp)).background(Color.Black)
            .semantics { contentDescription = "Camera preview. $text" },
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
    StepTitle("Voice and replies", "Both use what is installed on this phone, so they work without a connection.")

    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(MaterialTheme.colorScheme.surfaceContainer).padding(18.dp)) {
        Row(
            Modifier.fillMaxWidth().toggleable(value = settings.speak, role = Role.Switch) { v -> vm.update { it.copy(speak = v) } },
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text("Speak signed words aloud", style = MaterialTheme.typography.titleMedium)
                Text(
                    when {
                        voice.error != null -> voice.error!!
                        !voice.ready -> "Starting the phone's text-to-speech."
                        voice.onDevice -> "An installed voice is ready."
                        else -> "No installed English voice was found. Add one to keep speech offline."
                    },
                    style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Spacer(Modifier.width(12.dp))
            Switch(checked = settings.speak, onCheckedChange = null)
        }
        if (voice.ready && !voice.onDevice || voice.error != null) {
            OutlinedButton(onClick = { context.openSystem("com.android.settings.TTS_SETTINGS") }, modifier = Modifier.padding(top = 10.dp)) {
                Text("Text-to-speech settings")
            }
        }
    }
    Spacer(Modifier.height(12.dp))
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(MaterialTheme.colorScheme.surfaceContainer).padding(18.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Rounded.RecordVoiceOver, null, tint = if (reply.available) Palette.Green else Palette.Amber)
            Spacer(Modifier.width(12.dp))
            Text("Spoken replies", style = MaterialTheme.typography.titleMedium)
        }
        Spacer(Modifier.height(4.dp))
        Text(
            if (reply.available) "Ready. Tap Reply on the Talk tab and speak; your words appear large for the signer to read."
            else reply.note ?: "Checking for on-device speech recognition.",
            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        if (!reply.available && reply.note != null) {
            OutlinedButton(onClick = { context.openSystem(Settings.ACTION_VOICE_INPUT_SETTINGS) }, modifier = Modifier.padding(top = 10.dp)) {
                Text("Speech recognition settings")
            }
        }
    }
    Spacer(Modifier.height(16.dp))
    Row(verticalAlignment = Alignment.CenterVertically) {
        Icon(Icons.Rounded.Lock, null, Modifier.size(16.dp), tint = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.width(8.dp))
        Text("EchoGesture has no internet access. Video, audio and keypoints stay on this phone.",
            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

private fun Context.openSystem(action: String) {
    runCatching { startActivity(Intent(action).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) }
        .recoverCatching { startActivity(Intent(Settings.ACTION_SETTINGS).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) }
}
