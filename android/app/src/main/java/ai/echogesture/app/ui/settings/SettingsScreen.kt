package ai.echogesture.app.ui.settings

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.ui.components.SectionTitle

@Composable
fun SettingsScreen(vm: MainViewModel) {
    val s by vm.settings.collectAsStateWithLifecycle()
    val live by vm.live.collectAsStateWithLifecycle()
    val voice by vm.engine.speaker.state.collectAsStateWithLifecycle()
    val reply by vm.engine.listener.state.collectAsStateWithLifecycle()
    val info = if (live.ready) vm.engine.info else null
    var advanced by rememberSaveable { mutableStateOf(false) }

    Column(Modifier.fillMaxSize().statusBarsPadding().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 12.dp)) {
        Text("Settings", style = MaterialTheme.typography.headlineMedium)

        SectionTitle("Captions")
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            listOf("Eager", "Balanced", "Careful").forEachIndexed { i, name ->
                SegmentedButton(selected = s.caution == i, onClick = { vm.update { it.copy(caution = i) } },
                    shape = SegmentedButtonDefaults.itemShape(i, 3), enabled = s.continuous) { Text(name, maxLines = 1) }
            }
        }
        Spacer(Modifier.height(8.dp))
        Text(
            if (s.continuous) "Each word appears as you sign it; keep your hands up between signs. Eager shows more words, some of them wrong; Careful shows fewer, surer ones."
            else "Captions are waiting for the hands to rest after each sign (see Advanced).",
            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        SectionTitle("Voice")
        SwitchSetting(
            "Speak signed words aloud",
            when {
                voice.error != null -> voice.error!!
                !voice.ready -> "Starting the phone's text-to-speech."
                voice.onDevice -> "Uses a voice installed on this phone, so it works offline."
                else -> "No installed English voice was found. Add one in the phone's text-to-speech settings to keep speech offline."
            },
            s.speak,
        ) { v -> vm.update { it.copy(speak = v) } }
        AnimatedVisibility(s.speak) {
            SliderSetting(
                "Speaking speed", "%.2f×".format(s.speechRate), "Faster keeps up with quick signing.",
                s.speechRate, 0.8f..1.8f, 9,
            ) { v -> vm.update { it.copy(speechRate = v) } }
        }
        InfoLine(
            "Spoken replies",
            if (reply.available) "Tap Reply on the Live tab and speak; the words appear large for the signer. Recognised on this phone only."
            else reply.note ?: "Checking for on-device speech recognition.",
        )

        SectionTitle("Camera")
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            SegmentedButton(selected = s.backCamera, onClick = { vm.update { it.copy(backCamera = true) } },
                shape = SegmentedButtonDefaults.itemShape(0, 2)) { Text("Back: facing them") }
            SegmentedButton(selected = !s.backCamera, onClick = { vm.update { it.copy(backCamera = false) } },
                shape = SegmentedButtonDefaults.itemShape(1, 2)) { Text("Front: facing me") }
        }
        SwitchSetting("Show hand tracking", "Draws the tracked hands over the video.", s.showLandmarks) { v ->
            vm.update { it.copy(showLandmarks = v) }
        }

        // Tuning most people never need, folded away.
        Row(
            Modifier.fillMaxWidth().padding(top = 16.dp).clickable(role = Role.Button, onClickLabel = if (advanced) "Hide" else "Show") { advanced = !advanced }
                .padding(vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text("Advanced", style = MaterialTheme.typography.titleMedium)
                Text("Caption timing, taught-sign matching, details overlay", style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            val turn by animateFloatAsState(if (advanced) 180f else 0f, label = "chevron")
            Icon(Icons.Filled.ExpandMore, null, Modifier.rotate(turn))
        }
        AnimatedVisibility(advanced) {
            Column {
                SwitchSetting(
                    "Wait for the hands to rest",
                    "Off (recommended): each word appears as it is signed, hands kept up. On: one answer per sign, given only when the hands drop.",
                    !s.continuous,
                ) { v -> vm.update { it.copy(continuous = !v) } }
                AnimatedVisibility(!s.continuous) {
                    Column {
                        SliderSetting(
                            "Confidence needed", "%d%%".format(Math.round(100 * s.threshold)),
                            "Lower shows more words, including wrong ones; higher shows fewer, surer words.",
                            s.threshold, 0f..0.6f, 11,
                        ) { v -> vm.update { it.copy(threshold = v) } }
                        SliderSetting(
                            "Stability", if (s.agree == 1) "1 look" else "${s.agree} looks in a row",
                            "Hands up for over four seconds: how many consecutive looks must agree before a word is shown.",
                            s.agree.toFloat(), 1f..4f, 2,
                        ) { v -> vm.update { it.copy(agree = Math.round(v)) } }
                        SliderSetting(
                            "Listening window", "%.1f s".format(s.windowSec),
                            "Hands up for over four seconds: how much signing each look covers.",
                            s.windowSec, 1f..3f, 7,
                        ) { v -> vm.update { it.copy(windowSec = v) } }
                    }
                }
                SliderSetting(
                    "Taught-sign match", "%.2f".format(s.teachThreshold),
                    "How closely a sign must match one taught on the Teach tab. Raise it if taught signs appear when they should not.",
                    s.teachThreshold, 0.45f..0.65f, 3,
                ) { v -> vm.update { it.copy(teachThreshold = v) } }
                SwitchSetting("Show details on Live", "Best guesses, frame rate, timings and voice delay over the video.", s.showDetails) { v ->
                    vm.update { it.copy(showDetails = v) }
                }
                SwitchSetting("Track on the GPU", "Off: two CPU trackers take turns, 30 fps on a Galaxy S26. Try on if tracking is slow on another phone.", s.gpu) { v ->
                    vm.update { it.copy(gpu = v) }
                }
            }
        }

        SectionTitle("About")
        Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer)) {
            Column(Modifier.padding(18.dp)) {
                InfoRow("Vocabulary", info?.let { "%,d ASL signs".format(it.words) } ?: "loading")
                InfoRow("Accuracy (held-out test)", info?.let { "%.1f%% top-1 · %.1f%% top-5 on %,d clips".format(it.testTop1, it.testTop5, it.testClips) } ?: "–")
                InfoRow("Recogniser", info?.let { "INT8, %.1f MB".format(it.classifierMb) } ?: "–")
                InfoRow("Hand tracking", "MediaPipe Holistic, 13.7 MB", last = true)
            }
        }
        Spacer(Modifier.height(12.dp))
        Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)) {
            Row(Modifier.padding(18.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.Lock, null, tint = MaterialTheme.colorScheme.onPrimaryContainer)
                Spacer(Modifier.width(14.dp))
                Text(
                    "Everything runs on this phone. The app has no internet permission, so video, audio and keypoints cannot leave it.",
                    style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onPrimaryContainer,
                )
            }
        }
        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun SliderSetting(
    title: String, value: String, help: String, current: Float, range: ClosedFloatingPointRange<Float>, steps: Int,
    onChange: (Float) -> Unit,
) {
    Column(Modifier.padding(vertical = 8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(title, style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
            Text(value, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
        }
        Slider(value = current, onValueChange = onChange, valueRange = range, steps = steps)
        Text(help, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun SwitchSetting(title: String, help: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth().clickable(role = Role.Switch) { onChange(!checked) }.padding(vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge)
            Text(help, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Spacer(Modifier.width(12.dp))
        Switch(checked = checked, onCheckedChange = null)
    }
}

@Composable
private fun InfoLine(title: String, body: String) {
    Column(Modifier.fillMaxWidth().padding(vertical = 10.dp)) {
        Text(title, style = MaterialTheme.typography.bodyLarge)
        Text(body, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun InfoRow(label: String, value: String, last: Boolean = false) {
    Column(Modifier.padding(vertical = 6.dp)) {
        Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.bodyLarge)
    }
    if (!last) HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
}
