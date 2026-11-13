package ai.echogesture.app.ui.settings

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.ui.components.SectionTitle

private val VOCAB_NAMES = mapOf("core" to ("Core" to "100 signs"), "full" to ("Full" to "2,000 signs"), "union" to ("Extended" to "3,075 signs"))

@Composable
fun SettingsScreen(vm: MainViewModel) {
    val s by vm.settings.collectAsStateWithLifecycle()
    val live by vm.live.collectAsStateWithLifecycle()
    val info = if (live.ready) vm.engine.info else null

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp)) {
        Text("Settings", style = MaterialTheme.typography.headlineMedium)

        SectionTitle("Vocabulary")
        val vocabs = vm.engine.vocabularies
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            vocabs.forEachIndexed { i, v ->
                SegmentedButton(selected = s.vocabulary == v, onClick = { vm.update { it.copy(vocabulary = v) } },
                    shape = SegmentedButtonDefaults.itemShape(i, vocabs.size)) {
                    val (name, count) = VOCAB_NAMES[v] ?: (v to "")
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(name, maxLines = 1)
                        Text(count, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1)
                    }
                }
            }
        }
        Spacer(Modifier.height(6.dp))
        Text(
            "Core is the most accurate for everyday words. Full recognises every WLASL sign, with lower accuracy per word." +
                if ("union" in vocabs) " Extended adds 1,075 more signs from ASL Citizen." else "",
            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        SectionTitle("Recognition")
        SliderSetting(
            "Confidence needed", "%d%%".format((100 * s.threshold).toInt()),
            "Lower shows more words, including wrong ones; higher shows fewer, surer words.",
            s.threshold, 0.3f..0.8f, 9,
        ) { v -> vm.update { it.copy(threshold = v) } }
        SliderSetting(
            "Stability", if (s.agree == 1) "1 window" else "${s.agree} windows in a row",
            "How many consecutive looks must agree before a word is shown.",
            s.agree.toFloat(), 1f..4f, 2,
        ) { v -> vm.update { it.copy(agree = Math.round(v)) } }
        SliderSetting(
            "Listening window", "%.1f s".format(s.windowSec),
            "How much signing each look covers. Slow signers may want longer.",
            s.windowSec, 1f..3f, 7,
        ) { v -> vm.update { it.copy(windowSec = v) } }

        SliderSetting(
            "How closely a taught sign must match", "%.2f".format(s.teachThreshold),
            "For signs taught on the Teach tab. Lower catches more of them but mistakes more known words for them.",
            s.teachThreshold, 0.45f..0.65f, 3,
        ) { v -> vm.update { it.copy(teachThreshold = v) } }

        SectionTitle("Voice")
        val voice by vm.engine.speaker.state.collectAsStateWithLifecycle()
        SwitchSetting(
            "Speak captions aloud",
            when {
                voice.error != null -> voice.error!!
                !voice.ready -> "Starting the phone's text-to-speech engine."
                voice.onDevice -> "Each word is spoken as soon as it is recognised, by a voice installed on this phone."
                else -> "Each word is spoken as soon as it is recognised. No installed English voice was found: add one in the phone's text-to-speech settings to keep speech offline."
            },
            s.speak,
        ) { v -> vm.update { it.copy(speak = v) } }
        if (s.speak) {
            SliderSetting(
                "Speaking speed", "%.2f×".format(s.speechRate),
                "Faster speech keeps up with quick signing.",
                s.speechRate, 0.8f..1.8f, 9,
            ) { v -> vm.update { it.copy(speechRate = v) } }
            if (voice.spoken > 0) {
                Text(
                    "Last word sounded %.0f ms after the camera frame that settled it (median %.0f ms over %d words).".format(voice.lastMs, voice.medianMs, voice.spoken),
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        SectionTitle("Camera")
        Text("Which camera faces the signer", style = MaterialTheme.typography.bodyLarge)
        Spacer(Modifier.height(8.dp))
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            SegmentedButton(selected = s.backCamera, onClick = { vm.update { it.copy(backCamera = true) } },
                shape = SegmentedButtonDefaults.itemShape(0, 2)) { Text("Back (other person)") }
            SegmentedButton(selected = !s.backCamera, onClick = { vm.update { it.copy(backCamera = false) } },
                shape = SegmentedButtonDefaults.itemShape(1, 2)) { Text("Front (me)") }
        }
        SwitchSetting("Show hand tracking", "Draws the tracked hand skeleton over the video.", s.showLandmarks) { v ->
            vm.update { it.copy(showLandmarks = v) }
        }
        SwitchSetting("Use the GPU for tracking", "Faster on most phones. Turn off if the preview stutters or tracking fails.", s.gpu) { v ->
            vm.update { it.copy(gpu = v) }
        }

        SectionTitle("Model")
        Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer)) {
            Column(Modifier.padding(18.dp)) {
                InfoRow("Vocabulary", info?.let { "${it.words} ASL words" } ?: "loading")
                InfoRow("Accuracy (held-out test)", info?.let { "%.1f%% top-1 · %.1f%% top-5 on %d clips".format(it.testTop1, it.testTop5, it.testClips) } ?: "–")
                InfoRow("Classifier", info?.let { "INT8, %.1f MB, ensemble of %d".format(it.classifierMb, it.ensemble) } ?: "–")
                InfoRow("Tracking", "MediaPipe Holistic, 13.7 MB")
            }
        }

        SectionTitle("Privacy")
        Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)) {
            Row(Modifier.padding(18.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.Lock, null, tint = MaterialTheme.colorScheme.onPrimaryContainer)
                Spacer(Modifier.width(14.dp))
                Text(
                    "Everything runs on this phone. The app has no internet permission, so video and keypoints cannot leave the device.",
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
    Column(Modifier.padding(vertical = 6.dp)) {
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
    Row(Modifier.fillMaxWidth().padding(vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge)
            Text(help, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Spacer(Modifier.width(12.dp))
        Switch(checked = checked, onCheckedChange = onChange)
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Column(Modifier.padding(vertical = 6.dp)) {
        Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.bodyLarge)
    }
    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
}
