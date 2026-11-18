package ai.echogesture.app.ui.settings

import android.widget.Toast
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.data.Settings
import ai.echogesture.app.ui.components.SectionTitle
import ai.echogesture.app.ui.components.SubScreen
import ai.echogesture.app.ui.components.openAppSettings

/** What a person using the app may want to change. Tuning lives in [DeveloperScreen]. */
@Composable
fun SettingsScreen(vm: MainViewModel, onBack: () -> Unit, onDeveloper: () -> Unit, onSetUpAgain: () -> Unit) {
    val s by vm.settings.collectAsStateWithLifecycle()
    val voice by vm.engine.speaker.state.collectAsStateWithLifecycle()
    val reply by vm.engine.listener.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    var confirm by remember { mutableStateOf<Confirm?>(null) }

    SubScreen("Settings", onBack) {
        Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 20.dp)) {
            SectionTitle("Conversation", Modifier.padding(top = 4.dp))
            Choice(
                "Captions appear", listOf("As they sign", "After each sign"), if (s.continuous) 0 else 1,
                if (s.continuous) "Each word shows while the signer's hands stay up between signs."
                else "One word per sign, shown when the signer's hands come to rest.",
            ) { i -> vm.update { it.copy(continuous = i == 0) } }
            Choice(
                "Caption style", listOf("More words", "Balanced", "Fewer, surer"), s.caution,
                "More words shows more of what was signed, some of it wrong. Fewer, surer shows only words the app is confident about.",
                enabled = s.continuous,
            ) { i -> vm.update { it.copy(caution = i) } }
            val keep = listOf(1, 30, Settings.KEEP_FOREVER)
            Choice(
                "Keep conversations", listOf("1 day", "30 days", "Until I delete"), keep.indexOf(s.keepDays).coerceAtLeast(0),
                "Older conversations are deleted from History automatically.",
            ) { i -> vm.update { it.copy(keepDays = keep[i]) } }
            ActionRow("Delete all conversations", "Removes every conversation in History from this phone.", danger = true) {
                confirm = Confirm.CONVERSATIONS
            }

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
                if (reply.available) "Tap Reply on the Talk tab and speak; your words appear large for the signer. Recognised on this phone only."
                else reply.note ?: "Checking for on-device speech recognition.",
            )

            SectionTitle("Camera")
            Choice(
                "Camera on the Talk tab", listOf("Back: facing them", "Front: facing me"), if (s.backCamera) 0 else 1,
                "Back when you hold the phone toward someone signing; front when you sign to the phone yourself.",
            ) { i -> vm.update { it.copy(backCamera = i == 0) } }
            SwitchSetting("Highlight tracked hands", "Draws the hands the app is following over the video.", s.showLandmarks) { v ->
                vm.update { it.copy(showLandmarks = v) }
            }

            SectionTitle("Learning")
            ActionRow("Reset learning progress", "Forgets which signs you have learned and your practice days.", danger = true) {
                confirm = Confirm.PROGRESS
            }

            SectionTitle("About")
            About(vm, s.developer, onDeveloper)
            Spacer(Modifier.height(12.dp))
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)) {
                Column(Modifier.padding(start = 18.dp, end = 18.dp, top = 18.dp, bottom = 6.dp)) {
                    Row(verticalAlignment = Alignment.Top) {
                        Icon(Icons.Filled.Lock, null, tint = MaterialTheme.colorScheme.onPrimaryContainer)
                        Spacer(Modifier.width(14.dp))
                        Text(
                            "Everything runs on this phone. EchoGesture has no internet access, so video, audio and keypoints " +
                                "cannot leave it. Conversations and taught signs are stored only here.",
                            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onPrimaryContainer,
                        )
                    }
                    TextButton(onClick = { context.openAppSettings() }, modifier = Modifier.padding(start = 26.dp)) {
                        Text("App permissions")
                    }
                }
            }
            Spacer(Modifier.height(12.dp))
            Text(
                "Trained on the WLASL and ASL Citizen sign language datasets. Hand tracking by MediaPipe; recognition runs " +
                    "with ONNX Runtime.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(16.dp))
            OutlinedButton(onClick = onSetUpAgain, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("Set up again") }
            Spacer(Modifier.height(24.dp))
        }
    }

    confirm?.let { c ->
        AlertDialog(
            onDismissRequest = { confirm = null },
            title = { Text(if (c == Confirm.CONVERSATIONS) "Delete all conversations?" else "Reset learning progress?") },
            text = {
                Text(
                    if (c == Confirm.CONVERSATIONS) "Every conversation in History is deleted from this phone. This can't be undone."
                    else "Signs you have learned and your practice streak are forgotten. Taught signs are kept."
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    confirm = null
                    if (c == Confirm.CONVERSATIONS) vm.engine.clearTranscript() else vm.resetProgress()
                }) { Text(if (c == Confirm.CONVERSATIONS) "Delete all" else "Reset") }
            },
            dismissButton = { TextButton(onClick = { confirm = null }) { Text("Cancel") } },
        )
    }
}

private enum class Confirm { CONVERSATIONS, PROGRESS }

/** Version and vocabulary. Tapping the version seven times turns on developer options. */
@Composable
private fun About(vm: MainViewModel, developer: Boolean, onDeveloper: () -> Unit) {
    val context = LocalContext.current
    val version = remember {
        runCatching { context.packageManager.getPackageInfo(context.packageName, 0).versionName }.getOrNull() ?: "?"
    }
    var taps by remember { mutableIntStateOf(0) }
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer)) {
        Column(Modifier.padding(horizontal = 18.dp, vertical = 8.dp)) {
            Column(
                Modifier.fillMaxWidth().clickable(onClickLabel = null) {
                    if (developer) return@clickable
                    taps++
                    when {
                        taps >= DEVELOPER_TAPS -> {
                            vm.update { it.copy(developer = true) }
                            Toast.makeText(context, "Developer options are on", Toast.LENGTH_SHORT).show()
                        }
                        taps >= DEVELOPER_TAPS - 3 ->
                            Toast.makeText(context, "${DEVELOPER_TAPS - taps} more", Toast.LENGTH_SHORT).show()
                    }
                }.padding(vertical = 10.dp),
            ) {
                Text("EchoGesture $version", style = MaterialTheme.typography.bodyLarge)
                Text("Recognises about 2,000 ASL signs. Works offline.", style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            if (developer) {
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                Row(
                    Modifier.fillMaxWidth().clickable(role = Role.Button, onClick = onDeveloper).padding(vertical = 14.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text("Developer options", Modifier.weight(1f), style = MaterialTheme.typography.bodyLarge)
                    Icon(Icons.AutoMirrored.Filled.KeyboardArrowRight, null)
                }
            }
        }
    }
}

private const val DEVELOPER_TAPS = 7

@Composable
internal fun Choice(
    title: String, options: List<String>, selected: Int, help: String?, enabled: Boolean = true, onSelect: (Int) -> Unit,
) {
    Column(Modifier.padding(vertical = 8.dp)) {
        Text(title, style = MaterialTheme.typography.bodyLarge)
        Spacer(Modifier.height(8.dp))
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            options.forEachIndexed { i, name ->
                SegmentedButton(selected = selected == i, onClick = { onSelect(i) }, enabled = enabled,
                    shape = SegmentedButtonDefaults.itemShape(i, options.size)) { Text(name, maxLines = 1) }
            }
        }
        if (help != null) {
            Spacer(Modifier.height(6.dp))
            Text(help, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun ActionRow(title: String, help: String, danger: Boolean = false, onClick: () -> Unit) {
    Column(Modifier.fillMaxWidth().clickable(role = Role.Button, onClick = onClick).padding(vertical = 12.dp)) {
        Text(title, style = MaterialTheme.typography.bodyLarge, color = if (danger) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurface)
        Text(help, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
internal fun SliderSetting(
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
internal fun SwitchSetting(title: String, help: String, checked: Boolean, onChange: (Boolean) -> Unit) {
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
internal fun InfoLine(title: String, body: String) {
    Column(Modifier.fillMaxWidth().padding(vertical = 10.dp)) {
        Text(title, style = MaterialTheme.typography.bodyLarge)
        Text(body, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
internal fun InfoRow(label: String, value: String, last: Boolean = false) {
    Column(Modifier.padding(vertical = 6.dp)) {
        Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.bodyLarge)
    }
    if (!last) HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
}
