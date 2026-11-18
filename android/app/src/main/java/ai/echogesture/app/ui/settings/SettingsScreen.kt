package ai.echogesture.app.ui.settings

import ai.echogesture.app.R
import androidx.compose.ui.res.stringResource
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
import androidx.compose.material.icons.automirrored.rounded.KeyboardArrowRight
import androidx.compose.material.icons.rounded.Lock
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
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Slider
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.ui.platform.LocalDensity
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
import androidx.compose.ui.platform.LocalResources
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
fun SettingsScreen(vm: MainViewModel, onBack: () -> Unit, onDeveloper: () -> Unit, onSetUpAgain: () -> Unit, onLicences: () -> Unit,
    onHistory: () -> Unit,
) {
    val s by vm.settings.collectAsStateWithLifecycle()
    val voice by vm.engine.speaker.state.collectAsStateWithLifecycle()
    val reply by vm.engine.listener.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    var confirm by remember { mutableStateOf<Confirm?>(null) }

    SubScreen(stringResource(R.string.common_settings), onBack) {
        Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 20.dp)) {
            SectionTitle(stringResource(R.string.settings_conversation), Modifier.padding(top = 4.dp))
            Choice(
                stringResource(R.string.settings_captions_appear), listOf(stringResource(R.string.settings_as_they_sign), stringResource(R.string.settings_after_each_sign)), if (s.continuous) 0 else 1,
                if (s.continuous) stringResource(R.string.settings_each_word_shows_while_the)
                else stringResource(R.string.settings_one_word_per_sign_shown),
            ) { i -> vm.update { it.copy(continuous = i == 0) } }
            Choice(
                stringResource(R.string.settings_caption_style), listOf(stringResource(R.string.settings_more_words), stringResource(R.string.settings_balanced), stringResource(R.string.settings_fewer_surer)), s.caution,
                stringResource(R.string.settings_more_words_shows_more_of),
                enabled = s.continuous,
            ) { i -> vm.update { it.copy(caution = i) } }
            val keep = listOf(1, 30, Settings.KEEP_FOREVER)
            Choice(
                stringResource(R.string.settings_keep_conversations), listOf(stringResource(R.string.settings_1_day), stringResource(R.string.settings_30_days), stringResource(R.string.settings_until_i_delete)), keep.indexOf(s.keepDays).coerceAtLeast(0),
                stringResource(R.string.settings_older_conversations_are_deleted_from),
            ) { i -> vm.update { it.copy(keepDays = keep[i]) } }
            SwitchSetting(
                stringResource(R.string.settings_show_signs_in_capitals),
                stringResource(R.string.settings_writes_each_sign_as_a),
                s.glossCaps,
            ) { v -> vm.update { it.copy(glossCaps = v) } }
            ActionRow(stringResource(R.string.settings_past_conversations), stringResource(R.string.settings_past_conversations_help)) { onHistory() }
            ActionRow(stringResource(R.string.settings_delete_all_conversations), stringResource(R.string.settings_removes_every_conversation_in_history), danger = true) {
                confirm = Confirm.CONVERSATIONS
            }

            SectionTitle(stringResource(R.string.settings_voice))
            SwitchSetting(
                stringResource(R.string.settings_speak_signed_words_aloud),
                when {
                    voice.error != null -> voice.error!!
                    !voice.ready -> stringResource(R.string.settings_starting_the_phone_s_text)
                    voice.onDevice -> stringResource(R.string.settings_uses_a_voice_installed_on)
                    else -> stringResource(R.string.settings_no_installed_english_voice_was)
                },
                s.speak,
            ) { v -> vm.update { it.copy(speak = v) } }
            AnimatedVisibility(s.speak) {
                SliderSetting(
                    stringResource(R.string.settings_speaking_speed), "%.2f×".format(s.speechRate), stringResource(R.string.settings_faster_keeps_up_with_quick),
                    s.speechRate, 0.8f..1.8f, 9,
                ) { v -> vm.update { it.copy(speechRate = v) } }
            }
            InfoLine(
                stringResource(R.string.settings_spoken_replies),
                if (reply.available) stringResource(R.string.settings_tap_reply_on_the_talk)
                else reply.note ?: stringResource(R.string.settings_checking_for_on_device_speech),
            )

            SectionTitle(stringResource(R.string.settings_camera))
            Choice(
                stringResource(R.string.settings_camera_on_the_talk_tab), listOf(stringResource(R.string.settings_back_facing_them), stringResource(R.string.settings_front_facing_me)), if (s.backCamera) 0 else 1,
                stringResource(R.string.settings_back_when_you_hold_the),
            ) { i -> vm.update { it.copy(backCamera = i == 0) } }
            SwitchSetting(stringResource(R.string.settings_highlight_tracked_hands), stringResource(R.string.settings_draws_the_hands_the_app), s.showLandmarks) { v ->
                vm.update { it.copy(showLandmarks = v) }
            }

            SectionTitle(stringResource(R.string.settings_feedback))
            SwitchSetting(
                stringResource(R.string.settings_vibration),
                stringResource(R.string.settings_a_short_tap_when_a),
                s.vibrate,
            ) { v -> vm.update { it.copy(vibrate = v) } }

            SectionTitle(stringResource(R.string.settings_learning))
            ActionRow(stringResource(R.string.settings_reset_learning_progress), stringResource(R.string.settings_forgets_which_signs_you_have), danger = true) {
                confirm = Confirm.PROGRESS
            }

            SectionTitle(stringResource(R.string.settings_about))
            About(vm, s.developer, onDeveloper)
            Spacer(Modifier.height(12.dp))
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)) {
                Column(Modifier.padding(start = 18.dp, end = 18.dp, top = 18.dp, bottom = 6.dp)) {
                    Row(verticalAlignment = Alignment.Top) {
                        Icon(Icons.Rounded.Lock, null, tint = MaterialTheme.colorScheme.onPrimaryContainer)
                        Spacer(Modifier.width(14.dp))
                        Text(
                            stringResource(R.string.settings_everything_runs_on_this_phone),
                            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onPrimaryContainer,
                        )
                    }
                    TextButton(onClick = { context.openAppSettings() }, modifier = Modifier.padding(start = 26.dp)) {
                        Text(stringResource(R.string.settings_app_permissions))
                    }
                }
            }
            Spacer(Modifier.height(12.dp))
            Text(
                stringResource(R.string.settings_trained_on_the_wlasl_and),
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            ActionRow(stringResource(R.string.licences_open), stringResource(R.string.licences_intro)) { onLicences() }
            Spacer(Modifier.height(16.dp))
            OutlinedButton(onClick = onSetUpAgain, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text(stringResource(R.string.settings_set_up_again)) }
            Spacer(Modifier.height(24.dp))
        }
    }

    confirm?.let { c ->
        AlertDialog(
            onDismissRequest = { confirm = null },
            title = { Text(if (c == Confirm.CONVERSATIONS) stringResource(R.string.settings_delete_all_conversations_2) else stringResource(R.string.settings_reset_learning_progress_2)) },
            text = {
                Text(
                    if (c == Confirm.CONVERSATIONS) stringResource(R.string.settings_every_conversation_in_history_is)
                    else stringResource(R.string.settings_signs_you_have_learned_and)
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    confirm = null
                    if (c == Confirm.CONVERSATIONS) vm.engine.clearTranscript() else vm.resetProgress()
                }) { Text(if (c == Confirm.CONVERSATIONS) stringResource(R.string.settings_delete_all) else stringResource(R.string.settings_reset)) }
            },
            dismissButton = { TextButton(onClick = { confirm = null }) { Text(stringResource(R.string.action_cancel)) } },
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
    val resources = LocalResources.current
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer)) {
        Column(Modifier.padding(horizontal = 18.dp, vertical = 8.dp)) {
            Column(
                Modifier.fillMaxWidth().clickable(onClickLabel = null) {
                    if (developer) return@clickable
                    taps++
                    when {
                        taps >= DEVELOPER_TAPS -> {
                            vm.update { it.copy(developer = true) }
                            Toast.makeText(context, resources.getString(R.string.settings_developer_options_are_on), Toast.LENGTH_SHORT).show()
                        }
                        taps >= DEVELOPER_TAPS - 3 ->
                            Toast.makeText(context, resources.getQuantityString(R.plurals.settings_taps_left, DEVELOPER_TAPS - taps, DEVELOPER_TAPS - taps), Toast.LENGTH_SHORT).show()
                    }
                }.padding(vertical = 10.dp),
            ) {
                Text(stringResource(R.string.settings_echogesture, version), style = MaterialTheme.typography.bodyLarge)
                Text(stringResource(R.string.settings_recognises_about_2_000_asl), style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            if (developer) {
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                Row(
                    Modifier.fillMaxWidth().clickable(role = Role.Button, onClick = onDeveloper).padding(vertical = 14.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(stringResource(R.string.dev_developer_options), Modifier.weight(1f), style = MaterialTheme.typography.bodyLarge)
                    Icon(Icons.AutoMirrored.Rounded.KeyboardArrowRight, null)
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
        if (LocalDensity.current.fontScale > 1.3f) {
            // Large text: segments would cut their labels off ("After each…"), so the choices become a list.
            Column(Modifier.selectableGroup()) {
                options.forEachIndexed { i, name ->
                    Row(
                        Modifier.fillMaxWidth().heightIn(min = 48.dp)
                            .selectable(selected = selected == i, enabled = enabled, role = Role.RadioButton) { onSelect(i) },
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        RadioButton(selected = selected == i, onClick = null, enabled = enabled)
                        Spacer(Modifier.width(12.dp))
                        Text(name, style = MaterialTheme.typography.bodyLarge)
                    }
                }
            }
        } else {
            SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                options.forEachIndexed { i, name ->
                    SegmentedButton(selected = selected == i, onClick = { onSelect(i) }, enabled = enabled,
                        shape = SegmentedButtonDefaults.itemShape(i, options.size)) { Text(name, maxLines = 1) }
                }
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
