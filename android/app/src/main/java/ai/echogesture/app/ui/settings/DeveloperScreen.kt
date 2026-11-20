package ai.echogesture.app.ui.settings

import ai.echogesture.app.R
import androidx.compose.ui.res.stringResource
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.ui.components.SectionTitle
import ai.echogesture.app.ui.components.SubScreen

/**
 * Tuning and measurements for demos and debugging, unlocked by tapping the version in Settings seven times.
 * Nothing here is needed to use the app.
 */
@Composable
fun DeveloperScreen(vm: MainViewModel, onBack: () -> Unit) {
    val s by vm.settings.collectAsStateWithLifecycle()
    val live by vm.live.collectAsStateWithLifecycle()
    val overlay by vm.overlay.collectAsStateWithLifecycle()
    val info = if (live.ready) vm.engine.info else null

    SubScreen(stringResource(R.string.dev_developer_options), onBack) {
        Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 20.dp)) {
            SectionTitle(stringResource(R.string.dev_captions_after_each_sign), Modifier.padding(top = 4.dp))
            Text(
                if (s.continuous) stringResource(R.string.dev_used_when_settings_captions_appear)
                else stringResource(R.string.dev_used_now_settings_captions_appear),
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            SliderSetting(
                stringResource(R.string.dev_confidence_needed), "%d%%".format(Math.round(100 * s.threshold)),
                stringResource(R.string.dev_lower_shows_more_words_including),
                s.threshold, 0f..0.6f, 11,
            ) { v -> vm.update { it.copy(threshold = v) } }
            SliderSetting(
                stringResource(R.string.dev_stability), if (s.agree == 1) stringResource(R.string.dev_1_look) else stringResource(R.string.dev_looks_in_a_row, s.agree),
                stringResource(R.string.dev_hands_up_for_over_four),
                s.agree.toFloat(), 1f..4f, 2,
            ) { v -> vm.update { it.copy(agree = Math.round(v)) } }
            SliderSetting(
                stringResource(R.string.dev_listening_window), "%.1f s".format(s.windowSec),
                stringResource(R.string.dev_hands_up_for_over_four_2),
                s.windowSec, 1f..3f, 7,
            ) { v -> vm.update { it.copy(windowSec = v) } }

            SectionTitle(stringResource(R.string.dev_taught_signs))
            SliderSetting(
                stringResource(R.string.dev_taught_sign_match), "%.2f".format(s.teachThreshold),
                stringResource(R.string.dev_how_closely_a_sign_must),
                s.teachThreshold, 0.45f..0.65f, 3,
            ) { v -> vm.update { it.copy(teachThreshold = v) } }

            SectionTitle(stringResource(R.string.dev_talk_tab))
            SwitchSetting(stringResource(R.string.dev_show_details), stringResource(R.string.dev_best_guesses_frame_rate_timings), s.showDetails) { v ->
                vm.update { it.copy(showDetails = v) }
            }
            SwitchSetting(stringResource(R.string.dev_hand_skeleton), stringResource(R.string.dev_draws_the_tracked_bones_and), s.skeleton) { v ->
                vm.update { it.copy(skeleton = v) }
            }
            SwitchSetting(stringResource(R.string.tracking_track_on_the_gpu), stringResource(R.string.dev_off_two_cpu_trackers_take), s.gpu) { v ->
                vm.update { it.copy(gpu = v) }
            }
            AnimatedVisibility(live.ready) {
                Text(
                    stringResource(R.string.dev_now_0f_fps_tracking_0f).format(
                        overlay.fps, overlay.landmarksMs, overlay.modelMs, live.lanes, if (live.gpu) stringResource(R.string.talk_gpu) else stringResource(R.string.talk_cpu), if (live.lanes == 1) "" else "s",
                    ),
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            SectionTitle(stringResource(R.string.dev_model))
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer)) {
                Column(Modifier.padding(18.dp)) {
                    InfoRow(stringResource(R.string.dev_vocabulary), info?.let { "%,d ASL signs".format(it.words) } ?: "loading")
                    InfoRow(stringResource(R.string.dev_accuracy_held_out_test), info?.let { "%.1f%% top-1 · %.1f%% top-5 on %,d clips".format(it.testTop1, it.testTop5, it.testClips) } ?: "–")
                    InfoRow(stringResource(R.string.dev_recogniser), info?.let { stringResource(R.string.dev_int8_1f_mb).format(it.classifierMb) } ?: "–")
                    InfoRow(stringResource(R.string.dev_hand_tracking), stringResource(R.string.dev_mediapipe_holistic_13_7_mb), last = true)
                }
            }
            Text(
                stringResource(R.string.dev_identify_a_sign_in_a),
                Modifier.padding(top = 10.dp),
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(16.dp))
            OutlinedButton(onClick = { vm.update { it.copy(developer = false, showDetails = false) }; onBack() }, modifier = Modifier.fillMaxWidth()) {
                Text(stringResource(R.string.dev_turn_off_developer_options))
            }
            Spacer(Modifier.height(24.dp))
        }
    }
}
