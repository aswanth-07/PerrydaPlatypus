package ai.echogesture.app.ui.settings

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
    val info = if (live.ready) vm.engine.info else null

    SubScreen("Developer options", onBack) {
        Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 20.dp)) {
            SectionTitle("Captions after each sign", Modifier.padding(top = 4.dp))
            Text(
                if (s.continuous) "Used when Settings > Captions appear is \"After each sign\"; it is \"As they sign\" now."
                else "Used now: Settings > Captions appear is \"After each sign\".",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
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

            SectionTitle("Taught signs")
            SliderSetting(
                "Taught-sign match", "%.2f".format(s.teachThreshold),
                "How closely a sign must match one taught on this phone. Raise it if taught signs appear when they should not.",
                s.teachThreshold, 0.45f..0.65f, 3,
            ) { v -> vm.update { it.copy(teachThreshold = v) } }

            SectionTitle("Talk tab")
            SwitchSetting("Show details", "Best guesses, frame rate, timings and voice delay over the video.", s.showDetails) { v ->
                vm.update { it.copy(showDetails = v) }
            }
            SwitchSetting("Track on the GPU", "Off: two CPU trackers take turns, 30 fps on a Galaxy S26. Try on if tracking is slow on another phone.", s.gpu) { v ->
                vm.update { it.copy(gpu = v) }
            }
            AnimatedVisibility(live.ready) {
                Text(
                    "Now: %.0f fps · tracking %.0f ms · model %.1f ms · %d %s tracker%s".format(
                        live.fps, live.landmarksMs, live.modelMs, live.lanes, if (live.gpu) "GPU" else "CPU", if (live.lanes == 1) "" else "s",
                    ),
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            SectionTitle("Model")
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer)) {
                Column(Modifier.padding(18.dp)) {
                    InfoRow("Vocabulary", info?.let { "%,d ASL signs".format(it.words) } ?: "loading")
                    InfoRow("Accuracy (held-out test)", info?.let { "%.1f%% top-1 · %.1f%% top-5 on %,d clips".format(it.testTop1, it.testTop5, it.testClips) } ?: "–")
                    InfoRow("Recogniser", info?.let { "INT8, %.1f MB".format(it.classifierMb) } ?: "–")
                    InfoRow("Hand tracking", "MediaPipe Holistic, 13.7 MB", last = true)
                }
            }
            Text(
                "Identify a sign in a video (Learn tab) shows how long each step took while developer options are on.",
                Modifier.padding(top = 10.dp),
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(16.dp))
            OutlinedButton(onClick = { vm.update { it.copy(developer = false, showDetails = false) }; onBack() }, modifier = Modifier.fillMaxWidth()) {
                Text("Turn off developer options")
            }
            Spacer(Modifier.height(24.dp))
        }
    }
}
