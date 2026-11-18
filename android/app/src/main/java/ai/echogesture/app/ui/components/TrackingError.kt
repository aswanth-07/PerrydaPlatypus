package ai.echogesture.app.ui.components

import ai.echogesture.app.R
import androidx.compose.ui.res.stringResource
import ai.echogesture.app.engine.LiveState
import ai.echogesture.app.ui.theme.Palette
import ai.echogesture.app.ui.theme.glass
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ErrorOutline
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp

/**
 * Hand tracking could not start (the tracker failed to load, often the GPU delegate on an unusual phone). Says so
 * in plain words and offers the two things that help: trying again, and switching between CPU and GPU tracking.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun TrackingError(state: LiveState, gpu: Boolean, onRetry: () -> Unit, onSwitch: () -> Unit, modifier: Modifier = Modifier) {
    AnimatedVisibility(state.error != null, modifier, enter = fadeIn(), exit = fadeOut()) {
        Column(
            Modifier.widthIn(max = 420.dp).fillMaxWidth().glass(RoundedCornerShape(20.dp))
                .semantics { liveRegion = LiveRegionMode.Assertive }.padding(18.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Rounded.ErrorOutline, null, tint = Palette.Red)
                Spacer(Modifier.width(10.dp))
                Text(stringResource(R.string.tracking_hand_tracking_couldn_t_start), style = MaterialTheme.typography.titleMedium, color = Palette.Text)
            }
            Spacer(Modifier.height(12.dp))
            Text(
                stringResource(if (gpu) R.string.tracking_body_cpu else R.string.tracking_body_gpu),
                style = MaterialTheme.typography.bodyMedium, color = Palette.TextMuted,
            )
            state.error?.let {
                Text(it, Modifier.padding(top = 6.dp), style = MaterialTheme.typography.bodySmall, color = Palette.TextMuted.copy(alpha = 0.8f))
            }
            Spacer(Modifier.height(12.dp))
            FlowRow(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = onRetry, modifier = Modifier.heightIn(min = 48.dp)) { Text(stringResource(R.string.tracking_try_again)) }
                OutlinedButton(onClick = onSwitch, modifier = Modifier.heightIn(min = 48.dp)) {
                    Text(if (gpu) stringResource(R.string.tracking_track_on_the_cpu) else stringResource(R.string.tracking_track_on_the_gpu))
                }
            }
        }
    }
}
