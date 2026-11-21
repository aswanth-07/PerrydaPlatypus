package ai.echogesture.app.ui.clips

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.calculateStartPadding
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.VolumeUp
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.ErrorOutline
import androidx.compose.material.icons.rounded.Lock
import androidx.compose.material.icons.rounded.School
import androidx.compose.material.icons.rounded.Translate
import androidx.compose.material.icons.rounded.VideoLibrary
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.ClipProgress
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.R
import ai.echogesture.app.data.Demos
import ai.echogesture.app.engine.ClipResult
import ai.echogesture.app.engine.Speaker
import ai.echogesture.app.ui.components.ConfidenceRow
import ai.echogesture.app.ui.components.EmptyState
import ai.echogesture.app.ui.components.LocalCaption
import ai.echogesture.app.ui.components.LocalTabBarInset
import ai.echogesture.app.ui.components.Metric
import ai.echogesture.app.ui.components.TabHeader
import ai.echogesture.app.ui.components.VideoArt
import ai.echogesture.app.ui.theme.CaptionStyle
import ai.echogesture.app.ui.theme.Palette

/**
 * The Videos tab: identify the sign in a short video from the gallery, hear it spoken aloud, and jump straight to
 * watching or practising any matched sign in the offline 300-sign demonstration pack.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ClipsScreen(
    vm: MainViewModel,
    showTimings: Boolean,
    onWatchSign: (String) -> Unit,
    onPractiseSign: (String) -> Unit,
    onSettings: () -> Unit,
) {
    val clips by vm.clips.collectAsStateWithLifecycle()
    val progress by vm.clipProgress.collectAsStateWithLifecycle()
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri -> uri?.let(vm::recognise) }
    val busy = progress != null && progress?.error == null
    val context = LocalContext.current
    val demoWords = remember { Demos.load(context).keys }

    val bar = LocalTabBarInset.current
    Column(Modifier.fillMaxSize().padding(start = bar.calculateStartPadding(LayoutDirection.Ltr))) {
        TabHeader(stringResource(R.string.tab_videos), onSettings)
        LazyColumn(
            Modifier.fillMaxSize(),
            contentPadding = PaddingValues(start = 20.dp, end = 20.dp, top = 8.dp, bottom = 20.dp + bar.calculateBottomPadding()),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            item {
                Text(
                    stringResource(R.string.video_pick_a_short_video_of),
                    style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(Modifier.height(14.dp))
                Button(
                    onClick = { pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)) },
                    enabled = !busy,
                    modifier = Modifier.fillMaxWidth().height(54.dp),
                ) {
                    Icon(Icons.Rounded.VideoLibrary, null)
                    Spacer(Modifier.width(10.dp))
                    Text(if (busy) stringResource(R.string.video_working) else stringResource(R.string.video_choose_a_video))
                }
                Spacer(Modifier.height(10.dp))
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    InfoBadge(Icons.Rounded.Translate, stringResource(R.string.video_badge_vocab))
                    InfoBadge(Icons.Rounded.CheckCircle, stringResource(R.string.video_badge_offline))
                    InfoBadge(Icons.Rounded.Lock, stringResource(R.string.video_badge_private))
                }
            }
            progress?.let { p -> item { ProgressCard(p, onDismiss = vm::dismissClipError) } }
            if (clips.isEmpty() && progress == null) {
                item {
                    EmptyState(
                        { VideoArt() }, stringResource(R.string.video_no_videos_yet),
                        stringResource(R.string.video_each_result_shows_the_sign),
                        Modifier.fillMaxWidth(),
                    )
                }
            }
            items(clips, key = { it.atMillis }) { r ->
                ResultCard(
                    r = r,
                    showTimings = showTimings,
                    hasDemo = r.word.uppercase() in demoWords,
                    onSpeak = { vm.engine.speakNow(Speaker.sayable(r.word)) },
                    onWatch = { onWatchSign(r.word.uppercase()) },
                    onPractise = { onPractiseSign(r.word.uppercase()) },
                )
            }
            if (clips.isNotEmpty()) item {
                TextButton(onClick = vm::clearClips, modifier = Modifier.fillMaxWidth()) { Text(stringResource(R.string.video_clear_results)) }
            }
        }
    }
}

@Composable
private fun InfoBadge(icon: ImageVector, text: String) {
    Row(
        Modifier
            .clip(RoundedCornerShape(50))
            .background(MaterialTheme.colorScheme.surfaceContainer)
            .border(1.dp, Palette.Hairline, RoundedCornerShape(50))
            .padding(horizontal = 10.dp, vertical = 5.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(icon, null, Modifier.size(13.dp), tint = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.width(5.dp))
        Text(text, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun ProgressCard(p: ClipProgress, onDismiss: () -> Unit) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerHigh)) {
        Column(Modifier.padding(18.dp)) {
            if (p.error != null) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Rounded.ErrorOutline, null, tint = MaterialTheme.colorScheme.error)
                    Spacer(Modifier.width(10.dp))
                    Text(stringResource(R.string.video_could_not_recognise_this_clip), style = MaterialTheme.typography.titleMedium)
                }
                Spacer(Modifier.height(6.dp))
                Text(p.error, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                TextButton(onClick = onDismiss) { Text(stringResource(R.string.video_dismiss)) }
            } else {
                Text(p.stage, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Spacer(Modifier.height(10.dp))
                LinearProgressIndicator(progress = { p.fraction }, modifier = Modifier.fillMaxWidth())
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ResultCard(
    r: ClipResult,
    showTimings: Boolean,
    hasDemo: Boolean,
    onSpeak: () -> Unit,
    onWatch: () -> Unit,
    onPractise: () -> Unit,
) {
    val show = LocalCaption.current
    Box(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(22.dp))
            .background(MaterialTheme.colorScheme.surfaceContainer)
            .border(1.dp, Palette.Hairline, RoundedCornerShape(22.dp)),
    ) {
        Column(Modifier.padding(20.dp)) {
            Text(
                if (showTimings) stringResource(R.string.video_frames, r.name, r.frames) else r.name,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Spacer(Modifier.height(4.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Row(verticalAlignment = Alignment.Bottom) {
                        Text(show(r.word), style = CaptionStyle.copy(fontSize = 34.sp, lineHeight = 42.sp), color = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(10.dp))
                        Text(
                            if (r.taught) stringResource(R.string.video_taught_sign_match_2f).format(r.confidence)
                            else stringResource(R.string.video_sure, (100 * r.confidence).toInt()),
                            style = MaterialTheme.typography.titleSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(bottom = 5.dp),
                        )
                    }
                }
                FilledTonalIconButton(
                    onClick = onSpeak,
                    modifier = Modifier.size(44.dp),
                    colors = IconButtonDefaults.filledTonalIconButtonColors(
                        containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
                        contentColor = Palette.Text,
                    ),
                ) { Icon(Icons.AutoMirrored.Rounded.VolumeUp, stringResource(R.string.dict_hear_aloud)) }
            }
            if (r.also.isNotEmpty()) {
                Spacer(Modifier.height(4.dp))
                Text(
                    stringResource(R.string.video_same_sign, r.also.joinToString(" · ", transform = show)),
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.secondary,
                )
            }
            if (hasDemo) {
                Spacer(Modifier.height(10.dp))
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    AssistChip(
                        onClick = onWatch,
                        leadingIcon = { Icon(Icons.Rounded.Translate, null, Modifier.size(16.dp)) },
                        label = { Text(stringResource(R.string.video_watch_sign)) },
                    )
                    AssistChip(
                        onClick = onPractise,
                        leadingIcon = { Icon(Icons.Rounded.School, null, Modifier.size(16.dp)) },
                        label = { Text(stringResource(R.string.video_practise_sign)) },
                    )
                }
            }
            Spacer(Modifier.height(14.dp))
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                r.top5.forEachIndexed { i, (w, p) -> ConfidenceRow(show(w), p, highlight = i == 0) }
            }
            if (showTimings) {
                Spacer(Modifier.height(16.dp))
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Metric("reading", "%.0f ms".format(r.decodeMs))
                    Metric("tracking", "%.0f ms".format(r.landmarksMs))
                    Metric("model", "%.1f ms".format(r.modelMs))
                    Metric("total", "%.2f s".format(r.totalMs / 1000))
                }
            }
        }
    }
}
