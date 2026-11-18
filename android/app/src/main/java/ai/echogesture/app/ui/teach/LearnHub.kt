package ai.echogesture.app.ui.teach

import androidx.compose.ui.res.stringResource
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.KeyboardArrowRight
import androidx.compose.material.icons.rounded.AddCircleOutline
import androidx.compose.material.icons.rounded.LocalFireDepartment
import androidx.compose.material.icons.rounded.School
import androidx.compose.material.icons.rounded.VideoLibrary
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.R
import ai.echogesture.app.ui.components.LocalCaption
import ai.echogesture.app.ui.components.SectionTitle
import ai.echogesture.app.ui.components.TabHeader
import ai.echogesture.app.ui.theme.Palette

/** The Learn tab: practise signs, teach the app a sign, or identify the sign in a saved video. */
@Composable
fun LearnHubScreen(
    vm: MainViewModel, onLesson: (String) -> Unit, onLessons: () -> Unit, onPractise: () -> Unit, onTeach: () -> Unit,
    onVideo: () -> Unit, onSettings: () -> Unit,
) {
    val progress by vm.progress.collectAsStateWithLifecycle()
    val teach by vm.engine.teach.collectAsStateWithLifecycle()
    val streak = progress.streak()
    val show = LocalCaption.current

    Column(Modifier.fillMaxSize()) {
        TabHeader(stringResource(R.string.tab_learn), onSettings)
        Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)) {
            if (progress.learned.isNotEmpty()) {
                Row(
                    Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(MaterialTheme.colorScheme.primaryContainer)
                        .padding(horizontal = 18.dp, vertical = 14.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text(pluralStringResource(R.plurals.learn_learned, progress.learned.size, progress.learned.size),
                            style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.onPrimaryContainer)
                        Text(stringResource(R.string.learn_latest, progress.learned.takeLast(4).asReversed().joinToString(", ", transform = show)),
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onPrimaryContainer,
                            maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                    if (streak > 0) {
                        Spacer(Modifier.width(12.dp))
                        Icon(Icons.Rounded.LocalFireDepartment, null, tint = Palette.Amber)
                        Spacer(Modifier.width(4.dp))
                        Text(pluralStringResource(R.plurals.learn_streak, streak, streak), style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.onPrimaryContainer)
                    }
                }
            }
            ContinueCard(vm, onOpen = onLesson, onAll = onLessons)
            Entry(Icons.Rounded.School, stringResource(R.string.screen_practise_any_sign),
                stringResource(R.string.learn_pick_any_of_the_signs), onPractise)
            Entry(Icons.Rounded.AddCircleOutline, stringResource(R.string.screen_teach_the_app_a_sign),
                stringResource(R.string.learn_add_a_sign_it_doesn), onTeach)
            Entry(Icons.Rounded.VideoLibrary, stringResource(R.string.screen_identify_a_sign_in_a),
                stringResource(R.string.learn_pick_a_short_video_of), onVideo)

            SectionTitle(stringResource(R.string.learn_your_signs))
            if (teach.signs.isEmpty()) {
                Text(stringResource(R.string.learn_signs_you_teach_the_app),
                    style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            } else {
                Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(MaterialTheme.colorScheme.surfaceContainer)
                    .clickable(role = Role.Button, onClickLabel = stringResource(R.string.learn_manage_your_signs), onClick = onTeach).padding(horizontal = 18.dp, vertical = 12.dp)) {
                    teach.signs.forEach { sign ->
                        Row(Modifier.fillMaxWidth().padding(vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                            Text(show(sign.word), Modifier.weight(1f), style = MaterialTheme.typography.titleMedium, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            Text(pluralStringResource(R.plurals.teach_examples, sign.examples, sign.examples),
                                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }
            Spacer(Modifier.height(16.dp))
        }
    }
}

@Composable
private fun Entry(icon: ImageVector, title: String, body: String, onClick: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(MaterialTheme.colorScheme.surfaceContainer)
            .clickable(role = Role.Button, onClick = onClick).padding(horizontal = 16.dp, vertical = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.size(48.dp).clip(CircleShape).background(MaterialTheme.colorScheme.primaryContainer), contentAlignment = Alignment.Center) {
            Icon(icon, null, tint = MaterialTheme.colorScheme.primary)
        }
        Spacer(Modifier.width(16.dp))
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Text(body, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Icon(Icons.AutoMirrored.Rounded.KeyboardArrowRight, null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}
