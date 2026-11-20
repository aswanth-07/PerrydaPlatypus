package ai.echogesture.app.ui.teach

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.calculateStartPadding
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
import androidx.compose.material.icons.rounded.School
import androidx.compose.material.icons.rounded.Translate
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.ripple
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.R
import ai.echogesture.app.ui.components.LocalCaption
import ai.echogesture.app.ui.components.LocalTabBarInset
import ai.echogesture.app.ui.components.SectionTitle
import ai.echogesture.app.ui.components.TabHeader
import ai.echogesture.app.ui.components.rememberPress
import ai.echogesture.app.ui.theme.Palette

/** The Learn tab: guided lessons, offline Sign Dictionary & Text-to-Sign, free practice, and Teach a Sign. */
@Composable
fun LearnHubScreen(
    vm: MainViewModel,
    onLesson: (String) -> Unit,
    onLessons: () -> Unit,
    onDictionary: () -> Unit,
    onPractise: () -> Unit,
    onTeach: () -> Unit,
    onSettings: () -> Unit,
) {
    val teach by vm.engine.teach.collectAsStateWithLifecycle()
    val show = LocalCaption.current

    val bar = LocalTabBarInset.current
    Column(Modifier.fillMaxSize().padding(start = bar.calculateStartPadding(LayoutDirection.Ltr))) {
        TabHeader(stringResource(R.string.tab_learn), onSettings)
        Column(
            Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            ContinueCard(vm, onOpen = onLesson, onAll = onLessons)

            Entry(
                icon = Icons.Rounded.Translate,
                title = stringResource(R.string.dict_entry_title),
                body = stringResource(R.string.dict_entry_body),
                badge = stringResource(R.string.dict_entry_badge),
                onClick = onDictionary,
            )
            Entry(
                icon = Icons.Rounded.School,
                title = stringResource(R.string.screen_practise_any_sign),
                body = stringResource(R.string.learn_pick_any_of_the_signs),
                onClick = onPractise,
            )
            Entry(
                icon = Icons.Rounded.AddCircleOutline,
                title = stringResource(R.string.screen_teach_the_app_a_sign),
                body = stringResource(R.string.learn_add_a_sign_it_doesn),
                onClick = onTeach,
            )

            SectionTitle(stringResource(R.string.learn_your_signs), Modifier.padding(top = 8.dp, bottom = 2.dp))
            if (teach.signs.isEmpty()) {
                Row(
                    Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(20.dp))
                        .background(MaterialTheme.colorScheme.surfaceContainer)
                        .border(1.dp, Palette.Hairline, RoundedCornerShape(20.dp))
                        .clickable(role = Role.Button, onClick = onTeach)
                        .padding(horizontal = 18.dp, vertical = 16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        stringResource(R.string.learn_signs_you_teach_the_app),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.weight(1f),
                    )
                    Spacer(Modifier.width(12.dp))
                    Icon(Icons.AutoMirrored.Rounded.KeyboardArrowRight, null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            } else {
                Column(
                    Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(20.dp))
                        .background(MaterialTheme.colorScheme.surfaceContainer)
                        .border(1.dp, Palette.Hairline, RoundedCornerShape(20.dp))
                        .clickable(role = Role.Button, onClickLabel = stringResource(R.string.learn_manage_your_signs), onClick = onTeach)
                        .padding(horizontal = 18.dp, vertical = 12.dp),
                ) {
                    teach.signs.forEach { sign ->
                        Row(Modifier.fillMaxWidth().padding(vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                            Text(show(sign.word), Modifier.weight(1f), style = MaterialTheme.typography.titleMedium, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            Text(
                                pluralStringResource(R.plurals.teach_examples, sign.examples, sign.examples),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }
            Spacer(Modifier.height(16.dp + bar.calculateBottomPadding()))
        }
    }
}

@Composable
private fun Entry(
    icon: ImageVector,
    title: String,
    body: String,
    badge: String? = null,
    onClick: () -> Unit,
) {
    val press = rememberPress()
    Row(
        press.scale
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(MaterialTheme.colorScheme.surfaceContainer)
            .border(1.dp, Palette.Hairline, RoundedCornerShape(20.dp))
            .clickable(interactionSource = press.interaction, indication = ripple(), role = Role.Button, onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            Modifier.size(48.dp).clip(CircleShape).background(MaterialTheme.colorScheme.primaryContainer),
            contentAlignment = Alignment.Center,
        ) {
            Icon(icon, null, tint = MaterialTheme.colorScheme.primary)
        }
        Spacer(Modifier.width(16.dp))
        Column(Modifier.weight(1f)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(title, style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f, fill = false))
                if (badge != null) {
                    Box(
                        Modifier
                            .clip(RoundedCornerShape(50))
                            .background(Palette.DeepAzure)
                            .padding(horizontal = 8.dp, vertical = 2.dp),
                    ) {
                        Text(badge, style = MaterialTheme.typography.labelSmall, color = Palette.LightAzure)
                    }
                }
            }
            Spacer(Modifier.height(2.dp))
            Text(body, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Spacer(Modifier.width(4.dp))
        Icon(Icons.AutoMirrored.Rounded.KeyboardArrowRight, null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}
