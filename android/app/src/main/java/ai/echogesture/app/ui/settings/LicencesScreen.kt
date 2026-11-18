package ai.echogesture.app.ui.settings

import ai.echogesture.app.R
import ai.echogesture.app.ui.components.SubScreen
import ai.echogesture.app.ui.components.rememberPress
import ai.echogesture.app.ui.theme.Motion
import androidx.annotation.RawRes
import androidx.annotation.StringRes
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ExpandMore
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.ripple
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.platform.LocalResources
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** A component the app ships, who holds its copyright, its licence, and the licence's full text. */
private class Component(@StringRes val name: Int, @StringRes val holder: Int, @StringRes val licence: Int, @RawRes val text: Int)

private val COMPONENTS = listOf(
    Component(R.string.licence_mediapipe, R.string.licence_holder_google, R.string.licence_apache, R.raw.licence_apache_2_0),
    Component(R.string.licence_onnxruntime, R.string.licence_holder_microsoft, R.string.licence_mit, R.raw.licence_mit_onnxruntime),
    Component(R.string.licence_androidx, R.string.licence_holder_aosp, R.string.licence_apache, R.raw.licence_apache_2_0),
    Component(R.string.licence_kotlin, R.string.licence_holder_jetbrains, R.string.licence_apache, R.raw.licence_apache_2_0),
    Component(R.string.licence_icons, R.string.licence_holder_google, R.string.licence_apache, R.raw.licence_apache_2_0),
    Component(R.string.licence_atkinson, R.string.licence_holder_braille, R.string.licence_ofl, R.raw.licence_atkinson_hyperlegible),
    Component(R.string.licence_manrope, R.string.licence_holder_manrope, R.string.licence_ofl, R.raw.licence_manrope),
)

private val PARAGRAPH = Regex("""\n\s*\n""")
private val LINE_BREAK = Regex("""\s*\n\s*""")

/** Settings > About > Open-source licences: what EchoGesture is built with, each licence readable in full. */
@Composable
fun LicencesScreen(onBack: () -> Unit) {
    SubScreen(stringResource(R.string.licences_title), onBack) {
        LazyColumn(contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 4.dp, bottom = 24.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            item {
                Text(stringResource(R.string.licences_intro), Modifier.padding(horizontal = 4.dp, vertical = 4.dp),
                    style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            items(COMPONENTS, key = { it.name }) { ComponentCard(it) }
            item {
                Text(stringResource(R.string.licences_datasets), Modifier.padding(horizontal = 4.dp, vertical = 8.dp),
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@Composable
private fun ComponentCard(c: Component) {
    var open by rememberSaveable(c.name) { mutableStateOf(false) }
    val press = rememberPress()
    val turn by animateFloatAsState(if (open) 180f else 0f, tween(Motion.MEDIUM, easing = Motion.Standard), label = "chevron")
    val resources = LocalResources.current
    // Licence files are wrapped at 80 columns; each paragraph is joined so it flows to the phone's width instead.
    val text = remember(c.text, open) {
        if (!open) "" else resources.openRawResource(c.text).bufferedReader().use { it.readText() }
            .trim().split(PARAGRAPH).joinToString("\n\n") { para -> para.trim().replace(LINE_BREAK, " ") }
    }
    Column(press.scale.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(MaterialTheme.colorScheme.surfaceContainer)) {
        Row(
            Modifier.fillMaxWidth()
                .clickable(interactionSource = press.interaction, indication = ripple(), role = Role.Button,
                    onClickLabel = stringResource(if (open) R.string.licences_hide else R.string.licences_show)) { open = !open }
                .padding(horizontal = 18.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text(stringResource(c.name), style = MaterialTheme.typography.titleMedium)
                Text(stringResource(c.holder), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(stringResource(c.licence), style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
            }
            Icon(Icons.Rounded.ExpandMore, null, Modifier.rotate(turn))
        }
        AnimatedVisibility(open, enter = expandVertically(tween(Motion.MEDIUM)) + fadeIn(), exit = shrinkVertically(tween(Motion.MEDIUM)) + fadeOut()) {
            Text(
                text, Modifier.padding(start = 18.dp, end = 18.dp, bottom = 16.dp),
                fontFamily = FontFamily.Monospace, fontSize = 11.sp, lineHeight = 15.sp, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
