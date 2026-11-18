package ai.echogesture.app.ui.components

import ai.echogesture.app.ui.theme.Motion
import ai.echogesture.app.ui.theme.Palette
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandHorizontally
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkHorizontally
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.ripple
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp

/** One tab of the floating bar. */
class TabItem(val route: String, val label: String, val icon: ImageVector, val selectedIcon: ImageVector)

/**
 * The tab bar: a floating capsule, the selected tab a Dark Azure pill with its name, the others icons only. The
 * pill grows into place with a spring and each tab gives a little under the finger. [vertical] stands it on its
 * side for landscape, where the height is kept for the camera.
 */
@Composable
fun FloatingTabBar(items: List<TabItem>, current: String?, onSelect: (String) -> Unit, modifier: Modifier = Modifier, vertical: Boolean = false) {
    val haptics = LocalHaptics.current
    val shape = CircleShape
    val bar = modifier
        .shadow(18.dp, shape, ambientColor = Color.Black, spotColor = Color.Black)
        .clip(shape)
        .background(Palette.Onyx2.copy(alpha = 0.96f))
        .border(1.dp, Palette.Hairline, shape)
        .selectableGroup()
        .padding(6.dp)
    val tab: @Composable (TabItem) -> Unit = { item ->
        TabPill(item, selected = item.route == current, showLabel = !vertical) {
            if (item.route != current) haptics.tick()
            onSelect(item.route)
        }
    }
    if (vertical) {
        Column(bar, verticalArrangement = Arrangement.spacedBy(4.dp), horizontalAlignment = Alignment.CenterHorizontally) { items.forEach { tab(it) } }
    } else {
        Row(bar, horizontalArrangement = Arrangement.spacedBy(4.dp), verticalAlignment = Alignment.CenterVertically) { items.forEach { tab(it) } }
    }
}

@Composable
private fun TabPill(item: TabItem, selected: Boolean, showLabel: Boolean, onClick: () -> Unit) {
    val press = rememberPress(0.9f)
    val fill by animateColorAsState(if (selected) MaterialTheme.colorScheme.primaryContainer else Color.Transparent, tween(Motion.MEDIUM), label = "tab fill")
    val tint by animateColorAsState(
        if (selected) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSurfaceVariant, tween(Motion.MEDIUM), label = "tab tint",
    )
    Row(
        press.scale
            .clip(CircleShape)
            .background(fill)
            .selectable(selected = selected, interactionSource = press.interaction, indication = ripple(), role = Role.Tab, onClick = onClick)
            .heightIn(min = 52.dp).widthIn(min = 56.dp)
            .padding(horizontal = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center,
    ) {
        // The icon names the tab for TalkBack when its label is hidden.
        Icon(if (selected) item.selectedIcon else item.icon, if (selected && showLabel) null else item.label, Modifier.size(24.dp), tint = tint)
        AnimatedVisibility(
            selected && showLabel,
            enter = expandHorizontally(spring(dampingRatio = 0.75f, stiffness = Spring.StiffnessMediumLow)) + fadeIn(tween(Motion.MEDIUM)),
            exit = shrinkHorizontally(spring(dampingRatio = 0.9f, stiffness = Spring.StiffnessMedium)) + fadeOut(tween(Motion.SHORT)),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Spacer(Modifier.width(8.dp))
                Text(item.label, style = MaterialTheme.typography.labelLarge, color = tint, maxLines = 1)
            }
        }
    }
}

/** A press shared by an element's click and its scale: it gives a little under the finger and springs back. */
class Press(val interaction: MutableInteractionSource, val scale: Modifier)

@Composable
fun rememberPress(pressedScale: Float = 0.97f): Press {
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val s by animateFloatAsState(
        if (pressed) pressedScale else 1f,
        spring(dampingRatio = 0.55f, stiffness = Spring.StiffnessMediumLow), label = "press",
    )
    return Press(interaction, Modifier.graphicsLayer { scaleX = s; scaleY = s })
}

/** How far the floating tab bar reaches into the screen, so content can keep clear of it. */
val LocalTabBarInset = androidx.compose.runtime.compositionLocalOf { androidx.compose.foundation.layout.PaddingValues(0.dp) }

/** Space under scrolling content for the floating tab bar. */
@Composable
fun TabBarSpacer() {
    val inset = LocalTabBarInset.current
    Box(Modifier.padding(inset))
}
