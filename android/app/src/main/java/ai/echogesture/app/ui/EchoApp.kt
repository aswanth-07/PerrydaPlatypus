package ai.echogesture.app.ui

import android.Manifest
import ai.echogesture.app.R
import ai.echogesture.app.ui.components.rememberPermission
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.ui.res.stringResource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.FrontHand
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.School
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.VideoLibrary
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.ui.clips.ClipsScreen
import ai.echogesture.app.ui.live.LiveScreen
import ai.echogesture.app.ui.settings.SettingsScreen
import ai.echogesture.app.ui.teach.SignsScreen
import ai.echogesture.app.ui.transcript.TranscriptScreen

private enum class Tab(val route: String, val label: String, val icon: ImageVector) {
    Live("live", "Live", Icons.Filled.Videocam),
    Chat("chat", "Chat", Icons.AutoMirrored.Filled.Chat),
    Teach("teach", "Signs", Icons.Filled.School),
    Clips("clips", "Videos", Icons.Filled.VideoLibrary),
    Settings("settings", "Settings", Icons.Filled.Tune),
}

@Composable
fun EchoApp(vm: MainViewModel = viewModel()) {
    val nav = rememberNavController()
    val entry by nav.currentBackStackEntryAsState()
    val current = entry?.destination?.route ?: Tab.Live.route

    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        contentWindowInsets = WindowInsets(0),   // each screen handles the status bar; Live draws the camera under it
        bottomBar = {
            NavigationBar(containerColor = MaterialTheme.colorScheme.surfaceContainerLow) {
                Tab.entries.forEach { tab ->
                    NavigationBarItem(
                        selected = current == tab.route,
                        onClick = {
                            nav.navigate(tab.route) {
                                popUpTo(nav.graph.findStartDestination().id) { saveState = true }
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                        icon = { Icon(tab.icon, null) },
                        label = { Text(tab.label) },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = MaterialTheme.colorScheme.primary,
                            selectedTextColor = MaterialTheme.colorScheme.primary,
                            indicatorColor = MaterialTheme.colorScheme.primaryContainer,
                        ),
                    )
                }
            }
        },
    ) { padding ->
        NavHost(nav, startDestination = Tab.Live.route, modifier = Modifier.padding(padding)) {
            composable(Tab.Live.route) {
                CameraGate(
                    Icons.Filled.FrontHand, stringResource(R.string.gate_live_title), stringResource(R.string.gate_live_body),
                    listOf(Icons.Filled.Lock to stringResource(R.string.gate_point_private), Icons.Filled.Speed to stringResource(R.string.gate_point_realtime)),
                ) { LiveScreen(vm) }
            }
            composable(Tab.Clips.route) { ClipsScreen(vm) }
            composable(Tab.Teach.route) {
                CameraGate(
                    Icons.Filled.School, stringResource(R.string.gate_signs_title), stringResource(R.string.gate_signs_body),
                    listOf(Icons.Filled.Lock to stringResource(R.string.gate_point_private), Icons.Filled.Lock to stringResource(R.string.gate_point_keypoints)),
                ) { SignsScreen(vm) }
            }
            composable(Tab.Chat.route) { TranscriptScreen(vm) }
            composable(Tab.Settings.route) { SettingsScreen(vm) }
        }
    }
}

/**
 * Shows [content] once the camera is allowed, and before that a short explanation written for this screen. When
 * Android will no longer show its dialog, the button opens the app's settings page instead of doing nothing.
 */
@Composable
private fun CameraGate(
    icon: ImageVector, title: String, body: String, points: List<Pair<ImageVector, String>>,
    content: @Composable () -> Unit,
) {
    val camera = rememberPermission(Manifest.permission.CAMERA)
    if (camera.granted) {
        content()
        return
    }
    Box(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), contentAlignment = Alignment.Center) {
        Column(Modifier.widthIn(max = 420.dp).padding(28.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Box(
                Modifier.size(88.dp).clip(CircleShape).background(MaterialTheme.colorScheme.primaryContainer),
                contentAlignment = Alignment.Center,
            ) { Icon(icon, null, Modifier.size(40.dp), tint = MaterialTheme.colorScheme.primary) }
            Spacer(Modifier.height(20.dp))
            Text(title, style = MaterialTheme.typography.headlineMedium, textAlign = TextAlign.Center)
            Spacer(Modifier.height(10.dp))
            Text(body, style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center)
            Spacer(Modifier.height(20.dp))
            points.forEach { (i, t) -> Point(i, t) }
            Spacer(Modifier.height(24.dp))
            Button(onClick = camera::request, modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp)) {
                Text(stringResource(if (camera.blocked) R.string.action_open_settings else R.string.gate_allow_camera))
            }
            val note = when {
                camera.blocked -> R.string.gate_camera_blocked
                camera.denied -> R.string.gate_camera_denied
                else -> null
            }
            if (note != null) {
                Spacer(Modifier.height(12.dp))
                Text(stringResource(note), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.secondary,
                    textAlign = TextAlign.Center)
            }
        }
    }
}

@Composable
private fun Point(icon: ImageVector, text: String) {
    Row(Modifier.fillMaxWidth().padding(vertical = 6.dp), verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Start) {
        Icon(icon, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(20.dp))
        Spacer(Modifier.width(12.dp))
        Text(text, style = MaterialTheme.typography.bodyMedium)
    }
}
