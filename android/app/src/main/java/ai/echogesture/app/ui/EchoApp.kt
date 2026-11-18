package ai.echogesture.app.ui

import ai.echogesture.app.MainViewModel
import ai.echogesture.app.R
import ai.echogesture.app.ui.clips.ClipsScreen
import ai.echogesture.app.ui.components.CameraGate
import ai.echogesture.app.ui.components.SubScreen
import ai.echogesture.app.ui.history.ConversationScreen
import ai.echogesture.app.ui.history.HistoryScreen
import ai.echogesture.app.ui.live.LiveScreen
import ai.echogesture.app.ui.settings.DeveloperScreen
import ai.echogesture.app.ui.settings.SettingsScreen
import ai.echogesture.app.ui.setup.SetupScreen
import ai.echogesture.app.ui.teach.LearnHubScreen
import ai.echogesture.app.ui.teach.LearnScreen
import ai.echogesture.app.ui.teach.TeachScreen
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FrontHand
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.School
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material.icons.outlined.History
import androidx.compose.material.icons.outlined.School
import androidx.compose.material.icons.outlined.Videocam
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument

/** The three tabs. Everything else (Settings, a conversation, Teach…) opens over them with a back arrow. */
private enum class Tab(val route: String, val label: Int, val icon: ImageVector, val selectedIcon: ImageVector) {
    Talk("talk", R.string.tab_talk, Icons.Outlined.Videocam, Icons.Filled.Videocam),
    Learn("learn", R.string.tab_learn, Icons.Outlined.School, Icons.Filled.School),
    History("history", R.string.tab_history, Icons.Outlined.History, Icons.Filled.History),
}

private object Route {
    const val SETTINGS = "settings"
    const val DEVELOPER = "developer"
    const val SETUP = "setup"
    const val PRACTICE = "practice"
    const val TEACH = "teach"
    const val VIDEO = "video"
    const val CONVERSATION = "conversation/{start}"
    fun conversation(start: Long) = "conversation/$start"
}

@Composable
fun EchoApp(vm: MainViewModel = viewModel()) {
    val loaded by vm.settingsLoaded.collectAsStateWithLifecycle()
    val settings by vm.settings.collectAsStateWithLifecycle()
    // The surface gives every screen, including setup outside the tabs' scaffold, the app's background and text colour.
    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        when {
            // The saved settings take a few milliseconds to read; choosing a first screen from the defaults would flash setup.
            !loaded -> Unit
            // A benchmark replaying video through the real screens (--ez ui true) skips setup.
            !settings.onboarded && !vm.engine.replayOnly -> SetupScreen(vm, onDone = {})
            else -> Tabs(vm)
        }
    }
}

@Composable
private fun Tabs(vm: MainViewModel) {
    val nav = rememberNavController()
    val entry by nav.currentBackStackEntryAsState()
    val current = entry?.destination?.route
    val settings by vm.settings.collectAsStateWithLifecycle()
    val onTab = Tab.entries.any { it.route == current }

    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        contentWindowInsets = WindowInsets(0),   // each screen handles the status bar; Talk draws the camera under it
        bottomBar = {
            if (onTab) {
                NavigationBar(containerColor = MaterialTheme.colorScheme.surfaceContainerLow) {
                    Tab.entries.forEach { tab ->
                        val selected = current == tab.route
                        NavigationBarItem(
                            selected = selected,
                            onClick = { nav.toTab(tab.route) },
                            icon = { Icon(if (selected) tab.selectedIcon else tab.icon, null) },
                            label = { Text(stringResource(tab.label)) },
                            colors = NavigationBarItemDefaults.colors(
                                selectedIconColor = MaterialTheme.colorScheme.primary,
                                selectedTextColor = MaterialTheme.colorScheme.primary,
                                indicatorColor = MaterialTheme.colorScheme.primaryContainer,
                            ),
                        )
                    }
                }
            }
        },
    ) { padding ->
        NavHost(nav, startDestination = Tab.Talk.route, modifier = Modifier.padding(padding)) {
            val openSettings = { nav.navigate(Route.SETTINGS) { launchSingleTop = true } }
            composable(Tab.Talk.route) {
                CameraGate(
                    Icons.Filled.FrontHand, stringResource(R.string.gate_live_title), stringResource(R.string.gate_live_body),
                    listOf(Icons.Filled.Lock to stringResource(R.string.gate_point_private), Icons.Filled.Speed to stringResource(R.string.gate_point_realtime)),
                ) { LiveScreen(vm, onSettings = openSettings, onHistory = { nav.toTab(Tab.History.route) }) }
            }
            composable(Tab.Learn.route) {
                LearnHubScreen(
                    vm,
                    onPractise = { nav.navigate(Route.PRACTICE) },
                    onTeach = { nav.navigate(Route.TEACH) },
                    onVideo = { nav.navigate(Route.VIDEO) },
                    onSettings = openSettings,
                )
            }
            composable(Tab.History.route) {
                HistoryScreen(vm, onOpen = { nav.navigate(Route.conversation(it)) }, onSettings = openSettings)
            }

            composable(Route.PRACTICE) {
                Sub { SubScreen("Practise signs", nav::popBackStack) { SignsCameraGate { LearnScreen(vm) } } }
            }
            composable(Route.TEACH) {
                Sub { SubScreen("Teach the app a sign", nav::popBackStack) { SignsCameraGate { TeachScreen(vm) } } }
            }
            composable(Route.VIDEO) {
                Sub { SubScreen("Identify a sign in a video", nav::popBackStack) { ClipsScreen(vm, showTimings = settings.developer) } }
            }
            composable(Route.CONVERSATION, arguments = listOf(navArgument("start") { type = NavType.LongType })) { e ->
                Sub { ConversationScreen(vm, e.arguments?.getLong("start") ?: 0L, onBack = { nav.popBackStack() }) }
            }
            composable(Route.SETTINGS) {
                Sub {
                    SettingsScreen(
                        vm, onBack = { nav.popBackStack() },
                        onDeveloper = { nav.navigate(Route.DEVELOPER) },
                        onSetUpAgain = { nav.navigate(Route.SETUP) },
                    )
                }
            }
            composable(Route.DEVELOPER) { Sub { DeveloperScreen(vm, onBack = { nav.popBackStack() }) } }
            composable(Route.SETUP) {
                Sub { SetupScreen(vm, onDone = { nav.toTab(Tab.Talk.route) }) }
            }
        }
    }
}

/** Screens without the tab bar keep clear of the system navigation bar themselves. */
@Composable
private fun Sub(content: @Composable () -> Unit) {
    Box(Modifier.fillMaxSize().navigationBarsPadding()) { content() }
}

@Composable
private fun SignsCameraGate(content: @Composable () -> Unit) {
    CameraGate(
        Icons.Filled.School, stringResource(R.string.gate_signs_title), stringResource(R.string.gate_signs_body),
        listOf(Icons.Filled.Lock to stringResource(R.string.gate_point_private), Icons.Filled.Lock to stringResource(R.string.gate_point_keypoints)),
        content,
    )
}

/** Switch tabs keeping each tab's own back stack, and clear anything opened over the tabs. */
private fun NavController.toTab(route: String) {
    navigate(route) {
        popUpTo(graph.findStartDestination().id) { saveState = true }
        launchSingleTop = true
        restoreState = true
    }
}
