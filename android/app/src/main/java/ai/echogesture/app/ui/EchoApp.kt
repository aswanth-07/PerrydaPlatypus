package ai.echogesture.app.ui

import ai.echogesture.app.MainViewModel
import ai.echogesture.app.R
import ai.echogesture.app.ui.clips.ClipsScreen
import ai.echogesture.app.data.CaptionText
import ai.echogesture.app.ui.components.CameraGate
import ai.echogesture.app.ui.components.Haptics
import ai.echogesture.app.ui.components.LocalCaption
import ai.echogesture.app.ui.components.LocalHaptics
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalHapticFeedback
import ai.echogesture.app.ui.components.SubScreen
import ai.echogesture.app.ui.history.ConversationScreen
import ai.echogesture.app.ui.history.HistoryScreen
import ai.echogesture.app.ui.live.LiveScreen
import ai.echogesture.app.ui.settings.DeveloperScreen
import ai.echogesture.app.ui.settings.SettingsScreen
import ai.echogesture.app.ui.setup.SetupScreen
import ai.echogesture.app.ui.teach.LearnHubScreen
import ai.echogesture.app.ui.teach.LearnScreen
import ai.echogesture.app.ui.teach.LessonScreen
import ai.echogesture.app.ui.teach.LessonsScreen
import ai.echogesture.app.ui.teach.TeachScreen
import androidx.compose.foundation.layout.Box
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.calculateStartPadding
import androidx.compose.foundation.layout.displayCutout
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.ui.Alignment
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import ai.echogesture.app.ui.components.FloatingTabBar
import ai.echogesture.app.ui.components.LocalTabBarInset
import ai.echogesture.app.ui.components.TabItem
import ai.echogesture.app.ui.theme.Motion
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.FrontHand
import androidx.compose.material.icons.rounded.History
import androidx.compose.material.icons.rounded.Lock
import androidx.compose.material.icons.rounded.School
import androidx.compose.material.icons.rounded.Speed
import androidx.compose.material.icons.rounded.Videocam
import androidx.compose.material.icons.outlined.History
import androidx.compose.material.icons.outlined.School
import androidx.compose.material.icons.outlined.Videocam
import androidx.compose.material3.MaterialTheme
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.displayCutoutPadding
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.ui.platform.LocalWindowInfo
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

/**
 * The three tabs, in the bar's order: Learn, Talk in the middle (where the app opens), History. Everything else
 * (Settings, a conversation, Teach…) opens over them with a back arrow.
 */
private enum class Tab(val route: String, val label: Int, val icon: ImageVector, val selectedIcon: ImageVector) {
    Learn("learn", R.string.tab_learn, Icons.Outlined.School, Icons.Rounded.School),
    Talk("talk", R.string.tab_talk, Icons.Outlined.Videocam, Icons.Rounded.Videocam),
    History("history", R.string.tab_history, Icons.Outlined.History, Icons.Rounded.History),
}

private fun tabIndex(route: String?) = Tab.entries.indexOfFirst { it.route == route }

private object Route {
    const val SETTINGS = "settings"
    const val DEVELOPER = "developer"
    const val SETUP = "setup"
    const val PRACTICE = "practice"
    const val LESSONS = "lessons"
    const val LESSON = "lesson/{id}"
    fun lesson(id: String) = "lesson/$id"
    const val TEACH = "teach"
    const val VIDEO = "video"
    const val CONVERSATION = "conversation/{start}"
    fun conversation(start: Long) = "conversation/$start"
}

@Composable
fun EchoApp(vm: MainViewModel = viewModel()) {
    val loaded by vm.settingsLoaded.collectAsStateWithLifecycle()
    val settings by vm.settings.collectAsStateWithLifecycle()
    val teach by vm.engine.teach.collectAsStateWithLifecycle()
    // How glosses are written everywhere: taught signs are names, written in title case.
    val taught = remember(teach.signs) { teach.signs.map { it.word.uppercase() }.toSet() }
    val caption = remember(taught, settings.glossCaps) { { gloss: String -> CaptionText.show(gloss, taught, settings.glossCaps) } }
    val haptics = Haptics(LocalHapticFeedback.current, settings.vibrate)
    // The surface gives every screen, including setup outside the tabs' scaffold, the app's background and text colour.
    CompositionLocalProvider(LocalCaption provides caption, LocalHaptics provides haptics) {
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
}

@Composable
private fun Tabs(vm: MainViewModel) {
    val nav = rememberNavController()
    val entry by nav.currentBackStackEntryAsState()
    val current = entry?.destination?.route
    val settings by vm.settings.collectAsStateWithLifecycle()
    val onTab = Tab.entries.any { it.route == current }
    // Landscape keeps the height for the camera: the bar stands on its side at the left.
    val wide = LocalWindowInfo.current.containerSize.let { it.width > it.height }
    val navBars = WindowInsets.navigationBars.asPaddingValues()
    val cutout = WindowInsets.displayCutout.asPaddingValues()
    val layout = LocalLayoutDirection.current
    val inset = when {
        !onTab -> PaddingValues(0.dp)
        wide -> PaddingValues(start = cutout.calculateStartPadding(layout) + BAR_REACH)
        else -> PaddingValues(bottom = navBars.calculateBottomPadding() + BAR_REACH)
    }
    val items = Tab.entries.map { TabItem(it.route, stringResource(it.label), it.icon, it.selectedIcon) }

    Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
      CompositionLocalProvider(LocalTabBarInset provides inset) {
        NavHost(
            nav, startDestination = Tab.Talk.route, modifier = Modifier.fillMaxSize(),
            // Between tabs the screens slide toward the tab picked; a screen opened over a tab slides in from the
            // right while the tab recedes a little, and goes back the same way.
            enterTransition = {
                val from = tabIndex(initialState.destination.route)
                val to = tabIndex(targetState.destination.route)
                if (from >= 0 && to >= 0) slideInHorizontally(SLIDE) { w -> (if (to > from) w else -w) / 4 } + fadeIn(FADE)
                else slideInHorizontally(SLIDE) { it / 3 } + fadeIn(FADE)
            },
            exitTransition = {
                val from = tabIndex(initialState.destination.route)
                val to = tabIndex(targetState.destination.route)
                if (from >= 0 && to >= 0) slideOutHorizontally(SLIDE) { w -> (if (to > from) -w else w) / 4 } + fadeOut(FADE)
                else scaleOut(SCALE, targetScale = 0.96f) + fadeOut(FADE)
            },
            popEnterTransition = { scaleIn(SCALE, initialScale = 0.96f) + fadeIn(FADE) },
            popExitTransition = { slideOutHorizontally(SLIDE) { it / 3 } + fadeOut(FADE) },
        ) {
            val openSettings = { nav.navigate(Route.SETTINGS) { launchSingleTop = true } }
            composable(Tab.Talk.route) {
                CameraGate(
                    Icons.Rounded.FrontHand, stringResource(R.string.gate_live_title), stringResource(R.string.gate_live_body),
                    listOf(Icons.Rounded.Lock to stringResource(R.string.gate_point_private), Icons.Rounded.Speed to stringResource(R.string.gate_point_realtime)),
                ) { LiveScreen(vm, onSettings = openSettings, onHistory = { nav.toTab(Tab.History.route) }) }
            }
            composable(Tab.Learn.route) {
                LearnHubScreen(
                    vm,
                    onLesson = { nav.navigate(Route.lesson(it)) },
                    onLessons = { nav.navigate(Route.LESSONS) },
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
                Sub { SubScreen(stringResource(R.string.screen_practise_any_sign), nav::popBackStack) { SignsCameraGate { LearnScreen(vm) } } }
            }
            composable(Route.LESSONS) {
                Sub { LessonsScreen(vm, onOpen = { nav.navigate(Route.lesson(it)) }, onBack = { nav.popBackStack() }) }
            }
            composable(Route.LESSON, arguments = listOf(navArgument("id") { type = NavType.StringType })) { e ->
                Sub {
                    LessonScreen(
                        vm, e.arguments?.getString("id").orEmpty(), onBack = { nav.popBackStack() },
                        onNext = { next -> nav.navigate(Route.lesson(next)) { popUpTo(Route.LESSON) { inclusive = true } } },
                        gate = { content -> SignsCameraGate(content) },
                    )
                }
            }
            composable(Route.TEACH) {
                Sub { SubScreen(stringResource(R.string.screen_teach_the_app_a_sign), nav::popBackStack) { SignsCameraGate { TeachScreen(vm) } } }
            }
            composable(Route.VIDEO) {
                Sub { SubScreen(stringResource(R.string.screen_identify_a_sign_in_a), nav::popBackStack) { ClipsScreen(vm, showTimings = settings.developer) } }
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
        // The floating bar, over the content (the camera runs underneath it on Talk).
        AnimatedVisibility(
            onTab && !wide,
            modifier = Modifier.align(Alignment.BottomCenter).navigationBarsPadding().padding(bottom = 14.dp),
            enter = slideInVertically(SLIDE_IN) { it } + fadeIn(FADE), exit = slideOutVertically(SLIDE_IN) { it } + fadeOut(FADE),
        ) {
            FloatingTabBar(items, current, onSelect = { nav.toTab(it) })
        }
        AnimatedVisibility(
            onTab && wide,
            modifier = Modifier.align(Alignment.CenterStart).displayCutoutPadding().padding(start = 12.dp),
            enter = slideInHorizontally(SLIDE_IN) { -it } + fadeIn(FADE), exit = slideOutHorizontally(SLIDE_IN) { -it } + fadeOut(FADE),
        ) {
            FloatingTabBar(items, current, onSelect = { nav.toTab(it) }, vertical = true)
        }
    }
}

/** How far the floating bar (64 dp) and its margin reach into a tab's screen. */
private val BAR_REACH = 84.dp
private val SLIDE = tween<IntOffset>(Motion.MEDIUM + 50, easing = Motion.Standard)
private val SLIDE_IN = tween<IntOffset>(Motion.MEDIUM, easing = Motion.Enter)
private val FADE = tween<Float>(Motion.MEDIUM)
private val SCALE = tween<Float>(Motion.MEDIUM + 50, easing = Motion.Standard)

/** Screens without the tab bar keep clear of the system navigation bar themselves. */
@Composable
private fun Sub(content: @Composable () -> Unit) {
    Box(Modifier.fillMaxSize().navigationBarsPadding()) { content() }
}

@Composable
private fun SignsCameraGate(content: @Composable () -> Unit) {
    CameraGate(
        Icons.Rounded.School, stringResource(R.string.gate_signs_title), stringResource(R.string.gate_signs_body),
        listOf(Icons.Rounded.Lock to stringResource(R.string.gate_point_private), Icons.Rounded.Lock to stringResource(R.string.gate_point_keypoints)),
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
