package ai.echogesture.app.ui.components

import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.hapticfeedback.HapticFeedback
import androidx.compose.ui.hapticfeedback.HapticFeedbackType

/** Writes a recognised gloss for the screen: lower case, names in title case, or capitals (Settings). */
val LocalCaption = staticCompositionLocalOf<(String) -> String> { { it } }

/**
 * Short vibrations for moments the eyes may miss: a caption word settling, a correction, recording starting and
 * stopping, a countdown, a sign learned. All of them stop when Settings > Vibration is off.
 */
class Haptics(private val feedback: HapticFeedback?, private val enabled: Boolean) {
    fun tick() = play(HapticFeedbackType.SegmentTick)
    fun confirm() = play(HapticFeedbackType.Confirm)
    fun start() = play(HapticFeedbackType.GestureThresholdActivate)
    fun end() = play(HapticFeedbackType.GestureEnd)

    private fun play(type: HapticFeedbackType) {
        if (enabled) feedback?.performHapticFeedback(type)
    }
}

val LocalHaptics = staticCompositionLocalOf { Haptics(null, false) }
