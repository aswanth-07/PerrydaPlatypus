package ai.echogesture.app.ui.theme

import ai.echogesture.app.R
import androidx.compose.animation.core.CubicBezierEasing
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.spring
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontVariation
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** A dark HUD palette: near-black ink, one cyan accent for "the system speaking", amber for attention. */
object Palette {
    val Ink = Color(0xFF0A0E13)
    val Panel = Color(0xFF121821)
    val PanelHigh = Color(0xFF1A2230)
    val Line = Color(0xFF2A3544)        // decorative dividers only
    // Borders of controls (switch thumbs, outlined and segmented buttons): at least 3:1 on every surface here.
    val Outline = Color(0xFF66768C)
    val Text = Color(0xFFE8EEF5)
    val TextMuted = Color(0xFF93A1B2)
    val Cyan = Color(0xFF5AD2F4)
    val CyanDeep = Color(0xFF00313D)
    val Amber = Color(0xFFFFB547)
    val Green = Color(0xFF6BE38A)
    val Red = Color(0xFFFF6B6B)
    // Panels over live video: 88% ink keeps muted text above 4.5:1 even over a white wall (70% gave 2.8:1).
    val Glass = Color(0xE00A0E13)
    val GlassLight = Color(0x66121821)
    val Hairline = Color(0x24FFFFFF)    // the edge of a glass panel, so it reads as a pane over dark video too
    val Reply = Color(0xFFFFD58F)       // the hearing person's words: warm, distinct from the signer's cyan
    val ReplyDeep = Color(0xFF2E2210)
}

/** A glass panel over video: dark fill with a one-pixel light edge. */
fun Modifier.glass(shape: Shape, color: Color = Palette.Glass): Modifier =
    clip(shape).background(color, shape).border(Dp.Hairline, Palette.Hairline, shape)

/** Durations and easings shared by every animation, so the app moves with one rhythm. */
object Motion {
    const val SHORT = 150     // small state changes: a chip, a tick
    const val MEDIUM = 250    // things entering and leaving
    const val LONG = 400      // larger moves: a check drawing on, a countdown number
    val Standard = CubicBezierEasing(0.2f, 0f, 0f, 1f)
    val Enter = CubicBezierEasing(0.05f, 0.7f, 0.1f, 1f)
    val Exit = CubicBezierEasing(0.3f, 0f, 0.8f, 0.15f)
    /** A new caption word settles with a slight overshoot, so the eye catches it without a flash. */
    val wordSpring = spring(dampingRatio = 0.6f, stiffness = Spring.StiffnessMediumLow, visibilityThreshold = IntOffset(1, 1))
}

/** Atkinson Hyperlegible (Braille Institute, OFL): letterforms that stay distinct at a glance, for captions. */
val Atkinson = FontFamily(
    Font(R.font.atkinson_hyperlegible_regular, FontWeight.Normal),
    Font(R.font.atkinson_hyperlegible_bold, FontWeight.Bold),
)

private fun manrope(weight: Int) = Font(
    R.font.manrope_variable, FontWeight(weight), variationSettings = FontVariation.Settings(FontVariation.weight(weight)),
)

/** Manrope (variable, OFL) for headings and titles. */
val Manrope = FontFamily(manrope(400), manrope(500), manrope(600), manrope(700), manrope(800))

private val Scheme = darkColorScheme(
    primary = Palette.Cyan,
    onPrimary = Palette.CyanDeep,
    primaryContainer = Color(0xFF0E3A47),
    onPrimaryContainer = Color(0xFFBDEBFA),
    // Selections, slider tracks and chips use a quiet cyan tint; amber is kept for things that need attention.
    secondary = Color(0xFF9CCFDD),
    onSecondary = Color(0xFF00313D),
    secondaryContainer = Color(0xFF173742),
    onSecondaryContainer = Color(0xFFCDEFF8),
    tertiary = Palette.Green,
    background = Palette.Ink,
    onBackground = Palette.Text,
    surface = Palette.Ink,
    onSurface = Palette.Text,
    surfaceVariant = Palette.PanelHigh,
    onSurfaceVariant = Palette.TextMuted,
    surfaceContainerLowest = Palette.Ink,
    surfaceContainerLow = Palette.Panel,
    surfaceContainer = Palette.Panel,
    surfaceContainerHigh = Palette.PanelHigh,
    surfaceContainerHighest = Color(0xFF222C3B),
    outline = Palette.Outline,
    outlineVariant = Color(0xFF1F2835),
    error = Palette.Red,
)

private val AppTypography = Typography().let { t ->
    fun TextStyle.heading(weight: FontWeight) = copy(fontFamily = Manrope, fontWeight = weight)
    t.copy(
        displayLarge = t.displayLarge.heading(FontWeight.Bold),
        displayMedium = t.displayMedium.heading(FontWeight.Bold),
        displaySmall = t.displaySmall.heading(FontWeight.Bold).copy(letterSpacing = 0.sp),
        headlineLarge = t.headlineLarge.heading(FontWeight.Bold),
        headlineMedium = t.headlineMedium.heading(FontWeight.Bold),
        headlineSmall = t.headlineSmall.heading(FontWeight.Bold),
        titleLarge = t.titleLarge.heading(FontWeight.SemiBold),
        titleMedium = t.titleMedium.heading(FontWeight.Bold),
        titleSmall = t.titleSmall.heading(FontWeight.SemiBold),
        labelLarge = t.labelLarge.copy(fontWeight = FontWeight.SemiBold),
    )
}

/** Caption text over video: large, bold Atkinson Hyperlegible, generous line height for glanceable reading. */
val CaptionStyle = TextStyle(fontFamily = Atkinson, fontSize = 30.sp, lineHeight = 38.sp, fontWeight = FontWeight.Bold)

/** The hearing person's reply, shown for the signer to read: as large as a caption, in the regular weight. */
val ReplyStyle = TextStyle(fontFamily = Atkinson, fontSize = 28.sp, lineHeight = 36.sp, fontWeight = FontWeight.Normal)

/** Signed and spoken lines in History and the conversation sheet. */
val TranscriptStyle = TextStyle(fontFamily = Atkinson, fontSize = 22.sp, lineHeight = 30.sp, fontWeight = FontWeight.Normal)

@Composable
fun EchoTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = Scheme,
        typography = AppTypography,
        shapes = Shapes(
            small = RoundedCornerShape(10.dp),
            medium = RoundedCornerShape(16.dp),
            large = RoundedCornerShape(24.dp),
        ),
        content = content,
    )
}
