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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.geometry.Offset
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

/**
 * EchoGesture's palette: Onyx black, Dark Azure as the brand colour, Azure and Light Azure for accents, Paper
 * White for text, and the branding gradient (black to azure) for hero surfaces. Green, amber and red are kept
 * only as status signals: ready, needs attention, error.
 */
object Palette {
    // Brand colours.
    val Onyx = Color(0xFF000000)          // the background
    val Onyx2 = Color(0xFF141414)         // panels
    val Charcoal = Color(0xFF2B2B2B)      // raised panels
    val DarkAzure = Color(0xFF00638E)     // filled highlights: the selected tab, chosen options, replies
    val DeepAzure = Color(0xFF004A6B)
    val Azure = Color(0xFF8CB9CC)         // accent text and icons (9.6:1 on Onyx; Dark Azure is only 3.1:1)
    val LightAzure = Color(0xFFBFD8E3)
    val PaperWhite = Color(0xFFFFFFFF)

    // Roles used across the screens.
    val Ink = Onyx
    val Panel = Onyx2
    val PanelHigh = Color(0xFF1F1F1F)
    val Line = Charcoal                   // decorative dividers only
    // Borders of controls (switch thumbs, outlined and segmented buttons): at least 3:1 on Onyx, panels and Charcoal.
    val Outline = Color(0xFF767676)
    val Text = PaperWhite
    val TextMuted = Color(0xFFA6A6A6)     // 5.8:1 on Charcoal, 8.6:1 on Onyx
    val Cyan = Azure
    val CyanDeep = Color(0xFF002536)
    val Amber = Color(0xFFFFB547)         // status: needs attention
    val Green = Color(0xFF6BE38A)         // status: ready, right
    val Red = Color(0xFFFF6B6B)           // status: error, recording
    // Panels over live video: 88% Onyx keeps muted text above 4.5:1 even over a white wall.
    val Glass = Color(0xE0141414)
    val GlassLight = Color(0x66141414)
    val Hairline = Color(0x29FFFFFF)      // the edge of a glass panel, so it reads as a pane over dark video too
    val Reply = PaperWhite                // the hearing person's words: white on Dark Azure (6.8:1)
    val ReplyDeep = DarkAzure
    val FirstHand = PaperWhite            // the two tracked hands: white and azure, on video and in the figure

    /** The branding gradient, left to right: Onyx into Dark Azure into Azure. */
    val Brand = Brush.linearGradient(listOf(Onyx, DeepAzure, DarkAzure, Azure))
    /** The same, kept dark enough for white text across its whole width. */
    val BrandDeep = Brush.linearGradient(
        0f to Onyx, 0.55f to DeepAzure, 1f to DarkAzure,
        start = Offset.Zero, end = Offset(Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY),
    )
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
    primary = Palette.Azure,
    onPrimary = Palette.CyanDeep,
    primaryContainer = Palette.DarkAzure,
    onPrimaryContainer = Palette.PaperWhite,
    secondary = Palette.LightAzure,
    onSecondary = Color(0xFF00334A),
    secondaryContainer = Palette.DeepAzure,
    onSecondaryContainer = Palette.PaperWhite,
    tertiary = Palette.Green,
    background = Palette.Onyx,
    onBackground = Palette.Text,
    surface = Palette.Onyx,
    onSurface = Palette.Text,
    surfaceVariant = Palette.Charcoal,
    onSurfaceVariant = Palette.TextMuted,
    surfaceContainerLowest = Palette.Onyx,
    surfaceContainerLow = Palette.Onyx2,
    surfaceContainer = Color(0xFF1A1A1A),
    surfaceContainerHigh = Color(0xFF222222),
    surfaceContainerHighest = Palette.Charcoal,
    inverseSurface = Palette.LightAzure,
    inverseOnSurface = Palette.Onyx,
    outline = Palette.Outline,
    outlineVariant = Color(0xFF262626),
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
