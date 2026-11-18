package ai.echogesture.app.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** A dark HUD palette: near-black ink, one cyan accent for "the system speaking", amber for attention. */
object Palette {
    val Ink = Color(0xFF0A0E13)
    val Panel = Color(0xFF121821)
    val PanelHigh = Color(0xFF1A2230)
    val Line = Color(0xFF2A3544)
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
    val Reply = Color(0xFFFFD58F)       // the hearing person's words: warm, distinct from the signer's cyan
    val ReplyDeep = Color(0xFF2E2210)
}

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
    outline = Palette.Line,
    outlineVariant = Color(0xFF1F2835),
    error = Palette.Red,
)

private val AppTypography = Typography().let { t ->
    t.copy(
        displaySmall = t.displaySmall.copy(fontWeight = FontWeight.Bold, letterSpacing = 0.sp),
        headlineMedium = t.headlineMedium.copy(fontWeight = FontWeight.SemiBold),
        headlineSmall = t.headlineSmall.copy(fontWeight = FontWeight.SemiBold),
        titleMedium = t.titleMedium.copy(fontWeight = FontWeight.SemiBold),
        labelLarge = t.labelLarge.copy(fontWeight = FontWeight.SemiBold),
    )
}

/** Caption text over video: large, heavy, with generous line height for glanceable reading. */
val CaptionStyle = TextStyle(fontSize = 30.sp, lineHeight = 38.sp, fontWeight = FontWeight.Bold)

/** The hearing person's reply, shown for the signer to read: as large as a caption, one weight lighter. */
val ReplyStyle = TextStyle(fontSize = 28.sp, lineHeight = 36.sp, fontWeight = FontWeight.SemiBold)

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
