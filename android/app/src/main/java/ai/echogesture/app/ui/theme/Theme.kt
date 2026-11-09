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
    val Glass = Color(0xB30A0E13)       // panels over live video
    val GlassLight = Color(0x66121821)
}

private val Scheme = darkColorScheme(
    primary = Palette.Cyan,
    onPrimary = Palette.CyanDeep,
    primaryContainer = Color(0xFF0E3A47),
    onPrimaryContainer = Color(0xFFBDEBFA),
    secondary = Palette.Amber,
    onSecondary = Color(0xFF3A2500),
    secondaryContainer = Color(0xFF3B2C10),
    onSecondaryContainer = Color(0xFFFFDDAA),
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
