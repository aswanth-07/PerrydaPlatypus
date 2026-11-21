package ai.echogesture.app.ui.teach

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Pause
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Replay
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import ai.echogesture.app.R
import ai.echogesture.app.data.Demo
import ai.echogesture.app.data.SignToken
import ai.echogesture.app.ui.components.LocalCaption
import ai.echogesture.app.ui.theme.Motion
import ai.echogesture.app.ui.theme.Palette

/**
 * Plays a translated sequence of real-signer ASL [Demo] animations one after another (100% offline, from demos.bin).
 * Powers both "Reply in Sign" on the Talk tab and the Text-to-Sign translator in the Sign Dictionary.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun SignSequencePlayer(
    tokens: List<SignToken>,
    demos: Map<String, Demo>,
    modifier: Modifier = Modifier,
    figureHeight: Dp = 240.dp,
    mirror: Boolean = false,
    darkContainer: Color = Palette.Panel,
    onSelectSign: ((String) -> Unit)? = null,
    onActiveSignChange: ((String?) -> Unit)? = null,
) {
    val show = LocalCaption.current
    // Indices in `tokens` that have a matching real-signer demonstration.
    val playable = remember(tokens, demos) {
        tokens.mapIndexedNotNull { idx, tok -> tok.gloss?.takeIf { it in demos }?.let { idx to it } }
    }
    var seqPos by remember(playable) { mutableIntStateOf(0) }
    var playing by remember(playable) { mutableStateOf(true) }
    var slow by remember { mutableStateOf(false) }
    var zoom by remember { mutableStateOf(false) }
    var step by remember(playable) { mutableFloatStateOf(0f) }
    var progress by remember(playable) { mutableFloatStateOf(0f) }

    val activePair = playable.getOrNull(seqPos.coerceIn(0, maxOf(playable.lastIndex, 0)))
    val activeDemo = activePair?.second?.let { demos[it] }
    LaunchedEffect(activeDemo?.word) { onActiveSignChange?.invoke(activeDemo?.word) }

    // Frame-accurate playback across the sequence: smoothly plays each sign, pauses briefly between signs, then loops.
    LaunchedEffect(playable, seqPos, playing, slow) {
        val d = activeDemo ?: return@LaunchedEffect
        if (!playing) return@LaunchedEffect
        val playMs = (d.durationMs.coerceIn(600, 3200) * (if (slow) 2.0f else 1f))
        val holdMs = if (playable.size > 1) 420f else 650f
        val totalMs = playMs + holdMs
        var elapsedMs = (progress * totalMs).coerceIn(0f, totalMs)
        var lastNs = withFrameNanos { it }
        while (true) {
            val nowNs = withFrameNanos { it }
            val dtMs = ((nowNs - lastNs) / 1_000_000f).coerceIn(0f, 100f)
            lastNs = nowNs
            elapsedMs += dtMs
            if (elapsedMs >= totalMs) {
                progress = 0f
                step = 0f
                if (playable.size > 1) {
                    seqPos = (seqPos + 1) % playable.size
                    return@LaunchedEffect
                } else {
                    elapsedMs = 0f
                }
            } else {
                val frac = (elapsedMs / playMs).coerceIn(0f, 1f)
                step = frac * (d.frames - 1f)
                progress = (elapsedMs / totalMs).coerceIn(0f, 1f)
            }
        }
    }

    Column(
        modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(22.dp))
            .background(darkContainer)
            .border(1.dp, Palette.Hairline, RoundedCornerShape(22.dp))
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        if (activeDemo == null) {
            Text(
                stringResource(R.string.sign_reply_no_matches),
                style = MaterialTheme.typography.bodyMedium,
                color = Palette.TextMuted,
                modifier = Modifier.padding(8.dp),
            )
            return@Column
        }

        // Top bar: currently signing word badge + sequence counter + play/pause & replay.
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier
                    .clip(RoundedCornerShape(50))
                    .background(MaterialTheme.colorScheme.primaryContainer)
                    .padding(horizontal = 12.dp, vertical = 5.dp),
            ) {
                Text(
                    show(activeDemo.word),
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
            Spacer(Modifier.width(8.dp))
            Text(
                stringResource(R.string.sign_reply_counter, seqPos + 1, playable.size),
                style = MaterialTheme.typography.labelMedium,
                color = Palette.TextMuted,
                modifier = Modifier.weight(1f),
            )
            IconButton(
                onClick = { playing = !playing },
                modifier = Modifier.size(38.dp),
            ) {
                Icon(
                    if (playing) Icons.Rounded.Pause else Icons.Rounded.PlayArrow,
                    contentDescription = stringResource(if (playing) R.string.talk_pause else R.string.talk_resume),
                    tint = Palette.Text,
                )
            }
            IconButton(
                onClick = { seqPos = 0; progress = 0f; step = 0f; playing = true },
                modifier = Modifier.size(38.dp),
            ) {
                Icon(Icons.Rounded.Replay, contentDescription = stringResource(R.string.sign_reply_replay), tint = Palette.Text)
            }
        }

        // Real-signer animated figure.
        Box(
            Modifier
                .fillMaxWidth()
                .height(figureHeight)
                .clip(RoundedCornerShape(16.dp))
                .background(Palette.Onyx.copy(alpha = 0.7f)),
        ) {
            DemoFigure(
                demo = activeDemo,
                slow = slow,
                mirror = mirror,
                zoom = zoom,
                modifier = Modifier.fillMaxWidth().height(figureHeight).padding(10.dp),
                stepOverride = step,
            )
            // Subtle progress bar along the bottom edge of the stage.
            Box(
                Modifier
                    .align(Alignment.BottomStart)
                    .fillMaxWidth()
                    .height(3.dp)
                    .background(Color.White.copy(alpha = 0.10f)),
            ) {
                Box(
                    Modifier
                        .fillMaxWidth(progress.coerceIn(0f, 1f))
                        .height(3.dp)
                        .background(MaterialTheme.colorScheme.primary),
                )
            }
        }

        // Interactive sentence pills: tap any highlighted word to jump straight to its sign.
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            tokens.forEachIndexed { tokenIdx, tok ->
                val playIndex = playable.indexOfFirst { it.first == tokenIdx }
                val isPlayable = playIndex >= 0
                val isActive = playIndex == seqPos
                val bg by animateColorAsState(
                    when {
                        isActive -> MaterialTheme.colorScheme.primary
                        isPlayable -> Palette.PanelHigh
                        else -> Color.Transparent
                    },
                    tween(Motion.SHORT),
                    label = "pillBg",
                )
                val fg = when {
                    isActive -> MaterialTheme.colorScheme.onPrimary
                    isPlayable -> Palette.Text
                    else -> Palette.TextMuted.copy(alpha = 0.65f)
                }
                Box(
                    Modifier
                        .clip(RoundedCornerShape(50))
                        .background(bg)
                        .then(
                            if (isPlayable && !isActive) Modifier.border(1.dp, Palette.Hairline, RoundedCornerShape(50))
                            else Modifier
                        )
                        .clickable(enabled = isPlayable) {
                            seqPos = playIndex
                            progress = 0f
                            step = 0f
                            playing = true
                            tok.gloss?.let { onSelectSign?.invoke(it) }
                        }
                        .heightIn(min = 32.dp)
                        .padding(horizontal = if (isPlayable) 12.dp else 4.dp, vertical = 5.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        tok.spoken,
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = if (isActive) FontWeight.Bold else FontWeight.Medium,
                        color = fg,
                        maxLines = 1,
                    )
                }
            }
        }

        // Playback speed & hand-zoom chips (wraps cleanly at large accessibility font scales).
        FlowRow(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
            itemVerticalAlignment = Alignment.CenterVertically,
        ) {
            FilterChip(
                selected = slow,
                onClick = { slow = !slow },
                label = { Text(stringResource(R.string.practice_slow), maxLines = 1) },
            )
            FilterChip(
                selected = zoom,
                onClick = { zoom = !zoom },
                label = { Text(stringResource(R.string.practice_hands), maxLines = 1) },
            )
            Text(
                stringResource(R.string.sign_reply_offline_tag),
                style = MaterialTheme.typography.labelSmall,
                color = Palette.TextMuted,
                modifier = Modifier.padding(start = 4.dp),
            )
        }
    }
}
