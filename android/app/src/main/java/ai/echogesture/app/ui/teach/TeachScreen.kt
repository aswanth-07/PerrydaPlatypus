package ai.echogesture.app.ui.teach

import android.app.Activity
import android.view.WindowManager
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Cameraswitch
import androidx.compose.material.icons.rounded.Delete
import androidx.compose.material.icons.rounded.Edit
import androidx.compose.material.icons.rounded.FiberManualRecord
import androidx.compose.material.icons.rounded.Lock
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.data.TaughtSign
import ai.echogesture.app.engine.TeachExample
import ai.echogesture.app.ui.components.LocalCaption
import ai.echogesture.app.ui.components.LocalHaptics
import ai.echogesture.app.ui.components.SectionTitle
import ai.echogesture.app.ui.theme.Manrope
import ai.echogesture.app.ui.theme.Motion
import ai.echogesture.app.ui.theme.glass
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.animation.togetherWith
import ai.echogesture.app.ui.components.StatusPill
import ai.echogesture.app.ui.live.CameraPreview
import ai.echogesture.app.ui.live.HandOverlay
import ai.echogesture.app.ui.theme.Palette
import ai.echogesture.signcore.Embeddings
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

private const val MAX_EXAMPLES = 5
private const val EXAMPLE_MS = 2500L
private const val MIN_HAND_FRACTION = 0.3f
private const val CONSISTENT = 0.5f   // an example this far from the earlier ones is probably a different movement

/**
 * Teach the app a sign it does not know: a word and 3 to 5 short examples from the camera. The examples' pooled
 * embeddings form a prototype; live windows and clips close enough to it are captioned and spoken as that word.
 */
@Composable
fun TeachScreen(vm: MainViewModel) {
    val live by vm.live.collectAsStateWithLifecycle()
    val teach by vm.engine.teach.collectAsStateWithLifecycle()
    val settings by vm.settings.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var word by remember { mutableStateOf("") }
    val examples = remember { mutableStateListOf<TeachExample>() }
    var pending by remember { mutableStateOf<TeachExample?>(null) }
    var countdown by remember { mutableStateOf<Int?>(null) }
    var busy by remember { mutableStateOf(false) }
    var message by remember { mutableStateOf<String?>(null) }
    var renaming by remember { mutableStateOf<TaughtSign?>(null) }
    var deleting by remember { mutableStateOf<TaughtSign?>(null) }
    val haptics = LocalHaptics.current
    val show = LocalCaption.current

    DisposableEffect(Unit) {
        vm.engine.setPaused(false)   // a pause left on the Live tab would starve the recorder of frames
        vm.engine.acquireLive()
        val window = (context as? Activity)?.window
        window?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onDispose {
            vm.engine.releaseLive()
            window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }

    fun record() = scope.launch {
        busy = true
        message = null
        pending = null
        try {
            for (n in 3 downTo 1) { countdown = n; haptics.tick(); delay(700) }
            countdown = null
            haptics.start()
            val ex = vm.engine.recordExample(EXAMPLE_MS)
            haptics.end()
            val emb = ex.embedding
            val earlier = examples.mapNotNull { it.embedding }
            when {
                ex.handFraction < MIN_HAND_FRACTION -> message = "No hands were seen in that example. Keep your hands in view and record it again."
                emb == null -> message = "This model cannot learn new signs."
                earlier.isNotEmpty() && Embeddings.cosine(emb, Embeddings.prototype(earlier)) < CONSISTENT -> {
                    pending = ex
                    message = "That looked different from the earlier examples (match %.2f). Keep it, or record it again?"
                        .format(Embeddings.cosine(emb, Embeddings.prototype(earlier)))
                }
                else -> examples += ex
            }
        } catch (e: Exception) {
            message = e.message ?: "Recording failed"
        } finally {
            countdown = null
            busy = false
        }
    }

    Column(Modifier.fillMaxSize()) {
        // Camera, with the countdown and recording progress on top.
        Box(Modifier.fillMaxWidth().weight(0.45f).padding(horizontal = 12.dp, vertical = 8.dp).clip(RoundedCornerShape(24.dp)).background(Color.Black)) {
            CameraPreview(backCamera = settings.backCamera, analyzer = vm.engine::analyze, modifier = Modifier.fillMaxSize())
            if (settings.showLandmarks) HandOverlay(live, mirror = !settings.backCamera, skeleton = settings.developer && settings.skeleton, modifier = Modifier.fillMaxSize())
            StatusPill(
                when {
                    live.recording != null -> "Recording"
                    countdown != null -> "Get ready"
                    live.handsVisible -> "Hands in view"
                    live.bodyVisible -> "Show your hands"
                    else -> "Step back"
                },
                when {
                    live.recording != null -> Palette.Red
                    live.handsVisible -> Palette.Green
                    else -> Palette.Amber
                },
                Modifier.align(Alignment.TopStart).padding(12.dp),
            )
            FilledTonalIconButton(
                onClick = { vm.update { it.copy(backCamera = !it.backCamera) } },
                modifier = Modifier.align(Alignment.TopEnd).padding(12.dp).size(48.dp).glass(CircleShape),
                colors = IconButtonDefaults.filledTonalIconButtonColors(containerColor = Color.Transparent, contentColor = Palette.Text),
            ) { Icon(Icons.Rounded.Cameraswitch, "Switch camera") }
            // 3, 2, 1: each number lands large and fades as the next arrives.
            AnimatedContent(
                countdown, Modifier.align(Alignment.Center),
                transitionSpec = {
                    (scaleIn(tween(Motion.MEDIUM, easing = Motion.Enter), initialScale = 1.6f) + fadeIn(tween(Motion.SHORT))) togetherWith
                        (scaleOut(tween(Motion.SHORT), targetScale = 0.7f) + fadeOut(tween(Motion.SHORT)))
                },
                label = "countdown",
            ) { n ->
                if (n != null) Text("$n", color = Color.White, fontSize = 96.sp, fontWeight = FontWeight.ExtraBold, fontFamily = Manrope)
            }
            live.recording?.let { p ->
                LinearProgressIndicator(progress = { p }, modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth().height(6.dp),
                    color = Palette.Red)
            }
            // Try it: what the live captions would say right now, so a saved sign can be checked at once.
            if (live.recording == null && countdown == null && live.lastWord != null) {
                Text(
                    "Recognised: ${show(live.lastWord!!)}" + if (live.lastTaught) "  (taught)" else "",
                    Modifier.align(Alignment.BottomStart).padding(12.dp).glass(RoundedCornerShape(10.dp))
                        .padding(horizontal = 12.dp, vertical = 6.dp),
                    color = if (live.lastTaught) Palette.Amber else Palette.Text, style = MaterialTheme.typography.titleMedium,
                )
            }
        }

        Column(Modifier.fillMaxWidth().weight(0.55f).verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 8.dp)) {
            if (!teach.supported) {
                Text(teach.note ?: "Loading the model…", style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                return@Column
            }
            Text(
                "Add a sign it doesn't know, like a name sign. Type the word, then sign it 3 to 5 times.",
                style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(14.dp))
            OutlinedTextField(
                value = word, onValueChange = { word = it.take(24) }, singleLine = true, enabled = !busy,
                label = { Text("Word to show and speak") },
                keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Characters),
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(Modifier.height(12.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Button(
                    onClick = { record() },
                    enabled = !busy && word.isNotBlank() && examples.size < MAX_EXAMPLES && live.running,
                    modifier = Modifier.height(48.dp),
                ) {
                    Icon(Icons.Rounded.FiberManualRecord, null, tint = Palette.Red)
                    Spacer(Modifier.width(8.dp))
                    Text(if (examples.isEmpty()) "Record example" else "Record example ${examples.size + 1}")
                }
                Spacer(Modifier.width(16.dp))
                // One dot per example: filled when recorded; three or more is green.
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.semantics(mergeDescendants = true) {
                    contentDescription = "${examples.size} of $MAX_EXAMPLES examples recorded"
                }) {
                    repeat(MAX_EXAMPLES) { i ->
                        // An empty slot is an outline in muted text colour (7:1), so all five read as places to fill.
                        val dot = Modifier.size(12.dp).clip(CircleShape)
                        Box(
                            if (i >= examples.size) dot.border(1.5.dp, Palette.TextMuted, CircleShape)
                            else dot.background(if (examples.size >= 3) Palette.Green else MaterialTheme.colorScheme.primary)
                        )
                    }
                }
            }
            message?.let {
                Spacer(Modifier.height(10.dp))
                Text(it, style = MaterialTheme.typography.bodyMedium, color = Palette.Amber)
            }
            pending?.let { ex ->
                Row {
                    TextButton(onClick = { examples += ex; pending = null; message = null }) { Text("Keep it") }
                    TextButton(onClick = { pending = null; message = null }) { Text("Discard") }
                }
            }
            Spacer(Modifier.height(12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Button(
                    onClick = {
                        scope.launch {
                            busy = true
                            val w = word
                            runCatching { vm.engine.saveTaught(w, examples.map { it.clip }) }
                                .onSuccess {
                                    message = "Saved \u201c${w.trim()}\u201d. Sign it now to try it: it is captioned and spoken like any other word."
                                    examples.clear()
                                    word = ""
                                }
                                .onFailure { message = it.message }
                            busy = false
                        }
                    },
                    enabled = !busy && word.isNotBlank() && examples.isNotEmpty(),
                ) { Text("Save sign") }
                OutlinedButton(onClick = { examples.clear(); pending = null; message = null }, enabled = !busy && examples.isNotEmpty()) {
                    Text("Start over")
                }
            }
            if (examples.size in 1..2) {
                Spacer(Modifier.height(6.dp))
                Text("Three or more examples make it much more reliable.", style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Spacer(Modifier.height(14.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Rounded.Lock, null, Modifier.size(16.dp), tint = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.width(8.dp))
                Text("Only hand, body and face keypoints are stored on this phone, never video.",
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }

            SectionTitle("Taught on this phone")
            if (teach.signs.isEmpty()) {
                Text("None yet.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            teach.signs.forEach { sign ->
                Card(Modifier.fillMaxWidth().padding(vertical = 4.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer)) {
                    Row(Modifier.padding(start = 16.dp, end = 4.dp, top = 6.dp, bottom = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(show(sign.word), style = MaterialTheme.typography.titleMedium, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            Text("${sign.examples} example" + if (sign.examples == 1) "" else "s",
                                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        IconButton(onClick = { renaming = sign }) { Icon(Icons.Rounded.Edit, "Rename ${sign.word}") }
                        IconButton(onClick = { deleting = sign }) { Icon(Icons.Rounded.Delete, "Delete ${sign.word}") }
                    }
                }
            }
            Spacer(Modifier.height(12.dp))
            Text(
                "In offline tests on signs a recogniser had never seen, 3 examples were enough to be right about 9 times in 10. " +
                    "If a taught sign shows up when it shouldn't, delete it and teach it again with four or five examples.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }

    renaming?.let { sign ->
        var name by remember(sign.id) { mutableStateOf(sign.word) }
        AlertDialog(
            onDismissRequest = { renaming = null },
            title = { Text("Rename sign") },
            text = {
                OutlinedTextField(name, { name = it.take(24) }, singleLine = true,
                    keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Characters))
            },
            confirmButton = {
                TextButton(onClick = {
                    val n = name
                    renaming = null
                    if (n.isNotBlank()) scope.launch { vm.engine.renameTaught(sign.id, n) }
                }) { Text("Rename") }
            },
            dismissButton = { TextButton(onClick = { renaming = null }) { Text("Cancel") } },
        )
    }
    deleting?.let { sign ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("Delete ${sign.word}?") },
            text = { Text("Its ${sign.examples} recorded examples are removed from this phone.") },
            confirmButton = {
                TextButton(onClick = { deleting = null; scope.launch { vm.engine.deleteTaught(sign.id) } }) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("Cancel") } },
        )
    }
}
