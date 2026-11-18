package ai.echogesture.app.ui.teach

import android.app.Activity
import android.view.WindowManager
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Cameraswitch
import androidx.compose.material.icons.rounded.Close
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
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.data.CaptionText
import ai.echogesture.app.data.TaughtSign
import ai.echogesture.app.engine.LiveState
import ai.echogesture.app.engine.TeachExample
import ai.echogesture.app.ui.components.DrawnCheck
import ai.echogesture.app.ui.components.LocalCaption
import ai.echogesture.app.ui.components.LocalHaptics
import ai.echogesture.app.ui.components.SectionTitle
import ai.echogesture.app.ui.components.StatusPill
import ai.echogesture.app.ui.components.TrackingError
import ai.echogesture.app.ui.live.CameraPreview
import ai.echogesture.app.ui.live.HandOverlay
import ai.echogesture.app.ui.theme.Manrope
import ai.echogesture.app.ui.theme.Motion
import ai.echogesture.app.ui.theme.Palette
import ai.echogesture.app.ui.theme.glass
import ai.echogesture.signcore.Embeddings
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

private const val MIN_EXAMPLES = 3
private const val MAX_EXAMPLES = 5
private const val EXAMPLE_MS = 2500L
private const val MIN_HAND_FRACTION = 0.3f
private const val CONSISTENT = 0.5f   // an example this far from the earlier ones is probably a different movement

/** One recorded example, and whether it was recorded with the front camera (its replay is mirrored to match). */
private class Recorded(val example: TeachExample, val front: Boolean)

private enum class Step(val title: String) { NAME("Name"), RECORD("Record"), REVIEW("Review"), TRY("Try it") }

/**
 * Teach the app a sign it does not know, one step at a time: name it, record it 3 to 5 times (each example plays
 * back from its keypoints and can be deleted), look over the examples and save, then sign it once to see it
 * captioned. The examples' pooled embeddings form a prototype; live windows and clips close enough to it are
 * captioned and spoken as that word. Recording starts on the front camera, since people teach their own signs.
 */
@Composable
fun TeachScreen(vm: MainViewModel) {
    val live by vm.live.collectAsStateWithLifecycle()
    val teach by vm.engine.teach.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val haptics = LocalHaptics.current

    var step by rememberSaveable { mutableStateOf(Step.NAME) }
    var word by rememberSaveable { mutableStateOf("") }
    var front by rememberSaveable { mutableStateOf(true) }
    val examples = remember { mutableStateListOf<Recorded>() }
    var pending by remember { mutableStateOf<Recorded?>(null) }
    var countdown by remember { mutableStateOf<Int?>(null) }
    var busy by remember { mutableStateOf(false) }
    var message by remember { mutableStateOf<String?>(null) }

    DisposableEffect(Unit) {
        vm.engine.setPaused(false)   // a pause left on the Talk tab would starve the recorder of frames
        vm.engine.acquireLive()
        val window = (context as? Activity)?.window
        window?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onDispose {
            vm.engine.releaseLive()
            window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }
    // Recorded examples are not kept if Android ends the app; coming back to Review without them means recording.
    LaunchedEffect(Unit) { if (step == Step.REVIEW && examples.size < MIN_EXAMPLES) step = Step.RECORD }

    fun record() = scope.launch {
        busy = true
        message = null
        pending = null
        try {
            for (n in 3 downTo 1) { countdown = n; haptics.tick(); delay(700) }
            countdown = null
            haptics.start()
            val ex = Recorded(vm.engine.recordExample(EXAMPLE_MS), front)
            haptics.end()
            val emb = ex.example.embedding
            val earlier = examples.mapNotNull { it.example.embedding }
            when {
                ex.example.handFraction < MIN_HAND_FRACTION -> message = "No hands were seen in that example. Keep your hands in view and record it again."
                emb == null -> message = "This model cannot learn new signs."
                earlier.isNotEmpty() && Embeddings.cosine(emb, Embeddings.prototype(earlier)) < CONSISTENT -> {
                    pending = ex
                    message = "That one looked different from the earlier examples. Keep it, or record it again?"
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

    if (!teach.supported) {
        Box(Modifier.fillMaxSize().padding(28.dp), contentAlignment = Alignment.Center) {
            Text(teach.note ?: "Loading the model…", style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        return
    }

    Column(Modifier.fillMaxSize()) {
        Stepper(step)
        when (step) {
            Step.NAME -> NameStep(vm, word, teach.signs, onWord = { word = it }, onNext = { message = null; step = Step.RECORD })
            Step.RECORD -> CameraStep(
                vm, live, front, onFlip = { front = !front },
                overlay = { CountdownAndProgress(countdown, live) },
            ) {
                Text("Sign “${word.trim()}”", style = MaterialTheme.typography.titleLarge)
                Text(
                    "Record it $MIN_EXAMPLES to $MAX_EXAMPLES times, a little differently each time: nearer, further, faster.",
                    style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(Modifier.height(12.dp))
                ExampleStrip(examples, onDelete = { examples.remove(it) })
                message?.let {
                    Text(it, Modifier.padding(top = 10.dp).semantics { liveRegion = LiveRegionMode.Polite },
                        style = MaterialTheme.typography.bodyMedium, color = Palette.Amber)
                }
                pending?.let { ex ->
                    Row {
                        TextButton(onClick = { examples += ex; pending = null; message = null }) { Text("Keep it") }
                        TextButton(onClick = { pending = null; message = null }) { Text("Discard") }
                    }
                }
                Spacer(Modifier.height(12.dp))
                ActionRow {
                    Button(
                        onClick = { record() },
                        enabled = !busy && pending == null && examples.size < MAX_EXAMPLES && live.running,
                        modifier = Modifier.heightIn(min = 48.dp),
                    ) {
                        Icon(Icons.Rounded.FiberManualRecord, null, tint = Palette.Red)
                        Spacer(Modifier.width(8.dp))
                        Text(if (examples.isEmpty()) "Record" else "Record example ${examples.size + 1}")
                    }
                    OutlinedButton(onClick = { step = Step.REVIEW }, enabled = !busy && examples.size >= MIN_EXAMPLES,
                        modifier = Modifier.heightIn(min = 48.dp)) { Text("Review") }
                }
                if (examples.size < MIN_EXAMPLES) {
                    Text("${MIN_EXAMPLES - examples.size} more to go before saving.", Modifier.padding(top = 6.dp),
                        style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                TextButton(onClick = { step = Step.NAME }, enabled = !busy) { Text("Change the name") }
            }
            Step.REVIEW -> ReviewStep(word, examples, busy, message,
                onDelete = { examples.remove(it); if (examples.size < MIN_EXAMPLES) step = Step.RECORD },
                onMore = { step = Step.RECORD },
                onSave = {
                    scope.launch {
                        busy = true
                        val w = word
                        runCatching { vm.engine.saveTaught(w, examples.map { it.example.clip }) }
                            .onSuccess { examples.clear(); message = null; haptics.confirm(); vm.engine.clearCaption(); step = Step.TRY }
                            .onFailure { message = it.message ?: "Could not save the sign" }
                        busy = false
                    }
                },
            )
            Step.TRY -> TryStep(vm, live, word, front, onFlip = { front = !front },
                onAnother = { word = ""; step = Step.NAME },
            )
        }
    }
}

/** Where the person is in the four steps. */
@Composable
private fun Stepper(current: Step) {
    Row(
        Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 6.dp)
            .semantics(mergeDescendants = true) { contentDescription = "Step ${current.ordinal + 1} of 4: ${current.title}" },
        horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically,
    ) {
        Step.entries.forEach { s ->
            val on = s.ordinal <= current.ordinal
            Box(Modifier.weight(1f).height(4.dp).clip(CircleShape)
                .background(if (on) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceContainerHighest))
        }
        Spacer(Modifier.width(6.dp))
        Text(current.title, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ActionRow(content: @Composable () -> Unit) {
    FlowRow(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) { content() }
}

/** Step 1: what the sign means, and the signs already taught on this phone. */
@Composable
private fun NameStep(vm: MainViewModel, word: String, signs: List<TaughtSign>, onWord: (String) -> Unit, onNext: () -> Unit) {
    val scope = rememberCoroutineScope()
    val show = LocalCaption.current
    var renaming by remember { mutableStateOf<TaughtSign?>(null) }
    var deleting by remember { mutableStateOf<TaughtSign?>(null) }
    val name = storedName(word)
    val known = remember(word) { name.isNotEmpty() && name in vm.engine.labels() }
    val again = signs.any { it.word == name }

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 8.dp)) {
        Text("What does the sign mean?", Modifier.semantics { heading() }, style = MaterialTheme.typography.titleLarge)
        Text(
            "Usually a name sign, or a word EchoGesture doesn't know yet. It is captioned and spoken as you write it here.",
            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(14.dp))
        OutlinedTextField(
            value = word, onValueChange = { onWord(it.take(24)) }, singleLine = true,
            label = { Text("Word or name") },
            keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Words, imeAction = ImeAction.Next),
            keyboardActions = KeyboardActions(onNext = { if (word.isNotBlank()) onNext() }),
            modifier = Modifier.fillMaxWidth(),
        )
        val note = when {
            again -> "You already taught “${show(name)}”. These examples are added as another way to sign it."
            known -> "EchoGesture already knows “${show(name)}”. Teaching it adds your own way of signing it."
            else -> null
        }
        note?.let { Text(it, Modifier.padding(top = 8.dp), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary) }
        Spacer(Modifier.height(14.dp))
        Button(onClick = onNext, enabled = word.isNotBlank(), modifier = Modifier.heightIn(min = 48.dp)) { Text("Next: record it") }
        Spacer(Modifier.height(16.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Rounded.Lock, null, Modifier.size(16.dp), tint = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.width(8.dp))
            Text("Only hand, body and face keypoints are stored on this phone, never video.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }

        SectionTitle("Taught on this phone")
        if (signs.isEmpty()) {
            Text("None yet.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        signs.forEach { sign ->
            Card(Modifier.fillMaxWidth().padding(vertical = 4.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer)) {
                Row(Modifier.padding(start = 16.dp, end = 4.dp, top = 6.dp, bottom = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(show(sign.word), style = MaterialTheme.typography.titleMedium, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text("${sign.examples} example" + if (sign.examples == 1) "" else "s",
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    IconButton(onClick = { renaming = sign }) { Icon(Icons.Rounded.Edit, "Rename ${show(sign.word)}") }
                    IconButton(onClick = { deleting = sign }) { Icon(Icons.Rounded.Delete, "Delete ${show(sign.word)}") }
                }
            }
        }
        Spacer(Modifier.height(24.dp))
    }

    renaming?.let { sign ->
        var newName by remember(sign.id) { mutableStateOf(show(sign.word)) }
        AlertDialog(
            onDismissRequest = { renaming = null },
            title = { Text("Rename sign") },
            text = {
                OutlinedTextField(newName, { newName = it.take(24) }, singleLine = true,
                    keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Words))
            },
            confirmButton = {
                TextButton(onClick = {
                    val n = newName
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
            title = { Text("Delete “${show(sign.word)}”?") },
            text = { Text("Its ${sign.examples} recorded examples are removed from this phone.") },
            confirmButton = { TextButton(onClick = { deleting = null; scope.launch { vm.engine.deleteTaught(sign.id) } }) { Text("Delete") } },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("Cancel") } },
        )
    }
}

/** The name as it is stored and compared (the same cleaning the store applies). */
private fun storedName(word: String) = word.trim().replace(Regex("\\s+"), " ").uppercase()

/** The camera with its controls: stacked in portrait, side by side in landscape. */
@Composable
private fun CameraStep(
    vm: MainViewModel, live: LiveState, front: Boolean, onFlip: () -> Unit,
    overlay: @Composable () -> Unit = {},
    controls: @Composable () -> Unit,
) {
    val settings by vm.settings.collectAsStateWithLifecycle()
    val camera: @Composable (Modifier) -> Unit = { m ->
        Box(m.clip(RoundedCornerShape(24.dp)).background(Color.Black)) {
            CameraPreview(backCamera = !front, analyzer = vm.engine::analyze, modifier = Modifier.fillMaxSize())
            if (settings.showLandmarks) HandOverlay(live, mirror = front, skeleton = settings.developer && settings.skeleton, modifier = Modifier.fillMaxSize())
            StatusPill(
                when {
                    live.error != null -> "Hand tracking unavailable"
                    live.recording != null -> "Recording"
                    live.handsVisible -> "Hands in view"
                    live.bodyVisible -> "Show your hands"
                    else -> "Step back"
                },
                when {
                    live.error != null || live.recording != null -> Palette.Red
                    live.handsVisible -> Palette.Green
                    else -> Palette.Amber
                },
                Modifier.align(Alignment.TopStart).padding(12.dp),
            )
            FilledTonalIconButton(
                onClick = onFlip,
                modifier = Modifier.align(Alignment.TopEnd).padding(12.dp).size(48.dp).glass(CircleShape),
                colors = IconButtonDefaults.filledTonalIconButtonColors(containerColor = Color.Transparent, contentColor = Palette.Text),
            ) { Icon(Icons.Rounded.Cameraswitch, "Switch camera") }
            overlay()
            TrackingError(live, settings.gpu, onRetry = vm.engine::retryLive, onSwitch = { vm.update { it.copy(gpu = !it.gpu) } },
                modifier = Modifier.align(Alignment.Center).padding(12.dp))
        }
    }
    BoxWithConstraints(Modifier.fillMaxSize()) {
        if (maxWidth > maxHeight) {
            Row(Modifier.fillMaxSize().padding(horizontal = 12.dp, vertical = 8.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                camera(Modifier.weight(1f).fillMaxHeight())
                Column(Modifier.weight(1f).fillMaxHeight().verticalScroll(rememberScrollState()).padding(horizontal = 8.dp)) { controls() }
            }
        } else {
            Column(Modifier.fillMaxSize()) {
                camera(Modifier.fillMaxWidth().weight(0.48f).padding(horizontal = 12.dp, vertical = 8.dp))
                Column(Modifier.fillMaxWidth().weight(0.52f).verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 8.dp)) { controls() }
            }
        }
    }
}

/** 3, 2, 1 over the camera, then a red bar while the example records. */
@Composable
private fun CountdownAndProgress(countdown: Int?, live: LiveState) {
    Box(Modifier.fillMaxSize()) {
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
            LinearProgressIndicator(progress = { p }, modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth().height(6.dp), color = Palette.Red)
        }
    }
}

/** The examples so far, each playing back from its keypoints, with a delete button; empty places for the rest. */
@Composable
private fun ExampleStrip(examples: List<Recorded>, onDelete: (Recorded) -> Unit) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.semantics(mergeDescendants = false) {
        contentDescription = "${examples.size} of $MAX_EXAMPLES examples recorded"
    }) {
        repeat(MAX_EXAMPLES) { i ->
            val ex = examples.getOrNull(i)
            Box(
                Modifier.weight(1f).aspectRatio(0.8f).clip(RoundedCornerShape(12.dp))
                    .then(
                        if (ex == null) Modifier.border(1.5.dp, Palette.Outline, RoundedCornerShape(12.dp))
                        else Modifier.background(MaterialTheme.colorScheme.surfaceContainerHigh)
                    ),
                contentAlignment = Alignment.Center,
            ) {
                if (ex == null) {
                    Text("${i + 1}", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
                } else {
                    ExampleReplay(ex.example.clip, mirror = ex.front, modifier = Modifier.fillMaxSize().padding(4.dp))
                    IconButton(onClick = { onDelete(ex) }, modifier = Modifier.align(Alignment.TopEnd).size(32.dp)) {
                        Icon(Icons.Rounded.Close, "Delete example ${i + 1}", Modifier.size(18.dp))
                    }
                }
            }
        }
    }
}

/** Step 3: every example large, to spot one that went wrong, then save. */
@Composable
private fun ReviewStep(
    word: String, examples: List<Recorded>, busy: Boolean, message: String?,
    onDelete: (Recorded) -> Unit, onMore: () -> Unit, onSave: () -> Unit,
) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 8.dp)) {
        Text("Check your examples", Modifier.semantics { heading() }, style = MaterialTheme.typography.titleLarge)
        Text(
            "Each one plays back from the keypoints that are kept. Delete any that missed a hand or show a different movement.",
            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(12.dp))
        examples.chunked(2).forEach { pair ->
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(bottom = 10.dp)) {
                pair.forEach { ex ->
                    val n = examples.indexOf(ex) + 1
                    Box(Modifier.weight(1f).aspectRatio(0.85f).clip(RoundedCornerShape(16.dp)).background(MaterialTheme.colorScheme.surfaceContainer)) {
                        ExampleReplay(ex.example.clip, mirror = ex.front, modifier = Modifier.fillMaxSize().padding(10.dp))
                        Text("$n", Modifier.align(Alignment.TopStart).padding(10.dp), style = MaterialTheme.typography.labelLarge)
                        IconButton(onClick = { onDelete(ex) }, modifier = Modifier.align(Alignment.TopEnd)) {
                            Icon(Icons.Rounded.Delete, "Delete example $n")
                        }
                    }
                }
                if (pair.size == 1) Spacer(Modifier.weight(1f))
            }
        }
        message?.let { Text(it, Modifier.padding(bottom = 8.dp), style = MaterialTheme.typography.bodyMedium, color = Palette.Amber) }
        ActionRow {
            Button(onClick = onSave, enabled = !busy && examples.size >= MIN_EXAMPLES, modifier = Modifier.heightIn(min = 48.dp)) {
                Text("Save “${word.trim()}”")
            }
            if (examples.size < MAX_EXAMPLES) OutlinedButton(onClick = onMore, enabled = !busy, modifier = Modifier.heightIn(min = 48.dp)) { Text("Record another") }
        }
        Spacer(Modifier.height(24.dp))
    }
}

/** Step 4: sign it once; when the captions show it, it works. */
@Composable
private fun TryStep(vm: MainViewModel, live: LiveState, word: String, front: Boolean, onFlip: () -> Unit, onAnother: () -> Unit) {
    val haptics = LocalHaptics.current
    val shown = CaptionText.show(word.trim(), setOf(storedName(word)), caps = false)
    val heard = live.lastWord != null && live.lastWord.equals(storedName(word), ignoreCase = true)
    var worked by rememberSaveable { mutableStateOf(false) }
    var tries by remember { mutableIntStateOf(0) }
    val firstAt = remember { live.lastWordAt }
    LaunchedEffect(live.lastWordAt) {
        if (live.lastWordAt == firstAt) return@LaunchedEffect
        if (heard && !worked) { worked = true; haptics.confirm() } else if (!heard) tries++
    }
    CameraStep(vm, live, front, onFlip) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Saved. Now try it", Modifier.semantics { heading() }, style = MaterialTheme.typography.titleLarge)
                Text("Sign “$shown” once, as you recorded it.", style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            if (worked) DrawnCheck("tried-$word", 40.dp, modifier = Modifier.semantics { contentDescription = "It worked" })
        }
        Spacer(Modifier.height(12.dp))
        val (text, colour) = when {
            worked -> "It works: “$shown” appears in captions and is spoken like any other word." to Palette.Green
            live.lastWord != null && tries > 0 -> "That was captioned as “${LocalCaption.current(live.lastWord!!)}”. Try once more, or record more examples later." to Palette.Amber
            else -> "Waiting for the sign…" to Palette.TextMuted
        }
        Box(Modifier.fillMaxWidth().heightIn(min = 64.dp).clip(RoundedCornerShape(16.dp)).background(MaterialTheme.colorScheme.surfaceContainer)
            .semantics { liveRegion = LiveRegionMode.Polite }.padding(horizontal = 16.dp, vertical = 12.dp), contentAlignment = Alignment.CenterStart) {
            Text(text, style = MaterialTheme.typography.titleMedium, color = colour)
        }
        Spacer(Modifier.height(12.dp))
        ActionRow {
            Button(onClick = onAnother, modifier = Modifier.heightIn(min = 48.dp)) { Text("Teach another sign") }
        }
    }
}
