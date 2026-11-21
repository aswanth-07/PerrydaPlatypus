package ai.echogesture.app.ui.teach

import ai.echogesture.app.R
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
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
import androidx.compose.material3.AssistChip
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
import ai.echogesture.app.ui.components.description
import androidx.annotation.StringRes
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

private enum class Step(@StringRes val title: Int) { NAME(R.string.teach_name), RECORD(R.string.teach_record), REVIEW(R.string.teach_review), TRY(R.string.teach_try_it) }

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
    // Read here: the recording and saving below run in coroutines, where resources cannot be read by Compose.
    val noHands = stringResource(R.string.teach_no_hands_were_seen_in)
    val cannotLearn = stringResource(R.string.teach_this_model_cannot_learn_new)
    val looksDifferent = stringResource(R.string.teach_that_one_looked_different_from)
    val recordingFailed = stringResource(R.string.teach_recording_failed)
    val saveFailed = stringResource(R.string.teach_could_not_save_the_sign)

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
                ex.example.handFraction < MIN_HAND_FRACTION -> message = noHands
                emb == null -> message = cannotLearn
                earlier.isNotEmpty() && Embeddings.cosine(emb, Embeddings.prototype(earlier)) < CONSISTENT -> {
                    pending = ex
                    message = looksDifferent
                }
                else -> examples += ex
            }
        } catch (e: Exception) {
            message = e.message ?: recordingFailed
        } finally {
            countdown = null
            busy = false
        }
    }

    if (!teach.supported) {
        Box(Modifier.fillMaxSize().padding(28.dp), contentAlignment = Alignment.Center) {
            Text(teach.note ?: stringResource(R.string.practice_loading_the_model), style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
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
                Text(stringResource(R.string.teach_sign, word.trim()), style = MaterialTheme.typography.titleLarge)
                Text(
                    stringResource(R.string.teach_record_it_to_times_a, MIN_EXAMPLES, MAX_EXAMPLES),
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
                        TextButton(onClick = { examples += ex; pending = null; message = null }) { Text(stringResource(R.string.teach_keep_it)) }
                        TextButton(onClick = { pending = null; message = null }) { Text(stringResource(R.string.teach_discard)) }
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
                        Text(if (examples.isEmpty()) stringResource(R.string.teach_record) else stringResource(R.string.teach_record_example, examples.size + 1))
                    }
                    OutlinedButton(onClick = { step = Step.REVIEW }, enabled = !busy && examples.size >= MIN_EXAMPLES,
                        modifier = Modifier.heightIn(min = 48.dp)) { Text(stringResource(R.string.teach_review)) }
                }
                if (examples.size < MIN_EXAMPLES) {
                    Text(stringResource(R.string.teach_more_to_go_before_saving, MIN_EXAMPLES - examples.size), Modifier.padding(top = 6.dp),
                        style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                TextButton(onClick = { step = Step.NAME }, enabled = !busy) { Text(stringResource(R.string.teach_change_the_name)) }
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
                            .onFailure { message = it.message ?: saveFailed }
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
            .description(stringResource(R.string.teach_step_of_4, current.ordinal + 1, stringResource(current.title)), merge = true),
        horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically,
    ) {
        Step.entries.forEach { s ->
            val on = s.ordinal <= current.ordinal
            Box(Modifier.weight(1f).height(4.dp).clip(CircleShape)
                .background(if (on) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceContainerHighest))
        }
        Spacer(Modifier.width(6.dp))
        Text(stringResource(current.title), style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ActionRow(content: @Composable () -> Unit) {
    FlowRow(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) { content() }
}

/** Step 1: what the sign means, quick inspiration ideas, and the signs already taught on this phone. */
@OptIn(ExperimentalLayoutApi::class)
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
        Text(stringResource(R.string.teach_what_does_the_sign_mean), Modifier.semantics { heading() }, style = MaterialTheme.typography.titleLarge)
        Spacer(Modifier.height(4.dp))
        Text(
            stringResource(R.string.teach_usually_a_name_sign_or),
            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(14.dp))
        OutlinedTextField(
            value = word, onValueChange = { onWord(it.take(24)) }, singleLine = true,
            label = { Text(stringResource(R.string.teach_word_or_name)) },
            keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Words, imeAction = ImeAction.Next),
            keyboardActions = KeyboardActions(onNext = { if (word.isNotBlank()) onNext() }),
            shape = RoundedCornerShape(16.dp),
            modifier = Modifier.fillMaxWidth(),
        )
        if (word.isBlank()) {
            Spacer(Modifier.height(10.dp))
            Text(stringResource(R.string.teach_ideas_label), style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.height(4.dp))
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                listOf("Priya", "Namaste", "Clinic", "Chai").forEach { idea ->
                    AssistChip(onClick = { onWord(idea) }, label = { Text(idea) })
                }
            }
        }
        val note = when {
            again -> stringResource(R.string.teach_you_already_taught_these_examples, show(name))
            known -> stringResource(R.string.teach_echogesture_already_knows_teaching_it, show(name))
            else -> null
        }
        note?.let { Text(it, Modifier.padding(top = 8.dp), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary) }
        Spacer(Modifier.height(14.dp))
        Button(onClick = onNext, enabled = word.isNotBlank(), modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp)) {
            Text(stringResource(R.string.teach_next_record_it))
        }
        Spacer(Modifier.height(14.dp))
        // Compact "How it works" + privacy summary card.
        Column(
            Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(18.dp))
                .background(MaterialTheme.colorScheme.surfaceContainer)
                .border(1.dp, Palette.Hairline, RoundedCornerShape(18.dp))
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(stringResource(R.string.teach_how_it_works), style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
            Text(stringResource(R.string.teach_how_steps), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Rounded.Lock, null, Modifier.size(16.dp), tint = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.width(8.dp))
                Text(stringResource(R.string.teach_only_hand_body_and_face),
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }

        SectionTitle(stringResource(R.string.teach_taught_on_this_phone))
        if (signs.isEmpty()) {
            Box(
                Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(MaterialTheme.colorScheme.surfaceContainer)
                    .border(1.dp, Palette.Hairline, RoundedCornerShape(16.dp))
                    .padding(horizontal = 16.dp, vertical = 14.dp),
            ) {
                Text(stringResource(R.string.teach_none_yet_rich), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        signs.forEach { sign ->
            Card(Modifier.fillMaxWidth().padding(vertical = 4.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer)) {
                Row(Modifier.padding(start = 16.dp, end = 4.dp, top = 6.dp, bottom = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(show(sign.word), style = MaterialTheme.typography.titleMedium, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text(pluralStringResource(R.plurals.teach_examples, sign.examples, sign.examples),
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    IconButton(onClick = { renaming = sign }) { Icon(Icons.Rounded.Edit, stringResource(R.string.teach_rename, show(sign.word))) }
                    IconButton(onClick = { deleting = sign }) { Icon(Icons.Rounded.Delete, stringResource(R.string.teach_delete, show(sign.word))) }
                }
            }
        }
        Spacer(Modifier.height(24.dp))
    }

    renaming?.let { sign ->
        var newName by remember(sign.id) { mutableStateOf(show(sign.word)) }
        AlertDialog(
            onDismissRequest = { renaming = null },
            title = { Text(stringResource(R.string.teach_rename_sign)) },
            text = {
                OutlinedTextField(newName, { newName = it.take(24) }, singleLine = true,
                    keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Words))
            },
            confirmButton = {
                TextButton(onClick = {
                    val n = newName
                    renaming = null
                    if (n.isNotBlank()) scope.launch { vm.engine.renameTaught(sign.id, n) }
                }) { Text(stringResource(R.string.teach_rename_2)) }
            },
            dismissButton = { TextButton(onClick = { renaming = null }) { Text(stringResource(R.string.action_cancel)) } },
        )
    }
    deleting?.let { sign ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text(stringResource(R.string.teach_delete_2, show(sign.word))) },
            text = { Text(stringResource(R.string.teach_its_recorded_examples_are_removed, sign.examples)) },
            confirmButton = { TextButton(onClick = { deleting = null; scope.launch { vm.engine.deleteTaught(sign.id) } }) { Text(stringResource(R.string.action_delete)) } },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text(stringResource(R.string.action_cancel)) } },
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
    val context = LocalContext.current
    // Start Holistic trackers only on steps that use the camera (RECORD and TRY), not on NAME or REVIEW.
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
    val camera: @Composable (Modifier) -> Unit = { m ->
        Box(m.clip(RoundedCornerShape(24.dp)).background(Palette.Onyx2)
            .border(1.dp, Palette.Hairline, RoundedCornerShape(24.dp))) {
            CameraPreview(backCamera = !front, analyzer = vm.engine::analyze, modifier = Modifier.fillMaxSize())
            if (settings.showLandmarks) HandOverlay(vm.overlay, mirror = front, skeleton = settings.developer && settings.skeleton, modifier = Modifier.fillMaxSize())
            StatusPill(
                when {
                    live.error != null -> stringResource(R.string.talk_hand_tracking_unavailable)
                    live.recording != null -> stringResource(R.string.talk_recording)
                    live.handsVisible -> stringResource(R.string.teach_hands_in_view)
                    live.bodyVisible -> stringResource(R.string.talk_show_your_hands)
                    else -> stringResource(R.string.talk_step_back)
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
            ) { Icon(Icons.Rounded.Cameraswitch, stringResource(R.string.practice_switch_camera)) }
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
            Box(
                Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 16.dp)
                    .glass(RoundedCornerShape(50), Palette.Red.copy(alpha = 0.85f))
                    .padding(horizontal = 16.dp, vertical = 6.dp),
            ) {
                Text(stringResource(R.string.teach_sign_now), style = MaterialTheme.typography.labelLarge, color = Color.White, fontWeight = FontWeight.Bold)
            }
            LinearProgressIndicator(progress = { p }, modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth().height(6.dp), color = Palette.Red)
        }
    }
}

/** The examples so far, each playing back from its keypoints, with a delete button; empty places for the rest. */
@Composable
private fun ExampleStrip(examples: List<Recorded>, onDelete: (Recorded) -> Unit) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.description(stringResource(R.string.teach_of_examples_recorded, examples.size, MAX_EXAMPLES))) {
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
                        Icon(Icons.Rounded.Close, stringResource(R.string.teach_delete_example, i + 1), Modifier.size(18.dp))
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
        Text(stringResource(R.string.teach_check_your_examples), Modifier.semantics { heading() }, style = MaterialTheme.typography.titleLarge)
        Text(
            stringResource(R.string.teach_each_one_plays_back_from),
            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(12.dp))
        examples.chunked(2).forEach { pair ->
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(bottom = 10.dp)) {
                pair.forEach { ex ->
                    val n = examples.indexOf(ex) + 1
                    val handPct = (ex.example.handFraction * 100).toInt().coerceIn(0, 100)
                    Box(
                        Modifier.weight(1f).aspectRatio(0.85f).clip(RoundedCornerShape(16.dp))
                            .background(MaterialTheme.colorScheme.surfaceContainer)
                            .border(1.dp, Palette.Hairline, RoundedCornerShape(16.dp)),
                    ) {
                        ExampleReplay(ex.example.clip, mirror = ex.front, modifier = Modifier.fillMaxSize().padding(10.dp))
                        Text("$n", Modifier.align(Alignment.TopStart).padding(10.dp), style = MaterialTheme.typography.labelLarge)
                        Box(
                            Modifier
                                .align(Alignment.BottomStart)
                                .padding(8.dp)
                                .glass(RoundedCornerShape(50))
                                .padding(horizontal = 8.dp, vertical = 3.dp),
                        ) {
                            Text(
                                stringResource(R.string.teach_hands_pct, handPct),
                                style = MaterialTheme.typography.labelSmall,
                                color = if (handPct >= 65) Palette.Green else Palette.Amber,
                            )
                        }
                        IconButton(onClick = { onDelete(ex) }, modifier = Modifier.align(Alignment.TopEnd)) {
                            Icon(Icons.Rounded.Delete, stringResource(R.string.teach_delete_example, n))
                        }
                    }
                }
                if (pair.size == 1) Spacer(Modifier.weight(1f))
            }
        }
        message?.let { Text(it, Modifier.padding(bottom = 8.dp), style = MaterialTheme.typography.bodyMedium, color = Palette.Amber) }
        ActionRow {
            Button(onClick = onSave, enabled = !busy && examples.size >= MIN_EXAMPLES, modifier = Modifier.heightIn(min = 48.dp)) {
                Text(stringResource(R.string.teach_save, word.trim()))
            }
            if (examples.size < MAX_EXAMPLES) OutlinedButton(onClick = onMore, enabled = !busy, modifier = Modifier.heightIn(min = 48.dp)) { Text(stringResource(R.string.teach_record_another)) }
        }
        Spacer(Modifier.height(24.dp))
    }
}

/** Step 4: sign it once; when the captions show it, it works. */
@Composable
private fun TryStep(vm: MainViewModel, live: LiveState, word: String, front: Boolean, onFlip: () -> Unit, onAnother: () -> Unit) {
    val haptics = LocalHaptics.current
    // Sandbox mode: let the user test the newly taught sign without speaking it or writing it to Chat history.
    DisposableEffect(Unit) {
        vm.engine.sandboxMode = true
        onDispose {
            vm.engine.sandboxMode = false
            vm.engine.clearCaption()
        }
    }
    val shown = CaptionText.show(word.trim(), setOf(storedName(word)), caps = false)
    val seenWord = live.lastWord
    val heard = seenWord != null && seenWord.equals(storedName(word), ignoreCase = true)
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
                Text(stringResource(R.string.teach_saved_now_try_it), Modifier.semantics { heading() }, style = MaterialTheme.typography.titleLarge)
                Text(stringResource(R.string.teach_sign_once_as_you_recorded, shown), style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            if (worked) DrawnCheck("tried-$word", 40.dp, modifier = Modifier.description(stringResource(R.string.teach_it_worked)))
        }
        Spacer(Modifier.height(12.dp))
        val targetKey = storedName(word)
        val liveMatch = live.taughtMatch?.takeIf { it.first.equals(targetKey, ignoreCase = true) }?.second
        val (text, colour) = when {
            worked -> stringResource(R.string.teach_it_works_appears_in_captions, shown) to Palette.Green
            seenWord != null && tries > 0 -> stringResource(R.string.teach_that_was_captioned_as_try, LocalCaption.current(seenWord)) to Palette.Amber
            else -> stringResource(R.string.teach_waiting_for_the_sign) to Palette.TextMuted
        }
        Column(
            Modifier.fillMaxWidth().heightIn(min = 64.dp).clip(RoundedCornerShape(16.dp))
                .background(MaterialTheme.colorScheme.surfaceContainer)
                .border(1.dp, if (worked) Palette.Green.copy(alpha = 0.55f) else Palette.Hairline, RoundedCornerShape(16.dp))
                .semantics { liveRegion = LiveRegionMode.Polite }.padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(text, style = MaterialTheme.typography.titleMedium, color = colour)
            if (!worked && liveMatch != null && liveMatch > 0.15f) {
                val thresh = vm.settings.collectAsStateWithLifecycle().value.teachThreshold.coerceAtLeast(0.1f)
                val ratio = (liveMatch / thresh).coerceIn(0f, 1f)
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Box(Modifier.weight(1f).height(6.dp).clip(RoundedCornerShape(3.dp)).background(Color.White.copy(alpha = 0.10f))) {
                        Box(
                            Modifier.fillMaxWidth(ratio).fillMaxHeight().clip(RoundedCornerShape(3.dp))
                                .background(if (ratio >= 0.85f) Palette.Green else MaterialTheme.colorScheme.primary),
                        )
                    }
                    Text(
                        stringResource(R.string.teach_live_similarity, (liveMatch * 100).toInt()),
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
        Spacer(Modifier.height(12.dp))
        ActionRow {
            Button(onClick = onAnother, modifier = Modifier.heightIn(min = 48.dp)) { Text(stringResource(R.string.teach_teach_another_sign)) }
        }
    }
}
