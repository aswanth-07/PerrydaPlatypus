package ai.echogesture.app.ui.teach

import ai.echogesture.app.ui.theme.Motion
import androidx.compose.animation.core.tween
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.material3.ripple
import ai.echogesture.app.ui.components.rememberPress
import ai.echogesture.app.R
import androidx.compose.ui.res.stringResource
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.data.Lesson
import ai.echogesture.app.data.Lessons
import ai.echogesture.app.ui.components.DrawnCheck
import ai.echogesture.app.ui.components.LocalCaption
import ai.echogesture.app.ui.components.SubScreen
import ai.echogesture.app.ui.theme.Palette

/** Every lesson, by topic, with how many of its signs are learned. */
@Composable
fun LessonsScreen(vm: MainViewModel, onOpen: (String) -> Unit, onBack: () -> Unit) {
    val topics = rememberTopics(vm)
    val progress by vm.progress.collectAsStateWithLifecycle()
    val learned = progress.learned.toSet()
    SubScreen(stringResource(R.string.lesson_lessons), onBack) {
        if (topics.isEmpty()) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { CircularProgressIndicator() }
            return@SubScreen
        }
        val next = Lessons.next(topics, learned)
        LazyColumn(contentPadding = PaddingValues(start = 16.dp, end = 16.dp, bottom = 24.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            for (topic in topics) {
                item(key = topic.key) {
                    val words = topic.lessons.flatMap { it.words }
                    Row(Modifier.fillMaxWidth().padding(top = 16.dp, bottom = 2.dp, start = 4.dp), verticalAlignment = Alignment.Bottom) {
                        Text(topic.title, Modifier.weight(1f).semantics { heading() }, style = MaterialTheme.typography.titleMedium)
                        Text(stringResource(R.string.lesson_of, words.count { it in learned }, words.size), style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
                items(topic.lessons, key = { it.id }) { lesson ->
                    Box(Modifier.animateItem()) { LessonRow(lesson, learned, current = lesson.id == next?.id) { onOpen(lesson.id) } }
                }
            }
        }
    }
}

@Composable
private fun LessonRow(lesson: Lesson, learned: Set<String>, current: Boolean, onClick: () -> Unit) {
    val show = LocalCaption.current
    val done = lesson.words.count { it in learned }
    val all = done == lesson.words.size
    val press = rememberPress()
    Row(
        press.scale.fillMaxWidth().clip(RoundedCornerShape(18.dp))
            .background(if (current) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceContainer)
            .clickable(interactionSource = press.interaction, indication = ripple(), role = Role.Button,
                onClickLabel = stringResource(R.string.lesson_open_lesson), onClick = onClick)
            .padding(horizontal = 14.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.size(36.dp), contentAlignment = Alignment.Center) {
            if (all) DrawnCheck(lesson.id, 32.dp)
            else Box(Modifier.size(32.dp).clip(CircleShape).border(1.5.dp, Palette.Outline, CircleShape), contentAlignment = Alignment.Center) {
                Text("${lesson.number}", style = MaterialTheme.typography.labelLarge)
            }
        }
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text(if (current) stringResource(R.string.lesson_lesson_up_next, lesson.number) else stringResource(R.string.lesson_lesson, lesson.number), style = MaterialTheme.typography.labelLarge,
                color = if (current) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSurface)
            Text(lesson.words.joinToString(", ", transform = show), style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 2, overflow = TextOverflow.Ellipsis)
        }
        Spacer(Modifier.width(8.dp))
        Text("$done/${lesson.words.size}", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

/** One lesson: its signs practised in order, then a summary that leads to the next lesson. */
@Composable
fun LessonScreen(vm: MainViewModel, id: String, onBack: () -> Unit, onNext: (String) -> Unit, gate: @Composable (@Composable () -> Unit) -> Unit) {
    val topics = rememberTopics(vm)
    val progress by vm.progress.collectAsStateWithLifecycle()
    val lesson = Lessons.find(topics, id)
    var finished by rememberSaveable(id) { mutableStateOf(false) }
    SubScreen(lesson?.let { stringResource(R.string.lesson_lesson_2, it.title, it.number) } ?: stringResource(R.string.lesson_lesson_3), onBack) {
        if (lesson == null) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { CircularProgressIndicator() }
            return@SubScreen
        }
        gate { LearnScreen(vm, lesson, onLessonEnd = { finished = true }) }
    }
    if (finished && lesson != null) {
        val got = lesson.words.count { it in progress.learned }
        val next = Lessons.after(topics, id)
        AlertDialog(
            onDismissRequest = { finished = false },
            // A tick only when something was signed: "done" with nothing matched would not be true.
            icon = if (got > 0) ({ DrawnCheck("done-$id", 48.dp) }) else null,
            title = { Text(if (got > 0) stringResource(R.string.lesson_lesson_done) else stringResource(R.string.lesson_end_of_the_lesson)) },
            text = {
                Text(
                    when (got) {
                        lesson.words.size -> stringResource(R.string.lesson_you_signed_all_signs_in, lesson.words.size)
                        0 -> stringResource(R.string.lesson_none_of_these_signs_was)
                        else -> stringResource(R.string.lesson_you_signed_of_the_ones, got, lesson.words.size)
                    }
                )
            },
            confirmButton = {
                if (next != null) Button(onClick = { finished = false; onNext(next.id) }) { Text(stringResource(R.string.lesson_next_lesson)) }
                else Button(onClick = { finished = false; onBack() }) { Text(stringResource(R.string.talk_done)) }
            },
            dismissButton = { if (next != null) TextButton(onClick = { finished = false; onBack() }) { Text(stringResource(R.string.lesson_back_to_learn)) } },
        )
    }
}

/** The Learn tab's way in: the next lesson with a sign still to learn, and how far into it the learner is. */
@Composable
fun ContinueCard(vm: MainViewModel, onOpen: (String) -> Unit, onAll: () -> Unit) {
    val topics = rememberTopics(vm)
    val progress by vm.progress.collectAsStateWithLifecycle()
    val show = LocalCaption.current
    if (topics.isEmpty()) return
    val next = Lessons.next(topics, progress.learned)
    val total = topics.sumOf { t -> t.lessons.sumOf { it.words.size } }
    val learnedHere = topics.sumOf { t -> t.lessons.sumOf { l -> l.words.count { it in progress.learned } } }
    Column(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(24.dp)).background(Palette.BrandDeep)
            .border(1.dp, Palette.Hairline, RoundedCornerShape(24.dp)).padding(20.dp),
    ) {
        Text(if (next == null) stringResource(R.string.lesson_every_lesson_done) else if (learnedHere == 0) stringResource(R.string.lesson_start_learning) else stringResource(R.string.lesson_continue_learning),
            style = MaterialTheme.typography.labelLarge, color = Palette.LightAzure)
        Spacer(Modifier.height(4.dp))
        if (next != null) {
            Text(stringResource(R.string.lesson_lesson_2, next.title, next.number), style = MaterialTheme.typography.titleLarge)
            Text(next.words.joinToString(", ", transform = show), style = MaterialTheme.typography.bodyMedium,
                color = Palette.LightAzure, maxLines = 2, overflow = TextOverflow.Ellipsis)
        } else {
            Text(stringResource(R.string.lesson_you_have_signed_all_signs, total),
                style = MaterialTheme.typography.bodyMedium, color = Palette.LightAzure)
        }
        Spacer(Modifier.height(12.dp))
        // The bar fills to the new value rather than jumping, so a sign just learned shows as progress.
        val shown by animateFloatAsState(if (total == 0) 0f else learnedHere / total.toFloat(), tween(Motion.LONG, easing = Motion.Standard), label = "progress")
        LinearProgressIndicator(
            progress = { shown }, modifier = Modifier.fillMaxWidth(),
            color = Palette.LightAzure, trackColor = Palette.PaperWhite.copy(alpha = 0.18f),
        )
        Text(stringResource(R.string.lesson_of_signs, learnedHere, total), Modifier.padding(top = 6.dp), style = MaterialTheme.typography.labelMedium,
            color = Palette.LightAzure)
        Spacer(Modifier.height(12.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
            if (next != null) Button(onClick = { onOpen(next.id) }) { Text(if (next.words.any { it in progress.learned }) stringResource(R.string.setup_continue) else stringResource(R.string.setup_start)) }
            TextButton(onClick = onAll, colors = ButtonDefaults.textButtonColors(contentColor = Palette.PaperWhite)) { Text(stringResource(R.string.lesson_all_lessons)) }
        }
    }
}
