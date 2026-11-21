package ai.echogesture.app.ui.teach

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.VolumeUp
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.School
import androidx.compose.material.icons.rounded.Search
import androidx.compose.material.icons.rounded.Translate
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.echogesture.app.MainViewModel
import ai.echogesture.app.R
import ai.echogesture.app.data.Demos
import ai.echogesture.app.data.Lessons
import ai.echogesture.app.data.SignToken
import ai.echogesture.app.data.SignTranslator
import ai.echogesture.app.engine.Speaker
import ai.echogesture.app.ui.components.LocalCaption
import ai.echogesture.app.ui.components.SectionTitle
import ai.echogesture.app.ui.components.SubScreen
import ai.echogesture.app.ui.theme.Palette

/**
 * Offline Sign Dictionary & Text-to-Sign translator:
 *   - Type any English phrase or tap an example to watch real-signer ASL animations play back in sequence.
 *   - Search or filter all 300 demonstrated ASL signs by topic, hear pronunciation, or jump straight into
 *     camera-checked practice on any sign.
 * Works 100% offline using the bundled demos.bin keypoints and same-sign sets.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun DictionaryScreen(
    vm: MainViewModel,
    initialWord: String? = null,
    onPractiseWord: (String) -> Unit,
    onBack: () -> Unit,
) {
    val context = LocalContext.current
    val live by vm.live.collectAsStateWithLifecycle()
    val progress by vm.progress.collectAsStateWithLifecycle()
    val learned = remember(progress.learned) { progress.learned.toSet() }
    val show = LocalCaption.current
    val demos = remember { Demos.load(context) }
    val synonyms = remember(live.ready) { vm.engine.synonymMap() }
    val allWords = remember(demos) { demos.keys.toList() }
    val topics = remember(allWords) { Lessons.build(allWords) }

    var phrase by rememberSaveable { mutableStateOf(initialWord?.let { show(it) }.orEmpty()) }
    var selectedTopic by rememberSaveable { mutableStateOf<String?>(null) }
    var selectedWord by rememberSaveable {
        mutableStateOf(initialWord?.uppercase()?.takeIf { it in demos })
    }
    var playingWord by remember { mutableStateOf<String?>(null) }

    val tokens = remember(phrase, demos, synonyms, selectedWord) {
        when {
            phrase.isNotBlank() -> SignTranslator.translate(phrase, demos.keys, synonyms)
            selectedWord != null -> listOf(SignToken(show(selectedWord!!), selectedWord))
            else -> emptyList()
        }
    }
    val hasPlayable = remember(tokens, demos) { tokens.any { it.gloss != null && it.gloss in demos } }
    val practiceCandidate = remember(tokens, selectedWord, playingWord) {
        playingWord ?: tokens.firstOrNull { it.gloss != null }?.gloss ?: selectedWord
    }

    val topicWords = remember(topics, selectedTopic, allWords) {
        if (selectedTopic == null) allWords
        else topics.firstOrNull { it.key == selectedTopic }?.lessons?.flatMap { it.words } ?: allWords
    }
    val filteredWords = remember(phrase, topicWords) {
        // When the user typed a multi-word sentence, keep the full list below for browsing.
        if (' ' in phrase.trim()) topicWords else SignTranslator.search(phrase, topicWords, show)
    }

    SubScreen(stringResource(R.string.dict_title), onBack) {
        LazyColumn(
            Modifier.fillMaxSize(),
            contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 6.dp, bottom = 28.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item {
                Text(
                    stringResource(R.string.dict_subtitle, demos.size),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(Modifier.height(10.dp))
                OutlinedTextField(
                    value = phrase,
                    onValueChange = {
                        phrase = it.take(120)
                        if (it.isNotBlank()) selectedWord = null
                    },
                    singleLine = true,
                    leadingIcon = { Icon(if (' ' in phrase) Icons.Rounded.Translate else Icons.Rounded.Search, null) },
                    trailingIcon = {
                        if (phrase.isNotEmpty()) {
                            IconButton(onClick = { phrase = "" }) {
                                Icon(Icons.Rounded.Close, stringResource(R.string.dict_clear))
                            }
                        }
                    },
                    placeholder = { Text(stringResource(R.string.dict_placeholder)) },
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                    shape = RoundedCornerShape(16.dp),
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            // Sample sentences when input is empty so users immediately try Text-to-Sign in one tap.
            if (phrase.isBlank() && selectedWord == null) {
                item {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            stringResource(R.string.dict_try_phrase),
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.primary,
                        )
                        FlowRow(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp),
                        ) {
                            SignTranslator.SAMPLE_SENTENCES.forEach { sample ->
                                AssistChip(
                                    onClick = { phrase = sample; selectedWord = null },
                                    label = { Text(sample, maxLines = 1, overflow = TextOverflow.Ellipsis) },
                                )
                            }
                        }
                    }
                }
            }

            // Animated real-signer playback card whenever a sentence or sign is selected.
            if (tokens.isNotEmpty()) {
                item {
                    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        SignSequencePlayer(
                            tokens = tokens,
                            demos = demos,
                            figureHeight = 250.dp,
                            onActiveSignChange = { playingWord = it },
                        )
                        if (hasPlayable) {
                            Row(
                                Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(10.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                practiceCandidate?.let { w ->
                                    Button(
                                        onClick = { onPractiseWord(w) },
                                        modifier = Modifier.weight(1f).heightIn(min = 48.dp),
                                    ) {
                                        Icon(Icons.Rounded.School, null, Modifier.size(18.dp))
                                        Spacer(Modifier.width(8.dp))
                                        Text(stringResource(R.string.dict_practise_sign, show(w)), maxLines = 1, overflow = TextOverflow.Ellipsis)
                                    }
                                }
                                OutlinedButton(
                                    onClick = {
                                        val speakText = if (phrase.isNotBlank()) phrase else practiceCandidate?.let(Speaker::sayable).orEmpty()
                                        vm.engine.speakNow(speakText)
                                    },
                                    modifier = Modifier.heightIn(min = 48.dp),
                                ) {
                                    Icon(Icons.AutoMirrored.Rounded.VolumeUp, null, Modifier.size(18.dp))
                                    Spacer(Modifier.width(6.dp))
                                    Text(stringResource(R.string.dict_hear_short), maxLines = 1)
                                }
                            }
                        }
                    }
                }
            }

            // Topic filter strip for browsing all 300 demonstrated signs.
            item {
                SectionTitle(stringResource(R.string.dict_browse_title, filteredWords.size), Modifier.padding(top = 8.dp, bottom = 4.dp))
                LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    item {
                        FilterChip(
                            selected = selectedTopic == null,
                            onClick = { selectedTopic = null },
                            label = { Text(stringResource(R.string.dict_topic_all)) },
                        )
                    }
                    items(topics, key = { it.key }) { topic ->
                        FilterChip(
                            selected = selectedTopic == topic.key,
                            onClick = { selectedTopic = if (selectedTopic == topic.key) null else topic.key },
                            label = { Text(topic.title) },
                        )
                    }
                }
            }

            // 2-column grid of browsable signs.
            val pairs = filteredWords.chunked(2)
            items(pairs) { row ->
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    row.forEach { w ->
                        val isSelected = selectedWord == w || (phrase.trim().equals(show(w), ignoreCase = true))
                        val isLearned = w in learned
                        Row(
                            Modifier
                                .weight(1f)
                                .clip(RoundedCornerShape(16.dp))
                                .background(
                                    if (isSelected) MaterialTheme.colorScheme.primaryContainer
                                    else MaterialTheme.colorScheme.surfaceContainer,
                                )
                                .border(
                                    1.dp,
                                    if (isSelected) MaterialTheme.colorScheme.primary else Palette.Hairline,
                                    RoundedCornerShape(16.dp),
                                )
                                .clickable {
                                    selectedWord = w
                                    phrase = ""
                                }
                                .heightIn(min = 52.dp)
                                .padding(horizontal = 14.dp, vertical = 10.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text(
                                show(w),
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
                                color = if (isSelected) MaterialTheme.colorScheme.onPrimaryContainer else Palette.Text,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.weight(1f),
                            )
                            if (isLearned) {
                                Spacer(Modifier.width(6.dp))
                                Icon(Icons.Rounded.CheckCircle, stringResource(R.string.practice_learned), Modifier.size(18.dp), tint = Palette.Green)
                            }
                        }
                    }
                    if (row.size == 1) Spacer(Modifier.weight(1f))
                }
            }
        }
    }
}
