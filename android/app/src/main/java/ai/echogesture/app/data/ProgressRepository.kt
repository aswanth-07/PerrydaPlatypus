package ai.echogesture.app.data

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.core.stringSetPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import java.time.LocalDate

/** What the learner has done: signs learned, oldest first, and the days they practised (ISO dates). */
data class Progress(val learned: List<String> = emptyList(), val days: Set<String> = emptySet()) {
    fun streak(today: LocalDate = LocalDate.now()): Int = Streaks.count(days, today)
}

object Streaks {
    /** Days in a row with practice, ending today, or yesterday if today has none yet. */
    fun count(days: Set<String>, today: LocalDate): Int {
        var d = if (today.toString() in days) today else today.minusDays(1)
        var n = 0
        while (d.toString() in days) { n++; d = d.minusDays(1) }
        return n
    }
}

private val Context.progressStore by preferencesDataStore(name = "progress")

/** Learning progress on the phone, kept across restarts (Learn used to forget it when Android ended the app). */
class ProgressRepository(private val context: Context) {
    private object K {
        val learned = stringPreferencesKey("learned")   // newline-separated, oldest first
        val days = stringSetPreferencesKey("practice_days")
    }

    val progress: Flow<Progress> = context.progressStore.data.map { p ->
        Progress(p[K.learned].orEmpty().split('\n').filter { it.isNotBlank() }, p[K.days].orEmpty())
    }

    suspend fun markLearned(word: String, today: LocalDate = LocalDate.now()) {
        context.progressStore.edit { p ->
            val list = p[K.learned].orEmpty().split('\n').filter { it.isNotBlank() }
            if (word !in list) p[K.learned] = (list + word).joinToString("\n")
            p[K.days] = p[K.days].orEmpty() + today.toString()
        }
    }

    suspend fun reset() {
        context.progressStore.edit { it.clear() }
    }
}
