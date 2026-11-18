package ai.echogesture.app.data

import java.time.LocalDate
import kotlin.test.Test
import kotlin.test.assertEquals

class StreaksTest {
    private val today = LocalDate.of(2026, 11, 18)

    @Test
    fun countsDaysInARowEndingToday() {
        val days = setOf("2026-11-16", "2026-11-17", "2026-11-18", "2026-11-10")
        assertEquals(3, Streaks.count(days, today))
    }

    @Test
    fun aStreakSurvivesUntilTheEndOfTodayWithoutPractice() {
        assertEquals(2, Streaks.count(setOf("2026-11-16", "2026-11-17"), today))
    }

    @Test
    fun aMissedDayEndsTheStreak() {
        assertEquals(0, Streaks.count(setOf("2026-11-15", "2026-11-16"), today))
        assertEquals(0, Streaks.count(emptySet(), today))
    }
}
