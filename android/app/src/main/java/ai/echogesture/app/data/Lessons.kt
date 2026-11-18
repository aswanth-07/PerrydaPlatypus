package ai.echogesture.app.data

/** One lesson: about five signs of one topic, learned in order. [id] is "topic-number", e.g. "food-2". */
data class Lesson(val id: String, val topic: String, val title: String, val number: Int, val words: List<String>)

/** A topic and its lessons, in teaching order. */
data class Topic(val key: String, val title: String, val lessons: List<Lesson>)

/**
 * The 300 signs with a demonstration, grouped by topic for lessons: everyday signs first, then topics a
 * conversation reaches. A topic is split into lessons of about [SIZE] signs, as evenly as possible (12 signs make
 * three lessons of four, not 5 + 5 + 2). Signs with a demonstration that no topic lists go to "More signs", so a
 * new demonstration is never lost.
 */
object Lessons {
    const val SIZE = 5

    private val TOPICS = listOf(
        Triple("first", "First signs", "YES NO PLEASE HELP FINE GOOD BAD NAME SIGN DEAF HEARING LANGUAGE LEARN PRACTICE AGAIN MORE WANT NEED LIKE KNOW"),
        Triple("questions", "Questions", "WHAT WHERE WHO WHY HOW"),
        Triple("little", "Little words", "YOU YOUR HERE NOW ABOUT BECAUSE BUT WITH SINCE ALL SOME NONE MANY MOST SAME DIFFERENT RIGHT WRONG FIRST LAST BACK BEHIND"),
        Triple("people", "Family and people", "FAMILY MOTHER FATHER BROTHER SON DAUGHTER HUSBAND WIFE BABY CHILDREN BOY GIRL MAN WOMAN PERSON FRIEND COUSIN MEET KISS DIVORCE"),
        Triple("feelings", "Feelings", "HAPPY CRY LAUGH TIRED FEEL HOPE ENJOY HEART CRAZY CUTE COOL HUMBLE FRIENDLY MEAN CAREFUL"),
        Triple("health", "Health", "DOCTOR SICK HEADACHE COUGH MEDICINE"),
        Triple("food", "Food and drink", "EAT DRINK WATER MILK COFFEE TEA APPLE BANANA TOMATO CORN CHEESE EGG MEAT FISH PIZZA SANDWICH CEREAL CANDY SALT DELICIOUS COOK BAKE FULL RESTAURANT BAR"),
        Triple("home", "At home", "HOME HOUSE ROOM BED BATH BATHROOM DOOR WINDOW TABLE CHAIR BLANKET CLOCK LIGHT COMPUTER INTERNET"),
        Triple("things", "Things", "BOOK PAPER PENCIL LETTER KNIFE GLASSES MONEY BALL MOVIE MUSIC"),
        Triple("clothes", "Clothes and looks", "CLOTHES SHIRT PANTS DRESS JACKET HAT BACKPACK HAIR BEARD TALL SHORT FAT THIN SMALL UGLY"),
        Triple("colours", "Colours", "COLOR BLACK WHITE RED BLUE GREEN YELLOW BROWN PINK PURPLE ORANGE DARK"),
        Triple("nature", "Animals and weather", "ANIMAL DOG CAT BIRD COW RABBIT FLOWER SNOW HOT COLD ENVIRONMENT"),
        Triple("time", "Time", "TIME DAY WEEK YEAR HOUR SUNDAY THURSDAY YESTERDAY SOON LATER LATE BEFORE PAST FUTURE ALWAYS"),
        Triple("celebrations", "Celebrations", "BIRTHDAY CHRISTMAS HALLOWEEN THANKSGIVING PARTY"),
        Triple("moving", "Going places", "GO WALK RUN JUMP DANCE DRIVE RIDE TRAVEL VISIT ARRIVE LEAVE STAY WAIT HURRY TRAIN"),
        Triple("doing", "Everyday actions", "HAVE CAN MAKE GIVE TAKE BRING BUY PULL DROP CATCH SHOW CHANGE FINISH LOSE CHECK"),
        Triple("talking", "Talking", "TELL CALL CHAT DISCUSS ARGUE CONVINCE INFORM HEAR SHARE JOIN APPROVE DECIDE FAULT"),
        Triple("school", "School and work", "SCHOOL CLASS STUDENT TEACHER TEACH STUDY TEST COLLEGE GRADUATE READ WRITE DRAW PAINT COPY LIST WORK OFFICE BUSINESS SECRETARY PLAN INTEREST"),
        Triple("world", "The wider world", "CITY COUNTRY EAST SOUTH CENTER FAR AFRICA RUSSIA GOVERNMENT PRESIDENT POLICE LAW WAR"),
        Triple("play", "Games and sport", "PLAY GAME BASKETBALL FOOTBALL BOWLING CHAMPION MATCH BALANCE"),
        Triple("ideas", "Ideas and describing", "NEW OLD EASY HARD EXAMPLE THEORY PROBLEM IMPROVE ORDER TRADE REMEMBER FORGET LIVE CLOSE DELAY DISAPPEAR CHEAT CRASH ACCIDENT KILL"),
    )

    /** Lesson sizes for [n] signs: ceil(n / size) lessons, as even as possible, larger ones first. */
    fun split(n: Int, size: Int = SIZE): List<Int> {
        if (n <= 0) return emptyList()
        val count = (n + size - 1) / size
        return List(count) { i -> n / count + if (i < n % count) 1 else 0 }
    }

    /** Topics and lessons for the signs in [available] (those with a demonstration the loaded model knows). */
    fun build(available: Collection<String>): List<Topic> {
        val have = available.toSet()
        val placed = HashSet<String>()
        val topics = ArrayList<Topic>()
        fun add(key: String, title: String, words: List<String>) {
            if (words.isEmpty()) return
            var start = 0
            val lessons = split(words.size).mapIndexed { i, n ->
                Lesson("$key-${i + 1}", key, title, i + 1, words.subList(start, start + n)).also { start += n }
            }
            topics += Topic(key, title, lessons)
        }
        for ((key, title, list) in TOPICS) {
            val words = list.split(' ').filter { it in have && placed.add(it) }
            add(key, title, words)
        }
        add("more", "More signs", available.filter { placed.add(it) })
        return topics
    }

    /** The first lesson, in order, with a sign not yet in [learned]; null when every lesson is done. */
    fun next(topics: List<Topic>, learned: Collection<String>): Lesson? {
        val done = learned.toSet()
        return topics.asSequence().flatMap { it.lessons }.firstOrNull { l -> l.words.any { it !in done } }
    }

    fun find(topics: List<Topic>, id: String): Lesson? = topics.asSequence().flatMap { it.lessons }.firstOrNull { it.id == id }

    /** The lesson after [id], across topics; null after the last. */
    fun after(topics: List<Topic>, id: String): Lesson? {
        val all = topics.flatMap { it.lessons }
        val i = all.indexOfFirst { it.id == id }
        return if (i < 0) null else all.getOrNull(i + 1)
    }
}
