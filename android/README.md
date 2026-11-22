# EchoGesture for Android

Native Android app (Kotlin, Jetpack Compose, Material 3) that turns signing in front of the camera into
captions, entirely on the phone. The app has no internet permission: video and keypoints cannot leave it.

## Screens

Three tabs in a floating bar (a rail on the left in landscape): **Learn**, **Talk** in the middle, where the app
opens, and **Videos**. Everything else opens over the tabs with a back arrow.

| Screen | What it does |
|---|---|
| **Talk** | The camera fills the screen. A status pill worded for the camera in use ("Signing" / "Watching" / "Their hands aren't in view" / "Ask them to step back"), a 1-tap `Continuous · Balanced` quick-tune mode pill, a head-and-shoulders outline until a body is tracked (hidden during reply display), tracked hands drawn as a soft glow, and the caption card at the bottom: a 3 dp live lock-in bar across the top edge, an auto-scrolling full-width caption ribbon with a soft left-edge fade (newest word large in cyan, tentative word muted with anti-flicker hysteresis), and under it the sign's other names ("same sign: physician") or "taught on this phone". When the model is not sure, "Not right?" offers two other signs it could have been: a tap swaps the word on screen and in the conversation and speaks it. Compact labeled pills underneath let the user **Speak** the full phrase aloud, **Undo** the last word (long press clears all), or **Clear**; both undoable for 5 s. Frosted bottom dock: **Speak** (auto-speak toggle), **Reply** (offline voice recognition — automatically opens offline Type & Quick-Reply if STT is unavailable), **Type** (offline typed message + 10 curated quick phrases + direct animated ASL launch), and **Pause**. Replies show large high-contrast text with **Show to signer** (full screen + 180° Rotate) and **Watch in sign / Show in sign** (animated real-signer ASL sequence). The conversation under way pulls up in a sheet with copy, share, quick-reply, and All conversations. Keeps the screen on. |
| **Learn** | Unified hero progress card (streak badge, custom progress bar, "Continue lesson") over 300 signs in 21 topics (lessons of about five). A lesson practises its signs in order: a stick figure loops the sign (Play/Pause, full or half speed, "Hands" zoom, pronunciation speaker), the learner copies it on the bordered front camera beside it (with framing guide and live match meter), and the card says "Yes, that was DRINK" or what it looked like instead. **Sign dictionary** (offline text-to-sign translator & 300-sign visual browser), **Practise any sign**, and **Teach the app a sign** open from here. Progress is kept on the phone. |
| ↳ **Sign dictionary & Text-to-Sign** | Fully offline English-to-ASL translator and searchable 300-sign visual dictionary built directly on `demos.bin` (737 KB) and `sets.json` synonyms (zero extra model size, zero network). Type or pick a phrase to watch the animated real-signer ASL sequence with interactive highlighted gloss pills, speed/hand-zoom controls, pronunciation audio, and 1-tap **"Practise"** jump straight into live camera verification, or browse/filter all 300 signs by topic. |
| ↳ **Teach the app a sign** | Teach a sign the model does not know (a name sign, a word from another sign language) in four steps: name it (with quick idea chips; says when already known or taught); record 3 to 5 examples of 2.5 s after a 3-2-1 countdown, each replayed from its keypoints with a `"Hands N%"` visibility badge and deletable; review and save (builds multi-crop `Cores.windows` 768-d prototypes for timing-robust continuous recognition); sign it on the Try-it step with a live similarity match meter. Saved signs are captioned and spoken on Talk and Videos like any other word; rename and delete. Only keypoints are stored, never video. |
| **Videos** | Pick a video of one sign (system photo picker). Runs the same pipeline as the scored `predict()` and shows the sign, its other names, the five closest signs, and quick **Speak**, **Watch sign**, and **Practise** actions right on the result card; with developer options on, timings for frames, landmarks, model and total. |
| **Past conversations** | From Talk's conversation sheet or Settings. Searchable signed words and spoken/typed replies, grouped into conversations at pauses of more than 30 minutes, with detail, copy (with clipboard confirmation), share and delete. Kept in `files/conversation.tsv` for 30 days by default (1 day, 30 days or until deleted). |
| **Settings** | Conversation (captions appear as they sign / after each sign, caption style More words / Balanced / Fewer, surer, keep conversations, show signs in capitals, delete all), Voice (speak aloud, speed, whether spoken replies are available), Camera (which camera Talk uses, highlight tracked hands), Vibration, Reset learning progress, About (privacy note, app permissions, open-source licences, Set up again). **Developer options**, unlocked by tapping the version seven times: confidence, stability and listening window for captions after each sign, taught-sign match, show details (best guesses, frame rate, timings, voice delay), hand skeleton, track on the GPU (off by default: two CPU trackers are faster), model figures. |

First launch asks who holds the phone (back camera toward someone signing, or front camera toward yourself),
explains the camera before Android asks for it, checks the framing with a live outline, then checks the installed
voice and on-device speech recognition; Settings > Set up again repeats it. The Videos tab works without the camera.
Nothing the app keeps (conversations, taught signs, progress, settings) is backed up or transferred off the phone.

### Captions while the hands stay up

The live spotter (`signcore/Spotter.kt`, continuous mode) captions each sign as it is made; the signer does not
lower the hands between words. While a wrist is above a rest line (one shoulder width under the shoulders), and
for 600 ms after both have dropped, the last 1.2 s are classified every 160 ms by the unified EchoGesture model
(`src/models/echogesture_int8.onnx`, `5.70 MB`, packed once into `assets/` beside `live.json`, its measured
decoder settings — keeping total ML weight incl. Holistic strictly **19.38 MB < 20 MB** and loading a single
ONNX Runtime session in RAM). That model was trained on both whole signs and continuous windows and outputs word
probabilities, a transition ("no sign") probability, and `768-d` embeddings. A run of windows that name a sign is
one sign: each window adds its five best answers' probabilities as evidence, and the sign with the most evidence
is captioned when the run ends, or at once when the evidence is already enough. The leading word is shown muted
until then. Settings > Caption style (More words / Balanced / Fewer, surer) scales the evidence a caption needs.

With all 2,000 words open, on held-out test clips joined without rests, the unified 2-model diverse souped ensemble (`src/models/echogesture_int8.onnx`, `5.70 MB` INT8, **68.24%** whole-clip top-1) achieves cross-fitted INT8 word error rate of **40.3%** (**39.4%** mixed, **41.5%** with rests; shipped fixed `commit=2.0` INT8 config: **39.9% / 39.0% / 41.5%**; **26.7–28.9%** same-sign; `scripts/live_cont.py --onnx`; full table in the main README). `SpotterTest` replays six
streams through the Kotlin spotter with the unified INT8 model and requires the same words at the same frames
as Python, in both modes (`scripts/make_stream_fixture.py --live` writes `parity/streams.json`).

Settings > Captions appear > "After each sign" brings back the 15 Nov behaviour: a stretch opens when a wrist
rises above the rest line and closes when both wrists have been back below it for 200 ms; the whole stretch, with
300 ms of lead-in, is classified once, the way a clip is, and its word is shown if it reaches the confidence
slider (15% by default). A stretch longer than 4 s falls back to a sliding window (1.5 s every 5 frames, two
looks in a row above 35%).

### Tap to correct

With 2,000 words open the top word is the sign that was made for **81.48%** of test clips, and the right sign is usually
the next guess. When the model's confidence in the shown sign is under 80% (`SignEngine.ALT_BELOW`), the caption
card offers the best word of each of the next two signs in the ranking (`SameSignSets.alternatives`; names of the
shown sign are already listed as "same sign"). A tap replaces the newest word in the caption and in the conversation
(`SignEngine.correctLast`, `Spotter.replaceLast`: only while that word is still the newest) and speaks the
correction. `scripts/alternatives_eval.py` scores the same rule on the WLASL2000 test clips: sign on screen **93.09%**
with offers under 61.6% of captions (**93.88%** if always offered); captions shown without offers are right **96.74%** of the
time. `--es bench live --ez ui true` replays a video through the real screens with the camera off, to look at it.

### Spoken captions

Each word the spotter settles is spoken by the phone's own text-to-speech voice (an installed voice, so it works
offline; Settings warns when none is installed). The status line shows how long after the camera frame that
settled the word the sound started, last and median. Measured on a Galaxy S26 with the unified 2,000-word model
(`--es bench live --es file stream.mp4 --ez voice true`, replayed at 25 fps, 24 words spoken): the offline
text-to-speech engine starts **52 ms** (median) after the frame that settled the word, **91 ms** at p90 and 588 ms
at worst (`on_device: true`, 0 network). The engine's start is what is timed; the speaker's own output delay is not
included.

### Spoken, typed & animated ASL replies (two-way conversation)

Reply on the Talk tab turns the hearing person's speech into large text for the signer, so the conversation runs
both ways: signs to voice, voice/text to text and animated ASL. It uses Android's on-device recogniser only
(`SpeechRecognizer.createOnDeviceSpeechRecognizer`, Android 12 and later) and asks for the microphone the first
time; audio never leaves the phone and the app still has no internet permission. While it listens, the app's own
voice pauses so it does not hear itself. When offline speech recognition is unavailable (or whenever the user taps **Type**), the offline Type & Quick-Reply sheet lets them pick curated everyday phrases or type any reply. Whenever any words in a spoken or typed reply match the 300 bundled `demos.bin` signs (including 2-word phrases, everyday aliases, morphological stems, and `sets.json` synonyms via `SignTranslator.kt`), **"Watch in sign" / "Show in sign"** plays the reply back as an animated real-signer ASL sequence (`SignSequencePlayer.kt`) with 180° Rotate support for across-the-table conversation. Replies join the conversation beside the signed words.

### Teach a sign

The unified 2-model ensemble returns a pooled **768-number** unit-normalized ensemble embedding shared across live captions and clips. A taught sign is the unit mean of its examples' multi-crop (`Cores.windows`)
unit embeddings (a prototype); a live window or a clip whose embedding is at least 0.55 cosine from a prototype
(developer options, 0.45 to 0.65) is that word, the closest prototype winning; otherwise the classifier answers. The
examples are stored as model inputs (`files/taught/<id>/ex_N.bin`, about 120 KB each), not prototypes, so prototypes
rebuild automatically whenever the model updates. Measured
offline (`scripts/fewshot_eval.py`, baseline WLASL-only model, 20 ASL Citizen signs it never saw, examples and
test clips from different signers, 300 episodes): 79.3% right with 1 example, 90.3% with 3, 92.5% with 5.
On Indian Sign Language (INCLUDE, 23 signs, 0% training overlap, 11 taught per episode) the unified 2-model EchoGesture ensemble is right **91.52% / 95.81% / 96.82%** with 1 / 3 / 5 examples.

End to end on the emulator without a camera: `adb shell am start -n ai.echogesture.app/.MainActivity --es bench
teach --es words AXE,TWINS --ei examples 3 --es file teach_test.mp4` replays `files/teach_<word>_<n>.mp4` as the
camera while an example is recorded exactly as the tab records one, saves each sign, then replays the test video
through the live captions (a taught caption is logged with "taught") and deletes the signs again. With AXE and
TWINS from three ASL Citizen signers each, all 4 clips of them by other signers in a 10-clip stream were captioned
as the taught word and none of the other 6.

### Learn a sign

`app/.../data/Demos.kt` reads `assets/demos.bin` (`android/models/demos.bin`, from `scripts/make_demos.py`): for each of the 300 most
frequent WLASL words, one train or val clip (the one the EchoGesture model is surest of, with a hand seen in most
of the sign, never a test clip), cut to the sign with a short lead-in, 24 steps of 51 points (pose, both hands) as
bytes; 737 KB in all. The figure is scaled to each sign's own extent. While a word is practised,
`SignEngine.practiceTarget` makes the spotter report, for each settled sign, the practised word's same-sign set's
share of the evidence (`Spotter.watch`); at 0.5 or more the attempt passes. Check on a replayed video:
`--es bench live --es practice DRINK --es file teach_test.mp4` logs one `attempt` line per sign (DRINK passed on its
own clip and failed on the 9 others).

## Code

| Path | What it is |
|---|---|
| `signcore/` | Plain Kotlin shared with the desktop tests: point layout, normalisation (port of `src/normalise.py`), resampling, the ONNX Runtime classifier wrapper, the live `Spotter` (one answer per sign at rest, sliding window for signing without rests; port of `scripts/live_eval.py`), same-sign sets and taught-sign matching (`Taught.kt`) |
| `app/.../engine/SignEngine.kt` | MediaPipe `HolisticLandmarker` lanes (live stream for the camera, video mode for clips; see "Speed"), in-order reassembly, the signing-gated frame rate, the spotter on a worker thread, recording of taught-sign examples, state flows for the UI |
| `app/.../engine/VideoFrames.kt` | Picked videos: one sequential MediaCodec decode, YUV to ARGB at most 480 px, frames handed to tracking as they arrive |
| `app/.../engine/Bench.kt` | On-device benchmarks started from adb (see "Benchmarks on the phone") |
| `app/.../engine/Speaker.kt` | On-device text-to-speech and its frame-to-sound timing |
| `app/.../engine/Listener.kt` | Spoken replies: on-device speech recognition, partial results, error and language-pack handling |
| `app/.../data/TaughtStore.kt` | Taught signs on the phone's private storage |
| `app/.../data/SignTranslator.kt` | Pure-Kotlin offline English→ASL gloss matcher (2-word phrases, aliases, `sets.json` synonyms, stemmer, curated quick phrases) |
| `app/.../ui/teach/SignSequencePlayer.kt` | Frame-accurate multi-sign real-signer ASL animator with tappable highlighted sentence pills, speed & hand-zoom controls |
| `app/.../ui/teach/DictionaryScreen.kt` | Offline Text-to-Sign translator & searchable 300-sign topic-filtered visual dictionary |
| `app/.../engine/Landmarks.kt` | Holistic result → `RawFrame` (hands slotted by distance to the pose wrists), frame preparation (upright, ≤ 480 px like the training videos) |
| `app/.../ui/` | Compose screens, theme and components |
| `app/.../data/SettingsRepository.kt` | Settings in DataStore |

`signcore` has unit tests for the taught-sign logic (`TaughtTest`), core utilities (`CoresTest`), and a parity test that runs the real INT8 model on the desktop JVM and checks that the Kotlin
normaliser and the model's probabilities match Python on recorded test clips (normaliser within 6.8e-6,
worst probability diff 0.0040, **40/40 identical top-1**):

```bash
python -I scripts/make_parity_fixture.py data/index/WLASL_v0.3.json data/kp_hol src/models android/parity --subset 2000
gradle -p android :signcore:test
```

## Speed

Measured on a Samsung Galaxy S26 (SM-S942B, Exynos 2600, Android 17), release build, a signer video replayed like a
30 fps camera (frames that arrive while every tracker is busy are dropped, as CameraX does):

| Live tracking | Frames tracked | Holistic per frame p50 / p95 | App CPU |
|---|---|---|---|
| 1 tracker, GPU (the old default) | 10.2 fps | 80 / 100 ms | 0.6 cores |
| 1 tracker, CPU | 15.2 fps | 53 / 60 ms | 1.4 cores |
| 2 trackers, GPU | 20.9 fps | 76 / 90 ms | 1.1 cores |
| **2 trackers, CPU (default)** | **29.9 fps**, every camera frame | **41 / 48 ms** | 2.1 cores |
| 3 trackers, CPU, 60 fps feed (headroom) | 61 fps | 35 / 40 ms | 3.8 cores |

- **CPU, not GPU.** Holistic is a chain of small models with CPU work between them; on this phone the GPU delegate
  stays at 230–380 MHz of its 797 MHz and is slower per frame than XNNPACK on the CPU. The CPU is also the delegate
  the training keypoints were extracted with (Python MediaPipe's default).
- **Lanes.** Several `HolisticLandmarker`s take camera frames in turn (the next idle one gets the frame); results
  wait in a buffer until every earlier frame is back, so the spotter sees frames in order. Each lane keeps tracking
  across the frames it sees, and the landmarks come from the same model, so the classifier's input is unchanged.
- **Why the full rate matters.** On chained test clips (`scripts/live_eval.py --rate-div 2`), tracking at half the
  rate finds 46.7% of the words instead of 59.2% (word error rate 54.7 vs 48.9) and captions 280 ms later.
- **Signing-gated rate.** While no hand has been seen for 2 s, only one frame per 120 ms is tracked (about 8 fps,
  one lane); the first frame with a hand brings back every frame. No sign starts without a hand, so captions do not
  change; idle seconds cost less.
- **Ten minutes (590 s on final unified APK):** **30.09 fps** mean (`30.25 → 29.77 fps`, `1.6%` throughput drift), Holistic `34.4 / 41.0 ms` p50/p95 (`13.9%` thermal drift, PASS ≤ 20%), SoC `31.4 → 40.2 °C` (`+8.8 °C`, peak `41.0 °C`), skin `31.2 → 36.3 °C` (`+5.1 °C`), zero throttling (`thermal_status = 0`, headroom `0.711`), **0.0%** GPU busy, **179.1%** mean CPU (`176.0%` 60 s steady-state), **445.9 MB** mean PSS (`462.8 MB` 60 s max, single ORT session).
- **Videos tab:** 24 frames over four CPU video-mode trackers in parallel: **0.47 s** per clip p50 (**473.7 ms**, p95 **504.3 ms**, unified ensemble `9.50 ms`). `src/main.py` also supports `ECHO_LANES=2` and caches Holistic landmarkers by resolution.
- **Classifier:** Unified 2-model INT8 ensemble (`5.70 MB`, `FastFeaturesV2` + direct buffers), ONNX Runtime Android, 1 thread (`spin=off`): **3.21 / 3.30 ms** p50 / p95 back to back (**3.15 / 3.26 ms** on 2 threads `spin=on`; **2.98 / 3.02 ms** on 4 threads `spin=on`), `15.9 / 19.0 ms` when paced at 6 Hz (`11.5 / 13.8 ms` on 2t); runs at **13.06 / 16.27 ms** p50 / p95 beside 30.24 fps live tracking (`13.13 / 19.10 ms` over 10 min).
- **Power (on battery, airplane mode, Wi-Fi/BT off, whole phone screen on):** unified classifier at live 6 Hz rate **433 mW** (`+68 ± 14 mW` settled share above `360 mW` screen-on baseline); full 30 fps live pipeline **2,944 mW** (`2.94 W`); nobody in view signing-gated (7.57 fps, `33.6%` CPU) **686 mW** (**77% reduction**).

## Benchmarks on the phone

`engine/Bench.kt` runs repeatable benchmarks without the camera. Push a signer video first:
`adb push clip.mp4 /sdcard/Android/data/ai.echogesture.app/files/bench.mp4`. Then:

```bash
adb shell am start -S -n ai.echogesture.app/.MainActivity --es bench model --ei runs 400
adb shell am start -S -n ai.echogesture.app/.MainActivity --es bench live --ei seconds 600 --ei fps 30 --es lanes cpu,cpu
adb shell am start -S -n ai.echogesture.app/.MainActivity --es bench clip --ei runs 10
adb shell am start -S -n ai.echogesture.app/.MainActivity --es bench power --ei wait 90
adb shell am start -S -n ai.echogesture.app/.MainActivity --es bench power2 --ei wait 60 --ei threads 1
```

`lanes` lists one delegate per tracker (`gpu`, `cpu,cpu`, `gpu,cpu`, ...); `--ez blank true` replays an empty room
and `--ez gate true` turns on the signing-gated rate. `--es file NAME` replays another video from the same folder
(for instance several test clips chained into one), `--ez boundary false` captions with the sliding window alone,
and every caption word is written to logcat (`adb logcat -s EchoBench`), so the captions of a replayed stream can
be read back from the phone; `--ez voice true` speaks them as the app does and times each word from the frame that
settled it to the first sound. `power` waits, so the cable can be pulled, then runs signing on
two CPU lanes, an empty room gated and ungated, and signing on one GPU lane, on battery; `power2` measures the
screen-on baseline (app open, nothing tracked) and then the classifier alone at 6 windows a second (`--ei threads`,
`--ez spin`). Each run writes JSON lines
(every 10 s: frames tracked per second, Holistic and classifier p50/p95, app CPU, battery current averaged from 1 s
samples, voltage, temperatures, thermal headroom, memory) to the app's external files folder.
`scripts/phone_sampler.sh` samples GPU load and clock and the SoC, skin and battery temperatures over adb alongside;
`scripts/phone_bench_report.py` turns a run into the summary `scripts/results_csv.py --phone` reads. Power is taken
only from rows on battery: a charger feeds the phone, so the battery current is not the load.

## Building

The build uses Gradle, the standard Android build tool (Android Studio runs the same build).

1. Install the Android SDK (Android Studio, or the command-line tools) with platform 37 (Android 17, minor 2) and build-tools 36.0.0,
   and accept its licence.
2. `android/local.properties`: `sdk.dir=C:\\path\\to\\Android\\Sdk` (Android Studio writes this for you).
3. Nothing to copy by hand: the `copyModels` task fills the app's assets before every build from `src/models/`
   (`echogesture_int8.onnx`, `labels.json`, `sets.json`, `export.json`, `holistic_landmarker.task`) and
   `android/models/` (`live.json`, `demos.bin`). Total ML models packed into the APK: **19.38 MB** (< 20 MB). The assets folder is not kept in Git.
4. `gradle -p android :app:assembleRelease` → `app/build/outputs/apk/release/app-release.apk` (signed with the
   debug key so it installs directly; use a real key for distribution).

Emulator: the debug build type adds the x86_64 ABI, so `gradle -p android :app:assembleDebug` installs on an
x86_64 emulator image (tested on an Android 16 / API 36 Google APIs image: setup, Talk, Learn (lessons, practice
and Teach), Videos with a DOCTOR test clip showing "same sign: PHYSICIAN", past conversations, Settings, and the
Reply permission and language-pack flow; the emulator has no microphone or signer, so a recognised reply needs a phone; taught signs and Learn are checked
there with the replayed-video benches above). If `:app:mergeLibDexDebug` fails with a file "located outside the root directory", delete `android/.gradle`
and `android/app/build/.transforms` (stale execution history from another checkout path).

Release builds are minified by R8. `app/proguard-rules.pro` keeps `com.google.common.flogger`: MediaPipe's logger
finds its caller by class name on the stack, and without the rule the release app crashes as soon as hand tracking
starts. The arm64 ONNX Runtime library crashes under the emulator's ARM translation, so to try a release build on
an x86_64 emulator add `-PemulatorAbi`.

Windows notes: if Gradle fails with `Unable to establish loopback connection`, set
`JAVA_TOOL_OPTIONS=-Djdk.net.unixdomain.tmpdir=C:\Users\Public\xr2uds` (Java cannot bind Unix-domain sockets
under 8.3-style temp paths). For the desktop parity test only, ONNX Runtime needs a newer `msvcp140.dll` than
the one bundled with JDK 24; point `TEST_JAVA_HOME` at a JDK copy whose `bin` holds the System32 runtime DLLs.

Versions: AGP 9.5.1 (built-in Kotlin, so only the Compose compiler plugin is applied), Kotlin 2.4.21,
Compose BOM 2026.09.00, MediaPipe tasks-vision 1.2.2 (same as the Python package used for training),
ONNX Runtime Android 1.31.1, CameraX 1.6.2, compile SDK 37 (minor 2), target SDK 36, min SDK 28, arm64-v8a.
