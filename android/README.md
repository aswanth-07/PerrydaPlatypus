# EchoGesture for Android

Native Android app (Kotlin, Jetpack Compose, Material 3) that turns signing in front of the camera into
captions, entirely on the phone. The app has no internet permission: video and keypoints cannot leave it.

## Screens

| Tab | What it does |
|---|---|
| **Live** | The camera fills the screen. One status chip ("Signing" while the hands are up / "Watching" / "Show your hands" · vocabulary size), a switch-camera button, and the caption card at the bottom: the newest word large, earlier words beside it, and under it the sign's other names ("same sign: PHYSICIAN") or "taught on this phone"; when the model was not sure, two other signs it could have been ("or BIBLE · WALLET") as buttons for 8 s: a tap swaps the word on screen and in Chat and speaks it; a clear button in the card. Three labelled controls: **Voice** (speak signed words aloud), **Reply** (the hearing person speaks; their words appear large at the top for the signer, and fade after 15 s or on tap), **Pause**. Timings, voice delay and best guesses appear only with Settings > Advanced > "Show details". Keeps the screen on. |
| **Chat** | Both sides of the conversation as chat bubbles: signed words (grouped at pauses of more than 4 s) and spoken replies, with copy, share and clear (with a confirmation). |
| **Signs** | Two modes. **Learn a sign**: a stick figure (left) loops one of 300 common signs, at full or half speed ("Slow"), mirrored to match the front-camera preview beside it (right); the learner copies it and the card under them says "Yes, that was DRINK" or what it looked like instead; Skip / Next sign and Choose (search the 300). Practice is not spoken or kept in Chat. **Teach the app**: as below. |
| ↳ **Teach the app** | Teach a sign the model does not know (a name sign, a word from another sign language): type the word, record 1 to 5 examples of 2.5 s after a 3-2-1 countdown, with one progress dot per example. An example with no hands seen is refused; one far from the earlier examples (cosine below 0.5) is flagged to keep or redo. Saved signs are captioned and spoken on Live and Videos like any other word; rename and delete. Only keypoints are stored, never video. |
| **Videos** | Pick a video of one sign (system photo picker). Runs the same pipeline as the scored `predict()` and shows the sign, its other names, the five closest signs and timings for frames, landmarks, model and total. |
| **Settings** | Signs it knows (Core 100 / Full 2,000 / Extended 3,075), Voice (speak aloud, speed, whether spoken replies are available), Camera (which camera faces the signer, hand-tracking overlay); **Advanced**, folded away: caption each sign when the hands rest (on by default), confidence needed, stability and listening window (for signing without rests), taught-sign match, show details, track on the GPU (off by default: two CPU trackers are faster); About (vocabulary, held-out accuracy, size) and the privacy note. |

First launch explains what the camera is for before asking for it; the Clips tab works without the camera.

### Captions while the hands stay up

The live spotter (`signcore/Spotter.kt`, continuous mode) captions each sign as it is made; the signer does not
lower the hands between words. While a wrist is above a rest line (one shoulder width under the shoulders), and
for 600 ms after both have dropped, the last 1.2 s are classified every 160 ms by the vocabulary's live model
(`models/<vocab>/live_int8.onnx`, packed into the app's assets beside `live.json`, its measured decoder
settings). That model was trained on windows of continuous signing and answers with the sign at the window's
centre or with "no sign" between two. A run of windows that name a sign is one sign: each window adds its five
best answers' probabilities as evidence, and the sign with the most evidence is captioned when the run ends, or
at once when the evidence is already enough. The leading word is shown muted until then. Settings > Captions
(Eager / Balanced / Careful) scales the evidence a caption needs. A vocabulary packed without a live model uses
its clip model for the windows.

With all 2,000 words open, on held-out test clips joined without rests, the word error rate is
47.5% against 57.2% for the app as it was (44.4% against 47.1% when the signer does rest; `scripts/live_cont.py`,
settings cross-fitted on halves of the test clips; the table is in the main README). `SpotterTest` replays six
streams through the Kotlin spotter with the shipped INT8 models and requires the same words at the same frames
as Python, in both modes (`scripts/make_stream_fixture.py --live` writes `parity/streams.json`).

Settings > Advanced > "Wait for the hands to rest" brings back the 15 Nov behaviour: a stretch opens when a wrist
rises above the rest line and closes when both wrists have been back below it for 200 ms; the whole stretch, with
300 ms of lead-in, is classified once, the way a clip is, and its word is shown if it reaches the confidence
slider (15% by default). A stretch longer than 4 s falls back to a sliding window (1.5 s every 5 frames, two
looks in a row above 35%).

### Tap to correct

With 2,000 words open the top word is the sign that was made for 76.8% of test clips, and the right sign is usually
the next guess. When the model's confidence in the shown sign is under 80% (`SignEngine.ALT_BELOW`), the caption
card offers the best word of each of the next two signs in the ranking (`SameSignSets.alternatives`; names of the
shown sign are already listed as "same sign"). A tap replaces the newest word in the caption and in Chat
(`SignEngine.correctLast`, `Spotter.replaceLast`: only while that word is still the newest) and speaks the
correction. `scripts/alternatives_eval.py` scores the same rule on the WLASL2000 test clips: sign on screen 89.9%
with offers under 67% of captions (90.8% if always offered); captions shown without offers are right 95.1% of the
time. `--es bench live --ez ui true` replays a video through the real screens with the camera off, to look at it.

### Spoken captions

Each word the spotter settles is spoken by the phone's own text-to-speech voice (an installed voice, so it works
offline; Settings warns when none is installed). The status line shows how long after the camera frame that
settled the word the sound started, last and median. Measured on a Galaxy S26 with the 2,000-word model
(`--es bench live --es file stream.mp4 --ez voice true`, a stream of ten chained test signs replayed at 25 fps, 26
words spoken): the text-to-speech engine starts 70 ms (median) after the frame that settled the word, 368 ms at
p90 and 770 ms at worst, the slow ones being words that waited for the one before them to finish. With captions at rest (how this was measured) the word
settles 200 ms after the hands come to rest, so speech begins about 0.3 s after a sign ends. The engine's start is
what is timed; the speaker's own output delay is not included.

### Spoken replies (two-way conversation)

Reply on the Live tab turns the hearing person's speech into large text for the signer, so the conversation runs
both ways: signs to voice, voice to text. It uses Android's on-device recogniser only
(`SpeechRecognizer.createOnDeviceSpeechRecognizer`, Android 12 and later) and asks for the microphone the first
time; audio never leaves the phone and the app still has no internet permission. While it listens, the app's own
voice pauses so it does not hear itself. If the phone lacks the offline English speech pack, Android 13+ offers to
fetch it (the system downloads it, not the app); otherwise the app says how to add it. Replies join the Chat tab.

### Teach a sign

Every shipped model (Core, Full, Extended) also returns its pooled 384-number embedding; re-exporting Core and Extended for it left their accuracy unchanged. A taught sign is the unit mean of its examples'
unit embeddings (a prototype); a live window or a clip whose embedding is at least 0.55 cosine from a prototype
(Settings, 0.45 to 0.65) is that word, the closest prototype winning; otherwise the classifier answers. The
examples are stored as model inputs (`files/taught/<id>/ex_N.bin`, about 120 KB each), not prototypes, because each
vocabulary's model has its own embedding, so prototypes are rebuilt with whichever model is loaded. Measured
offline (`scripts/fewshot_eval.py`, a model trained on WLASL only, 20 ASL Citizen signs it never saw, examples and
test clips from different signers, 300 episodes): 79.3% right with 1 example, 90.3% with 3, 92.5% with 5; at the
0.55 threshold 2% of clips of known words trigger a taught sign and 80.5% of taught-sign clips are caught.
On Indian Sign Language (INCLUDE, 23 signs, 11 taught per episode, split by clip because the set does not label
signers) the shipped 2,000-word model is right 87.5% / 95.1% / 96.5% with 1 / 3 / 5 examples, but at 0.55 about 37%
of clips of untaught ISL signs also match a taught one: raise the threshold when teaching many signs of one other
language. Vocabularies whose model has no embedding output show a one-line note on the Teach tab instead.

End to end on the emulator without a camera: `adb shell am start -n ai.echogesture.app/.MainActivity --es bench
teach --es words AXE,TWINS --ei examples 3 --es file teach_test.mp4` replays `files/teach_<word>_<n>.mp4` as the
camera while an example is recorded exactly as the tab records one, saves each sign, then replays the test video
through the live captions (a taught caption is logged with "taught") and deletes the signs again. With AXE and
TWINS from three ASL Citizen signers each, all 4 clips of them by other signers in a 10-clip stream were captioned
as the taught word and none of the other 6.

### Learn a sign

`app/.../data/Demos.kt` reads `assets/full/demos.bin` (from `scripts/make_demos.py`): for each of the 300 most
frequent WLASL words, one train or val clip (the one the shipped classifier is surest of, with a hand seen in most
of the sign, never a test clip), cut to the sign with a short lead-in, 24 steps of 51 points (pose, both hands) as
bytes; 737 KB in all. The figure is scaled to each sign's own extent. While a word is practised,
`SignEngine.practiceTarget` makes the spotter report, for each settled sign, the practised word's same-sign set's
share of the evidence (`Spotter.watch`); at 0.5 or more the attempt passes. Check on a replayed video:
`--es bench live --es practice DRINK --es file teach_test.mp4` logs one `attempt` line per sign (DRINK passed on its
own clip and failed on the 9 others).

## Code

| Path | What it is |
|---|---|
| `signcore/` | Plain Kotlin shared with the desktop tests: point layout, normalisation (port of `scripts/data_v2.py`), resampling, the ONNX Runtime classifier wrapper, the live `Spotter` (one answer per sign at rest, sliding window for signing without rests; port of `scripts/live_eval.py`), same-sign sets and taught-sign matching (`Taught.kt`) |
| `app/.../engine/SignEngine.kt` | MediaPipe `HolisticLandmarker` lanes (live stream for the camera, video mode for clips; see "Speed"), in-order reassembly, the signing-gated frame rate, the spotter on a worker thread, recording of taught-sign examples, state flows for the UI |
| `app/.../engine/VideoFrames.kt` | Picked videos: one sequential MediaCodec decode, YUV to ARGB at most 480 px, frames handed to tracking as they arrive |
| `app/.../engine/Bench.kt` | On-device benchmarks started from adb (see "Benchmarks on the phone") |
| `app/.../engine/Speaker.kt` | On-device text-to-speech and its frame-to-sound timing |
| `app/.../engine/Listener.kt` | Spoken replies: on-device speech recognition, partial results, error and language-pack handling |
| `app/.../data/TaughtStore.kt` | Taught signs on the phone's private storage |
| `app/.../engine/Landmarks.kt` | Holistic result → `RawFrame` (hands slotted by distance to the pose wrists), frame preparation (upright, ≤ 480 px like the training videos) |
| `app/.../ui/` | Compose screens, theme and components |
| `app/.../data/SettingsRepository.kt` | Settings in DataStore |

`signcore` has unit tests for the taught-sign logic (`TaughtTest`) and a parity test that runs the real INT8 model on the desktop JVM and checks that the Kotlin
normaliser and the model's probabilities match Python on recorded test clips (normaliser within 7e-6,
probabilities within 0.02, top-1 identical except where Python itself scores a near-tie; with the shipped 2,000-word
model, 39/40 identical and the other clip a 0.0671 vs 0.0668 tie):

```bash
python -I scripts/make_parity_fixture.py data/index/WLASL_v0.3.json data/kp_hol src/models/full android/parity --subset 2000
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
- **Ten minutes:** 29.9 fps in the first and in the last minute (no thermal drift), p95 49 ms, SoC 38.1 → 41.6 °C
  (peak 43.2), skin +2.3 °C, no throttling (thermal status 0, headroom 0.77), PSS 458 MB.
- **Videos tab:** 24 frames over two video-mode trackers at once: 0.57 s per clip p50 (1.69 s with one CPU tracker,
  2.03 s with one GPU tracker). `src/main.py` has the same as an option (`ECHO_LANES=2`: on the first 600 test clips
  78.8% vs 79.5% top-1, a difference within noise, at 1.8× the speed); the scored `predict()` keeps one tracker,
  the configuration every published number was measured with.
- **Classifier:** INT8, ONNX Runtime, one thread with its workers sleeping between calls: 2.7 / 4.5 ms p50 / p95
  back to back, 14.1 / 15.5 ms when paced like the spotter (6 a second; the cores clock down in between), 12 ms
  p95 beside live tracking. Two threads with ONNX Runtime's default spin-waiting were no faster when paced and
  kept a second core busy.
- **Power, on battery, whole phone with the screen on:** live at 30 fps 2,869 mW; nobody in view 724 mW with the
  signing-gated rate, 1,583 mW without; the app open with nothing tracked 712 mW; the classifier alone at 6 windows
  a second 395 mW above that (measured with 2 spinning threads). About 80 mJ per tracked frame on CPU or GPU.

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

1. Install the Android SDK (Android Studio, or the command-line tools) with platform 36 and build-tools 36.1.0,
   and accept its licence.
2. `android/local.properties`: `sdk.dir=C:\\path\\to\\Android\\Sdk` (Android Studio writes this for you).
3. `src/models/` (in the repository) holds `sign_int8.onnx`, `labels.json`, `export.json`, and for the 2,000-word model
   `sets.json` (from `scripts/export_v2.py`) and
   `holistic_landmarker.task`; the `copyModels` task puts them in the app's assets before every build.
4. `gradle -p android :app:assembleRelease` → `app/build/outputs/apk/release/app-release.apk` (signed with the
   debug key so it installs directly; use a real key for distribution).

Emulator: the debug build type adds the x86_64 ABI, so `gradle -p android :app:assembleDebug` installs on an
x86_64 emulator image (tested on an Android 16 / API 36 Google APIs image: Live, Teach recording and its hand check,
Videos with a DOCTOR test clip showing "same sign: PHYSICIAN", Chat, Settings, and the Reply permission and
language-pack flow; the emulator has no microphone or signer, so a recognised reply needs a phone; taught signs and Learn are checked
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
ONNX Runtime Android 1.31.1, CameraX 1.6.2, compile/target SDK 36, min SDK 28, arm64-v8a.
