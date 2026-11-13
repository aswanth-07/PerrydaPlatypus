# EchoGesture for Android

Native Android app (Kotlin, Jetpack Compose, Material 3) that turns signing in front of the camera into
captions, entirely on the phone. The app has no internet permission: video and keypoints cannot leave it.

## Screens

| Tab | What it does |
|---|---|
| **Live** | Full-screen camera with the tracked hand skeleton, a caption band (newest word highlighted), the current top-3 guesses with confidence bars, tracking status, fps and per-stage latency; speak-aloud, flip camera, pause, clear, landmarks toggle. Under the newest word: its other names when WLASL lists the same signing under several words ("same sign: PHYSICIAN"), or "taught on this phone". Keeps the screen on. |
| **Clips** | Pick a video (system photo picker). Runs the same pipeline as the scored `predict()` and shows the word, its other names, top-5 bars and timings for frames, landmarks, model and total. |
| **Teach** | Teach a sign the model does not know (a name sign, a word from another sign language): type the word, record 1 to 5 examples of 2.5 s after a 3-2-1 countdown. An example with no hands seen is refused; one far from the earlier examples (cosine below 0.5) is flagged to keep or redo. Saved signs are captioned and spoken on Live and Clips like any other word; rename and delete. Only keypoints are stored, never video. |
| **Transcript** | Every word captioned on Live, grouped into lines at pauses of more than 4 s, with copy, share and clear. |
| **Settings** | Vocabulary, confidence needed, stability (windows that must agree), listening window, how closely a taught sign must match, voice on/off and speed (with the measured frame-to-sound delay), which camera faces the signer, hand-tracking overlay, GPU tracking; model card (vocabulary, held-out accuracy, size); privacy note. |

First launch explains what the camera is for before asking for it; the Clips tab works without the camera.

### Spoken captions

Each word the spotter settles is spoken by the phone's own text-to-speech voice (an installed voice, so it works
offline; Settings warns when none is installed). The status line shows how long after the camera frame that
settled the word the sound started, last and median.

### Teach a sign

The 2,000-word model also returns its pooled 384-number embedding. A taught sign is the unit mean of its examples'
unit embeddings (a prototype); a live window or a clip whose embedding is at least 0.55 cosine from a prototype
(Settings, 0.45 to 0.65) is that word, the closest prototype winning; otherwise the classifier answers. The
examples are stored as model inputs (`files/taught/<id>/ex_N.bin`, about 120 KB each), not prototypes, because each
vocabulary's model has its own embedding, so prototypes are rebuilt with whichever model is loaded. Measured
offline (`scripts/fewshot_eval.py`, a model trained on WLASL only, 20 ASL Citizen signs it never saw, examples and
test clips from different signers, 300 episodes): 79.3% right with 1 example, 90.3% with 3, 92.5% with 5; at the
0.55 threshold 2% of clips of known words trigger a taught sign and 80.5% of taught-sign clips are caught.
Vocabularies whose model has no embedding output show a one-line note on the Teach tab instead.

## Code

| Path | What it is |
|---|---|
| `signcore/` | Plain Kotlin shared with the desktop tests: point layout, normalisation (port of `scripts/data_v2.py`), resampling, the ONNX Runtime classifier wrapper, the live `Spotter`, same-sign sets and taught-sign matching (`Taught.kt`) |
| `app/.../engine/SignEngine.kt` | MediaPipe `HolisticLandmarker` (live stream for the camera, video mode for clips), the spotter on a worker thread, recording of taught-sign examples, state flows for the UI |
| `app/.../engine/Speaker.kt` | On-device text-to-speech and its frame-to-sound timing |
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

Windows notes: if Gradle fails with `Unable to establish loopback connection`, set
`JAVA_TOOL_OPTIONS=-Djdk.net.unixdomain.tmpdir=C:\Users\Public\xr2uds` (Java cannot bind Unix-domain sockets
under 8.3-style temp paths). For the desktop parity test only, ONNX Runtime needs a newer `msvcp140.dll` than
the one bundled with JDK 24; point `TEST_JAVA_HOME` at a JDK copy whose `bin` holds the System32 runtime DLLs.

Versions: AGP 9.5.1 (built-in Kotlin, so only the Compose compiler plugin is applied), Kotlin 2.4.21,
Compose BOM 2026.09.00, MediaPipe tasks-vision 1.2.2 (same as the Python package used for training),
ONNX Runtime Android 1.31.1, CameraX 1.6.2, compile/target SDK 36, min SDK 28, arm64-v8a.
