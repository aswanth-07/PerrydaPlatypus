# EchoGesture for Android

Native Android app (Kotlin, Jetpack Compose, Material 3) that turns signing in front of the camera into
captions, entirely on the phone. The app has no internet permission: video and keypoints cannot leave it.

## Screens

| Tab | What it does |
|---|---|
| **Live** | Full-screen camera with the tracked hand skeleton, a caption band (newest word highlighted), the current top-3 guesses with confidence bars, tracking status, fps and per-stage latency; flip camera, pause, clear, landmarks toggle. Keeps the screen on. |
| **Clips** | Pick a video (system photo picker). Runs the same pipeline as the scored `predict()` and shows the word, top-5 bars and timings for frames, landmarks, model and total. |
| **Transcript** | Every word captioned on Live, grouped into lines at pauses of more than 4 s, with copy, share and clear. |
| **Settings** | Confidence needed, stability (windows that must agree), listening window, which camera faces the signer, hand-tracking overlay, GPU tracking; model card (vocabulary, held-out accuracy, size); privacy note. |

First launch explains what the camera is for before asking for it; the Clips tab works without the camera.

## Code

| Path | What it is |
|---|---|
| `signcore/` | Plain Kotlin shared with the desktop tests: point layout, normalisation (port of `scripts/data_v2.py`), resampling, the ONNX Runtime classifier wrapper, the live `Spotter` |
| `app/.../engine/SignEngine.kt` | MediaPipe `HolisticLandmarker` (live stream for the camera, video mode for clips), the spotter on a worker thread, state flows for the UI |
| `app/.../engine/Landmarks.kt` | Holistic result → `RawFrame` (hands slotted by distance to the pose wrists), frame preparation (upright, ≤ 480 px like the training videos) |
| `app/.../ui/` | Compose screens, theme and components |
| `app/.../data/SettingsRepository.kt` | Settings in DataStore |

`signcore` has a parity test that runs the real INT8 model on the desktop JVM and checks that the Kotlin
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
3. `src/models/` (in the repository) holds `sign_int8.onnx`, `labels.json`, `export.json` (from `scripts/export_v2.py`) and
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
