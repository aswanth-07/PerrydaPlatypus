# Demo video plan (2–3 minutes)

What the video must show, in the order the jury scores it: the app works with no network, the captions are
right and quick, it holds up when conditions are poor, and it does something useful beyond the brief. Every
number on screen comes from a file in the repository (named at the end); nothing is staged with edited output.

## Setup

- Phone: any arm64 Android 9+ phone with the release APK installed (`android/app/build/outputs/apk/release/`).
- Second phone or a camera on a tripod records the screen and the signer together; or screen-record the phone
  and film the signer separately and place the two side by side.
- Signer stands 1.5–2 m from the phone, upper body and both hands in frame, plain background for the first
  shots.
- Settings before recording: vocabulary **Full (2,000 words)**, Voice **on** (Live tab), Show hand tracking
  **on**, back camera facing the signer, Settings > Advanced > Show details **on** for shot 2 only. Before going
  offline, tap Reply once on an Android 13+ phone so it fetches the offline English speech pack (or add it in
  the phone's speech settings), then check Reply works in airplane mode.
- Rehearse the six signs below until each is captioned on the first try; keep one take per shot.

## Shots

| # | Time | Shot | On screen / voice-over |
|---|---|---|---|
| 1 | 0:00–0:12 | Swipe down the quick settings: airplane mode on, Wi-Fi off. Open EchoGesture. | "No network. Everything you see runs on this phone." Title card: EchoGesture, team, track. |
| 2 | 0:12–0:40 | Live tab, details on. Signer signs 5–6 everyday words with short pauses: HELLO, THANK YOU, BOOK, DRINK, MOTHER, HELP (pick the six that rehearse best). The phone speaks each word. | The caption card fills word by word; the details panel shows about 30 fps from 2 CPU trackers, tracking and model milliseconds, "voice N ms after the frame" and the best guesses. Voice-over: real time on the phone, every camera frame tracked by two trackers taking turns, then a 3 MB INT8 model, about 10 ms per window on the phone. Turn details off after this shot. |
| 3 | 0:40–0:52 | Signer signs DOCTOR. | Caption shows DOCTOR with "same sign: PHYSICIAN" under it. Voice-over: WLASL lists the same signing under several words; the app shows every name instead of guessing one. |
| 3b | 0:52–1:12 | The hearing person taps **Reply** and says "Nice to meet you, how are you?". | "Listening…" with the words appearing as they are spoken, then the reply large at the top for the signer; the Chat tab shows both sides. Voice-over: the conversation runs both ways, signs to voice and voice to text, all on the phone. |
| 4 | 1:12–1:32 | Dim the room lights (or turn off a lamp), then a hand partly behind a cup or the other arm. Signer repeats two words from shot 2. | Still captioned. Cut-in card with the measured drops on all 2,878 test clips (see "Numbers card"). |
| 5 | 1:32–2:12 | Teach tab. Type a word the model does not know (a name sign, or an Indian Sign Language word such as the ISL sign for a colour). Record three examples with the 3-2-1 countdown. Save. Sign it on the Teach preview or the Live tab. | The progress dots fill and turn green at three; Save; then the caption shows the new word with "taught on this phone", spoken aloud. Voice-over: only keypoints are stored, never video; in offline tests on signs a recogniser had never seen, 3 examples were right about 9 times in 10. |
| 6 | 2:12–2:30 | Videos tab: pick a saved video of one sign. | The sign, its other names, the five closest signs and per-stage timings: the same pipeline as the scored `predict()`. |
| 7 | 2:30–2:50 | Numbers card, then end card. | See below. End card: repository URL, "fully on-device, no internet permission". |

## Numbers card (fill from the shipped files on the day of recording)

| What | Value | Source |
|---|---|---|
| Vocabulary | 2,000 ASL words (WLASL2000) | `src/models/full/labels.json` |
| Top-1 / top-5, held-out test split | from `int8.test` | `src/models/full/export.json` |
| Classifier size, latency p95 (1 CPU thread) | from `int8.size_mb`, `int8.one_thread.p95_ms` | `src/models/full/export.json` |
| Hand tracking | MediaPipe Holistic, 13.7 MB | `src/models/holistic_landmarker.task` |
| Robustness, moderate (drop in points) | six conditions | `results/results.csv`, rows `robustness_drop_moderate_*` |
| Voice delay on the phone | median "Voice N ms after the frame" | the app's status line; write it into the README |
| Live speed on the phone | 29.9 fps, tracking p95 49 ms, no drift in 10 min | `results/results.csv`, rows `live_fps_phone`, `holistic_p95_phone`, `thermal_drift_10min` |
| Video recognition on the phone | 0.57 s per clip | `results/results.csv`, row `video_recognition_p50_phone` |
| Teach a sign | 90.3% (3 examples), 92.5% (5) | `runs/analysis_2026-11-12/fewshot_wlaslonly.json` |

## Wording rules for the video

- Say what was measured and where; never "state of the art" or "novel".
- The 70% target is not met at 2,000 words; say the measured number plainly and that the 100-word core model is
  at 92%.
- Occlusion and the combined condition are the weakest; if a take fails there, keep it and say so rather than
  cutting to a lucky one.
