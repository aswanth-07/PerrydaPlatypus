# Demo video plan (under 3 minutes)

What the video must show, in the order the jury scores it: the app works with no network, the captions are
right and quick, it holds up when conditions are poor, and it does something useful beyond the brief. Every
number on screen comes from a file in the repository (named at the end); nothing is staged with edited output.

## Setup

- Phone: any arm64 Android 12+ phone with the APK from the latest GitHub release installed.
- Second phone or a camera on a tripod records the screen and the signer together; or screen-record the phone
  and film the signer separately and place the two side by side.
- Signer stands 1.5–2 m from the phone, upper body and both hands in frame, plain background for the first
  shots.
- Settings before recording: finish first-launch setup with "I'm talking with someone who signs" (back camera
  facing the signer), **Speak words on** (Talk tab), Highlight tracked hands **on**; unlock developer options
  (Settings, tap the version seven times) and turn Show details **on** for shot 2 only. Before going
  offline, tap Reply once on an Android 13+ phone so it fetches the offline English speech pack (or add it in
  the phone's speech settings), then check Reply works in airplane mode.
- Rehearse the six signs below until each is captioned on the first try; keep one take per shot.
- Sign one word after another with the hands kept up; there is no need to lower them between words. The word
  under way is shown muted, then settles and is spoken as the sign ends. Make each sign fully and leave a short
  beat between signs; lower the hands only at the end of the sentence.

## Shots

| # | Time | Shot | On screen / voice-over |
|---|---|---|---|
| 1 | 0:00–0:12 | Swipe down the quick settings: airplane mode on, Wi-Fi off. Open EchoGesture. | "No network. Everything you see runs on this phone." Title card: EchoGesture, team, track. |
| 2 | 0:12–0:40 | Talk tab, details on. Signer signs 5–6 everyday words one after another, hands kept up: HELLO, THANK YOU, BOOK, DRINK, MOTHER, HELP (pick the six that rehearse best). The phone speaks each word. | The caption card fills word by word, each word muted until it settles; the details panel shows about 30 fps from 2 CPU trackers, tracking and model milliseconds, "voice N ms after the frame" and the best guesses. Voice-over: real time on the phone, every camera frame tracked, then the EchoGesture INT8 model in under 4 ms. Turn details off after this shot. |
| 3 | 0:40–0:52 | Signer signs DOCTOR. | Caption shows "doctor" with "same sign: physician" under it. Voice-over: WLASL lists the same signing under several words; the app shows every name instead of guessing one. |
| 3a | within shot 2 or 3 | When a word comes out wrong, tap "Not right?" and then the right sign. | The word changes and the phone says the correction. Voice-over: when it is not sure it says so, and the right sign is one tap away (on screen for 9 of 10 test clips). Keep one such correction in the video rather than cutting it. |
| 3b | 0:52–1:12 | The hearing person taps **Reply** (or **Type** offline quick-phrases) and says "Yes, thank you, please wait". | "Listening…" with the words appearing as they are spoken, then the reply banner shows **"Show to signer"** and **"Watch in sign"**: tap **"Watch in sign"** (with 180° Rotate) to play the reply as an animated real-signer ASL sequence right on screen; the conversation sheet shows both sides. Voice-over: true two-way conversation — signs to voice, speech or text back to animated ASL, all offline. |
| 4 | 1:12–1:32 | Dim the room lights (or turn off a lamp), then a hand partly behind a cup or the other arm. Signer repeats two words from shot 2. | Still captioned. Cut-in card with the measured drops on all 2,878 test clips (see "Numbers card"). |
| 5 | 1:32–1:47 | Learn tab: open **Sign dictionary** to type a short English phrase and watch the offline ASL animation, then tap **Practise** (or open DRINK directly). The stick figure shows DRINK; a learner copies it with the front camera and live match meter. | "Yes, that was DRINK." Voice-over: the app also translates English phrases to ASL and teaches 300 signs from keypoints of real signers, checking the attempt live on camera. |
| 5a | 1:47–2:15 | Learn tab, **Teach the app a sign**. Type a word the model does not know (a name sign, or an Indian Sign Language word such as the ISL sign for a colour). Record three examples with the 3-2-1 countdown. Review and save. Sign it on the "Try it" step or the Talk tab. | Each example replays from its keypoints; Save; then the caption shows the new word with "taught on this phone", spoken aloud. Voice-over: only keypoints are stored, never video; in offline tests on signs a recogniser had never seen, 3 examples were right about 9 times in 10. |
| 6 | 2:15–2:30 | Videos tab: pick a saved video of one sign. | The sign, its other names, the five closest signs and (developer options on) per-stage timings: the same pipeline as the scored `predict()`. |
| 7 | 2:30–2:50 | Numbers card, then end card. | See below. End card: repository URL, "fully on-device, no internet permission". |

## Numbers card (fill from the shipped files on the day of recording)

| What | Value | Source |
|---|---|---|
| Vocabulary | 2,000 ASL words (WLASL2000) | `src/models/labels.json` |
| WLASL2000 official test split (INT8, 2,878 clips) | **68.24%** top-1 (`0.00 pt` quant drop from FP32), **94.51%** top-5 (**81.48%** same-sign) | `results/results.csv`, `src/models/export.json` |
| The same unified model on WLASL's 1,000 / 300 / 100 most common words | **79.96% / 90.72% / 93.41%** top-1 | `results/results.csv`, rows `top1_accuracy_wlasl*` |
| Live captions, word error rate on chained test clips | **40.3%** no-rest (**39.9%** fixed), **39.4%** mixed (**39.0%** fixed), **41.5%** rest (**26.7–28.9%** same-sign) | `results/results.csv`, rows `live_caption_word_error_rate_*` |
| Model size, latency p95 on the phone (1 CPU thread) | **5.70 MB** unified ensemble (**19.38 MB** total w/ Holistic across repo, ZIP & APK), **3.30 ms** p95 (`3.02 ms` 4t) | `results/results.csv`, rows `model_size*`, `latency_p95_phone` |
| Power (Galaxy S26, battery, airplane mode) | **433 mW** whole phone w/ model at 6 Hz (`+68 mW` settled share over `360 mW` baseline); **686 mW** idle-gated (`2,944 mW` active 30 fps) | `results/results.csv`, rows `avg_power*`, `power_idle_gated` |
| Hand tracking | MediaPipe Holistic, 13.68 MB (19.38 MB total with unified ensemble) | `src/models/holistic_landmarker.task` |
| Robustness, moderate (drop in points) | **6 / 6 PASS, all ≤ 4.0 pt** (angled 0.59, low light 0.87, moving head 0.90, low res 2.36, occlusion 3.75, combined 3.99) | `results/results.csv`, rows `robustness_drop_moderate_*` |
| Voice delay on the phone | speech starts **52 ms** p50 (**91 ms** p90) after the frame that settles the word (`on_device: true`) | `results/results.csv`, row `voice_delay_p50_phone`; live on the app's status line with details on |
| Live speed on the phone | **30.24 fps**, every camera frame tracked, **16.27 ms** unified ensemble p95 under load, thermal drift **13.9%** over 10 min (`thermal_status = 0`) | `results/results.csv`, rows `live_fps_phone`, `classifier_p95_phone_under_load`, `thermal_drift_10min` |
| Video recognition on the phone | **0.47 s** per clip (`473.7 ms` p50, `504.3 ms` p95) | `results/results.csv`, row `video_recognition_p50_phone` |
| Teach a sign | **91.5% / 95.8% / 96.8%** Indian Sign Language (1/3/5-shot); 90.3% / 92.5% unseen ASL | `results/results.csv`, README |

## Wording rules for the video

- Say what was measured and where; never "state of the art" or "novel".
- The 70% target is not met at 2,000 words; say the measured number plainly (**68.24%**), and that the same model file is at
  **79.96%** when it answers from WLASL's 1,000 most common words (`85.41%` on unseen ASL Citizen signers).
- Occlusion and the combined condition are the weakest; if a take fails there, keep it and say so rather than
  cutting to a lucky one.
