# Presentation pack

Everything the deck needs, slide by slide, with the source of every number. The organizers ask the deck to cover:
theme ID, project title and team details; the problem in our own words; the solution and an architecture diagram;
tools and tech stack; innovation highlights, results and limitations. Name the file **`VIT_Vellore_PerrydaPlatypus`**
(CollegeName_TeamName), as PPT or PDF.

All numbers below are measured, on held-out data or on the phone; `results/results.csv` holds each with its device
and runtime. Round as shown here, and do not add numbers that are not in this file or in `results/results.csv`.

---

## 1. Title

- **Theme:** Track 2, EchoGesture: Sign Language Prediction on Edge Devices
- **Project:** EchoGesture, on-device sign language recognition
- **Team PerrydaPlatypus**, Vellore Institute of Technology, Vellore: A Aswanth Raj, V. Nikhil
- Tagline: *Sign in, speech out. Speech in, text out. Nothing leaves the phone.*

## 2. The problem, in our words

- A Deaf signer and a hearing person who does not sign cannot talk without an interpreter, and interpreters are
  rarely there for everyday moments: a shop counter, a clinic desk, a classroom.
- Cloud translation is not acceptable: it would stream video of the signer and of bystanders off the phone.
- So the whole chain must run on the phone: camera → keypoints → model → caption and voice, in real time, without
  draining the battery, and still work with poor light, a moving camera, an angled signer or hidden hands.
- The brief's budgets: INT8 model ≤ 20 MB, ≤ 50 ms latency, no cloud, 2,000 WLASL words.

## 3. Our solution

One line: **a 3.3 MB INT8 model that reads signs from body and hand keypoints, inside an Android app that turns
signing into spoken captions and speech into text, fully offline.**

- MediaPipe Holistic finds the body, both hands and the face in each camera frame.
- The **EchoGesture model** names the sign from 64 steps of keypoints: 2,000 ASL words.
- The app captions each sign as it is made, says it aloud, and lets the hearing person reply by voice.
- The app has no internet permission at all.

## 4. Architecture

Use `docs/architecture.png` (or `docs/architecture.svg`). Talking points:

1. Camera frame → **MediaPipe Holistic** (13.7 MB): body, hands, face, plus metric 3D hand points.
2. **Normalise**: 119 points per frame, centred on the shoulders, scaled by shoulder width, gaps filled. The same
   code runs in Python and Kotlin, matched to within 7e-6.
3. **EchoGesture model** (3.32 MB INT8 ONNX, 2.5 M parameters): handshape in each palm's own 3D frame, motion and
   position features computed inside the graph; convolution + transformer blocks; the same-sign answer rule built
   into the graph.
4. Outputs → `predict()` for the scoring, and the Android app for people.

## 5. Results against the brief

| Metric | Target | Ours | |
|---|---|---|---|
| Top-1 accuracy, 2,000 words | ≥ 70% | **64.4%** (64.6% end to end from raw video) | below |
| Top-1, same model, WLASL's 1,000 / 300 / 100 most common words | ≥ 70% | **77.7% / 88.6% / 91.5%** | pass |
| Latency p95 | ≤ 50 ms | **4.4 ms** on the phone, 1 CPU thread | pass |
| Model size | ≤ 20 MB | **3.32 MB** (17.0 MB with the hand tracker) | pass |
| Robustness drop | ≤ 5 points | **≤ 2.3** for angled, moving head, low light, low resolution; 5.0 occlusion; 5.5 all combined | 5 of 6 |
| Average power | ≤ 500 mW | **457 mW**, the whole phone with the screen on, model running at the live rate | pass |
| Thermal drift, 10 min | ≤ 20% | **12.1%**, no throttling | pass |

Sources: `results/results.csv` rows `top1_accuracy`, `top1_accuracy_wlasl*`, `latency_p95_phone`, `model_size`,
`model_size_with_landmarker`, `accuracy_drop_robustness`, `robustness_drop_moderate_*`, `avg_power`,
`thermal_drift_10min`.

Say it plainly: 70% is not reached at 2,000 words; published pose and video systems report about 55 to 63% on this
split. The same file passes 70% for any word list up to 1,000 words.

## 6. Accuracy in detail

| | Top-1 | Top-5 |
|---|---|---|
| WLASL2000 test split (2,878 clips) | 64.4% | 92.0% |
| End to end, `predict()` on the raw videos | 64.6% | 92.0% |
| Counting another name of the same sign as right | 77.5% | |
| People in none of the training data (ASL Citizen, 11 signers, 20,117 clips) | 83.0% | |

- Models were chosen by the unseen-signer score, never by WLASL test accuracy.
- Sources: `results/results.csv` (`top1_accuracy`, `top5_accuracy`, `top1_end_to_end`, `top1_unseen_signers`);
  README "Accuracy" and "One sign, several names".

## 7. Robustness

Every one of the 2,878 test videos was degraded in its pixels, then landmarked and scored again.

| Top-1 drop (points) | Angled | Moving head | Low light | Low res | Occlusion | All combined |
|---|---|---|---|---|---|---|
| Moderate | −0.5 | 0.3 | 0.5 | 2.3 | 5.0 | 5.5 |
| Harsh | −0.2 | 2.1 | 5.8 | 9.6 | 8.4 | 10.6 |

What did it: training on keypoints from degraded copies of every training video, and a clean-view teacher (each
degraded copy is also trained to agree with the model's answer on the clean clip). Source: `results/results.csv`
rows `robustness_drop_*`.

## 8. On the phone (Samsung Galaxy S26, Exynos 2600, Android 17)

| | |
|---|---|
| Live tracking | **30 fps**, every camera frame (two CPU trackers taking turns) |
| Hand tracking per frame | 35 ms p50, 42 ms p95 |
| EchoGesture model | **4.4 ms** p95, one thread; 14.6 ms p95 beside live tracking |
| Power, airplane mode, on battery | screen-on baseline 451 mW; with the model at the live rate **457 mW** (model's own share +6 ± 28 mW); full live pipeline 3.05 W, almost all hand tracking |
| 10 minutes of live captions | drift **12.1%** (hand-tracking p95 latency +12.1%, throughput 30.2 → 29.8 fps, first vs last minute); SoC 28.7 → 40.2 °C, skin +6.7 °C, thermal status 0 (no throttling); the phone was charging |
| Sign to voice | speech starts **70 ms** after the frame that settles the word (median) |
| A picked video | 0.57 s per clip |

Sources: `results/results.csv` rows `live_fps_phone`, `holistic_p95_phone`, `latency_p95_phone`,
`classifier_p95_phone_under_load`, `avg_power*`, `power_baseline_screen_on`, `thermal_drift_10min`,
`soc_temp_rise_10min`, `voice_delay_p50_phone`, `video_recognition_p50_phone`.

## 9. Live captions: no pause needed between signs

- A live-caption copy of the model, fine-tuned on 1.2 s windows of continuous signing built from isolated clips
  (signs joined by direct movements, held poses or relaxed waits).
- Classified six times a second while a hand is up; one caption per sign, settled as the sign ends (280 ms before
  its last frame at the median).

| Word error rate, 2,000 words open | Hands never rest | Mixed | Rest after most signs |
|---|---|---|---|
| Strict | 46.8% | 46.2% | 45.3% |
| Another name of the same sign counts | 35.7% | 35.1% | 34.7% |

Source: `results/results.csv` rows `live_caption_word_error_rate_*`, `live_caption_delay_p50`. Caveat to say: the
streams are joins of isolated test clips, not natural conversation.

## 10. Innovation highlights

1. **One sign, several names.** WLASL's own index files the same video segment under two words for 590 word pairs
   (DOCTOR / PHYSICIAN, PRISON / JAIL). We group them into 308 same-sign sets (from train and validation listings
   only) and build an answer rule into the model graph: **+1.4 top-1** (63.0 → 64.4), and captions that show every
   name ("same sign: PHYSICIAN").
2. **Pose-invariant handshape.** Each hand's metric 3D points are rotated into its own palm frame inside the model,
   so the handshape reads the same however the hand is turned. Holistic keypoints with these palm-frame features
   gave **+8.5 top-1** in our design experiments.
3. **Real time on the phone's CPU.** Two hand trackers take frames in turn, so every frame of a 30 fps camera is
   tracked; tracking at half the rate finds 46.7% of words instead of 59.2%.
4. **Captions without pauses.** A live model trained on synthetic continuous signing captions each sign while the
   hands stay up.
5. **INT8 chosen without peeking at labels.** Dynamic INT8 with the first layer kept in FP32 agrees with the float
   model on 97.0% of clips (static INT8: 92.6%); the scheme was picked by that agreement, not by test accuracy.
6. **Private by construction.** The app has no internet permission; only keypoints are ever stored.

## 11. Beyond the brief (brownie features)

| Feature | What it does | Measured |
|---|---|---|
| Spoken captions | The phone's own offline voice says each word | voice starts 70 ms after the settling frame |
| Two-way reply | The hearing person speaks; on-device speech recognition shows it as large text | audio never leaves the phone |
| Tap to correct | When unsure, the caption offers the two next most likely signs | the right sign is on screen for 89.9% of test clips (76.8% for the top word alone) |
| Learn a sign | The app teaches 300 common signs with a moving figure and checks the learner's attempt | 737 KB for all 300; check passed only on the right clip in a 10-clip test |
| Teach a sign | Add a new sign (a name sign, another sign language) from 3 to 5 examples, no retraining | 90.3% right with 3 examples, 92.5% with 5 (new signs, other people); Indian Sign Language 95.1% with 3 |
| Videos | Pick a video of a sign; same pipeline as `predict()` | 0.57 s per clip on the phone |

## 12. Tools and tech stack

- **Model and training:** Python 3.11, PyTorch 2.11 (CUDA, one RTX 5080 laptop GPU), MediaPipe Holistic 1.2.2,
  ONNX, ONNX Runtime 1.31.1 (INT8 quantization and inference), NumPy, OpenCV.
- **Data:** WLASL (2,000 words), ASL Citizen (41 more signers), ASL-LEX 2.0 phonology labels; INCLUDE (Indian Sign
  Language) only to test teach-a-sign.
- **App:** Kotlin, Jetpack Compose, Material 3, CameraX, MediaPipe Tasks Vision, ONNX Runtime Android, Android
  on-device speech recognition and text-to-speech.
- **Reproducibility:** Docker (`predict()`), Gradle, Kotlin parity tests against Python.
- **AI assistance:** disclosed in `AI_DISCLOSURE.md`.

## 13. Demo and local setup

```bash
pip install -r src/requirements.txt
python src/main.py path/to/clips/ --out results/predictions.jsonl
docker build -t echogesture . && docker run --rm -v "$PWD/clips:/clips" echogesture /clips
```

APK: the latest GitHub release. Video: `docs/demo_video_plan.md`.

## 14. Limitations and next steps

- 64.4% at 2,000 words, below 70%. The biggest lever is more signers per word (rare words score far lower), for
  example the Sem-Lex dataset (41 deaf signers, 91k clips).
- Occlusion and combined degradations drop 5.0 and 5.5 points: a hand hidden for most of a sign cannot be recovered
  from keypoints.
- Live captions were measured on joined isolated clips, not natural conversation.
- The model meets the power limit; real-time hand tracking for the camera costs about 3 W on this phone.

## 15. Thank you

Repository link, team names, "fully on-device, no internet permission".
