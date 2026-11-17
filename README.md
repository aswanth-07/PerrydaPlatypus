# EchoGesture: on-device sign language recognition

Samsung XR Hackathon 2026, Track 2 (EchoGesture: sign language prediction on edge devices).
Team **PerrydaPlatypus**, Vellore Institute of Technology, Vellore: A Aswanth Raj and V. Nikhil.

Video of a signer in, an English word out, entirely on the device. MediaPipe Holistic finds the body and hands,
and the **EchoGesture model** (3.3 MB, INT8 ONNX, 2,000 ASL words) names the sign. A native Android app turns this
into live captions that are spoken aloud, and lets the hearing person reply by voice. It can also teach a sign to a
learner, and learn a new sign from three examples. Nothing leaves the device: the app has no internet permission.

![Architecture](docs/architecture.svg)

## Submission checklist

| Item | Where |
|---|---|
| `predict(video_path) -> dict` | [`src/main.py`](src/main.py), with everything it needs in `src/` ([`src/README.md`](src/README.md)) |
| Results sheet | [`results/results.csv`](results/results.csv), the brief's columns: metric, target, measured, unit, device, runtime, pass |
| Predictions | [`results/predictions.jsonl`](results/predictions.jsonl), one JSON line per WLASL2000 test clip (also as one list in `predictions.json`) |
| Quantized model | [`src/models/echogesture_int8.onnx`](src/models/echogesture_int8.onnx), 3.32 MB INT8 ONNX |
| Conversion script | [`scripts/export.py`](scripts/export.py) (PyTorch → ONNX → INT8), run by [`scripts/finalize_model.sh`](scripts/finalize_model.sh) |
| Reproducible setup, Docker | [Quick start](#quick-start) below, [`Dockerfile`](Dockerfile), [`requirements.txt`](requirements.txt) |
| Source code | `src/` (inference), `scripts/` (training, export, evaluation), `android/` (the app) |
| APK | the latest [GitHub release](../../releases/latest); build steps in [`android/README.md`](android/README.md) |
| Numbers for the presentation | [`docs/presentation.md`](docs/presentation.md), [`docs/architecture.png`](docs/architecture.png) |
| Demo video plan | [`docs/demo_video_plan.md`](docs/demo_video_plan.md) |
| AI disclosure | [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md), summarised [at the end](#ai-disclosure) |

## Results at a glance

The shipped EchoGesture model (`src/models/echogesture_int8.onnx`) against the brief's targets. Each row is in
`results/results.csv` with the device and runtime it was measured on.

| Metric | Target | Measured | |
|---|---|---|---|
| Top-1 accuracy | ≥ 70% | **64.42%** on all 2,878 WLASL2000 test clips (91.97% top-5); **64.63%** end to end from the raw videos. The same model file answering from WLASL's 1,000 most common words: **77.7%**; 300: 88.6%; 100: 91.5% | met up to 1,000 words; below target at 2,000 |
| Latency p95 | ≤ 50 ms | **4.4 ms** on a Galaxy S26 (one CPU thread); 4.3 ms on a laptop CPU thread | pass |
| Model size | ≤ 20 MB | **3.32 MB**; with the MediaPipe Holistic landmarker 17.0 MB | pass |
| Robustness (top-1 drop) | ≤ 5 points | all 2,878 test clips degraded before landmarking: angled −0.5, moving head 0.3, low light 0.5, low resolution 2.3, occlusion 5.0, all combined 5.5 | five of six |
| Average power | ≤ 500 mW | **457 mW** for the whole phone, screen on, while the model runs at the live caption rate; the model's own share is +6 ± 28 mW over the screen-on baseline | pass |
| Thermal drift, 10 min | ≤ 20% | **12.1%** (Galaxy S26, 10 minutes of live captions): throughput 30.2 → 29.8 fps, hand-tracking p95 latency +12.1%; no throttling | pass |

WLASL2000 is a hard benchmark: published pose-based and video-based systems report roughly 55 to 63% top-1 on
this split. Our number counts every clip of the official test split, including 189 clips whose source segment
overlaps a train or validation clip (`scripts/leak_audit.py`, `results/leak_audit.json`). Many of the remaining
errors are one sign filed under two English names (next section): counting those as right, top-1 is 77.5%.

## Quick start

`predict()` needs Python 3.11 and runs on the CPU:

```bash
pip install -r src/requirements.txt
python src/main.py path/to/clips/ --out results/predictions.jsonl
```

```python
import sys; sys.path.insert(0, "src")
from main import predict
predict("clip.mp4")
# {"clip_id": "clip", "prediction": "DOCTOR", "top5": ["DOCTOR", "PHYSICIAN", "NURSE", "DETECTIVE", "DUTY"],
#  "confidence": 0.7411, "latency_ms": 8.96, "pipeline_ms": 1443.1, "keypoints_ms_per_frame": 28.77}
```

With Docker:

```bash
docker build -t echogesture .
docker run --rm -v "$PWD/clips:/clips" -v "$PWD/results:/out" echogesture /clips --out /out/predictions.jsonl
```

`ECHO_VOCAB_FILE=words.txt` restricts the answers to a published word list (one word per line). The other options
are in [`src/README.md`](src/README.md). For the phone, install the APK from the latest release (Android 9 or
newer, arm64; Reply needs Android 12).

## How it works

1. **Keypoints** (`src/holistic_kp.py`). MediaPipe Holistic in video mode gives the pose, both hands, the face, and
   metric 3D points for each hand. Each hand is assigned to a side by its distance to the pose wrists, because
   MediaPipe's own handedness label was wrong on about 10% of frames.
2. **Normalisation** (`src/normalise.py`). 119 points per frame, made isotropic, centred on the shoulders and
   scaled by shoulder width; tracking gaps are filled by interpolation, and the clip is resampled to 64 steps. The
   phone runs a line-for-line Kotlin port of this code, which matches Python to within 7e-6 (`ParityTest`).
3. **The EchoGesture model** (`scripts/model.py`, 2.5 M parameters). Its features are computed inside the graph:
   positions, hand-local shape, motion, and each hand's metric 3D points rotated into its own palm frame, so the
   handshape does not depend on how the hand is turned towards the camera. Depthwise convolution blocks with
   channel attention alternate with transformer blocks. The dominant-hand flip and the same-sign answer rule are
   part of the exported graph.
4. **Export** (`scripts/export.py`). ONNX, then dynamic INT8: per-channel INT8 weights on every Conv, Gemm and
   MatMul, except the input projection, which takes 802 features of very different scales and stays FP32. The INT8
   model agrees with the float model's top-1 on 97.0% of test clips (95.7% with the input projection quantized,
   92.6% with static INT8). The scheme was chosen by that label-free agreement, not by test accuracy.

## Accuracy

| | Top-1 | Top-5 |
|---|---|---|
| WLASL2000 official test split (2,878 clips), INT8 model | **64.42%** | 91.97% |
| The same, end to end from the raw videos (`predict()`) | **64.63%** | 92.01% |
| Counting another name of the same sign as right | 77.5% | |
| Signers in none of the training data (ASL Citizen test split, 11 people, 20,117 clips) | 83.0% | |

- **Unseen signers** decide every model choice, not WLASL test accuracy. They are scored with matched runs that
  never saw those 11 people.
- **End to end**, `predict()` landmarks every frame of a clip (up to 96). That scores 1.2 points higher than 24
  evenly spaced frames (63.45%), because Holistic follows the hands better across consecutive frames. A clip takes
  2.9 s at the median on a laptop CPU when six clips run side by side; `ECHO_FRAMES=24` takes about a third of that.
- **Smaller word lists, same file.** WLASL is published at four sizes. The 2,000-word model, answering only from
  each list (`ECHO_VOCAB_FILE`), scores on the test clips of those words (`scripts/vocab_subsets.py`):

| Word list | Test clips | Top-1 | Top-5 |
|---|---|---|---|
| WLASL100 | 258 | 91.47% | 96.90% |
| WLASL300 | 668 | 88.62% | 96.86% |
| WLASL1000 | 1,876 | **77.72%** | 95.15% |
| WLASL2000 | 2,878 | 64.42% | 91.97% |

So the brief's 70% is met by this one file for any list up to about 1,000 words, and a published scoring
vocabulary needs only its word list, not retraining. Random lists of the same sizes give 93.9%, 87.1% and 75.0%.

## One sign, several names

WLASL's own index lists the same source video segment under two glosses for 590 pairs of its 2,000 words, using
the train and validation listings only (no test listing is used). Joined into sets, 308 sets hold 780 words and
37.9% of the test clips: PRISON and JAIL, STORY and TALE, DOCTOR and PHYSICIAN, HURRY and RUSH are the same signing
under two English names. On words with no such twin the network is right 74.2% of the time; on words with one,
44.8%, and most of those errors are the twin. Two things follow (`scripts/signsets.py`):

- **An answer rule inside the graph.** The set's total probability is kept and shared among its words in
  proportion to p(word)^0.25 × (WLASL clips of the word)^2, following how often WLASL uses each name. It was chosen
  on the validation split with a model that never trained on it (57.10 → 58.55, `scripts/set_rule.py`) and then
  applied unchanged: test top-1 63.03 → **64.42**. A global word-frequency prior gives nothing, so the gain comes
  from the set structure. It is a prior matched to WLASL's naming (on ASL Citizen's labels it costs 1.6 points),
  so `ECHO_SET_RULE=0` answers with the network's own probabilities. The graph returns `probs` (with the rule),
  `set_probs`, `embedding` and `raw_probs`.
- **Captions that name the sign, not a guess between its names.** The app shows the other names under the caption
  ("same sign: PHYSICIAN"), from `src/models/sets.json`.

## Robustness

Every one of the 2,878 test videos was degraded in its pixels before MediaPipe (angled camera, moving head, low
light, occlusion, low resolution, and low light + moving head + occlusion combined) at two strengths, landmarked
again, and scored with the float checkpoint the INT8 file was exported from, answering with the same-sign rule as
the shipped graph does (`scripts/extract_degraded.py`, `scripts/eval_suite.py`):

| Top-1 drop (points) | angled | moving head | low light | low res | occlusion | combined |
|---|---|---|---|---|---|---|
| moderate | −0.5 | 0.3 | 0.5 | 2.3 | 5.0 | 5.5 |
| harsh | −0.2 | 2.1 | 5.8 | 9.6 | 8.4 | 10.6 |

Training on keypoints from degraded copies of every training video, and a clean-view teacher (each degraded copy
also matches the model's answer on its clean clip), keep the moderate drops small. Occlusion and the combined
condition are the weak spots: when a hand is hidden for most of a sign its handshape is gone, and filling short
tracking gaps cannot bring it back (under moderate occlusion only 0.8 of the 4.5 extra points of missing hand
frames are gaps of five frames or fewer).

## On the phone

Samsung Galaxy S26 (SM-S942B, Exynos 2600, Android 17), the release APK, a signer video replayed like a 30 fps
camera (`android/app/.../engine/Bench.kt`, `scripts/phone_bench_report.py`):

| | Measured |
|---|---|
| Live frames tracked | **30.0 fps** over 10 minutes, every frame of a 30 fps camera |
| Holistic per frame | 35 ms p50, 42 ms p95 (two CPU trackers taking frames in turn) |
| EchoGesture model | **4.4 ms** p95 back to back (one thread); 17.3 ms p95 paced like live captions with the phone otherwise idle; 14.6 ms p95 beside live tracking |
| Thermal, 10 minutes of live captions | drift **12.1%** (hand-tracking p95 latency +12.1%, throughput 30.2 → 29.8 fps, first vs last minute); SoC 28.7 → 40.2 °C, skin +6.7 °C, thermal status 0 (no throttling); the phone was charging |
| Power, whole phone, screen on, on battery, airplane mode | screen-on baseline 451 mW; with the model at the live caption rate **457 mW** (its own share +6 ± 28 mW, three paired rounds); full live pipeline at 30 fps **3.05 W** |
| Sign to voice | the phone starts speaking a word **70 ms** (median) after the camera frame that settled it, with an installed voice and no network |
| A picked video (Videos tab) | 0.57 s per clip at the median: decoding, 24 frames tracked, the model |

Two choices made it real time: on this phone MediaPipe's CPU path is faster than its GPU path, and two Holistic
trackers take frames in turn, with their results put back in order. The rate matters for accuracy, not just
smoothness: tracking at half the rate finds 46.7% of the words in a stream instead of 59.2%. While no hand is in
view for two seconds, the app tracks about 8 frames a second and goes back to every frame when a hand appears.

Power follows the tracking rate. The model itself costs almost nothing (it runs six times a second for a few
milliseconds), so the brief's 500 mW limit is met by the model, not by the whole pipeline: real-time hand tracking
costs about 80 mJ per frame on this phone's CPU and GPU alike.

## Live captions

The app captions each sign while the hands stay up; the signer does not have to lower them between words
(`Spotter.kt` in `android/signcore`, `scripts/live_cont.py`).

- **A live-caption copy of the EchoGesture model** (`android/models/echogesture_live_int8.onnx`, 3.3 MB),
  fine-tuned on 1.2 s windows of continuous signing made from the isolated training clips. A window holds the core
  of a sign with the end of another before it and the start of another after it, joined by a direct movement, a
  pose held in the air, or a relaxed wait. It is labelled with the sign at its centre, or "no sign" when it is
  centred on the movement between two. On whole clips it still scores 64.1% top-1.
- **One caption per sign, settled when the sign ends.** The window is classified six times a second while a wrist
  is up. A run of windows that name a sign is one sign; when the run ends, the sign with the most evidence is
  captioned. Until then the leading word is shown muted, the way live speech captions settle a word.

Measured on held-out WLASL test clips joined into streams of six signs (120 streams of each kind), with the INT8
model the phone runs; the decoder settings are cross-fitted on halves of the test clips:

| Word error rate, 2,000 words open | Hands never rest | Mixed | A rest after most signs |
|---|---|---|---|
| Strict | **46.8%** | **46.2%** | **45.3%** |
| Another name of the same sign counts as right | 35.7% | 35.1% | 34.7% |

The caption appears 280 ms before the sign's last frame at the median (280 ms after it at the 90th percentile).
The error rate that remains is mostly the classifier's own: with 2,000 words open it names 64% of whole clips. The
Kotlin spotter is checked against the Python one word for word and frame for frame on six test streams
(`SpotterTest`). The streams are joins of isolated clips; no gloss-annotated continuous ASL was available, and a
real signer's transitions will differ.

## The Android app

`android/`: Kotlin, Jetpack Compose, Material 3. Tabs: **Live**, **Chat**, **Signs**, **Videos**, **Settings**
(screens and build steps in [`android/README.md`](android/README.md)).

- **Live captions, spoken aloud** by the phone's own installed voice, offline.
- **Reply.** The hearing person taps Reply and speaks; Android's on-device speech recognition turns it into large
  text for the signer, and both sides join the Chat tab. Together with the spoken captions this is the two-way
  conversation the brief asks for: signs to voice, voice to text. Audio never leaves the phone.
- **Same-sign captions.** "same sign: PHYSICIAN" under DOCTOR.
- **Tap to correct.** When the model is unsure, the caption offers the two next most likely signs ("or BIBLE ·
  WALLET"); one tap swaps the word and speaks the correction. On the 2,878 test clips the sign that was made is
  then on screen for 89.9% of clips, against 76.8% for the top word alone (`scripts/alternatives_eval.py`). It
  costs nothing extra: it reads the ranking the model already returns.
- **Learn a sign.** The app teaches signs to anyone learning ASL. A moving stick figure shows one of 300 common
  signs (keypoints of a real signer from a WLASL training clip, never video; 737 KB for all 300,
  `scripts/make_demos.py`), mirrored to match the front camera beside it. The learner copies it, and the app says
  whether that was the sign: the practised word's share of the evidence must be at least half. With a replayed
  10-clip stream while DRINK was practised, it passed on the DRINK clip and on none of the other nine.
- **Teach a sign.** Type a word and sign it three to five times. The app averages the examples' embeddings into a
  prototype and from then on captions and speaks that word. A name sign, a regional sign, or a sign from another
  sign language can be added without retraining; only keypoints are stored. Measured offline
  (`scripts/fewshot_eval.py`, a model trained on WLASL only, 20 ASL Citizen signs it never saw, examples and test
  clips from different people): 79.3% right with one example, 90.3% with three, 92.5% with five, while 2% of clips
  of known words trigger a taught sign. On Indian Sign Language (INCLUDE, 23 signs) it is right 87.5% / 95.1% /
  96.5% with 1 / 3 / 5 examples. On the phone (replayed video as the camera), AXE and TWINS were taught from three
  signers each: all four of their clips by other signers were captioned as the taught word, and none of six other
  clips.
- **Videos.** Pick a video of one sign; the same pipeline as `predict()` names it.

## Training

The model was trained on keypoints of WLASL and ASL Citizen (41 more signers) on one laptop GPU (RTX 5080). What
each step bought, measured on held-out data:

| Step | Effect |
|---|---|
| Conv + transformer hybrid instead of a depthwise temporal CNN | +3.7 top-1 (WLASL's 100-word subset, 3 seeds) |
| Holistic keypoints with palm-frame metric 3D hand features | +8.5 |
| Dominant-hand canonicalisation | +2.0 (3 of 3 seeds) |
| Keypoints from degraded copies of every training video | +1.0 clean; robustness drops cut by 3 to 6 points |
| Auxiliary phoneme heads from ASL-LEX 2.0 (training only) | +1.6 (5 of 5 paired seeds) |
| Final fit on train + validation | +4.3 |
| ASL Citizen clips | unseen-signer accuracy 49 → 81; +3.7 top-1 |
| Pretraining on 3,075 signs (WLASL + ASL Citizen), then fine-tuning to the 2,000 WLASL words | +1.9 top-1, +1.6 unseen signers |
| Weight-averaged soup of fine-tunes sharing that pretraining (16 in the shipped model) | +0.9 top-1 over the best single run with 8 members, same size |
| Clean-view teacher in half of the soup | occlusion drop 5.8 → 5.0, degraded top-1 +0.6 |
| Same-sign answer rule in the graph | +1.4 top-1 (63.03 → 64.42) |
| Dynamic INT8 with an FP32 input projection, instead of static INT8 | agreement with the float model 92.6% → 97.0%; +0.5 top-1 |

Tried and rejected, each with its measurement: mixup, a cosine head, a 3× larger model, distillation, test-time
low-light enhancement, phonological re-ranking, phonology-weighted label smoothing, a word-frequency prior,
longer fine-tunes, other INT8 calibration methods, ensembles of two soups, a set-level training loss, relabelled
ASL Citizen clips, stopping early when the model is confident, a nearest-neighbour memory beside the model, and
filling short hand-tracking gaps against occlusion.

## Reproduce

Data (not in the repository): WLASL videos and `WLASL_v0.3.json`, and ASL Citizen. With the training environment
(`pip install -r requirements.txt`, PyTorch 2.11 with CUDA):

1. `src/extract.py`: Holistic keypoints for every WLASL clip; `scripts/extract_aslc.py` the same for ASL Citizen
   (`scripts/aslc_fetch.py` downloads it), `scripts/aslc_cache.py` caches it.
2. `src/normalise.py <kp_dir> <cache.pt>`: the normalised training cache.
3. `scripts/extract_degraded.py`: degraded training copies and the robustness test sets.
4. `scripts/run_exp.sh` → `scripts/train.py`: one training run; `scripts/recipe_train.sh` holds the runs behind the
   shipped model.
5. `scripts/soup.py`: the weight-averaged soup; `scripts/recipe_soup.sh` builds and scores it.
6. `scripts/eval_suite.py`: WLASL test, unseen signers and both robustness strengths, with and without the rule;
   `scripts/signsets.py` builds the same-sign sets.
7. `EXPORT_ARGS="--dynamic --ops Conv,Gemm,MatMul --exclude stem/Conv --set-rule" scripts/finalize_model.sh
   runs/soup_mix16/best.pt`: export to INT8 and copy into `src/models`.
8. `scripts/predict_test.py`: `predict()` end to end on the raw test videos; `scripts/make_results.sh`: the results
   sheet.

The app's extras: `scripts/live_cont.py` and `scripts/export_live.sh` (live captions), `scripts/make_demos.py`
(Learn), `scripts/fewshot_eval.py` (Teach), `scripts/make_parity_fixture.py` and `scripts/make_stream_fixture.py`
(the Kotlin tests' fixtures).

## Repository layout

```
src/           predict() and everything it needs: main.py, normalise.py, extract.py, holistic_kp.py, models/
scripts/       training, export, evaluation and measurement
android/       the app (app/), the shared Kotlin core with its tests (signcore/), the app-only models (models/)
results/       results.csv, predictions, the leak audit
docs/          presentation numbers, architecture diagram, demo video plan
```

## Data and licences

- WLASL (C-UDA, research use): training and the official test split.
- ASL Citizen (Microsoft Research, research licence): extra signers, and the unseen-signer test.
- ASL-LEX 2.0 phonology (CC BY 4.0): auxiliary training heads.
- MediaPipe Holistic landmarker (Apache-2.0).
- INCLUDE (Indian Sign Language, Zenodo 4010759, CC BY 4.0): only to measure teach-a-sign; never trained on.
- How2Sign was not used: the track scores isolated words.

## Limitations

- Top-1 on all 2,000 WLASL words is 64.4%, below the brief's 70% (met at 1,000 words and below). The largest
  remaining lever is more signers per word: in an analysis of the model one step before the shipped soup, words
  with 15 or more training clips scored 76.6% and words with 6 or fewer 42.8%.
- Occlusion and combined degradations drop 5.0 and 5.5 points; a hand hidden for most of a sign cannot be
  recovered from keypoints.
- Live captions were measured on joins of isolated clips, not on natural continuous signing.
- The power limit is met by the model; real-time hand tracking for the live camera costs about 3 W on this phone.

## AI disclosure

An AI coding assistant was used throughout the project, under the team's direction: to write most of the code,
run the experiments and measurements, analyse results and draft documentation. The model is our own network,
trained from scratch on public sign language datasets; no AI service runs at inference time. The full disclosure
(the tool, what it did, what the team did, and what is not AI-generated) is in [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md).
