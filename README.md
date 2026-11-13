# EchoGesture — on-device ASL sign recognition

Samsung XR Hackathon 2026, Track 2 (EchoGesture). Team **PerrydaPlatypus**, Vellore Institute of Technology,
Vellore: A Aswanth Raj and V. Nikhil.

Video of a signer in, an English word out, entirely on the device: MediaPipe Holistic keypoints, a 3.3 MB INT8
classifier, and a native Android app with live captions spoken aloud, and signs the user can teach it in three
examples. Nothing leaves the device; the Android app has no internet permission.

## Submission at a glance

| Brief requirement | Target | Measured (shipped `src/models/full`, 2,000 words) | |
|---|---|---|---|
| Top-1 accuracy | ≥ 70% | **64.42%** on the 2,878-clip WLASL2000 test split (91.97% top-5); 77.5% when a word WLASL lists for the same signing counts; 92.3% at 100 words | below target at 2,000 words, see below |
| Inference latency p95 | ≤ 50 ms | **4.3 ms** classifier, one laptop CPU thread; **4.5 ms** on a Galaxy S26 (one thread; 15.5 ms when paced like live captions, cores idle between calls) | pass |
| Model size | ≤ 20 MB | **3.32 MB** classifier + 13.7 MB Holistic = 17.0 MB | pass |
| Robustness (top-1 drop) | ≤ 5 points | all 2,878 test clips, moderate degradations: angled −0.5, moving head 0.3, low light 0.5, low resolution 2.3, occlusion 5.0, all combined 5.5 | five of six pass |
| Thermal drift (10 min) | ≤ 20% | **0.0%**: Galaxy S26, live pipeline at 29.9 fps in the first and the last minute, p95 tracking 49 ms; SoC 38.1 → 41.6 °C, no throttling | pass |
| Power | ≤ 500 mW | **395 mW** (±113) for the INT8 classifier at the live caption rate, above a 712 mW screen-on baseline, on battery. The whole live pipeline at 30 fps draws **2.87 W** for the whole phone, 2.16 W above baseline, almost all of it hand tracking (see "On the phone") | classifier passes; the real-time pipeline does not |
| Real time on the phone | | **29.9 fps** live (every frame of a 30 fps camera), Holistic 41 / 49 ms p50 / p95, captions from the 2,000-word model; a picked video in **0.57 s** | |
| `predict()` output | `clip_id, prediction, top5, confidence, latency_ms` | `src/main.py`, end to end on all 2,878 raw test videos: **63.45%** top-1, 91.45% top-5 | pass |

`results/predictions.jsonl` and `results/results.csv` are the scored outputs; `results.csv` uses the brief's columns
(metric, target, measured, unit, device, runtime, pass). WLASL2000 is a hard benchmark: the best published results
are in the low 60s top-1, most of them using RGB video as well as pose, and our number counts every clip of the
official test split, including the 189 clips that share an overlapping source segment with a train or val clip
(audit: `scripts/leak_audit.py`, `results/leak_audit.json`; we report the number over all clips).

## Results (held-out test splits, INT8 models as shipped)

| `src/models/…` | Vocabulary | Top-1 | Top-5 | Unseen signers | Size | Classifier p95 |
|---|---|---|---|---|---|---|
| `full` (default) | 2,000 WLASL words | **64.42%** | 91.97% | 83.0% | 3.32 MB | 4.3 ms |
| `core` (app live mode) | 100 WLASL words | **92.25%** | 98.45% | 96.0% | 4.15 MB | 10.7 ms |
| `union` (app "Extended") | 3,075 (WLASL + ASL Citizen) | 60.88% | 90.62% | 83.2% / 77.4% | 3.28 MB | 5.7 ms |
| MediaPipe Holistic | | | | | 13.7 MB | about 30 ms per frame (laptop); 41 ms on a Galaxy S26 CPU |

- Top-1/top-5: the WLASL official test split (2,878 clips at 2,000 words, 258 at 100), never used for training.
  `union` is scored on the same 2,878 clips with all 3,075 words open. Model choices rest on the unseen-signer score
  below; the robustness sets (degraded copies of the 258 WLASL100 test clips) also informed the occlusion decisions,
  and each INT8 scheme is picked by label-free agreement with its float model on the test clips.
- Unseen signers: ASL Citizen's test split, 11 people in none of the training data (20,117 clips at 2,000 words,
  1,705 at 100, 32,941 at 3,075), scored on matched runs that held those people out. **Models are chosen by this
  number**, not by WLASL test accuracy.
- Latency: ONNX Runtime 1.31.1, `CPUExecutionProvider`, one thread, Intel Core Ultra 9 275HX laptop.
- End to end (`predict()` on the raw test videos, decoding + 24 Holistic frames + classifier, one thread per clip):
  63.45% top-1 and 91.45% top-5 on all 2,878 videos (the cached-keypoint score above is 64.42); 1,054 ms p50 / 1,267 ms
  p95 per clip with six clips processed side by side on a busy machine (turn 1, one process alone: 771 ms p50).
  Holistic is about 30–40 ms per frame, so one tracker limits a live camera to 15–25 fps; the app runs two
  trackers that take frames in turn and reaches the camera's 30 fps (see "On the phone"). `ECHO_LANES=2` does the
  same for `predict()`: 383 vs 686 ms per clip alone on the laptop, 78.8% vs 79.5% top-1 on the first 600 test clips
  (within noise); the scored path keeps one tracker.

Robustness: every one of the 2,878 WLASL2000 test videos was degraded in pixels before MediaPipe (angled camera,
moving head, low light, occlusion, low resolution, and low light + moving head + occlusion combined) at two
strengths, landmarked again, and scored with the float checkpoint the INT8 file was exported from, answering with
the same-sign rule as the shipped graph does (`scripts/eval_suite.py`, `runs/soup_mix16/suite_both.json`):

| Top-1 drop (points), `full` | angled | moving head | low light | low res | occlusion | combined |
|---|---|---|---|---|---|---|
| moderate | −0.5 | 0.3 | 0.5 | 2.3 | 5.0 | 5.5 |
| harsh | −0.2 | 2.1 | 5.8 | 9.6 | 8.4 | 10.6 |

Occlusion and the combined condition are the weak spots: when a hand is hidden for most of a sign its handshape is
gone, and filling short tracking gaps cannot bring it back (under moderate occlusion only 0.8 of the 4.5 extra
points of missing hand frames are gaps of five frames or fewer). The earlier 258-clip WLASL100 sets, where one clip
is 0.39 points, had made low light and low resolution look like 5-point drops; on all 2,878 clips they are 0.5 and
2.3. For `core` and `union` the 258-clip sets remain the measurement: core ≤ 4.3 on five conditions and 5.8 on
occlusion; union ≤ 4.3 on all six. Training on extra occlusion-only degraded copies helped the core model (about
2 points under moderate occlusion in a paired test), which ships with it.

Live captions (60 chained streams of 6 held-out signs each, the app's spotter rules): 16.9% word error rate, 86.7%
of words found, 5.6 false words per 100 with the core model; with all 2,000 words open, 53.1%, 50.0% and 25.6, which
is why the app's live mode starts in the core vocabulary.

## One sign, several names

The WLASL index lists one source video segment under two glosses for 590 pairs of its 2,000 words (train and
validation listings only; no test listing is used). Joined into sets, 308 sets hold 780 words and 37.9% of the
test clips: PRISON and JAIL, STORY and TALE, DOCTOR and PHYSICIAN, HURRY and RUSH are the same signing under two
English names. On words with no such twin the shipped network alone is right 74.2% of the time; on words with one,
44.8%, and most of those errors are its twin.

Two things follow from it (`scripts/signsets.py`):

- **An answer rule inside the graph.** The set's total probability is kept and shared among its words in
  proportion to p(word)^0.25 × (WLASL clips of the word)^2, which follows how often WLASL uses each name. It was
  chosen on the validation split with a model that never trained on it (57.10 → 58.55) and then applied unchanged:
  test top-1 63.03 → **64.42**, top-5 91.73 → 92.11 (INT8 64.42 / 91.97). A global word-frequency prior gives
  nothing (on the previous soup, 62.72 → 62.4), so the gain comes from the set structure. It is a prior matched to WLASL's naming: on ASL
  Citizen's labels it costs 1.6 points (82.99 → 81.42), and `ECHO_SET_RULE=0` answers with the network's own
  probabilities. The graph returns `probs` (with the rule), `set_probs`, `embedding` and `raw_probs`.
- **Captions that name the sign rather than guess between its names.** Counting a word of the same set as right,
  top-1 is 77.5%. The app shows the other names under the caption ("same sign: PHYSICIAN"), from the `sets.json`
  shipped beside the model.

Training with a set-level loss instead adds nothing measurable beyond the rule (one seed: +0.5 WLASL with the rule,
−0.3 unseen signers).

## Quick start: `predict()`

```bash
pip install -r src/requirements.txt          # Python 3.11
python src/main.py path/to/clips/ --out results/predictions.jsonl
```

```python
import sys; sys.path.insert(0, "src")
from main import predict
predict("clip.mp4")
# {"clip_id": "clip", "prediction": "BOOK", "top5": [...], "confidence": 0.93, "latency_ms": 8.6, ...}
```

Everything `predict()` needs is in `src/` (code, `src/models/`, `src/requirements.txt`); see `src/README.md` for the
options (`ECHO_VOCAB=full|core|union`, `ECHO_VOCAB_FILE` to restrict answers to a published word list,
`ECHO_FRAMES`, `ECHO_LANES=2` for two Holistic trackers per clip, 1.8x faster) and the `Dockerfile` for a container.

## Pipeline

1. **Keypoints** (`src/holistic_kp.py`): MediaPipe Holistic in VIDEO mode, thresholds 0.3; hands get slots by
   distance to the pose wrists (MediaPipe's handedness label was wrong on ~10% of frames). One landmarker is kept
   across clips, with a blank frame fed between clips so tracking restarts (same accuracy, 450 ms faster per clip).
2. **Normalisation** (`src/data_v2.py`): isotropic coordinates centred on the shoulders and scaled by shoulder width,
   gaps interpolated, hand world points wrist-relative. Ported line for line to Kotlin (parity within 7e-6).
3. **Classifier** (`scripts/model_v2.py`, 2.5 M parameters at 2,000 words): features computed inside the graph — positions,
   hand-local shape, motion and palm-frame metric 3D hand features (each hand rotated into its own palm frame, so
   handshape is pose-invariant); depthwise-conv blocks with channel attention interleaved with transformer blocks.
   The dominant-hand flip is part of the exported graph.
4. **Export** (`scripts/export_v2.py`): ONNX, INT8. The 2,000-word model uses dynamic INT8 (per-channel weights,
   activations quantized on the fly) on every Conv and MatMul except the input projection, which takes 802 features
   of very different scales and stays FP32: the INT8 model agrees with the float model's top-1 on 97.0% of test
   clips, against 95.7% with the input projection quantized and 92.6% with static INT8. The core and union models
   use static per-channel INT8 on Conv/Gemm, which was the better of the two for them. Coordinate maths stays FP32.
   The scheme is chosen per model by that label-free agreement, not by test accuracy.

## Training recipe (what each step bought, measured on held-out data)

| Step | Effect |
|---|---|
| Conv+transformer hybrid vs depthwise TCN (same keypoints) | +3.7 top-1 (WLASL100, 3 seeds) |
| Holistic + palm-frame metric hand features | +8.5 |
| Dominant-hand canonicalisation | +2.0 (3/3 seeds) |
| Training on keypoints from degraded video (low light, tilt, shake, occluders, low res) | +1.0 clean, robustness drops cut by 3–6 points |
| Auxiliary phoneme heads from ASL-LEX 2.0 (training only) | +1.6 (5/5 paired seeds) |
| Train+val final fit (2,000 words) | +4.3 |
| ASL Citizen clips (41 more signers) | unseen-signer accuracy 49 → 81 at 2,000 words; +3.7 top-1 |
| Warm start from a 3,075-word (WLASL + ASL Citizen) model, then back to 2,000 | +1.9 top-1 (59.8 → 61.7, deploy runs), +1.6 unseen signers (80.7 → 82.3) |
| Model soup: weight average of 8 fine-tunes sharing that ancestor | +0.9 top-1 over the best single run (61.78 → 62.72), same size |
| Same-sign answer rule in the graph (above) | +1.4 top-1 (63.03 → 64.42), +0.4 top-5 |
| Clean-view teacher: each degraded training copy also matches the model's answer on its clean clip (`--consist 1.0`); half of a 16-member soup trained with it | at equal soup size (4 seeds each) unseen signers unchanged (85.29 vs 85.28 same-set), degraded top-1 +0.6, WLASL +0.2; occlusion drop 5.8 → 5.0. Alone it costs 0.5 on unseen signers, so it ships only mixed |
| Dynamic INT8 with the input projection in FP32, instead of static INT8 (2,000 words) | top-1 agreement with the float model 92.6% → 97.0%; +0.5 top-1 |
| Core (100 words) warm-started from the 2,000-word model | +1.9 top-1 |

Tried and rejected (kept in the record, each with its measurement): mixup, cosine head, a 3× larger model,
distillation, test-time low-light enhancement, phonological re-ranking, phonology-weighted label smoothing, a
label-prior correction for WLASL's word frequencies, a third warm-start cycle, longer fine-tunes, other INT8
calibration methods, ensembles of two soups (no gain after INT8 over one larger soup), a set-level training loss
(inside the noise), 3,137 ASL Citizen clips relabelled to the WLASL word the model maps them to (+0.3 WLASL, −0.1
unseen), stopping early when the model is confident (62.16 at 22 frames, below a fixed 24), a nearest-neighbour
memory beside the classifier (+0.25 unseen for an 18 MB store), and filling short hand-tracking gaps against
occlusion (the gaps are long).

Reproduce: `src/extract_v2.py` (keypoints) → `src/data_v2.py` (cache) → `scripts/extract_degraded.py` (degraded
copies, robustness sets) → `scripts/run_exp.sh` / `scripts/train_v2.py` (`--consist 1.0` for the clean-view
teacher) → `scripts/soup.py` → `scripts/eval_suite.py` (WLASL, unseen signers, prefixes and both robustness
strengths, with and without the rule) → `EXPORT_ARGS="--dynamic --ops Conv,Gemm,MatMul --exclude stem/Conv
--set-rule" scripts/finalize_model.sh full 2000 runs/soup_mix16/best.pt` (export, robustness, copy to
`src/models`) → `scripts/predict_test.py` (end to end) → `scripts/results_csv.py --suite …`. The training and
scoring chains behind the shipped model are `scripts/chain_cons.sh` and `scripts/chain_soups.sh`.
`scripts/signsets.py` builds the same-sign sets; `scripts/eval_aslc.py` scores unseen signers;
`scripts/live_eval.py` scores live captions (word error rate and delay); `scripts/fewshot_eval.py` measures
teach-a-sign; `scripts/aslc_fetch.py`, `scripts/extract_aslc.py` and `scripts/include_prep.py` fetch and landmark
ASL Citizen and INCLUDE.

## On the phone

Samsung Galaxy S26 (SM-S942B, Exynos 2600, Android 17), release APK, 2,000-word model, a signer video replayed like
a 30 fps camera (`engine/Bench.kt`, `scripts/phone_bench_report.py`; `android/README.md` has the commands):

| | Measured |
|---|---|
| Live frames tracked | **29.9 fps** (every camera frame), steady for 10 minutes |
| Holistic per frame | 41 ms p50, 49 ms p95 (two CPU trackers; the old single GPU tracker: 80 / 100 ms, 10 fps) |
| Classifier | 4.5 ms p95 back to back (1 thread); 15.5 ms p95 paced like live captions with the phone otherwise idle; 12 ms p95 beside live tracking |
| Thermal drift, 10 min | 0.0% throughput, −0.9% latency; SoC 38.1 → 41.6 °C (peak 43.2), skin +2.3 °C, thermal status 0 |
| App CPU, memory | 2.1 cores of 10, 458 MB PSS; GPU idle |
| Power, whole phone, screen on, on battery | live at 30 fps **2,869 mW**; nobody in view 724 mW (signing-gated) vs 1,583 mW (gating off); the old single GPU tracker at 10 fps 1,180 mW; the app open with nothing tracked 712 mW |
| Power of the classifier | 395 mW above that baseline at 6 windows a second (2 threads, spin-waiting; the app now uses 1 thread without spinning) |
| Picked video (Videos tab) | 0.57 s p50 per clip, decode + 24 frames + classifier (2.73 s this morning) |

What made it real time: on this phone MediaPipe's CPU path is faster than its GPU path (the GPU stays at a third of
its clock for this chain of small models), and two Holistic trackers take frames in turn, their results put back
in order before the caption logic. Both keep the landmarks the classifier was trained on: the training keypoints
were extracted on the CPU too. The rate matters for accuracy, not just smoothness: on chained test clips, tracking
at half the rate finds 46.7% of the words instead of 59.2% and captions 280 ms later (`scripts/live_eval.py
--rate-div 2`). While no hand is in view for 2 s the app tracks about 8 frames a second on one tracker and goes back
to every frame as soon as a hand appears.

Power follows the tracking rate: about 80 mJ per tracked frame on the CPU and on the GPU alike, so a real-time
tracker cannot fit in 500 mW on this phone, and the brief's limit is met by the model, not the pipeline. Android
measures only the whole phone (screen, radios, system and the app together); the 712 mW baseline is the phone
with the screen on and the app open doing nothing, and the rows above it are what the app adds. The classifier's
own share was inflated by ONNX Runtime's worker threads spin-waiting between calls; the app now runs it on one
thread without spinning, which is as fast when paced like the live captions (15.5 vs 15.8 ms p95).

## Android app

`android/` — Kotlin, Jetpack Compose, Material 3. Tabs: Live (full-screen camera, caption card, Voice / Reply /
Pause), Chat (both sides of the conversation), Teach, Videos (pick a video: the same pipeline as `predict()`),
Settings (Core 100 / Full 2,000 / Extended 3,075 vocabulary, voice, camera; tuning folded under Advanced). The build copies the models from
`src/models`. `signcore` is plain Kotlin with a parity test against Python on 40 test clips, running the shipped
INT8 model: normaliser within 7e-6, top-1 identical on 40/40, probabilities within 0.025 (dynamic INT8 itself moves
by up to 0.037 when its inputs move by 7e-6; the FP32 export does not move), and unit tests for the taught-sign
logic.

Beyond captions:

- **Spoken captions.** Each settled word is said by the phone's own installed voice, offline; the HUD shows how
  many milliseconds after the camera frame that settled the word the sound started.
- **Spoken replies.** The hearing person taps Reply and speaks; Android's on-device recogniser (Android 12+) turns
  it into large text for the signer, and the line joins the Chat tab. Audio stays on the phone. Together with spoken
  captions this is the two-way conversation the brief describes: signs to voice, voice to text.
- **Same-sign captions.** "same sign: PHYSICIAN" under DOCTOR (see "One sign, several names").
- **Teach a sign.** Type a word and record it 1 to 5 times; the app averages the examples' embeddings (the
  classifier's pooled 384 features) into a prototype and from then on captions and speaks that word when a window
  matches it (cosine ≥ 0.55, adjustable). Only keypoints are stored. Measured offline (`scripts/fewshot_eval.py`)
  with a model trained on WLASL only, on 20 ASL Citizen signs it never saw, examples and test clips from different
  people: 79.3% right with 1 example, 90.3% with 3, 92.5% with 5; at 0.55, 2% of clips of known words trigger a
  taught sign and 80.5% of taught-sign clips are caught. A name sign, a regional sign or a word from another sign
  language can be added this way without retraining. On Indian Sign Language (INCLUDE: 23 signs, 472 clips, 11
  taught per episode) the shipped model is right 87.5% of the time with 1 example, 95.1% with 3 and 96.5% with 5.
  INCLUDE does not label its signers, so examples and test clips may come from the same person and these numbers
  are optimistic; and at the same 0.55 threshold, 35-39% of clips of ISL signs that were *not* taught match one
  that was, because signs of one language sit closer to each other than to the ASL words the threshold was set on.
  Someone teaching many signs of another language should raise the threshold in Settings.

The release APK is attached to the GitHub release; `android/README.md` says how to build it.

## Data and licences

- WLASL (C-UDA, research use): training and the official test split.
- ASL Citizen (Microsoft Research, non-commercial research licence, accepted by the team): extra signers.
- ASL-LEX 2.0 phonology (CC BY 4.0) for the auxiliary heads.
- MediaPipe Holistic model: Apache-2.0.
- INCLUDE (Indian Sign Language, Zenodo 4010759, CC BY 4.0): only to measure teach-a-sign on another sign
  language; never trained on.
- How2Sign was not used: the track scores isolated words.
