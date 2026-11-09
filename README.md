# EchoGesture — on-device ASL sign recognition

Samsung XR Hackathon 2026, Track 2 (EchoGesture). Team **PerrydaPlatypus**, Vellore Institute of Technology,
Vellore: A Aswanth Raj and V. Nikhil.

Video of a signer in, an English word out, entirely on the device: MediaPipe Holistic keypoints, a 3.2 MB INT8
classifier, and a native Android app with live captions. Nothing leaves the device; the Android app has no
internet permission.

## Submission at a glance

| Brief requirement | Target | Measured (shipped `src/models/full`, 2,000 words) | |
|---|---|---|---|
| Top-1 accuracy | ≥ 70% | **62.72%** on the 2,878-clip WLASL2000 test split (91.56% top-5); 92.3% at 100 words | below target at 2,000 words, see below |
| Inference latency p95 | ≤ 50 ms | **3.4 ms** classifier, one laptop CPU thread | pass |
| Model size | ≤ 20 MB | **3.18 MB** classifier + 13.7 MB Holistic = 16.9 MB | pass |
| Robustness (top-1 drop) | ≤ 5 points | moderate degradations: ≤ 2.7 on angled camera, moving head, low light; 6.6 occlusion, 5.0 low resolution, 6.2 all combined (union model passes all six) | partly |
| Power, thermal drift | | not measured: needs the phone | open |
| `predict()` output | `clip_id, prediction, top5, confidence, latency_ms` | `src/main.py`, end to end on all 2,878 raw test videos: 61.88% top-1 | pass |

`results/predictions.jsonl` and `results/results.csv` are the scored outputs; `results.csv` uses the brief's columns
(metric, target, measured, unit, device, runtime, pass). WLASL2000 is a hard benchmark: the best published results
are in the low 60s top-1, most of them using RGB video as well as pose, and our number counts every clip of the
official test split, including the 189 clips that share an overlapping source segment with a train or val clip
(without them the shipped model scores 64.75% end to end; we report the lower number; `scripts/leak_audit.py`,
`results/leak_audit.json`).

## Results (held-out test splits, INT8 models as shipped)

| `src/models/…` | Vocabulary | Top-1 | Top-5 | Unseen signers | Size | Classifier p95 |
|---|---|---|---|---|---|---|
| `full` (default) | 2,000 WLASL words | **62.72%** | 91.56% | 83.1% | 3.18 MB | 3.4 ms |
| `core` (app live mode) | 100 WLASL words | **92.25%** | 98.45% | 96.0% | 4.17 MB | 9.1 ms |
| `union` (app "Extended") | 3,075 (WLASL + ASL Citizen) | 60.88% | 90.62% | 83.2% / 77.4% | 3.30 MB | 7.8 ms |
| MediaPipe Holistic | | | | | 13.7 MB | about 30 ms per frame |

- Top-1/top-5: the WLASL official test split (2,878 clips at 2,000 words, 258 at 100), never used for training.
  `union` is scored on the same 2,878 clips with all 3,075 words open. Model choices rest on the unseen-signer score
  below; the robustness sets (degraded copies of the 258 WLASL100 test clips) also informed the occlusion decisions,
  and each INT8 scheme is picked by label-free agreement with its float model on the test clips.
- Unseen signers: ASL Citizen's test split, 11 people in none of the training data (20,117 clips at 2,000 words,
  1,705 at 100, 32,941 at 3,075), scored on matched runs that held those people out. **Models are chosen by this
  number**, not by WLASL test accuracy.
- Latency: ONNX Runtime 1.31.1, `CPUExecutionProvider`, one thread, Intel Core Ultra 9 275HX laptop.
- End to end (`predict()` on the raw test videos, decoding + 24 Holistic frames + classifier, one thread per clip):
  984 ms p50 / 1,136 ms p95 per clip with six clips processed side by side; a single process measured 771 ms p50.
  Holistic is about 30–40 ms per frame, so a live camera at 15–25 fps is limited by Holistic, not the classifier.

Robustness: top-1 drop vs clean on the 258 WLASL100 test clips, degraded in pixels before MediaPipe (angled camera,
moving head, low light, occlusion, low resolution, all combined), each model answering from its whole vocabulary and
measured with the float checkpoints its INT8 file was exported from. At moderate strength: full ≤ 2.7 on angled,
moving head and low light, 6.6 on occlusion, 5.0 on low resolution, 6.2 on all combined; core ≤ 4.3 on five
conditions and 5.8 on occlusion; union ≤ 4.3 on all six. At harsh strength the worst drops are 12.0 (full), 9.7
(core) and 8.9 (union). The drops are on 258 clips, so one clip is 0.39 points. Training on extra
occlusion-only degraded copies raised a single model's accuracy under moderate occlusion by about 2 points and under
all combined by about 4 in a paired test, at no measurable cost on unseen signers; the core model ships with it. The
2,000-word soup gained nothing from it (it was already as robust), so `full` ships without it.

Live captions (60 chained streams of 6 held-out signs each, the app's spotter rules): 16.9% word error rate, 86.7%
of words found, 5.6 false words per 100 with the core model; with all 2,000 words open, 53.1%, 50.0% and 25.6, which
is why the app's live mode starts in the core vocabulary.

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
`ECHO_FRAMES`) and the `Dockerfile` for a container.

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
| Dynamic INT8 with the input projection in FP32, instead of static INT8 (2,000 words) | top-1 agreement with the float model 92.6% → 97.0%; +0.5 top-1 |
| Core (100 words) warm-started from the 2,000-word model | +1.9 top-1 |

Tried and rejected (kept in the record, each with its measurement): mixup, cosine head, a 3× larger model,
distillation, test-time low-light enhancement, phonological re-ranking, phonology-weighted label smoothing, a
label-prior correction for WLASL's word frequencies, a third warm-start cycle, longer fine-tunes, other INT8
calibration methods, ensembles of two soups (no gain after INT8 over one larger soup).

Reproduce: `src/extract_v2.py` (keypoints) → `src/data_v2.py` (cache) → `scripts/extract_degraded.py` (degraded
copies, robustness sets) → `scripts/run_exp.sh` / `scripts/train_v2.py` → `scripts/soup.py` →
`scripts/finalize_model.sh` (export, robustness, copy to `src/models`) → `scripts/predict_test.py` (end to end) →
`scripts/results_csv.py`. `scripts/eval_aslc.py` scores unseen signers; `scripts/live_eval.py` scores live captions;
`scripts/aslc_fetch.py` and `scripts/extract_aslc.py` fetch and landmark ASL Citizen.

## Android app

`android/` — Kotlin, Jetpack Compose, Material 3. Tabs: Live (camera HUD: captions, top-3 confidence bars, tracking
state, timings), Clips (pick a video: the same pipeline as `predict()`), Transcript, Settings (Core 100 / Full 2,000
/ Extended 3,075 vocabulary, sensitivity, camera, GPU). The build copies the models from `src/models`. `signcore` is
plain Kotlin with a parity test against Python on 40 test clips, running the shipped INT8 model: normaliser within
7e-6, probabilities within 0.017, top-1 identical on 39/40 (the other clip is one Python scores as a tie, 0.0671 vs
0.0668). The release APK is attached to the GitHub release `v1.0-submission`; `android/README.md` says how to build
it.

## Data and licences

- WLASL (C-UDA, research use): training and the official test split.
- ASL Citizen (Microsoft Research, non-commercial research licence, accepted by the team): extra signers.
- ASL-LEX 2.0 phonology (CC BY 4.0) for the auxiliary heads.
- MediaPipe Holistic model: Apache-2.0.
- How2Sign was not used: the track scores isolated words.
