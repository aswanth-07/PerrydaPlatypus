# EchoGesture — on-device ASL sign recognition (Samsung XR Hackathon 2026, Track 2)

Video of a signer in, an English word out, entirely on the device: MediaPipe Holistic keypoints, a small INT8
classifier, and a native Android app with live captions. No cloud: the Android app has no internet permission.

## Results (held-out test splits, INT8 models as shipped)

| Model | Vocabulary | Top-1 | Top-5 | Unseen signers | Size | Classifier p95 |
|---|---|---|---|---|---|---|
| `runs/deploy/core` | 100 WLASL words | **91.5%** | 98.1% | 94.5% | 4.2 MB | 15.6 ms |
| `runs/deploy/full` (default) | 2,000 WLASL words | **60.9%** | 91.0% | 80.7% | 5.7 MB | 22.2 ms |
| `runs/deploy/core_v1` | 100 | 89.5% | 97.3% | 93.1% | 4.2 MB | passes every moderate robustness condition |
| `runs/deploy/core_d128` | 100 | 88.8% | | | 2.2 MB | 10.1 ms |
| `runs/deploy/core_wlasl_only`, `full_wlasl_only` | 100 / 2,000 | 88.4% / 57.2% | 96.9% / 89.1% | 78.1% / 48.6% | | trained on WLASL alone |
| MediaPipe Holistic | | | | | 13.7 MB | about 29 ms per frame |

Top-1/top-5 are on the WLASL official test split (258 clips at 100 words, 2,878 at 2,000), never used for training
or model choice. Unseen signers: ASL Citizen's test split, 11 people in none of the training data (1,705 clips at
100 words, 20,117 at 2,000), scored by matched runs that held them out. Latency is on one laptop CPU thread.

Robustness (top-1 drop vs clean on the 258 WLASL100 test clips, degraded in pixels before MediaPipe): at moderate
strength the core model loses ≤ 3.9 points on angled, moving head, low light, low resolution and all combined,
and 6.6 on occlusion; `core_v1` stays ≤ 4.7 on all six. At harsh strength every model loses 7–12 points in low
light, occlusion, low resolution and combined. Live captions (60 chained streams of 6 held-out signs, the app's
spotter rules): 19.4% word error rate, 83.9% of words found, 6.7 false words per 100 with the core model; with all
2,000 words open, 54.4%, 49.2% and 26.4, which is why the app's live mode starts in the core vocabulary.
`runs/deploy/*/results.csv` holds every metric in the brief's format, with targets and pass/fail.

## Quick start: `predict()`

```bash
pip install -r requirements.txt          # Python 3.11
python src/main.py path/to/clip.mp4      # or a folder of .mp4 files; writes predictions.jsonl
```

```python
from src.main import predict
predict("clip.mp4")
# {"clip_id": "clip", "prediction": "BOOK", "top5": [...], "confidence": 0.93, "latency_ms": ...}
```

`ECHO_VOCAB=core|full|core_v1|...` picks the model folder under `runs/deploy` (default `full`, since the scored
vocabulary is not published); `ECHO_FRAMES` sets how many evenly spaced frames are landmarked (default 24,
measured equal to all frames). One Holistic landmarker is kept across clips, with a blank frame fed before each
clip so tracking starts afresh; `ECHO_FRESH_LANDMARKER=1` builds one per clip instead (same accuracy, about 450 ms
slower per clip).

End to end on the 258 raw WLASL100 test videos (`scripts/predict_test.py`, output in
`runs/deploy/core/predictions.jsonl`): 91.5% top-1, 97.7% top-5, 574 ms p50 / 612 ms p95 per whole clip on one
idle laptop core (24 Holistic frames at about 23 ms each, classifier 7.6 ms).
`ECHO_VOCAB_FILE=words.txt` (one word per line, or a JSON list) restricts answers to a published word list:
the full model's probabilities are masked to those words and renormalised.

## Pipeline

1. **Keypoints** (`scripts/holistic_kp.py`): MediaPipe Holistic in VIDEO mode, thresholds 0.3; hands are given
   slots by distance to the pose wrists (MediaPipe's handedness label was wrong on ~10% of frames).
2. **Normalisation** (`scripts/data_v2.py`): isotropic coordinates, centred on the shoulders, scaled by shoulder
   width, gaps interpolated; hand world points made wrist-relative. Ported line for line to Kotlin.
3. **Classifier** (`scripts/model_v2.py`): features computed inside the graph — positions, hand-local shape,
   motion, and *palm-frame metric 3D hand features* (each hand rotated into its own palm frame: pose-invariant
   handshape); 1D depthwise-conv blocks with channel attention interleaved with transformer blocks; the
   dominant-hand flip is part of the exported graph.
4. **Export** (`scripts/export_v2.py`): ONNX, static INT8 (per-channel) on Conv/Gemm weights; the coordinate
   maths and attention products stay FP32 (quantizing them cost 6 points).

## Training recipe (what each step bought, measured on held-out data)

| Step | Effect |
|---|---|
| Conv+transformer hybrid vs depthwise TCN (same keypoints) | +3.7 top-1 (WLASL100, 3 seeds) |
| Holistic + palm-frame metric hand features | +8.5 |
| Dominant-hand canonicalisation | +2.0 (3/3 seeds) |
| Training on keypoints from degraded video (low light, tilt, shake, occluders, low res) | +1.0 clean, robustness drops cut by 3–6 points |
| Auxiliary phoneme heads from ASL-LEX 2.0 (training only) | +1.6 (5/5 paired seeds) |
| Warm start from the 2,000-word model + gentle fine-tune | +3.9 to +7.4 |
| Train+val final fit (2,000 words) | +4.3 |
| ASL Citizen clips (41 signers) | unseen-signer accuracy 76 → 93 (100 words), 49 → 81 (2,000 words); +3.7 top-1 at 2,000 |
| Core warm-started from the 2,000-word ASL Citizen model | +1.9 top-1 (89.5 → 91.5) |

Tried and rejected (kept in the record): mixup, cosine head, 3× larger model, distillation, test-time low-light
enhancement, phonological re-ranking, shoulder-relative flip, stronger occlusion augmentation.

Reproduce: `scripts/extract_v2.py` (keypoints) → `scripts/data_v2.py` (cache) → `scripts/extract_degraded.py`
(degraded copies, robustness sets) → `scripts/train_v2.py` → `scripts/export_v2.py` → `scripts/eval_v2.py`,
`scripts/eval_aslc.py`, `scripts/live_eval.py`, `scripts/results_csv.py`. `scripts/aslc_fetch.py` and
`scripts/extract_aslc.py` fetch and landmark ASL Citizen.

## Android app

`android/` — Kotlin, Jetpack Compose, Material 3. Tabs: Live (camera HUD: captions, top-3 confidence bars,
tracking state, timings), Clips (pick a video: same pipeline as `predict()`), Transcript, Settings (core/full
vocabulary, sensitivity, camera, GPU). `signcore` is plain Kotlin with a parity test against Python (normaliser
within 7e-6; INT8 top-1 agreement 40/40). See `android/README.md` to build.

## Data and licences

- WLASL (C-UDA, research use) and How2Sign (CC BY-NC 4.0) are the brief's datasets.
- ASL Citizen (Microsoft Research): non-commercial research licence, accepted by the team.
- ASL-LEX 2.0 phonology (CC BY 4.0) for the auxiliary heads.
- MediaPipe Holistic model: Apache-2.0.
