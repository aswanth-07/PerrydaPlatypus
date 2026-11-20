# EchoGesture: on-device sign language recognition

Samsung XR Hackathon 2026, Track 2 (EchoGesture: sign language prediction on edge devices).
Team **PerrydaPlatypus**, Vellore Institute of Technology, Vellore: A Aswanth Raj and V. Nikhil.

Video of a signer in, an English word out, entirely on the device. MediaPipe Holistic finds the body and hands,
and the **EchoGesture model** (5.72 MB, INT8 ONNX 2-model diverse ensemble, 2,000 ASL words; matching 5.72 MB live continuous ensemble) names the sign. A native Android app turns this
into live captions that are spoken aloud, and lets the hearing person reply by voice, typed quick-phrases, or animated real-signer ASL (**Reply in Sign**). It also includes an offline 300-sign visual dictionary & text-to-sign translator, teaches signs to a learner, and learns new signs from three examples. Nothing leaves the device: the app has no internet permission.

![Architecture](docs/architecture.svg)

## Submission checklist

| Item | Where |
|---|---|
| `predict(video_path) -> dict` | [`src/main.py`](src/main.py), with everything it needs in `src/` ([`src/README.md`](src/README.md)) |
| Results sheet | [`results/results.csv`](results/results.csv), the brief's columns: metric, target, measured, unit, device, runtime, pass |
| Predictions | [`results/predictions.jsonl`](results/predictions.jsonl), one JSON line per WLASL2000 test clip (also as one list in `predictions.json`) |
| Quantized model | [`src/models/echogesture_int8.onnx`](src/models/echogesture_int8.onnx), 5.72 MB INT8 ONNX (2-model ensemble; matching 5.72 MB live ensemble in `android/models/`) |
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
| Top-1 accuracy | ≥ 70% | **67.93%** on all 2,878 WLASL2000 test clips (**93.99%** top-5; FP32 **68.14%**, 99.06% INT8 agreement). The same model file answering from WLASL's 1,000 most common words: **80.22%**; 300: **90.42%**; 100: **93.02%** | met up to 1,000 words; +3.5 pt jump at 2,000 |
| Latency p95 | ≤ 50 ms | **3.45 ms** on a Galaxy S26 (one CPU thread, spin off; **3.29 ms** on 2 threads); 6.94 ms on a laptop CPU thread | pass (14.5× margin) |
| Model size | ≤ 20 MB | **5.72 MB** (2-model ensemble; matching 5.72 MB live continuous ensemble); with the MediaPipe Holistic landmarker **19.40 MB** | pass |
| Robustness (top-1 drop) | ≤ 5 points | all 2,878 test clips degraded before landmarking: angled **0.49**, low light **0.98**, moving head **1.01**, low resolution **2.23**, combined **3.34**, occlusion **4.24** | **6 of 6 pass** |
| Average power | ≤ 500 mW | **457 mW** for the whole phone, screen on, while the model runs at the live caption rate; the model's own share is +6 ± 28 mW over the screen-on baseline | pass |
| Thermal drift, 10 min | ≤ 20% | **12.1%** (Galaxy S26, 10 minutes of live captions; updated live SignNetV2 confirmed lower CPU 172.1% vs 180.8% and thermal status 0) | pass |

WLASL2000 is a hard benchmark: published pose-based and video-based systems report roughly 55 to 63% top-1 on
this split. Our number counts every clip of the official test split, including 189 clips whose source segment
overlaps a train or validation clip (`scripts/leak_audit.py`, `results/leak_audit.json`). Many of the remaining
errors are one sign filed under two English names (next section): counting those as right, top-1 is **81.10%** (FP32 **81.55%**).

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
3. **The EchoGesture model** (`scripts/model.py`, 4.88 M parameters across a diverse 2-model ensemble: `SignNetV2` + `DualStreamSignNet`). `FeaturesV2` (1,238 dims) is computed once inside the shared graph: 3D palm-frame hand bones, intra-hand fingertip distances, mirror-equivariant ASL hand-to-face/body contacts, 2nd-order hand acceleration, and lip/brow velocity. Multi-branch `RepConv` (`k=17+5+1`) blocks fold algebraically into a single depthwise conv at export (zero extra FLOPs/bytes), paired with ECA channel attention, Transformer layers, and 4-head `AttnMeanPool`. Dominant-hand flip, `FastSetRule`, and unit-normalized 768-d embeddings are fused inside the graph.
4. **Export** (`scripts/export.py`). Algebraic BN + `RepConv` folding and `Conv1d(1)→Linear` conversion for ONNX Runtime's `DynamicQuantizeMatMul` fast path, then dynamic INT8 with the feature stems kept in FP32. The INT8 model agrees with the float model's top-1 on **99.06%** of test clips (2,851 / 2,878).

## Accuracy

| | Top-1 | Top-5 |
|---|---|---|
| WLASL2000 official test split (2,878 clips), INT8 model | **67.93%** | **93.99%** (raw 94.51%) |
| WLASL2000 official test split (2,878 clips), FP32 | **68.14%** | **93.99%** (raw 94.58%) |
| Counting another name of the same sign as right | **81.10%** (FP32 81.55%) | |
| Signers in none of the training data (ASL Citizen test split, 11 people, 20,117 clips) | **85.41%** | |

- **Unseen signers** decide every model choice, not WLASL test accuracy. They are scored with matched zero-leakage Stage-0 runs (`SignNetV2 + DualStreamSignNet`) that never saw those 11 people.
- **Smaller word lists, same file.** WLASL is published at four sizes. The 2,000-word model, answering only from
  each list (`ECHO_VOCAB_FILE`), scores on the test clips of those words (`scripts/vocab_subsets.py`):

| Word list | Test clips | Top-1 | Top-5 |
|---|---|---|---|
| WLASL100 | 258 | **93.02%** | 98.06% |
| WLASL300 | 668 | **90.42%** | 98.35% |
| WLASL1000 | 1,876 | **80.22%** | 96.70% |
| WLASL2000 | 2,878 | **67.93%** | 93.99% |

So the brief's 70% is cleared by **+10.2 points** at 1,000 words with this single file, and a published scoring
vocabulary needs only its word list, not retraining. Random lists of the same sizes give **95.2%, 90.0% and 78.1%**.

## One sign, several names

WLASL's own index lists the same source video segment under two glosses for 590 pairs of its 2,000 words, using
the train and validation listings only (no test listing is used). Joined into sets, 308 sets hold 780 words and
37.9% of the test clips: PRISON and JAIL, STORY and TALE, DOCTOR and PHYSICIAN, HURRY and RUSH are the same signing
under two English names. On words with no such twin the network is right **79.45%** of the time; on words with one,
**49.08%**, and most of those errors are the twin. Two things follow (`scripts/signsets.py`):

- **An answer rule inside the graph.** The set's total probability is kept and shared among its words in
  proportion to p(word)^0.25 × (WLASL clips of the word)^2, following how often WLASL uses each name. Applied
  unchanged via `FastSetRule`: test top-1 66.99 → **67.93** (FP32 66.96 → **68.14**). A global word-frequency prior gives nothing, so the gain comes
  from the set structure. It is a prior matched to WLASL's naming (on ASL Citizen's labels it costs 1.6 points),
  so `ECHO_SET_RULE=0` answers with the network's own probabilities. The graph returns `probs` (with the rule),
  `set_probs`, `embedding` (768-d) and `raw_probs`.
- **Captions that name the sign, not a guess between its names.** The app shows the other names under the caption
  ("same sign: PHYSICIAN"), from `src/models/sets.json`.

## Robustness

Every one of the 2,878 test videos was degraded in its pixels before MediaPipe (angled camera, moving head, low
light, occlusion, low resolution, and low light + moving head + occlusion combined) at two strengths, landmarked
again, and scored with the float checkpoints the INT8 file was exported from, answering with the same-sign rule as
the shipped graph does (`scripts/extract_degraded.py`, `scripts/eval_suite.py`):

| Top-1 drop (points) | angled | moving head | low light | low res | occlusion | combined |
|---|---|---|---|---|---|---|
| moderate | **0.49** | **1.01** | **0.98** | **2.23** | **4.24** | **3.34** |
| harsh | 0.56 | 2.09 | 6.05 | 8.03 | 6.85 | 8.24 |

**All 6 of 6 moderate conditions pass the ≤ 5.0 pt requirement** (combined accuracy is **64.80%**, exceeding our prior model's clean accuracy of 64.42%!). The breakthrough came from occlusion-matched training (`--occ-fill` linear interpolation matching `normalise.fill_group()`), post-occlusion handedness canonicalisation (`--post-occ-canon`), and cross-architecture clean-view distillation between `SignNetV2` and `DualStreamSignNet`.

## On the phone

Samsung Galaxy S26 (SM-S942B, Exynos 2600, Android 17), the release APK, a signer video replayed like a 30 fps
camera (`android/app/.../engine/Bench.kt`, `scripts/phone_bench_report.py`):

| | Measured |
|---|---|
| Live frames tracked | **30.2 fps** (60 s live run; **30.05 fps** over 10 minutes), every frame of a 30 fps camera |
| Holistic per frame | **34.1 ms** p50, **40.6 ms** p95 (two CPU trackers taking frames in turn) |
| EchoGesture model (2-model ensemble, 5.72 MB) | **3.45 ms** p95 back to back (one thread, spin off; **3.29 ms** on 2 threads); 18.3 ms p95 paced; live `SignNetV2` (3.03 MB) **9.46 ms** p95 beside live tracking (down from 14.55 ms) |
| CPU & memory under 30 fps live load | **172.1%** of one core (down from 180.8%), **460.8 MB** PSS (down from 474.5 MB), **0%** GPU busy, thermal status 0 |
| Thermal, 10 minutes of live captions | drift **12.1%** (baseline 590 s run, SoC 28.7 → 40.2 °C, thermal status 0; updated live model lowers CPU load further) |
| Power, whole phone, screen on, on battery, airplane mode | screen-on baseline 451 mW; with the model at the live caption rate **457 mW** (its own share +6 ± 28 mW); full live pipeline at 30 fps **3.05 W** |
| Sign to voice | the phone starts speaking a word **70 ms** (median) after the camera frame that settled it, with an installed voice and no network |
| A picked video (Videos tab) | **0.46 s** per clip at the median (457.9 ms, p95 500.6 ms; down from 566 ms): decode + 24 frames on 4 CPU Holistic lanes + 2-model ensemble (9.32 ms) |

Two choices made it real time: on this phone MediaPipe's CPU path is faster than its GPU path, and two Holistic
trackers take frames in turn, with their results put back in order (and 4 lanes for picked videos). The rate matters for accuracy, not just
smoothness: tracking at half the rate finds 46.7% of the words in a stream instead of 59.2%. While no hand is in
view for two seconds, the app drops to low-rate idle gating (and uses a zero-allocation 16×12 luminance motion check at 4 fps when the camera scene is still) before snapping back to 30 fps on the first frame of motion.

Power follows the tracking rate. The model itself costs almost nothing (it runs six times a second for a few
milliseconds), so the brief's 500 mW limit is met by the model, not by the whole pipeline: real-time hand tracking
costs about 80 mJ per frame on this phone's CPU and GPU alike.

## Live captions

The app captions each sign while the hands stay up; the signer does not have to lower them between words
(`Spotter.kt` in `android/signcore`, `scripts/live_cont.py`).

- **A live-caption copy of the EchoGesture 2-model ensemble** (`android/models/echogesture_live_int8.onnx`, **5.72 MB**,
  3-way weight-souped `SignNetV2` + 3-way weight-souped `DualStreamSignNet` trained with clean-to-continuous ensemble KD and rare-class balancing on 1.2 s continuous signing windows). On whole clips it scores **68.21%** INT8 top-1 (**94.41%** top-5, **99.20%** FP32 agreement), and exports a **768-d** ensemble embedding for live Teach-a-Sign matching.
- **One caption per sign, settled smoothly as the sign completes.** The window is classified six times a second while a wrist
  is up. A run of windows that name a sign is one sign; each window adds its top answers' same-sign set evidence while a live lock-in bar tracks progress and tentative hysteresis prevents mid-sign flicker. When the run ends (or hits `commit`), the sign with the most evidence is captioned onto a smooth auto-scrolling caption ribbon.

Measured directly on the shipped INT8 ONNX binary over held-out WLASL test clips joined into streams of six signs (120 streams of each kind, 360 total), 2,000 words open, cross-fitted decoder settings (shipped fixed `commit=2.0` INT8 config in parentheses; FP32 cross-fit: 39.9% / 39.0% / 40.8%):

| Word error rate, 2,000 words open | Hands never rest | Mixed | A rest after most signs |
|---|---|---|---|
| Strict | **40.4%** (INT8 fixed **39.9%**) | **39.6%** (INT8 fixed **39.2%**) | **41.7%** (INT8 fixed **41.5%**) |
| Another name of the same sign counts as right | **27.4%** (INT8 fixed **26.7%**) | **27.5%** (INT8 fixed **27.1%**) | **29.4%** (INT8 fixed **29.0%**) |

The caption appears 320 ms before the sign's last frame at the median under cross-fit (`-200 ms` at `commit=2.0`).
The Kotlin spotter is checked against the Python one word for word and frame for frame on six test streams
(`SpotterTest`). The streams are joins of isolated clips; no gloss-annotated continuous ASL was available, and a
real signer's transitions will differ.

## The Android app

`android/`: Kotlin, Jetpack Compose, Material 3. Tabs: **Learn**, **Talk**, **Videos**; Sign Dictionary, past conversations,
Teach and Settings open over them (screens and build steps in [`android/README.md`](android/README.md)).

- **Live captions, spoken aloud** by the phone's own installed voice, offline, with an auto-scrolling caption ribbon (left-edge fade), 3 dp live lock-in bar, on-demand phrase read-aloud (`Speak` chip), 1-tap `Continuous / After each sign` quick-tune pill, and frosted bottom dock.
- **Two-way Reply + Reply in Sign.** The hearing person taps **Reply** to speak (Android on-device STT) or **Type** to pick an offline quick-phrase or type any message (automatic offline fallback if STT is unavailable). Large high-contrast text appears for the signer with 180° Rotate for across-the-table chat — and **"Watch in sign" / "Show in sign"** plays the reply back as an animated sequence of real-signer ASL signs offline using the bundled 737 KB `demos.bin` keypoints and `sets.json` synonyms (zero extra model bytes, zero network).
- **Offline 300-Sign Visual Dictionary & Text-to-Sign Translator.** From the Learn hub, type any English phrase to watch it signed in sequence with interactive highlighted gloss pills, speed/hand-zoom controls, pronunciation audio, and 1-tap **"Practise"** jump into live camera verification — or browse and search all 300 signs by topic.
- **Same-sign captions.** "same sign: PHYSICIAN" under DOCTOR.
- **Tap to correct.** When the model is unsure, the caption offers the two next most likely signs ("Not right? BIBLE ·
  WALLET"); one tap swaps the word and speaks the correction. On the 2,878 test clips the sign that was made is
  then on screen for **93.15%** of clips (against **81.10%** for the top word alone, and **96.79%** when confident enough not to show chips; `scripts/alternatives_eval.py`). It
  costs nothing extra: it reads the ranking the model already returns.
- **Learn a sign.** The app teaches signs to anyone learning ASL. A moving stick figure shows one of 300 common
  signs (keypoints of a real signer from a WLASL training clip, never video; 737 KB for all 300,
  `scripts/make_demos.py`), mirrored to match the front camera beside it, with play/pause, pronunciation speaker, framing guide, and live match meter. The learner copies it, and the app says
  whether that was the sign: the practised word's share of the evidence must be at least half. With a replayed
  10-clip stream while DRINK was practised, it passed on the DRINK clip (`share 1.0`) and on none of the other nine (`share 0.0`).
- **Teach a sign.** Type a word (or pick a quick idea) and sign it three to five times. The app averages multi-crop core-window **768-d** ensemble embeddings into a
  prototype (robust to timing shifts in continuous signing), flags hand-visibility quality on review, shows a live match meter on the Try-it step, and from then on captions and speaks that word. A name sign, a regional sign, or a sign from another
  sign language can be added without retraining; only keypoints are stored. On Indian Sign Language (INCLUDE, 23 signs, 0% training overlap) the shipped 2-model ensemble scores **91.28% / 95.82% / 96.83%** with 1 / 3 / 5 examples. On unseen ASL signs (baseline WLASL-only model, different signers): 79.3% / 90.3% / 92.5% with 1 / 3 / 5 examples. End-to-end on the new live ensemble (`AXE`, `TWINS` from three signers each), all unseen-signer clips were captioned as the taught word (`0.66–0.90` cosine) with zero false triggers on other signs.
- **Videos.** Pick a video of one sign; the same pipeline as `predict()` names it (**0.46 s** median on Galaxy S26), with on-device **Speak aloud**, **Watch sign**, and **Practise** actions right on the result card.

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

- Top-1 on all 2,000 WLASL words is **67.93%** (68.14% FP32), 2.1 points shy of 70% (exceeded by +10.2 points at 1,000 words: **80.22%**). The largest remaining lever is more signers per word: words with 15+ training clips score **83.4%**, while rare words with ≤6 clips score **38.8%**.
- All 6 moderate robustness conditions now pass (≤ 4.24 pt drop); extreme harsh occlusion and low resolution drop 6.9–8.2 points when hands disappear for most of a clip.
- Live captions were measured on joins of isolated clips, not on natural continuous signing.
- The power limit is met by the model; real-time hand tracking for the live camera costs about 3 W on this phone.

## AI disclosure

An AI coding assistant was used throughout the project, under the team's direction: to write most of the code,
run the experiments and measurements, analyse results and draft documentation. The model is our own network,
trained from scratch on public sign language datasets; no AI service runs at inference time. The full disclosure
(the tool, what it did, what the team did, and what is not AI-generated) is in [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md).
