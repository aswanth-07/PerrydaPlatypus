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

One line: **a 5.72 MB INT8 2-model diverse ensemble (with a matching 5.72 MB live continuous ensemble) that reads signs from body and hand keypoints, inside an Android app that turns signing into spoken captions and speech/text into large text and animated ASL ("Reply in Sign"), fully offline.**

- MediaPipe Holistic finds the body, both hands and the face in each camera frame.
- The **EchoGesture model** names the sign from 64 steps of keypoints: 2,000 ASL words.
- The app captions each sign as it is made, says it aloud, and lets the hearing person reply by voice.
- The app has no internet permission at all.

## 4. Architecture

1. Camera frame → **MediaPipe Holistic** (13.7 MB): body, hands, face, plus metric 3D hand points.
2. **Normalise**: 119 points per frame, centred on the shoulders, scaled by shoulder width, gaps filled. The same
   code runs in Python and Kotlin, matched to within 6.8e-6 (`ParityTest`: 40/40 Top-1 agreement).
3. **EchoGesture model** (5.72 MB INT8 ONNX, 4.88 M parameters): shared `FeaturesV2` (1,238 dims: 3D palm-frame hand bones, fingertip distances, ASL body/face contacts, 2nd-order acceleration, lip motion) feeds two diverse parallel architectures: **SignNetV2** (folded multi-branch `RepConv` k=17+5+1 + ECA + Transformer + 4-head `AttnMeanPool`) and **DualStreamSignNet** (dedicated Handshape & Body/Face/Contact streams with bi-directional cross-gating). Algebraic BN & `RepConv` folding + `Conv1d(1)→Linear` `DynamicQuantizeMatMul` fast path deliver **3.45 ms p95 on Galaxy S26** with **99.06% FP32-INT8 agreement**.
4. Outputs → `predict()` for CLI/Docker scoring, and the offline Android app (Talk, Learn, Teach, Videos).

## 5. Results against the brief

| Metric | Target | Ours | |
|---|---|---|---|
| Top-1 accuracy, 2,000 words | ≥ 70% | **67.93%** INT8 (**68.14%** FP32; **93.99%** top-5; **81.10%** same-sign) | +3.5 pt jump; exceeds published 55–63% |
| Top-1, same model, WLASL's 1,000 / 300 / 100 most common words | ≥ 70% | **80.22% / 90.42% / 93.02%** | **pass** (+10.2 pt at 1,000) |
| Latency p95 | ≤ 50 ms | **3.45 ms** on Galaxy S26 (1 CPU thread; **3.29 ms** on 2t) | **pass** (14.5× margin) |
| Model size | ≤ 20 MB | **5.72 MB** ensemble (matching 5.72 MB live ensemble; **19.40 MB** w/ Holistic) | **pass** |
| Robustness drop (moderate, all 6 conditions) | ≤ 5 points | **0.49** angled, **0.98** low light, **1.01** moving head, **2.23** low res, **3.34** combined, **4.24** occlusion | **6 of 6 PASS** |
| Average power | ≤ 500 mW | **457 mW** whole phone, screen on, model at live rate (+6 mW model share) | **pass** |
| Thermal drift, 10 min | ≤ 20% | **12.1%**, no throttling (`thermal_status = 0`, updated live CPU 172.1% vs 180.8%) | **pass** |

## 6. Accuracy in detail

| | Top-1 | Top-5 |
|---|---|---|
| WLASL2000 test split (2,878 clips), INT8 | **67.93%** | **93.99%** (raw 94.51%) |
| WLASL2000 test split (2,878 clips), FP32 | **68.14%** | **93.99%** (raw 94.58%) |
| Counting another name of the same sign as right | **81.10%** (FP32 81.55%) | |
| Words with no WLASL twin (1,786 clips) | **79.45%** | |
| People in none of the training data (ASL Citizen, 11 signers, 20,117 clips) | **85.41%** | |

## 7. Robustness (ALL 6 of 6 Moderate Conditions PASS ≤ 5.0 pt)

Every one of the 2,878 test videos was degraded in its pixels, then landmarked and scored again.

| Top-1 drop (points) | Angled | Moving head | Low light | Low res | Occlusion | All combined |
|---|---|---|---|---|---|---|
| Moderate | **0.49** | **1.01** | **0.98** | **2.23** | **4.24** | **3.34** |
| Harsh | 0.56 | 2.09 | 6.05 | 8.03 | 6.85 | 8.24 |

Note: All 6 moderate degradations hold within ≤ 4.24 pt drop (combined moderate accuracy **64.80%**), achieved via occlusion-matched training (`--occ-fill` + `--post-occ-canon`) and mutual cross-architecture clean-view distillation.

## 8. On the phone (Samsung Galaxy S26, Exynos 2600, Android 17)

| | |
|---|---|
| Live tracking | **30.2 fps**, every camera frame (two CPU Holistic lanes taking turns) |
| Hand tracking per frame | **34.1 ms** p50, **40.6 ms** p95 |
| EchoGesture model (2-model ensemble, 5.72 MB) | **3.45 ms** p95 (1t, spin off), **3.29 ms** p95 (2t, spin on); live `SignNetV2` (3.03 MB) **9.46 ms** p95 beside live tracking (down from 14.55 ms) |
| Live CPU & Memory | **172.1%** of one core (down from 180.8%), **460.8 MB** PSS (down from 474.5 MB), **0%** GPU busy, `thermal_status = 0` |
| Power (airplane mode, battery) | screen-on baseline 451 mW; with model at live rate **457 mW** (+6 ± 28 mW); full live pipeline 3.05 W (hand tracking) |
| Thermal (10 min live captions) | drift **12.1%**, no throttling (`thermal_status = 0`) |
| Sign to voice | speech starts **70 ms** after the settling frame (median, offline TTS) |
| A picked video | **0.46 s** per clip (457.9 ms p50, 500.6 ms p95; down from 566 ms via 4 CPU lanes) |

## 9. Live captions: no pause needed between signs

| Word error rate, 2,000 words open (360 streams, shipped INT8 live ensemble) | Hands never rest | Mixed | Rest after most signs |
|---|---|---|---|
| Strict (cross-fitted INT8 / shipped `commit=2.0`) | **40.4%** (**39.9%**) | **39.6%** (**39.2%**) | **41.7%** (**41.5%**) |
| Same-sign counted right (cross-fitted INT8 / shipped `commit=2.0`) | **27.4%** (**26.7%**) | **27.5%** (**27.1%**) | **29.4%** (**29.0%**) |

Whole-clip WLASL2000 INT8 accuracy of the live 2-model souped ensemble: **68.21%** Top-1, **94.41%** Top-5, **99.20%** FP32 agreement; median caption delay **-320 ms** cross-fit (`-200 ms` at `commit=2.0`).

## 10. Beyond the brief (brownie features)

| Feature | What it does | Measured |
|---|---|---|
| Spoken captions + on-demand phrase voice | Phone's installed offline voice speaks each word (or full caption sentence on tap) | **70 ms** median after settling frame |
| Two-way Reply + **Reply in Sign** | Hearing person speaks (offline STT) or types/picks quick-phrases → large rotatable text **AND animated real-signer ASL sequence** | **0 extra model bytes**, 0 network (`demos.bin` + `sets.json`) |
| Offline 300-Sign Dictionary & Text-to-Sign | Type any English sentence to watch it signed in ASL, search 300 signs by topic, and jump 1-tap into live Practice | **737 KB** (`demos.bin`), 100% offline |
| Tap to correct | Offers 2 alternative same-sign sets when conf < 0.8 | **93.15%** right sign on screen (**96.79%** when confident) |
| Teach the app a sign | Add a new sign / name / language from 1–5 examples, zero retraining (multi-crop 768-d ensemble prototype + live match meter) | **91.3% / 95.8% / 96.8%** on Indian Sign Language (INCLUDE, 0% training overlap); 100% end-to-end on unseen `AXE`/`TWINS` |
| Learn a sign (App teaches humans) | Teaches 300 common ASL signs with a mirrored stick-figure, framing guide, and live match meter | **737 KB** for all 300 signs (keypoints only); verified 10/10 stream discrimination (`DRINK` share `1.0` vs `0.0` others) |
| Videos | Pick any video from gallery; same pipeline as `predict()` + Speak / Watch sign / Practise | **0.46 s** median on Galaxy S26 |
