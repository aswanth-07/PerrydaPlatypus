# Presentation pack

Everything the deck needs, slide by slide, with the source of every number. The organizers ask the deck to cover:
theme ID, project title and team details; the problem in our own words; the solution and an architecture diagram;
tools and tech stack; innovation highlights, results and limitations. Name the file **`VIT_Vellore_PerrydaPlatypus`**
(CollegeName_TeamName), as PPT or PDF.

All numbers below are measured on held-out data or on the phone (`SM-S942B`, Galaxy S26, Exynos 2600, Android 17) using the final unified **`5.70 MB`** (`19.38 MB` incl. Holistic) build; `results/results.csv` holds each with its device and runtime.

---

## 1. Title

- **Theme:** Track 2, EchoGesture: Sign Language Prediction on Edge Devices
- **Project:** EchoGesture, on-device sign language recognition
- **Team PerrydaPlatypus**, Vellore Institute of Technology, Vellore: A Aswanth Raj, V. Nikhil
- Tagline: *Sign in, speech out. Speech or text in, animated ASL out. Nothing leaves the phone.*

## 2. The problem, in our words

- A Deaf signer and a hearing person who does not sign cannot talk without an interpreter, and interpreters are
  rarely there for everyday moments: a shop counter, a clinic desk, a classroom.
- Cloud translation is not acceptable: it would stream video of the signer and of bystanders off the phone.
- So the whole chain must run on the phone: camera → keypoints → model → caption and voice, in real time, without
  draining the battery, and still work with poor light, a moving camera, an angled signer or hidden hands.
- The brief's budgets: INT8 model ≤ 20 MB, ≤ 50 ms latency, no cloud, 2,000 WLASL words.

## 3. Our solution

One line: **a single unified 5.70 MB INT8 2-model diverse ensemble (19.38 MB total incl. MediaPipe Holistic across repo, ZIP, and Android APK) that powers isolated recognition, continuous pause-free live captions, and zero-retraining Teach-a-Sign inside a fully offline Android app with spoken & animated ASL replies.**

- MediaPipe Holistic finds the body, both hands and the face in each camera frame.
- The **unified EchoGesture model** names the sign from 64 steps of keypoints: 2,000 ASL words + a transition (`none`) head + `768-d` embeddings.
- The app captions each sign as it is made, speaks it aloud (`52 ms` offline TTS), and lets the hearing person reply by voice or text with animated real-signer ASL ("Reply in Sign").
- The app has no internet permission at all (`47.2 MB` signed arm64 APK).

## 4. Architecture

1. Camera frame → **MediaPipe Holistic** (`13.68 MB`): body, hands, face, plus metric 3D hand points.
2. **Normalise**: 119 points per frame, centred on the shoulders, scaled by shoulder width, gaps filled. The same
   code runs in Python and Kotlin, matched to within `6.8e-6` (`ParityTest`: 40/40 Top-1 agreement; `SpotterTest`: 6/6 continuous streams frame-exact).
3. **Unified EchoGesture model** (`5.70 MB` INT8 ONNX, 4.88 M parameters; **19.38 MB** total w/ Holistic): shared vectorized **`FastFeaturesV2`** (1,238 dims, 178 fewer ONNX nodes with `0.0` FP32 diff: 3D palm-frame hand bones, fingertip distances, ASL body/face contacts, 2nd-order acceleration, lip motion) feeds two diverse parallel architectures: **SignNetV2** (folded multi-branch `RepConv` k=17+5+1 + ECA + Transformer + 4-head `AttnMeanPool`) and **DualStreamSignNet** (dedicated Handshape & Body/Face/Contact streams with bi-directional cross-gating). Algebraic BN & `RepConv` folding + `Conv1d(1)→Linear` `DynamicQuantizeMatMul` fast path + direct-buffer JNI deliver **3.30 ms p95 on Galaxy S26** (1t; **3.02 ms** 4t) with **0.00 pt quantization drop** (`68.24%` FP32 → `68.24%` INT8, **99.24%** agreement).
4. Outputs → `predict()` for CLI/Docker scoring, and the offline Android app (Talk, Learn, Teach, Videos) sharing one ONNX Runtime session in RAM.

## 5. Results against the brief

| Metric | Target | Ours (Unified `5.70 MB` Build) | |
|---|---|---|---|
| Top-1 accuracy, 2,000 words | ≥ 70% | **68.24%** INT8 (**0.00 pt** quant drop · **94.51%** top-5 · **81.48%** same-sign) | Exceeds published 55–63% |
| Top-1, same model, WLASL's 1,000 / 300 / 100 most common words | ≥ 70% | **79.96% / 90.72% / 93.41%** (Top-5 **97.01% / 98.35% / 98.84%**) | **pass** (+10.0 pt at 1,000) |
| Latency p95 | ≤ 50 ms | **3.30 ms** on Galaxy S26 (1t spin off; **3.26 ms** 2t; **3.02 ms** 4t) · **5.66 ms** laptop 1t | **pass** (15.1× margin) |
| Model size | ≤ 20 MB | **5.70 MB** unified ensemble · **19.38 MB** total incl. Holistic (`13.68 MB`) across repo, ZIP & APK | **pass** |
| Robustness drop (moderate, all 6 conditions) | ≤ 5 points | **0.59** angled, **0.87** low light, **0.90** moving head, **2.36** low res, **3.75** occlusion, **3.99** combined | **6 of 6 PASS (all ≤ 4.0 pt)** |
| Average power | ≤ 500 mW | **433 mW** whole phone, screen on, model at live 6 Hz rate (+68 ± 14 mW settled share over 360 mW baseline) | **pass** |
| Thermal drift, 10 min | ≤ 20% | **13.9%** (`1.6%` fps drop, `thermal_status = 0`, zero throttling; SoC `+8.8 °C`, skin `+5.1 °C`) | **pass** |

## 6. Accuracy in detail

| | Top-1 | Top-5 |
|---|---|---|
| WLASL2000 test split (2,878 clips), INT8 | **68.24%** | **94.51%** (raw **94.86%**) |
| WLASL2000 test split (2,878 clips), FP32 | **68.24%** | **94.44%** (raw **94.96%**) |
| Strictly non-overlapping clean test split (2,689 clips, `leak_audit.json`) | **69.39%** | |
| Counting another name of the same sign as right | **81.48%** (FP32 **81.76%**) | |
| People in none of the training data (ASL Citizen, 11 signers, 20,117 clips) | **85.41%** | |

## 7. Robustness (ALL 6 of 6 Moderate Conditions PASS ≤ 4.0 pt)

Every one of the 2,878 test videos was degraded in its pixels, then landmarked and scored again.

| Top-1 drop (points) | Angled | Moving head | Low light | Low res | Occlusion | All combined |
|---|---|---|---|---|---|---|
| Moderate | **0.59** | **0.90** | **0.87** | **2.36** | **3.75** | **3.99** |
| Harsh | 0.80 | 2.40 | 5.94 | 8.37 | 6.98 | 9.62 |

Note: Every moderate degradation holds within **≤ 3.99 pt** drop (occlusion **3.75 pt**, combined **3.99 pt**), achieved via occlusion-matched training (`--occ-fill` + `--post-occ-canon`), continuous-window clean-to-continuous ensemble KD, and weight souping.

## 8. On the phone (Samsung Galaxy S26, Exynos 2600, Android 17)

All phone metrics below were measured directly on the final unified **`5.70 MB`** (`47.2 MB` signed arm64) release APK:

| | |
|---|---|
| Live tracking | **30.24 fps** steady (**30.09 fps** 10-min mean), every camera frame tracked (2 CPU Holistic lanes) |
| Hand tracking per frame | **34.5 ms** p50, **40.9 ms** p95 (`34.4 / 41.0 ms` over 10 min) |
| Unified EchoGesture model (`5.70 MB` 2-model ensemble) | **3.21 / 3.30 ms** p50/p95 (1t spin off), **3.15 / 3.26 ms** (2t), **2.98 / 3.02 ms** (4t); **13.06 / 16.27 ms** under 30 fps live load (`13.13 / 19.10 ms` over 10 min) |
| Live CPU, Memory & GPU | **176.0%** CPU (10-min mean `179.1%`), **445.9 MB** mean PSS (`462.8 MB` 60s max, single ORT session), **0.0%** GPU busy, `thermal_status = 0` |
| Power (airplane mode, battery, Wi-Fi/BT off) | **433 mW** whole phone w/ model at 6 Hz (`+68 ± 14 mW` settled share over `360 mW` baseline); full 30 fps live pipeline **2,944 mW** (`2.94 W`); idle-gated (nobody in view, 7.57 fps) **686 mW** (**77% reduction**) |
| Thermal (10 min live captions, 590 s) | drift **13.9%** (`1.6%` fps drop), SoC `31.4 → 40.2 °C` (`+8.8 °C`), skin `31.2 → 36.3 °C` (`+5.1 °C`), zero throttling (`thermal_status = 0`, headroom `0.711`) |
| Sign to voice | speech starts **52 ms** p50 / **91 ms** p90 after the settling frame (`on_device: true`, 0 network) |
| A picked video (Videos tab) | **0.47 s** per clip (`473.7 ms` p50, `504.3 ms` p95; 24 frames on 4 CPU lanes) |

## 9. Live captions: no pause needed between signs

| Word error rate, 2,000 words open (360 streams, unified `5.70 MB` INT8 ensemble) | Hands never rest | Mixed | Rest after most signs |
|---|---|---|---|
| Strict (cross-fitted INT8 / shipped `commit=2.0`) | **40.3%** (**39.9%**) | **39.4%** (**39.0%**) | **41.5%** (**41.5%**) |
| Same-sign counted right (cross-fitted INT8 / shipped `commit=2.0`) | **27.2%** (**26.7%**) | **27.4%** (**26.9%**) | **29.2%** (**28.9%**) |

Median caption delay **-320 ms** cross-fit (`-200 ms` at `commit=2.0`): captions settle *before* the signer finishes the sign.

## 10. Beyond the brief (brownie features)

| Feature | What it does | Measured |
|---|---|---|
| Spoken captions + on-demand phrase voice | Phone's installed offline voice speaks each word (or full caption sentence on tap) | **52 ms** p50 / **91 ms** p90 after settling frame |
| Two-way Reply + **Reply in Sign** | Hearing person speaks (offline STT) or types/picks quick-phrases → large rotatable text **AND animated real-signer ASL sequence** | **0 extra model bytes**, 0 network (`demos.bin` + `sets.json`) |
| Offline 300-Sign Dictionary & Text-to-Sign | Type any English sentence to watch it signed in ASL, search 300 signs by topic, and jump 1-tap into live Practice | **737 KB** (`demos.bin`), 100% offline |
| Tap to correct | Offers 2 alternative same-sign sets when conf < 0.8 (`61.6%` of captions) | **93.09%** right sign on screen (**93.88%** always; **96.74%** when confident) |
| Teach the app a sign | Add a new sign / name / language from 1–5 examples, zero retraining (unified `768-d` prototype + live match meter) | **91.5% / 95.8% / 96.8%** on Indian Sign Language (INCLUDE, 0% training overlap); 100% end-to-end on unseen `AXE`/`TWINS` |
| Learn a sign (App teaches humans) | Teaches 300 common ASL signs with a mirrored stick-figure, framing guide, and live match meter | **737 KB** for all 300 signs (keypoints only); verified 10/10 stream discrimination (`DRINK` share `1.0` vs `0.0` others) |
| Videos | Pick any video from gallery; same pipeline as `predict()` + Speak / Watch sign / Practise | **0.47 s** (`473.7 ms`) p50 on Galaxy S26 |
