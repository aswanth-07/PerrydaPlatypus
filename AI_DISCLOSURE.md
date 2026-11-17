# AI disclosure

Team PerrydaPlatypus (Vellore Institute of Technology, Vellore): A Aswanth Raj and V. Nikhil.
Samsung XR Hackathon 2026, Track 2 (EchoGesture).

## The tool we used

We used **Claude**, Anthropic's AI model, through **Claude Code**, Anthropic's coding assistant, throughout the
project (November 2026). It ran on our own laptop and phone, with our permission for each kind of action.

## What it did

- **Code.** It wrote and edited most of the code under our direction: keypoint extraction, normalisation, the
  model and its training, export and INT8 quantization scripts, the evaluation and measurement scripts,
  `predict()`, the Docker setup, and the Android app (Kotlin, Jetpack Compose).
- **Experiments.** It ran the training runs, exports and evaluations on our laptop's GPU and CPU, and the
  on-device benchmarks on our phone over USB, and recorded the results.
- **Analysis.** It analysed the results (for example, finding the same-sign sets in the WLASL index and testing the
  answer rule on the validation split before the test split was scored), and proposed designs, which we accepted
  or rejected.
- **Documents.** It drafted the README files, the results table and the documents in `docs/`.

## What we did

- Chose the problem framing, the priorities (accuracy and robustness first, everything on the device, no cloud),
  the datasets and their licences, and which features the app should have.
- Directed the work, made the design decisions, set the rules a result had to meet before it was adopted, and
  reviewed what was produced.
- Tested the app on our own phone and did the physical parts of the measurements (for example, unplugging the phone
  and switching it to airplane mode for the power test).
- Made the presentation and the demo video.

We are responsible for everything in this submission.

## What is not AI-generated

- **The model.** The EchoGesture model is our own network, trained from scratch by our training code on public
  sign language datasets (WLASL, ASL Citizen, ASL-LEX 2.0 phonology labels; INCLUDE only to measure teach-a-sign,
  never trained on). It contains no pretrained language or foundation model, and no AI-generated training data.
- **The data.** Every training and test clip is a real signer from those datasets. The stick figures in the app's
  Learn mode are keypoints of real WLASL training clips, not generated images.
- **The numbers.** Every number in the README, `results/results.csv` and `docs/` comes from running the code on
  held-out data or on the phone; each is traced to the script and file that produced it. None was estimated or
  written from memory.

## At run time

No AI service is called. `predict()` and the Android app run entirely on the device; the app has no internet
permission. The components that run are the EchoGesture model, Google's MediaPipe Holistic landmarker
(Apache-2.0), and, in the app, Android's own on-device speech recognition and text-to-speech.
