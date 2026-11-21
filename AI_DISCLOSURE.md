# AI disclosure

Team PerrydaPlatypus (Vellore Institute of Technology, Vellore): A Aswanth Raj and V. Nikhil.
Samsung XR Hackathon 2026, Track 2 (EchoGesture).

## Statement

We conceived EchoGesture, designed the system and its experiments, directed and checked every part of the work,
and are responsible for everything in this submission. AI coding assistants (Claude Code by Anthropic, and
Antigravity by Google) assisted under our direction with implementing code, running experiments and benchmarks,
and drafting and editing documents, the slide visuals and the video edit.

## What we did

- **The idea and its scope.** We set out to build two-way communication, not only recognition: sign to speech, and
  speech or text back to sign, entirely on the phone. We chose the features that make it useful in practice (live
  captions, replies in sign, a 300-sign dictionary, Learn, Teach-a-Sign) and the extension to Indian Sign Language.
- **The design.** We set the priorities (accuracy and robustness first, everything on the device, no cloud),
  decided to work from keypoints only so that no video is kept, chose the two-network ensemble, the same-sign rule
  and the INT8 export, and accepted or rejected each proposed design.
- **Data and evaluation.** We chose the datasets and checked their licences, kept the test split held out, and set
  the rule that a result had to be measured on held-out data before it was adopted.
- **Testing on the phone.** We tested the app on our own Galaxy S26 and did the physical parts of the measurements
  (for example, unplugging the phone and switching to airplane mode for the power test).
- **Review.** We reviewed what was produced and caught and corrected errors in drafts, including figures.
- **The presentation and the video.** We decided the story, what each slide says and shows, and the storyboard
  of the video; we chose the narrator's voice, reviewed every slide and cut, and published the video.

## How AI tools assisted

- **Code.** Implementing and refactoring code from our specifications: keypoint extraction, normalisation, the
  model and its training, the export and INT8 scripts, evaluation and measurement scripts, `predict()`, Docker and
  the Android app (Kotlin, Jetpack Compose). We reviewed and tested what it produced.
- **Experiments.** Running the training runs, exports and evaluations on our laptop, and the benchmarks on our
  phone over USB, with our permission, and recording the results.
- **Analysis.** Analysing results and proposing designs, for example finding the same-sign sets in the WLASL index
  and testing the answer rule on the validation split before the test split was scored.
- **Documents and media.** Drafting and editing the README files, the results table and the diagrams in `docs/`;
  rendering the slide visuals and assembling the deck; editing and rendering the demo video.

## AI-generated media

- **Two illustrations.** The pharmacy scenes on slides 2 and 3 are AI-generated images, labelled
  "AI-generated illustration" on the slides.
- **The narrator's voice.** The narrator in the demo video is AI text-to-speech (Qwen3-TTS, run locally). The
  voice that speaks captions in the video is the app's own: Android's on-device text-to-speech.

## What is not AI-generated

- **The model.** EchoGesture is our own network, trained from scratch by our training code on public sign language
  datasets (WLASL, ASL Citizen, ASL-LEX 2.0 phonology labels; INCLUDE only to measure Teach-a-Sign, never trained
  on). It contains no pretrained language or foundation model and no AI-generated training data.
- **The data and the footage.** Every training and test clip is a real signer from those datasets. The app footage
  in the video is the real app on our Galaxy S26, with held-out WLASL and Indian Sign Language test clips fed in as
  the camera. The stick figures in the app and on the slides are drawn from keypoints of real signers' clips. The
  background music was synthesized by our own code.
- **The numbers.** Every number in the README, `results/results.csv`, the slides and `docs/` comes from running the
  code on held-out data or on the phone, and is traced to the script and file that produced it.

## At run time

No AI service is called. `predict()` and the Android app run entirely on the device, and the app has no internet
permission. What runs is the EchoGesture model, Google's MediaPipe Holistic landmarker (Apache-2.0), and, in the
app, Android's own on-device speech recognition and text-to-speech.
