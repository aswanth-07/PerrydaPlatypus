# EchoGesture privacy policy

Last updated: 18 November 2026

EchoGesture captions American Sign Language on your phone, speaks the captions aloud, and shows spoken replies
as text. It is built so that nothing you sign, say or teach it leaves your phone.

## The short version

- EchoGesture has no internet permission. Android does not let it open a network connection, so it cannot send
  anything anywhere.
- There are no accounts, no ads, no analytics and no crash reporting.
- Video and audio are processed in memory and never saved.
- What the app keeps (conversations, signs you teach it, your learning progress and settings) stays in its
  private storage on your phone. It is excluded from Android backups and device transfers, and it is deleted
  when you uninstall the app.

## What the app uses, and why

**Camera.** Frames from the camera are passed to hand and body tracking (MediaPipe) and a sign recogniser
(ONNX Runtime), both running on the phone. Each frame is discarded once it is processed. No photo or video is
saved.

**Microphone.** Only when you tap Reply. Your speech is turned into text by Android's on-device speech
recogniser (Android 12 and later), so the audio stays on the phone. The audio is not saved; the recognised text
is added to the conversation.

**Text-to-speech.** When "Speak signed words aloud" is on, caption words are passed to the phone's
text-to-speech engine. EchoGesture chooses a voice installed on the phone. If no installed English voice is
found, the phone's text-to-speech app may use a voice of its own that needs a connection; EchoGesture says so in
Settings so you can install an offline voice.

**A video you pick.** "Identify a sign in a video" reads the video you choose with Android's photo picker,
processes it in memory and keeps only the result while the app is open. The video is not copied or saved.

## What the app keeps on your phone

| What | Where | How long |
| --- | --- | --- |
| Conversations (recognised words and spoken replies, with times) | App-private file | 1 day, 30 days (default) or until you delete them: Settings > Keep conversations. Delete one in History or all in Settings. |
| Signs you teach it | App-private folder, as hand, body and face keypoints (numbers), never video | Until you delete the sign in Learn > Teach the app a sign |
| Learning progress (signs learned, days practised) | App-private settings | Until you reset it in Settings > Reset learning progress |
| Settings | App-private settings | Until you uninstall the app |

Other apps cannot read this storage, and uninstalling EchoGesture deletes all of it.

## Sharing

EchoGesture shares nothing by itself. If you tap Copy or Share on a conversation, the text goes to the
clipboard or to the app you choose, as with any other text you share.

## Children

EchoGesture is not directed at children and collects no personal information from anyone.

## Changes

If this policy changes, the new version will be published here with a new date.

## Contact

Questions about this policy: [contact email to be added before publishing].
