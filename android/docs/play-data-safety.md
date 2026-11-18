# Play Console: data safety answers for EchoGesture

How to fill in the Data safety form, and why. Checked against the app at versionName 1.0: the merged manifest has
CAMERA and RECORD_AUDIO and no INTERNET or ACCESS_NETWORK_STATE permission (`aapt2 dump permissions`), and
onnxruntime-android's telemetry provider is removed from it.

## Data collection and security

- **Does your app collect or share any of the required user data types?** No.
  Google's definition of "collect" is transmitting data off the device. EchoGesture has no internet permission,
  so it cannot transmit anything. Processing on the device, and data kept only on the device, is not collection.
- **Is all of the user data collected by your app encrypted in transit?** Not applicable (nothing is collected).
- **Do you provide a way for users to request that their data is deleted?** Not applicable; users delete their
  own data in the app (History, Settings > Delete all conversations, Teach > delete a sign, Settings > Reset
  learning progress) and by uninstalling.

With "No" to the first question, Play shows "No data collected" and "No data shared with third parties".

## Notes for the review

- Camera and microphone: used on the device only (MediaPipe, ONNX Runtime, Android's on-device speech
  recogniser). No frames or audio are stored.
- Text-to-speech: the app prefers an installed voice and warns in Settings when none is installed. The
  text-to-speech engine is a separate app on the phone with its own policy.
- Photo picker: a video the user picks is read in memory to identify the sign and not stored.
- Backups: `android:allowBackup="false"` and data extraction rules exclude everything from cloud backup and
  device transfer.

## Other Play Console items

- **Privacy policy URL:** host `privacy-policy.md` (for example as a GitHub Pages page) and add the contact email
  first.
- **App access:** no login; all functions are available without special access.
- **Ads:** no ads.
- **Content rating:** no user-generated content is shared, no violence; expected rating Everyone / PEGI 3.
- **Target audience:** not designed for children.
- **Health apps declaration:** EchoGesture is an accessibility and learning tool, not a medical device.
- **Datasets:** the model is trained on WLASL and ASL Citizen. Whether their licences allow a free Play release
  is still to be checked before publishing.
