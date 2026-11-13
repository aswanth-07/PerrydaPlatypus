# Native code calls back into these classes by name; keep them whole.
-keep class com.google.mediapipe.** { *; }
-keep class ai.onnxruntime.** { *; }
-keep class com.google.protobuf.** { *; }
# MediaPipe logs through Flogger, which finds its caller by looking for its own class names on the stack; renamed,
# Graph's static initialiser throws "no caller found on the stack" and the release build crashes on launch.
-keep class com.google.common.flogger.** { *; }
-dontwarn com.google.common.flogger.**
-dontwarn com.google.mediapipe.**
-dontwarn com.google.protobuf.**
-dontwarn com.google.auto.value.**
-dontwarn javax.annotation.**
-dontwarn org.checkerframework.**
