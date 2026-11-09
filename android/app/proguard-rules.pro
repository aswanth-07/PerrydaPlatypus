# Native code calls back into these classes by name; keep them whole.
-keep class com.google.mediapipe.** { *; }
-keep class ai.onnxruntime.** { *; }
-keep class com.google.protobuf.** { *; }
-dontwarn com.google.mediapipe.**
-dontwarn com.google.protobuf.**
-dontwarn com.google.auto.value.**
-dontwarn javax.annotation.**
-dontwarn org.checkerframework.**
