plugins {
    id("com.android.application")
    // AGP 9 compiles Kotlin itself; only the Compose compiler plugin is added.
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "ai.echogesture.app"
    // Current AndroidX (Compose BOM 2026.09) needs compile SDK 37; the app still targets 36 and runs from 28.
    compileSdk {
        version = release(37) { minorApiLevel = 2 }
    }
    buildToolsVersion = "36.0.0"

    defaultConfig {
        applicationId = "ai.echogesture.app"
        minSdk = 28            // MediaMetadataRetriever.getFrameAtIndex for the clip screen
        targetSdk = 36
        versionCode = 1
        versionName = "0.2"
        ndk { abiFilters += listOf("arm64-v8a") }
    }

    buildTypes {
        debug {
            ndk { abiFilters += listOf("x86_64") }   // also runs on the x86_64 emulator
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.getByName("debug")  // installable demo build
        }
    }

    packaging {
        jniLibs { useLegacyPackaging = true }   // compress native libraries in the APK: a much smaller download
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures { compose = true }

    androidResources {
        // MediaPipe memory-maps its bundle and ONNX Runtime reads the model whole; keep both uncompressed.
        noCompress += listOf("task", "onnx")
    }
}

dependencies {
    implementation(project(":signcore"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.1")
    implementation("com.google.mediapipe:tasks-vision:1.2.2")

    implementation(platform("androidx.compose:compose-bom:2026.09.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended:1.7.8")
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.11.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.11.0")
    implementation("androidx.navigation:navigation-compose:2.10.2")
    implementation("androidx.datastore:datastore-preferences:1.3.0")

    implementation("androidx.camera:camera-core:1.6.2")
    implementation("androidx.camera:camera-camera2:1.6.2")
    implementation("androidx.camera:camera-lifecycle:1.6.2")
    implementation("androidx.camera:camera-view:1.6.2")
}

// The Holistic bundle and the exported vocabularies (core: 100 words, full: 2,000, union: 3,075 when exported)
// are copied from src/models (the same files predict() loads, kept in Git) into assets before each build; Sync
// also removes anything stale from earlier layouts.
val modelDir = rootProject.file("../src/models")
val copyModels = tasks.register<Sync>("copyModels") {
    into(layout.projectDirectory.dir("src/main/assets"))
    from(modelDir) { include("holistic_landmarker.task") }
    for (v in listOf("core", "full", "union").filter { File(modelDir, it).isDirectory }) {
        from(File(modelDir, v)) { include("sign_int8.onnx", "labels.json", "export.json", "sets.json"); into(v) }
    }
}
tasks.named("preBuild") { dependsOn(copyModels) }
