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
            // `-PemulatorAbi` also packs x86_64, so the shrunk release build can be tried on an x86_64 emulator
            // (arm64 ONNX Runtime crashes while loading under the emulator's ARM translation).
            if (providers.gradleProperty("emulatorAbi").isPresent) ndk { abiFilters += listOf("x86_64") }
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

// Assets are generated, not kept in Git: before every build Sync copies the EchoGesture model, its labels, same-sign
// sets and export record and the MediaPipe Holistic bundle from src/models (the files predict() loads), and the
// live-caption model, its decoder settings and the Learn demonstrations from android/models.
val copyModels = tasks.register<Sync>("copyModels") {
    into(layout.projectDirectory.dir("src/main/assets"))
    from(rootProject.file("../src/models")) {
        include("echogesture_int8.onnx", "labels.json", "export.json", "sets.json", "holistic_landmarker.task")
    }
    from(rootProject.file("models")) { include("echogesture_live_int8.onnx", "live.json", "demos.bin") }
}
tasks.named("preBuild") { dependsOn(copyModels) }
