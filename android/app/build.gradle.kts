import java.util.Properties

plugins {
    id("com.android.application")
    // AGP 9 compiles Kotlin itself; only the Compose compiler plugin is added.
    id("org.jetbrains.kotlin.plugin.compose")
}

// Release signing: android/keystore.properties (ignored by Git) names the Play upload key:
//   storeFile=C:/keys/echogesture-upload.jks
//   storePassword=…
//   keyAlias=upload
//   keyPassword=…
// Without it a release build is signed with the debug key: installable for testing, not accepted by Play.
val uploadKey: Properties? = rootProject.file("keystore.properties").takeIf { it.exists() }
    ?.let { f -> Properties().apply { f.inputStream().use { load(it) } } }

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
        versionCode = 2
        versionName = "1.0"
        ndk { abiFilters += listOf("arm64-v8a") }
    }

    signingConfigs {
        if (uploadKey != null) create("upload") {
            storeFile = file(uploadKey.getProperty("storeFile"))
            storePassword = uploadKey.getProperty("storePassword")
            keyAlias = uploadKey.getProperty("keyAlias")
            keyPassword = uploadKey.getProperty("keyPassword")
        }
    }

    buildTypes {
        debug {
            ndk { abiFilters += listOf("x86_64") }   // also runs on the x86_64 emulator
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.findByName("upload") ?: signingConfigs.getByName("debug")
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

    testOptions {
        unitTests.all { it.useJUnitPlatform() }
    }

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
    implementation("androidx.core:core-splashscreen:1.0.1")

    implementation("androidx.camera:camera-core:1.6.2")
    implementation("androidx.camera:camera-camera2:1.6.2")
    implementation("androidx.camera:camera-lifecycle:1.6.2")
    implementation("androidx.camera:camera-view:1.6.2")

    // JVM unit tests for the app's pure logic (conversations, progress, lessons, caption text).
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit5:2.4.21")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

// Assets are generated, not kept in Git: before every build Sync copies the unified EchoGesture INT8 model, its labels,
// same-sign sets, export record and MediaPipe Holistic bundle from src/models (the files predict() loads), and the
// continuous decoder settings and Learn demonstrations from android/models.
val copyModels = tasks.register<Sync>("copyModels") {
    into(layout.projectDirectory.dir("src/main/assets"))
    from(rootProject.file("../src/models")) {
        include("echogesture_int8.onnx", "labels.json", "export.json", "sets.json", "holistic_landmarker.task")
    }
    from(rootProject.file("models")) { include("live.json", "demos.bin") }
}
tasks.named("preBuild") { dependsOn(copyModels) }
