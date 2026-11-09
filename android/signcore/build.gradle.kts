import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    kotlin("jvm")
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

dependencies {
    // The same ai.onnxruntime API ships in onnxruntime (desktop JVM) and onnxruntime-android; the app picks the Android one.
    compileOnly("com.microsoft.onnxruntime:onnxruntime:1.31.1")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.31.1")
    testImplementation("org.json:json:20240303")
    testImplementation(kotlin("test-junit5"))
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform()
    // Optional: a JDK whose bundled C++ runtime is new enough for ONNX Runtime on Windows (see README).
    System.getenv("TEST_JAVA_HOME")?.let {
        executable = "$it/bin/java" + if (System.getProperty("os.name").startsWith("Windows")) ".exe" else ""
    }
    systemProperty("parity.dir", rootProject.file("parity").absolutePath)
    inputs.dir(rootProject.file("parity")).optional()  // a new fixture or model reruns the tests
    testLogging { showStandardStreams = true; events("passed", "failed") }
}
