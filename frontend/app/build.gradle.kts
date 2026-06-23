import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.hilt.android)
    alias(libs.plugins.ksp)
}

// Lee keystore.properties desde la raíz del proyecto Gradle (frontend/).
// En este proyecto rootProject == frontend/ (single-module), por lo que
// `keystore.properties` se resuelve a frontend/keystore.properties,
// que SI esta en .gitignore (linea 45). NO anteponer "frontend/" al nombre.
val keystorePropertiesFile = rootProject.file("keystore.properties")
val keystoreProperties = Properties().apply {
    if (keystorePropertiesFile.exists()) load(keystorePropertiesFile.inputStream())
}

// Default backend URL — overridable from `keystore.properties` via `lsec.defaultBackendUrl`
// or via the LSEC_DEFAULT_BACKEND_URL env var (used by the release GitHub Actions workflow).
val defaultBackendUrl: String = (System.getenv("LSEC_DEFAULT_BACKEND_URL")
    ?: (keystoreProperties["lsec.defaultBackendUrl"] as String?)
    ?: "https://calvin261-lsec-api.hf.space/")

// Default API key baked into release builds. Empty by default — set via LSEC_API_KEY
// env var (CI/CD) or `keystore.properties` (`lsec.apiKey`). Empty in debug builds.
val defaultApiKey: String = (System.getenv("LSEC_API_KEY")
    ?: (keystoreProperties["lsec.apiKey"] as String?)
    ?: "")

android {
    namespace = "com.lsec.translator"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.lsec.translator"
        minSdk = 26
        targetSdk = 34
        versionCode = (project.findProperty("versionCode") as String?)?.toIntOrNull() ?: 1
        versionName = (project.findProperty("versionName") as String?) ?: "1.0.0"

        // URL base del backend. Sobrescrita por buildTypes. Cambiable desde Configuracion > URL del servidor.
        buildConfigField("String", "LSEC_BASE_URL", "\"$defaultBackendUrl\"")
        buildConfigField("String", "DEFAULT_BACKEND_URL", "\"$defaultBackendUrl\"")
        // THREAT MODEL para BUNDLED_API_KEY:
        //   - Es un String publico final en BuildConfig; apktool + strings lo recupera.
        //   - Aceptable porque el APK solo se distribuye via GitHub Releases a ~10 usuarios
        //     de confianza (ver AGENTS.md). El interceptor de NetworkModule prioriza la
        //     clave guardada por el usuario en DataStore sobre esta baked-in.
        //   - Mitigaciones: ProGuard + rotacion trimestral re-tagging v1.x.0.
        //   - NO hardcodear aqui valores reales — siempre via LSEC_API_KEY (env var) o
        //     frontend/keystore.properties (gitignored).
        buildConfigField("String", "BUNDLED_API_KEY", "\"$defaultApiKey\"")

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        vectorDrawables {
            useSupportLibrary = true
        }

        // Configuración para TensorFlow Lite
        ndk {
            // Default (debug): all 4 ABIs for emulator testing.
            // Release overrides below to drop x86/x86_64 (emulator-only).
            abiFilters += listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64")
        }
    }

    signingConfigs {
        create("release") {
            // Prefer env vars (GitHub Actions secrets), fall back to keystore.properties (local).
            val envStoreFile = System.getenv("LSEC_STORE_FILE")
            val envStorePass = System.getenv("LSEC_STORE_PASS")
            val envKeyAlias = System.getenv("LSEC_KEY_ALIAS") ?: "lsec"
            val envKeyPass = System.getenv("LSEC_KEY_PASS")

            val propStoreFile = keystoreProperties["storeFile"] as String?
            val propStorePass = keystoreProperties["storePassword"] as String?
            val propKeyAlias = keystoreProperties["keyAlias"] as String?
            val propKeyPass = keystoreProperties["keyPassword"] as String?

            storeFile = rootProject.file(envStoreFile ?: propStoreFile ?: "keystore/release.keystore")
            storePassword = envStorePass ?: propStorePass ?: ""
            keyAlias = envKeyAlias ?: propKeyAlias ?: "lsec"
            keyPassword = envKeyPass ?: propKeyPass ?: ""
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            signingConfig = signingConfigs.getByName("release")
            buildConfigField("String", "LSEC_BASE_URL", "\"$defaultBackendUrl\"")
            buildConfigField("String", "DEFAULT_BACKEND_URL", "\"$defaultBackendUrl\"")
            // Release preserva la API key embebida (seteada via env/keystore.properties)
            buildConfigField("String", "BUNDLED_API_KEY", "\"$defaultApiKey\"")
            // Release targets physical phones only — drop emulator ABIs to
            // roughly halve the APK and shave ~30% off build time. Debug
            // keeps all 4 (set in defaultConfig above) for local emulator testing.
            ndk {
                abiFilters.clear()
                abiFilters += listOf("arm64-v8a", "armeabi-v7a")
            }
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isMinifyEnabled = false
            // Debug usa los valores de keystore.properties / env vars; si no hay
            // ninguno, cae a defaults razonables (PC local). Cambiable desde
            // Configuracion > URL del servidor en runtime.
            buildConfigField("String", "LSEC_BASE_URL", "\"$defaultBackendUrl\"")
            buildConfigField("String", "DEFAULT_BACKEND_URL", "\"$defaultBackendUrl\"")
            buildConfigField("String", "BUNDLED_API_KEY", "\"$defaultApiKey\"")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        buildConfig = true
        compose = true
        mlModelBinding = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
            excludes += "META-INF/DEPENDENCIES"
        }
    }
}

ksp {
    arg("room.schemaLocation", "$projectDir/schemas")
    arg("room.incremental", "true")
}

dependencies {
    // Core Android
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.activity.compose)

    // Jetpack Compose
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.material.icons.extended)

    // Navigation
    implementation(libs.androidx.navigation.compose)

    // CameraX
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
    implementation(libs.androidx.camera.extensions)

    // MediaPipe
    implementation(libs.mediapipe.hands)

    // TensorFlow Lite (kept for compatibility)
    implementation(libs.tensorflow.lite)
    implementation(libs.tensorflow.lite.gpu)
    implementation(libs.tensorflow.lite.support)
    implementation(libs.tensorflow.lite.metadata)

    // ONNX Runtime for Android (sign classifier models downloaded from server)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.3")

    // Room Database
    implementation(libs.androidx.room.runtime)
    implementation(libs.androidx.room.ktx)
    ksp(libs.androidx.room.compiler)

    // Hilt Dependency Injection
    implementation(libs.hilt.android)
    ksp(libs.hilt.compiler)
    implementation(libs.androidx.hilt.navigation.compose)

    // Coroutines
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.coroutines.core)

    // DataStore
    implementation(libs.androidx.datastore.preferences)

    // Accompanist (Permissions)
    implementation(libs.accompanist.permissions)

    // Networking (API communication)
    implementation(libs.retrofit.core)
    implementation(libs.retrofit.moshi)
    implementation(libs.okhttp.core)
    implementation(libs.okhttp.logging)
    implementation(libs.moshi.kotlin)
    ksp(libs.moshi.codegen)

    // Firebase (Analytics)
    implementation(platform(libs.firebase.bom))
    implementation(libs.firebase.analytics)

    // Testing
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.ui.test.junit4)
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.ui.test.manifest)
}
