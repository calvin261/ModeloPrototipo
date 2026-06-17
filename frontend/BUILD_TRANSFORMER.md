# Build APK with the SignTransformer ONNX

This is the recipe for building the Android APK that ships the
`SignTransformerClassifier` (ONNX Runtime) and falls back to the
legacy GRU (`DynamicSignClassifier`) when the transformer is missing.

## Prerequisites

- **Android Studio Hedgehog (2023.1.1) or newer.** The project uses
  Kotlin 2.0.0 + AGP 8.13.2 + Compose BOM 2024.02.
- **JDK 17** (bundled with Android Studio).
- **Infinix GT30 Pro** (Dimensity 8350, 8 GB) for the target smoke test,
  or any arm64-v8a device.
- **ONNX Runtime Android 1.17.3** is already pinned in
  `app/build.gradle.kts:137` — supports opset 18 (the export uses 18).
- **Trained ONNX** at
  `backend/models/exports/sign_transformer_v20260614_134616.onnx`
  (397 KB FP32, 12 classes). This file is the artifact that gets
  shipped to the phone by `ModelDownloadManager`.

## Steps

1. **Open the project in Android Studio.**

   File → Open → `frontend/`. Wait for the Gradle sync to finish.

2. **Add the model to the bundled fallback (optional, but recommended
   for offline runs).**

   The app first looks in `filesDir/models/sign_transformer.onnx`
   (downloaded from the backend at runtime). If you want a working APK
   without the backend, drop the file into
   `frontend/app/src/main/assets/sign_transformer.tflite` (the TFLite
   fallback path; the current model is ONNX, so this is a no-op until
   you also ship a TFLite version).

3. **Configure the backend URL.**

   Edit `frontend/app/build.gradle.kts` and set `LSEC_BASE_URL` for the
   `debug` build type to point at your dev box:

   ```kotlin
   buildConfigField("String", "LSEC_BASE_URL", "\"http://192.168.1.4:8000/\"")
   ```

   The release build uses the public Hugging Face Space by default.

4. **Verify the backend exposes the transformer endpoint.**

   With the backend running (`cd backend && start_backend.bat`), hit:

   ```bash
   curl http://localhost:8000/api/v1/models/latest/transformer
   ```

   You should get:

   ```json
   {
     "version": "20260614_134616",
     "download_url": "/api/v1/models/download/transformer/20260614_134616",
     "size_kb": 406.78,
     "accuracy": 0.0,
     "num_classes": 12
   }
   ```

   If you get HTTP 400/404, the registry hasn't picked up the
   transformer entry. Re-run `python -m scripts.export_transformer_onnx`
   from `backend/` so `models/exports/registry.json` is updated.

5. **Build the debug APK.**

   ```bash
   cd frontend
   gradlew.bat assembleDebug
   ```

   The signed-with-debug APK lands at
   `app/build/outputs/apk/debug/app-debug.apk` (~25 MB including
   ONNX Runtime native libs for all four ABIs).

   For a release APK with R8/proguard:

   ```bash
   gradlew.bat assembleRelease
   ```

   You'll need a `frontend/keystore.properties` with the four keys
   (`storeFile`, `storePassword`, `keyAlias`, `keyPassword`).

6. **Install on the device.**

   ```bash
   adb install -r app/build/outputs/apk/debug/app-debug.apk
   ```

7. **Smoke-test the transformer end-to-end.**

   1. Launch the app, allow camera permission.
   2. Go to Camera → "Dinámico" mode.
   3. Make the "hola" sign in front of the front camera.
   4. The bottom card should show `hola` with a confidence > 0.6 and
      the `LiveConfidenceBar` should track the softmax in real time.
   5. Check logcat:

      ```
      adb logcat -s SignTransformer:* DynamicSignClassifier:* SignClassifier:*
      ```

      You should see `Loaded sign_transformer ONNX (classes=12, ...)`
      on first launch and a per-frame latency under 50ms.

8. **Calibration (optional, 30s flow).**

   1. Tap "Calibrar (30s)" in the Camera screen.
   2. The screen is a stub — fill in the camera preview + countdown
      using the TODOs in `ui/screens/camera/CalibrationScreen.kt`.
   3. When calibration finishes, the facade
      (`SignClassifier.processDynamicFrame`) will route classifications
      through `CentroidClassifier.classify(...)` first, falling back to
      the transformer's logits if no centroid matches.

9. **Reformulate (optional, wired later).**

   The "Reformular" button placeholder is in
   `CameraScreen.kt::ActionButtonsRow`. The `onReformulate` callback
   has a `TODO(llm-subagent)` comment — the
   `lsc-llm-postprocessor` sub-agent will wire it to `Phi3PostProcessor`.

## Common failures

- **App crashes with `OrtSession` error**: the downloaded ONNX is
  truncated. Re-download via
  `modelDownloadManager.downloadLatestModel("transformer")` or wipe
  `filesDir/models/sign_transformer.onnx` and restart the app.

- **HTTP 400 from `getLatestModel("transformer")`**: the backend
  router was not updated. Check that
  `backend/api/routers/models_router.py` includes `transformer` in
  `_VALID_MODEL_TYPES` and that you restarted the FastAPI server.

- **Latency > 50ms**: the 2-thread `setIntraOpNumThreads(2)` in
  `SignTransformerClassifier.initialize()` caps the CPU budget. If
  it's still too slow, lower the thread count to 1 or switch the
  ONNX file to the INT8-quantized version
  (`sign_transformer_v*_int8.onnx`, ~120 KB). The router's
  `get_latest_model` already prefers the INT8 file when present.

- **Camera permission denied**: the `PermissionRequest` composable in
  `CameraScreen.kt` shows a button to re-request. If it still fails,
  check `AndroidManifest.xml` for `<uses-permission android:name="android.permission.CAMERA"/>`.

## Files added/changed by this integration

| File | Status |
|---|---|
| `app/src/main/java/com/lsec/translator/ml/SignTransformerClassifier.kt` | NEW |
| `app/src/main/java/com/lsec/translator/ml/SignClassifier.kt` | MODIFIED (transformer + centroid preference) |
| `app/src/main/java/com/lsec/translator/data/remote/ModelDownloadManager.kt` | MODIFIED (adds "transformer" type) |
| `app/src/main/java/com/lsec/translator/ui/screens/camera/CalibrationScreen.kt` | NEW (stub) |
| `app/src/main/java/com/lsec/translator/ui/screens/camera/CameraScreen.kt` | MODIFIED (Calibrar + Reformular + live confidence bar) |
| `app/src/main/java/com/lsec/translator/ui/navigation/LSECNavHost.kt` | MODIFIED (calibration route) |
| `backend/api/routers/models_router.py` | MODIFIED (accepts "transformer" model_type) |
| `docs/CHANGELOG.md` | MODIFIED (code · config entry) |
