# SETUP_LLM.md — Post-procesador Phi-3-mini INT4 (opcional)

## Por que Phi-3-mini + llama.cpp

| Opcion | Pros | Contras | Veredicto |
|---|---|---|---|
| **llama.cpp Android** | GGUF maduro, community enorme, JNI directo, binarios pequenos | Setup JNI manual, primer token lento | **Recomendado** |
| MLC LLM | Startup rapido, compila a NDK con Vulkan | APK +600 MB, no soporta Phi-3 bien | Rechazado (APK) |
| ONNX Runtime GenAI | API simple, oficial MS para Phi-3 ONNX | Binario ~80 MB, sin cuantizacion <INT4 | Backup si JNI falla |

**Recomendacion: llama.cpp.** El APK queda <30 MB de `.so` (vs +600 MB de
MLC), y el modelo INT4 GGUF (2.2 GB) se descarga on-demand. El primer
token tarda 3-5s en el GT30 Pro — la UI debe mostrar "Pensando...".

## Pasos

1. **Descargar el modelo GGUF** (una vez, ~2.2 GB):
   ```bash
   # Recomendado: Q4_K_M de TheBloke — buen balance tamano/calidad
   wget -O frontend/app/src/main/assets/phi3-mini-int4.gguf \
     https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi3-mini-int4.gguf
   # (Mejor: NO bundlearlo. Descargarlo on-demand — ver paso 5.)
   ```
   **NO** incluir el `.gguf` en el APK (excede 2.2 GB Play Store limit).

2. **Compilar llama.cpp para Android** (una vez, ~10 min):
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp/examples/llama.android
   ./gradlew assembleRelease   # genera libllama.so por ABI
   # Output: build/intermediates/.../jni/<abi>/libllama.so
   ```
   Alternativa minima (sin example app): usar `CMakeLists.txt` de la app
   con `add_subdirectory(llama.cpp)` y `target_link_libraries(... llama)`.

3. **Copiar `.so` a la app** (4 ABIs):
   ```
   frontend/app/src/main/jniLibs/
   ├── arm64-v8a/libllama.so    # obligatorio (GT30 Pro)
   ├── armeabi-v7a/libllama.so  # 32 bits legacy
   ├── x86/libllama.so          # emulador
   └── x86_64/libllama.so       # emulador 64
   ```

4. **Wire en Gradle** — agregar a `frontend/app/build.gradle.kts`:
   ```kotlin
   android {
       externalNativeBuild {
           cmake {
               path = file("src/main/cpp/CMakeLists.txt")
               version = "3.22.1"
           }
       }
   }
   dependencies {
       implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
   }
   ```

5. **Trigger de descarga on-demand** (recomendado):
   - Llamar `modelDownloadManager.downloadPhi3Gguf { progress -> ... }` desde
     SettingsScreen cuando el usuario activa el toggle LLM.
   - Guarda en `filesDir/models/phi3-mini-int4.gguf`.
   - `Phi3LlamaCppPostProcessor` lo busca ahi en el primer `rephrase()`.

6. **Implementar los JNI bindings** — ver `Phi3LlamaCppBridge.kt`:
   - 4 funciones `external fun`: `init`, `complete`, `release`, (opcional `tokenize`).
   - Nombres C++: `Java_com_lsec_translator_llm_Phi3LlamaCppBridge_<nombre>`.
   - Detalles tecnicos en KDoc dentro del archivo `.kt`.

## Verificacion

```bash
cd frontend && ./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
# En el device: Settings → LLM → toggle ON → esperar descarga → "Reformular" en camara
```

## Pitfalls

- **2.2 GB APK limit**: modelo SIEMPRE on-demand, nunca bundled.
- **<6 GB RAM**: auto-desactivar (ya implementado en `hasEnoughRam()`).
- **First-token 3-5s**: UI debe mostrar "Pensando..." con un spinner.
- **JNI fiddly**: si libllama.so no carga, todos los `external fun` lanzan
  `UnsatisfiedLinkError` — el wrapper lo atrapa y cae al fallback de
  concatenacion simple. La app sigue funcionando, sin LLM.
