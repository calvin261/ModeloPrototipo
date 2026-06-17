# Arquitectura Técnica - LSEC Translator

## Visión General

Este documento describe la arquitectura técnica del sistema de reconocimiento de Lenguaje de Señas Ecuatoriano (LSEC) en tiempo real.

## Diagrama de Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CAPA DE PRESENTACIÓN                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  HomeScreen │  │CameraScreen │  │ Dictionary  │  │  Settings   │        │
│  │             │  │             │  │   Screen    │  │   Screen    │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                │
│         └────────────────┼────────────────┼────────────────┘                │
│                          │                │                                 │
│                    ┌─────┴─────┐    ┌─────┴─────┐                           │
│                    │  Camera   │    │Dictionary │                           │
│                    │ ViewModel │    │ ViewModel │                           │
│                    └─────┬─────┘    └─────┬─────┘                           │
└──────────────────────────┼────────────────┼─────────────────────────────────┘
                           │                │
┌──────────────────────────┼────────────────┼─────────────────────────────────┐
│                          │  CAPA DE DOMINIO                                 │
├──────────────────────────┼────────────────┼─────────────────────────────────┤
│  ┌───────────────────────┴────────────────┴───────────────────────┐         │
│  │                         USE CASES                               │         │
│  ├─────────────────┬──────────────────┬───────────────────────────┤         │
│  │ DetectHandsUC   │ ClassifySignUC   │ SynthesizeSpeechUC        │         │
│  └────────┬────────┴────────┬─────────┴─────────────┬─────────────┘         │
└───────────┼─────────────────┼───────────────────────┼───────────────────────┘
            │                 │                       │
┌───────────┼─────────────────┼───────────────────────┼───────────────────────┐
│           │     CAPA DE DATOS / ML                  │                       │
├───────────┼─────────────────┼───────────────────────┼───────────────────────┤
│  ┌────────┴────────┐ ┌──────┴───────┐ ┌─────────────┴───────────────┐       │
│  │  HandDetector   │ │SignClassifier│ │       TTSEngine             │       │
│  │ (YOLOv8+MPipe)  │ │   (LSTM)     │ │   (Google TTS + Custom)     │       │
│  └────────┬────────┘ └──────┬───────┘ └─────────────────────────────┘       │
│           │                 │                                               │
│  ┌────────┴─────────────────┴────────────────────────────────────────┐      │
│  │                    TensorFlow Lite Runtime                         │      │
│  │                    (GPU Delegate / INT8 Quantization)              │      │
│  └────────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                    Room Database (SQLite)                           │     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │     │
│  │  │  SignDao    │  │  SignEntity │  │ Repository  │                  │     │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                    DataStore Preferences                            │     │
│  │                    (User Settings)                                  │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Pipeline de Procesamiento de Video

### Flujo de Datos

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   CameraX   │───▶│   YOLOv8    │───▶│  MediaPipe  │───▶│    LSTM     │
│  (30 FPS)   │    │   (ROI)     │    │  (Landmarks)│    │ (Classify)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
   640x480           Bounding Box      21 landmarks x 3    Sign Label
   YUV_420_888       [x,y,w,h,conf]    [x,y,z] normalized   + Confidence
       │                  │                  │                  │
       │                  │                  │                  ▼
       │                  │                  │           ┌─────────────┐
       │                  │                  │           │  TTSEngine  │
       │                  │                  │           │  (Speak)    │
       │                  │                  │           └─────────────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
    ~33ms              ~10ms              ~15ms              ~100ms
```

### Desglose de Latencia

| Etapa | Tiempo (ms) | Optimización |
|-------|-------------|--------------|
| Captura de frame | 33 | CameraX ImageAnalysis |
| Preprocesamiento | 5 | Native buffer operations |
| YOLOv8 inference | 10-15 | TFLite GPU delegate |
| MediaPipe Hands | 10-15 | Pre-built SDK |
| Feature extraction | 2-3 | Kotlin native |
| LSTM inference | 20-30 | INT8 quantization |
| Post-processing | 2 | Efficient Kotlin |
| TTS synthesis | 100-200 | Cached audio |
| **Total** | **~200-300ms** | **< 500ms target ✓** |

## Modelo LSTM

### Arquitectura

```
Input Shape: (30, 63)  # 30 frames × (21 landmarks × 3 coords)
    │
    ▼
┌─────────────────────────────────────┐
│         BatchNormalization          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│    Bidirectional LSTM (128 units)   │
│         return_sequences=True       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│           Dropout (0.3)             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│    Bidirectional LSTM (128 units)   │
│         return_sequences=False      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│           Dropout (0.3)             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│          Dense (256, ReLU)          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│         BatchNormalization          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│           Dropout (0.3)             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│       Dense (N classes, Softmax)    │
└─────────────────────────────────────┘
    │
    ▼
Output: Probability distribution over N sign classes
```

### Cuantización INT8

- **Tamaño original**: ~2.5 MB
- **Tamaño cuantizado**: ~650 KB
- **Speedup**: ~2x en CPU, ~1.5x con GPU delegate
- **Pérdida de precisión**: <1%

## Estructura de Datos

### SignEntity (Room)

```kotlin
@Entity(tableName = "signs")
data class SignEntity(
    @PrimaryKey val id: Long,
    val name: String,                    // "Hola"
    val spanishTranslation: String,      // "Hola"
    val ecuadorianVariant: String,       // "¡Qué más!"
    val description: String,             // Instrucciones
    val category: SignCategory,          // GREETINGS
    val difficulty: SignDifficulty,      // BEGINNER
    val isDynamic: Boolean,              // true
    val videoUrl: String?,               // Tutorial video
    val usageCount: Int,                 // Analytics
    val isFavorite: Boolean              // User preference
)
```

### Landmark Features

```kotlin
// 21 landmarks por mano, cada uno con x, y, z
data class HandLandmarks(
    val landmarks: FloatArray  // Size: 63 (21 × 3)
)

// Índices de landmarks de MediaPipe Hands:
// 0: WRIST
// 1-4: THUMB (CMC, MCP, IP, TIP)
// 5-8: INDEX (MCP, PIP, DIP, TIP)
// 9-12: MIDDLE (MCP, PIP, DIP, TIP)
// 13-16: RING (MCP, PIP, DIP, TIP)
// 17-20: PINKY (MCP, PIP, DIP, TIP)
```

## Inyección de Dependencias (Hilt)

```
@HiltAndroidApp
LSECApplication
    │
    ├── @Singleton
    │   ├── LSECDatabase
    │   ├── SignDao
    │   ├── SignRepository
    │   ├── HandDetector
    │   ├── SignClassifier
    │   ├── TTSEngine
    │   └── UserPreferences
    │
    └── @ViewModelScoped
        ├── CameraViewModel
        ├── DictionaryViewModel
        └── SettingsViewModel
```

## Estrategia de TTS con Ecuatorianismos

### Mapeo de Vocabulario

```
┌─────────────────────────────────────────────────────────────────┐
│                    TTSEngine                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  speak("Chévere")                                              │
│       │                                                        │
│       ▼                                                        │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ ecuadorianAudioMap.containsKey("Chévere")?          │       │
│  └─────────────────────────────────────────────────────┘       │
│       │                                                        │
│       ├── YES ──▶ Play audio_chevere.mp3 (cached)             │
│       │                                                        │
│       └── NO ───▶ Google TTS (es-EC) ──▶ AudioTrack            │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

### Banco de Audio Pregrabado

```
res/raw/
├── audio_que_mas.mp3      # "¡Qué más!"
├── audio_pana.mp3         # "Pana"
├── audio_chevere.mp3      # "Chévere"
├── audio_full.mp3         # "Full"
├── audio_bacan.mp3        # "Bacán"
├── audio_guagua.mp3       # "Guagua"
├── audio_nanio.mp3        # "Ñaño"
└── ...
```

## Post-procesador LLM Opcional (Phi-3-mini INT4)

Capa **opcional** y **off-by-default** que toma la secuencia de glosas
producida por el MLP/GRU y devuelve una oración natural en español.
NO es un clasificador: no se entrena, no reemplaza al MLP ni al GRU.

```
                ┌────────────────────────────────────────────────────┐
                │          CameraScreen / ViewModel                  │
                │  Glosas detectadas: ["hola", "como", "estas"]      │
                └───────────────────────┬────────────────────────────┘
                                        │
                          (toggle llmEnabled = true)
                                        │
                                        ▼
                ┌────────────────────────────────────────────────────┐
                │     LlmPostProcessor (interfaz, suspend)           │
                │     Phi3LlamaCppPostProcessor (impl)               │
                │   - check: isEnabled? (preferencia AND >=6GB RAM) │
                │   - check: isReady?   (lazy init via ensureLoaded) │
                │   - build prompt: "Seyas: hola, como, estas -> …"  │
                └───────────────────────┬────────────────────────────┘
                                        │ withContext(Dispatchers.IO)
                                        ▼
                ┌────────────────────────────────────────────────────┐
                │         Phi3LlamaCppBridge (object, JNI)          │
                │   - System.loadLibrary("llama")                   │
                │   - external fun init / complete / release         │
                └───────────────────────┬────────────────────────────┘
                                        │ JNI
                                        ▼
                ┌────────────────────────────────────────────────────┐
                │     libllama.so (C++, en jniLibs/<abi>/)           │
                │   - llama_model_load_from_file(gguf)               │
                │   - llama_decode + sampler en bucle                │
                │   - llama_free al terminar                         │
                └────────────────────────────────────────────────────┘
```

### Decisiones de diseño

1. **Lazy init**: el modelo de 2.2 GB NO se carga al abrir la app.
   Se carga on-demand en el primer `rephrase()` (o cuando el usuario
   pulsa el botón "Reformular" tras la descarga). El primer token
   tarda 3-5s — la UI muestra un spinner "Pensando...".

2. **Modelo on-demand, NUNCA bundleado**: el `.gguf` se descarga vía
   `ModelDownloadManager.downloadPhi3Gguf()` cuando el usuario
   activa el toggle. Razón: el límite de APK de Play Store es 2.2 GB
   y dejaría la app inutilizable en devices con poco almacenamiento.

3. **Auto-disable en <6 GB RAM**: `Phi3LlamaCppPostProcessor.hasEnoughRam()`
   chequea `ActivityManager.getMemoryInfo()` y fuerza `isEnabled=false`
   sin tocar la preferencia persistida. 8 GB en el GT30 Pro es el floor
   funcional — con menos, el sistema mata el proceso durante inferencia.

4. **Mutex + Dispatchers.IO**: el bridge nativo es bloqueante; un
   solo `rephrase()` corre a la vez. No vale la pena streaming con
   callback — la salida es <30 tokens (2-4 tok/s = ~10s máximo).

5. **Fallback degradado**: si el modelo no está listo o la inferencia
   falla, `rephrase()` devuelve `signSequence.joinToString(" ")`. La
   app sigue funcionando como antes — el LLM es puramente aditivo.

6. **System prompt fijo**: "Eres un asistente que convierte secuencias
   de seyas LSC a oraciones naturales en espanol. Manten el significado
   original. Se conciso." Anclado en `LlmPostProcessor.SYSTEM_PROMPT`.

7. **Temperatura 0.3**: determinista-suave. 0.0 es robótico, >0.7
   inventa contenido. El rephrase debe ser fiel al significado de
   las glosas.

### Comparación de backends

| Backend | APK impact | Startup | Phi-3 support | Veredicto |
|---|---|---|---|---|
| **llama.cpp + JNI** | +20-30 MB (.so) | 1-2s | Excelente (GGUF) | **Elegido** |
| MLC LLM | +400-600 MB | <500ms (Vulkan) | Limitado, no INT4 limpio | Rechazado (APK) |
| ONNX Runtime GenAI | +80 MB | ~1s | Nativo (Phi-3 ONNX oficial) | Backup si JNI falla |

Razones detalladas y pasos de compilación en `frontend/SETUP_LLM.md`.

### Flujo del usuario

1. Settings → "Reformular con IA" → toggle ON.
2. App descarga GGUF (2.2 GB, una vez, con progreso visible).
3. En cámara, modo "dynamic", tras 2-3 señas reconocidas:
   - **Manual**: usuario pulsa el botón "Reformular" junto a la glosa.
   - **Auto** (si `llmAutoTrigger=true`): cuando se acumulan 3+ glosas
     en 6 segundos, se invoca `rephrase()` automáticamente.
4. UI muestra "Pensando..." durante 3-5s, luego actualiza la frase
   mostrada encima del botón con la oración en español.

## Estrategias de Optimización

### 1. Frame Skipping Adaptativo

```kotlin
class AdaptiveFrameProcessor {
    private var lastProcessedTime = 0L
    private var targetIntervalMs = 33L  // 30 FPS default
    
    fun shouldProcessFrame(): Boolean {
        val now = System.currentTimeMillis()
        if (now - lastProcessedTime >= targetIntervalMs) {
            lastProcessedTime = now
            return true
        }
        return false
    }
    
    fun adjustForBattery(isBatterySaver: Boolean) {
        targetIntervalMs = if (isBatterySaver) 66L else 33L  // 15 vs 30 FPS
    }
}
```

### 2. Pipeline Asíncrono

```kotlin
viewModelScope.launch {
    cameraFlow
        .buffer(Channel.CONFLATED)  // Drop old frames
        .mapLatest { frame ->
            withContext(Dispatchers.Default) {
                detectHands(frame)
            }
        }
        .filter { it != null }
        .mapLatest { landmarks ->
            withContext(Dispatchers.Default) {
                classifySign(landmarks)
            }
        }
        .collect { result ->
            updateUI(result)
            speakIfNeeded(result)
        }
}
```

### 3. Model Caching

```kotlin
@Singleton
class ModelCache @Inject constructor() {
    private var interpreter: Interpreter? = null
    
    fun getOrCreate(context: Context): Interpreter {
        return interpreter ?: createInterpreter(context).also {
            interpreter = it
        }
    }
}
```

## Pruebas

### Unit Tests

```kotlin
@Test
fun `classify should return correct sign for valid sequence`() {
    val landmarks = generateTestLandmarks()
    val result = classifier.classify(landmarks)
    
    assertThat(result.confidence).isGreaterThan(0.5f)
    assertThat(result.signName).isNotEmpty()
}
```

### Integration Tests

```kotlin
@HiltAndroidTest
class CameraIntegrationTest {
    @Test
    fun camera_shouldDetectHands_whenVisible() {
        // Setup camera preview
        // Send test frame
        // Verify hand detection
    }
}
```

## Métricas de Rendimiento

### Profiling en Dispositivo

```kotlin
class PerformanceTracker {
    fun measureLatency(block: () -> Unit): Long {
        val start = System.nanoTime()
        block()
        return (System.nanoTime() - start) / 1_000_000  // ms
    }
}
```

### Firebase Performance

```kotlin
Firebase.performance.newTrace("sign_detection").apply {
    start()
    // Detection logic
    putMetric("latency_ms", latency)
    putAttribute("sign_detected", signName)
    stop()
}
```

## Consideraciones de Seguridad

- ✅ Procesamiento on-device (sin envío de datos a servidores)
- ✅ Sin almacenamiento de frames de cámara
- ✅ Permisos mínimos requeridos (solo cámara)
- ✅ ProGuard para ofuscación de código

## Compatibilidad

- **API mínimo**: 26 (Android 8.0)
- **API objetivo**: 34 (Android 14)
- **Arquitecturas**: armeabi-v7a, arm64-v8a, x86, x86_64
- **RAM mínima**: 2GB
- **Almacenamiento**: ~100MB
