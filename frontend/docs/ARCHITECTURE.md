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
