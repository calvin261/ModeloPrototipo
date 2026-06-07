# Plan: App Móvil de Reconocimiento LSEC en Tiempo Real

Desarrollar una aplicación Android nativa que reconozca el Lenguaje de Señas Ecuatoriano (LSEC) mediante visión por computadora y deep learning híbrido, traduciendo en tiempo real a voz sintetizada con ecuatorianismos. Arquitectura: YOLOv8-nano + MediaPipe Hands + LSTM/Transformer, con inferencia on-device para alcanzar <500ms de latencia y ≥85% de precisión.

---

## Steps

1. **Configurar proyecto Android nativo con Kotlin** - Crear estructura base con Jetpack Compose, CameraX, Hilt para inyección de dependencias, y TensorFlow Lite como runtime de ML.

2. **Implementar pipeline de captura y detección** - Integrar CameraX para captura de frames, YOLOv8-nano (exportado a TFLite) para detección de manos/ROI, y MediaPipe Hands SDK para extracción de 21 landmarks 3D por mano.

3. **Desarrollar modelo de clasificación temporal** - Diseñar red LSTM/Bi-LSTM (2 capas, 128-256 unidades) o Transformer ligero para clasificar secuencias de landmarks en señas LSEC; exportar a TFLite cuantizado (INT8).

4. **Crear motor TTS con ecuatorianismos** - Combinar Google TTS on-device para texto estándar con banco de audio pregrabado para expresiones ecuatorianas (ej: "pana", "qué más", "chévere"), con mapeo vocabulario→audio.

5. **Optimizar para latencia <500ms** - Aplicar cuantización INT8, GPU delegate de TFLite, pipeline asíncrono con Kotlin Coroutines, y frame skipping adaptativo para garantizar rendimiento en dispositivos mid-range.

6. **Implementar UI/UX y módulo de diccionario** - Diseñar interfaz con vista de cámara en vivo, indicador de seña reconocida, controles de voz, y base de datos local (Room) con diccionario LSEC expandible.

---

## Arquitectura Técnica Recomendada

### Stack Tecnológico

| Capa | Tecnología | Versión |
|------|------------|---------|
| **Plataforma** | Android (Kotlin) | API 26+ (Android 8.0+) |
| **UI Framework** | Jetpack Compose | Latest stable |
| **Cámara** | CameraX | 1.3.x |
| **Hand Tracking** | MediaPipe Hands | 0.10.x |
| **Object Detection** | YOLOv8-nano | TFLite export |
| **Sequence Model** | LSTM/GRU | TFLite (custom) |
| **ML Runtime** | TensorFlow Lite | 2.14+ |
| **Database** | Room | 2.6.x |
| **TTS** | Google TTS + Custom Audio | - |
| **Dependency Injection** | Hilt | 2.48+ |
| **Async** | Kotlin Coroutines + Flow | 1.7.x |
| **Analytics** | Firebase Analytics | - |

### Pipeline de Procesamiento

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Camera Frame                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           YOLOv8-nano (Hand Detection & ROI)                │
│           - Detects hands in frame                           │
│           - Provides bounding boxes                          │
│           - ~5ms inference on mobile                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           MediaPipe Hands (Landmark Extraction)             │
│           - 21 3D landmarks per hand                         │
│           - Normalized coordinates                           │
│           - ~10ms on mobile                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│      Feature Engineering (Preprocessing)                     │
│      - Relative coordinates                                  │
│      - Velocity/acceleration features                        │
│      - Hand orientation                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│     LSTM/Bi-LSTM (Temporal Sequence Classification)         │
│     - 2 layers, 128-256 units                               │
│     - Dropout for regularization                             │
│     - ~20ms inference                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Dense + Softmax (Sign Classification)             │
│           - Output: LSEC sign classes                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               TTS Engine (Voice Synthesis)                   │
│               - Ecuadorian Spanish expressions               │
│               - Local TTS with custom vocabulary             │
└─────────────────────────────────────────────────────────────┘
```

### Desglose de Latencia (Target <500ms)

| Etapa | Target | Estrategia de Optimización |
|-------|--------|---------------------------|
| Captura de frame | 33ms (30 FPS) | CameraX con ImageAnalysis |
| YOLOv8 inference | 10-15ms | YOLOv8-nano, TFLite GPU delegate |
| MediaPipe hands | 10-15ms | Pre-built Android SDK |
| LSTM inference | 20-30ms | Quantized INT8 model |
| Post-processing | 5ms | Efficient native code |
| TTS synthesis | 100-200ms | Cached audio, local TTS |
| **Total** | **~200-300ms** | **Buffer for variations** |

---

## Arquitecturas Alternativas a Considerar

1. **MediaPipe Holistic + Transformer**
   - Captura manos, cara y pose (contexto corporal completo)
   - Mejor para señas que requieren expresiones faciales
   - Mayor costo computacional

2. **MoveNet + Custom CNN**
   - Estimación de pose ligera
   - Bueno para reconocimiento de señas de cuerpo completo

3. **Graph Neural Networks (GCN)**
   - Modela datos de landmarks como grafos
   - Prometedor para reconocimiento de gestos de manos

4. **Vision Transformer (ViT) Ligero**
   - MobileViT o EfficientFormer
   - Mejor captura de relaciones espaciales

---

## Estrategia TTS con Ecuatorianismos

### Mapeo de Vocabulario

```
Seña → Español Estándar → Variante Ecuatoriana
"Hola" → "¡Qué más!" / "¡Qué fue!"
"Amigo" → "Pana" / "Llave"
"Dinero" → "Plata" / "Billete"
"Bien" → "Chévere" / "Bacán"
"Mucho" → "Full" / "Un montón"
```

### Implementación Híbrida

1. **Audio pregrabado** para ecuatorianismos comunes (cache local)
2. **Google TTS fallback** para español estándar
3. **Preferencia de usuario** para estilo de habla
4. **SSML customization** para pronunciación y énfasis

---

## Desafíos Clave y Soluciones

| Desafío | Solución |
|---------|----------|
| **Escasez de Dataset LSEC** | 1) Crear dataset con signantes LSEC nativos 2) Data augmentation 3) Transfer learning desde ASL/LSM |
| **Señas Dinámicas vs Estáticas** | LSTM para secuencias temporales, CNN para deletreo estático |
| **Variaciones de Iluminación** | Preprocesamiento robusto de MediaPipe, data augmentation |
| **Fondo Complejo** | Detección ROI con YOLOv8, segmentación por tono de piel |
| **Variaciones Regionales** | Diccionario modular, personalización por usuario |
| **Consumo de Batería** | Optimización de modelo, frame rate adaptativo, dark mode |
| **Target 85% Accuracy** | Métodos ensemble, umbrales de confianza, feedback de corrección |

---

## Further Considerations

1. **Dataset LSEC** - ¿Se cuenta con un dataset existente de señas ecuatorianas o se debe crear desde cero? Recomiendo contactar organizaciones como FENASEC (Federación Nacional de Sordos del Ecuador) para colaboración.

2. **Alcance inicial del vocabulario** - ¿Cuántas señas debe reconocer el prototipo? Recomiendo MVP de 50-100 señas (alfabeto + números + frases comunes) antes de escalar.

3. **Dispositivos objetivo** - ¿Cuál es el rango de dispositivos Android objetivo (API mínimo, RAM, procesador)? Esto impacta la complejidad del modelo y optimizaciones necesarias.

4. **Metodología de desarrollo** - ¿El proyecto académico requiere metodología específica (SCRUM, XP, RUP)? Definir sprints y entregables.

5. **Validación con usuarios** - Plan de pruebas con comunidad sorda ecuatoriana para validar usabilidad y precisión.
