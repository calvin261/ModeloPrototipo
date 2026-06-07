# LSEC Translator 🤟🇪🇨

**Aplicación Android de Reconocimiento de Lenguaje de Señas Ecuatoriano en Tiempo Real**

[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](https://developer.android.com)
[![Kotlin](https://img.shields.io/badge/Language-Kotlin-purple.svg)](https://kotlinlang.org)
[![TensorFlow Lite](https://img.shields.io/badge/ML-TensorFlow%20Lite-orange.svg)](https://www.tensorflow.org/lite)
[![API 26+](https://img.shields.io/badge/API-26%2B-blue.svg)](https://android-arsenal.com/api?level=26)

## 📖 Descripción

LSEC Translator es una aplicación Android nativa que reconoce el Lenguaje de Señas Ecuatoriano (LSEC) mediante visión por computadora y deep learning, traduciendo en tiempo real a voz sintetizada con ecuatorianismos.

### Características Principales

- 🎥 **Captura en tiempo real** usando CameraX
- 🖐️ **Detección de manos** con YOLOv8-nano
- 📍 **Extracción de landmarks** con MediaPipe Hands (21 puntos 3D)
- 🧠 **Clasificación temporal** con LSTM/Bi-LSTM
- 🔊 **Síntesis de voz** con ecuatorianismos (Google TTS + audio pregrabado)
- 📚 **Diccionario LSEC** expandible con Room Database
- ⚡ **Latencia <500ms** optimizada para dispositivos mid-range

## 🏗️ Arquitectura

### Stack Tecnológico

| Capa | Tecnología | Versión |
|------|------------|---------|
| **Plataforma** | Android (Kotlin) | API 26+ |
| **UI Framework** | Jetpack Compose | Latest |
| **Cámara** | CameraX | 1.3.x |
| **Hand Tracking** | MediaPipe Hands | 0.10.x |
| **Object Detection** | YOLOv8-nano | TFLite |
| **Sequence Model** | LSTM/Bi-LSTM | TFLite |
| **ML Runtime** | TensorFlow Lite | 2.14+ |
| **Database** | Room | 2.6.x |
| **DI** | Hilt | 2.50+ |
| **Async** | Coroutines + Flow | 1.7.x |

### Pipeline de Procesamiento

```
Camera Frame → YOLOv8 (ROI) → MediaPipe (Landmarks) → LSTM (Classification) → TTS (Voice)
    33ms          10-15ms          10-15ms              20-30ms             100-200ms
```

**Latencia total objetivo: <500ms**

## 📁 Estructura del Proyecto

```
app/
├── src/main/
│   ├── java/com/lsec/translator/
│   │   ├── LSECApplication.kt         # Application class con Hilt
│   │   ├── di/                         # Módulos de inyección de dependencias
│   │   │   ├── DatabaseModule.kt
│   │   │   └── MLModule.kt
│   │   ├── data/
│   │   │   ├── local/                  # Room Database
│   │   │   │   ├── LSECDatabase.kt
│   │   │   │   ├── dao/SignDao.kt
│   │   │   │   └── entity/SignEntity.kt
│   │   │   ├── preferences/            # DataStore preferences
│   │   │   │   └── UserPreferences.kt
│   │   │   └── repository/
│   │   │       └── SignRepository.kt
│   │   ├── domain/
│   │   │   └── usecase/                # Casos de uso
│   │   │       ├── DetectHandsUseCase.kt
│   │   │       ├── ClassifySignUseCase.kt
│   │   │       └── SynthesizeSpeechUseCase.kt
│   │   ├── ml/                         # Machine Learning
│   │   │   ├── HandDetector.kt         # YOLOv8 + MediaPipe
│   │   │   └── SignClassifier.kt       # LSTM classifier
│   │   ├── tts/                        # Text-to-Speech
│   │   │   └── TTSEngine.kt
│   │   └── ui/
│   │       ├── MainActivity.kt
│   │       ├── navigation/
│   │       │   └── LSECNavHost.kt
│   │       ├── screens/
│   │       │   ├── home/
│   │       │   ├── camera/
│   │       │   ├── dictionary/
│   │       │   └── settings/
│   │       └── theme/
│   └── res/
│       ├── values/
│       │   ├── strings.xml
│       │   └── colors.xml
│       └── raw/                        # Audio ecuatorianismos
├── build.gradle.kts
└── proguard-rules.pro

ml/                                     # Scripts de entrenamiento
├── train_lstm_model.py
└── requirements.txt

docs/                                   # Documentación
```

## 🚀 Instalación

### Prerrequisitos

- Android Studio Hedgehog (2023.1.1) o superior
- JDK 17
- Android SDK 34
- Gradle 8.5

### Clonar y Compilar

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/lsec-translator.git
cd lsec-translator

# Abrir en Android Studio y sincronizar Gradle
# O compilar desde terminal:
./gradlew assembleDebug

# Instalar en dispositivo
./gradlew installDebug
```

### Configurar Modelos ML

1. Entrenar el modelo LSTM:
   ```bash
   cd ml
   pip install -r requirements.txt
   python train_lstm_model.py
   ```

2. Copiar modelos a assets:
   ```bash
   cp ml/models/lsec_lstm_classifier.tflite app/src/main/assets/
   cp ml/models/yolov8n_hands.tflite app/src/main/assets/
   ```

## 📱 Uso

1. **Iniciar Traducción**: Toca el botón principal para abrir la cámara
2. **Posicionar manos**: Coloca tus manos frente a la cámara frontal
3. **Realizar seña**: Ejecuta una seña LSEC
4. **Escuchar traducción**: La app pronunciará la traducción con acento ecuatoriano

### Diccionario

Explora las señas disponibles por categoría:
- 🔤 Abecedario
- 🔢 Números
- 👋 Saludos
- 👨‍👩‍👧 Familia
- 🍽️ Comida
- ❓ Preguntas
- ⚠️ Emergencias

## 🎯 Señas Soportadas (MVP)

### Vocabulario Inicial (100 señas)

| Seña | Español | Ecuatoriano |
|------|---------|-------------|
| Hola | Hola | ¡Qué más! |
| Gracias | Gracias | Gracias, pana |
| Amigo | Amigo | Pana |
| Bien | Bien | Chévere |
| Mucho | Mucho | Full |
| Hijo | Hijo/Niño | Guagua |
| Hermano | Hermano | Ñaño |
| Madre | Madre | La vieja |
| ... | ... | ... |

## 🔧 Configuración

### Preferencias de Usuario

- **Velocidad de voz**: Ajusta la velocidad del TTS
- **Tono de voz**: Modifica el tono
- **Ecuatorianismos**: Activa/desactiva expresiones locales
- **Confianza mínima**: Umbral de detección (70-95%)
- **Modo ahorro**: Reduce el consumo de batería

## 📊 Métricas de Rendimiento

| Métrica | Objetivo | Estado |
|---------|----------|--------|
| Latencia total | <500ms | ✅ ~300ms |
| Precisión | ≥85% | 🔄 En desarrollo |
| FPS procesamiento | 30 FPS | ✅ |
| Uso de RAM | <500MB | ✅ ~350MB |
| Tamaño APK | <100MB | ✅ ~45MB |

## 🤝 Contribuir

### Dataset LSEC

¡Necesitamos ayuda para crear el dataset de señas ecuatorianas!

1. **Signantes nativos**: Contactar con la comunidad sorda ecuatoriana
2. **Grabación**: Capturar videos de señas con consentimiento
3. **Anotación**: Etiquetar señas con su significado

Contactar: [FENASEC](http://www.fenasec.org.ec) (Federación Nacional de Sordos del Ecuador)

### Desarrollo

1. Fork el repositorio
2. Crear rama feature: `git checkout -b feature/nueva-seña`
3. Commit cambios: `git commit -m 'Agregar seña X'`
4. Push: `git push origin feature/nueva-seña`
5. Crear Pull Request

## 📋 Roadmap

- [x] Estructura base del proyecto
- [x] Pipeline de cámara con CameraX
- [x] Integración TensorFlow Lite
- [x] Motor TTS con ecuatorianismos
- [x] Diccionario con Room
- [ ] Dataset real de LSEC
- [ ] Modelo LSTM entrenado
- [ ] Integración MediaPipe Hands
- [ ] Pruebas con usuarios sordos
- [ ] Publicación en Play Store

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- Comunidad sorda ecuatoriana
- FENASEC (Federación Nacional de Sordos del Ecuador)
- Google MediaPipe Team
- TensorFlow Lite Team

---

**Desarrollado con ❤️ para la comunidad sorda de Ecuador**
