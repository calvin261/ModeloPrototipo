# Documentacion del Proyecto LSEC Translator

## 1. Resumen

LSEC Translator es una aplicacion Android nativa orientada al reconocimiento de Lengua de Senas Ecuatoriana. El proyecto combina captura de camara en tiempo real, extraccion de landmarks de mano con MediaPipe, clasificacion de senas mediante modelos locales ONNX/TFLite, sintesis de voz y herramientas internas para capturar datos, entrenar modelos en un backend y descargar nuevas versiones.

El estado actual del repositorio corresponde a un prototipo funcional con varias piezas preparadas para evolucionar: la app movil, la base local del diccionario, el pipeline de vision, la integracion con un backend FastAPI esperado y scripts iniciales de entrenamiento/descarga de modelos.

## 2. Tecnologias Principales

| Area | Tecnologia |
| --- | --- |
| Plataforma | Android API 26+, Kotlin, JDK 17 |
| UI | Jetpack Compose, Material 3, Navigation Compose |
| Camara | CameraX |
| ML en dispositivo | MediaPipe Tasks Vision, TensorFlow Lite, ONNX Runtime Android |
| Persistencia local | Room, SQLite, DataStore Preferences |
| Inyeccion de dependencias | Hilt |
| Red | Retrofit, OkHttp, Moshi |
| Concurrencia | Kotlin Coroutines, Flow, StateFlow |
| Analitica | Firebase Analytics |
| Entrenamiento auxiliar | Python, TensorFlow/Keras, NumPy, scikit-learn, MediaPipe |

## 3. Estructura General

```text
.
|-- app/                         Modulo Android principal.
|-- docs/                        Documentacion tecnica y guias del dataset.
|-- gradle/                      Version catalog y wrapper de Gradle.
|-- ml/                          Scripts y dependencias de entrenamiento.
|-- build.gradle.kts             Configuracion raiz de Gradle.
|-- settings.gradle.kts          Incluye el modulo :app.
|-- download_models.py           Descarga el modelo MediaPipe hand_landmarker.task.
|-- README.md                    Descripcion general inicial del proyecto.
```

## 4. Modulo Android `app`

El modulo `app` contiene la aplicacion movil completa. Su `namespace` y `applicationId` son `com.lsec.translator`. Compila con SDK 34, usa minSdk 26 y genera version `1.0.0`.

La configuracion esta en `app/build.gradle.kts`:

- Activa Compose, BuildConfig y ML Model Binding.
- Define URLs base distintas para `debug` y `release`.
- Incluye soporte para ABI `armeabi-v7a`, `arm64-v8a`, `x86` y `x86_64`.
- Configura firma de release leyendo `keystore.properties`.
- Usa ProGuard y shrink resources en release.

## 5. Entrada de la Aplicacion

### `LSECApplication.kt`

Clase `Application` anotada con `@HiltAndroidApp`. Es el punto de arranque de Hilt y permite inyectar dependencias en ViewModels, repositorios, componentes ML y servicios.

### `MainActivity.kt`

Actividad principal de la app. Carga la interfaz Compose y el host de navegacion. La actividad se declara como launcher en `AndroidManifest.xml` y fuerza orientacion vertical.

### `AndroidManifest.xml`

Declara:

- Permiso de camara: `android.permission.CAMERA`.
- Permiso de internet para comunicacion con backend y posibles servicios asociados.
- Requisito de hardware de camara.
- `networkSecurityConfig` y `usesCleartextTraffic=true`, necesario para usar servidores locales HTTP durante desarrollo.

## 6. Navegacion y Pantallas

La navegacion se centraliza en `ui/navigation/LSECNavHost.kt`. Define estas rutas:

| Ruta | Pantalla | Funcion |
| --- | --- | --- |
| `home` | `HomeScreen` | Menu principal de la app. |
| `camera` | `CameraScreen` | Traduccion en tiempo real con camara. |
| `dictionary` | `DictionaryScreen` | Diccionario local de senas. |
| `settings` | `SettingsScreen` | Preferencias de usuario, servidor y API key. |
| `data_capture` | `DataCaptureScreen` | Captura de landmarks para dataset. |
| `training` | `TrainingScreen` | Entrenamiento remoto y gestion de modelos. |

### Home

`HomeScreen` presenta accesos a traduccion, diccionario, configuracion, captura de datos y entrenamiento. Es la primera pantalla del flujo.

### Camara

`CameraScreen` y `CameraViewModel` coordinan el flujo principal:

1. Recibir frames de CameraX.
2. Enviar cada frame a `DetectHandsUseCase`.
3. Acumular landmarks en un buffer temporal.
4. Clasificar con `ClassifySignUseCase`.
5. Actualizar UI con sena, variante ecuatoriana, confianza y latencia.
6. Reproducir voz con `SynthesizeSpeechUseCase` cuando la confianza supera el umbral.

El ViewModel usa un cooldown para no repetir continuamente la misma traduccion y un `Mutex` para evitar que varias llamadas TTS se acumulen.

### Diccionario

`DictionaryScreen` y `DictionaryViewModel` muestran senas almacenadas en Room. Incluyen busqueda, filtros por categoria, favoritos, senas frecuentes y reproduccion TTS de la traduccion.

El ViewModel tambien carga datos iniciales si el diccionario esta vacio.

### Configuracion

`SettingsScreen` y `SettingsViewModel` permiten modificar preferencias guardadas en DataStore:

- Activar/desactivar TTS.
- Velocidad y tono de voz.
- Preferir variantes ecuatorianas.
- Camara frontal/trasera.
- Confianza minima.
- Modo ahorro de bateria.
- Modo oscuro.
- URL del servidor.
- API key.
- Actualizacion de diccionario y limpieza de cache.

### Captura de Datos

`DataCaptureScreen` y `DataCaptureViewModel` permiten recolectar landmarks desde la camara:

- Modo estatico: captura un frame de 63 valores para una etiqueta.
- Modo secuencia: graba multiples frames para senas dinamicas.
- Mantiene conteos locales por etiqueta.
- Sube muestras capturadas al backend mediante `UploadLandmarksUseCase`.

Las etiquetas estaticas por defecto son `A-Z` y `SPACE`. Las dinamicas por defecto incluyen `hola`, `gracias`, `por_favor`, `si`, `no`, `ayuda`, `agua`, `comida`, `bano`, `nombre`, `como_estas`, `bien`, `mal`, `perdon` y `adios`.

### Entrenamiento

`TrainingScreen` y `TrainingViewModel` gestionan entrenamiento remoto:

- Verifican conexion consultando estadisticas del dataset.
- Inician entrenamiento de clasificador estatico o secuencial.
- Hacen polling del estado cada 3 segundos.
- Consultan historial de entrenamiento.
- Descargan modelos actualizados.
- Eliminan etiquetas del dataset remoto.
- Piden al servidor exportar un modelo si la descarga inicial falla.

## 7. Capas de Arquitectura

El proyecto sigue una separacion sencilla por capas:

```text
UI Compose
  -> ViewModels
    -> Use cases
      -> Repositorios
        -> Room / DataStore / Retrofit / ML / TTS
```

### Capa UI

Ubicada en `app/src/main/java/com/lsec/translator/ui`. Contiene pantallas Compose, navegacion y tema visual.

### Capa de Dominio

Ubicada en `domain/usecase`. Expone acciones concretas para la UI:

- `DetectHandsUseCase`: extrae landmarks desde un frame.
- `ClassifySignUseCase`: clasifica una secuencia de landmarks.
- `SynthesizeSpeechUseCase`: convierte texto a voz.
- `UploadLandmarksUseCase`: sube landmarks al backend.
- `StartTrainingUseCase`: inicia entrenamiento remoto.
- `CheckTrainingStatusUseCase`: consulta progreso del entrenamiento.
- `DownloadModelUseCase`: descarga modelos desde el servidor.

### Capa de Datos

Ubicada en `data`. Incluye persistencia local, preferencias, red y repositorios.

### Capa ML

Ubicada en `ml`. Contiene deteccion de manos, normalizacion y clasificadores.

### Capa TTS

Ubicada en `tts`. Maneja Text-to-Speech y audio personalizado.

## 8. Inyeccion de Dependencias

Los modulos Hilt estan en `di`:

| Archivo | Responsabilidad |
| --- | --- |
| `DatabaseModule.kt` | Provee `LSECDatabase` y `SignDao`. |
| `MLModule.kt` | Provee `HandDetector`, `SignClassifier` y `TTSEngine`. |
| `NetworkModule.kt` | Provee Moshi, OkHttp, Retrofit y `LSECApiService`. |

`NetworkModule` lee la URL y API key desde `UserPreferences`. La API key se inyecta como header `X-API-Key` cuando existe.

## 9. Persistencia Local

### Room

La base local se llama `lsec_database` y esta definida en `LSECDatabase.kt`. Actualmente contiene la entidad `SignEntity`.

`SignEntity` representa una sena del diccionario con:

- Nombre.
- Traduccion al espanol.
- Variante ecuatoriana.
- Descripcion.
- Categoria.
- Dificultad.
- Video o imagen opcional.
- Tipo de sena dinamica/estatica.
- Variante regional.
- Conteo de uso.
- Favorito.
- Fechas de creacion y actualizacion.

`SignDao` permite listar, filtrar, buscar, insertar, actualizar, borrar, marcar favoritos e incrementar uso.

### DataStore

`UserPreferences.kt` guarda configuraciones del usuario:

- TTS, velocidad, tono y variantes ecuatorianas.
- Camara y umbral de confianza.
- Ahorro de bateria y tema oscuro.
- URL del backend y API key.
- Versiones de modelos.
- Actualizacion automatica.
- Modo de reconocimiento (`static` o `dynamic`).

## 10. Pipeline de Reconocimiento

El pipeline actual es:

```text
CameraX frame
  -> HandDetector
  -> MediaPipe HandLandmarker
  -> LandmarkNormalizer
  -> SignClassifier
  -> StaticSignClassifier o DynamicSignClassifier
  -> resultado con confianza
  -> TTSEngine
```

### HandDetector

`HandDetector.kt` usa MediaPipe `HandLandmarker` con el asset `hand_landmarker.task`. Detecta una mano y extrae 21 landmarks 3D. Cada frame termina como `FloatArray(63)`.

Aunque documentacion anterior menciona YOLO, el codigo actual ya no lo usa en el detector Android. MediaPipe realiza deteccion y landmarks en una sola etapa.

### LandmarkNormalizer

`LandmarkNormalizer.kt` normaliza landmarks para hacerlos mas invariantes:

1. Traslada la muneca, landmark 0, al origen.
2. Calcula la distancia entre muneca y landmark 9.
3. Escala todos los puntos usando esa distancia.
4. Devuelve los 63 valores aplanados.

### SignClassifier

`SignClassifier.kt` es una fachada que carga:

- Labels de `filesDir/models/*.json` si fueron descargados.
- Labels desde assets si existen.
- Labels por defecto si no hay archivos externos.

Luego delega a:

- `StaticSignClassifier` para senas estaticas.
- `DynamicSignClassifier` para senas dinamicas.

Tambien enriquece el resultado con variantes ecuatorianas para ciertas etiquetas.

### StaticSignClassifier

Clasifica un solo frame de landmarks normalizados. Busca modelo en este orden:

1. `filesDir/models/sign_mlp.onnx`.
2. `filesDir/models/sign_mlp.tflite`.
3. `assets/sign_mlp.tflite`.

Usa ONNX Runtime si hay modelo ONNX; si no, intenta TensorFlow Lite.

### DynamicSignClassifier

Clasifica secuencias. Mantiene un buffer de hasta 60 frames y clasifica cuando:

- El buffer llega al maximo.
- La mano desaparece durante varios frames y ya hay suficientes muestras.

Busca modelo en este orden:

1. `filesDir/models/sign_gru.onnx`.
2. `filesDir/models/sign_gru.tflite`.
3. `assets/sign_gru.tflite`.

## 11. Text-to-Speech

`TTSEngine.kt` usa `TextToSpeech` con locale `es-EC` cuando esta disponible. Tambien tiene un mapa de audios personalizados para expresiones ecuatorianas.

Flujo:

1. Se recibe texto.
2. Se revisa si existe audio personalizado.
3. Si existe, se reproduce con `MediaPlayer`.
4. Si no existe, se sintetiza con Android TTS.

La pantalla de camara evita reproducir resultados con baja confianza y aplica cooldown para que la misma frase no se repita cada frame.

## 12. Backend Esperado

El cliente Android espera un backend con endpoints bajo `api/v1`. La interfaz esta en `LSECApiService.kt`.

### Datos

- `POST api/v1/data/upload/static`: sube landmarks estaticos.
- `POST api/v1/data/upload/sequence`: sube secuencias por multipart.
- `GET api/v1/data/stats`: obtiene estadisticas del dataset.
- `DELETE api/v1/data/static/{label}`: borra muestras estaticas por etiqueta.
- `DELETE api/v1/data/sequence/{label}`: borra secuencias por etiqueta.

### Entrenamiento

- `POST api/v1/train/classifier`: entrena clasificador estatico.
- `POST api/v1/train/sequence`: entrena modelo de secuencias.
- `GET api/v1/train/status/{taskId}`: consulta estado.
- `GET api/v1/train/history`: consulta historial.

### Modelos

- `GET api/v1/models/list`: lista modelos.
- `GET api/v1/models/latest/{modelType}`: obtiene version mas reciente.
- `GET api/v1/models/download/{modelType}/{version}`: descarga modelo.
- `GET api/v1/models/labels/{modelType}`: descarga labels.
- `POST api/v1/models/export/{modelType}`: exporta modelo entrenado.

`modelType` esperado: `classifier` o `sequence`.

## 13. Gestion de Modelos

`ModelDownloadManager.kt` descarga modelos y labels al directorio interno:

```text
filesDir/models/
|-- sign_mlp.onnx o sign_mlp.tflite
|-- sign_gru.onnx o sign_gru.tflite
|-- classifier_labels.json
|-- sequence_labels.json
```

Los clasificadores leen estos archivos en el arranque o despues de llamar a recarga. Si no hay modelos descargados ni assets equivalentes, el clasificador devuelve `null` y la app mostrara resultado desconocido o baja confianza.

## 14. Scripts de Machine Learning

### `download_models.py`

Descarga `hand_landmarker.task` desde los modelos oficiales de MediaPipe y lo coloca en:

```text
app/src/main/assets/hand_landmarker.task
```

Este asset si esta presente en el repositorio actual.

### `ml/train_lstm_model.py`

Script inicial para entrenar un clasificador Bi-LSTM con secuencias de landmarks:

- Entrada: 30 frames por secuencia, 63 features por frame.
- Modelo: BatchNorm, dos capas Bi-LSTM, Dropout, Dense y Softmax.
- Salida: modelo Keras, modelo TFLite cuantizado y labels JSON.

Importante: el script genera datos sinteticos para pruebas. Para produccion debe conectarse a un dataset real de LSEC.

### `ml/requirements.txt`

Lista dependencias para entrenamiento: TensorFlow, NumPy, scikit-learn, OpenCV, MediaPipe, pandas, matplotlib y tqdm.

## 15. Assets y Recursos

### Assets ML

Actualmente existe:

```text
app/src/main/assets/hand_landmarker.task
```

Los clasificadores tambien buscan assets opcionales:

```text
app/src/main/assets/sign_mlp.tflite
app/src/main/assets/sign_gru.tflite
app/src/main/assets/classifier_labels.json
app/src/main/assets/sequence_labels.json
```

Si no estan incluidos, la app depende de modelos descargados desde el backend.

### Recursos Android

`res/values` contiene colores, strings y temas. `values-night` contiene tema nocturno. `res/xml` contiene reglas de backup, extraccion de datos y configuracion de seguridad de red.

## 16. Configuracion de Build

Comandos habituales:

```bash
./gradlew assembleDebug
./gradlew installDebug
./gradlew assembleRelease
```

En Windows:

```powershell
.\gradlew.bat assembleDebug
```

Para preparar MediaPipe:

```bash
python download_models.py
```

Para entrenar el script Python experimental:

```bash
cd ml
pip install -r requirements.txt
python train_lstm_model.py
```

## 17. Documentacion Existente

| Archivo | Contenido |
| --- | --- |
| `README.md` | Vision general, instalacion, arquitectura inicial y roadmap. |
| `docs/ARCHITECTURE.md` | Arquitectura tecnica propuesta y pipeline de procesamiento. |
| `docs/DATASET_GUIDE.md` | Guia de recoleccion, anotacion y preparacion de dataset LSEC. |
| `docs/PROJECT_DOCUMENTATION.md` | Esta documentacion integral basada en el codigo actual. |

Nota: algunos documentos existentes muestran caracteres danados por codificacion. Tambien mencionan piezas historicas como YOLO, pero el codigo Android actual usa MediaPipe HandLandmarker directamente.

## 18. Puntos Pendientes o Riesgos

- No se observan pruebas unitarias o instrumentadas especificas para el pipeline ML, repositorios o ViewModels.
- El README y otros docs tienen problemas de codificacion de caracteres.
- La documentacion antigua menciona YOLOv8, pero el detector actual no lo usa.
- `NetworkModule` crea Retrofit con la URL guardada al iniciar; si el usuario cambia la URL, puede requerir reiniciar la app para recrear Retrofit.
- El script `train_lstm_model.py` usa datos sinteticos, no un dataset real.
- Los modelos `sign_mlp` y `sign_gru` no aparecen como assets en el listado actual, asi que la clasificacion real depende de descarga o de agregar esos archivos.
- `fallbackToDestructiveMigration()` en Room borra datos locales ante cambios de esquema.
- Hay archivos sensibles de firma (`keystore.properties`, `.jks`) presentes en el workspace; conviene revisar si deben estar versionados o protegidos.

## 19. Flujo de Desarrollo Recomendado

1. Configurar Android Studio con JDK 17 y SDK 34.
2. Ejecutar `python download_models.py` si falta `hand_landmarker.task`.
3. Compilar `assembleDebug`.
4. Configurar la URL del backend desde la pantalla de configuracion.
5. Capturar datos desde la pantalla de captura.
6. Subir muestras al backend.
7. Entrenar clasificador o secuencia desde la pantalla de entrenamiento.
8. Descargar modelos entrenados.
9. Probar reconocimiento en la pantalla de camara.
10. Consolidar labels/modelos como assets si se necesita funcionamiento offline.

## 20. Glosario Interno

| Termino | Significado |
| --- | --- |
| Landmark | Punto 3D detectado en la mano. MediaPipe entrega 21 por mano. |
| `FloatArray(63)` | 21 landmarks por 3 coordenadas `x`, `y`, `z`. |
| Sena estatica | Sena reconocible con un solo frame o postura de mano. |
| Sena dinamica | Sena que requiere movimiento y varios frames. |
| `classifier` | Modelo estatico, normalmente `sign_mlp`. |
| `sequence` | Modelo temporal, normalmente `sign_gru`. |
| Variante ecuatoriana | Texto de salida adaptado a expresiones locales. |
| TTS | Text-to-Speech, sintesis de voz. |
