# Ficha técnica LSEC Translator

> Auditoría técnica del proyecto `ModeloPrototipo` ejecutada el 2026-06-17
> contra el commit `506c726` (branch `codex/backend-last`).
> Responde al plan `PLAN_DESCUBRIMIENTO_LSEC.md` aplicando todas sus fases.
> Cada afirmación se cita con `archivo:línea`.

---

## 1. Identidad

| Campo | Valor |
|---|---|
| Tipo de proyecto | Monorepo: **Backend Python/FastAPI** + **App Android nativa Kotlin/Compose** |
| Repositorio | `https://github.com/calvin261/ModeloPrototipo.git` |
| Versión | App `versionName=1.0.0`, `versionCode=1` (`frontend/app/build.gradle.kts:25-26`); API `1.0.0` (`backend/api/main.py:96-98`); último commit `506c726` |
| Licencia | **NO PRESENTE** — no existe archivo `LICENSE` en el repo (búsqueda exhaustiva) |
| Autor git | Jefferson Jacome `<jeffesjacome@gmail.com>` |

---

## 2. Stack tecnológico

### 2.1 Lenguajes y frameworks

| Capa | Tecnología | Versión | Fuente |
|---|---|---|---|
| Backend | Python | `>=3.10` | `backend/pyproject.toml:6` |
| Backend HTTP | FastAPI + Uvicorn | `>=0.109.0` / `>=0.27.0` | `backend/api/requirements.txt:1-2` |
| Backend training | PyTorch (CPU-only) | `>=2.0.0` | `backend/api/requirements.txt:5` |
| Android | Kotlin | `2.0.0` | `gradle/libs.versions.toml:4` |
| Android UI | Jetpack Compose + Material 3 | BOM `2024.02.00` | `libs.versions.toml:10,114-115` |
| Android navigation | navigation-compose | `2.7.7` | `libs.versions.toml:11` |
| Build Android | Gradle + AGP | `8.13` / `8.13.2` | `gradle/wrapper/gradle-wrapper.properties:3`, `libs.versions.toml:3` |
| JVM target | Java | `17` | `frontend/app/build.gradle.kts:74-79` |

### 2.2 ML / Data Science

| Ítem | Valor | Fuente |
|---|---|---|
| Framework inferencia móvil primario | **ONNX Runtime Android** | `1.17.3` (`frontend/app/build.gradle.kts:137`) |
| Framework inferencia móvil fallback | TFLite | `2.14.0` + GPU `0.4.4` | `libs.versions.toml:131-134` |
| Detector landmarks | MediaPipe Hands Tasks Vision | `0.10.9` | `libs.versions.toml:128` |
| Framework backend (activo) | PyTorch CPU-only | `torch>=2.0.0` | `backend/api/requirements.txt:5` |
| Framework backend (legado, no usado) | TensorFlow + onnx-tf | `>=2.14.0` / `>=1.10.0` | `backend/api/requirements.txt:11-12` |
| Tipos de modelos | MLP (24 clases), Bi-GRU+att (12 clases), SignTransformer SPOTER-style (12 clases, 564K params) | `hand_model/{classifier,sequence_model,spoter_model}.py` |
| Tamaño en repo | MLP 217 KB · GRU 1.77 MB · Transformer 2.17 MB · ONNX FP32 ~397 KB · MediaPipe .task 7.46 MB | filesystem |
| Cuantización | INT8 dinámico via PyTorch `quantize_dynamic` | `backend/scripts/export_transformer_onnx.py` |
| Formatos | `.pth`, `.onnx` (+`.onnx.data` para pesos), `.task`, `.npy` | tree |

### 2.3 Persistencia

| Capa | Tecnología | Detalle |
|---|---|---|
| Android DB estructurada | **Room `2.6.1`** | DB `lsec_database`, 1 entidad `SignEntity` (15 cols) + DAO con Flow (`libs.versions.toml:140-142`) |
| Android key-value | **DataStore Preferences `1.0.0`** | `user_settings`: 17 prefs (serverUrl, apiKey, llmEnabled, llmAutoTrigger, …) |
| Android SharedPreferences raw | `adaptive_thresholds` | thresholds serializados como string `key=value;…` |
| Android filesystem | `filesDir/models/`, `filesDir/calibration/<userId>.bin` (binario custom), `getExternalFilesDir(DOCUMENTS)/<tester>_round<N>.csv` | `ModelDownloadManager.kt:29`, `CentroidClassifier.kt:117`, `SignTestLogger.kt:67` |
| Backend | filesystem plano (CSV + .npy + JSON manifest/splits) | `backend/data/` |
| Remoto | REST FastAPI `/api/v1/*` con X-API-Key opcional | `backend/api/routers/*` |

### 2.4 Dependencias top

**Backend** (`backend/pyproject.toml:7-19` + `backend/api/requirements.txt:1-24`):

`torch`, `torchvision`, `mediapipe`, `ultralytics`, `fastapi`, `uvicorn[standard]`, `python-multipart`, `onnx`, `onnxruntime`, `onnx-tf`, `tensorflow`, `opencv-python-headless`, `pyyaml`, `numpy`, `pandas`, `scikit-learn`, `pyttsx3`, `seaborn`.

**Android** (`gradle/libs.versions.toml`):

Compose BOM `2024.02.00`, Navigation Compose `2.7.7`, Lifecycle `2.7.0`, Hilt `2.50` (+KSP), Room `2.6.1`, CameraX `1.3.1`, MediaPipe Tasks Vision `0.10.9`, TFLite `2.14.0`, **ONNX Runtime Android `1.17.3`**, Retrofit `2.9.0` + OkHttp `4.12.0` + Moshi `1.15.0`, Coroutines `1.7.3`, DataStore `1.0.0`, Accompanist Permissions `0.34.0`, Firebase BOM `32.7.2` (solo Analytics).

**Ausencias**: no Coil/Glide (no hay galería), no Google Play Services Auth/MLKit/Maps/Crashlytics, no appcompat ni Material 2.

---

## 3. Arquitectura

### 3.1 Topología de carpetas (resumen, nivel 2-3)

```
backend/
├── api/                  FastAPI (main.py, routers/, services/, schemas/, tasks/, config.py)
├── hand_model/           Paquete ML (classifier, sequence_model, spoter_model, landmarks, augmentations, config)
├── scripts/              33 CLIs (prepare_*, train_*, evaluate*, export*, build_*_splits, full_pipeline.bat)
├── tests/                5 archivos pytest (34 tests)
├── data/                 Generados (CSV landmarks, .npy sequences, .avi LSC50, .jpg Kaggle, splits/)
├── models/               Checkpoints .pth, ONNX exports, registry.json
├── datasets/             hands.yaml, labels_map.json, lsec_alphabet_labels.json, lsec_vocabulary.json
├── reports/              Métricas F0/F3 + orchestrator_log.md
└── pyproject.toml, Dockerfile, docker-compose.yml, deploy.sh, cloudflare-tunnel.sh

frontend/
└── app/src/main/java/com/lsec/translator/
    ├── ui/               MainActivity, navigation/LSECNavHost, screens/{home,camera,calibration,dictionary,settings,capture,training}
    ├── data/             local/{LSECDatabase, dao, entity}, preferences/UserPreferences, remote/{LSECApiService, ModelDownloadManager}, repository/
    ├── domain/usecase/   ClassifySign, DetectHands, DownloadModel, StartTraining, CheckTrainingStatus, UploadLandmarks, SynthesizeSpeech
    ├── di/               DatabaseModule, NetworkModule, MLModule (vacío)
    ├── ml/               HandDetector, LandmarkNormalizer, SignClassifier (façade), StaticSignClassifier, DynamicSignClassifier (GRU), SignTransformerClassifier, CentroidClassifier, AdaptiveThresholds
    ├── llm/              LlmPostProcessor, Phi3LlamaCppBridge (stub JNI)
    ├── tts/              TTSEngine
    └── test/             SignTestLogger (F7 harness runtime, NO tests JUnit)

docs/                     README.md, AGENTS.md, CHANGELOG.md (839 líneas), STATUS_F0_F3.md,
                          ARCHITECTURE.md, DATASET_GUIDE.md, PROJECT_DOCUMENTATION.md,
                          F7_TEST_PROTOCOL.md, F7_USER_TEST_REPORT.template.md,
                          INSTRUCCIONES/{README,FUENTES,GUIA_PASO_A_PASO}.md,
                          INTEGRACION_FRONTEND_BACKEND.md
```

### 3.2 Punto de entrada y DI

- **Backend**: `backend/api/main.py:96-102` sirve `/` y `/health`. Routers bajo `/api/v1` (`backend/api/config.py:11`). Singletons manuales: `ModelRegistry`, `DataService` (`main.py:40-58`). Lifespan pattern.
- **Android**: `MainActivity.kt:19` (`@AndroidEntryPoint ComponentActivity`) arranca `LSECNavHost` desde `home`. Hilt root en `LSECApplication.kt:10`. Módulos en `di/{DatabaseModule,NetworkModule,MLModule}.kt`. Componentes ML por constructor (`@Singleton + @Inject constructor`).
- **Multi-agente**: `opencode.json` registra 5 sub-agentes `lsc-*` + skill `lsc-orchestrator` para coordinar el trabajo.

### 3.3 Patrón arquitectónico

- **Backend**: capas (router → service → model), hexagonal-light. Path injection manual de `backend/` en `sys.path` desde cada CLI.
- **Android**: Clean Architecture lite (`ui/` → `domain/usecase/` → `data/` → `ml/`) con MVVM (`@HiltViewModel` por pantalla) + unidirectional data flow (StateFlow).

### 3.4 Componentes principales y mapa de llamadas

| # | Componente | Responsabilidad | Quién llama a quién |
|---|---|---|---|
| 1 | MediaPipe HandLandmarker (asset 7.46 MB) | Detectar 21 landmarks × 3D | `HandDetector.kt:84-87` |
| 2 | `LandmarkNormalizer` | wrist-relative + scale-invariant + (opcional) rotation+handedness invariant | `HandDetector.kt:105`, `landmarks.py:9-66` |
| 3 | `SignClassifier` façade | Orquesta MLP / SignTransformer / GRU / CentroidClassifier | `CameraViewModel` → `ClassifySignUseCase` |
| 4 | `SignTransformerClassifier` (ONNX Runtime) | Inferencia secuencia → 12 logits + (opcional) embedding 128-dim | `SignClassifier.kt:115-172` |
| 5 | `CentroidClassifier` | Re-decisión con cosine similarity sobre embeddings (per-user) | `SignClassifier.kt:163-192` |
| 6 | `AdaptiveThresholds` | Auto-tune umbrales del state machine | `SignTransformerClassifier`, `DynamicSignClassifier` |
| 7 | FastAPI app (uvicorn :8000) | Sirve `/api/v1/{data,train,models}/*` | App Android (Retrofit) |
| 8 | `LSECApiService` (Retrofit + OkHttp + Moshi) | Cliente HTTP desde móvil | Repositorios `data/repository/*` |
| 9 | `SignRepository` / `ModelRepository` / `TrainingRepository` | Passthrough + descarga modelos | ViewModels |
| 10 | `SignTransformer` (PyTorch) | 564K params, 4 layers, embed=128, CLS-pool | `train_transformer.py`, `export_transformer_onnx.py` |

### 3.5 API surface (`backend/api/`)

- **App-level** (`backend/api/main.py:96-102`):
  - `GET /` → público
  - `GET /health` → público
  - `GET /docs` (Swagger, FastAPI default) → público
- **Data router** (`backend/api/routers/data_router.py`, prefix `/api/v1/data`, requiere `X-API-Key`):
  - `POST /api/v1/data/upload/static`
  - `POST /api/v1/data/upload/sequence` (multipart)
  - `GET /api/v1/data/stats`
  - `DELETE /api/v1/data/static/{label}`
  - `DELETE /api/v1/data/sequence/{label}`
- **Models router** (`backend/api/routers/models_router.py`, prefix `/api/v1/models`, requiere `X-API-Key`). Tipos válidos: `classifier | sequence | transformer`:
  - `GET /api/v1/models/list`
  - `GET /api/v1/models/latest/{model_type}`
  - `GET /api/v1/models/download/{model_type}/{version}` (binary FileResponse)
  - `POST /api/v1/models/export/{model_type}` (lanza export en background; transformer debe lanzarse via `scripts/export_transformer_onnx.py`)
  - `GET /api/v1/models/labels/{model_type}`
- **Training router** (`backend/api/routers/training_router.py`, prefix `/api/v1/train`, requiere `X-API-Key`):
  - `POST /api/v1/train/classifier`
  - `POST /api/v1/train/sequence`
  - `GET /api/v1/train/status/{task_id}`
  - `GET /api/v1/train/history`
- **NO existe `/api/v1/train/transformer`** — debe invocarse via `python -m scripts.train_transformer`.

### 3.6 Configuración (dataclass, sin YAML)

**`backend/api/config.py:7-27` (`ServerConfig`)**:
- `host="0.0.0.0"`, `port=8000`, `api_prefix="/api/v1"`
- Paths: `data_landmarks_dir`, `data_sequences_dir`, `models_classifier_dir`, `models_sequence_dir`, `models_exports_dir`, `datasets_dir`, `classifier_labels_path`, `vocabulary_path`
- Método helper `resolve(relative_path) -> Path`

**`backend/hand_model/config.py`** define 4 dataclasses:

| Dataclass | Campos destacados |
|---|---|
| `TrainConfig` (l.4-13) | YOLO hand detector: `weights`, `data`, `imgsz=640`, `epochs=100`, `batch=-1`, `amp=true`, `seed=42` |
| `ClassifierConfig` (l.16-29) | MLP alfabético: `input_size=63`, `hidden_sizes=(256,128)`, `num_classes=27` (activo: 24), `dropout=0.3`, `lr=1e-3`, `epochs=150`, `batch_size=64`, `seed=42` |
| `SequenceConfig` (l.32-66) | GRU/Transformer: `input_size=63`, `hidden_size=128`, `num_layers=2`, `num_classes=50` (activo: 12), `max_seq_len=60`, `dropout=0.3`, `lr=1e-3`, `epochs=120`, `batch_size=32`, `aug_factor=20`, `use_extended_augmentations=True`, `cutmix_prob=0.3`, `rotation_invariant=True`, `model_type="gru"` (también `"transformer"`) |
| `AppConfig` (l.69-81) | App runtime: `mode="static"`, `camera_index=0`, `camera_width=640`, `camera_height=480`, `confidence_threshold=0.7`, `confirm_frames=10`, `tts_rate=150`, `tts_lang="es"`, `landmark_model`, `classifier_model`, `sequence_model` |

---

## 4. Modelo ML

### 4.1 Datasets — discrepancia **12.071** vs realidad

| Número | Dónde | Estado |
|---|---|---|
| **12.071 / 11.971** | **NO aparece en ningún `.md`, `.py`, `.kt`, `.json` del repo.** Solo en `PLAN_DESCUBRIMIENTO_LSEC.md:65,68,201` (checklist externo) | **Sin reconciliar en el código** |
| **4.068** muestras alfabéticas | `backend/data/processed/landmarks/manifest.json:24` (2.845 Kaggle + 1.223 mobile_upload) | ✅ Real |
| **2.880** videos LSC50 | filesystem | ✅ Real |
| **4.200** imágenes Kaggle alfabeto | filesystem (21 letras × 200) | ✅ Real |
| **510** secuencias dinámicas en `manifest.json` | `data/sequences/manifest.json:33` | Consistente (suma interna `by_word` y `by_signer` = 510) |
| **492** `.npy` reales en disco | filesystem enumerado | **510 − 18 = 492** (18 saltados en re-extracción rotation-invariant; `reports/orchestrator_log.md:90`) |
| **1.002** en `splits.json` (train 556 + val 169 + test 277) | `data/sequences/splits.json:46-50` | Suma de **muestras augmentadas** (cada orig × ~2.04 por augmentation), no originales |

**Veredicto**: el código nunca menciona 12.071 ni 11.971. Los números comparables son:
- **Alfabético**: 4.068 muestras pre-augmentación (24 clases, train/val/test = 2.844/612/612)
- **Dinámico**: 492 secuencias pre-augmentación (12 clases × 18 signantes) o 1.002 con augmentation train-side

### 4.2 Distribución por clases

**Alfabeto (24 clases estáticas, J y Z excluidas por requerir movimiento)** — `manifest.json:14-17,57-83`:
- Rango: 50 (S, G, H) → 200+ (algunas letras Kaggle); media ≈ 169 muestras/clase

**Dinámico (12 palabras activas)** — `data/sequences/manifest.json:5-13` + `vocabulary.json:2-13`:
- `adios(55) · ayuda(54) · bien(54) · perdon(53) · hola(53) · mal(51) · por_favor(55) · como_estas(29) · no(28) · bano(26) · gracias(26) · si(26)` — suma 510
- Mapeo LSC50→vocab: Hola, Gracias, Por favor, Sí, No, Ayudar (≈ayuda), Baño (=bano), Cómo está (=como_estas), Bien, Mal, Perdón, Chao (=adios)
- **Faltan del vocabulario**: agua, comida, nombre — no en LSC50

### 4.3 Origen y consentimiento

- **LSC50**: dataset público PMC (https://pmc.ncbi.nlm.nih.gov/articles/PMC11655960/) → `backend/data/public/LSC50/`
- **Kaggle "Dataset LSC Modelo"**: https://www.kaggle.com/datasets/oscarstep/dataset-lsc-modelo → `backend/data/public/kaggle/archive/`
- **Mobile_upload.csv**: 1.223 muestras del signante original (consentimiento **no documentado**)
- **No hay IRB, no hay comité de ética, no hay carta de aprobación** en el repo. `docs/F7_TEST_PROTOCOL.md:54,66` y `docs/CHANGELOG.md:320-321` lo declaran explícitamente: *"consentimiento informal, sin IRB"*. Solo se anota `consent=si/no` en bitácora del investigador.

### 4.4 Partición

| Split | Alfabeto (`manifest.json:18-23`) | Dinámico (`splits.json:25-31`) |
|---|---|---|
| Método | Estratificado 70/15/15, **seed=42** | signer-stratified (held-out by signer) |
| Train | 2.844 | 10 signantes (1,17,19,22,23,27,29,31,33,34), 556 augmented |
| Val | 612 | 3 signantes (3,21,28), 169 augmented |
| Test | 612 | 5 signantes (24,25,26,30,32), 277 augmented |

**Adicional**: `data/sequences/loso_splits.json` define **18 folds LOSO canónicos**, uno por signante, seed=42.

### 4.5 Aumento de datos — pipeline de 12 técnicas

`backend/hand_model/augmentations.py`:

| # | Función | Descripción |
|---|---|---|
| 1 | `gaussian_noise` (l.41) | σ=0.02 aditivo por landmark |
| 2 | `joint_jitter` (l.46) | σ=0.015 por articulación |
| 3 | `landmark_dropout` (l.58) | 10% pares (frame,joint) → wrist-origin |
| 4 | `time_warp` (l.73) | Resample 0.7-1.4× + interp back |
| 5 | `frame_dropout` (l.96) | Drop 10% frames + resample |
| 6 | `speed_change` (l.115) | Time warp más estricto ±30% |
| 7 | `cutmix_temporal` (l.120) | Mitad-a + mitad-b misma clase |
| 8 | `rotation_2d` (l.148) | Rotación Z ±35° |
| 9 | `rotation_3d` (l.161) | Roll + pitch ±25° |
| 10 | `scale_jitter` (l.186) | × uniform[0.7,1.4] |
| 11 | `translation_xy` (l.192) | Shift ±0.15 |
| 12 | `mirror_x` (l.202) | Flip X = mano opuesta |

Composición: `augment_sequence()` aplica 11 técnicas (excluye cutmix), `augment_sequence_with_cutmix()` añade cutmix con `cutmix_prob=0.2-0.3`.

### 4.6 Arquitecturas

#### MLP alfabético — `backend/hand_model/classifier.py:8-36`
- **Capas**: `Linear(63→256) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(256→128) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(128→num_classes)`
- **Parámetros**: ≈ 53.5K (~52 KB modelo)
- **Hiperparámetros**: Adam lr=1e-3, weight_decay=1e-4, CosineAnnealing T_max=150, label_smoothing=0.1, batch=64, grad clip 5.0
- **Pérdida**: CrossEntropyLoss
- **Test acc (612 held-out)**: **79.25%** (485/612), gap val-test 0.16% — `docs/CHANGELOG.md:209-221`
- **Confusiones top**: M↔N (5), K↔V (4), C↔O, D↔Y, W↔T, X↔F, I↔Y (3 c/u)
- **Clases <70%**: M (47%), D (58%), X (61%), Q (65%)

#### Bi-GRU + Attention (dinámico legacy) — `sequence_model.py:8-54`
- **Capas**: `nn.GRU(bidirectional=True, hidden_size=128, num_layers=2)` + `Linear(256,1)` attention (softmax temporal) + head `Linear(256→64→12)` ReLU Dropout(0.3)
- **Parámetros**: ≈ 462K
- **LOSO 18-fold honesto (legacy, old norm)**: **11.67% ± 7.84%** (min 0% signer 32, max 30% signer 28) — `reports/f0_loso_baseline_gru.json:114-119`
- **LOSO 18-fold (clean, rotation-invariant re-train)**: **84.01% ± 17.01%** (min 42.86% signer 32, max 100% signers 17/19/23/26/27) — `reports/gru_clean_loso.json:114-119`

#### SignTransformer (SPOTER-style, F3) — `spoter_model.py:54-216`
- **Capas**: `Linear(63→128)` + CLS token aprendido + PE aprendido + `TransformerEncoder(4 layers, 4 heads, d_ff=256, GELU, pre-norm, dropout=0.1)` + CLS pool + head `Linear(128→128→12)` GELU Dropout(0.1)
- **Parámetros**: **564.492** (registry `n_params:564492`, `models/exports/registry.json:179,204`)
- **Docstring explícito**: *"originally inspired by SPOTER [Bohacek & Hruz, WACV 2022], then simplified based on empirical evidence that the per-joint / per-channel embedding trick from SPOTER **hurts** with our small training set"* (`spoter_model.py:3-7`). **Adaptación divergente, no reproducción.**
- **LOSO clean (post-recuperación)**: **77.12% ± 15.74%** (min 52.63% signer 31, max 100% signer 17) — `reports/transformer_clean_loso.json:114-119`
- **LOSO contaminated (pre-recuperación, RETRACTADO)**: 87.39% ± 13.73% — mezcla de dos espacios de normalización

### 4.7 Métricas honestas resumidas (CHANGELOG.md:70-73)

| Modelo | LOSO mean ± std | Test (5 signers) | Val (3 signers) | Conf | Params |
|---|---|---|---|---|---|
| Legacy GRU (old norm) | 11.67% ± 7.84% | 75.19%* | 67.06% | 0.78 | 462K |
| GRU clean (rotation-inv) | 84.01% ± 17.01% | 72.93% | 69.41% | 0.73 | 462K |
| SignTransformer (clean) | 77.12% ± 15.74% | 69.17% | 75.29% | 0.73 | 564K |
| SignTransformer (contaminated) | 87.39% ± 13.73% | 75.19%* | 77.65%* | 0.83 | 564K |

*El "75.19% test" del legacy GRU y del Transformer "contaminado" es **engañoso** — los 5 signers de test (24/25/26/30/32) son sistemáticamente más fáciles que los otros 13 (ver `docs/STATUS_F0_F3.md:7-33`).*

### 4.8 Features / preprocesamiento

| Aspecto | Implementación |
|---|---|
| **Landmarks** | **Solo mano**: 21 keypoints × 3D = **63 dims/frame**. **NO usa pose ni face** (no es MediaPipe Holistic). `numHands=1`. |
| **Normalización** | wrist-relative + scale-invariant (landmark 9 como referencia) — `landmarks.py:9-22`. Variante **rotation+handedness-invariant** (`landmarks.py:25-66`): mirror si zurdo, rota vector wrist→MCP_9 al eje +Y. Patrón de Zimmermann & Brox 2017. |
| **Secuencia** | Variable; pad/trunc a `max_seq_len=60` en inferencia (`sequence_model.py:122-128`). Padding mask `(seq.abs().sum(dim=(2,3)) < 1e-6)` para transformer. |
| **Coordenadas** | 3D, normalizadas a escala unitaria (~1.0) |

### 4.9 Latencia (PC CPU)

| Modelo | batch=1 p50 / p99 | batch=32 p50 / p99 | Fuente |
|---|---|---|---|
| GRU (legacy) | 23.9 ms / 36 ms | 30.9 ms / 72.1 ms | `reports/f0_latency_baseline.json:5-74` |
| SignTransformer (clean) | (no medido batch=1) | 2.5 ms / 18.3 ms | `reports/f3_latency_transformer.json:5-74` |

**Latencia móvil (Infinix GT30 Pro)**: **NO medida todavía**. Objetivo README: <500 ms end-to-end → cumplido en PC con margen amplio.

### 4.10 Comparación con Baihan 2024

**NO hay ninguna referencia a "Baihan"** en `.md`, `.py`, `.kt`, `.json` del proyecto. Las únicas dos coincidencias están en `PLAN_DESCUBRIMIENTO_LSEC.md:82,195` (el propio checklist). **No se cita, no se reproduce.**

---

## 5. Pipeline de inferencia

### 5.1 Captura

| Aspecto | Valor | Fuente |
|---|---|---|
| Cámara | trasera por defecto (toggle a frontal) | `UserPreferences.useFrontCamera` |
| Resolución | 640×480 | `config.py:73-74` |
| FPS procesamiento | `timestampMs += 33 ms/frame` (asume ~30 fps) | `HandDetector.kt:85` |
| MediaPipe confidence | detection/presence/tracking = 0.5 | `HandDetector.kt:44-47` |
| Modo MediaPipe | VIDEO (no IMAGE/STREAM) | `HandDetector.kt:43` |
| Suavizado temporal | ninguno (1 frame = 1 inferencia) | — |

### 5.2 Preprocesamiento

- `HandDetector.kt:111-141`: YUV_420_888 → JPEG → Bitmap → `BitmapImageBuilder` → `HandLandmarker.detectForVideo`
- Output: 21 landmarks × 3 floats → `LandmarkNormalizer.normalize(...)` → `FloatArray(63)`
- Padding/truncado a `max_seq_len=60` en `SignTransformerClassifier.processDynamicFrame`

### 5.3 Inferencia

| Aspecto | Valor |
|---|---|
| **Dispositivo** | 100% on-device. **No se sube video a ningún servidor.** Solo landmarks y CSVs llegan al backend, nunca píxeles. |
| **Backend ONNX** | `com.microsoft.onnxruntime:onnxruntime-android:1.17.3`, `intraOpNumThreads=2`, `OptLevel.NO_OPT` (`SignTransformerClassifier.kt:120-124`) |
| **GPU/NNAPI delegate** | **NO usado**. ONNX CPU + 2 threads. TFLite tampoco usa GPU delegate. |
| **Batch** | Fijo = 1 (producción) |

### 5.4 Postprocesamiento

- **Decodificación**: argmax sobre softmax de logits (`SignTransformerClassifier.kt:542-549`)
- **Manejo de señas no reconocidas**: rechazo si `confidence < threshold` (default 0.35, calibrado por `AdaptiveThresholds`) + CentroidClassifier retorna `null` si cosine < 0.5
- **State machine** (`SignTransformerClassifier.kt:88,193-289`):
  ```
  IDLE → (motion > threshold) → BUFFERING
  BUFFERING → (motion estable N frames) → infer → (conf ≥ thresh) → emit
  BUFFERING → (sin mano N frames) → infer si buffer >10
  emit → COOLDOWN (12 frames) → IDLE
  ```
- **Variantes regionales (Ecuador)**: hard-coded map que reescribe glosas — "hola" → "¡Qué más!", "si" → "Sí, pues", "gracias" → "Gracias, pana" (`SignClassifier.kt:47-63`)

---

## 6. UI/UX y producto

### 6.1 Pantallas (rutas Compose Navigation, `LSECNavHost.kt:19-87`)

| Ruta | Composable | ViewModel |
|---|---|---|
| `home` (start) | `HomeScreen` | (stateless) |
| `camera` | `CameraScreen` + `LiveConfidenceBar` + botón "Reformular" (stub TODO) | `CameraViewModel` |
| `calibration` | `CalibrationScreen` (stub TODO 30s) | (none) |
| `dictionary` | `DictionaryScreen` (15 categorías, búsqueda, favoritos) | `DictionaryViewModel` |
| `settings` | `SettingsScreen` (server URL, API key, modelos, LLM toggles, theme, TTS) | `SettingsViewModel` |
| `data_capture` | `DataCaptureScreen` (grabar muestras y subir al backend) | `DataCaptureViewModel` |
| `training` | `TrainingScreen` (lanzar reentrenamiento via API, descargar modelo) | `TrainingViewModel` |

Fuera de la nav (acceso manual vía adb): `TestSessionScreen` en `SignTestLogger.kt:253` (F7 harness).

### 6.2 Internacionalización

- **Solo español (Ecuador)** — `values/strings.xml` (default). Único `values-night/` es dark-mode. **No existe `values-en/`, `values-es/`, ni `values-qu/`.** **No hay soporte Kichwa en i18n, TTS ni vocabulario del modelo.** Las "variantes regionales" son ecuatorianismos del español (QUITO/GUAYAQUIL/CUENCA/NATIONAL en `SignEntity.regionalVariant`), no lengua indígena.

### 6.3 Accesibilidad

| Aspecto | Estado |
|---|---|
| `contentDescription` en iconos | ✅ 57 ocurrencias (botones back, search, favorito, audio, etc.) |
| `contentDescription` en preview cámara | ❌ `PreviewView` sin `semantics{}`, TalkBack anuncia "sin etiqueta" |
| Live regions para glosa detectada | ❌ Ninguna |
| TalkBack-aware announcements | ❌ Ninguno |
| Decorative `null` | ✅ 20+ iconos marcados como decorativos |
| Soporte RTL | ✅ `supportsRtl="true"`, usa `paddingStart/End` |
| Tamaño táctil | ✅ Material defaults (Cards 100dp altura) |
| Contraste de color | ⚠️ Sin auditoría WCAG; texto blanco sobre negro translúcido 70% |
| Pruebas con comunidad sorda | ⚠️ Solo planeadas en F7 (5 testers reclutados pero **aún no ejecutadas** — `docs/F7_TEST_PROTOCOL.md` define el protocolo, no hay CSVs todavía) |

### 6.4 Métricas de usabilidad

- **SUS score: NO medido**.
- **Tasa de éxito por tarea / tiempo promedio: NO medidos**.

---

## 7. Despliegue y operación

| Aspecto | Valor | Fuente |
|---|---|---|
| Build type | APK (debug + release). **No AAB, no per-ABI splits** | `frontend/app/build.gradle.kts` (sin `splits { abi {} }`) |
| Tamaño APK estimado | ~70-90 MB debug universal / ~40-60 MB release minified universal (4 ABIs incluidos sin split) | cálculo basado en deps (MediaPipe ~7 MB, ONNX Runtime ~10-15 MB, Compose, TFLite, Firebase) |
| Min Android | **API 26 (Android 8.0)** | `frontend/app/build.gradle.kts:23` |
| Target / Compile SDK | **34** | `frontend/app/build.gradle.kts:19,24` |
| Permisos | `CAMERA`, `INTERNET` | `AndroidManifest.xml:6,9` |
| Features requeridas | `android.hardware.camera` (required), `autofocus` (optional) | `AndroidManifest.xml:12-13` |
| Privacidad video | **Procesamiento 100% on-device**. Solo landmarks (FloatArray 63) y CSV subidos al backend. Píxeles nunca salen del móvil. | arquitectura |
| LOPDP / Ecuador compliance | **No documentado**. `network_security_config.xml` permite cleartext global sin scope. | `network_security_config.xml:8` |
| HTTPS prod | `cleartextTrafficPermitted="true"` global. Production default: `https://placeholder-lsec-api.hf.space/` (Hugging Face Space) | `app/build.gradle.kts:59` |
| Backend | Docker (Debian), `backend/Dockerfile`, `docker-compose.yml`, `deploy.sh` + `cloudflare-tunnel.sh`. CPU-only torch pinned `2.3.1+cpu` | `backend/api/Dockerfile` |
| Deploy command | `cd backend && ./deploy.sh install` una vez; luego `./deploy.sh up|rebuild|restart|stop|logs|status|ip` | `docs/AGENTS.md` |

---

## 8. Testing y calidad

### 8.1 Cobertura

- **Backend**: 5 archivos pytest, **34 `def test_*`** (≈36 IDs con parametrize, claim en `orchestrator_log.md:52`). Marcador `slow` definido en `pyproject.toml:32-35`.
- **Android**: **0 tests JUnit**. `frontend/app/src/test/` y `frontend/app/src/androidTest/` no existen. Solo deps declaradas (`junit:4.13.2`, `androidx.test.ext:junit`, `espresso-core`, `compose ui-test-junit4`) — nunca usadas.

### 8.2 Desglose de tests backend (`backend/tests/`)

| Archivo | Tests | Cubre |
|---|---|---|
| `test_data.py` | 1 | `hand_model.data.get_processed_paths()` retorna dict con `images`/`labels` |
| `test_normalization.py` | 7 | `normalize_landmarks` y `normalize_landmarks_invariant` (wrist=origin, MCP_9 unit length, X→+Y rotation, handedness mirror); `LandmarkExtractor` constructor con `rotation_invariant` |
| `test_augmentations.py` | 18 | Smoke tests de las 11 augmentaciones (shape, dtype, NoN/NoInf, stochasticidad, cutmix halves) |
| `test_sign_transformer.py` | 7 (+2 parametrize) | Construcción; forward pass `(B,T,21,3)` y `(B,T,63)` → `(B, num_classes)`; CLS vs mean pool; ONNX export + onnxruntime match PyTorch within `1e-4` |
| `test_end_to_end.py` | 1 (`slow`) | Pipeline sintético: load → augment 5 → 2 epochs → eval en signers held-out; verifica `test_acc >= 1/C` y checkpoint escrito |

### 8.3 Pruebas externas

- **Pruebas en dispositivos reales**: NO realizadas. Plan F7 (Infinix GT30 Pro) documentado pero no ejecutado. Sin CSVs de tester en el repo.
- **Condiciones adversas (poca luz, fondo complejo, oclusión)**: NO probadas sistemáticamente.

---

## 9. Documentación y deuda técnica

| Ítem | Estado | Fuente |
|---|---|---|
| `README.md` (raíz) | ✅ Breve, apunta a `docs/` y `frontend/README.md` | `README.md` |
| `docs/CHANGELOG.md` | ✅ Excelente, 839 líneas, formato Keep a Changelog | `docs/CHANGELOG.md` |
| `docs/STATUS_F0_F3.md` | ✅ Tracker por fase con callouts críticos | `docs/STATUS_F0_F3.md` |
| `docs/AGENTS.md` | ✅ Onboarding para AI agents (88 líneas) | `docs/AGENTS.md` |
| `docs/ARCHITECTURE.md` | ✅ Diagramas + secciones por componente | `docs/ARCHITECTURE.md` |
| `docs/INSTRUCCIONES/` | ✅ README, FUENTES, GUIA_PASO_A_PASO en español | `docs/INSTRUCCIONES/` |
| `docs/INTEGRACION_FRONTEND_BACKEND.md` | ✅ Contrato Android↔backend | `docs/INTEGRACION_FRONTEND_BACKEND.md` |
| Diagramas UML / secuencia | ❌ No hay | — |
| Comentarios en código | ✅ Density alta en Python (docstrings), moderado en Kotlin (KDoc en clases ML) | tree |
| Licencia | ❌ **NO hay archivo LICENSE** | tree |
| Trazabilidad commits | ✅ Excelente: cada commit documenta fase + métricas | `git log` |
| Deuda técnica notable | (a) Stale `num_classes` en `ClassifierConfig=27` y `SequenceConfig=50` cuando el activo es 24/12; (b) `_train_tmp.csv` / `_val_tmp.csv` con schema flat (63 cols) inconsistente con resto; (c) `train_lstm_model.py:293` entrena con `generate_synthetic_data()` (legacy); (d) `extractEmbedding()` retorna `null` hasta re-exportar ONNX con `output_names=["logits","embedding"]`; (e) `splits.json` (1002 augmented) inconsistente con `manifest.json` (510 originales); (f) `evaluate.py`/`export.py`/`train_yolo.py` son stubs vacíos; (g) CalibrationScreen y botones "Reformular" son TODOs | múltiples |

---

## 10. Verificación cruzada con tesis

| Afirmación de la tesis | Realidad en código | Coincide? |
|---|---|---|
| MLP+GRU | Ambos existen; **SignTransformer es el backend activo en producción**. GRU solo como fallback | **PARCIAL** |
| 12.071 muestras | NO aparece en repo; comparable = 4.068 alfabético + 492 dinámico pre-augment = 4.560 | **NO** |
| 90-96% accuracy | Alfabético 79.25%; dinámico LOSO 77-84%; solo Transformer "contaminado" llegó a 87% (retractado) | **NO** |
| Latencia <500 ms | GRU p99 72 ms, Transformer p99 18 ms en PC CPU | **SÍ** (en PC) |
| MediaPipe Holistic (21 mano × 2 + 33 pose + 468 face) | **Solo mano**: 21 × 1 × 3 = 63 dims | **NO** (solo mano) |
| Aumento de datos | Implementado: 12 técnicas en `augmentations.py` (ruido, jitter, dropout, time_warp, frame_dropout, speed_change, cutmix, rotations 2D/3D, scale, translation, mirror) | **SÍ** |
| Split train/val/test | 70/15/15 alfabético (seed=42); signer-stratified dinámico; LOSO 18 folds canónicos | **SÍ** |
| Consentimiento / ética | Informal, sin IRB, sin comité de ética | **Documentado pero no formal** |

---

## 11. Citas externas

| Cita | Dónde | Contexto |
|---|---|---|
| **SPOTER [Bohacek & Hruz, WACV 2022]** | `backend/hand_model/spoter_model.py:3` | Inspiración arquitectónica del SignTransformer; **modificada**, no reproducida (per-joint embedding removido empíricamente) |
| **Zimmermann & Brox 2017 "Hand Keypoint Detection in Real Images"** | `docs/AGENTS.md:62`, `CHANGELOG.md:311-312,710-711`, `INSTRUCCIONES/FUENTES.md:41` | Patrón wrist-relative + scale-invariant. Sin DOI |
| **Google "ML Kit latency best practices" (2023)** | `docs/CHANGELOG.md:343` | Estándar de medición latencia end-to-end mobile ML |
| **Nielsen Norman Group "How Many Test Users" (1993, rev. 2012)** | `docs/CHANGELOG.md:340-341` | Justificación de 5 testers F7 |
| **WLASL-2000** | `spoter_model.py:28`, `orchestrator_log.md:126` | Referencia de tamaño de dataset, no usado |
| **CSLR foundational (Camgöz et al. 2018)** | `docs/CHANGELOG.md:89,488` | Mencionado, no citado formalmente |
| **Baihan 2024** | **NO CITADO en ningún archivo** del proyecto | — |

---

## 12. Modelo de riesgos y hallazgos críticos

### Hallazgos que requieren acción del autor

1. **Discrepancia 12.071 vs realidad**: el número de la tesis **no coincide** con ningún conteo del código. Decisión necesaria: ¿revisar la tesis o re-explicar la metodología?

2. **"90-96% accuracy" no coincide con métricas honestas**:
   - Alfabético: 79.25% (gap -11pp)
   - Dinámico LOSO: 77-84% (gap -6 a -13pp)
   - Solo Transformer "contaminado" llegó a 87% (retractado)
   - Decisión: ¿revisar la tesis, actualizar el target, o usar el número retractado?

3. **Sin IRB ni comité de ética**: si la defensa de tesis lo requiere (varios programas universitarios lo exigen), hay que conseguir aprobación retroactiva o documentar por qué no aplica (entorno académico local sin IRB).

4. **Sin tests Android**: `frontend/app/src/test/` y `androidTest/` no existen. Si la rúbrica exige cobertura Android, hay que empezar.

5. **Sin per-ABI APK splits**: APK universal ~70-90 MB. Si el target es Play Store, conviene AAB con per-ABI splits (reduciría a ~25-30 MB por ABI).

6. **Sin Kichwa**: si la rúbrica o el contexto sociocultural lo requiere, hay que planificar i18n + TTS.

7. **API key en texto plano en DataStore**: `UserPreferences.apiKey` se almacena sin encriptación. Considerar EncryptedSharedPreferences o Android Keystore.

8. **F7 todavía no ejecutado**: el protocolo está documentado pero no hay CSVs. Si la tesis dice "validado con 5 usuarios", hay que correr F7 antes de la defensa.

### Hallazgos menores (deuda técnica, no bloquean defensa)

- Stale `num_classes` defaults en `ClassifierConfig`/`SequenceConfig`
- `_train_tmp.csv` / `_val_tmp.csv` schema flat inconsistente
- `train_lstm_model.py` entrena con datos sintéticos (legacy no usado, pero no borrado)
- `extractEmbedding()` retorna `null` (requiere re-export ONNX con output adicional)
- Stubs vacíos: `evaluate.py`, `export.py`, `train_yolo.py`
- TODOs abiertos: CalibrationScreen, botones "Reformular", SignTestLogger helpers

---

## 13. Estimación de tiempo real vs plan

| Fase | Plan estimado | Real (con paralelización) |
|---|---|---|
| 0-1 (Estructura) | 35 min | ~10 min |
| 2 (Stack) | 20 min | ~3 min |
| 3 (ML/Datos) | 45 min | ~15 min |
| 4 (Inferencia) | 30 min | ~5 min |
| 5-6 (UI/Despliegue) | 35 min | ~10 min |
| 7-8 (Testing/Docs) | 35 min | ~5 min |
| **Total** | **~3.5 h** | **~50 min** |

Aceleradores: 2 `explore` agents en paralelo (frontend + backend ML), `git log --oneline` para historial, `docs/CHANGELOG.md` (839 líneas) con casi todas las decisiones pre-documentadas.

---

## Anexo A — Inventario de archivos `backend/data/` commiteados

| Path | Cantidad | Notas |
|---|---|---|
| `data/landmarks/*.csv` | 4 archivos (8 KB - 830 KB) | `mobile_upload.csv` (1223 samples), `collected_data.csv` (vacío), `_train_tmp.csv` / `_val_tmp.csv` (legacy flat schema) |
| `data/processed/landmarks/*.csv` | 5 archivos | `alphabet_kaggle.csv` (2845), `train.csv` (2844), `val.csv` (612), `test.csv` (612), `manifest.json` |
| `data/public/LSC50/Videos Colors/<signer>/Colores/<color>/<n>.avi` | ~1080 videos, 12 signers × 11 colores × 5 reps | videos crudos |
| `data/public/LSC50/Videos Courtesy/<signer>/<word>/<n>.avi` | 510 videos, 18 signers × 12 palabras | videos crudos para secuencias |
| `data/public/LSC50/Videos Numbers/...` | videos restantes del dataset | videos crudos |
| `data/public/kaggle/archive/<letter>/*.jpg` | 4200 imágenes, 21 letras × 200 | imágenes crudas alfabeto |
| `data/sequences/<word>/<signer>_<word>_<idx>.npy` | **492 archivos** (T,63) float16 | secuencias reales re-extraídas con normalizador invariante |
| `data/sequences/manifest.json` | 1 | metadata por archivo, 510 entries (18 skipped <12 frames) |
| `data/sequences/splits.json` | 1 | signer-stratified train/val/test (augmented: 556/169/277) |
| `data/sequences/loso_splits.json` | 1 | 18 folds LOSO canónicos |
| `data/sequences/vocabulary.json` | 1 | 12 palabras indizadas |

## Anexo B — Inventario de archivos `backend/models/`

| Path | Tamaño | Notas |
|---|---|---|
| `models/classifier/sign_mlp.pth` | 217 KB | MLP alfabético activo |
| `models/sequence/sign_gru_clean.pth` | 1.77 MB | GRU baseline re-entrenado clean |
| `models/sequence/sign_transformer.pth` | 2.17 MB | SignTransformer F3 (clean) |
| `models/mediapipe/hand_landmarker.task` | 7.46 MB | detector MediaPipe bundled |
| `models/exports/sign_mlp_*.onnx` (14 archivos) | 71-212 KB | historial completo de exports MLP |
| `models/exports/sign_gru_*.onnx` (5 archivos) | 1809-1918 KB | historial completo de exports GRU |
| `models/exports/sign_transformer_*.onnx` (3 archivos) | ~397 KB + `.onnx.data` | historial completo de exports Transformer (FP32 + INT8) |
| `models/exports/*_labels_*.json` (19 archivos) | 50-150 B | label maps versionados |
| `models/exports/registry.json` | 208 líneas | registry activo con todos los modelos |

## Anexo C — Tests disponibles (resumen ejecutable)

```bash
# Backend
cd backend && .venv\Scripts\python -m pytest -q tests

# Saltar slow test
cd backend && .venv\Scripts\python -m pytest -q tests -m "not slow"

# Android (no hay tests; deps declaradas)
cd frontend && ./gradlew test          # FAIL: no source set
cd frontend && ./gradlew connectedCheck  # FAIL: no source set
```

## Anexo D — Comandos de reproducibilidad

```bash
# Re-extracción de datos (corre en background, ~17 min)
cd backend && .venv\Scripts\python.exe -m scripts.run_reextract_background

# Entrenamiento SignTransformer (background, ~35 min para 80 epochs)
cd backend && .venv\Scripts\python.exe -m scripts.run_train_background --name train_v4 \
    --module scripts.train_transformer -- \
    --epochs 80 --aug-factor 10 --cutmix-prob 0.2 \
    --splits-json data/sequences/splits.json --batch-size 32

# Eval-only LOSO (~2 segundos)
cd backend && .venv\Scripts\python.exe -m scripts.loso_test_only --ckpt models/sequence/sign_transformer.pth

# Full pipeline (raw videos → ONNX)
backend\scripts\full_pipeline.bat                    # full
backend\scripts\full_pipeline.bat --fast             # epochs reducidos
backend\scripts\full_pipeline.bat --no-extract       # skip re-extracción

# Export ONNX FP32 + INT8
cd backend && python -m scripts.export_transformer_onnx

# Comparar GRU vs Transformer
cd backend && python -m scripts.compare_models --export reports/gru_vs_transformer.json

# Evaluar calibración centroid
cd backend && python -m scripts.evaluate_centroid_calibration --n-shots 5

# Backend dev
cd backend && start_backend.bat
# o
cd backend && python start_backend.py

# Docker
cd backend && ./deploy.sh install   # una vez
cd backend && ./deploy.sh up
```

## Anexo E — Preguntas para el autor de la tesis

1. **¿"12.071 muestras" de dónde sale?** No aparece en el código. ¿Es de la documentación externa, del paper, o de la metodología previa?
2. **¿"90-96% accuracy" sobre qué evaluación?** Las métricas honestas no lo alcanzan. ¿Cuál es el protocolo que justifica esa cifra?
3. **¿MediaPipe Holistic o solo mano?** El código usa solo mano (63 dims). ¿La tesis dice Holistic?
4. **¿Baihan 2024 fue reproducido?** No hay ninguna referencia en el repo. ¿Hay material auxiliar no commiteado?
5. **¿IRB / comité de ética fue aprobado?** El protocolo F7 asume consentimiento informal. ¿La universidad lo acepta?
6. **¿Kichwa es objetivo?** El código no lo soporta. ¿Está en el alcance de la tesis?
7. **¿F7 ya ejecutado?** El protocolo está documentado pero no hay CSVs. ¿Cuándo se corre?
8. **¿Cuál es la fecha de defensa?** Afecta el plan de remediación de los hallazgos arriba.

---

> **Refs**:
> - Plan ejecutado: `PLAN_DESCUBRIMIENTO_LSEC.md`
> - Bitácora del proyecto: `docs/CHANGELOG.md`
> - Tracker por fase: `docs/STATUS_F0_F3.md`
> - Onboarding técnico: `docs/AGENTS.md`
> - Contrato Android↔backend: `docs/INTEGRACION_FRONTEND_BACKEND.md`
> - Logs de sesión del orquestador multi-agente: `backend/reports/orchestrator_log.md`
> - Métricas crudas: `backend/reports/*.json`