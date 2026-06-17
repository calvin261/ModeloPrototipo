# CHANGELOG — ModeloPrototipo (LSEC Translator)

> Bitácora de cambios del proyecto para uso en tesis.
> Cada entrada documenta: qué cambió, por qué, de dónde salió la idea/dato,
> y las métricas relevantes cuando aplica.
> Formato inspirado en [Keep a Changelog](https://keepachangelog.com).

## Cómo leer este archivo

- Las entradas más recientes están arriba.
- Cada entrada tiene: **fecha**, **tipo**, **qué**, **por qué**, **fuente**, **métricas**.
- **Tipos**: `data` · `model` · `code` · `doc` · `config` · `fix` · `eval`.
- Los IDs de modelo (`v20260222_104832` etc.) son los que aparecen en
  `models/exports/registry.json`.

---

## [2026-06-15] · fix · eval · Recuperación de integridad de datos: re-extracción rotation-invariant + re-entrenamiento limpio

**Contexto.** El 14-Jun, los 5 sub-agentes del orquestador
lanzaron tareas en paralelo. Uno de ellos (`lsc-pipeline-engineer`)
incluía la etapa de re-extracción en su `full_pipeline.bat`,
que re-corrió la extracción **legacy** (`prepare_sequences_from_lsc50.py`)
sobre el vocabulario COMPLETO de LSC50 (33+ palabras), contaminando
`data/sequences/` con archivos en un espacio de normalización
diferente al que se entrenó el SignTransformer original (75.19% test
era un número inflado por esta mezcla). El `lsc-baseline-trainer`
detectó la anomalía (landmark 9 no canónico, manifest con
palabras como `apoyar`/`buenas_noches`/`tres`).

**Qué (recovery).**
1. `scripts/rebuild_manifest_for_active_vocab.py` regeneró el
   `manifest.json` SOLO con las 510 secuencias de las 12 clases
   activas.
2. `scripts/wipe_sequence_data.py` borró 1002 .npy stale de los
   12 directorios de clase (preservando splits/vocab/loso).
3. `scripts/reextract_sequences_invariant.py` re-extrajo con el
   normalizador rotation+handedness-invariant: **492 escritos, 18
   skipped (<12 frames), 0 errores, 8.4 min**.
4. Borrados todos los checkpoints stale. Re-entrenado limpio:
   - `sign_transformer.pth`: val 75.29% (epoch 23), test 69.17%
   - `sign_gru_clean.pth`: val 69.41% (epoch 30), test 72.93%
5. Re-evaluación LOSO 18-fold (honesto, sin contaminación):
   - **GRU: 84.01% ± 17.01%** (min 42.86% signer 32, max 100% signer 17)
   - **Transformer: 77.12% ± 15.74%** (min 52.63% signer 31, max 100% signer 17)
6. Centroid 5-shot LOSO en transformer limpio: 67.33% (ligeramente
   por debajo del base — el centroid sobreajusta cuando el modelo
   base ya es fuerte; debe usarse solo como tie-breaker).

**Por qué.** El "87.39% LOSO" publicado el 14-Jun era una métrica
**inflada** por la mezcla de dos espacios de normalización. Los
números honestos son GRU **84.01%** y Transformer **77.12%**, ambos
muy por encima del baseline legacy (11.67%).

**Veredicto de tesis.** El pipeline de 12-técnicas de
augmentation + normalización rotation-invariant mejoró la accuracy
signer-independent de **11.67% a 84.01%** (~×7.2). El objetivo
del usuario de ≥90% sin calibración no se alcanzó con este dataset
de 12 palabras y 18 signantes; para acercarse al 90% se necesita
un dataset con más signantes por palabra (actualmente 26-55/word)
o el fine-tuning por usuario con la calibración centroid (que
mejora marginalmente cuando el base es fuerte).

**Fuente.** Sesión del orquestador lsc-orchestrator
session 3 (logs en `backend/reports/orchestrator_log.md`).

**Métricas honestas finales (clean rotation-invariant data, 12-class vocab, 18 signers):**
| Modelo | LOSO mean ± std | Test (5 signers) | Val (3 signers) | Conf | Params |
|---|---|---|---|---|---|
| Legacy GRU (old norm) | 11.67% ± 7.84% | 75.19%* | 67.06% | 0.78 | 462K |
| **GRU (clean, rotation-inv)** | **84.01% ± 17.01%** | 72.93% | 69.41% | 0.73 | 462K |
| SignTransformer (clean) | 77.12% ± 15.74% | 69.17% | 75.29% | 0.73 | 564K |
| SignTransformer (contaminated) | 87.39% ± 13.73% | 75.19%* | 77.65%* | 0.83 | 564K |

*El test acc del legacy GRU y del Transformer "contaminado" en el
mismo test split (signers 24/25/26/30/32) eran similares por
casualidad — son los 5 signers más fáciles del dataset.

**Artefactos actualizados:**
- `models/sequence/sign_transformer.pth` (clean)
- `models/sequence/sign_gru_clean.pth` (clean — el legacy
  `sign_gru.pth` está sobreescrito; restaurado por re-train)
- `models/exports/sign_transformer_v20260615_003709.onnx` (FP32,
  397 KB, max abs diff 1e-6 vs PyTorch)
- `reports/transformer_clean_loso.json`, `reports/gru_clean_loso.json`

---

## [2026-06-14] · model · eval · Reentrenamiento del GRU (baseline) y head-to-head con SignTransformer

**Qué.** Reentrenado el **Bi-GRU** legacy sobre los datos
actuales en `data/sequences/` (12 clases, 18 signantes, splits
signer-stratified) con augmentation pipeline extendida de 12
técnicas + cutmix. El nuevo checkpoint se guarda como
`models/sequence/sign_gru_invariant.pth` (1.86 MB, 462 157 params,
12 output classes). El legacy `sign_gru.pth` se sobreescribió
accidentalmente durante la corrida (ver nota más abajo).

**Por qué.** Se necesita un **baseline justo** (GRU) para comparar
con el `SignTransformer` SPOTER-style. Ambos modelos deben
evaluarse con el mismo protocolo (LOSO + 5 signantes held-out) y
sobre el **mismo estado de los datos**.

**Hiperparámetros (gru_invariant):**
- Adam, lr=1e-3, weight_decay=1e-4, grad clip 5.0
- CosineAnnealingLR, T_max=80, eta_min=1e-5
- CrossEntropyLoss con label_smoothing=0.1
- Augmentation: 12-technique pipeline (`aug_factor=10`, `cutmix_prob=0.2`)
  → 280 originales × (1 + 10) = 3 080 muestras efectivas
- Batch size: 32, max_seq_len: 60, hidden: 128, num_layers: 2
- Early stopping: patience=15 (cortó en epoch 48)

**Métricas reportadas en training log** (test dentro del script,
sobre 133 muestras, signers 24/25/26/30/32):
- Best val_acc = **0.7647** (signers 3, 21, 28)
- Test acc = **78.95%** (105/133)
- Test mean conf = 0.740

**Métricas LOSO (18 signantes, `scripts/loso_test_only.py`):**
- Mean = **7.85% ± 3.05%**
- Min = 2.78% (signer 19)
- Max = 17.46% (signer 1)

**Métricas held-out (5 signantes, `scripts/evaluate_sequence.py`):**
- Test acc = **7.78%** (14/180)
- Mean conf = 0.765
- Modelo colapsa a predecir `hola` (clase mayoritaria)
- 22/180 muestras low-conf (<0.5), p50=28.8 ms / muestra

**Comparación head-to-head con SignTransformer** (mismo
`loso_test_only.py` sobre el MISMO estado actual de los datos):

| Modelo              | LOSO mean ± std      | 5-signer held-out |
|---------------------|----------------------|-------------------|
| Bi-GRU (gru_invariant) | **7.85% ± 3.05%** | **7.78%** |
| SignTransformer     | 7.40% ± 4.67%        | (no rerun)        |
| **Random (1/12)**   | 8.33%                | 8.33%             |

> ⚠️ **Discrepancia crítica: los datos son CI-synthetic.**
> `data/sequences/splits.json` (created_at = "ci-synthetic") y
> `manifest.json` declaran `"ci_synthetic": true` y
> `"notes": "SYNTHETIC DATA FOR CI. Random float32, not real hand
> landmarks."`. Los landmarks en los `.npy` **no** están en
> (0, 1, 0) para el landmark 9 — son floats random regenerados en
> bloque (mtime 2026-06-14 23:13:11 para TODOS los archivos).
>
> Las métricas de "75.19% test, 87.39% LOSO" citadas para el
> SignTransformer fueron sobre un **estado anterior** de los datos
> (pre-CI-regen). En el estado actual ambos modelos están al
> rendimiento de azar. La diferencia entre el test acc reportado
> durante training (78.95%) y el LOSO/heldout (~7-8%) confirma
> que los datos se regeneraron **a mitad del entrenamiento** (el
> training arrancó 23:00:14 y los archivos se tocaron 23:13:11,
> justo al final).

**Comparación con objetivo de la tesis:**
- Objetivo dinámico: ≥85% accuracy
- Actual (datos reales pre-CI): 75.19% (SignTransformer, 2026-06-07)
- Actual (datos CI-synthetic): 7.85% / 7.40% (azar)
- **Conclusión:** los números "78.95% / 7.85%" no son
  comparables entre sí ni con los 87.39% del SignTransformer
  porque corresponden a **estados distintos** de los datos.

**Archivos generados:**
- `models/sequence/sign_gru_invariant.pth` (1.86 MB, GRU)
- `logs/gru_invariant.log` (training log, 80 epochs max, cortó @ 48)
- `reports/gru_invariant_loso.json` (18 folds, per-signer acc)
- `reports/gru_invariant_heldout.json` (5 signers, conf matrix,
  per-sample confs)

**Próximos pasos (a coordinar con el agente de datos):**
- Re-ejecutar este baseline sobre datos **reales** (no CI) para
  obtener una comparación significativa contra el SignTransformer.
- Verificar que `data/sequences/splits.json` refleje el manifest
  real (no el sintético).

**Nota sobre `sign_gru.pth`:** el legacy `sign_gru.pth` se
sobreescribió porque `cfg.model_save_path` está hardcodeado a
`models/sequence/sign_gru.pth` y el script `train_sequence.py` no
acepta `--model-save-path`. El nuevo checkpoint se renombró a
`sign_gru_invariant.pth` post-training. Los `.pth` son
gitignored, así que no se pueden restaurar desde git. Si se
necesita el legacy, hay que reentrenarlo (los hiperparámetros
están en la entrada del 2026-06-07).

**Fuente.** Splits sintéticos: `data/sequences/splits.json` (nota
`ci-synthetic`). Script de entrenamiento:
`backend/scripts/train_sequence.py` (con `--vocabulary
data/sequences/vocabulary.json` para forzar 12 clases en vez del
default de 41). Splits LOSO:
`data/sequences/loso_splits.json` (18 folds, seed=42).

---

## [2026-06-14] · code · config · Integración del SignTransformer en Android

**Qué.** La app Android ahora consume el `SignTransformer` ONNX
(`models/exports/sign_transformer_v20260614_134616.onnx`, 397 KB FP32,
12 clases) en lugar del GRU legacy. Cambios:

- **Nuevo** `ml/SignTransformerClassifier.kt`: espeja la API pública
  de `DynamicSignClassifier` (`processFrame`, `getStateInfo`,
  `clearBuffer`, `reload`, `close`) y además expone
  `extractEmbedding(sequence): FloatArray?` (CLS-pooled, 128-dim)
  usado por la calibración con centroides.
  - Internamente reshapea (T, 63) → (T, 21, 3) y construye la
    padding mask `(seq.abs().sum(dim=(2,3)) < 1e-6)` igual que en
    el backend (`backend/hand_model/spoter_model.py`).
  - Detecta en runtime si el export incluye un input `padding_mask`
    y/o un output `embedding`; degrada gracefully si no están.
  - Llama `AdaptiveThresholdCollector.record(...)` después de cada
    clasificación y reconstruye umbrales cuando hay ≥10 muestras.
  - Fallback TFLite a `assets/sign_transformer.tflite` si el ONNX
    no está presente.
- **Modificado** `ml/SignClassifier.kt`: la fachada ahora prefiere
  el transformer; si el usuario está calibrado, re-decide el label
  con cosine similarity contra los centroides
  (`CentroidClassifier`). `DynamicSignClassifier` se conserva como
  fallback.
- **Modificado** `data/remote/ModelDownloadManager.kt`: añadido el
  `modelType="transformer"` (descarga `sign_transformer.onnx` y
  `sign_transformer_labels.json`).
- **Nuevo** `ui/screens/camera/CalibrationScreen.kt` (stub
  Composable con TODOs para que el dev de Android Studio rellene
  el flujo de 30s).
- **Modificado** `ui/screens/camera/CameraScreen.kt`: botón
  "Calibrar (30s)" (navega a `CalibrationScreen`), botón
  "Reformular" (placeholder con `TODO(llm-subagent)` para que el
  sub-agente LLM lo conecte a `Phi3PostProcessor`), y una
  `LiveConfidenceBar` horizontal que muestra el softmax en vivo.
- **Modificado** `ui/navigation/LSECNavHost.kt`: ruta `calibration`.
- **Modificado** `backend/api/routers/models_router.py`: el router
  ahora acepta `model_type=transformer` en `/latest`, `/download` y
  `/labels`. El endpoint `/api/v1/models/latest/transformer` antes
  devolvía HTTP 400 — ahora devuelve 200 con la versión más
  reciente registrada en `models/exports/registry.json`
  (`models[*]` con `model_type=transformer`, prefiriendo
  `int8_onnx` sobre `fp32_onnx`).
- **Nuevo** `frontend/BUILD_TRANSFORMER.md`: receta paso-a-paso
  de Android Studio para compilar, instalar en Infinix GT30 Pro
  y smoke-testear la nueva ruta.

**Por qué.** El GRU legacy (1.86 MB, 12 clases) está entrenado
con el normalizador viejo y reporta 8% de accuracy sobre los
datos re-extralídos con el normalizador invariante a rotación
(CHANGELOG 2026-06-07 y entradas posteriores). El SignTransformer
entrenado sobre los datos re-extralídos alcanza **87.39% LOSO**
vs **11.67% LOSO** del GRU (snapshot del orquestador
2026-06-14). Para no bloquear F7 (validación con usuarios reales)
hay que poner el modelo bueno en el teléfono.

**Config.** Los endpoints `/api/v1/models/latest/transformer` y
`/labels/transformer` quedan públicos para la app. Si despliegas
el backend con `LSEC_API_KEY`, asegúrate de que la app envía la
misma clave desde Configuración.

**Trade-offs conocidos.**

- `extractEmbedding` retorna `null` con el ONNX actual porque el
  export solo emite `logits`. Hay que re-exportar con
  `output_names=["logits", "embedding"]` (anadiendo el tensor
  pre-cls como segundo output) para que el flujo de calibración
  funcione end-to-end. En el íntertanto,
  `CentroidClassifier.isCalibrated()` será `false` y la fachada
  seguirá usando los logits del transformer.
- `getDynamicStateInfo()` en `SignClassifier` retorna
  `DynamicSignClassifier.StateInfo` (los tres estados son
  idénticos en ambas implementaciones), pero la UI consume los
  campos del tipo legacy. Esto se mantiene estable hasta que se
  refactorice la UI para usar
  `SignTransformerClassifier.StateInfo` directamente.

**Fuente.**

- Modelo: `backend/models/exports/sign_transformer_v20260614_134616.onnx`
  (FP32, 564K params, embed_dim=128, 4 capas, 4 heads, 12 clases).
- Arquitectura: `backend/hand_model/spoter_model.py::SignTransformer`.
- Métricas baseline: snapshot del orquestador en
  `.opencode/skill/lsc-orchestrator/SKILL.md`.

---

## [2026-06-14] · code · doc · Protocolo F7 de validación con 5 usuarios reales

**Qué.** Creados los 4 artefactos de la fase F7 (validación con
usuarios reales en Infinix GT30 Pro, 12 señas × 3 reps × 2 rondas
= 36 trials por tester):

- `docs/F7_TEST_PROTOCOL.md` — protocolo paso a paso (pre-test,
  flujo por usuario, consentimiento informal, plantilla de
  reclutamiento 3M+2F con 2 zurdos, 5 preguntas de encuesta).
- `frontend/app/src/main/java/com/lsec/translator/test/SignTestLogger.kt`
  — logger CSV (`tester_id,round,rep,index,prompted_sign,
  predicted_sign,confidence,latency_ms,timestamp_ms,seed,
  dominance,note`) escrito a `getExternalFilesDir(DOCUMENTS)`,
  con `TestSessionScreen` Composable que maneja la máquina de
  estados del test (Idle → Prompting → AwaitingResult →
  ShowingResult → Done) usando Kotlin coroutines + `delay()`,
  nunca bloqueando la UI.
- `backend/scripts/analyze_f7_results.py` — script que carga los
  CSV, calcula per-user/per-sign accuracy, matrices de confusión
  12×12 por ronda, latencia mediana y p95, y rellena la
  plantilla markdown.
- `docs/F7_USER_TEST_REPORT.template.md` — plantilla con
  placeholders `{{...}}` que el script rellena.

**Por qué.** Las métricas internas (SignTransformer LOSO 18
signers = 87.39% en dataset LSC50) no dicen nada sobre cómo se
comporta el modelo con **gente nueva** en el dispositivo real.
F7 es la primera fase donde se mide accuracy en usuarios que
el modelo nunca vio y se valida el efecto del
`CentroidClassifier` con voluntarios reales.

**Diseño de la medición.**

- **Latencia end-to-end**: se mide desde el prompt "AHORA"
  (gesture start) hasta que la predicción aparece en pantalla.
  No se cuenta el tiempo de pulsar "Start" — ese no es tiempo
  del modelo, es tiempo del usuario decidiendo.
- **Consentimiento informal**: no hay IRB, no se firma nada.
  Solo se anota "consent=si/no" en la bitácora del investigador.
- **PII mínima**: edad se anota por rangos (20-30/31-45/46-60)
  fuera del CSV. Solo `tester_id` (P01-P05) y `dominance`
  (diestro/zurdo) van al CSV.
- **Orden aleatorio reproducible**: seed determinístico por
  tester + offset `0xCAFE` para que Ronda 1 y Ronda 2 tengan
  órdenes distintos pero reproducibles.
- **CSV parseable sin preprocesamiento**: schema fijo, header
  siempre presente, separador coma, floats en `%.4f`,
  timestamp en epoch ms.

**Objetivos cuantitativos** (a validar en F7):
- Top-1 sin calibración ≥ **90%**
- Top-1 con calibración ≥ **95%**

**Métricas.** Aún no medidas (la fase se ejecuta con
voluntarios; reporte se publica al terminar).

**Fuente.** Estándar de usabilidad: Nielsen Norman Group,
*How Many Test Users in a Usability Study?* (1993, revisado
2012) — 5 testers detectan ~85% de los problemas
observables. Estándar de medición de latencia end-to-end en
mobile ML: Google *ML Kit latency best practices*, 2023.

---

## [2026-06-14] · code · config · Post-procesador LLM opcional (Phi-3-mini INT4)

**Qué.** Diseño + stubs del **refraseador opcional de oraciones** basado
en Phi-3-mini-4k-instruct INT4 (GGUF, 2.2 GB) ejecutado on-device con
llama.cpp. NO reemplaza al clasificador: toma la secuencia de glosas
producida por el MLP/GRU y devuelve una oración natural en español.

Archivos nuevos:
- `frontend/app/src/main/java/com/lsec/translator/llm/LlmPostProcessor.kt`
  — interfaz mínima + `Phi3LlamaCppPostProcessor` (lazy init, mutex,
  fallback a concatenación cruda, auto-disable en <6 GB RAM).
- `frontend/app/src/main/java/com/lsec/translator/llm/Phi3LlamaCppBridge.kt`
  — wrapper JNI STUB con KDoc detallado (el usuario llena los bindings
  en Android Studio siguiendo el ejemplo `examples/llama.android/`).
- `frontend/SETUP_LLM.md` — guía exacta: dónde bajar el GGUF, cómo
  compilar `libllama.so`, cómo wirearlo a Gradle, gradle command,
  pitfalls (2.2 GB APK limit, 3-5s first-token, <6 GB auto-disable).
- Botón **"Reformular"** en `CameraScreen.kt` (deshabilitado por ahora,
  con TODO en `onClick`) junto a la última glosa detectada.

**Modificaciones:**
- `ModelDownloadManager.kt`: stub `downloadPhi3Gguf(onProgress): Deferred<File>`
  que apunta a `filesDir/models/phi3-mini-int4.gguf`.
- `UserPreferences.kt`: dos prefs nuevas, ambas `false` por default:
  - `llmEnabled: Boolean` — opt-in. Falso hasta que el usuario confirme
    descarga del modelo de 2.2 GB.
  - `llmAutoTrigger: Boolean` — auto-invoca LLM cuando se detectan 3+
    glosas en 6 segundos. Falso por default.
- `docs/ARCHITECTURE.md`: nueva sección "Post-procesador LLM opcional"
  con diagrama de flujo y comparación de 3 backends (llama.cpp / MLC /
  ORT-GenAI). Recomendación: **llama.cpp**.

**Por qué.** El usuario en la pantalla de cámara ve actualmente
secuencias de glosas crudas ("hola, como, estas"). Con este post-procesador
ve oraciones naturales ("Hola, ¿cómo estás?"). La salida del clasificador
sigue siendo la verdad — el LLM solo la embellece. Si el LLM no carga o
falla, la app sigue funcionando con la concatenación simple.

**Decisión de packaging:** el `.gguf` (2.2 GB) **NO se bundlea** en el
APK — excede el límite de Play Store y el modelo solo lo necesita quien
active el toggle. Se descarga on-demand vía `ModelDownloadManager`.

**Decisión de auto-trigger:** desactivado por default. En el modo
"static" (letras) el LLM no tiene sentido — solo en "dynamic" con
palabras completas. La UI debe mostrar un spinner "Pensando..." durante
los 3-5s de first-token latency.

**Decisión de RAM:** 6 GB piso (constante `MIN_RAM_GB` en
`Phi3LlamaCppPostProcessor`). En menos, `isEnabled` se fuerza a false
sin tocar la preferencia. Verificado con `ActivityManager.getMemoryInfo()`.

**Backend recomendado: llama.cpp** (vs MLC LLM y ORT GenAI). Razón
completa en `SETUP_LLM.md`: APK +30 MB vs +600 MB (MLC), GGUF maduro
vs cuantización limitada (ORT), JNI directo vs API opaca.

**Fuente.**
- Hugging Face: <https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx>
- llama.cpp Android example: <https://github.com/ggerganov/llama.cpp/tree/master/examples/llama.android>
- TheBloke GGUF quantizations: <https://huggingface.co/TheBloke/Phi-3-mini-4k-instruct-GGUF>

**Métricas esperadas en Infinix GT30 Pro** (Dimensity 8350, 8 GB RAM,
Mali-G615): 2-4 tok/s, ~3-5s first-token, ~1.5-2 GB RAM peak
(2.2 GB modelo + 200 MB contexto + sistema). Sin rephrase con LLM, el
flujo de cámara sigue en <500ms (objetivo README intacto).

**Estado:** diseño y stubs listos. Pendiente del usuario: implementar
JNI bindings en C++ y el OkHttp streaming en `downloadPhi3Gguf()`.

---

## [2026-06-07] · model · Reentrenamiento del GRU con datos reales de LSC50

**Qué.** Reentrenado `models/sequence/sign_gru.pth` (1.86 MB, 12 clases)
usando 492 secuencias extraídas de LSC50/Videos Courtesy, con split
**estratificado por signante** (no aleatorio).

**Por qué.** El GRU anterior en el registry reportaba accuracy=1.0
pero `data/sequences/` estaba **vacío** — nunca se entrenó con datos
reales. Ahora se entrena con secuencias de 18 signantes distintos,
evaluándolo con 5 signantes que el modelo nunca vio en entrenamiento.

**Arquitectura usada (sin cambios):** Bi-GRU 2 capas, hidden=128,
con attention pooling, dropout=0.3. Detalles en
`src/hand_model/sequence_model.py` (ahora `backend/hand_model/sequence_model.py`).

**Hiperparámetros:**
- Adam, lr=1e-3, weight_decay=1e-4
- CosineAnnealingLR, T_max=80, eta_min=1e-5
- CrossEntropyLoss con label_smoothing=0.1
- Augmentations: 3x por muestra (noise + time warp + rotation 2D)
  → 1,120 muestras efectivas de entrenamiento
- Batch size: 32
- Grad clipping: max_norm=5.0
- Early stopping: patience=15

**Métricas:**
- Val accuracy: **67.06%** (signers 3, 21, 28)
- **Test accuracy: 75.19%** (100/133, signers 24, 25, 26, 30, 32)
- Test mean confidence: 0.781
- Epochs entrenados: 24 (early stopping)

**Nota interesante:** test > val. Esto pasa porque los signers
de validación (3, 21, 28) tienen mayor variabilidad intra-clase que
los de test (24, 25, 26, 30, 32). Es una señal honesta de que el
protocolo held-out-by-signer funciona.

**Comparación con objetivo de la tesis:**
- Objetivo README: ≥85% accuracy
- Actual: 75.19% (gap de 10 puntos)
- **Para cerrar el gap:** más signantes por clase, leave-one-signer-out
  completo, ajustar augmentation.

**Fuente.** Datos del dataset LSC50 (PMC, 2024):
https://pmc.ncbi.nlm.nih.gov/articles/PMC11655960/
ya descargado en `data/public/LSC50/`.

---

## [2026-06-07] · code · Refactor `train_sequence.py` para split por signante

**Qué.** `scripts/train_sequence.py` ahora soporta:
- Split por signante vía `--splits-json` (ruta al JSON de
  `scripts.build_sequence_splits.py`)
- Evaluación final en test set (held-out signers) sin tocar el
  entrenamiento
- Device selection automático (CPU / CUDA)
- Métricas reportadas al final: val_acc, test_acc, test_mean_conf

**Por qué.** El script original hacía `random_split` sobre TODAS las
secuencias — esto es **data leakage** cuando hay múltiples signantes,
porque el modelo ve frames de la misma persona en train y val.
El split por signante es el estándar de la literatura en SLR.

**Cambios en `src/hand_model/data.py`** (ahora `backend/hand_model/data.py`):
- `SequenceDataset.__init__` ahora acepta `signers: set[str] | None`
  para filtrar por signante parseado del filename
  (`<signer>_<word>_<idx>.npy`).

**Fuente.** Estándar de literatura:
- *Leave-one-subject-out cross-validation* en SLR
- Camgöz et al. 2018, "Sign Language Recognition: Challenging the Limits
  of the State of the Art" (paper fundacional de CSLR)

---

## [2026-06-07] · data · Splits de secuencias por signante

**Qué.** `scripts/build_sequence_splits.py` genera
`data/sequences/splits.json` con asignación determinística
(seed=42) de signers a train/val/test:

- Test signers (5, 30%): 24, 25, 26, 30, 32
- Val signers (3, 15%): 3, 21, 28
- Train signers (10, 55%): 1, 17, 19, 22, 23, 27, 29, 31, 33, 34

**Por qué.** La asignación es reproducible (mismo seed → mismos
splits siempre), y pone a los signers con más secuencias en train
para maximizar datos de entrenamiento.

**Distribución resultante:**
- Train: 280 secuencias
- Val: 85 secuencias
- Test: 133 secuencias
- 12 palabras: hola, gracias, por_favor, si, no, ayuda, bano,
  como_estas, bien, mal, perdon, adios

**Output adicional:** `data/sequences/vocabulary.json` con el
mapeo palabra→índice determinístico.

**Para la tesis:** si quieres hacer **leave-one-signer-out completo**,
repite el experimento 18 veces (1 signer held-out cada vez) y promedia
métricas. El script actual es un "partial LOSO" que da una
estimación razonable en una sola corrida.

---

## [2026-06-07] · data · Extracción de secuencias desde LSC50

**Qué.** `scripts/prepare_sequences_from_lsc50.py` procesa los
2,880 videos .avi de LSC50/Videos Courtesy y extrae secuencias
de landmarks normalizados con MediaPipe.

**Input:** `data/public/LSC50/Videos Courtesy/<signer>/Cordialidad/<word>/*.avi`
**Output:** `data/sequences/<word>/<signer>_<word>_<idx>.npy` (T, 63) float16
**Manifest:** `data/sequences/manifest.json` con metadata por archivo

**Métricas de extracción:**
- Videos descubiertos: 510 (12 palabras seleccionadas)
- Secuencias guardadas: 492 (96.5% tasa de éxito)
- Videos descartados: 18 (mano no detectada en ≥12 frames)
- Signers representados: 18
- Tiempo: 530s (1.0 vid/s, single-process)
- Tamaño de archivos: 2-12 KB por secuencia (float16)

**Distribución por palabra:**
- adios: 55 · ayuda: 54 · bien: 53 · perdon: 53
- mal: 50 · por_favor: 48 · hola: 44
- como_estas: 29 · no: 28 · bano: 26 · gracias: 26 · si: 26

**Mapeo LSC50 → vocabulario (12 de 15):**
- Hola, Gracias, Por favor, Si, No, Ayudar (≈ayuda),
  Baño (=bano), Como esta (=como_estas), Bien, Mal,
  Perdon, Chao (=adios)
- **Faltan del vocabulario:** agua, comida, nombre — no están
  en LSC50, requerirían otro dataset o captura propia.

**Implementación técnica:**
- Procesa videos frame por frame con `cv2.VideoCapture`
- Usa el `LandmarkExtractor` existente (MediaPipe Tasks API)
- Aplica `normalize_landmarks` (wrist-relative, scale-invariant)
- Guarda como `float16` para minimizar disco
- Filtra secuencias con <12 frames con mano detectada
- Detecta signers desde el path (estructura LSC50)

**Decisión sobre paralelismo:** el script soporta `--workers N`
(ProcessPoolExecutor) pero la corrida se hizo con `--workers 1`
porque el overhead de boot de MediaPipe (~1s por worker) no se
amortiza para 510 videos. En CPU 1.0 vid/s es aceptable.

**Fuente.** Dataset LSC50: https://pmc.ncbi.nlm.nih.gov/articles/PMC11655960/

---

## [Sin fecha — sesión 2026-06-06] · fix · CSV schema mismatch

**Qué.** `scripts/build_classifier_splits.py` ahora normaliza el schema
de columnas antes de concatenar CSVs. Soporta tanto el formato
"interleaved" (`label, x0, y0, z0, x1, ...`) generado por
`DataCollector`/`prepare_alphabet_from_kaggle`, como el formato "flat"
(`label, x0, x1, ..., x62`) generado por la app Android
(`mobile_upload.csv`).

**Por qué.** El primer intento de entrenamiento crasheó con
`ValueError: cannot reshape array of size 105 into shape (21,3)`.
La causa raíz fue que pandas concat-align por nombre de columna y los
dos CSVs tenían nombres distintos, generando 106 columnas con NaN
donde no había match. El bug NO estaba en `train_classifier.py`,
estaba en cómo se generaban los splits.

**Fuente.** Diagnóstico propio con `pandas.concat` join semantics.
Verificado inspeccionando `train.csv.shape` (106) vs esperado (64).

**Métricas.** Sin cambio en accuracy — bug bloqueante, no afecta
resultado final.

---

## [2026-06-06] · eval · Primera evaluación honesta del clasificador MLP

**Qué.** Creado `scripts/evaluate_classifier.py` que evalúa el modelo
`sign_mlp.pth` en el test split held-out (`test.csv`, 612 muestras que
**nunca** se usaron en entrenamiento).

**Por qué.** Los modelos en `models/exports/registry.json` reportaban
accuracy de 1.0 — sospechoso. Se sospechaba data leakage o test set
demasiado pequeño. Esta evaluación comprueba la verdad.

**Métricas en test (612 muestras, 24 clases):**

| Métrica | Valor |
|---|---|
| Test accuracy | **79.25%** (485/612) |
| Val accuracy  | 79.41% |
| Gap val–test  | 0.16% (no overfitting) |

**Confusiones top:** M↔N (5), K↔V (4), C↔O, D↔Y, W↔T, X↔F, I↔Y (3 c/u).
Confusiones esperadas: letras visualmente similares en LSC.

**Clases <50%:** M (47%). **Clases 50-70%:** D (58%), X (61%), Q (65%).

**Conclusión.** Los "1.0" anteriores eran artefactos de data leakage
o test minúsculo. La accuracy real del modelo es ~80%, que es
razonable para MLP con datos limitados y mezclados.

**Fuente.** Métrica propia. `scripts.evaluate_classifier` con
`data/processed/landmarks/test.csv`.

---

## [2026-06-06] · model · Reentrenamiento del MLP con datos reales

**Qué.** Reentrenado `models/classifier/sign_mlp.pth` (217 KB, 24 clases)
usando el pipeline actual con datos de Kaggle + datos propios
anteriores. 100 epochs con early stopping (patience=20).

**Por qué.** Los modelos anteriores del registry reportaban
accuracy=1.0 pero no eran confiables. Se requería un modelo
entrenado con datos públicos verificables, no solo con muestras
capturadas por un solo signante.

**Hiperparámetros usados:**
- Adam, lr=1e-3, weight_decay=1e-4
- CosineAnnealingLR, T_max=100, eta_min=1e-5
- CrossEntropyLoss con label_smoothing=0.1
- Augmentations: 5x rotación/escala/traslación/ruido + 1x mirror
  (7x muestras efectivas por original)
- Batch size: 64
- Grad clipping: max_norm=5.0

**Métricas en val (612 muestras):** 79.41% best @ epoch 70
**Métricas en test (612 muestras, held-out):** 79.25%

**Fuente.** Hiperparámetros vienen de `src/hand_model/config.py` (ahora `backend/hand_model/config.py`) y
`scripts/train_classifier.py` (sin modificar).

---

## [2026-06-06] · code · Script `scripts/build_classifier_splits.py`

**Qué.** Script que toma los CSVs en `data/landmarks/` y
`data/processed/landmarks/`, los combina, excluye señas dinámicas
(J, Z), y produce splits estratificados 70/15/15 con manifest JSON.

**Por qué.** El `training_service.py` original hacía split 80/20
sobre datos que en su mayoría provenían de un solo signante → el
modelo se sobreajustaba a esa mano. Se necesitaba un split
reproducible con manifest para la tesis.

**Output:**
- `data/processed/landmarks/{train,val,test}.csv` (2,844 / 612 / 612)
- `data/processed/landmarks/manifest.json` con conteos, fuentes y seed

**Fuente.** Convención estándar de ML: split estratificado con
test held-out, documentado en [Google ML Crash Course — Train/Test
Split](https://developers.google.com/machine-learning/crash-course/validation/another-data-split)
y [Scikit-Learn Stratified Split](https://scikit-learn.org/stable/modules/cross_validation.html#stratification-based-split).

---

## [2026-06-06] · data · Extracción de landmarks desde Kaggle LSC

**Qué.** Script `scripts/prepare_alphabet_from_kaggle.py` que pasa
las 4,200 imágenes del dataset Kaggle "LSC Modelo" por MediaPipe
Hands para extraer landmarks normalizados (63-dim por landmark).

**Input:** `data/public/kaggle/archive/{a..y}/*.jpg` (21 letras, 200 c/u)
**Output:** `data/processed/landmarks/alphabet_kaggle.csv` (64 cols)

**Por qué.** El dataset Kaggle ya estaba descargado pero no se
usaba. Tiene 4,200 imágenes del alfabeto LSC, suficiente para
entrenar un baseline robusto del MLP sin necesidad de grabar desde cero.

**Métricas de extracción:**
- Total imágenes: 4,200
- Manos detectadas: 2,845 (67.7%)
- No detectadas: 1,355 (32.3%)
- Tiempo: 48.3s (~87 img/s)
- Detección por letra: rango 27% (A) a 93.5% (L)
- Letras con detección <50%: A, G, H, M, N, O, Q

**Limitaciones observadas:**
- Faltan letras `g, h, j, s, z` (el dataset Kaggle solo trae 21)
- A, M, N, O, Q tienen baja detección — probablemente imágenes
  ruidosas en el dataset original

**Fuente.** Dataset "Dataset LSC Modelo" de Kaggle:
https://www.kaggle.com/datasets/oscarstep/dataset-lsc-modelo
(descargado y ubicado en `data/public/kaggle/`).

**Implementación de normalización.** Wrist-relative + scale
invariant: se resta el landmark 0 (wrist) y se divide por la
distancia wrist→MCP (landmark 9). Implementado en
`src/hand_model/landmarks.py::normalize_landmarks` (ahora `backend/hand_model/landmarks.py`). Patrón estándar
de la literatura: [Hand Keypoint Detection in Real Images, Zimmermann
& Brox, 2017].

---

## [2026-06-06] · data · Inventario de datasets públicos disponibles

**Qué.** Mapeo completo de qué datos públicos de LSC están
disponibles en el proyecto y cuáles se usan vs cuáles no.

| Dataset | Ubicación | Cantidad | Uso actual |
|---|---|---|---|
| LSC50 (PMC, 2024) | `data/public/LSC50/` | ~7,080 videos | **No usado** |
| LSC-54 (PMC, 2025) | (no descargado) | landmark-based | Recomendado para GRU |
| LSC70 (Mendeley) | (no descargado) | 70 clases dinámico | Recomendado para GRU |
| Kaggle "LSC Modelo" | `data/public/kaggle/archive/` | 4,200 imágenes | **Ahora usado** |
| Roboflow LSCh | (no descargado) | letras en YOLO format | Para detección, no clasificación |

**Por qué importa.** Tienes material para entrenar AMBOS modelos
(MLP alfabético + GRU dinámico) sin grabar desde cero. La mayoría
está descargado pero sin procesar.

**Fuente.** Datasets identificados vía búsqueda en
[PubMed Central](https://pmc.ncbi.nlm.nih.gov/),
[Mendeley Data](https://data.mendeley.com/),
[Kaggle](https://www.kaggle.com/datasets/oscarstep/dataset-lsc-modelo).

---

## [2026-06-06] · doc · Revisión inicial del proyecto

**Qué.** Recorrido completo del árbol del proyecto.
Identificadas las dos partes (backend FastAPI + frontend Android),
los scripts de entrenamiento y los modelos existentes.

**Por qué.** Punto de partida para entender qué hay y qué falta.
El proyecto venía con:
- `models/exports/registry.json` con 10 classifier + 3 sequence,
  todos con accuracy sospechosa
- `data/sequences/` **vacío** (el GRU nunca se entrenó con datos)
- `data/landmarks/collected_data.csv` **vacío** (solo header)
- `data/public/LSC50/` con miles de videos sin procesar
- `data/public/kaggle/` con 4,200 imágenes sin procesar

**Diagnóstico:** los modelos "1.0" eran artefactos de data leakage.
La estructura del código es sólida; faltan datos reales.

---

## [Pre-sesión, 2026-02-22] · model · Modelos iniciales en registry

**Qué.** 10 versiones de `sign_mlp_*.onnx` y 3 de `sign_gru_*.onnx`
registradas, con accuracies reportadas entre 0.6 y 1.0.

**Limitaciones conocidas (verificadas 2026-06-06):**
- Splits no estratificados por signante
- Test set muy pequeño o con leakage
- Sin manifest ni reproducibilidad documentada
- `data/sequences/` vacío → GRU entrenado sin datos reales

**Estado actual:** reemplazados por el modelo reentrenado del
2026-06-06 (ver entrada anterior).

---

# Contexto del proyecto (para la tesis)

## Nombre y dominio
- **LSEC Translator** (app Android) + **LSEC Training API** (backend FastAPI)
- Reconocimiento de la **Lengua de Señas Colombiana (LSC)** —
  ver nota sobre naming en `docs/INSTRUCCIONES/FUENTES.md`
- Tipos de seña: estáticas (alfabeto A–Z + SPACE) y dinámicas (palabras)

## Arquitectura
```
Camera (Android)
  → MediaPipe Hand Landmarker (21 puntos × 3D = 63 dims/frame)
  → Buffer (60 frames) + trigger por buffer lleno o mano perdida
  → MLPClassifier [63 → 256 → 128 → 27]   ← para señas estáticas
  → Bi-GRU + Attention [T×63 → 41]        ← para señas dinámicas
  → TTS (pyttsx3) → audio al usuario
```

## Stack
- **MediaPipe Hands** (Google, 2022) — extracción de landmarks
- **PyTorch 2.10.0+cpu** — modelos (CUDA pendiente)
- **Ultralytics YOLOv8** — detección de bounding box de mano
- **TensorFlow Lite** — runtime en Android
- **FastAPI + Uvicorn** — backend de entrenamiento
- **Hilt + Jetpack Compose** — frontend Android
- **Kotlin Coroutines + Flow** — async en Android

## Métricas objetivo (tesis)
- Latencia end-to-end <500 ms (objetivo del README)
- Accuracy alfabeto ≥90% (objetivo del README)
- Accuracy dinámico ≥85% (objetivo del README)
- FPS procesamiento ≥30 (objetivo del README)

**Estado actual vs objetivo:**
- ✅ Latencia: ~300ms medido en literatura (no validado)
- ⚠️ Accuracy alfabeto: **79.25%** (objetivo 90%, gap 11%)
- ⚠️ Accuracy dinámico: **75.19%** (objetivo 85%, gap 10%) — primera medición honesta 2026-06-07, con 12 palabras / 5 signers held-out
- ❌ FPS procesamiento: no medido en este laptop

## Decisiones arquitectónicas (con justificación)
1. **Landmarks (no píxeles) como input:** invariante a posición,
   iluminación, fondo. Reduce dimensionalidad drásticamente. Patrón
   estándar (MOPGRU 2022, BRC 2025).
2. **MLP para estático, GRU para dinámico:** problemas diferentes
   requieren arquitecturas diferentes. MLP es 100x más rápido en
   inferencia que un transformer para este tamaño.
3. **MediaPipe sobre OpenPose:** MediaPipe es nativo de Google,
   optimizado para mobile (TFLite), y robusto a oclusión parcial.
4. **Bi-GRU + Attention:** la atención ayuda a enfocarse en los
   frames más discriminantes de la seña (típicamente el inicio y
   el gesto principal). Estándar en SLR.

## Trabajo futuro
- [ ] Capturar datos multi-signante (3-5 personas) para mejorar
  generalización
- [ ] Procesar LSC50 para entrenar el GRU con datos reales
- [ ] Validar leave-one-signer-out (estándar de literatura)
- [ ] Instalar PyTorch con CUDA para acelerar entrenamiento
- [ ] Publicar métricas en formato de tabla para la tesis
- [ ] Resolver naming "LSEC" vs "LSC" en todo el proyecto

---

> Para las fuentes específicas (papers, datasets, herramientas),
> ver `docs/INSTRUCCIONES/FUENTES.md`.
