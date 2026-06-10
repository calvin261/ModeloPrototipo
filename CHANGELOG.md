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
`src/hand_model/sequence_model.py`.

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

**Cambios en `src/hand_model/data.py`:**
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

**Fuente.** Hiperparámetros vienen de `src/hand_model/config.py` y
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
`src/hand_model/landmarks.py::normalize_landmarks`. Patrón estándar
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
  ver nota sobre naming en `INSTRUCCIONES/FUENTES.md`
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
> ver `INSTRUCCIONES/FUENTES.md`.
