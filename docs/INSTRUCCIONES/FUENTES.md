# Fuentes consultadas (buenas prácticas y guías oficiales)

## Python
- PEP 8 — Style Guide for Python Code: https://peps.python.org/pep-0008/
- PEP 257 — Docstring Conventions: https://peps.python.org/pep-0257/
- PEP 20 — The Zen of Python: https://peps.python.org/pep-0020/
- Packaging Python Projects: https://packaging.python.org/en/latest/tutorials/packaging-projects/
- Writing pyproject.toml: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/

## Modelos / CV
- Ultralytics YOLOv8 Overview: https://docs.ultralytics.com/models/yolov8/
- Ultralytics Quickstart: https://docs.ultralytics.com/quickstart/
- Ultralytics Train Mode: https://docs.ultralytics.com/modes/train/
- Ultralytics Export Mode: https://docs.ultralytics.com/modes/export/
- Ultralytics Datasets (detect): https://docs.ultralytics.com/datasets/detect/

## MediaPipe / Edge
- MediaPipe Hand Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- LiteRT Guide (TFLite): https://ai.google.dev/edge/litert/guide
- Keras RNNs Guide: https://www.tensorflow.org/guide/keras/working_with_rnns

## Datasets LSC (Lengua de Señas Colombiana)

- **LSC-54** (2025) — landmark-based dataset, 54 clases. PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12557509/
  *Relevancia para este proyecto: viene en formato CSV de landmarks normalizados, mismo formato que produce este pipeline. Ideal para combinar con datos propios.*
- **LSC70** — Dynamic LSC database, 70 clases (videos + imágenes). Mendeley: https://data.mendeley.com/datasets/9ssyn8tff5/2
  *Relevancia: cobertura amplia de señas dinámicas (palabras), útil para entrenar el GRU.*
- **LSC50** (2024) — Video + IMU, 50 clases, multi-signante. PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11655960/
  *Relevancia: ya está descargado en `data/public/LSC50/`. 7,080 videos, 34+ signantes. Estructura: `Videos <tipo>/<signante>/<categoria>/<palabra>/*.mp4`.*
- **Dataset LSC Modelo** (Kaggle) — alfabeto LSC, 21 letras, 200 imágenes por letra. https://www.kaggle.com/datasets/oscarstep/dataset-lsc-modelo
  *Relevancia: ya está descargado en `data/public/kaggle/archive/`. Usado en el reentrenamiento del 2026-06-06 (2,845/4,200 detecciones).*
- **Señas de LSCh** (Roboflow) — letras en formato YOLO. https://universe.roboflow.com/entrenamiento-lsch/senas-de-lsch
  *Relevancia: para detección de bounding box de mano (etapa previa al landmark), no para clasificación.*

## Papers y arquitectura

- **MediaPipe-optimized GRU (MOPGRU)** for Indian Sign Language. PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9279358/
  *Referencia directa para el modelo GRU+attention del proyecto.*
- **Bidirectional Reservoir Computing (BRC)** for sign language, 2025. arXiv: https://arxiv.org/abs/2512.00777
  *Alternativa más liviana que GRU; podría explorarse en trabajo futuro para reducir latencia en mobile.*
- **Hand Keypoint Detection in Real Images** (Zimmermann & Brox, 2017). Patrón de normalización wrist-relative + scale-invariant (mismo que `landmarks.py::normalize_landmarks`).
- **CosineAnnealingLR for training stability.** PyTorch docs: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
- **Label Smoothing** (Szegedy et al., 2016). Paper original: https://arxiv.org/abs/1512.00567
  *Justifica el `label_smoothing=0.1` en `train_classifier.py`.*

## Buenas prácticas de dataset / split

- **Stratified train/val/test split** — scikit-learn docs: https://scikit-learn.org/stable/modules/cross_validation.html#stratification-based-split
- **Leave-one-subject-out cross-validation** — estándar en reconocimiento de señas (importante para multi-signante).
- **Keep a Changelog** (formato de CHANGELOG): https://keepachangelog.com/

## Herramientas usadas

- **MediaPipe Tasks API** (Python): https://developers.google.com/mediapipe/solutions/tasks
- **PyTorch** (CPU/CUDA builds): https://pytorch.org/get-started/locally/
- **Ultralytics YOLOv8**: https://docs.ultralytics.com/models/yolov8/
