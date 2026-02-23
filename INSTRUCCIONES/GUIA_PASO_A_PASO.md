# Guía Paso a Paso — Proyecto Python para Detección de Manos

> Objetivo: crear y entrenar **múltiples modelos de detección de manos** en Python, partiendo de modelos preentrenados, añadiendo nuevas imágenes y exportando a formatos optimizados.

## 1) Estructura recomendada del proyecto (clean code)

```
ModeloPrototipo/
├── pyproject.toml
├── README.md
├── .gitignore
├── data/
│   ├── raw/                  # Imágenes originales
│   ├── interim/              # Datos intermedios (limpios, sin etiquetas)
│   ├── processed/            # Dataset en formato YOLO
│   └── splits/               # train/val/test
├── datasets/
│   ├── hands.yaml            # Config YOLO (rutas y clases)
│   └── labels_map.json       # Mapeo de clases
├── models/
│   ├── pretrained/           # Pesos preentrenados
│   ├── finetuned/            # Pesos reentrenados
│   ├── exports/              # ONNX / TFLite
│   └── mediapipe/            # Modelos .task
├── notebooks/                # Opcional (solo exploración rápida)
├── scripts/
│   ├── prepare_dataset.py
│   ├── train_yolo.py
│   ├── evaluate.py
│   └── export.py
├── src/
│   └── hand_model/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── train.py
│       ├── export.py
│       └── utils.py
└── tests/
    └── test_data.py
```

## 2) Entorno Python (recomendado)

- Python 3.10+.
- Entorno virtual aislado.
- Dependencias declaradas en `pyproject.toml`.

Dependencias mínimas sugeridas:

- `ultralytics` (YOLOv8)
- `opencv-python` (o `opencv-python-headless` en servidor)
- `numpy`, `pandas`
- `torch`, `torchvision`
- `pyyaml`
- `mediapipe` (si se usará Landmarker en Python)

## 3) Descarga de modelos preentrenados

### A. YOLOv8 (detección)

Modelos recomendados (calidad vs velocidad):

- `yolov8n.pt` (más rápido, menor tamaño)
- `yolov8s.pt` (mejor precisión, aún ligero)

Se descargarán automáticamente al llamar a `YOLO("yolov8n.pt")`.
Guárdalos en `models/pretrained/`.

### B. MediaPipe Hand Landmarker (landmarks)

Descarga el **model bundle** oficial (archivo `.task`) y colócalo en:

```
models/mediapipe/hand_landmarker.task
```

Esto permite extraer 21 landmarks por mano en Python si se necesita pipeline híbrido.

## 4) Preparación del dataset (YOLO format)

Estructura en `data/processed/`:

```
data/processed/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Formato de labels (`.txt` por imagen):

```
<class_id> <x_center> <y_center> <width> <height>
```

- Coordenadas normalizadas (0–1).
- Una sola clase recomendada: `hand`.

`datasets/hands.yaml`:

```
path: data/processed
train: images/train
val: images/val
test: images/test
names:
  0: hand
```

## 5) Entrenar múltiples modelos

Define una lista de modelos base (n/s) y reentrena con el mismo dataset para comparar:

- Modelo A: `yolov8n.pt`
- Modelo B: `yolov8s.pt`

Guarda los resultados en `models/finetuned/` y versiona con fecha.

### Buenas prácticas

- Entrena con `imgsz=640` (balance velocidad/precisión).
- Usa `batch=-1` para auto batch según GPU.
- Habilita `amp=True` para mixed precision.
- Fija `seed` para reproducibilidad.

## 6) Exportación para despliegue

Exporta los mejores pesos a:

- **ONNX** (para validación cruzada)
- **TFLite** (para Android)

Parámetros clave:

- `int8=True` para dispositivos móviles.
- `imgsz=640` consistente con entrenamiento.

Guarda en `models/exports/`.

## 7) Optimización de recursos (máximo rendimiento)

- Usa `yolov8n` para inferencia en tiempo real.
- Reduce resolución si no se alcanza latencia objetivo.
- Activa `cache=ram` solo si hay suficiente memoria.
- Exporta TFLite con INT8 y usa datos representativos.
- Controla el tamaño del dataset (balance calidad vs costo).

## 8) Reentrenar con nuevas imágenes

Flujo recomendado:

1. Añadir imágenes nuevas a `data/raw/`.
2. Etiquetar (Roboflow/LabelImg o similar).
3. Convertir y mover a `data/processed/`.
4. Actualizar `datasets/hands.yaml`.
5. Reentrenar y comparar métricas.

## 9) Checklist de calidad

- [ ] Dataset balanceado (variación de iluminación, fondo, tono de piel).
- [ ] Labels consistentes.
- [ ] Métricas mAP evaluadas en `val`.
- [ ] Exportación TFLite verificada.
- [ ] Guardado de modelos y configs versionados.

## 10) Recomendaciones de clean code

- Mantener funciones pequeñas (1 responsabilidad).
- Evitar lógica en scripts, mover a `src/`.
- Configuración central en `config.py`.
- Tests básicos para carga de dataset.

---

Si deseas, puedo crear los archivos base del proyecto Python (`pyproject.toml`, scripts y estructura) automáticamente.
