# Instrucciones para el Modelo (Python) — Detección de Manos

Este paquete documenta, paso a paso, cómo levantar el proyecto de **entrenamiento y exportación** del modelo de detección de manos en **Python**, con enfoque en rendimiento y buenas prácticas. Está pensado para que otro agente de IA pueda seguirlo sin contexto adicional.

## Alcance

- Entrenamiento de **modelos de detección de manos** (YOLOv8/Ultralytics) y preparación de modelos auxiliares.
- Descarga y uso de **modelos preentrenados** (YOLOv8n/s y MediaPipe Hand Landmarker).
- Preparación de datasets y reentrenamiento con nuevas imágenes.
- Exportación a formatos de despliegue (ONNX/TFLite) optimizados.

## Archivos clave

- `GUIA_PASO_A_PASO.md`: guía completa de setup, datos, entrenamiento y exportación.
- `FUENTES.md`: fuentes web consultadas (buenas prácticas y guías oficiales).

## Principios de diseño (resumen)

- Estructura tipo **src layout** y `pyproject.toml`.
- Código legible (PEP 8), docstrings claras (PEP 257).
- Entrenamiento reproducible (semillas, configs, logging).
- Optimización de recursos (batch auto, imágenes 640, mixed precision, export INT8).
