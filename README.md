# ModeloPrototipo

Proyecto unificado para LSEC: backend Python/FastAPI para entrenamiento y gestion de modelos, mas frontend Android/Kotlin para captura, entrenamiento asistido y consumo de modelos.

## Estructura

```text
ModeloPrototipo/
|-- backend/      # API FastAPI
|-- frontend/     # App Android copiada desde Prototipo
|-- src/          # Paquete Python hand_model
|-- scripts/      # Scripts de entrenamiento, inferencia y exportacion
|-- data/         # Datos de entrenamiento
|-- models/       # Checkpoints y exports
|-- datasets/     # Labels y configuraciones
`-- tests/        # Tests del backend y servicios
```

## Documentacion

- Swagger backend: `http://localhost:8000/docs`
- Guia de integracion frontend/backend: `docs/INTEGRACION_FRONTEND_BACKEND.md`
- Documentacion del frontend Android: `frontend/README.md`

## Inicio rapido

1. Backend: instalar dependencias Python y ejecutar `python start_backend.py`.
2. Verificar API: abrir `http://localhost:8000/health` o `http://localhost:8000/docs`.
3. Frontend: abrir `frontend/` en Android Studio.
4. En la app Android, configurar la URL del servidor como `http://<IP-de-tu-PC>:8000/`.
5. Para emulador Android, usar normalmente `http://10.0.2.2:8000/`.
