# Integracion Frontend Android + Backend FastAPI

Este repositorio contiene ambos proyectos en una sola carpeta:

- `backend/`: API FastAPI de entrenamiento, datos, modelos y scripts.
- `frontend/`: app Android/Kotlin copiada desde `C:\Users\jeffe\Documents\Proyectos\Prototipo`.
- `docs/`: documentacion del proyecto (incluyendo este archivo).

## Backend

Desde la carpeta `backend/`:

```bash
cd backend
python start_backend.py
```

O en Windows: `backend\start_backend.bat`.

Endpoints utiles:

```text
http://localhost:8000/health
http://localhost:8000/docs
http://localhost:8000/openapi.json
```

Si configuras `LSEC_API_KEY` en `backend/.env`, la app Android debe enviar la misma clave desde la pantalla de ajustes.

## Frontend Android

Abre esta carpeta en Android Studio:

```text
frontend/
```

La app consume el backend usando rutas como:

```text
/api/v1/data/stats
/api/v1/train/classifier
/api/v1/models/latest/{modelType}
/api/v1/models/download/{modelType}/{version}
/api/v1/models/labels/{modelType}
```

`{modelType}` acepta los valores `classifier` | `sequence` |
`transformer`. El `transformer` es el modelo nuevo (SignTransformer
ONNX, 12 clases LSEC) y se introdujo para que el cliente Android
pueda descargarlo desde `ModelDownloadManager` con el mismo flujo
que ya usaba para el MLP y el GRU legacy.

## URL del servidor

En dispositivo fisico:

```text
http://<IP-de-tu-PC>:8000/
```

Ejemplo:

```text
http://192.168.1.4:8000/
```

En emulador Android:

```text
http://10.0.2.2:8000/
```

## Archivos no copiados

No se copiaron artefactos generados ni secretos locales del frontend original:

- `.git/`
- `.gradle/`
- `.idea/`
- `.kotlin/`
- `build/`
- `keystore.properties`
- archivos `.jks` o `.keystore`
- APK/AAB generados

Si necesitas firmar una version release, vuelve a colocar manualmente los archivos de keystore dentro de `frontend/` en tu entorno local.

## Validacion rapida

Backend:

```bash
cd backend
.venv\Scripts\python.exe -m pytest -q tests
```

Frontend:

```bash
cd frontend
gradlew.bat tasks
```

Para compilar debug:

```bash
cd frontend
gradlew.bat assembleDebug
```
