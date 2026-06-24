# Plan de producción: HF Spaces + GitHub Releases

Costo total: **$0/mes** (asumiendo subdominio `is-a.dev` o dominio opcional $1/año).

---

## FASE 0 · Pre-requisitos (10 min)

```bash
# Cuentas necesarias
# 1. huggingface.co (gratis) — para Spaces + Hub
# 2. github.com (gratis) — para código + releases
# 3. (opcional) cloudflare.com — para dominio custom + proxy TLS

# Autenticación local
pip install huggingface_hub
huggingface-cli login        # pega token de https://huggingface.co/settings/tokens
gh auth login                # OAuth con tu cuenta de GitHub
```

---

## FASE 1 · Subir modelos a Hugging Face Hub (30 min)

### 1.1 Crear repos de modelos

```bash
# Repo principal de modelos LSEC
huggingface-cli repo create lsec-models --type model --organization <tu-user>

# Repo de datos (opcional, para checkpoints de training)
huggingface-cli repo create lsec-data --type dataset --organization <tu-user>
```

### 1.2 Subir artefactos

```bash
cd "C:\Users\jeffe\Documents\Proyectos\ModeloPrototipo\backend\models\exports"

# Modelos actuales
huggingface-cli upload <tu-user>/lsec-models sign_mlp_v20260621_220556.onnx .
huggingface-cli upload <tu-user>/lsec-models sign_gru_v20260607_010151.onnx .
huggingface-cli upload <tu-user>/lsec-models sign_transformer_20260614_134616.onnx .
huggingface-cli upload <tu-user>/lsec-models classifier_labels_v20260621_220556.json .
huggingface-cli upload <tu-user>/lsec-models sequence_labels_v20260607_010151.json .
huggingface-cli upload <tu-user>/lsec-models transformer_labels_20260614_134616.json .

# Registry (sobrescribe con la versión patcheada de tu disco)
huggingface-cli upload <tu-user>/lsec-models registry.json . --repo-type model
```

URLs resultantes (formato raw):

```
https://huggingface.co/<tu-user>/lsec-models/resolve/main/sign_transformer_20260614_134616.onnx
https://huggingface.co/<tu-user>/lsec-models/resolve/main/transformer_labels_20260614_134616.json
```

### 1.3 Generar checksums SHA256

```powershell
Get-FileHash backend\models\exports\*.onnx,backend\models\exports\*.json -Algorithm SHA256 | Format-Table
# Anotar los hashes para agregarlos a registry.json
```

---

## FASE 2 · Backend en HF Spaces (1 hora)

### 2.1 Crear Space con Docker

En https://huggingface.co/new-space:

- Name: `lsec-api`
- License: MIT
- SDK: **Docker**
- Hardware: **CPU basic** (gratis)
- Visibility: Public

### 2.2 Estructura de archivos en el repo `lsec-api`

```
huggingface.co/<tu-user>/lsec-api/
├── Dockerfile
├── README.md
├── requirements.txt
├── api/
│   ├── main.py
│   ├── routers/
│   │   └── models_router.py
│   ├── services/
│   │   └── model_registry.py
│   └── config.py
└── start.sh
```

### 2.3 `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema (MediaPipe + ORT no requieren nada extra)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Descarga modelos desde HF Hub al build (más rápido que runtime)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='<tu-user>/lsec-models', local_dir='/data/models')"

ENV MODELS_DIR=/data/models
ENV ENVIRONMENT=production
ENV PORT=7860
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

EXPOSE 7860

CMD ["gunicorn", "api.main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "2", \
     "-b", "0.0.0.0:7860", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
```

### 2.4 `requirements.txt`

```
fastapi==0.115.*
uvicorn[standard]==0.32.*
gunicorn==23.*
huggingface-hub==0.26.*
python-multipart==0.0.*
numpy<2
pydantic==2.*
pydantic-settings==2.*
```

### 2.5 Modificar `models_router.py` para producción

```python
# backend/api/routers/models_router.py

from huggingface_hub import hf_hub_download
from pathlib import Path

HF_REPO = os.getenv("HF_MODEL_REPO", "<tu-user>/lsec-models")
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/data/models"))

@app.get("/api/v1/models/download/{model_type}/{version}")
def download_model(model_type: str, version: str):
    entry = model_registry.get(model_type, version)
    if not entry:
        raise HTTPException(404, "Model version not found")

    # En HF Spaces los modelos viven en /data/models (montado vía snapshot_download)
    local_path = MODELS_DIR / Path(entry["filename"]).name
    if not local_path.exists():
        # Fallback runtime download (si la imagen se reusó sin /data)
        local_path = Path(hf_hub_download(
            repo_id=HF_REPO, filename=entry["filename"]))

    sha = entry.get("sha256", "")
    return FileResponse(
        local_path,
        headers={"X-Model-SHA256": sha, "X-Model-Format": entry.get("format", "onnx")},
    )
```

### 2.6 Variable de entorno en el Space

En https://huggingface.co/<tu-user>/lsec-api/settings → Variables:

```
ENVIRONMENT=production
HF_MODEL_REPO=<tu-user>/lsec-models
```

### 2.7 Push y verificar

```bash
git clone https://huggingface.co/<tu-user>/lsec-api
cd lsec-api
# ... copia archivos arriba ...
git add . && git commit -m "initial deploy" && git push
```

URL del Space: `https://<tu-user>-lsec-api.hf.space` (HTTPS automático).

Verificar:

```bash
curl https://<tu-user>-lsec-api.hf.space/health
# {"status":"ok"}
```

---

## FASE 3 · Endurecer el APK (1 hora)

### 3.1 Generar keystore de release

```bash
keytool -genkey -v -keystore release.keystore -alias lsec -keyalg RSA -keysize 4096 -validity 10000
# Anotar: alias=lsec, password, name=LSEC Translator
```

Guardar como `keystore/release.keystore` y agregar a `.gitignore`.

### 3.2 Configurar `signingConfigs` en `frontend/app/build.gradle.kts`

```kotlin
android {
    signingConfigs {
        create("release") {
            storeFile = file("../keystore/release.keystore")
            storePassword = System.getenv("LSEC_STORE_PASS") ?: ""
            keyAlias = System.getenv("LSEC_KEY_ALIAS") ?: "lsec"
            keyPassword = System.getenv("LSEC_KEY_PASS") ?: ""
        }
    }
    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.getByName("release")
            buildConfigField("String", "DEFAULT_BACKEND_URL", "\"https://<tu-user>-lsec-api.hf.space/\"")
        }
    }
}
```

### 3.3 Actualizar `UserPreferences.kt` y `SettingsViewModel.kt`

Reemplazar los 3 sitios donde está hardcodeado `http://192.168.18.4:8000/`:

```kotlin
serverUrl = BuildConfig.DEFAULT_BACKEND_URL
```

### 3.4 Limpiar `network_security_config.xml`

```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <!-- Producción: cleartext completamente deshabilitado -->
    <base-config cleartextTrafficPermitted="false">
        <trust-anchors>
            <certificates src="system" />
        </trust-anchors>
    </base-config>
</network-security-config>
```

Quitar cualquier `<domain-config>` que apunte a 192.168.18.4.

### 3.5 ProGuard rules (`frontend/app/proguard-rules.pro`)

```proguard
# ONNX Runtime (carga nativa por reflexión)
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# MediaPipe Tasks
-keep class com.google.mediapipe.** { *; }
-dontwarn com.google.mediapipe.**

# Hilt / Dagger
-keep class dagger.hilt.** { *; }

# Moshi reflection
-keep class kotlin.reflect.jvm.internal.** { *; }
-keepattributes *Annotation*
-keepclassmembers class * {
    @com.squareup.moshi.* <methods>;
}

# DataStore
-keep class androidx.datastore.*.** { *; }
```

### 3.6 Build release

```bash
cd frontend
export LSEC_STORE_PASS=...
export LSEC_KEY_ALIAS=lsec
export LSEC_KEY_PASS=...
./gradlew.bat assembleRelease
# → app/build/outputs/apk/release/app-release.apk
```

### 3.7 Smoke test en la tablet

```bash
adb -s R52XB09FEKE install -r frontend/app/build/outputs/apk/release/app-release.apk
# Verificar:
#   - App arranca
#   - Carga modelos desde HF (no desde 192.168.18.4)
#   - Calibración funciona (12 señas → centroides)
#   - Reconocimiento usa centroides
```

---

## FASE 4 · Distribución vía GitHub Releases (30 min)

### 4.1 Subir código a GitHub

```bash
# Crear repo
gh repo create lsec-translator --public --source=. --remote=origin

# Push
git remote add origin https://github.com/<tu-user>/lsec-translator.git
git push -u origin main
```

### 4.2 CI/CD: build APK en cada tag (`.github/workflows/release.yml`)

```yaml
name: release
on:
  push:
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with: { distribution: 'temurin', java-version: '17' }

      - name: Decode keystore
        run: |
          mkdir -p keystore
          echo "${{ secrets.KEYSTORE_BASE64 }}" | base64 -d > keystore/release.keystore

      - name: Build release APK
        working-directory: frontend
        env:
          LSEC_STORE_PASS: ${{ secrets.STORE_PASS }}
          LSEC_KEY_ALIAS: ${{ secrets.KEY_ALIAS }}
          LSEC_KEY_PASS: ${{ secrets.KEY_PASS }}
        run: ./gradlew assembleRelease

      - name: Upload APK to release
        uses: softprops/action-gh-release@v2
        with:
          files: frontend/app/build/outputs/apk/release/app-release.apk
          generate_release_notes: true
```

### 4.3 Secrets en GitHub

En https://github.com/<tu-user>/lsec-translator/settings/secrets/actions:

- `KEYSTORE_BASE64` → `base64 -w 0 keystore/release.keystore | pbcopy`
- `STORE_PASS`, `KEY_ALIAS`, `KEY_PASS` (los del keytool)

### 4.4 Crear primer release

```bash
git tag v1.0.0
git push origin v1.0.0
# GitHub Actions builds + adjunta APK al release
```

URL: `https://github.com/<tu-user>/lsec-translator/releases/latest`

### 4.5 (Opcional) Landing page con GitHub Pages

`docs/index.md`:

```markdown
# LSEC Translator

App Android para reconocer lengua de señas ecuatoriana.

## Descargar
[Última versión](https://github.com/<tu-user>/lsec-translator/releases/latest)

## Código
[github.com/<tu-user>/lsec-translator](https://github.com/<tu-user>/lsec-translator)

## API
https://<tu-user>-lsec-api.hf.space
```

Activar Pages en `Settings → Pages → Source: main /docs`.

---

## FASE 5 · Monitoreo gratis (15 min)

### 5.1 UptimeRobot

- Monitor: `https://<tu-user>-lsec-api.hf.space/health`
- Intervalo: 5 min
- Alerta: email cuando 2 checks fallen seguidos

### 5.2 Sentry (crash reporting)

En `frontend/app/build.gradle.kts`:

```kotlin
implementation("io.sentry:sentry-android:7.13.0")
```

Crear proyecto en https://sentry.io (free, 5K events/mes).

`MainActivity.kt` (onCreate):

```kotlin
Sentry.init { options ->
    options.dsn = "https://...@sentry.io/..."
    options.environment = if (BuildConfig.DEBUG) "debug" else "production"
}
```

---

## FASE 6 · Hardening de seguridad (30 min)

### 6.1 Backend: autenticación en `/download/*`

```python
# api/routers/models_router.py
from fastapi import Header, HTTPException

API_KEY = os.getenv("LSEC_API_KEY", "dev-key-change-me")

async def verify_key(x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(403, "Invalid API key")

@app.get("/api/v1/models/download/{model_type}/{version}", dependencies=[Depends(verify_key)])
def download_model(...):
    ...
```

Variable de entorno en HF Space: `LSEC_API_KEY=<random-32-chars>`.

Cliente Android: agregar `OkHttp interceptor` que inyecte `X-API-Key: <key>` (se puede hardcodear en build por ahora para uso no-público).

### 6.2 Rate limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/v1/models/list")
@limiter.limit("60/minute")
def list_models(request: Request):
    ...
```

`requirements.txt` agregar: `slowapi==0.1.*`

---

## FASE 7 · Distribución ampliada (1-2 días, opcional)

### 7.1 F-Droid

- Hacer repo público + agregar `metadata/<package>.yml` con sources, license.
- F-Droid revisa el build reproducible y publica (1-2 semanas de review).

### 7.2 Huawei AppGallery

- Crear cuenta de desarrollador (gratis).
- Subir APK firmado + AAB.

### 7.3 Samsung Galaxy Store

- Seller Portal gratis.
- Mismo APK.

---

## Línea de tiempo

| Día | Tarea |
|---|---|
| **Día 1 (3h)** | Fase 1 (modelos) + Fase 2 (HF Space) + Fase 3 (keystore + APK release) |
| **Día 1 (2h)** | Fase 4 (GitHub + CI/CD) + Fase 5 (monitoreo) |
| **Día 2 (1h)** | Fase 6 (auth + rate limit) + smoke tests |
| **Día 2 (1h)** | Tag v1.0.0 + primer release público |
| **Semana 2+** | Fase 7 (F-Droid, etc.) + iteración con feedback de usuarios |

**Costo mensual**: $0
**Costo único**: ~$0 (todo en software libre)

---

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| HF Space se duerme / cold start | HF CPU basic no duerme las instancias activas. Si la app queda idle, sigue corriendo. |
| API key leak en APK | Ofuscar con ProGuard + rotación trimestral. Para uso privado (e.g. solo 10 usuarios), no es crítico. |
| Cambios incompatibles en `models_router.py` | Mantener el patch de `_resolve_registry_path` del commit `a927a72` — resuelve paths con/sin `backend/`. |
| R8 quita clases de ORT/MediaPipe | Las keep rules en `proguard-rules.pro` lo cubren. Smoke test obligatorio con APK release. |
| Updates del modelo en HF Hub | El cliente puede consultar `/api/v1/models/list` cada vez que abre Settings. |
| Sin HTTPS en WiFi local | `cleartextTrafficPermitted=false` bloquea HTTP; el release apunta solo a `https://*.hf.space`. |

---

## Primer milestone: v1.0.0 en producción (1 fin de semana)

1. ✅ Modelos en HF Hub
2. ✅ Backend en HF Space con `/health` 200
3. ✅ APK firmado con `BuildConfig.DEFAULT_BACKEND_URL` apuntando a HF
4. ✅ Smoke test en tablet: app carga modelos, calibra, reconoce
5. ✅ Tag v1.0.0 → release en GitHub con APK
6. ✅ UptimeRobot + Sentry configurados
