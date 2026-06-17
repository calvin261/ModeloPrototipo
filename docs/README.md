# ModeloPrototipo

LSC sign-language recognition project. Unified repo with three top-level sections:

- `backend/` — Python/FastAPI training backend, ML models, training/export scripts, datasets.
- `frontend/` — Standalone Android app (Kotlin, Jetpack Compose, CameraX, TFLite).
- `docs/` — All project documentation (this folder).

## Quick start

1. **Backend (dev):** `cd backend && python start_backend.py`
   (or Windows: `backend\start_backend.bat`)
2. **Verify API:** open `http://localhost:8000/health` or `http://localhost:8000/docs`
3. **Frontend:** open `frontend/` in Android Studio
4. **Android device:** in the app, set server URL to `http://<your-PC-IP>:8000/`
5. **Android emulator:** use `http://10.0.2.2:8000/`

## Where to read more

| Topic | File |
|---|---|
| Agent / OpenCode instructions | `docs/AGENTS.md` |
| Thesis bitácora (decisions, metrics) | `docs/CHANGELOG.md` |
| Step-by-step training guide | `docs/INSTRUCCIONES/GUIA_PASO_A_PASO.md` |
| Training sources & references | `docs/INSTRUCCIONES/FUENTES.md` |
| Android↔backend integration contract | `docs/INTEGRACION_FRONTEND_BACKEND.md` |
| Android app README | `frontend/README.md` |
