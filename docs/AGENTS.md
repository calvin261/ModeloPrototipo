# AGENTS.md — ModeloPrototipo

LSC sign-language recognition system: a **Python/FastAPI** training backend plus an **Android/Kotlin** app that consumes the exported models. Single repo, three top-level sections: `backend/`, `frontend/`, `docs/`.

> Read `docs/CHANGELOG.md` for the thesis bitácora (decisions, metrics, rationale) and update it whenever you add an entry of type `data · model · code · doc · config · fix · eval`. It's the project memory.

## Repo layout (what owns what)

```
backend/
├── api/                  FastAPI service (main.py, routers/, services/, schemas/, config.py, tasks/)
├── hand_model/           Reusable Python package (MLP, Bi-GRU+Attention, MediaPipe helpers, configs)
├── scripts/              One-off CLIs: prepare_*, train_*, evaluate*, export*, build_*_splits, collect_data, run_app
├── tests/                pytest
├── data/                 Generated: data/landmarks/*.csv, data/sequences/<word>/*.npy, data/processed/landmarks/
├── models/               Generated: models/classifier/sign_mlp.pth, models/sequence/sign_gru.pth, models/exports/*.onnx + registry.json
├── datasets/             Static config: hands.yaml, labels_map.json, lsec_alphabet_labels.json, lsec_vocabulary.json
├── notebooks/            Jupyter notebooks (data exploration)
├── logs/                 Dev logs and one-off helper scripts
├── pyproject.toml        Python packaging
├── Dockerfile / docker-compose.yml
└── start_backend.{py,bat}, deploy.sh, cloudflare-tunnel.sh, .env.example, .dockerignore, .gitignore

frontend/                 Standalone Android Studio project (Hilt + Compose + CameraX + TFLite)

docs/                     README.md, AGENTS.md (this file), CHANGELOG.md, INTEGRACION_FRONTEND_BACKEND.md, INSTRUCCIONES/
```

`backend/api/main.py` (was `backend/main.py`) and every script in `backend/scripts/` inject `backend/` (the directory containing this file's `parent.parent`) into `sys.path` so the `hand_model` package is importable without a venv activation step. **Do not break that path setup** — every CLI in `scripts/` depends on it.

## Dev environment

- **Python ≥ 3.10**. Use `.venv/` at the repo root (`python -m venv .venv`). It's gitignored; create it. The `.vscode/launch.json` configs and `start_backend.bat` point at `<root>/.venv/Scripts/python.exe`.
- **Windows-first.** `backend\start_backend.bat` activates `.venv\Scripts\activate.bat`; `backend/start_backend.py` shells out to `python -m uvicorn` (fallback when `uvicorn` isn't on PATH).
- **No CI, no lint/format config, no Makefile, no `.editorconfig`, no pre-commit.** Don't invent new tooling; match the existing style (PEP 8, type hints, dataclass configs in `backend/hand_model/config.py` and `backend/api/config.py`).
- **VS Code:** `.vscode/launch.json` defines 5 debug configs (Backend uvicorn, start_backend.py, Train Classifier, Train Sequence, Tests) — all set `cwd=${workspaceFolder}/backend` and `PYTHONPATH=${workspaceFolder}/backend`. When running scripts directly, `cd backend` first.
- **Heavy stack is already declared in `backend/pyproject.toml`:** ultralytics, torch, mediapipe, opencv-python, pyttsx3, scikit-learn, seaborn. The Docker image installs **CPU-only** torch (see `backend/api/Dockerfile`) — match this for reproducibility; don't grab CUDA wheels by default.

## Common commands

All commands assume a shell at the repo root unless noted.

| Task | Command |
|---|---|
| Backend (dev, hot-reload) | `cd backend && start_backend.bat` *or* `cd backend && python start_backend.py` |
| Backend (manual) | `cd backend && .venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload` |
| Tests | `cd backend && .venv\Scripts\python -m pytest -q tests` |
| API docs | `http://localhost:8000/docs` (Swagger), `http://localhost:8000/health` for liveness |
| Rebuild classifier splits | `cd backend && python scripts/build_classifier_splits.py` |
| Rebuild sequence splits | `cd backend && python scripts/build_sequence_splits.py` |
| Train MLP (alphabet) | `cd backend && python scripts/train_classifier.py` |
| Train Bi-GRU (words) | `cd backend && python scripts/train_sequence.py --splits-json data/sequences/splits.json` |
| Train SignTransformer (F3) | `cd backend && python -m scripts.train_transformer --splits-json data/sequences/splits.json` |
| Extract LSC50 sequences | `cd backend && python scripts/prepare_sequences_from_lsc50.py --workers 1` (single-process is faster here — see CHANGELOG 2026-06-07) |
| **Full pipeline (raw videos → ONNX)** | `backend\scripts\full_pipeline.bat` (or `--fast` / `--no-extract`); logs in `backend\logs\*_pipeline.log` |
| Container (Debian) | `cd backend && ./deploy.sh install` once, then `./deploy.sh up\|rebuild\|restart\|stop\|logs\|status\|ip` |
| Public tunnel | `cd backend && ./cloudflare-tunnel.sh quick\|install\|status\|stop` |

## Architecture notes (non-obvious)

- **Two model families, two directories.** Static (alphabet, MLP 63→256→128→27) lives in `backend/models/classifier/sign_mlp.pth`; dynamic (words, Bi-GRU 2×128 + attention, 50 classes) lives in `backend/models/sequence/sign_gru.pth`. Both are exported to ONNX in `backend/models/exports/`, and a versioned `registry.json` tracks which export is current. **J and Z are excluded from the static classifier** — they require movement (see `backend/api/services/training_service.py`).
- **Landmarks are the universal interface.** 21 keypoints × 3D = 63 dims/frame, normalized wrist-relative + scale-invariant (`backend/hand_model/landmarks.py::normalize_landmarks`, pattern from Zimmermann & Brox 2017). Anything that doesn't produce 63-dim landmarks is out-of-band.
- **Sequence file naming encodes the split:** `<signer>_<word>_<idx>.npy` (T, 63) float16, parsed by `SequenceDataset` in `backend/hand_model/data.py`. **The GRU split is held-out-by-signer, not random** — random splits leak frames of the same person across train/val and inflate accuracy (the 1.0 metrics from 2026-02 were that bug; see CHANGELOG).
- **API surface is `/api/v1/...`** (see `backend/api/config.py:11`). Only `/` and `/health` are public; everything else requires `X-API-Key` when `LSEC_API_KEY` is set. If the env var is empty, auth is silently disabled (dev mode) — `backend/api/main.py` logs which mode.
- **Android↔backend integration contract** lives in `docs/INTEGRACION_FRONTEND_BACKEND.md`. From a device, the app hits `http://<PC-IP>:8000/`; from the Android emulator use `http://10.0.2.2:8000/`. CORS is `allow_origins=["*"]` in `backend/api/main.py` so preflight is never the failure.

## Conventions & gotchas

- **Configs are dataclasses, not YAML.** `backend/hand_model/config.py` defines `TrainConfig`, `ClassifierConfig`, `SequenceConfig`, `AppConfig`; `backend/api/config.py` defines `ServerConfig`. Hardcode the test of new knobs there, not in scripts.
- **Generated artifacts are gitignored** (see `.gitignore` at the repo root): `.venv/`, `backend/data/`, `*.pt`, `*.onnx`, `*.tflite`, `*.task`, `*.log`, `backend/logs/`, `.pytest_cache/`. **Only the ONNX exports and `registry.json` survive in `backend/models/exports/`.** Don't commit `.pt` checkpoints.
- **Naming inconsistency:** README says "LSC" (Colombian) but code says "LSEC Translator" and the frontend README still claims Ecuadorian context (FENASEC, etc.). The active project is the Colombian one; the frontend README is stale — trust `docs/CHANGELOG.md` and the backend code over it.
- **No frontend build/CI hook from the Python side.** The `frontend/` Gradle project is self-contained; changes there don't affect the backend. Don't add cross-subproject scripts.
- **Logs in dev:** the dev backend and `start_backend.py` log to stdout/stderr (capture with `> backend/logs/backend.out.log 2> backend/logs/backend.err.log` if you want files). Container logs come from `docker compose logs -f` (or `cd backend && ./deploy.sh logs`).
- **PyTorch install in the container is CPU-only and pinned** (`torch==2.3.1+cpu`, `torchvision==0.18.1+cpu` in `backend/api/Dockerfile`). Local venvs can use whatever `pip install torch` gives you, but stay CPU-only unless explicitly switching to GPU.
- **Android assets** (TFLite + hand_landmarker.task) are produced by training/export scripts and copied into `frontend/app/src/main/assets/`. See `frontend/README.md` §"Configurar Modelos ML".
- **Dev helper scripts in `backend/logs/`** (`filter_log.py`, `parse_ui.py`) are not part of the API — they're ad-hoc tools for debugging dev runs.

## Testing

- **Test suite is intentionally minimal** — `backend/tests/test_data.py` was the seed; we now have `test_augmentations.py`, `test_normalization.py`, `test_sign_transformer.py`, and the slow `test_end_to_end.py`. Run with `cd backend && .venv\Scripts\python -m pytest -q tests`. The `slow` marker is registered in `backend/pyproject.toml`; the end-to-end test is skipped in PR CI via `-m "not slow"`.
- **No integration tests for the API.** Verify endpoints manually via `/docs` (Swagger has a "Try it out" UI) or `curl http://localhost:8000/health`.
- **Expensive things to avoid in tests/CI:** model training, MediaPipe extraction over full datasets, anything touching `backend/data/public/LSC50/` (thousands of videos). The CI strategy here is "don't run it" — keep heavy work in scripts, not in the test path. The end-to-end test runs in ~3 s and exists only to smoke-test the data→augment→train→eval plumbing; do not grow it into a real benchmark.

## When you change something

- Bump / add an entry in `docs/CHANGELOG.md` with date, type tag, what/why/source, and metrics (if it's a model or data change).
- Keep `backend/models/exports/registry.json` consistent: training scripts update it via the `ModelRegistry` injected by `backend/api/main.py`. The export service (`backend/api/services/export_service.py`) is what writes new ONNX files — re-export after retraining or the Android app will keep using the stale `*.tflite` derived from the old ONNX.
- If you change a path constant, update **both** `backend/api/config.py` (Python side) and `docs/INTEGRACION_FRONTEND_BACKEND.md` (Android side) in the same commit.
