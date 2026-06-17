"""FastAPI backend for LSEC model training and management."""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is in path so `hand_model` package is importable
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from api.config import ServerConfig
from api.routers import data_router, models_router, training_router
from api.services.data_service import DataService
from api.services.model_registry import ModelRegistry
from api.services import training_service

config = ServerConfig()

# ── API-Key authentication ─────────────────────────────────────────────────────
# Set LSEC_API_KEY env var to enable protection. If empty, auth is disabled (dev mode).
_API_KEY = os.environ.get("LSEC_API_KEY", "").strip()
# Endpoints that are always public (health probes, root info)
_PUBLIC_PATHS = {"/", "/health"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize services
    data_service = DataService(
        landmarks_dir=str(config.resolve(config.data_landmarks_dir)),
        sequences_dir=str(config.resolve(config.data_sequences_dir)),
    )
    data_router.init(data_service)

    exports_dir = str(config.resolve(config.models_exports_dir))
    registry = ModelRegistry(registry_path=f"{exports_dir}/registry.json")
    models_router.init(registry, config)

    # Allow training service to register exported models
    training_service.init_export(registry, exports_dir)

    if _API_KEY:
        print("API key protection: ENABLED")
    else:
        print("API key protection: DISABLED (set LSEC_API_KEY to enable)")

    yield


app = FastAPI(
    title="LSEC Training API",
    description="Backend for training and serving LSEC sign language models",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Reject requests that don't carry the correct X-API-Key header.

    Skipped when LSEC_API_KEY is not configured (dev/local mode).
    Public paths (/ and /health) are always allowed.
    """
    if _API_KEY and request.url.path not in _PUBLIC_PATHS:
        key = request.headers.get("X-API-Key", "")
        if key != _API_KEY:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key"},
                headers={"WWW-Authenticate": "ApiKey"},
            )
    return await call_next(request)


# Register routers
app.include_router(data_router.router, prefix=config.api_prefix)
app.include_router(training_router.router, prefix=config.api_prefix)
app.include_router(models_router.router, prefix=config.api_prefix)


@app.get("/")
async def root():
    return {"message": "LSEC Training API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "ok"}
