"""Endpoints for model download and versioning."""

import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from api.schemas.model_schemas import ModelInfo, ModelLatestResponse
from api.services.model_registry import ModelRegistry

router = APIRouter(prefix="/models", tags=["models"])

# Initialized in main.py lifespan
model_registry: ModelRegistry = None
_config = None

# The registry keeps classifier/sequence entries in top-level buckets but
# transformer entries (added by `scripts.export_transformer_onnx`) are stored
# in a separate "models" array. We dispatch on this so the same endpoints
# (`/latest/{modelType}`, `/download/{modelType}/{version}`,
# `/labels/{modelType}`) work for all three model families.
_VALID_MODEL_TYPES = ("classifier", "sequence", "transformer")


def _resolve_registry_path(p: str) -> Path:
    """Resolve a registry path (relative or absolute) to an absolute Path.

    If the path was registered without the `backend/` prefix (a known
    historical bug when models were trained from the repo root), also try
    the corrected `<repo>/backend/<p>` location before giving up.
    """
    path = Path(p)
    if not path.is_absolute():
        path = _config.resolve(p) if _config else path
    if path.exists():
        return path
    # Fallback: if p looks like "<root>/models/exports/..." and missing the
    # `backend/` segment, also check "<root>/backend/models/exports/...".
    parts = path.parts
    if "models" in parts and "exports" in parts:
        try:
            mi = parts.index("models")
            candidate = Path(*parts[:mi], "backend", *parts[mi:])
            if candidate.exists():
                return candidate
        except ValueError:
            pass
    return path


def _get_transformer_entry():
    """Find the most-recent transformer entry in the registry.

    Returns the entry dict, or None if no transformer has been registered.
    """
    if model_registry is None:
        return None
    all_entries = model_registry.list_all()
    transformer_entries = [e for e in all_entries if e.get("model_type") == "transformer"]
    if not transformer_entries:
        return None
    return transformer_entries[-1]


def init(registry: ModelRegistry, config):
    global model_registry, _config
    model_registry = registry
    _config = config


@router.get("/list")
async def list_models() -> list[ModelInfo]:
    """List all available model versions."""
    all_models = model_registry.list_all()
    return [
        ModelInfo(
            name=m.get("model_type", "unknown"),
            version=m["version"],
            accuracy=m.get("accuracy", 0.0),
            size_kb=m.get("size_kb", 0.0),
            format=m.get("format", "onnx"),
            num_classes=m.get("num_classes", 0),
            created_at=m.get("created_at", ""),
        )
        for m in all_models
    ]


@router.get("/latest/{model_type}", response_model=ModelLatestResponse)
async def get_latest_model(model_type: str):
    """Get information about the latest model version.

    model_type: 'classifier' | 'sequence' | 'transformer'
    """
    if model_type not in _VALID_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"model_type must be one of {list(_VALID_MODEL_TYPES)}",
        )

    if model_type == "transformer":
        entry = _get_transformer_entry()
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail="No transformer model registered yet. Run scripts/export_transformer_onnx.py first.",
            )
        version = entry["version"]
        # Prefer INT8 (3x smaller) but fall back to FP32.
        onnx_path = entry.get("int8_onnx") or entry.get("fp32_onnx")
        size_kb = 0.0
        if onnx_path:
            try:
                size_kb = round(Path(_resolve_registry_path(onnx_path)).stat().st_size / 1024, 2)
            except OSError:
                pass
        return ModelLatestResponse(
            version=version,
            download_url=f"/api/v1/models/download/{model_type}/{version}",
            size_kb=size_kb,
            accuracy=0.0,
            num_classes=entry.get("n_classes", 0),
        )

    latest = model_registry.get_latest(model_type)
    if latest is None:
        raise HTTPException(
            status_code=404, detail=f"No {model_type} model registered yet. Train a model first."
        )

    return ModelLatestResponse(
        version=latest["version"],
        download_url=f"/api/v1/models/download/{model_type}/{latest['version']}",
        size_kb=latest.get("size_kb", 0.0),
        accuracy=latest.get("accuracy", 0.0),
        num_classes=latest.get("num_classes", 0),
    )


@router.get("/download/{model_type}/{version}")
async def download_model(model_type: str, version: str):
    """Download a model file (ONNX or TFLite)."""
    if model_type not in _VALID_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"model_type must be one of {list(_VALID_MODEL_TYPES)}",
        )

    if model_type == "transformer":
        entry = _get_transformer_entry()
        if entry is None or entry.get("version") != version:
            raise HTTPException(status_code=404, detail="Transformer version not found")
        # Prefer INT8 if it exists, fall back to FP32.
        model_path = entry.get("int8_onnx") or entry.get("fp32_onnx")
        if not model_path:
            raise HTTPException(status_code=404, detail="Transformer model file missing")
        model_path = _resolve_registry_path(model_path)
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model file not found on disk")
        return FileResponse(
            path=str(model_path),
            media_type="application/octet-stream",
            filename=model_path.name,
            headers={
                "X-Model-Version": version,
                "X-Model-Format": "onnx",
            },
        )

    entries = model_registry.get_all(model_type)
    entry = next((e for e in entries if e["version"] == version), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="Model version not found")

    model_path = entry.get("tflite_path") or entry.get("onnx_path")
    if not model_path:
        raise HTTPException(status_code=404, detail="Model file not found on disk")
    model_path = _resolve_registry_path(model_path)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found on disk")

    return FileResponse(
        path=model_path,
        media_type="application/octet-stream",
        filename=Path(model_path).name,
        headers={
            "X-Model-Version": version,
            "X-Model-Accuracy": str(entry.get("accuracy", 0.0)),
            "X-Model-Format": entry.get("format", "onnx"),
        },
    )


@router.post("/export/{model_type}")
async def export_model(model_type: str, background_tasks: BackgroundTasks):
    """Manually trigger ONNX export of the latest trained model.

    Useful after training completes or if auto-export failed.
    model_type: 'classifier' | 'sequence' | 'transformer'
    """
    if model_type not in _VALID_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"model_type must be one of {list(_VALID_MODEL_TYPES)}",
        )
    if model_type == "transformer":
        raise HTTPException(
            status_code=400,
            detail="Transformer export must be triggered via scripts/export_transformer_onnx.py",
        )

    exports_dir = str(_config.resolve(_config.models_exports_dir))

    def do_export():
        from api.services.export_service import (
            export_classifier_onnx_only,
            export_sequence_onnx_only,
        )
        import torch

        if model_type == "classifier":
            checkpoint_path = str(_config.resolve(_config.models_classifier_dir) / "sign_mlp.pth")
            if not Path(checkpoint_path).exists():
                return
            entry = export_classifier_onnx_only(checkpoint_path, exports_dir)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            entry["accuracy"] = float(checkpoint.get("best_accuracy", 0.0))
        else:
            checkpoint_path = str(_config.resolve(_config.models_sequence_dir) / "sign_gru.pth")
            if not Path(checkpoint_path).exists():
                return
            entry = export_sequence_onnx_only(checkpoint_path, exports_dir)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            entry["accuracy"] = float(checkpoint.get("best_accuracy", 0.0))

        model_registry.register(model_type, entry)

    background_tasks.add_task(do_export)
    return {"status": "export_started", "model_type": model_type}


@router.get("/labels/{model_type}")
async def get_labels(model_type: str):
    """Get the label map for a model type."""
    if model_type not in _VALID_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"model_type must be one of {list(_VALID_MODEL_TYPES)}",
        )

    # Transformer labels are stored alongside the model in `labels`.
    if model_type == "transformer":
        entry = _get_transformer_entry()
        if entry is None:
            raise HTTPException(status_code=404, detail="No transformer model registered yet")
        labels_path = entry.get("labels")
        if not labels_path:
            raise HTTPException(status_code=404, detail="Transformer labels path missing")
        labels_path = _resolve_registry_path(labels_path)
        if not labels_path.exists():
            raise HTTPException(status_code=404, detail="Labels file not found on disk")
        with open(labels_path) as f:
            return json.load(f)

    # First try the exported labels (most up-to-date)
    latest = model_registry.get_latest(model_type)
    if latest and latest.get("labels_path") and Path(latest["labels_path"]).exists():
        with open(latest["labels_path"]) as f:
            return json.load(f)

    # Fallback to static dataset labels
    if model_type == "classifier":
        labels_path = _config.resolve(_config.classifier_labels_path)
    elif model_type == "sequence":
        labels_path = _config.resolve(_config.vocabulary_path)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"model_type must be one of {list(_VALID_MODEL_TYPES)}",
        )

    if not labels_path.exists():
        raise HTTPException(status_code=404, detail="Labels file not found")

    with open(labels_path) as f:
        return json.load(f)
