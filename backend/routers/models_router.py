"""Endpoints for model download and versioning."""

import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from backend.schemas.model_schemas import ModelInfo, ModelLatestResponse
from backend.services.model_registry import ModelRegistry

router = APIRouter(prefix="/models", tags=["models"])

# Initialized in main.py lifespan
model_registry: ModelRegistry = None
_config = None


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

    model_type: 'classifier' or 'sequence'
    """
    if model_type not in ("classifier", "sequence"):
        raise HTTPException(
            status_code=400, detail="model_type must be 'classifier' or 'sequence'"
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
    if model_type not in ("classifier", "sequence"):
        raise HTTPException(
            status_code=400, detail="model_type must be 'classifier' or 'sequence'"
        )

    entries = model_registry.get_all(model_type)
    entry = next((e for e in entries if e["version"] == version), None)

    if entry is None:
        raise HTTPException(status_code=404, detail="Model version not found")

    # Try TFLite first, then ONNX
    model_path = entry.get("tflite_path") or entry.get("onnx_path")
    if not model_path or not Path(model_path).exists():
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
    model_type: 'classifier' or 'sequence'
    """
    if model_type not in ("classifier", "sequence"):
        raise HTTPException(
            status_code=400, detail="model_type must be 'classifier' or 'sequence'"
        )

    exports_dir = str(_config.resolve(_config.models_exports_dir))

    def do_export():
        from backend.services.export_service import (
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
            status_code=400, detail="model_type must be 'classifier' or 'sequence'"
        )

    if not labels_path.exists():
        raise HTTPException(status_code=404, detail="Labels file not found")

    with open(labels_path) as f:
        return json.load(f)
