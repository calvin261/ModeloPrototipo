"""Endpoints for uploading and managing training data."""

import io
import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.schemas.data_schemas import (
    DataStatsResponse,
    SequenceUploadResponse,
    StaticLandmarkUpload,
    UploadResponse,
)
from backend.services.data_service import DataService

router = APIRouter(prefix="/data", tags=["data"])

# Initialized in main.py lifespan
data_service: DataService = None


def init(service: DataService):
    global data_service
    data_service = service


@router.post("/upload/static", response_model=UploadResponse)
async def upload_static_landmarks(data: StaticLandmarkUpload):
    """Upload static landmark samples for classifier training."""
    if not data.samples:
        raise HTTPException(status_code=400, detail="No samples provided")

    for sample in data.samples:
        if len(sample.landmarks) != 63:
            raise HTTPException(
                status_code=400,
                detail=f"Each sample must have 63 landmarks, got {len(sample.landmarks)}",
            )

    samples_dicts = [s.model_dump() for s in data.samples]
    added = data_service.append_static_samples(samples_dicts)
    total = data_service.get_total_static()

    return UploadResponse(status="ok", samples_added=added, total_samples=total)


@router.post("/upload/sequence", response_model=SequenceUploadResponse)
async def upload_sequence(
    file: UploadFile = File(...),
    label: str = Form(...),
):
    """Upload a sequence of landmarks as .npy file for GRU training."""
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="File must be .npy format")

    contents = await file.read()
    sequence = np.load(io.BytesIO(contents))

    if sequence.ndim != 2 or sequence.shape[1] != 63:
        raise HTTPException(
            status_code=400,
            detail=f"Sequence must have shape (T, 63), got {sequence.shape}",
        )

    frames = data_service.save_sequence(label, sequence)
    return SequenceUploadResponse(status="ok", label=label, frames=frames)


@router.get("/stats", response_model=DataStatsResponse)
async def get_data_stats():
    """Get statistics about stored training data."""
    stats = data_service.get_stats()
    return DataStatsResponse(**stats)


@router.delete("/static/{label}")
async def delete_static_label(label: str):
    """Delete all static samples for a given label."""
    deleted = data_service.delete_static_label(label)
    return {"status": "ok", "label": label, "deleted": deleted}


@router.delete("/sequence/{label}")
async def delete_sequence_label(label: str):
    """Delete all sequence files for a given label."""
    deleted = data_service.delete_sequence_label(label)
    return {"status": "ok", "label": label, "deleted": deleted}
