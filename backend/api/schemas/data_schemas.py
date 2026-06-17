"""Pydantic schemas for data endpoints."""

from pydantic import BaseModel


class LandmarkSample(BaseModel):
    label: str
    landmarks: list[float]


class StaticLandmarkUpload(BaseModel):
    samples: list[LandmarkSample]


class UploadResponse(BaseModel):
    status: str
    samples_added: int
    total_samples: int


class SequenceUploadResponse(BaseModel):
    status: str
    label: str
    frames: int


class DataStatsResponse(BaseModel):
    static_samples: int
    sequence_samples: int
    static_labels: dict[str, int]
    sequence_labels: dict[str, int]
