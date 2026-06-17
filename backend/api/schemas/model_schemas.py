"""Pydantic schemas for model endpoints."""

from typing import Optional

from pydantic import BaseModel


class ModelInfo(BaseModel):
    name: str
    version: str
    accuracy: float
    size_kb: float
    format: str
    num_classes: int
    created_at: str


class ModelLatestResponse(BaseModel):
    version: str
    download_url: str
    size_kb: float
    accuracy: float
    num_classes: int
