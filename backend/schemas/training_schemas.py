"""Pydantic schemas for training endpoints."""

from typing import Optional

from pydantic import BaseModel


class TrainingParams(BaseModel):
    epochs: int = 50
    learning_rate: float = 0.001
    batch_size: int = 64


class TrainingStartResponse(BaseModel):
    task_id: str
    status: str


class TrainingStatusResponse(BaseModel):
    task_id: str
    status: str  # "pending", "running", "completed", "failed"
    epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_accuracy: float = 0.0
    best_accuracy: float = 0.0
    error: Optional[str] = None


class TrainingHistoryEntry(BaseModel):
    task_id: str
    model_type: str
    status: str
    best_accuracy: float
    epochs: int
    started_at: str
    completed_at: Optional[str] = None
