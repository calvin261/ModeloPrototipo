"""Endpoints for training control and monitoring."""

from fastapi import APIRouter, HTTPException

from api.schemas.training_schemas import (
    TrainingHistoryEntry,
    TrainingParams,
    TrainingStartResponse,
    TrainingStatusResponse,
)
from api.services.training_service import (
    get_all_tasks,
    get_task_status,
    start_classifier_training,
    start_sequence_training,
)

router = APIRouter(prefix="/train", tags=["training"])


@router.post("/classifier", response_model=TrainingStartResponse)
async def train_classifier(params: TrainingParams):
    """Start classifier (MLP) training as a background task."""
    task_id = start_classifier_training(
        epochs=params.epochs,
        learning_rate=params.learning_rate,
        batch_size=params.batch_size,
    )
    return TrainingStartResponse(task_id=task_id, status="started")


@router.post("/sequence", response_model=TrainingStartResponse)
async def train_sequence(params: TrainingParams):
    """Start sequence model (GRU) training as a background task."""
    task_id = start_sequence_training(
        epochs=params.epochs,
        learning_rate=params.learning_rate,
        batch_size=params.batch_size,
    )
    return TrainingStartResponse(task_id=task_id, status="started")


@router.get("/status/{task_id}", response_model=TrainingStatusResponse)
async def training_status(task_id: str):
    """Get the status of a training task."""
    status = get_task_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return TrainingStatusResponse(**status)


@router.get("/history")
async def training_history() -> list[TrainingHistoryEntry]:
    """Get history of all training runs."""
    tasks = get_all_tasks()
    return [
        TrainingHistoryEntry(
            task_id=t["task_id"],
            model_type=t["model_type"],
            status=t["status"],
            best_accuracy=t["best_accuracy"],
            epochs=t["total_epochs"],
            started_at=t["started_at"],
            completed_at=t.get("completed_at"),
        )
        for t in tasks
    ]
