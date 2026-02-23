"""Service for managing model training as background tasks."""

import sys
import uuid
from datetime import datetime
from pathlib import Path
from threading import Thread

# Add project root to path so training scripts can be imported
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, str(Path(project_root) / "src"))

from hand_model.config import ClassifierConfig, SequenceConfig

# Registry and exports dir are set by main.py after initialization
_model_registry = None
_exports_dir: str = None


def init_export(registry, exports_dir: str):
    """Called by main.py to inject registry and exports directory."""
    global _model_registry, _exports_dir
    _model_registry = registry
    _exports_dir = exports_dir


# In-memory storage for training task status
_training_tasks: dict[str, dict] = {}


def get_task_status(task_id: str) -> dict | None:
    """Get the status of a training task."""
    return _training_tasks.get(task_id)


def get_all_tasks() -> list[dict]:
    """Get all training tasks."""
    return list(_training_tasks.values())


def _prepare_classifier_data() -> tuple[str, str]:
    """Merge all CSVs in data/landmarks/ and split 80/20 into train/val temp files.

    Returns (train_csv_path, val_csv_path) as absolute paths.
    Handles labels with only 1 sample by placing them in training only.
    """
    import pandas as pd

    landmarks_dir = Path(project_root) / "data" / "landmarks"
    csv_files = [
        f for f in landmarks_dir.glob("*.csv") if not f.name.startswith("_")
    ]

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {landmarks_dir}. "
            "Capture some samples from the app first."
        )

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        # Standardize column names: label + x0..x(N-1) — LandmarkDataset reads by position
        cols = ["label"] + [f"x{i}" for i in range(df.shape[1] - 1)]
        df.columns = cols
        dfs.append(df)

    combined = (
        pd.concat(dfs, ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # J and Z require movement (dynamic signs) — exclude from static classifier
    dynamic_signs = {"J", "Z"}
    excluded = combined[combined["label"].isin(dynamic_signs)]["label"].unique().tolist()
    combined = combined[~combined["label"].isin(dynamic_signs)].reset_index(drop=True)
    if excluded:
        print(
            f"Note: Excluded {excluded} from static classifier "
            "(they are dynamic signs — capture them as sequences instead)."
        )

    if combined.empty:
        raise ValueError("No data remaining after excluding dynamic signs (J, Z).")

    # Stratified 80/20 split; labels with 1 sample go entirely to train
    train_rows, val_rows = [], []
    for _label, group in combined.groupby("label"):
        n = len(group)
        if n <= 1:
            train_rows.append(group)
        else:
            n_val = max(1, int(n * 0.2))
            val_rows.append(group.iloc[:n_val])
            train_rows.append(group.iloc[n_val:])

    train_df = (
        pd.concat(train_rows).sample(frac=1, random_state=42).reset_index(drop=True)
    )
    # If all labels had 1 sample, use a small subset of train as val
    if not val_rows:
        n_val = max(1, int(len(train_df) * 0.2))
        val_df = train_df.iloc[:n_val].reset_index(drop=True)
        train_df = train_df.iloc[n_val:].reset_index(drop=True)
    else:
        val_df = (
            pd.concat(val_rows).sample(frac=1, random_state=42).reset_index(drop=True)
        )

    train_path = str(landmarks_dir / "_train_tmp.csv")
    val_path = str(landmarks_dir / "_val_tmp.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    return train_path, val_path


def start_classifier_training(
    epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    train_csv: str = None,
    val_csv: str = None,
) -> str:
    """Start classifier training in a background thread.

    Returns a task_id for polling progress.
    """
    task_id = str(uuid.uuid4())

    cfg = ClassifierConfig()
    cfg.epochs = epochs
    cfg.learning_rate = learning_rate
    cfg.batch_size = batch_size

    # Auto-detect and prepare data if no explicit CSVs provided
    if train_csv and val_csv:
        cfg.train_csv = train_csv
        cfg.val_csv = val_csv
    else:
        try:
            cfg.train_csv, cfg.val_csv = _prepare_classifier_data()
        except Exception as e:
            # Record failure immediately without starting thread
            fail_id = str(uuid.uuid4())
            _training_tasks[fail_id] = {
                "task_id": fail_id,
                "model_type": "classifier",
                "status": "failed",
                "epoch": 0,
                "total_epochs": epochs,
                "train_loss": 0.0,
                "val_accuracy": 0.0,
                "best_accuracy": 0.0,
                "error": str(e),
                "started_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
            }
            return fail_id

    _training_tasks[task_id] = {
        "task_id": task_id,
        "model_type": "classifier",
        "status": "pending",
        "epoch": 0,
        "total_epochs": epochs,
        "train_loss": 0.0,
        "val_accuracy": 0.0,
        "best_accuracy": 0.0,
        "error": None,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
    }

    def progress_callback(epoch, total, loss, accuracy, best_acc):
        _training_tasks[task_id].update({
            "status": "running",
            "epoch": epoch,
            "total_epochs": total,
            "train_loss": round(loss, 4),
            "val_accuracy": round(accuracy, 4),
            "best_accuracy": round(best_acc, 4),
        })

    def run_training():
        try:
            _training_tasks[task_id]["status"] = "running"
            from scripts.train_classifier import train
            train(cfg, progress_callback=progress_callback)
            _training_tasks[task_id]["status"] = "completed"
            _training_tasks[task_id]["completed_at"] = datetime.now().isoformat()
            # Auto-export after successful training
            _auto_export_classifier(task_id, cfg)
        except Exception as e:
            _training_tasks[task_id]["status"] = "failed"
            _training_tasks[task_id]["error"] = str(e)
            _training_tasks[task_id]["completed_at"] = datetime.now().isoformat()

    thread = Thread(target=run_training, daemon=True)
    thread.start()

    return task_id


def start_sequence_training(
    epochs: int = 80,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    sequences_dir: str = None,
    vocabulary_path: str = None,
) -> str:
    """Start sequence model training in a background thread.

    Returns a task_id for polling progress.
    """
    task_id = str(uuid.uuid4())

    cfg = SequenceConfig()
    cfg.epochs = epochs
    cfg.learning_rate = learning_rate
    cfg.batch_size = batch_size
    if sequences_dir:
        cfg.sequences_dir = sequences_dir
    if vocabulary_path:
        cfg.vocabulary_path = vocabulary_path

    _training_tasks[task_id] = {
        "task_id": task_id,
        "model_type": "sequence",
        "status": "pending",
        "epoch": 0,
        "total_epochs": epochs,
        "train_loss": 0.0,
        "val_accuracy": 0.0,
        "best_accuracy": 0.0,
        "error": None,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
    }

    def progress_callback(epoch, total, loss, accuracy, best_acc):
        _training_tasks[task_id].update({
            "status": "running",
            "epoch": epoch,
            "total_epochs": total,
            "train_loss": round(loss, 4),
            "val_accuracy": round(accuracy, 4),
            "best_accuracy": round(best_acc, 4),
        })

    def run_training():
        try:
            _training_tasks[task_id]["status"] = "running"
            from scripts.train_sequence import train
            train(cfg, progress_callback=progress_callback)
            _training_tasks[task_id]["status"] = "completed"
            _training_tasks[task_id]["completed_at"] = datetime.now().isoformat()
            # Auto-export after successful training
            _auto_export_sequence(task_id, cfg)
        except Exception as e:
            _training_tasks[task_id]["status"] = "failed"
            _training_tasks[task_id]["error"] = str(e)
            _training_tasks[task_id]["completed_at"] = datetime.now().isoformat()

    thread = Thread(target=run_training, daemon=True)
    thread.start()

    return task_id


# ---------------------------------------------------------------------------
# Internal auto-export helpers
# ---------------------------------------------------------------------------

def _auto_export_classifier(task_id: str, cfg):
    """Export classifier .pth -> ONNX and register. Called after training."""
    if _model_registry is None or _exports_dir is None:
        return
    try:
        from backend.services.export_service import export_classifier_onnx_only
        checkpoint_path = str(Path(project_root) / cfg.model_save_path)
        entry = export_classifier_onnx_only(checkpoint_path, _exports_dir)
        entry["accuracy"] = _training_tasks[task_id].get("best_accuracy", 0.0)
        _model_registry.register("classifier", entry)
        _training_tasks[task_id]["export_status"] = "exported"
        _training_tasks[task_id]["model_version"] = entry["version"]
    except Exception as e:
        _training_tasks[task_id]["export_status"] = f"export_failed: {e}"


def _auto_export_sequence(task_id: str, cfg):
    """Export sequence .pth -> ONNX and register. Called after training."""
    if _model_registry is None or _exports_dir is None:
        return
    try:
        from backend.services.export_service import export_sequence_onnx_only
        checkpoint_path = str(Path(project_root) / cfg.model_save_path)
        entry = export_sequence_onnx_only(checkpoint_path, _exports_dir)
        entry["accuracy"] = _training_tasks[task_id].get("best_accuracy", 0.0)
        _model_registry.register("sequence", entry)
        _training_tasks[task_id]["export_status"] = "exported"
        _training_tasks[task_id]["model_version"] = entry["version"]
    except Exception as e:
        _training_tasks[task_id]["export_status"] = f"export_failed: {e}"
