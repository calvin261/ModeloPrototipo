"""Model registry for versioning and metadata management."""

import json
from datetime import datetime
from pathlib import Path
from threading import Lock


class ModelRegistry:
    """Tracks model versions, accuracy, and file locations."""

    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._data = self._load()

    def _load(self) -> dict:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {"classifier": [], "sequence": []}

    def _save(self):
        with open(self.registry_path, "w") as f:
            json.dump(self._data, f, indent=2)

    def register(self, model_type: str, entry: dict):
        """Register a new model version."""
        with self._lock:
            if model_type not in self._data:
                self._data[model_type] = []
            self._data[model_type].append(entry)
            self._save()

    def get_latest(self, model_type: str) -> dict | None:
        """Get the latest registered model for a type."""
        entries = self._data.get(model_type, [])
        if not entries:
            return None
        return entries[-1]

    def get_all(self, model_type: str) -> list[dict]:
        """Get all versions for a model type."""
        return self._data.get(model_type, [])

    def list_all(self) -> list[dict]:
        """List all models across all types."""
        result = []
        for model_type, entries in self._data.items():
            for entry in entries:
                result.append({"model_type": model_type, **entry})
        return result
