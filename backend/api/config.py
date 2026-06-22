"""Backend configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ServerConfig:
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = int(os.environ.get("PORT", "7860"))
    api_prefix: str = "/api/v1"

    # Paths relative to project root (overridden by MODELS_DIR env var when set,
    # e.g. on Hugging Face Spaces where /data/models is mounted from the Hub).
    project_root: str = field(default_factory=lambda: str(Path(__file__).parent.parent))
    data_landmarks_dir: str = "data/landmarks"
    data_sequences_dir: str = "data/sequences"
    models_classifier_dir: str = "models/classifier"
    models_sequence_dir: str = "models/sequence"
    models_exports_dir: str = os.environ.get("MODELS_DIR", "models/exports")
    datasets_dir: str = "datasets"

    # Label files
    classifier_labels_path: str = "datasets/lsec_alphabet_labels.json"
    vocabulary_path: str = "datasets/lsec_vocabulary.json"

    def resolve(self, relative_path: str) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            return path
        return Path(self.project_root) / relative_path
