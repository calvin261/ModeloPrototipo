"""Backend configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    api_prefix: str = "/api/v1"

    # Paths relative to project root
    project_root: str = field(default_factory=lambda: str(Path(__file__).parent.parent))
    data_landmarks_dir: str = "data/landmarks"
    data_sequences_dir: str = "data/sequences"
    models_classifier_dir: str = "models/classifier"
    models_sequence_dir: str = "models/sequence"
    models_exports_dir: str = "models/exports"
    datasets_dir: str = "datasets"

    # Label files
    classifier_labels_path: str = "datasets/lsec_alphabet_labels.json"
    vocabulary_path: str = "datasets/lsec_vocabulary.json"

    def resolve(self, relative_path: str) -> Path:
        return Path(self.project_root) / relative_path
