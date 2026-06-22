from .config import TrainConfig, ClassifierConfig, SequenceConfig, AppConfig
from .data import get_processed_paths, LandmarkDataset, SequenceDataset
from .utils import ensure_dir, augment_landmarks, mirror_landmarks
from .landmarks import LandmarkExtractor, normalize_landmarks
from .classifier import SignClassifierMLP, StaticSignRecognizer
from .sequence_model import SignSequenceModel, DynamicSignRecognizer

# NOTE: training/export/collection helpers (which pull in ultralytics) and the
# desktop TTS/app helpers (which pull in pyttsx3) are intentionally NOT
# re-exported here so the FastAPI service can boot in a slim production image
# without those packages installed. Import them directly when running scripts/
# or the desktop CLI, e.g.:
#     from hand_model.train import train_model
#     from hand_model.export import export_model
#     from hand_model.collect import DataCollector
#     from hand_model.tts import TTSEngine
#     from hand_model.app import SignLanguageApp
__all__ = [
    # Config
    "TrainConfig",
    "ClassifierConfig",
    "SequenceConfig",
    "AppConfig",
    # Data
    "get_processed_paths",
    "LandmarkDataset",
    "SequenceDataset",
    # Landmarks
    "LandmarkExtractor",
    "normalize_landmarks",
    # Classifiers
    "SignClassifierMLP",
    "StaticSignRecognizer",
    "SignSequenceModel",
    "DynamicSignRecognizer",
    # Utils
    "ensure_dir",
    "augment_landmarks",
    "mirror_landmarks",
]
