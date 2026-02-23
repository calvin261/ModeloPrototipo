from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    weights: str = "models/pretrained/yolov8n.pt"
    data: str = "datasets/hands.yaml"
    imgsz: int = 640
    epochs: int = 100
    batch: int = -1
    amp: bool = True
    seed: int = 42
    project: str = "models/finetuned"


@dataclass
class ClassifierConfig:
    input_size: int = 63
    hidden_sizes: tuple = (256, 128)  # Wider network for better feature extraction
    num_classes: int = 27
    dropout: float = 0.3
    learning_rate: float = 1e-3
    epochs: int = 150  # More epochs; early stopping prevents overfitting
    batch_size: int = 64
    seed: int = 42
    model_save_path: str = "models/classifier/sign_mlp.pth"
    labels_path: str = "datasets/lsec_alphabet_labels.json"
    train_csv: str = "data/landmarks/lsec_alphabet_train.csv"
    val_csv: str = "data/landmarks/lsec_alphabet_val.csv"


@dataclass
class SequenceConfig:
    input_size: int = 63
    hidden_size: int = 128
    num_layers: int = 2
    num_classes: int = 50
    max_seq_len: int = 60
    dropout: float = 0.3
    learning_rate: float = 1e-3
    epochs: int = 120  # More epochs; early stopping (patience=15) prevents overfitting
    batch_size: int = 32
    seed: int = 42
    model_save_path: str = "models/sequence/sign_gru.pth"
    vocabulary_path: str = "datasets/lsec_vocabulary.json"
    sequences_dir: str = "data/sequences"


@dataclass
class AppConfig:
    mode: str = "static"
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    confidence_threshold: float = 0.7
    confirm_frames: int = 10
    tts_rate: int = 150
    tts_lang: str = "es"
    landmark_model: str = "models/mediapipe/hand_landmarker.task"
    classifier_model: str = "models/classifier/sign_mlp.pth"
    sequence_model: str = "models/sequence/sign_gru.pth"
