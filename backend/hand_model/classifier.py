from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class SignClassifierMLP(nn.Module):
    """MLP with BatchNorm for static sign classification from landmark vectors.

    Architecture: Linear → BN → ReLU → Dropout (x2 hidden layers) → Linear output.
    BatchNorm improves generalisation with small datasets by stabilising activations.
    """

    def __init__(
        self,
        input_size: int = 63,
        hidden_sizes: tuple = (256, 128),
        num_classes: int = 27,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[1], num_classes),
        )

    def forward(self, x):
        return self.net(x)


class StaticSignRecognizer:
    """Inference wrapper that combines landmark extraction with MLP
    classification for real-time static sign recognition."""

    def __init__(self, model_path: str, landmark_extractor):
        self.extractor = landmark_extractor
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.label_map = checkpoint["label_map"]
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        cfg = checkpoint["config"]
        self.model = SignClassifierMLP(
            input_size=cfg.input_size,
            hidden_sizes=cfg.hidden_sizes,
            num_classes=cfg.num_classes,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, frame: np.ndarray) -> Optional[Tuple[str, float]]:
        """Predict the sign shown in a BGR frame.

        Returns (label, confidence) or None if no hand is detected.
        """
        landmarks = self.extractor.extract(frame)
        if landmarks is None:
            return None
        with torch.no_grad():
            tensor = torch.tensor(landmarks).unsqueeze(0)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, idx = torch.max(probs, dim=1)
        label = self.inv_label_map[idx.item()]
        return label, confidence.item()
