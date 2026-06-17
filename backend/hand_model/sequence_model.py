from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class SignSequenceModel(nn.Module):
    """Bidirectional GRU with attention for dynamic sign recognition.

    Takes a sequence of landmark vectors (T, 63) and classifies
    the sign/word being performed.
    """

    def __init__(
        self,
        input_size: int = 63,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 50,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        feat_size = hidden_size * direction_factor

        self.attention = nn.Linear(feat_size, 1)
        self.classifier = nn.Sequential(
            nn.Linear(feat_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        output, _ = self.gru(x)  # (batch, seq_len, hidden*2)

        # Attention pooling
        attn_weights = torch.softmax(
            self.attention(output), dim=1
        )  # (batch, seq_len, 1)
        context = torch.sum(output * attn_weights, dim=1)  # (batch, feat_size)

        return self.classifier(context)


class DynamicSignRecognizer:
    """Inference wrapper for sequence-based sign recognition.

    Buffers landmark frames and runs the GRU model when the buffer
    is full or when the hand disappears (end of sign).
    """

    def __init__(
        self,
        model_path: str,
        landmark_extractor,
        max_seq_len: int = 60,
    ):
        self.extractor = landmark_extractor
        self.max_seq_len = max_seq_len

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.vocabulary = checkpoint["vocabulary"]
        self.inv_vocabulary = {v: k for k, v in self.vocabulary.items()}
        cfg = checkpoint["config"]

        self.model = SignSequenceModel(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_classes=cfg.num_classes,
            dropout=0.0,  # No dropout at inference
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Frame buffer
        self.buffer = []
        self.no_hand_count = 0
        self.NO_HAND_THRESHOLD = 15  # frames without hand to trigger classification

    def predict(self, frame: np.ndarray) -> Optional[Tuple[str, float]]:
        """Process a frame and return prediction when a sign sequence ends.

        Returns (word, confidence) when a complete sequence is classified,
        or None while still buffering frames.
        """
        landmarks = self.extractor.extract(frame)

        if landmarks is not None:
            self.no_hand_count = 0
            self.buffer.append(landmarks)
            # If buffer is full, classify
            if len(self.buffer) >= self.max_seq_len:
                return self._classify_buffer()
        else:
            self.no_hand_count += 1
            # If hand disappeared after collecting frames, classify
            if self.no_hand_count >= self.NO_HAND_THRESHOLD and len(self.buffer) > 10:
                return self._classify_buffer()

        return None

    def _classify_buffer(self) -> Optional[Tuple[str, float]]:
        if not self.buffer:
            return None

        seq = np.array(self.buffer, dtype=np.float32)

        # Pad or truncate
        if len(seq) > self.max_seq_len:
            seq = seq[: self.max_seq_len]
        elif len(seq) < self.max_seq_len:
            pad = np.zeros(
                (self.max_seq_len - len(seq), 63), dtype=np.float32
            )
            seq = np.concatenate([seq, pad], axis=0)

        self.buffer.clear()

        with torch.no_grad():
            tensor = torch.tensor(seq).unsqueeze(0)  # (1, seq_len, 63)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, idx = torch.max(probs, dim=1)

        word = self.inv_vocabulary.get(idx.item(), "???")
        return word, confidence.item()
