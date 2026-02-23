from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def get_processed_paths(base="data/processed"):
    base_path = Path(base)
    return {
        "images": base_path / "images",
        "labels": base_path / "labels",
    }


class LandmarkDataset(Dataset):
    """PyTorch dataset for static sign landmark CSVs.

    Each CSV row has: label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20
    """

    def __init__(self, csv_path: str, label_map: dict = None):
        df = pd.read_csv(csv_path)
        self.labels = df.iloc[:, 0].values
        self.features = df.iloc[:, 1:].values.astype(np.float32)

        if label_map is None:
            unique = sorted(set(self.labels))
            self.label_map = {label: i for i, label in enumerate(unique)}
        else:
            self.label_map = label_map

        self.label_indices = np.array(
            [self.label_map[l] for l in self.labels]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]),
            torch.tensor(self.label_indices[idx], dtype=torch.long),
        )


class SequenceDataset(Dataset):
    """PyTorch dataset for dynamic sign sequences (.npy files).

    Each .npy file has shape (T, 63) where T = number of frames.
    Sequences are padded/truncated to max_seq_len.
    """

    def __init__(self, sequences_dir: str, vocabulary: dict, max_seq_len: int = 60):
        self.max_seq_len = max_seq_len
        self.samples = []
        base = Path(sequences_dir)
        for word, idx in vocabulary.items():
            word_dir = base / word
            if not word_dir.exists():
                continue
            for f in word_dir.glob("*.npy"):
                self.samples.append((str(f), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = np.load(path).astype(np.float32)

        if len(seq) > self.max_seq_len:
            seq = seq[: self.max_seq_len]
        elif len(seq) < self.max_seq_len:
            pad = np.zeros(
                (self.max_seq_len - len(seq), seq.shape[1]),
                dtype=np.float32,
            )
            seq = np.concatenate([seq, pad], axis=0)

        return (
            torch.tensor(seq),
            torch.tensor(label, dtype=torch.long),
        )
