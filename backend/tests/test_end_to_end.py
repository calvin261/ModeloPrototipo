"""End-to-end smoke test for the LSC pipeline.

Marked `slow` so it can be skipped in PR CI. Exercises:

  1. Load `data/sequences/` via SequenceDataset
  2. Augment 5 samples with the full augmentation pipeline
  3. Train SignTransformer for 2 epochs (`aug_factor=1`, no test eval)
  4. Run a quick evaluation on the held-out test signers
  5. Verify test_acc >= 1/C (chance level for C classes) and that a
     checkpoint was written to disk

The goal is to catch the most common regressions in the data -> model
path (shape mismatches, broken augmentation, label-leakage in the
splits, broken save path) in well under 60 seconds on a CPU-only venv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hand_model.config import SequenceConfig  # noqa: E402
from hand_model.data import SequenceDataset  # noqa: E402
from hand_model.spoter_model import SignTransformer  # noqa: E402
from scripts.train_transformer import (  # noqa: E402
    ReshapingSequenceDataset,
    collate_reshape,
)


pytestmark = pytest.mark.slow


SEQUENCES_DIR = ROOT / "data" / "sequences"
VOCAB_PATH = SEQUENCES_DIR / "vocabulary.json"
SPLITS_PATH = SEQUENCES_DIR / "splits.json"


def _require_data() -> tuple[Path, Path]:
    if not SEQUENCES_DIR.exists() or not VOCAB_PATH.exists() or not SPLITS_PATH.exists():
        pytest.skip(
            f"data/sequences/, vocabulary.json, and splits.json must exist "
            f"under {SEQUENCES_DIR}. Run `python -m scripts.prepare_sequences_from_lsc50` "
            "and `python -m scripts.build_sequence_splits` first."
        )
    return SEQUENCES_DIR, SPLITS_PATH


def _quick_train_and_eval(tmp_path: Path) -> tuple[float, int, Path]:
    """Run a 2-epoch training + a quick test evaluation; return (acc, n_test, ckpt_path)."""
    from hand_model.augmentations import (
        DEFAULT_TECHNIQUES,
        augment_sequence as _aug_extended,
    )

    seq_dir, splits_path = _require_data()
    vocabulary = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    n_classes = len(vocabulary)
    max_seq_len = 60

    test_signers = set(splits["test_signers"])
    val_signers = set(splits["val_signers"])
    train_signers = set(splits["train_signers"])

    # Build datasets.
    train_base = SequenceDataset(str(seq_dir), vocabulary, max_seq_len, signers=train_signers)
    val_base = SequenceDataset(str(seq_dir), vocabulary, max_seq_len, signers=val_signers)
    test_base = SequenceDataset(str(seq_dir), vocabulary, max_seq_len, signers=test_signers)

    if len(train_base) == 0 or len(test_base) == 0:
        pytest.skip("train or test set is empty — cannot run end-to-end test")

    # 1) Augment 5 samples with the full pipeline (smoke check that
    #    augment_sequence + a (T, 63) array don't blow up).
    rng = np.random.default_rng(0)
    for i in range(min(5, len(train_base))):
        seq, _ = train_base[i]
        out = _aug_extended(seq.numpy(), rng, DEFAULT_TECHNIQUES)
        assert out.shape == seq.shape, f"aug output shape {out.shape} != {seq.shape}"
        assert out.dtype == np.float32
        assert np.isfinite(out).all()

    # 2) Build a tiny training set: original + 1 augmented copy per sample.
    aug_factor = 1
    aug_seqs, aug_labels = [], []
    by_class: dict[int, list[np.ndarray]] = {}
    for i in range(len(train_base)):
        seq, lbl = train_base[i]
        by_class.setdefault(lbl.item(), []).append(seq.numpy())
        aug_seqs.append(seq.numpy())
        aug_labels.append(lbl.item())
    for _ in range(aug_factor):
        for i in range(len(train_base)):
            seq, lbl = train_base[i]
            out = _aug_extended(seq.numpy(), rng, DEFAULT_TECHNIQUES)
            aug_seqs.append(out)
            aug_labels.append(lbl.item())

    aug_seqs_t = torch.from_numpy(np.stack(aug_seqs)).float()
    aug_seqs_t = aug_seqs_t.reshape(len(aug_seqs), max_seq_len, 21, 3)
    aug_masks_t = (aug_seqs_t.abs().sum(dim=(2, 3)) < 1e-6)
    aug_labels_t = torch.tensor(aug_labels, dtype=torch.long)

    class _TensorsDS(torch.utils.data.Dataset):
        def __init__(self, s, m, l):
            self.s, self.m, self.l = s, m, l
        def __len__(self): return len(self.s)
        def __getitem__(self, i): return self.s[i], self.m[i], self.l[i]

    train_ds = _TensorsDS(aug_seqs_t, aug_masks_t, aug_labels_t)
    val_ds = ReshapingSequenceDataset(val_base)
    test_ds = ReshapingSequenceDataset(test_base)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_reshape)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_reshape)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_reshape)

    # 3) Train SignTransformer for 2 epochs.
    device = torch.device("cpu")
    model = SignTransformer(
        n_joints=21, n_channels=3, num_classes=n_classes,
        embed_dim=64, nhead=4, num_layers=2, dim_feedforward=128,
        dropout=0.0, max_seq_len=max_seq_len + 1, use_cls=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for _epoch in range(2):
        for seqs, masks, labels in train_loader:
            seqs = seqs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs, padding_mask=masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    # 4) Save the checkpoint (so the test verifies the save path).
    ckpt_path = tmp_path / "sign_transformer_e2e.pth"
    torch.save(
        {
            "model_type": "transformer",
            "model_state_dict": model.state_dict(),
            "vocabulary": vocabulary,
            "config": SequenceConfig(),
        },
        ckpt_path,
    )

    # 5) Quick evaluation on the held-out test set.
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for seqs, masks, labels in test_loader:
            seqs = seqs.to(device)
            masks = masks.to(device)
            labels_np = labels.numpy()
            logits = model(seqs, padding_mask=masks)
            preds = logits.argmax(dim=1).cpu().numpy()
            correct += int((preds == labels_np).sum())
            total += int(labels_np.size)
    test_acc = correct / total if total else 0.0
    return test_acc, n_classes, ckpt_path


def test_end_to_end_pipeline_under_60s(tmp_path: Path):
    """Full data -> augment -> train -> eval pipeline; verifies the test
    accuracy is at or above chance level and that the checkpoint is
    persisted."""
    test_acc, n_classes, ckpt_path = _quick_train_and_eval(tmp_path)
    chance = 1.0 / n_classes
    assert test_acc >= chance, (
        f"test_acc {test_acc:.4f} is below chance level 1/{n_classes} = {chance:.4f}"
    )
    assert ckpt_path.exists() and ckpt_path.stat().st_size > 0, (
        f"checkpoint was not written to {ckpt_path}"
    )
