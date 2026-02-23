"""Train the dynamic sign sequence model (GRU) on recorded sequences."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from hand_model.config import SequenceConfig
from hand_model.data import SequenceDataset
from hand_model.sequence_model import SignSequenceModel
from hand_model.utils import ensure_dir


def _augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Augment a (T, 63) landmark sequence with noise, time warp and rotation."""
    T, D = seq.shape

    # Per-frame Gaussian noise (preserves motion shape)
    seq = seq + rng.normal(0, 0.01, size=seq.shape).astype(np.float32)

    # Time warp: resample to 80-120% length then resize back to T
    warp_factor = rng.uniform(0.8, 1.2)
    new_len = max(2, int(T * warp_factor))
    idx_orig = np.linspace(0, T - 1, new_len)
    idx_new = np.linspace(0, new_len - 1, T)
    warped = np.zeros_like(seq)
    for d in range(D):
        warped[:, d] = np.interp(idx_new, np.arange(new_len),
                                  np.interp(idx_orig, np.arange(T), seq[:, d]))
    seq = warped

    # Global 2D rotation (+/- 20 deg) applied uniformly to all frames
    angle = rng.uniform(-20, 20) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    coords = seq.reshape(T, 21, 3)
    xy = coords[:, :, :2].copy()
    coords[:, :, 0] = xy[:, :, 0] * cos_a - xy[:, :, 1] * sin_a
    coords[:, :, 1] = xy[:, :, 0] * sin_a + xy[:, :, 1] * cos_a
    return coords.reshape(T, D).astype(np.float32)


def train(cfg: SequenceConfig, progress_callback=None):
    """Train the GRU sequence model.

    Improvements:
    - Sequence augmentation (noise, time warp, rotation) x3 per sample
    - CosineAnnealingLR scheduler
    - Label smoothing=0.1
    - Gradient clipping
    - Early stopping with patience=15

    Args:
        cfg: Training configuration.
        progress_callback: Optional callable(epoch, total, loss, accuracy, best_acc).
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # Load vocabulary
    with open(cfg.vocabulary_path) as f:
        vocabulary = json.load(f)

    num_classes = len(vocabulary)

    dataset = SequenceDataset(
        sequences_dir=cfg.sequences_dir,
        vocabulary=vocabulary,
        max_seq_len=cfg.max_seq_len,
    )

    if len(dataset) == 0:
        print("No sequence data found. Use collect_data.py to record sequences.")
        return

    # Split 80/20
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    # Offline augmentation: expand training set 4x (original + 3 augmented)
    aug_factor = 3
    all_sequences, all_labels = [], []
    for seq, lbl in train_ds:
        seq_np = seq.numpy()
        all_sequences.append(seq_np)
        all_labels.append(lbl.item())
        for _ in range(aug_factor):
            all_sequences.append(_augment_sequence(seq_np, rng))
            all_labels.append(lbl.item())

    aug_sequences = torch.tensor(np.stack(all_sequences), dtype=torch.float32)
    aug_labels = torch.tensor(all_labels, dtype=torch.long)
    aug_train_ds = torch.utils.data.TensorDataset(aug_sequences, aug_labels)
    aug_train_size = len(aug_train_ds)

    train_loader = DataLoader(
        aug_train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False
    )

    print(f"Classes: {num_classes}  |  Train: {aug_train_size} (aug)  |  Val: {val_size}")

    model = SignSequenceModel(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        num_classes=num_classes,
        dropout=cfg.dropout,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.01
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    patience = 15
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * sequences.size(0)

        scheduler.step()
        avg_loss = total_loss / aug_train_size

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                outputs = model(sequences)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total if total > 0 else 0
        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
            _save_checkpoint(model, vocabulary, cfg)
        else:
            epochs_no_improve += 1

        if progress_callback:
            progress_callback(epoch, cfg.epochs, avg_loss, acc, best_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{cfg.epochs}  "
                f"Loss: {avg_loss:.4f}  "
                f"Val Acc: {acc:.4f}  "
                f"Best: {best_acc:.4f}  "
                f"LR: {scheduler.get_last_lr()[0]:.5f}"
            )

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")
    print(f"Model saved to: {cfg.model_save_path}")


def _save_checkpoint(model, vocabulary, cfg):
    ensure_dir(Path(cfg.model_save_path).parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocabulary": vocabulary,
            "config": cfg,
        },
        cfg.model_save_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train dynamic sign sequence model (GRU)."
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequences-dir", default=None)
    parser.add_argument("--vocabulary", default=None)
    args = parser.parse_args()

    cfg = SequenceConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.lr:
        cfg.learning_rate = args.lr
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.sequences_dir:
        cfg.sequences_dir = args.sequences_dir
    if args.vocabulary:
        cfg.vocabulary_path = args.vocabulary

    train(cfg)


if __name__ == "__main__":
    main()
