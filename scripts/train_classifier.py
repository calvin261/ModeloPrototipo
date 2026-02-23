"""Train the static sign classifier (MLP) on landmark data."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hand_model.classifier import SignClassifierMLP
from hand_model.config import ClassifierConfig
from hand_model.data import LandmarkDataset
from hand_model.utils import ensure_dir, augment_landmarks, mirror_landmarks


class AugmentedLandmarkDataset(LandmarkDataset):
    """LandmarkDataset with aggressive online data augmentation.

    Per original sample produces:
      - aug_factor copies with rotation/scale/translation/noise
      - 1 mirrored copy (simulates opposite-hand perspective)

    With aug_factor=5: each sample -> 7 training samples (1 orig + 5 aug + 1 mirror).
    """

    def __init__(self, csv_path, label_map=None, augment=True, aug_factor=5):
        super().__init__(csv_path, label_map)
        self.augment = augment
        self.aug_factor = aug_factor
        self.rng = np.random.default_rng(42)

        if augment:
            orig_features = self.features.copy()
            orig_indices = self.label_indices.copy()
            orig_labels = self.labels.copy()
            all_features = [orig_features]
            all_indices = [orig_indices]
            all_labels = [orig_labels]

            # Geometric + noise augmentations
            for _ in range(aug_factor):
                aug_features = np.array(
                    [augment_landmarks(f, self.rng) for f in orig_features]
                )
                all_features.append(aug_features)
                all_indices.append(orig_indices)
                all_labels.append(orig_labels)

            # Mirror augmentation (simulates mirrored camera / other hand)
            mirrored_features = np.array(
                [mirror_landmarks(f) for f in orig_features]
            )
            all_features.append(mirrored_features)
            all_indices.append(orig_indices)
            all_labels.append(orig_labels)

            self.features = np.concatenate(all_features, axis=0)
            self.label_indices = np.concatenate(all_indices, axis=0)
            self.labels = np.concatenate(all_labels, axis=0)


def train(cfg: ClassifierConfig, progress_callback=None):
    """Train the MLP classifier.

    Improvements:
    - Mirror augmentation (aug_factor=5 + 1 mirror = 7x per sample)
    - CosineAnnealingLR scheduler (smooth LR decay)
    - Label smoothing=0.1 (reduces overconfidence on small datasets)
    - Weight decay (L2 regularisation)
    - Gradient clipping (stable training with BatchNorm)
    - Early stopping with patience=20

    Args:
        cfg: Training configuration.
        progress_callback: Optional callable(epoch, total, loss, accuracy, best_acc).
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load datasets
    train_ds = AugmentedLandmarkDataset(
        cfg.train_csv, augment=True, aug_factor=5
    )
    val_ds = LandmarkDataset(cfg.val_csv, label_map=train_ds.label_map)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False
    )

    num_classes = len(train_ds.label_map)
    print(f"Classes: {num_classes}  |  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    model = SignClassifierMLP(
        input_size=cfg.input_size,
        hidden_sizes=cfg.hidden_sizes,
        num_classes=num_classes,
        dropout=cfg.dropout,
    )

    # Adam with L2 regularisation
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4
    )
    # Cosine decay: LR goes from learning_rate → ~0 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.01
    )
    # Label smoothing reduces overconfidence when data is scarce
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    patience = 20
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping prevents instability with BatchNorm + small batches
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * features.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_ds)

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
            _save_checkpoint(model, train_ds.label_map, cfg)
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

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")
    print(f"Model saved to: {cfg.model_save_path}")


def _save_checkpoint(model, label_map, cfg):
    ensure_dir(Path(cfg.model_save_path).parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_map": label_map,
            "config": cfg,
        },
        cfg.model_save_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train static sign classifier MLP."
    )
    parser.add_argument(
        "--train-csv", default=None,
        help="Path to training CSV (default: from config)",
    )
    parser.add_argument(
        "--val-csv", default=None,
        help="Path to validation CSV (default: from config)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = ClassifierConfig()
    if args.train_csv:
        cfg.train_csv = args.train_csv
    if args.val_csv:
        cfg.val_csv = args.val_csv
    if args.epochs:
        cfg.epochs = args.epochs
    if args.lr:
        cfg.learning_rate = args.lr
    if args.batch_size:
        cfg.batch_size = args.batch_size

    train(cfg)


if __name__ == "__main__":
    main()
