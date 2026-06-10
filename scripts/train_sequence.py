"""Train the dynamic sign sequence model (GRU) on recorded sequences.

Supports two split modes:
  - Default: 80/20 random split (old behaviour).
  - Signer-stratified: pass --splits-json pointing at the JSON produced
    by scripts.build_sequence_splits.py. This creates 3 disjoint signer
    sets (train/val/test) and evaluates on the held-out test signers
    at the end. This is the realistic evaluation for sign language.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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

    seq = seq + rng.normal(0, 0.01, size=seq.shape).astype(np.float32)

    warp_factor = rng.uniform(0.8, 1.2)
    new_len = max(2, int(T * warp_factor))
    idx_orig = np.linspace(0, T - 1, new_len)
    idx_new = np.linspace(0, new_len - 1, T)
    warped = np.zeros_like(seq)
    for d in range(D):
        warped[:, d] = np.interp(
            idx_new,
            np.arange(new_len),
            np.interp(idx_orig, np.arange(T), seq[:, d]),
        )
    seq = warped

    angle = rng.uniform(-20, 20) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    coords = seq.reshape(T, 21, 3)
    xy = coords[:, :, :2].copy()
    coords[:, :, 0] = xy[:, :, 0] * cos_a - xy[:, :, 1] * sin_a
    coords[:, :, 1] = xy[:, :, 0] * sin_a + xy[:, :, 1] * cos_a
    return coords.reshape(T, D).astype(np.float32)


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Return (accuracy, mean_confidence) on `loader`."""
    model.eval()
    correct = 0
    total = 0
    conf_sum = 0.0
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            logits = model(sequences)
            probs = torch.softmax(logits, dim=1)
            conf, preds = torch.max(probs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            conf_sum += conf.sum().item()
    acc = correct / total if total > 0 else 0.0
    mean_conf = conf_sum / total if total > 0 else 0.0
    return acc, mean_conf


def train(
    cfg: SequenceConfig,
    progress_callback=None,
    splits_json: str | None = None,
    evaluate_test: bool = True,
) -> dict:
    """Train the GRU sequence model.

    Returns a dict with final metrics (val_acc, test_acc if available,
    per_split_counts, etc.).
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    with open(cfg.vocabulary_path) as f:
        vocabulary = json.load(f)

    num_classes = len(vocabulary)

    full_dataset = SequenceDataset(
        sequences_dir=cfg.sequences_dir,
        vocabulary=vocabulary,
        max_seq_len=cfg.max_seq_len,
    )
    if len(full_dataset) == 0:
        print("No sequence data found. Use collect_data.py to record sequences.")
        return {"status": "no_data"}

    # Build train/val/test by signer if --splits-json given
    test_signers_set: set[str] | None = None
    if splits_json and Path(splits_json).exists():
        with open(splits_json) as f:
            splits = json.load(f)
        # The vocabulary in splits.json may be a subset of cfg.vocabulary_path;
        # build the actual vocabulary from the sequences that are listed in
        # the splits, mapped to the same indices as cfg.vocabulary_path.
        train_signers = set(splits["train_signers"])
        val_signers = set(splits["val_signers"])
        test_signers_set = set(splits["test_signers"])
        train_ds = SequenceDataset(
            cfg.sequences_dir, vocabulary, cfg.max_seq_len, signers=train_signers
        )
        val_ds = SequenceDataset(
            cfg.sequences_dir, vocabulary, cfg.max_seq_len, signers=val_signers
        )
        test_ds = SequenceDataset(
            cfg.sequences_dir, vocabulary, cfg.max_seq_len, signers=test_signers_set
        )
        print(
            f"Splits: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  "
            f"(signers train={len(train_signers)}, val={len(val_signers)}, "
            f"test={len(test_signers_set)})"
        )
    else:
        # Fallback: random 80/20, no test
        val_size = max(1, int(len(full_dataset) * 0.2))
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        test_ds = None
        test_signers_set = None
        print(f"Splits (random 80/20): train={train_size}  val={val_size}")

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
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes: {num_classes}  |  Train (aug): {aug_train_size}  |  Val: {len(val_ds)}")

    model = SignSequenceModel(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        num_classes=num_classes,
        dropout=cfg.dropout,
    ).to(device)
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
            sequences = sequences.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * sequences.size(0)
        scheduler.step()
        avg_loss = total_loss / aug_train_size

        val_acc, _ = _evaluate(model, val_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            _save_checkpoint(model, vocabulary, cfg)
        else:
            epochs_no_improve += 1

        if progress_callback:
            progress_callback(epoch, cfg.epochs, avg_loss, val_acc, best_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{cfg.epochs}  "
                f"Loss: {avg_loss:.4f}  "
                f"Val Acc: {val_acc:.4f}  "
                f"Best: {best_acc:.4f}  "
                f"LR: {scheduler.get_last_lr()[0]:.5f}"
            )

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")
    print(f"Model saved to: {cfg.model_save_path}")

    # Reload best checkpoint and evaluate on test (if any)
    result = {
        "status": "ok",
        "best_val_acc": float(best_acc),
        "test_acc": None,
        "test_mean_conf": None,
        "test_signers": sorted(test_signers_set) if test_signers_set else None,
    }
    if evaluate_test and test_ds is not None and len(test_ds) > 0:
        # Load best checkpoint
        ckpt = torch.load(cfg.model_save_path, map_location=device, weights_only=False)
        best_model = SignSequenceModel(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_classes=num_classes,
            dropout=0.0,
        ).to(device)
        best_model.load_state_dict(ckpt["model_state_dict"])
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
        test_acc, test_conf = _evaluate(best_model, test_loader, device)
        result["test_acc"] = float(test_acc)
        result["test_mean_conf"] = float(test_conf)
        print()
        print("=" * 60)
        print(f"  TEST (held-out signers: {result['test_signers']})")
        print(f"  Test accuracy:       {test_acc * 100:.2f}%  ({int(test_acc * len(test_ds))}/{len(test_ds)})")
        print(f"  Mean confidence:    {test_conf:.3f}")
        print("=" * 60)
    return result


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
    parser.add_argument(
        "--splits-json",
        default=None,
        help="Path to splits.json from scripts.build_sequence_splits. "
        "Enables held-out-by-signer evaluation.",
    )
    parser.add_argument(
        "--no-test-eval",
        action="store_true",
        help="Skip the final test-set evaluation (use only val).",
    )
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

    train(
        cfg,
        splits_json=args.splits_json,
        evaluate_test=not args.no_test_eval,
    )


if __name__ == "__main__":
    main()
