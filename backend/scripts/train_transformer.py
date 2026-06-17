"""Train the Sign Transformer (SPOTER-style) on hand-landmark sequences.

This script is the F3 counterpart of `train_sequence.py` for the new
architecture. It uses the same dataset loaders and augmentation pipeline
but routes data through a (B, T, 21, 3) reshape so the transformer can
use its per-joint embeddings.

Differences vs `train_sequence.py`:
  - Model: SignTransformer (joint+channel embeddings, 4-layer encoder)
  - Default aug_factor=20, 12-technique extended pipeline, cutmix p=0.3
  - Default max_seq_len=60, embed_dim=128, num_layers=4, nhead=4
  - Same LOSO/held-out-by-signer evaluation support

Usage (from backend/):
    python -m scripts.train_transformer
    python -m scripts.train_transformer --epochs 80 --batch-size 32
    python -m scripts.train_transformer --splits-json data/sequences/splits.json
    python -m scripts.train_transformer --aug-factor 30 --cutmix-prob 0.4
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from hand_model.augmentations import (
    DEFAULT_TECHNIQUES,
    augment_sequence as _aug_extended,
    augment_sequence_with_cutmix,
)
from hand_model.config import SequenceConfig
from hand_model.data import SequenceDataset
from hand_model.spoter_model import SignTransformer
from hand_model.utils import ensure_dir


class ReshapingSequenceDataset(Dataset):
    """Wraps a SequenceDataset and reshapes each (T, 63) sample to (T, 21, 3).

    Also returns a padding mask so the transformer can ignore the zero-padded
    tail. The mask is (T,) with True where the value is exactly zero (i.e.
    padding). Sequences shorter than max_seq_len are right-padded with zeros
    inside `SequenceDataset.__getitem__`; longer ones are truncated.
    """

    def __init__(self, base: SequenceDataset, eps: float = 1e-6):
        self.base = base
        self.eps = eps

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        flat, label = self.base[idx]                 # (T, 63), scalar
        T, D = flat.shape
        reshaped = flat.reshape(T, 21, 3)
        # Build padding mask: True where ALL 63 dims are ~0 (i.e. pure pad).
        flat_abs = flat.abs().sum(dim=1)              # (T,)
        is_padded = flat_abs < self.eps
        return reshaped, is_padded, label


def collate_reshape(batch):
    """Collate (T, 21, 3) + (T,) mask + label into batched tensors."""
    seqs, masks, labels = zip(*batch)
    seqs = torch.stack(seqs, dim=0)        # (B, T, 21, 3)
    masks = torch.stack(masks, dim=0)      # (B, T)
    labels = torch.tensor(labels, dtype=torch.long)
    return seqs, masks, labels


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    conf_sum = 0.0
    with torch.no_grad():
        for seqs, masks, labels in loader:
            seqs = seqs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            logits = model(seqs, padding_mask=masks)
            probs = torch.softmax(logits, dim=1)
            conf, preds = torch.max(probs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            conf_sum += conf.sum().item()
    return (correct / total if total else 0.0, conf_sum / total if total else 0.0)


def train(
    cfg: SequenceConfig,
    progress_callback=None,
    splits_json: str | None = None,
    evaluate_test: bool = True,
) -> dict:
    """Train the Sign Transformer.

    Returns a dict with final metrics (val_acc, test_acc, etc.).
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    with open(cfg.vocabulary_path, encoding="utf-8") as f:
        vocabulary = json.load(f)
    num_classes = len(vocabulary)

    base_full = SequenceDataset(
        sequences_dir=cfg.sequences_dir,
        vocabulary=vocabulary,
        max_seq_len=cfg.max_seq_len,
    )
    if len(base_full) == 0:
        return {"status": "no_data"}

    test_signers_set = None
    if splits_json and Path(splits_json).exists():
        with open(splits_json, encoding="utf-8") as f:
            splits = json.load(f)
        train_signers = set(splits["train_signers"])
        val_signers = set(splits["val_signers"])
        test_signers_set = set(splits["test_signers"])
        train_base = SequenceDataset(cfg.sequences_dir, vocabulary, cfg.max_seq_len, signers=train_signers)
        val_base = SequenceDataset(cfg.sequences_dir, vocabulary, cfg.max_seq_len, signers=val_signers)
        test_base = SequenceDataset(cfg.sequences_dir, vocabulary, cfg.max_seq_len, signers=test_signers_set)
        print(
            f"Splits: train={len(train_base)}  val={len(val_base)}  test={len(test_base)}  "
            f"(signers train={len(train_signers)}, val={len(val_signers)}, test={len(test_signers_set)})"
        )
    else:
        val_size = max(1, int(len(base_full) * 0.2))
        train_size = len(base_full) - val_size
        train_base, val_base = random_split(
            base_full, [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        test_base = None
        test_signers_set = None
        print(f"Splits (random 80/20): train={train_size}  val={val_size}")

    # Augmentation. Because the transformer needs (T, 21, 3), we augment the
    # flat (T, 63) sequence and reshape at the end.
    aug_factor = cfg.aug_factor
    use_extended = cfg.use_extended_augmentations
    cutmix_prob = cfg.cutmix_prob

    # Pre-index by class for cutmix.
    by_class: dict[int, list[np.ndarray]] = {}
    for i in range(len(train_base)):
        seq, lbl = train_base[i]
        by_class.setdefault(lbl.item(), []).append(seq.numpy())

    aug_seqs, aug_labels = [], []
    for i in range(len(train_base)):
        seq_np, lbl = train_base[i]
        seq_np = seq_np.numpy()
        lbl = lbl.item()
        aug_seqs.append(seq_np)
        aug_labels.append(lbl)

    for _ in range(aug_factor):
        for i in range(len(train_base)):
            seq_np, lbl = train_base[i]
            seq_np = seq_np.numpy()
            lbl = lbl.item()
            if cutmix_prob > 0 and rng.random() < cutmix_prob:
                partners = by_class[lbl]
                if len(partners) > 1:
                    partner = partners[rng.integers(0, len(partners))]
                    out = augment_sequence_with_cutmix(
                        seq_np, partner, rng, max_seq_len=cfg.max_seq_len
                    )
                else:
                    out = _aug_extended(seq_np, rng, DEFAULT_TECHNIQUES)
            else:
                out = _aug_extended(seq_np, rng, DEFAULT_TECHNIQUES)
            aug_seqs.append(out)
            aug_labels.append(lbl)

    # Reshape all to (T, 21, 3) and build padding masks.
    aug_seqs_t = torch.from_numpy(np.stack(aug_seqs)).float()  # (N, T, 63)
    aug_seqs_t = aug_seqs_t.reshape(len(aug_seqs), cfg.max_seq_len, 21, 3)
    aug_masks_t = (aug_seqs_t.abs().sum(dim=(2, 3)) < 1e-6)   # (N, T) bool
    aug_labels_t = torch.tensor(aug_labels, dtype=torch.long)

    # Custom dataset that returns (seq, mask, label) triples.
    class _TensorsDS(Dataset):
        def __init__(self, seqs, masks, labels):
            self.seqs = seqs
            self.masks = masks
            self.labels = labels
        def __len__(self):
            return len(self.seqs)
        def __getitem__(self, i):
            return self.seqs[i], self.masks[i], self.labels[i]

    train_ds = _TensorsDS(aug_seqs_t, aug_masks_t, aug_labels_t)
    val_ds = ReshapingSequenceDataset(val_base)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_reshape, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_reshape,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes: {num_classes}  |  Train (aug): {len(train_ds)}  |  Val: {len(val_ds)}")
    print(
        f"Aug pipeline: {'12-technique extended' if use_extended else 'legacy 3-technique'} "
        f"(factor={aug_factor}, cutmix_prob={cutmix_prob})"
    )

    model = SignTransformer(
        n_joints=21,
        n_channels=3,
        num_classes=num_classes,
        embed_dim=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=cfg.dropout,
        max_seq_len=cfg.max_seq_len + 1,
        use_cls=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: SignTransformer, {n_params:,} params, embed=128, layers=4, nhead=4")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.01
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    patience = 20
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for seqs, masks, labels in train_loader:
            seqs = seqs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs, padding_mask=masks)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
        scheduler.step()
        avg_loss = total_loss / max(len(train_ds), 1)
        train_time = time.time() - t0

        val_acc, val_conf = _evaluate(model, val_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            _save_checkpoint(model, vocabulary, cfg)
        else:
            epochs_no_improve += 1

        if progress_callback:
            progress_callback(epoch, cfg.epochs, avg_loss, val_acc, best_acc)

        # Print every epoch (unbuffered) so we can see the trajectory
        # even if the process gets killed.
        print(
            f"Epoch {epoch:3d}/{cfg.epochs}  "
            f"Loss: {avg_loss:.4f}  "
            f"Val Acc: {val_acc:.4f}  "
            f"Best: {best_acc:.4f}  "
            f"Conf: {val_conf:.3f}  "
            f"LR: {scheduler.get_last_lr()[0]:.5f}  "
            f"Time: {train_time:.1f}s",
            flush=True,
        )

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).", flush=True)
            break

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")
    print(f"Model saved to: {cfg.model_save_path}")

    result = {
        "status": "ok",
        "model_type": "transformer",
        "best_val_acc": float(best_acc),
        "test_acc": None,
        "test_mean_conf": None,
        "test_signers": sorted(test_signers_set) if test_signers_set else None,
        "n_params": n_params,
    }
    if evaluate_test and test_base is not None and len(test_base) > 0:
        ckpt = torch.load(cfg.model_save_path, map_location=device, weights_only=False)
        best_model = SignTransformer(
            n_joints=21, n_channels=3, num_classes=num_classes,
            embed_dim=128, nhead=4, num_layers=4, dim_feedforward=256,
            dropout=0.0, max_seq_len=cfg.max_seq_len + 1, use_cls=True,
        ).to(device)
        best_model.load_state_dict(ckpt["model_state_dict"])
        test_ds = ReshapingSequenceDataset(test_base)
        test_loader = DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False,
            collate_fn=collate_reshape,
        )
        test_acc, test_conf = _evaluate(best_model, test_loader, device)
        result["test_acc"] = float(test_acc)
        result["test_mean_conf"] = float(test_conf)
        print()
        print("=" * 60)
        print(f"  TEST (held-out signers: {result['test_signers']})")
        print(f"  Test accuracy:    {test_acc * 100:.2f}%  ({int(test_acc * len(test_ds))}/{len(test_ds)})")
        print(f"  Mean confidence:  {test_conf:.3f}")
        print("=" * 60)
    return result


def _save_checkpoint(model, vocabulary, cfg):
    ensure_dir(Path(cfg.model_save_path).parent)
    torch.save(
        {
            "model_type": "transformer",
            "model_state_dict": model.state_dict(),
            "vocabulary": vocabulary,
            "config": cfg,
        },
        cfg.model_save_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Sign Transformer (SPOTER-style).")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequences-dir", default=None)
    parser.add_argument("--vocabulary", default=None)
    parser.add_argument("--splits-json", default=None)
    parser.add_argument("--no-test-eval", action="store_true")
    parser.add_argument("--aug-factor", type=int, default=None)
    parser.add_argument("--cutmix-prob", type=float, default=None)
    parser.add_argument("--save-path", default=None,
                        help="Override model save path (default: cfg.model_save_path)")
    args = parser.parse_args()

    cfg = SequenceConfig()
    # Transformer training needs a different default save path
    if args.save_path:
        cfg.model_save_path = args.save_path
    else:
        cfg.model_save_path = "models/sequence/sign_transformer.pth"
    # Default to the 12-class active vocabulary (overrides the 50-class
    # default in SequenceConfig) unless the caller explicitly overrides it.
    if not args.vocabulary:
        cfg.vocabulary_path = str(PROJECT_ROOT / "data" / "sequences" / "vocabulary.json")
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
    if args.aug_factor is not None:
        cfg.aug_factor = args.aug_factor
    if args.cutmix_prob is not None:
        cfg.cutmix_prob = args.cutmix_prob

    train(
        cfg,
        splits_json=args.splits_json,
        evaluate_test=not args.no_test_eval,
    )


if __name__ == "__main__":
    raise SystemExit(main())
