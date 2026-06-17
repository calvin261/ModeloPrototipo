"""Quick LOSO evaluation of an already-trained model (no retraining).

This script evaluates a single trained checkpoint on each LOSO fold
as the test set, reporting per-signer accuracy and the macro average
across all signers. Much faster than `loso_evaluate.py` (which retrains
per fold) — useful for A/B comparisons between checkpoints.

Usage:
    python -m scripts.loso_test_only --ckpt models/sequence/sign_gru.pth
    python -m scripts.loso_test_only --ckpt models/sequence/sign_transformer.pth
    python -m scripts.loso_test_only --export reports/loso_test_only.json
    python -m scripts.loso_test_only --signers 1 3 17   # only a few signers
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from hand_model.data import SequenceDataset
from hand_model.sequence_model import SignSequenceModel
from hand_model.spoter_model import SignTransformer
from scripts.train_transformer import ReshapingSequenceDataset, collate_reshape


def _load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    vocab = ckpt["vocabulary"]
    model_type = ckpt.get("model_type", "gru")
    if model_type == "transformer":
        model = SignTransformer(
            n_joints=21, n_channels=3, num_classes=len(vocab),
            embed_dim=128, nhead=4, num_layers=4, dim_feedforward=256,
            dropout=0.0, max_seq_len=cfg.max_seq_len + 1, use_cls=True,
        )
    else:
        model = SignSequenceModel(
            input_size=cfg.input_size, hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers, num_classes=len(vocab), dropout=0.0,
        )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab, cfg, model_type


@torch.no_grad()
def _eval_one(model, ds: SequenceDataset, model_type: str, vocab_inv: dict[int, str]):
    preds, labels = [], []
    if model_type == "transformer":
        rds = ReshapingSequenceDataset(ds)
        loader = DataLoader(rds, batch_size=32, shuffle=False, collate_fn=collate_reshape)
        for seqs, masks, lbls in loader:
            logits = model(seqs, padding_mask=masks)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend([vocab_inv[int(p)] for p in pred])
            labels.extend([vocab_inv[int(l)] for l in lbls.numpy()])
    else:
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        for seqs, lbls in loader:
            logits = model(seqs)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend([vocab_inv[int(p)] for p in pred])
            labels.extend([vocab_inv[int(l)] for l in lbls.numpy()])
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    return correct, len(preds), preds, labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval-only LOSO on a trained model.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--loso-splits", type=Path,
        default=PROJECT_ROOT / "data" / "sequences" / "loso_splits.json",
    )
    parser.add_argument(
        "--signers", nargs="+", default=None,
        help="Subset of test signers (default: all 18).",
    )
    parser.add_argument(
        "--export", type=Path, default=None,
    )
    args = parser.parse_args()

    if not args.ckpt.exists():
        print(f"[!] Checkpoint not found: {args.ckpt}")
        return 1
    if not args.loso_splits.exists():
        print(f"[!] LOSO splits missing: {args.loso_splits}")
        return 1

    with open(args.loso_splits, encoding="utf-8") as f:
        loso = json.load(f)
    folds = loso["folds"]
    if args.signers:
        folds = [f for f in folds if f["test_signer"] in args.signers]

    print("=" * 60)
    try:
        rel = args.ckpt.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = args.ckpt
    print(f"  Eval-only LOSO — {rel}")
    print("=" * 60)
    print(f"Folds: {len(folds)}")
    model, vocab, cfg, model_type = _load_model(args.ckpt)
    print(f"Model: {model_type}, {len(vocab)} classes, max_seq={cfg.max_seq_len}")

    per_fold = []
    accs = []
    t0 = time.time()
    vocab_inv = {v: k for k, v in vocab.items()}
    for fold in folds:
        signer = fold["test_signer"]
        ds = SequenceDataset(
            sequences_dir=str(PROJECT_ROOT / "data" / "sequences"),
            vocabulary=vocab, max_seq_len=cfg.max_seq_len,
            signers={signer},
        )
        if len(ds) == 0:
            print(f"  signer {signer}: 0 samples, skipping")
            continue
        c, n, _, _ = _eval_one(model, ds, model_type, vocab_inv)
        acc = c / n if n else 0
        accs.append(acc)
        per_fold.append({"signer": signer, "n": n, "correct": c, "acc": acc})
        elapsed = time.time() - t0
        print(
            f"  signer {signer:>3}  acc={acc * 100:>6.2f}%  ({c}/{n})  "
            f"running mean={np.mean(accs) * 100:.2f}%  elapsed={elapsed:.1f}s"
        )

    if accs:
        import statistics
        mean = statistics.mean(accs)
        std = statistics.stdev(accs) if len(accs) > 1 else 0
        print()
        print("=" * 60)
        print(f"  LOSO eval summary ({model_type})")
        print("=" * 60)
        print(f"Signers evaluated: {len(accs)}")
        print(f"Mean accuracy:     {mean * 100:.2f}%  +/-{std * 100:.2f}%")
        print(f"Min accuracy:      {min(accs) * 100:.2f}%  (signer {per_fold[accs.index(min(accs))]['signer']})")
        print(f"Max accuracy:      {max(accs) * 100:.2f}%  (signer {per_fold[accs.index(max(accs))]['signer']})")
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": str(args.ckpt),
                    "model_type": model_type,
                    "per_fold": per_fold,
                    "summary": {
                        "mean_acc": mean if accs else 0,
                        "std_acc": std if accs else 0,
                        "min_acc": min(accs) if accs else 0,
                        "max_acc": max(accs) if accs else 0,
                    },
                },
                f, indent=2,
            )
        try:
            rel = args.export.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = args.export
        print(f"\nExported to: {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
