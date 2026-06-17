"""Side-by-side comparison: GRU (legacy) vs Sign Transformer (F3) on the
same test split.

Reports per-class accuracy, latency, model size, and confusion-matrix
top pairs for each. Useful as a single command to validate that the F3
architecture change actually improves over the F0 baseline.

Usage:
    python -m scripts.compare_models
    python -m scripts.compare_models --export reports/gru_vs_transformer.json
    python -m scripts.compare_models --transformer-ckpt models/sequence/sign_transformer.pth
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

from hand_model.config import SequenceConfig
from hand_model.data import SequenceDataset
from hand_model.sequence_model import SignSequenceModel
from hand_model.spoter_model import SignTransformer
from hand_model.augmentations import ReshapingSequenceDataset  # may not exist; fallback
from scripts.train_transformer import ReshapingSequenceDataset, collate_reshape


def _load_gru(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    vocab = ckpt["vocabulary"]
    model = SignSequenceModel(
        input_size=cfg.input_size, hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers, num_classes=len(vocab), dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab, cfg


def _load_transformer(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    vocab = ckpt["vocabulary"]
    model = SignTransformer(
        n_joints=21, n_channels=3, num_classes=len(vocab),
        embed_dim=128, nhead=4, num_layers=4, dim_feedforward=256,
        dropout=0.0, max_seq_len=cfg.max_seq_len + 1, use_cls=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab, cfg


def _predict_gru(model, ds: SequenceDataset, device):
    preds_all, confs_all, labels_all, latencies = [], [], [], []
    inv_vocab = {v: k for k, v in ds.label_map.items() if v in [s[1] for s in ds.samples]}
    label_set = set(s[1] for s in ds.samples)
    inv = {v: k for k, v in ds.label_map.items() if v in label_set}
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    with torch.no_grad():
        for seqs, labels in loader:
            seqs = seqs.to(device)
            t0 = time.perf_counter()
            logits = model(seqs)
            elapsed = (time.perf_counter() - t0) * 1000.0 / seqs.size(0)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            conf, pred = probs.max(axis=1), probs.argmax(axis=1)
            preds_all.extend([inv[int(p)] for p in pred])
            confs_all.extend(conf.tolist())
            labels_all.extend([inv[int(l)] for l in labels.numpy()])
            latencies.extend([elapsed] * seqs.size(0))
    return preds_all, confs_all, labels_all, latencies


def _predict_transformer(model, ds: SequenceDataset, device):
    rds = ReshapingSequenceDataset(ds)
    preds_all, confs_all, labels_all, latencies = [], [], [], []
    inv = {v: k for k, v in ds.label_map.items() if v in set(s[1] for s in ds.samples)}
    loader = DataLoader(rds, batch_size=32, shuffle=False, collate_fn=collate_reshape)
    with torch.no_grad():
        for seqs, masks, labels in loader:
            seqs = seqs.to(device)
            masks = masks.to(device)
            t0 = time.perf_counter()
            logits = model(seqs, padding_mask=masks)
            elapsed = (time.perf_counter() - t0) * 1000.0 / seqs.size(0)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            conf, pred = probs.max(axis=1), probs.argmax(axis=1)
            preds_all.extend([inv[int(p)] for p in pred])
            confs_all.extend(conf.tolist())
            labels_all.extend([inv[int(l)] for l in labels.numpy()])
            latencies.extend([elapsed] * seqs.size(0))
    return preds_all, confs_all, labels_all, latencies


def _summary(name, preds, confs, labels, latencies):
    n = len(preds)
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    acc = correct / n
    per_class = defaultdict(lambda: [0, 0])  # [correct, total]
    for p, l in zip(preds, labels):
        per_class[l][1] += 1
        if p == l:
            per_class[l][0] += 1
    low = [(lbl, c, t, c / t * 100) for lbl, (c, t) in per_class.items() if t > 0 and c / t < 0.7]
    confusions = defaultdict(int)
    for p, l in zip(preds, labels):
        if p != l:
            confusions[(l, p)] += 1
    top_pairs = sorted(confusions.items(), key=lambda x: -x[1])[:5]
    return {
        "name": name,
        "accuracy": acc,
        "n": n,
        "mean_confidence": float(np.mean(confs)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "classes_below_70pct": sorted(low, key=lambda x: x[3]),
        "top_confusions": [{"true": t, "pred": p, "n": n} for (t, p), n in top_pairs],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare GRU vs Sign Transformer.")
    parser.add_argument(
        "--gru-ckpt", type=Path,
        default=PROJECT_ROOT / "models" / "sequence" / "sign_gru.pth",
    )
    parser.add_argument(
        "--transformer-ckpt", type=Path,
        default=PROJECT_ROOT / "models" / "sequence" / "sign_transformer.pth",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--signers", nargs="+", default=["24", "25", "26", "30", "32"],
        help="Test signer set (default: the held-out test signers from splits.json).",
    )
    parser.add_argument(
        "--export", type=Path, default=None,
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  GRU vs Sign Transformer comparison")
    print("=" * 60)
    device = torch.device(args.device)

    # Load datasets using each model's own config (they should match for fair comparison)
    gru_model, gru_vocab, gru_cfg = _load_gru(args.gru_ckpt)
    gru_ds = SequenceDataset(
        sequences_dir=str(PROJECT_ROOT / "data" / "sequences"),
        vocabulary=gru_vocab, max_seq_len=gru_cfg.max_seq_len,
        signers=set(args.signers),
    )
    print(f"GRU test set:         {len(gru_ds)} samples (signers={args.signers})")

    if args.transformer_ckpt.exists():
        tx_model, tx_vocab, tx_cfg = _load_transformer(args.transformer_ckpt)
        tx_ds = SequenceDataset(
            sequences_dir=str(PROJECT_ROOT / "data" / "sequences"),
            vocabulary=tx_vocab, max_seq_len=tx_cfg.max_seq_len,
            signers=set(args.signers),
        )
        print(f"Transformer test set: {len(tx_ds)} samples")
    else:
        print(f"Transformer ckpt missing: {args.transformer_ckpt}  (skip)")
        tx_model = None

    print()
    print("Running GRU inference...")
    gru_preds, gru_confs, gru_labels, gru_lats = _predict_gru(gru_model, gru_ds, device)
    gru_summary = _summary("GRU (Bi-GRU + attention)", gru_preds, gru_confs, gru_labels, gru_lats)

    if tx_model is not None:
        print("Running Transformer inference...")
        tx_preds, tx_confs, tx_labels, tx_lats = _predict_transformer(tx_model, tx_ds, device)
        tx_summary = _summary("Sign Transformer", tx_preds, tx_confs, tx_labels, tx_lats)

    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  {'Model':<30} {'Acc':>7} {'Conf':>7} {'p50ms':>7} {'p95ms':>7}  N")
    for s in [gru_summary] + ([tx_summary] if tx_model is not None else []):
        print(
            f"  {s['name']:<30} "
            f"{s['accuracy'] * 100:>6.2f}% "
            f"{s['mean_confidence']:>6.3f} "
            f"{s['latency_p50_ms']:>6.2f} "
            f"{s['latency_p95_ms']:>6.2f}  "
            f"{s['n']}"
        )
    if tx_model is not None:
        delta = tx_summary["accuracy"] - gru_summary["accuracy"]
        print(f"\n  Transformer vs GRU delta: {delta * 100:+.2f} pp")
    print()
    print("Top confusions:")
    print(f"  GRU:          {gru_summary['top_confusions'][:3]}")
    if tx_model is not None:
        print(f"  Transformer:  {tx_summary['top_confusions'][:3]}")

    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(
                {"gru": gru_summary, "transformer": tx_summary if tx_model else None},
                f, indent=2,
            )
        print(f"\nExported to: {args.export.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
