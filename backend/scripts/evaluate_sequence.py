"""Diagnostic evaluator for the dynamic sign (sequence/GRU) model.

Reports:
  - overall accuracy on the test split
  - 12x12 confusion matrix (or NxN for any vocabulary size)
  - per-class accuracy (flags classes below 50% and 70%)
  - top 10 confusions (true -> predicted)
  - confidence distribution (mean, median, low-conf count)
  - entropy of predictions (high entropy = model unsure)
  - latency per inference (p50, p95, p99)
  - per-frame latency breakdown
  - export of all per-sample results to a JSON file for post-hoc analysis

Usage:
    python -m scripts.evaluate_sequence
    python -m scripts.evaluate_sequence --ckpt models/sequence/sign_gru.pth
    python -m scripts.evaluate_sequence --export results/sequence_eval.json
    python -m scripts.evaluate_sequence --signers 24 25 26 30 32   # custom test signers
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


def _build_test_loader(
    sequences_dir: Path,
    vocabulary: dict,
    max_seq_len: int,
    signers: set[str] | None,
    batch_size: int,
) -> DataLoader:
    ds = SequenceDataset(
        sequences_dir=str(sequences_dir),
        vocabulary=vocabulary,
        max_seq_len=max_seq_len,
        signers=signers,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False), ds


def _latency_stats(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {}
    arr = np.array(latencies_ms)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy (base e) of a probability distribution."""
    p = probs[probs > 1e-9]
    return float(-(p * np.log(p)).sum())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnostic evaluation of the dynamic sign sequence model."
    )
    parser.add_argument(
        "--ckpt", type=Path,
        default=PROJECT_ROOT / "models" / "sequence" / "sign_gru.pth",
    )
    parser.add_argument(
        "--vocabulary", type=Path,
        default=PROJECT_ROOT / "data" / "sequences" / "vocabulary.json",
    )
    parser.add_argument(
        "--sequences-dir", type=Path,
        default=PROJECT_ROOT / "data" / "sequences",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=None,
        help="Override max_seq_len (default: from ckpt config).",
    )
    parser.add_argument(
        "--signers", nargs="+", default=None,
        help="Restrict test set to these signer IDs. Default: any test signer from splits.json, "
             "or all signers if no splits.json.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--export", type=Path, default=None,
        help="If set, dump per-sample results (true/pred/conf/latency) to this JSON file.",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup batches before measuring latency.",
    )
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    vocabulary = ckpt["vocabulary"]
    inv_vocabulary = {v: k for k, v in vocabulary.items()}
    cfg = ckpt["config"]
    if args.max_seq_len:
        cfg.max_seq_len = args.max_seq_len

    print("=" * 60)
    print("  Sequence model diagnostic evaluation")
    print("=" * 60)
    try:
        rel = args.ckpt.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = args.ckpt
    print(f"Checkpoint:    {rel}")
    print(f"Vocabulary:    {len(vocabulary)} classes")
    print(f"Max seq len:   {cfg.max_seq_len}")
    print(f"Hidden:        {cfg.hidden_size} x {cfg.num_layers}")
    print(f"Test signers:  {args.signers if args.signers else '(all)'}")
    print()

    # Build loader
    signers = set(args.signers) if args.signers else None
    loader, ds = _build_test_loader(
        args.sequences_dir, vocabulary, cfg.max_seq_len, signers, args.batch_size
    )
    if len(ds) == 0:
        print("[!] No test samples found.")
        return 1
    print(f"Test samples:  {len(ds)}")
    print()

    # Build model
    n_classes = len(vocabulary)
    model_type = ckpt.get("model_type", "gru")
    if model_type == "transformer":
        from hand_model.spoter_model import SignTransformer
        model = SignTransformer(
            n_joints=21, n_channels=3, num_classes=n_classes,
            embed_dim=128, nhead=4, num_layers=4, dim_feedforward=256,
            dropout=0.0, max_seq_len=cfg.max_seq_len + 1, use_cls=True,
        )
    else:
        model = SignSequenceModel(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_classes=n_classes,
            dropout=0.0,
        )
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.eval()
    print(f"Device:        {device}")
    print()

    # Inference with per-batch and per-sample timing
    per_sample_results: list[dict] = []
    per_batch_latencies: list[float] = []
    per_sample_latencies: list[float] = []
    n_classes = len(vocabulary)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    with torch.no_grad():
        for batch_idx, (sequences, labels) in enumerate(loader):
            sequences = sequences.to(device)
            labels_np = labels.numpy()

            if model_type == "transformer":
                # Reshape (B, T, 63) -> (B, T, 21, 3) and build padding mask.
                B, T, D = sequences.shape
                seq_4d = sequences.reshape(B, T, 21, 3)
                padding_mask = seq_4d.abs().sum(dim=(2, 3)) < 1e-6
            else:
                seq_4d = None
                padding_mask = None

            if device.type == "cpu":
                t0 = time.perf_counter()
                if model_type == "transformer":
                    logits = model(seq_4d, padding_mask=padding_mask)
                else:
                    logits = model(sequences)
                elapsed = time.perf_counter() - t0
            else:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                if model_type == "transformer":
                    logits = model(seq_4d, padding_mask=padding_mask)
                else:
                    logits = model(sequences)
                end.record()
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end) / 1000.0

            if batch_idx >= args.warmup:
                per_batch_latencies.append(elapsed * 1000.0)
                per_sample_latencies.extend(
                    [elapsed * 1000.0 / max(sequences.size(0), 1)] * sequences.size(0)
                )

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            confs, preds = probs.max(axis=1), probs.argmax(axis=1)
            for i in range(sequences.size(0)):
                conf_matrix[labels_np[i], int(preds[i])] += 1
                per_sample_results.append(
                    {
                        "idx": len(per_sample_results),
                        "true": inv_vocabulary[int(labels_np[i])],
                        "pred": inv_vocabulary[int(preds[i])],
                        "conf": float(confs[i]),
                        "entropy": _entropy(probs[i]),
                        "latency_ms": per_sample_latencies[-1] if per_sample_latencies else 0.0,
                    }
                )

    # Metrics
    correct = sum(1 for r in per_sample_results if r["true"] == r["pred"])
    total = len(per_sample_results)
    acc = correct / total if total else 0.0
    print("=" * 60)
    print(f"  Test accuracy: {acc * 100:.2f}%  ({correct}/{total})")
    print("=" * 60)
    print()

    print("Per-class accuracy on test:")
    print(f"  {'label':<14} {'correct':>8} {'total':>6} {'acc':>6}   {'mean_conf':>10}")
    low = []
    very_low = []
    for i, lbl in enumerate(vocabulary.keys()):
        mask = [r for r in per_sample_results if r["true"] == lbl]
        n_total = len(mask)
        n_correct = sum(1 for r in mask if r["true"] == r["pred"])
        pct = n_correct / n_total * 100 if n_total else 0
        mean_conf = float(np.mean([r["conf"] for r in mask])) if mask else 0
        flag = ""
        if pct < 50:
            flag = "   <- VERY LOW"
            very_low.append((lbl, pct, n_correct, n_total))
        elif pct < 70:
            flag = "   <- low"
            low.append((lbl, pct, n_correct, n_total))
        print(f"  {lbl:<14} {n_correct:>8} {n_total:>6} {pct:>5.1f}%   {mean_conf:>10.3f}{flag}")

    confusions = defaultdict(int)
    for r in per_sample_results:
        if r["true"] != r["pred"]:
            confusions[(r["true"], r["pred"])] += 1

    print()
    print("Top 10 confusions (true -> predicted):")
    for (t, p), n in sorted(confusions.items(), key=lambda x: -x[1])[:10]:
        print(f"  {t:<14} -> {p:<14}  ({n}x)")

    print()
    confidences = [r["conf"] for r in per_sample_results]
    entropies = [r["entropy"] for r in per_sample_results]
    print(f"Mean confidence:    {np.mean(confidences):.3f}")
    print(f"Median confidence:  {np.median(confidences):.3f}")
    print(f"Mean entropy:       {np.mean(entropies):.3f} nats  (max={np.log(n_classes):.3f})")
    low_conf = sum(1 for c in confidences if c < 0.5)
    print(
        f"Low-conf samples:   {low_conf}/{total} "
        f"({low_conf / total * 100:.1f}%)"
    )

    print()
    print("Latency per inference (batch-level):")
    for k, v in _latency_stats(per_batch_latencies).items():
        print(f"  {k:<10} {v}")
    print()
    print("Latency per sample (estimated = batch_ms / batch_size):")
    for k, v in _latency_stats(per_sample_latencies).items():
        print(f"  {k:<10} {v}")

    # Confusion matrix
    print()
    print("Confusion matrix (rows = true, cols = predicted):")
    labels_sorted = list(vocabulary.keys())
    name_w = max(8, max(len(l) for l in labels_sorted))
    header = " " * (name_w + 2) + " ".join(f"{l[:6]:>6}" for l in labels_sorted)
    print(header)
    for i, lbl in enumerate(labels_sorted):
        row = "  ".join(f"{conf_matrix[i, j]:>6}" for j in range(n_classes))
        print(f"  {lbl:<{name_w}} {row}")

    # Pairs to attack (off-diagonal > 0)
    pair_attacks = [
        (lbl_t, lbl_p, int(conf_matrix[i, j]))
        for i, lbl_t in enumerate(labels_sorted)
        for j, lbl_p in enumerate(labels_sorted)
        if i != j and conf_matrix[i, j] > 0
    ]
    pair_attacks.sort(key=lambda x: -x[2])
    if pair_attacks:
        print()
        print("Confusion pairs ranked (top 15 to attack with hard-negative mining):")
        for t, p, n in pair_attacks[:15]:
            print(f"  {t:<14} -> {p:<14}  ({n}x)")

    # Per-class low-confidence share (samples where conf < 0.5)
    print()
    print("Uncertainty by class (samples with conf < 0.5):")
    for lbl in labels_sorted:
        mask = [r for r in per_sample_results if r["true"] == lbl]
        if not mask:
            continue
        share = sum(1 for r in mask if r["conf"] < 0.5) / len(mask) * 100
        print(f"  {lbl:<14} {share:>5.1f}%  ({sum(1 for r in mask if r['conf'] < 0.5)}/{len(mask)})")

    # Export
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        export_data = {
            "checkpoint": str(args.ckpt),
            "vocabulary": vocabulary,
            "test_signers": sorted(signers) if signers else "all",
            "summary": {
                "accuracy": acc,
                "n_samples": total,
                "n_correct": correct,
                "mean_confidence": float(np.mean(confidences)),
                "mean_entropy": float(np.mean(entropies)),
                "latency_p50_ms": _latency_stats(per_batch_latencies).get("p50_ms"),
                "latency_p95_ms": _latency_stats(per_batch_latencies).get("p95_ms"),
                "latency_p99_ms": _latency_stats(per_batch_latencies).get("p99_ms"),
            },
            "confusion_matrix": conf_matrix.tolist(),
            "confusion_pairs": [
                {"true": t, "pred": p, "n": n} for t, p, n in pair_attacks
            ],
            "per_sample": per_sample_results,
        }
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print()
        try:
            print(f"Per-sample results exported to: {args.export.relative_to(PROJECT_ROOT)}")
        except ValueError:
            print(f"Per-sample results exported to: {args.export}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
