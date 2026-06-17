"""Latency profiler for the sequence model.

Loads a checkpoint, runs N inferences on dummy data, and reports:
  - p50 / p95 / p99 latency
  - per-stage breakdown if torch profiler is enabled
  - JSON export for comparison across model versions

Usage:
    python -m scripts.profile_sequence_latency
    python -m scripts.profile_sequence_latency --ckpt models/sequence/sign_gru.pth --n 200
    python -m scripts.profile_sequence_latency --export results/latency.json
    python -m scripts.profile_sequence_latency --threads 1 2 4
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

from hand_model.sequence_model import SignSequenceModel


def _measure(model: torch.nn.Module, x: torch.Tensor, n: int, warmup: int) -> list[float]:
    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        for _ in range(n):
            t0 = time.perf_counter()
            _ = model(x)
            latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def _stats(latencies: list[float]) -> dict:
    if not latencies:
        return {}
    arr = np.array(latencies)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "std_ms": float(arr.std()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Latency profile of the sequence model.")
    parser.add_argument(
        "--ckpt", type=Path,
        default=PROJECT_ROOT / "models" / "sequence" / "sign_gru.pth",
    )
    parser.add_argument("--n", type=int, default=100, help="Number of timed inferences per batch size.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32],
    )
    parser.add_argument(
        "--threads", type=int, nargs="+", default=[None],
        help="Optional list of intra-op thread counts to compare. Default: whatever torch picks.",
    )
    parser.add_argument(
        "--export", type=Path, default=None,
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
    )
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    n_classes = len(ckpt.get("vocabulary", {})) or cfg.num_classes
    model_type = ckpt.get("model_type", "gru")
    try:
        rel = args.ckpt.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = args.ckpt
    print("=" * 60)
    print(f"  Latency profile — {rel}")
    print("=" * 60)
    if model_type == "transformer":
        from hand_model.spoter_model import SignTransformer
        print(f"Model:    {model_type}  classes={n_classes}  max_seq={cfg.max_seq_len}")
    else:
        print(f"Model:    input={cfg.input_size} hidden={cfg.hidden_size}x{cfg.num_layers} "
              f"classes={n_classes} max_seq={cfg.max_seq_len}")
    print(f"Runs:     {args.n} per (batch, threads) — warmup {args.warmup}")
    print()

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
    model.eval()
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model.to(device)

    results: list[dict] = []
    for n_threads in args.threads:
        if n_threads is not None:
            torch.set_num_threads(n_threads)
        else:
            torch.set_num_threads(torch.get_num_threads())
        actual_threads = torch.get_num_threads()
        for batch_size in args.batch_sizes:
            if model_type == "transformer":
                # Transformer takes (B, T, 21, 3) input
                x = torch.randn(
                    batch_size, cfg.max_seq_len, 21, 3,
                    device=device, dtype=torch.float32,
                )
            else:
                x = torch.randn(
                    batch_size, cfg.max_seq_len, cfg.input_size,
                    device=device, dtype=torch.float32,
                )
            latencies = _measure(model, x, args.n, args.warmup)
            stats = _stats(latencies)
            stats.update({
                "batch_size": batch_size,
                "threads": actual_threads,
                "device": str(device),
            })
            results.append(stats)
            print(
                f"  batch={batch_size:>3}  threads={actual_threads:>3}  "
                f"p50={stats['p50_ms']:>6.2f}ms  p95={stats['p95_ms']:>6.2f}ms  "
                f"p99={stats['p99_ms']:>6.2f}ms  mean={stats['mean_ms']:>6.2f}ms"
            )
    print()

    # Per-sample latency = batch latency / batch size (for single-sample inference)
    print("Per-sample (batch latency / batch size):")
    for r in results:
        ps = {k: v for k, v in r.items() if k != "batch_size"}
        per_sample = {k: (v / r["batch_size"] if isinstance(v, float) and k.endswith("_ms") else v) for k, v in r.items()}
        print(
            f"  batch={r['batch_size']:>3}  threads={r['threads']:>3}  "
            f"p50/sample={per_sample['p50_ms']:>6.3f}ms  "
            f"p99/sample={per_sample['p99_ms']:>6.3f}ms"
        )

    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": str(args.ckpt), "results": results}, f, indent=2)
        print()
        try:
            print(f"Latency results exported to: {args.export.relative_to(PROJECT_ROOT)}")
        except ValueError:
            print(f"Latency results exported to: {args.export}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
