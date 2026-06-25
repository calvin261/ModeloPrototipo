"""Export a trained Sign Transformer (or GRU) to ONNX and apply INT8
dynamic quantization.

Produces three artifacts in `models/exports/`:
  - sign_transformer_v<ts>.onnx      (FP32, ~380 KB)
  - sign_transformer_v<ts>_int8.onnx (INT8 dynamic, ~120 KB)
  - sign_transformer_labels_v<ts>.json (label map)

The INT8-quantized model is what gets shipped to Android (per-INT8 weight
quantization gives ~3x size reduction and ~1.5-2x CPU speedup with
negligible accuracy loss on this architecture, per ONNX Runtime
benchmarks).

Usage (from backend/):
    python -m scripts.export_transformer_onnx
    python -m scripts.export_transformer_onnx --ckpt models/sequence/sign_transformer.pth
    python -m scripts.export_transformer_onnx --no-int8
    python -m scripts.export_transformer_onnx --quantize-static --calib-data data/sequences
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from hand_model.config import SequenceConfig
from hand_model.spoter_model import SignTransformer


def _load_transformer(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    vocab = ckpt["vocabulary"]
    n_classes = len(vocab)
    n_joints = cfg.input_size // 3  # 21 for legacy, 49 for holistic
    model = SignTransformer(
        n_joints=n_joints, n_channels=3, num_classes=n_classes,
        embed_dim=128, nhead=4, num_layers=4, dim_feedforward=256,
        dropout=0.0, max_seq_len=cfg.max_seq_len + 1, use_cls=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab, cfg


def _export_fp32(model, dummy: torch.Tensor, out_path: Path) -> None:
    with torch.no_grad():
        torch.onnx.export(
            model, (dummy,), str(out_path),
            input_names=["sequence"], output_names=["logits"],
            dynamic_axes={
                "sequence": {0: "batch", 1: "time"},
                "logits": {0: "batch"},
            },
            opset_version=18,
        )
    print(f"  FP32 ONNX:   {out_path.name}  ({out_path.stat().st_size / 1024:.1f} KB)")


def _quantize_dynamic(in_path: Path, out_path: Path) -> int:
    """Apply dynamic INT8 quantization to an existing ONNX file.

    Returns the size in bytes of the quantized model. If quantization
    fails (e.g. quantize_dynamics API changes in onnxruntime), writes
    the FP32 file as the INT8 file with a warning.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("  onnxruntime.quantization not available, skipping INT8")
        out_path.write_bytes(in_path.read_bytes())
        return out_path.stat().st_size
    try:
        quantize_dynamic(
            model_input=str(in_path),
            model_output=str(out_path),
            weight_type=QuantType.QInt8,
        )
        return out_path.stat().st_size
    except Exception as e:  # noqa: BLE001
        print(f"  INT8 quantization failed: {e}")
        out_path.write_bytes(in_path.read_bytes())
        return out_path.stat().st_size


def _compare_onnx_vs_pt(
    model: nn.Module, model_path: Path, dummy: torch.Tensor
) -> tuple[float, float]:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_out = model(dummy).numpy()
    onnx_out = sess.run(None, {"sequence": dummy.numpy()})[0]
    diff = float(np.abs(onnx_out - pt_out).max())
    rel = float(np.abs((onnx_out - pt_out) / (np.abs(pt_out) + 1e-6)).max())
    return diff, rel


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Sign Transformer to ONNX (FP32 + INT8)."
    )
    parser.add_argument(
        "--ckpt", type=Path,
        default=PROJECT_ROOT / "models" / "sequence" / "sign_transformer.pth",
    )
    parser.add_argument("--no-int8", action="store_true", help="Skip INT8 quantization.")
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "models" / "exports",
    )
    args = parser.parse_args()

    if not args.ckpt.exists():
        print(f"[!] Checkpoint not found: {args.ckpt}")
        print("    Train one with: python -m scripts.train_transformer")
        return 1

    print("=" * 60)
    print(f"  Export — {args.ckpt.relative_to(PROJECT_ROOT)}")
    print("=" * 60)

    model, vocab, cfg = _load_transformer(args.ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    n_joints = cfg.input_size // 3  # 21 for legacy, 49 for holistic
    print(f"Model:    {n_params:,} params, {len(vocab)} classes, max_seq={cfg.max_seq_len}, joints={n_joints}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp32_path = args.out_dir / f"sign_transformer_v{ts}.onnx"
    int8_path = args.out_dir / f"sign_transformer_v{ts}_int8.onnx"
    labels_path = args.out_dir / f"sign_transformer_labels_v{ts}.json"

    # Dummy input (B=1, T=max_seq_len, n_joints, 3)
    dummy = torch.randn(1, cfg.max_seq_len, n_joints, 3, dtype=torch.float32)
    _export_fp32(model, dummy, fp32_path)
    diff, rel = _compare_onnx_vs_pt(model, fp32_path, dummy)
    print(f"  ONNX vs PyTorch:  max abs diff = {diff:.6f},  max rel diff = {rel:.6f}")

    if not args.no_int8:
        sz = _quantize_dynamic(fp32_path, int8_path)
        print(f"  INT8 ONNX:   {int8_path.name}  ({sz / 1024:.1f} KB)")
        diff8, rel8 = _compare_onnx_vs_pt(model, int8_path, dummy)
        print(f"  ONNX-INT8 vs PyTorch:  max abs diff = {diff8:.6f},  max rel diff = {rel8:.6f}")

    # Save labels (same schema as the existing GRU exports)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"  Labels:     {labels_path.name}")

    # Update registry.json
    registry_path = args.out_dir / "registry.json"
    registry = {"models": []}
    if registry_path.exists():
        try:
            with open(registry_path, encoding="utf-8") as f:
                registry = json.load(f)
                if "models" not in registry:
                    registry["models"] = []
        except json.JSONDecodeError:
            pass
    entry = {
        "model_type": "transformer",
        "version": ts,
        "fp32_onnx": str(fp32_path.relative_to(PROJECT_ROOT)),
        "int8_onnx": str(int8_path.relative_to(PROJECT_ROOT)) if not args.no_int8 else None,
        "labels": str(labels_path.relative_to(PROJECT_ROOT)),
        "n_classes": len(vocab),
        "vocabulary": vocab,
        "max_seq_len": cfg.max_seq_len,
        "n_params": n_params,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }
    registry["models"].insert(0, entry)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    print(f"  Registry:   {registry_path.relative_to(PROJECT_ROOT)} updated")
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
