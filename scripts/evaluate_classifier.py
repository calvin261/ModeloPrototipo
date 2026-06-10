"""Evaluate the trained MLP on the held-out test split.

Reports:
  - overall accuracy
  - per-class accuracy (flags classes below 50% and below 70%)
  - top 5 confusions (true -> predicted)
  - confidence distribution
"""

import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from hand_model.classifier import SignClassifierMLP


def main() -> int:
    test_csv = PROJECT_ROOT / "data" / "processed" / "landmarks" / "test.csv"
    ckpt_path = PROJECT_ROOT / "models" / "classifier" / "sign_mlp.pth"

    test_df = pd.read_csv(test_csv)
    print(f"Test samples: {len(test_df)}")
    print()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    label_map = ckpt["label_map"]
    inv_label_map = {v: k for k, v in label_map.items()}
    cfg = ckpt["config"]

    model = SignClassifierMLP(
        input_size=cfg.input_size,
        hidden_sizes=cfg.hidden_sizes,
        num_classes=len(label_map),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Model: input={cfg.input_size} hidden={cfg.hidden_sizes} classes={cfg.num_classes}")
    print(f"Classes: {sorted(label_map.keys())}")
    print()

    features = test_df.iloc[:, 1:].values.astype(np.float32)
    labels = test_df.iloc[:, 0].values
    label_idx = np.array([label_map[l] for l in labels])

    with torch.no_grad():
        logits = model(torch.tensor(features))
        probs = torch.softmax(logits, dim=1)
        conf, preds = torch.max(probs, dim=1)

    preds_np = preds.numpy()
    correct = (preds_np == label_idx).sum()
    total = len(label_idx)
    acc = correct / total
    print("=" * 60)
    print(f"  Test accuracy: {acc * 100:.2f}%  ({correct}/{total})")
    print("=" * 60)

    print()
    print("Per-class accuracy on test:")
    print("  label  correct  total    acc")
    low = []
    very_low = []
    for lbl in sorted(label_map.keys()):
        mask = labels == lbl
        n_total = int(mask.sum())
        n_correct = int((preds_np[mask] == label_idx[mask]).sum())
        pct = n_correct / n_total * 100 if n_total else 0
        flag = ""
        if pct < 50:
            flag = "  <- VERY LOW"
            very_low.append((lbl, pct, n_correct, n_total))
        elif pct < 70:
            flag = "  <- low"
            low.append((lbl, pct, n_correct, n_total))
        print(f"  {lbl:>5}  {n_correct:>7}  {n_total:>5}  {pct:5.1f}%{flag}")

    confusions = defaultdict(int)
    for true_i, pred_i in zip(label_idx, preds_np):
        if true_i != pred_i:
            confusions[(inv_label_map[int(true_i)], inv_label_map[int(pred_i)])] += 1

    print()
    print("Top 10 confusions (true -> predicted):")
    for (t, p), n in sorted(confusions.items(), key=lambda x: -x[1])[:10]:
        print(f"  {t} -> {p}  ({n}x)")

    print()
    mean_conf = conf.mean().item()
    median_conf = conf.median().item()
    low_conf_count = int((conf < 0.5).sum().item())
    print(f"Mean confidence:   {mean_conf:.3f}")
    print(f"Median confidence: {median_conf:.3f}")
    print(
        f"Low-conf samples:  {low_conf_count} / {len(conf)} "
        f"({low_conf_count / len(conf) * 100:.1f}%)"
    )

    if very_low:
        print()
        print("Classes below 50% accuracy (need attention):")
        for lbl, pct, c, t in very_low:
            print(f"  {lbl}: {pct:.1f}%  ({c}/{t})")
    if low:
        print()
        print("Classes 50-70% accuracy (could improve):")
        for lbl, pct, c, t in low:
            print(f"  {lbl}: {pct:.1f}%  ({c}/{t})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
