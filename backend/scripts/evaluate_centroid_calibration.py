"""On-device personalization for sign recognition.

Implements a **centroid-based** few-shot adapter: given 3-5 example
recordings per sign, we extract the model's penultimate-layer embedding
for each, average them into a centroid, and classify new sign
recordings by cosine similarity to the nearest centroid.

This approach was chosen over LoRA fine-tuning for the first version
because it requires **zero gradient updates** — only forward passes —
so it runs in pure inference mode on Android (no autograd needed,
no on-device training subgraph). If the simple centroid approach
proves insufficient, the same interface supports a future LoRA
implementation.

Usage (offline evaluation):
    python -m scripts.evaluate_centroid_calibration \\
        --ckpt models/sequence/sign_transformer.pth \\
        --n-shots 3

    # Sweep over multiple shot counts:
    for k in 1 2 3 5 8; do
      python -m scripts.evaluate_centroid_calibration --n-shots $k
    done
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from hand_model.config import SequenceConfig
from hand_model.data import SequenceDataset
from hand_model.spoter_model import SignTransformer
from hand_model.sequence_model import SignSequenceModel


def _load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    vocab = ckpt["vocabulary"]
    model_type = ckpt.get("model_type", "gru")
    if model_type == "transformer":
        n_classes = len(vocab)
        model = SignTransformer(
            n_joints=21, n_channels=3, num_classes=n_classes,
            embed_dim=128, nhead=4, num_layers=4, dim_feedforward=256,
            dropout=0.0, max_seq_len=cfg.max_seq_len + 1, use_cls=True,
        )
    else:
        n_classes = len(vocab)
        model = SignSequenceModel(
            input_size=cfg.input_size, hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers, num_classes=n_classes, dropout=0.0,
        )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab, cfg, model_type


@torch.no_grad()
def _embedding_transformer(model: SignTransformer, seq_flat: torch.Tensor) -> torch.Tensor:
    """(T, 63) -> (D,) where D = embed_dim (128)."""
    if seq_flat.dim() == 2:
        seq_flat = seq_flat.unsqueeze(0)
    B, T, D = seq_flat.shape
    seq = seq_flat.reshape(B, T, 21, 3)
    mask = (seq.abs().sum(dim=(2, 3)) < 1e-6)  # (B, T)
    x = model._embed(seq)  # (B, T, D)
    if model.use_cls:
        cls = model.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=1)
    L = x.size(1)
    x = x + model.pos_embed[:, :L, :]
    out = model.encoder(x, src_key_padding_mask=mask)
    if model.use_cls:
        pooled = out[:, 0, :]
    else:
        m = (~mask).float().unsqueeze(-1)
        pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    return model.norm(pooled).squeeze(0)


@torch.no_grad()
def _embedding_gru(model: SignSequenceModel, seq_flat: torch.Tensor) -> torch.Tensor:
    """(T, 63) -> (hidden*2,) = 256-d."""
    if seq_flat.dim() == 2:
        seq_flat = seq_flat.unsqueeze(0)
    out, _ = model.gru(seq_flat)
    attn = torch.softmax(model.attention(out), dim=1)
    context = (out * attn).sum(dim=1)  # (1, hidden*2)
    return context.squeeze(0)


def _all_embeddings(model, ds: SequenceDataset, model_type: str):
    fn = _embedding_transformer if model_type == "transformer" else _embedding_gru
    by_class: dict[int, list[np.ndarray]] = defaultdict(list)
    by_signer: dict[str, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for i in range(len(ds)):
        seq, lbl = ds[i]
        signer = Path(ds.samples[i][0]).stem.split("_", 1)[0]
        emb = fn(model, seq).cpu().numpy()
        by_class[lbl.item()].append(emb)
        by_signer[signer].setdefault(lbl.item(), []).append(emb)
    return by_class, by_signer


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(a @ b / (na * nb))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate centroid-based few-shot calibration."
    )
    parser.add_argument(
        "--ckpt", type=Path,
        default=PROJECT_ROOT / "models" / "sequence" / "sign_transformer.pth",
    )
    parser.add_argument(
        "--sequences-dir", type=Path,
        default=PROJECT_ROOT / "data" / "sequences",
    )
    parser.add_argument(
        "--vocabulary", type=Path,
        default=PROJECT_ROOT / "data" / "sequences" / "vocabulary.json",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=60,
    )
    parser.add_argument(
        "--n-shots", type=int, default=3,
        help="Number of examples per class to use as the calibration set (k-shot).",
    )
    parser.add_argument(
        "--mode", choices=["leave-one-signer-out", "within-signer"], default="leave-one-signer-out",
        help="How to split calibration vs evaluation. Default LOSO.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.ckpt.exists():
        print(f"[!] Checkpoint not found: {args.ckpt}")
        return 1

    print("=" * 60)
    print(f"  Centroid calibration — n_shots={args.n_shots}, mode={args.mode}")
    print("=" * 60)
    try:
        rel = args.ckpt.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = args.ckpt
    print(f"Checkpoint:  {rel}")

    model, vocab, cfg, model_type = _load_model(args.ckpt)
    print(f"Model:       {model_type}, {len(vocab)} classes, max_seq={cfg.max_seq_len}")

    ds = SequenceDataset(
        sequences_dir=str(args.sequences_dir),
        vocabulary=vocab,
        max_seq_len=args.max_seq_len,
    )
    print(f"Dataset:     {len(ds)} samples")
    print("Extracting embeddings...")
    by_class, by_signer = _all_embeddings(model, ds, model_type)
    print(f"Embeddings:  {sum(len(v) for v in by_class.values())} total")
    print()

    rng = np.random.default_rng(args.seed)
    inv_vocab = {v: k for k, v in vocab.items()}

    if args.mode == "leave-one-signer-out":
        # For each signer, take K examples per class as calibration,
        # use the rest of that signer's data as the test.
        signers = sorted(by_signer.keys())
        per_signer_accs = []
        for test_signer in signers:
            # Calibration = n_shots samples per class drawn from OTHER signers
            # (mixed-signer centroids — this is the realistic scenario:
            # the model is calibrated on a generic pool and applied to a
            # new signer).
            calib_emb: dict[int, list[np.ndarray]] = defaultdict(list)
            for other in signers:
                if other == test_signer:
                    continue
                for lbl, embs in by_signer[other].items():
                    if len(calib_emb[lbl]) < args.n_shots:
                        idxs = rng.choice(len(embs), size=min(args.n_shots, len(embs)), replace=False)
                        for i in idxs:
                            if len(calib_emb[lbl]) < args.n_shots:
                                calib_emb[lbl].append(embs[i])
            centroids = {
                lbl: np.mean(embs, axis=0) for lbl, embs in calib_emb.items() if embs
            }
            # Test = all of test_signer's samples
            n_correct = 0
            n_total = 0
            for true_lbl, embs in by_signer[test_signer].items():
                for emb in embs:
                    best_lbl = max(centroids.keys(), key=lambda l: _cosine(emb, centroids[l]))
                    if best_lbl == true_lbl:
                        n_correct += 1
                    n_total += 1
            acc = n_correct / n_total if n_total else 0
            per_signer_accs.append(acc)
            print(f"  test_signer={test_signer:>3}  acc={acc * 100:.2f}%  ({n_correct}/{n_total})")
        mean_acc = float(np.mean(per_signer_accs))
        std_acc = float(np.std(per_signer_accs))
        print()
        print(f"Mean accuracy across {len(signers)} signers: {mean_acc * 100:.2f}% +/-{std_acc * 100:.2f}%")
    else:
        # Within-signer: pick n_shots per class from EACH signer, then evaluate
        # on the rest of that signer's data.
        per_signer_accs = []
        for signer, class_to_embs in by_signer.items():
            centroids = {}
            used_idxs = {}  # lbl -> set of indices used as calibration
            for lbl, embs in class_to_embs.items():
                if len(embs) <= args.n_shots:
                    continue
                idxs = rng.choice(len(embs), size=args.n_shots, replace=False)
                centroids[lbl] = np.mean([embs[i] for i in idxs], axis=0)
                used_idxs[lbl] = set(int(i) for i in idxs)
            if not centroids:
                continue  # skip signers with no usable calibration data
            n_correct = 0
            n_total = 0
            for lbl, embs in class_to_embs.items():
                for i, emb in enumerate(embs):
                    if lbl in used_idxs and i in used_idxs[lbl]:
                        continue
                    best_lbl = max(centroids.keys(), key=lambda l: _cosine(emb, centroids[l]))
                    if best_lbl == lbl:
                        n_correct += 1
                    n_total += 1
            if n_total == 0:
                continue
            acc = n_correct / n_total
            per_signer_accs.append(acc)
            print(f"  signer={signer:>3}  acc={acc * 100:>6.2f}%  ({n_correct}/{n_total})  "
                  f"calib_classes={len(centroids)}")
        if not per_signer_accs:
            print("[!] No signers had enough samples for within-signer calibration.")
            return 0
        mean_acc = float(np.mean(per_signer_accs))
        std_acc = float(np.std(per_signer_accs))
        print()
        print(f"Within-signer mean accuracy: {mean_acc * 100:.2f}% +/-{std_acc * 100:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
