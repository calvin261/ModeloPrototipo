"""Build Leave-One-Signer-Out (LOSO) cross-validation splits for sign language data.

The current `splits.json` in `data/sequences/` uses a single held-out signer set
(5 signers for test, 3 for val, 10 for train). That's an OK estimate but it
underestimates the true generalization gap and under-uses the data.

This script produces N=F(K) folds, one per signer, where the test set is
that signer's samples and the remaining K-1 signers are split into train+val
with a stratified split (val fraction = 0.15 of remaining).

Outputs:
  data/sequences/loso_splits.json — list of N fold dicts:
      {
        "fold": 0,
        "test_signer": "24",
        "train_signers": [...],
        "val_signers":   [...],
        "train_count":   245,
        "val_count":      41,
        "test_count":     35
      }

Usage:
    python -m scripts.build_loso_splits
    python -m scripts.build_loso_splits --output data/sequences/loso_splits.json
    python -m scripts.build_loso_splits --min-test-samples 3
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Leave-One-Signer-Out splits for sequence data."
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
        "--output", type=Path,
        default=PROJECT_ROOT / "data" / "sequences" / "loso_splits.json",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.15,
        help="Fraction of remaining (non-test) signers to put in the val set.",
    )
    parser.add_argument(
        "--min-test-samples", type=int, default=1,
        help="Skip signers with fewer than this many total samples.",
    )
    parser.add_argument(
        "--include-existing-splits", action="store_true",
        help="If set, add a fold that uses the original splits.json train/val/test "
             "partitioning, useful for reproducibility of prior results.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.vocabulary, encoding="utf-8") as f:
        vocabulary = json.load(f)

    # Discover all signer IDs by scanning the .npy filenames under each word dir.
    signer_to_files: dict[str, list[Path]] = {}
    for word in vocabulary:
        word_dir = args.sequences_dir / word
        if not word_dir.exists():
            continue
        for npy in word_dir.glob("*.npy"):
            signer = npy.stem.split("_", 1)[0]
            signer_to_files.setdefault(signer, []).append(npy)

    if not signer_to_files:
        print(f"[!] No .npy sequences found under {args.sequences_dir}")
        return 1

    signers = sorted(signer_to_files.keys(), key=lambda s: int(s) if s.isdigit() else s)
    print(f"Found {len(signers)} signers with data:")
    for s in signers:
        n = len(signer_to_files[s])
        flag = "" if n >= args.min_test_samples else "  (skipped)"
        print(f"  signer {s:>3}: {n:>4} samples{flag}")
    print()

    import random
    rng = random.Random(args.seed)

    folds = []
    for i, test_signer in enumerate(signers):
        test_count = len(signer_to_files[test_signer])
        if test_count < args.min_test_samples:
            continue
        # Remaining signers split into train/val
        rest = [s for s in signers if s != test_signer]
        rng.shuffle(rest)
        n_val = max(1, int(round(len(rest) * args.val_fraction)))
        val_signers = sorted(rest[:n_val])
        train_signers = sorted(rest[n_val:])

        train_count = sum(len(signer_to_files[s]) for s in train_signers)
        val_count = sum(len(signer_to_files[s]) for s in val_signers)

        folds.append(
            {
                "fold": i,
                "test_signer": test_signer,
                "train_signers": train_signers,
                "val_signers": val_signers,
                "train_count": train_count,
                "val_count": val_count,
                "test_count": test_count,
            }
        )

    # Optionally embed the original splits.json as fold -1 for reproducibility.
    if args.include_existing_splits:
        original = args.sequences_dir / "splits.json"
        if original.exists():
            with open(original, encoding="utf-8") as f:
                orig = json.load(f)
            folds.insert(
                0,
                {
                    "fold": -1,
                    "test_signer": "split",
                    "train_signers": sorted(orig["train_signers"]),
                    "val_signers": sorted(orig["val_signers"]),
                    "test_signers": sorted(orig["test_signers"]),
                    "train_count": orig["per_split_counts"]["train"],
                    "val_count": orig["per_split_counts"]["val"],
                    "test_count": orig["per_split_counts"]["test"],
                    "note": "mirrors the original 80/15/15 splits.json for reproducibility",
                },
            )

    output = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "leave-one-signer-out (LOSO)",
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "n_folds": len(folds),
        "n_signers": len(signers),
        "min_test_samples": args.min_test_samples,
        "vocabulary": vocabulary,
        "folds": folds,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print()
    try:
        rel = args.output.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = args.output
    print(f"Wrote {len(folds)} folds to: {rel}")
    print()
    print("Per-fold test counts:")
    for fold in folds[:25]:
        print(
            f"  fold {fold['fold']:>2}  test_signer={fold['test_signer']:>3}  "
            f"train={fold['train_count']:>3}  val={fold['val_count']:>2}  test={fold['test_count']:>2}"
        )
    if len(folds) > 25:
        print(f"  ... and {len(folds) - 25} more folds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
