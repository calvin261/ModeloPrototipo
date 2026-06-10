"""Build signer-stratified train/val/test splits for the sequence model.

Reads the manifest produced by scripts.prepare_sequences_from_lsc50 and
assigns each signer to exactly one of {train, val, test}. The split is
held-out-by-signer, which is the realistic evaluation protocol for sign
language recognition (the model never sees the test signer's hand during
training).

Outputs:
  data/sequences/splits.json
      { "train": [signers...], "val": [signers...], "test": [signers...],
        "per_class_counts": {"train": {...}, "val": {...}, "test": {...}},
        ... }
  data/sequences/vocabulary.json
      { "hola": 0, "gracias": 1, ... }   (12 classes, sorted)

The signer assignment is deterministic given the seed (default 42), so
re-running this script always produces the same splits (reproducibility).
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
MANIFEST_PATH = SEQUENCES_DIR / "manifest.json"
SPLITS_PATH = SEQUENCES_DIR / "splits.json"
VOCAB_PATH = SEQUENCES_DIR / "vocabulary.json"

# Target ratios. With 18 signers and ~6-7 test signers (~35%) we
# get a meaningful held-out set without losing too much training data.
TEST_FRAC = 0.30
VAL_FRAC = 0.15
SEED = 42


def main() -> int:
    if not MANIFEST_PATH.exists():
        print(f"[!] Manifest not found: {MANIFEST_PATH}")
        print("    Run scripts.prepare_sequences_from_lsc50 first.")
        return 1

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    files = manifest.get("files", [])
    if not files:
        print("[!] Manifest has no files.")
        return 1

    signers = sorted({f["signer"] for f in files})
    words = sorted({f["word"] for f in files})
    print("=" * 60)
    print("  Build signer-stratified splits for sequence model")
    print("=" * 60)
    print(f"Signers: {len(signers)}")
    print(f"Words:   {len(words)}")
    print(f"Total sequences: {len(files)}")
    print()

    # Sort by total count descending so signers with more data go to train
    signer_counts = Counter(f["signer"] for f in files)
    signers_sorted = sorted(signer_counts.keys(), key=lambda s: -signer_counts[s])
    print("Signers by sequence count:")
    for s in signers_sorted:
        print(f"  signer {s:>3}: {signer_counts[s]} sequences")

    # Assign signers to test/val/train with the given fractions.
    # Use deterministic shuffle so re-runs match.
    rng = random.Random(SEED)
    pool = list(signers_sorted)
    rng.shuffle(pool)

    n_test = max(1, round(len(pool) * TEST_FRAC))
    n_val = max(1, round(len(pool) * VAL_FRAC))
    test_signers = pool[:n_test]
    val_signers = pool[n_test : n_test + n_val]
    train_signers = pool[n_test + n_val :]

    print()
    print(f"Test signers ({len(test_signers)}):  {test_signers}")
    print(f"Val signers  ({len(val_signers)}):  {val_signers}")
    print(f"Train signers ({len(train_signers)}): {train_signers}")

    # Per-class counts per split
    def _per_class(items: list[dict]) -> dict[str, int]:
        c = Counter(f["word"] for f in items)
        return dict(sorted(c.items()))

    train_files = [f for f in files if f["signer"] in train_signers]
    val_files = [f for f in files if f["signer"] in val_signers]
    test_files = [f for f in files if f["signer"] in test_signers]

    print()
    print("Per-split totals:")
    print(f"  train: {len(train_files)} sequences")
    print(f"  val:   {len(val_files)} sequences")
    print(f"  test:  {len(test_files)} sequences")
    print()
    print("Per-split per-class counts:")
    header = f"  {'word':<14}" + "".join(f"{w:>8}" for w in words)
    print(header)
    for split_name, items in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ]:
        per = Counter(f["word"] for f in items)
        row = f"  {split_name:<14}" + "".join(f"{per.get(w, 0):>8}" for w in words)
        print(row)

    # Vocabulary: deterministic order, sorted alphabetically
    vocabulary = {w: i for i, w in enumerate(words)}

    # Write vocabulary
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, indent=2, ensure_ascii=False)
    print()
    print(f"Wrote vocabulary: {VOCAB_PATH.relative_to(PROJECT_ROOT)} ({len(vocabulary)} classes)")

    # Write splits
    splits = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "signer-stratified (held-out by signer)",
        "seed": SEED,
        "test_fraction": TEST_FRAC,
        "val_fraction": VAL_FRAC,
        "n_total_signers": len(signers),
        "train_signers": train_signers,
        "val_signers": val_signers,
        "test_signers": test_signers,
        "vocabulary": vocabulary,
        "per_split_counts": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        },
        "per_split_per_class": {
            "train": _per_class(train_files),
            "val": _per_class(val_files),
            "test": _per_class(test_files),
        },
        "notes": (
            "Test signers are NEVER seen during training. "
            "This is a realistic evaluation for sign-language recognition. "
            "For full leave-one-signer-out, repeat the experiment with each "
            "signer held out and average metrics."
        ),
    }
    with open(SPLITS_PATH, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
    print(f"Wrote splits:     {SPLITS_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
