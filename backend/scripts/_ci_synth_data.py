"""Generate a tiny synthetic .npy dataset for CI pipeline smoke tests.

The real LSC50 hand-landmark .npy files are gitignored (see .gitignore at
the repo root), so the CI cannot run `backend/scripts/full_pipeline.bat`
against the real data. This script builds a minimal but well-formed
synthetic dataset that has the same shape as the real data, so the
full pipeline can be exercised end-to-end.

The synthetic dataset is written to `backend/tests/fixtures/synth_sequences/`
(NOT `backend/data/sequences/`, so a developer's real data is never
touched). The CI workflow copies / symlinks the synthetic data into
`backend/data/sequences/` for the duration of the run only.

  - 12 words, 18 signers, 3 samples per (word, signer)
  - each .npy has shape (T, 63) with random float32 values
  - vocabulary.json + splits.json are written so the train/eval
    scripts can run unchanged

The resulting model will learn nothing (random targets), but every
stage of the pipeline (extract -> loso splits -> train -> loso eval ->
held-out eval -> ONNX export -> summary) will run, and the pipeline
script's error handling / RC propagation will be exercised.

Usage (from backend/):
    python -m scripts._ci_synth_data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTH_DIR = PROJECT_ROOT / "tests" / "fixtures" / "synth_sequences"

WORDS = [
    "adios", "ayuda", "bano", "bien", "como_estas", "gracias",
    "hola", "mal", "no", "perdon", "por_favor", "si",
]
SIGNERS = [str(i) for i in [
    1, 3, 17, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
]]
SAMPLES_PER_SINGER = 3
MIN_T, MAX_T = 30, 60
SEED = 42


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CI synthetic sequence data.")
    parser.add_argument(
        "--out-dir", type=Path, default=SYNTH_DIR,
        help="Output directory. Defaults to backend/tests/fixtures/synth_sequences.",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    if out_dir.exists():
        # Clear pre-existing synth data so reruns are deterministic.
        for child in out_dir.iterdir():
            if child.is_dir():
                for f in child.glob("*.npy"):
                    f.unlink()
                for f in child.glob("*.json"):
                    f.unlink()
                child.rmdir()
            else:
                child.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    n_per_word: dict[str, int] = {}
    for w in WORDS:
        (out_dir / w).mkdir(parents=True, exist_ok=True)

    for signer in SIGNERS:
        for w in WORDS:
            for k in range(1, SAMPLES_PER_SINGER + 1):
                T = int(rng.integers(MIN_T, MAX_T + 1))
                seq = rng.normal(0.0, 0.3, size=(T, 63)).astype(np.float32)
                path = out_dir / w / f"{signer}_{w}_{k:03d}.npy"
                np.save(path, seq)
                n_per_word[w] = n_per_word.get(w, 0) + 1

    vocabulary = {w: i for i, w in enumerate(WORDS)}
    # Mirror the 5-signer held-out split used by the real pipeline.
    test_signers = ["24", "25", "26", "30", "32"]
    val_signers = ["3", "21", "28"]
    train_signers = [s for s in SIGNERS if s not in test_signers and s not in val_signers]

    counts = {"train": 0, "val": 0, "test": 0}
    per_split_per_class: dict[str, dict[str, int]] = {
        "train": {w: 0 for w in WORDS},
        "val":   {w: 0 for w in WORDS},
        "test":  {w: 0 for w in WORDS},
    }
    for signer in SIGNERS:
        split = "test" if signer in test_signers else "val" if signer in val_signers else "train"
        for w in WORDS:
            counts[split] += SAMPLES_PER_SINGER
            per_split_per_class[split][w] += SAMPLES_PER_SINGER

    splits = {
        "created_at": "ci-synthetic",
        "method": "signer-stratified (CI synthetic)",
        "seed": SEED,
        "test_fraction": 0.3,
        "val_fraction": 0.15,
        "n_total_signers": len(SIGNERS),
        "train_signers": train_signers,
        "val_signers": val_signers,
        "test_signers": test_signers,
        "vocabulary": vocabulary,
        "per_split_counts": counts,
        "per_split_per_class": per_split_per_class,
        "notes": "SYNTHETIC DATA FOR CI. Random float32, not real hand landmarks.",
    }
    (out_dir / "vocabulary.json").write_text(
        json.dumps(vocabulary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "splits.json").write_text(
        json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "manifest.json").write_text(
        json.dumps({"ci_synthetic": True, "n_files": sum(n_per_word.values())}, indent=2),
        encoding="utf-8",
    )

    print(
        f"[ci-synth] wrote {sum(n_per_word.values())} .npy files "
        f"({len(SIGNERS)} signers x {len(WORDS)} words x {SAMPLES_PER_SINGER})"
    )
    print(f"[ci-synth] out_dir: {out_dir}")
    print(f"[ci-synth] splits: train={counts['train']} val={counts['val']} test={counts['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
