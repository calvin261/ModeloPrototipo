"""Build stratified train/val/test splits for the static alphabet classifier.

Reads:
  - data/processed/landmarks/alphabet_kaggle.csv  (primary source)
  - data/landmarks/collected_data.csv             (any locally captured data)
  - data/landmarks/mobile_upload.csv              (any data uploaded from app)

Writes (under data/processed/landmarks/):
  - train.csv   (70%)
  - val.csv     (15%)
  - test.csv    (15%)   <- NEVER used during training; only final evaluation
  - manifest.json (counts, sources, timestamp)

The split is stratified by label (so each letter keeps its proportion).
If a class has fewer than 3 samples it is placed entirely in train.

Run from project root:
    python -m scripts.build_classifier_splits
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_CSV = PROCESSED_DIR / "alphabet_kaggle.csv"
OPTIONAL_CSVS = [
    PROJECT_ROOT / "data" / "landmarks" / "collected_data.csv",
    PROJECT_ROOT / "data" / "landmarks" / "mobile_upload.csv",
]

# Signs whose motion is required to disambiguate from other static letters
# (J traces a hook, Z draws a line). The static classifier cannot learn them
# from a single frame, so we drop them if present.
DYNAMIC_SIGNS = {"J", "Z"}

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Canonical 63 column names for 21 hand landmarks (x, y, z per landmark).
CANONICAL_FEATURE_COLS = [f"{c}{i}" for i in range(21) for c in ("x", "y", "z")]
CANONICAL_COLS = ["label"] + CANONICAL_FEATURE_COLS


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with the canonical 64-column schema.

    Different data sources use different column naming conventions for the
    63 landmark features. The two we have encountered:

      - "interleaved" (Python DataCollector + alphabet_kaggle.csv):
            label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20
      - "flat" (Android mobile_upload.csv):
            label, x0, x1, x2, ..., x62

    Both store the data in landmark-major order (landmark 0's x,y,z, then
    landmark 1's x,y,z, ...). The data is identical, only the column NAMES
    differ. We rename to the canonical interleaved schema so concatenation
    aligns properly.
    """
    if df.shape[1] == 64 and list(df.columns[:1]) == ["label"] and list(df.columns[1:]) == CANONICAL_FEATURE_COLS:
        return df  # already canonical

    if df.shape[1] == 64 and df.columns[0] != "label":
        df = df.rename(columns={df.columns[0]: "label"})

    cols = list(df.columns)
    if len(cols) != 64 or cols[0] != "label":
        raise ValueError(
            f"Unexpected schema: {len(cols)} columns, first is {cols[0]!r}. "
            f"Expected 64 columns starting with 'label'."
        )

    feature_cols = cols[1:]
    # If the features are 63 sequential indices (x0, x1, x2, ..., x62)
    # rename them to the canonical interleaved x0,y0,z0,x1,y1,z1,... order.
    if feature_cols == [f"x{i}" for i in range(63)]:
        rename_map = {f"x{i}": CANONICAL_FEATURE_COLS[i] for i in range(63)}
        df = df.rename(columns=rename_map)
    elif feature_cols == CANONICAL_FEATURE_COLS:
        pass  # already in canonical order
    else:
        raise ValueError(
            f"Cannot recognize feature column schema. First few: {feature_cols[:5]}"
        )

    return df[["label"] + CANONICAL_FEATURE_COLS]


def _load_all_sources() -> tuple[pd.DataFrame, list[dict]]:
    """Read every available CSV and concatenate them with a source tag."""
    sources: list[dict] = []
    frames: list[pd.DataFrame] = []

    if PRIMARY_CSV.exists():
        df = pd.read_csv(PRIMARY_CSV)
        df = _normalize_columns(df)
        df["__source__"] = PRIMARY_CSV.name
        frames.append(df)
        sources.append({"file": str(PRIMARY_CSV), "rows": int(len(df))})
    else:
        print(f"[!] Primary CSV missing: {PRIMARY_CSV}")
        print("    Run scripts.prepare_alphabet_from_kaggle first.")

    for path in OPTIONAL_CSVS:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:  # noqa: BLE001
            print(f"[!] Could not read {path}: {e}")
            continue
        if len(df) == 0:
            continue
        try:
            df = _normalize_columns(df)
        except ValueError as e:
            print(f"[!] Skipping {path.name}: {e}")
            continue
        df["__source__"] = path.name
        frames.append(df)
        sources.append({"file": str(path), "rows": int(len(df))})

    if not frames:
        raise SystemExit("[!] No data found. Run prepare_alphabet_from_kaggle first.")

    combined = pd.concat(frames, ignore_index=True)
    return combined, sources


def _stratified_split(
    df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split that keeps label proportions in each subset.

    Classes with <3 samples go entirely to train.
    Classes with 3-5 samples: 1 each to val and test, rest to train.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    train_parts, val_parts, test_parts = [], [], []
    for label, group in df.groupby("label"):
        group = group.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(group)
        if n < 3:
            train_parts.append(group)
        else:
            n_test = max(1, int(round(n * test_frac)))
            n_val = max(1, int(round(n * val_frac)))
            # ensure at least 1 sample stays in train
            n_train = n - n_test - n_val
            if n_train < 1:
                # take 1 from val or test to keep at least 1 in train
                if n_val > 1:
                    n_val -= 1
                else:
                    n_test -= 1
                n_train = 1
            test_parts.append(group.iloc[:n_test])
            val_parts.append(group.iloc[n_test : n_test + n_val])
            train_parts.append(group.iloc[n_test + n_val :])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1, random_state=42).reset_index(drop=True) if val_parts else train_df.iloc[:0]
    test_df = (
        pd.concat(test_parts).sample(frac=1, random_state=42).reset_index(drop=True) if test_parts else train_df.iloc[:0]
    )
    return train_df, val_df, test_df


def main() -> int:
    print("=" * 60)
    print("  Build stratified train / val / test splits")
    print("=" * 60)

    combined, sources = _load_all_sources()
    print(f"Loaded {len(combined)} rows from {len(sources)} source file(s):")
    for s in sources:
        print(f"  - {Path(s['file']).name}: {s['rows']} rows")

    # Drop rows with no label or unknown label
    before = len(combined)
    combined = combined.dropna(subset=["label"])
    combined["label"] = combined["label"].astype(str).str.strip()
    combined = combined[combined["label"] != ""]
    after = len(combined)
    if before != after:
        print(f"Dropped {before - after} rows with missing/empty label")

    # Drop dynamic signs (J, Z) — they need movement, not a single frame
    excluded = sorted(set(combined["label"]) & DYNAMIC_SIGNS)
    if excluded:
        before = len(combined)
        combined = combined[~combined["label"].isin(DYNAMIC_SIGNS)].reset_index(drop=True)
        print(f"Excluded dynamic signs {excluded}: dropped {before - len(combined)} rows")

    # Class distribution
    print()
    print("Class distribution (all data):")
    counts = combined["label"].value_counts().sort_index()
    for label, n in counts.items():
        print(f"  {label:>3}: {n}")
    print(f"  TOTAL: {len(combined)} samples, {len(counts)} classes")

    # Split
    train_df, val_df, test_df = _stratified_split(combined, TRAIN_FRAC, VAL_FRAC, TEST_FRAC)
    print()
    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # Drop helper column before writing
    cols_to_drop = ["__source__"]
    for part in (train_df, val_df, test_df):
        for c in cols_to_drop:
            if c in part.columns:
                part.drop(columns=[c], inplace=True)

    # Write
    out_train = PROCESSED_DIR / "train.csv"
    out_val = PROCESSED_DIR / "val.csv"
    out_test = PROCESSED_DIR / "test.csv"
    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)
    test_df.to_csv(out_test, index=False)
    print()
    print(f"Wrote: {out_train}")
    print(f"Wrote: {out_val}")
    print(f"Wrote: {out_test}")

    # Per-split class counts
    print()
    print("Per-split class counts:")
    train_counts = train_df["label"].value_counts().sort_index()
    val_counts = val_df["label"].value_counts().sort_index() if len(val_df) else pd.Series(dtype=int)
    test_counts = test_df["label"].value_counts().sort_index() if len(test_df) else pd.Series(dtype=int)
    all_labels = sorted(set(counts.index) | set(train_counts.index) | set(val_counts.index) | set(test_counts.index))
    print(f"  {'label':>5}  {'train':>5}  {'val':>5}  {'test':>5}")
    for lbl in all_labels:
        print(
            f"  {lbl:>5}  {train_counts.get(lbl, 0):>5}  "
            f"{val_counts.get(lbl, 0):>5}  {test_counts.get(lbl, 0):>5}"
        )

    # Manifest
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "primary_source": str(PRIMARY_CSV),
        "sources": sources,
        "excluded_dynamic_signs": list(DYNAMIC_SIGNS),
        "split_fractions": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
        "random_seed": 42,
        "total_samples": int(len(combined)),
        "num_classes": int(len(counts)),
        "classes": sorted(counts.index.tolist()),
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "per_class_counts": {
            "all": {k: int(v) for k, v in counts.to_dict().items()},
            "train": {k: int(v) for k, v in train_counts.to_dict().items()},
            "val": {k: int(v) for k, v in val_counts.to_dict().items()},
            "test": {k: int(v) for k, v in test_counts.to_dict().items()},
        },
        "notes": "Test split is HELD OUT and must NOT be used during training. "
        "Use it only for the final evaluation reported in the thesis.",
    }
    manifest_path = PROCESSED_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print()
    print(f"Wrote manifest: {manifest_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
