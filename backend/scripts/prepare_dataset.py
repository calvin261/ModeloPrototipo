"""Extract MediaPipe landmarks from a directory of labeled images.

Expected input structure (e.g. ASL Alphabet dataset from Kaggle):
    input_dir/
        A/
            img001.jpg
            img002.jpg
        B/
            ...

Outputs a CSV with columns: label, x0, y0, z0, ..., x20, y20, z20
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hand_model.landmarks import LandmarkExtractor, normalize_landmarks


def extract_landmarks_from_images(
    input_dir: str,
    output_dir: str = "data/landmarks",
    max_per_class: int = 0,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Input directory does not exist: {input_path}")
        return

    extractor = LandmarkExtractor(
        model_path="models/mediapipe/hand_landmarker.task"
    )

    samples = []
    class_dirs = sorted(
        [d for d in input_path.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    print(f"Found {len(class_dirs)} classes in {input_path}")

    for class_dir in class_dirs:
        label = class_dir.name.upper()
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if max_per_class > 0:
            images = images[:max_per_class]

        success = 0
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            raw = extractor.extract_raw(frame)
            if raw is None:
                continue
            landmarks = normalize_landmarks(raw).astype(np.float32)
            samples.append([label] + landmarks.tolist())
            success += 1

        print(f"  {label}: {success}/{len(images)} images processed")

    extractor.close()

    if not samples:
        print("No landmarks extracted. Check your images and model path.")
        return

    columns = ["label"] + [
        f"{c}{i}" for i in range(21) for c in ("x", "y", "z")
    ]
    df = pd.DataFrame(samples, columns=columns)

    # Split into train / val / test
    train_df, temp_df = train_test_split(
        df, test_size=(test_size + val_size), stratify=df["label"],
        random_state=seed,
    )
    relative_val = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val), stratify=temp_df["label"],
        random_state=seed,
    )

    train_df.to_csv(output_path / "lsec_alphabet_train.csv", index=False)
    val_df.to_csv(output_path / "lsec_alphabet_val.csv", index=False)
    test_df.to_csv(output_path / "lsec_alphabet_test.csv", index=False)

    print(f"\nDataset saved to {output_path}/")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Extract landmarks from labeled image directories."
    )
    parser.add_argument(
        "input_dir",
        help="Root directory with subdirectories per class (e.g. asl_alphabet/)",
    )
    parser.add_argument(
        "--output-dir", default="data/landmarks",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=0,
        help="Max images per class (0 = all)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.15,
        help="Fraction for test split",
    )
    parser.add_argument(
        "--val-size", type=float, default=0.15,
        help="Fraction for validation split",
    )
    args = parser.parse_args()

    extract_landmarks_from_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_per_class=args.max_per_class,
        test_size=args.test_size,
        val_size=args.val_size,
    )


if __name__ == "__main__":
    main()
