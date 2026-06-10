"""Extract hand landmarks from the Kaggle alphabet images using MediaPipe.

Reads images from data/public/kaggle/archive/<letter>/*.jpg (21 letter folders,
~200 images each), runs the project's LandmarkExtractor on every image, and
writes a single CSV at data/processed/landmarks/alphabet_kaggle.csv.

CSV columns: label, x0, y0, z0, x1, ..., x20, y20, z20  (21 landmarks x 3 = 63)

Images where MediaPipe does not detect a hand are skipped (and reported).
Run from project root:
    python -m scripts.prepare_alphabet_from_kaggle
"""

import csv
import sys
import time
from pathlib import Path

# Make project root + src importable when run as a module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2  # noqa: E402

from hand_model.landmarks import LandmarkExtractor  # noqa: E402

INPUT_DIR = PROJECT_ROOT / "data" / "public" / "kaggle" / "archive"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "landmarks" / "alphabet_kaggle.csv"
MEDIAPIPE_MODEL = PROJECT_ROOT / "models" / "mediapipe" / "hand_landmarker.task"

# Columns: label + 21 landmarks * (x, y, z)
COLUMNS = ["label"] + [f"{c}{i}" for i in range(21) for c in ("x", "y", "z")]


def main() -> int:
    if not INPUT_DIR.exists():
        print(f"[!] Input directory not found: {INPUT_DIR}")
        print("    Make sure the Kaggle alphabet images are in data/public/kaggle/archive/")
        return 1

    if not MEDIAPIPE_MODEL.exists():
        print(f"[!] MediaPipe model not found: {MEDIAPIPE_MODEL}")
        return 1

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    letter_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])
    if not letter_dirs:
        print(f"[!] No letter folders found in {INPUT_DIR}")
        return 1

    print("=" * 60)
    print("  Prepare alphabet landmarks from Kaggle dataset")
    print("=" * 60)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Letters: {[d.name.upper() for d in letter_dirs]}")
    print()

    # Count totals upfront for progress reporting
    total_images = 0
    per_letter_count: dict[str, int] = {}
    for d in letter_dirs:
        n = len(list(d.glob("*.jpg")))
        per_letter_count[d.name.upper()] = n
        total_images += n
    print(f"Total images to process: {total_images}")
    print()

    extractor = LandmarkExtractor(model_path=str(MEDIAPIPE_MODEL))

    results: list[tuple[str, list[float]]] = []
    processed = 0
    detected = 0
    failed = 0
    failed_per_letter: dict[str, int] = {k: 0 for k in per_letter_count}
    start_time = time.time()

    try:
        for letter_dir in letter_dirs:
            label = letter_dir.name.upper()
            images = sorted(letter_dir.glob("*.jpg"))
            letter_detected = 0
            letter_failed = 0
            for img_path in images:
                processed += 1
                frame = cv2.imread(str(img_path))
                if frame is None:
                    failed += 1
                    letter_failed += 1
                    failed_per_letter[label] += 1
                    continue
                try:
                    landmarks = extractor.extract(frame)
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    letter_failed += 1
                    failed_per_letter[label] += 1
                    if failed <= 3:
                        print(f"  [!] Error on {img_path.name}: {e}")
                    continue
                if landmarks is None:
                    failed += 1
                    letter_failed += 1
                    failed_per_letter[label] += 1
                    continue
                results.append((label, landmarks.tolist()))
                detected += 1
                letter_detected += 1

                if processed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (total_images - processed) / rate if rate > 0 else 0
                    print(
                        f"  [{label}] {processed}/{total_images} "
                        f"({detected} ok, {failed} skip) "
                        f"{rate:.1f} img/s, ETA {eta:.0f}s"
                    )

            print(
                f"  [{label}] done -> detected {letter_detected}/{len(images)} "
                f"({letter_detected / len(images) * 100:.1f}%)"
                if images
                else f"  [{label}] no images"
            )
    finally:
        extractor.close()

    # Write CSV
    print()
    print(f"Writing {len(results)} samples to {OUTPUT_CSV} ...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for label, feats in results:
            writer.writerow([label] + feats)

    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"Total images seen:   {total_images}")
    print(f"Hand detected:       {detected}  ({detected / total_images * 100:.1f}%)")
    print(f"No hand / failed:    {failed}  ({failed / total_images * 100:.1f}%)")
    print(f"Wall time:           {elapsed:.1f}s ({processed / elapsed:.1f} img/s)")
    print(f"Output CSV:          {OUTPUT_CSV}")
    print()
    print("Per-letter results:")
    for label, n in per_letter_count.items():
        ok = n - failed_per_letter[label]
        pct = ok / n * 100 if n else 0
        print(f"  {label:>3}: {ok:>4}/{n:>4}  ({pct:5.1f}%)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
