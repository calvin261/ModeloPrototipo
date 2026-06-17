"""Re-extract hand-landmark sequences from LSC50 videos using the new
rotation+handedness-invariant normalizer.

This script reads the existing `manifest.json` and re-processes every
source .avi file with `LandmarkExtractor(rotation_invariant=True)`. The
existing .npy files (extracted with the legacy normalizer) are overwritten
in place.

Why this is needed:
  The legacy normalizer (wrist-relative + scale) leaves the hand in
  whatever in-plane rotation the user held it in. The new normalizer
  also rotates the hand so the wrist→middle-MCP vector aligns with the
  +Y axis, and mirrors left-handed hands. This removes two major
  sources of inter-user variance: how the user holds the phone, and
  whether they sign with the left or right hand.

Usage (from backend/):
    python -m scripts.reextract_sequences_invariant
    python -m scripts.reextract_sequences_invariant --manifest data/sequences/manifest.json
    python -m scripts.reextract_sequences_invariant --dry-run
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from hand_model.landmarks import LandmarkExtractor  # noqa: E402

MEDIAPIPE_MODEL = PROJECT_ROOT / "models" / "mediapipe" / "hand_landmarker.task"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "sequences" / "manifest.json"


def _re_extract_one(
    video_path: Path,
    extractor: LandmarkExtractor,
    min_frames: int,
) -> np.ndarray | None:
    """Read a single video, extract invariant landmarks frame-by-frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frames_out: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            try:
                # Use the new path that returns handedness too.
                raw, handedness = extractor.extract_raw_with_handedness(frame)
            except Exception:  # noqa: BLE001
                continue
            if raw is None:
                continue
            from hand_model.landmarks import normalize_landmarks_invariant
            lm = normalize_landmarks_invariant(raw, handedness or "Right")
            frames_out.append(lm)
    finally:
        cap.release()
    if len(frames_out) < min_frames:
        return None
    return np.stack(frames_out, axis=0).astype(np.float16)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-extract LSC50 sequences with rotation-invariant normalizer."
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--mediapipe-model", type=Path, default=MEDIAPIPE_MODEL,
    )
    parser.add_argument(
        "--min-frames", type=int, default=12,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N files (for testing).",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"[!] Manifest not found: {args.manifest}")
        return 1
    if not args.mediapipe_model.exists():
        print(f"[!] MediaPipe model missing: {args.mediapipe_model}")
        return 1

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    files = manifest["files"]
    if args.limit:
        files = files[: args.limit]

    print("=" * 60)
    print("  Re-extract with rotation+handedness invariance")
    print("=" * 60)
    print(f"Manifest:        {args.manifest.relative_to(PROJECT_ROOT)}")
    print(f"Files to process: {len(files)}")
    print(f"Min frames:      {args.min_frames}")
    print(f"Mode:            {'DRY RUN' if args.dry_run else 'WRITE'}")
    print()

    if args.dry_run:
        for f in files[:10]:
            print(f"  would re-extract: {f['file']} <- {f['source']}")
        print(f"  ... and {len(files) - 10} more")
        return 0

    extractor = LandmarkExtractor(
        model_path=str(args.mediapipe_model), rotation_invariant=True
    )
    print("Initialized MediaPipe with rotation_invariant=True")
    print()

    t0 = time.time()
    written = 0
    skipped = 0
    errors = 0
    print_interval = max(20, len(files) // 20)
    for i, entry in enumerate(files, 1):
        rel_npy = entry["file"]
        rel_avi = entry["source"]
        npy_path = PROJECT_ROOT / rel_npy
        avi_path = PROJECT_ROOT / rel_avi
        if not avi_path.exists():
            errors += 1
            continue
        try:
            seq = _re_extract_one(avi_path, extractor, args.min_frames)
        except Exception as e:  # noqa: BLE001
            errors += 1
            if errors <= 5:
                print(f"  ERROR on {rel_avi}: {e}")
            continue
        if seq is None:
            skipped += 1
            continue
        # Overwrite in place. Cast to float16 to match the legacy format
        # and keep disk usage low.
        np.save(npy_path, seq.astype(np.float16))
        written += 1
        if i % print_interval == 0 or i == len(files):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 0.1)
            eta = (len(files) - i) / max(rate, 0.1)
            print(
                f"  [{i}/{len(files)}] {rate:.1f} vid/s, ETA {eta:.0f}s, "
                f"written={written} skipped={skipped} errors={errors}"
            )
    extractor.close()
    print()
    print("=" * 60)
    print(f"  Wrote:   {written}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {errors}")
    print(f"  Wall:    {time.time() - t0:.1f}s")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
