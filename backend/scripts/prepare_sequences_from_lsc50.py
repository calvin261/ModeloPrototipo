"""Extract hand-landmark sequences from LSC50 video dataset using MediaPipe.

Walks the LSC50 directory tree (default: data/public/LSC50/Videos Courtesy)
and extracts normalized (63-dim) landmarks for every frame of every video.
Saves one .npy file per video at data/sequences/<word>/<signer>_<word>_<idx>.npy
with shape (T, 63) where T = number of frames where a hand was detected.

Usage (from project root):
    python -m scripts.prepare_sequences_from_lsc50
    python -m scripts.prepare_sequences_from_lsc50 --signers 1 2 3 --words Hola Gracias
    python -m scripts.prepare_sequences_from_lsc50 --workers 4
    python -m scripts.prepare_sequences_from_lsc50 --words-mapping "Hola=hola" "Gracias=gracias"

Notes:
- Sequences with < MIN_FRAMES_DETECTED valid frames are skipped.
- All sequences are saved as float16 to keep disk usage small
  (~3KB per sequence); the training loader casts back to float32.
- A manifest.json is written alongside the sequences with per-file
  metadata (signer, word, n_frames, source_path).
"""

import argparse
import json
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from hand_model.landmarks import LandmarkExtractor  # noqa: E402

DEFAULT_INPUT = PROJECT_ROOT / "data" / "public" / "LSC50" / "Videos Courtesy"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "sequences"
MEDIAPIPE_MODEL = PROJECT_ROOT / "models" / "mediapipe" / "hand_landmarker.task"

# Minimum number of frames with a detected hand to keep a sequence.
# Below this the model has too little temporal information to learn.
MIN_FRAMES_DETECTED = 12


def _slugify(text: str) -> str:
    """Lowercase, strip accents, replace spaces with underscores."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip().replace(" ", "_")
    return text


def _build_word_mapping(overrides: list[str]) -> dict[str, str]:
    """Map LSC50 word folder names (display) -> vocabulary keys (snake_case).

    Default mapping uses slugify on the LSC50 name. CLI overrides of the
    form "LSCName=vocab_key" win.
    """
    mapping: dict[str, str] = {}
    if overrides:
        for ov in overrides:
            if "=" not in ov:
                print(f"[!] Ignoring --words-mapping value (no '='): {ov!r}")
                continue
            src, dst = ov.split("=", 1)
            mapping[src.strip()] = dst.strip()
    return mapping


def _process_one_video(
    video_path: Path,
    word_display: str,
    signer_id: str,
    extractor: LandmarkExtractor,
    min_frames: int,
) -> Optional[np.ndarray]:
    """Extract a (T, 63) float16 sequence from a single .avi file.

    Returns None if fewer than `min_frames` frames had a detected hand.
    """
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
                landmarks = extractor.extract(frame)
            except Exception:  # noqa: BLE001
                continue
            if landmarks is not None:
                frames_out.append(landmarks)
    finally:
        cap.release()

    if len(frames_out) < min_frames:
        return None
    return np.stack(frames_out, axis=0).astype(np.float16)


def _process_all_serial(
    video_jobs: list[tuple[Path, str, str]],
    out_root: Path,
    min_frames: int,
    mediapipe_model: Path,
    word_mapping: dict[str, str],
) -> list[dict]:
    """Process all videos sequentially. Returns the manifest list."""
    extractor = LandmarkExtractor(model_path=str(mediapipe_model))
    manifest: list[dict] = []
    print_interval = max(50, len(video_jobs) // 20)
    t0 = time.time()
    try:
        for i, (video_path, word_display, signer_id) in enumerate(video_jobs, 1):
            word_key = word_mapping.get(word_display, _slugify(word_display))
            seq = _process_one_video(
                video_path, word_display, signer_id, extractor, min_frames
            )
            if seq is None:
                if i % print_interval == 0:
                    elapsed = time.time() - t0
                    rate = i / max(elapsed, 0.1)
                    eta = (len(video_jobs) - i) / max(rate, 0.1)
                    print(
                        f"  [{i}/{len(video_jobs)}] {rate:.1f} vid/s, "
                        f"ETA {eta:.0f}s, kept {len(manifest)} so far"
                    )
                continue
            word_dir = out_root / word_key
            word_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(word_dir.glob(f"{signer_id}_{word_key}_*.npy"))
            idx = len(existing) + 1
            out_path = word_dir / f"{signer_id}_{word_key}_{idx:03d}.npy"
            np.save(out_path, seq)
            manifest.append(
                {
                    "file": str(out_path.relative_to(PROJECT_ROOT)),
                    "signer": signer_id,
                    "word": word_key,
                    "n_frames": int(seq.shape[0]),
                    "source": str(video_path.relative_to(PROJECT_ROOT)),
                }
            )
            if i % print_interval == 0:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 0.1)
                eta = (len(video_jobs) - i) / max(rate, 0.1)
                print(
                    f"  [{i}/{len(video_jobs)}] {rate:.1f} vid/s, "
                    f"ETA {eta:.0f}s, kept {len(manifest)} so far"
                )
    finally:
        extractor.close()
    return manifest


def _discover_jobs(input_dir: Path, only_signers: Optional[list[str]],
                   only_words: Optional[list[str]]) -> list[tuple[Path, str, str]]:
    """Find (video_path, word, signer_id) triples under input_dir.

    LSC50 layout:
        Videos Courtesy/<signer_id>/Cordialidad/<word>/*.avi
    """
    jobs: list[tuple[Path, str, str]] = []
    if not input_dir.exists():
        return jobs
    for signer_dir in sorted(input_dir.iterdir()):
        if not signer_dir.is_dir():
            continue
        signer_id = signer_dir.name
        if only_signers and signer_id not in only_signers:
            continue
        # The category subfolder is "Cordialidad" / "Colores" / "Numeros"
        for cat_dir in signer_dir.iterdir():
            if not cat_dir.is_dir():
                continue
            for word_dir in sorted(cat_dir.iterdir()):
                if not word_dir.is_dir():
                    continue
                word = word_dir.name
                if only_words and word not in only_words:
                    continue
                for video in sorted(word_dir.glob("*.avi")):
                    jobs.append((video, word, signer_id))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract hand-landmark sequences from LSC50 videos."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT,
        help=f"LSC50 category root (default: {DEFAULT_INPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help=f"Where to write .npy sequences (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--mediapipe-model", type=Path, default=MEDIAPIPE_MODEL,
        help="Path to hand_landmarker.task",
    )
    parser.add_argument(
        "--signers", nargs="+", default=None,
        help="Restrict to these signer IDs (e.g., 1 2 3). Default: all.",
    )
    parser.add_argument(
        "--words", nargs="+", default=None,
        help='Restrict to these LSC50 word names (e.g., "Hola" "Gracias"). Default: all.',
    )
    parser.add_argument(
        "--words-mapping", nargs="+", default=[],
        help='Override word mapping, e.g. "Gracias=gracias". Multiple allowed.',
    )
    parser.add_argument(
        "--min-frames", type=int, default=MIN_FRAMES_DETECTED,
        help=f"Minimum hand-detected frames to keep a sequence (default: {MIN_FRAMES_DETECTED}).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default 1 = serial). Each worker boots its own MediaPipe model.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just list what would be processed; do not extract.",
    )
    args = parser.parse_args()

    if not args.mediapipe_model.exists():
        print(f"[!] MediaPipe model missing: {args.mediapipe_model}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    word_mapping = _build_word_mapping(args.words_mapping)

    print("=" * 60)
    print("  LSC50 -> sequence extraction")
    print("=" * 60)
    print(f"Input:        {args.input_dir.relative_to(PROJECT_ROOT)}")
    print(f"Output:       {args.output_dir.relative_to(PROJECT_ROOT)}")
    print(f"Min frames:   {args.min_frames}")
    print(f"Workers:      {args.workers}")
    print()

    jobs = _discover_jobs(args.input_dir, args.signers, args.words)
    if not jobs:
        print(f"[!] No videos found under {args.input_dir}")
        return 1

    # Stats before extraction
    signer_counts: Counter[str] = Counter(j[2] for j in jobs)
    word_counts: Counter[str] = Counter(j[1] for j in jobs)
    print(f"Discovered {len(jobs)} videos from {len(signer_counts)} signers, {len(word_counts)} words")
    print(f"Signers: {sorted(signer_counts.keys())[:10]}{'...' if len(signer_counts) > 10 else ''}")
    print()

    if args.dry_run:
        print("DRY RUN — first 20 jobs:")
        for j in jobs[:20]:
            print(f"  signer={j[2]:>3} word={j[1]:<20} -> {j[0].name}")
        print(f"  ... and {len(jobs) - 20} more")
        return 0

    start = time.time()
    if args.workers == 1:
        manifest = _process_all_serial(
            jobs, args.output_dir, args.min_frames, args.mediapipe_model, word_mapping
        )
    else:
        # Parallel path: each worker reuses the same LandmarkExtractor init code
        # via a worker initializer.
        from concurrent.futures import ProcessPoolExecutor

        # Chunk the jobs so each worker boots its detector once.
        # We send extractor-relevant args only, not the live extractor object
        # (extractor cannot be pickled).
        chunks = [jobs[i::args.workers] for i in range(args.workers)]

        def _worker(chunk: list[tuple[Path, str, str]]) -> list[dict]:
            extractor = LandmarkExtractor(model_path=str(args.mediapipe_model))
            out: list[dict] = []
            try:
                for video_path, word_display, signer_id in chunk:
                    word_key = word_mapping.get(word_display, _slugify(word_display))
                    seq = _process_one_video(
                        video_path, word_display, signer_id, extractor, args.min_frames
                    )
                    if seq is None:
                        continue
                    word_dir = args.output_dir / word_key
                    word_dir.mkdir(parents=True, exist_ok=True)
                    existing = sorted(word_dir.glob(f"{signer_id}_{word_key}_*.npy"))
                    idx = len(existing) + 1
                    out_path = word_dir / f"{signer_id}_{word_key}_{idx:03d}.npy"
                    np.save(out_path, seq)
                    out.append(
                        {
                            "file": str(out_path.relative_to(PROJECT_ROOT)),
                            "signer": signer_id,
                            "word": word_key,
                            "n_frames": int(seq.shape[0]),
                            "source": str(video_path.relative_to(PROJECT_ROOT)),
                        }
                    )
            finally:
                extractor.close()
            return out

        manifest = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for chunk_result in ex.map(_worker, chunks):
                manifest.extend(chunk_result)

    elapsed = time.time() - start

    # Write manifest
    manifest_path = args.output_dir / "manifest.json"
    by_word = Counter(m["word"] for m in manifest)
    by_signer = Counter(m["signer"] for m in manifest)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(args.input_dir.relative_to(PROJECT_ROOT)),
        "min_frames": args.min_frames,
        "word_mapping_overrides": word_mapping,
        "total_sequences": len(manifest),
        "by_word": dict(by_word.most_common()),
        "by_signer": dict(sorted(by_signer.items())),
        "files": manifest,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"Videos seen:        {len(jobs)}")
    print(f"Sequences saved:    {len(manifest)}")
    print(f"Wall time:          {elapsed:.1f}s ({len(jobs) / max(elapsed, 0.1):.1f} vid/s)")
    print(f"Manifest:           {manifest_path.relative_to(PROJECT_ROOT)}")
    print()
    print("Sequences per word:")
    for w, n in by_word.most_common():
        print(f"  {w:<20} {n:>4}")
    print()
    print(f"Signers represented: {len(by_signer)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
