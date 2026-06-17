"""Build a fresh manifest for the 12-class active vocabulary from LSC50 videos.

This is the missing step between a legacy `prepare_sequences_from_lsc50.py` run
(which leaves a stale manifest with words outside the 12-class vocabulary) and
the rotation-invariant re-extraction. It walks the LSC50 source tree, picks
only the directories/words that map into the active vocabulary, and writes a
fresh manifest.json. After running this, run:

    python -m scripts.run_reextract_background
"""

import json
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT = PROJECT_ROOT / "data" / "public" / "LSC50" / "Videos Courtesy"
MEDIAPIPE_MODEL = PROJECT_ROOT / "models" / "mediapipe" / "hand_landmarker.task"
MIN_FRAMES_DETECTED = 12


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().strip().replace(" ", "_")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Build a fresh manifest for the 12-class active vocabulary.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--vocabulary", type=Path, default=PROJECT_ROOT / "data" / "sequences" / "vocabulary.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "sequences" / "manifest.json")
    args = parser.parse_args()

    with open(args.vocabulary, encoding="utf-8") as f:
        active_vocab = set(json.load(f).keys())

    # Default mapping (same as the legacy script's)
    default_mapping = {
        "Hola": "hola",
        "Gracias": "gracias",
        "Por favor": "por_favor",
        "Si": "si",
        "No": "no",
        "Ayudar": "ayuda",
        "Ba\u00f1o": "bano",
        "Como esta": "como_estas",
        "Bien": "bien",
        "Mal": "mal",
        "Perdon": "perdon",
        "Chao": "adios",
    }
    print(f"Active vocabulary ({len(active_vocab)} classes): {sorted(active_vocab)}")

    files = []
    for signer_dir in sorted(args.input_dir.iterdir()):
        if not signer_dir.is_dir():
            continue
        signer_id = signer_dir.name
        for cat_dir in signer_dir.iterdir():
            if not cat_dir.is_dir():
                continue
            for word_dir in sorted(cat_dir.iterdir()):
                if not word_dir.is_dir():
                    continue
                word_display = word_dir.name
                word_key = default_mapping.get(word_display, _slugify(word_display))
                if word_key not in active_vocab:
                    continue
                for video in sorted(word_dir.glob("*.avi")):
                    files.append({
                        "file": f"data\\sequences\\{word_key}\\{signer_id}_{word_key}_{len([f for f in files if f['word'] == word_key]) + 1:03d}.npy",
                        "signer": signer_id,
                        "word": word_key,
                        "n_frames": 0,  # placeholder, will be updated by re-extraction
                        "source": str(video.relative_to(PROJECT_ROOT)),
                    })

    by_word = {}
    by_signer = {}
    for f in files:
        by_word[f["word"]] = by_word.get(f["word"], 0) + 1
        by_signer[f["signer"]] = by_signer.get(f["signer"], 0) + 1

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(args.input_dir.relative_to(PROJECT_ROOT)),
        "min_frames": MIN_FRAMES_DETECTED,
        "word_mapping_overrides": default_mapping,
        "active_vocabulary": sorted(active_vocab),
        "total_sequences": len(files),
        "by_word": dict(sorted(by_word.items())),
        "by_signer": dict(sorted(by_signer.items())),
        "files": files,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print(f"Wrote manifest with {len(files)} entries to: {args.output.relative_to(PROJECT_ROOT)}")
    print(f"By word: {by_word}")
    print(f"By signer: {by_signer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
