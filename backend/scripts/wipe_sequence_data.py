"""Wipe the .npy files in data/sequences/ so the re-extraction starts clean.

Preserves:
- data/sequences/vocabulary.json
- data/sequences/splits.json
- data/sequences/loso_splits.json
- data/sequences/manifest.json (will be re-read by the re-extractor)

Removes:
- all <word>/*.npy files
- any stale <word>/ directories that are not in the active vocabulary
"""

import json
import shutil
from pathlib import Path

ROOT = Path(r'C:\Users\jeffe\Documents\Proyectos\ModeloPrototipo\backend')
SEQ_DIR = ROOT / 'data' / 'sequences'

with open(SEQ_DIR / 'vocabulary.json', encoding='utf-8') as f:
    active = set(json.load(f).keys())

print(f'Active vocabulary: {sorted(active)}')

# Also read manifest to see what files the re-extractor will need
with open(SEQ_DIR / 'manifest.json', encoding='utf-8') as f:
    manifest = json.load(f)
print(f'Manifest entries: {len(manifest["files"])}')

# Count current .npy
current = {}
for d in SEQ_DIR.iterdir():
    if d.is_dir() and d.name in active:
        current[d.name] = len(list(d.glob('*.npy')))
print(f'Current .npy per word: {current}')

# Wipe .npy files in active word directories
total = 0
for d in SEQ_DIR.iterdir():
    if d.is_dir() and d.name in active:
        for f in d.glob('*.npy'):
            f.unlink()
            total += 1
print(f'Wiped {total} .npy files from active word directories.')

# Remove any extra dirs not in active vocabulary
for d in SEQ_DIR.iterdir():
    if d.is_dir() and d.name not in active and d.name != 'fixtures':
        n = len(list(d.glob('*.npy')))
        if n:
            print(f'Removing stale dir {d.name}/ ({n} files)')
            shutil.rmtree(d, ignore_errors=True)
        else:
            d.rmdir()

# Verify
remaining = sum(1 for _ in SEQ_DIR.glob('*/*.npy'))
print(f'Remaining .npy files: {remaining} (should be 0)')
