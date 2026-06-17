"""Clean up extra word directories that are not in the 12-class active vocabulary.

After a legacy re-extraction (or someone running prepare_sequences_from_lsc50.py
with the full vocabulary), extra directories like 'apoyar', 'buenas_noches', etc.
may exist in data/sequences/. The rotation-invariant re-extraction expects only
the 12 active classes. This script removes the extras.

The 12 active classes are: adios, ayuda, bano, bien, como_estas, gracias,
hola, mal, no, perdon, por_favor, si.
"""

import json
import shutil
from pathlib import Path

ROOT = Path(r'C:\Users\jeffe\Documents\Proyectos\ModeloPrototipo\backend')
SEQ_DIR = ROOT / 'data' / 'sequences'

with open(SEQ_DIR / 'vocabulary.json', encoding='utf-8') as f:
    active = set(json.load(f).keys())

print(f'Active vocabulary: {sorted(active)}')
print()

current = {d.name for d in SEQ_DIR.iterdir() if d.is_dir()}
extras = current - active - {'fixtures'}  # keep fixtures if present
print(f'Current directories: {sorted(current)}')
print(f'Extra (non-vocabulary) directories: {sorted(extras)}')

if not extras:
    print('\nNo extras to remove.')
    raise SystemExit(0)

# Count files in extras
total = 0
for d in extras:
    n = len(list((SEQ_DIR / d).glob('*.npy')))
    total += n
    print(f'  {d}/  {n} files')

print(f'\nWill remove {len(extras)} dirs ({total} .npy files).')

# Use --dry-run via env var or just print
import os
if os.environ.get('CONFIRM') != 'yes':
    print('\nDry run. Set CONFIRM=yes to actually delete.')
    raise SystemExit(0)

for d in extras:
    full = SEQ_DIR / d
    print(f'Removing {full}')
    shutil.rmtree(full, ignore_errors=True)

# Also remove the old manifest so the re-extraction builds a fresh one
manifest = SEQ_DIR / 'manifest.json'
if manifest.exists():
    print(f'Removing old manifest {manifest}')
    manifest.unlink()

print('Done.')
