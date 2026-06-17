"""
Script para descargar los modelos MediaPipe necesarios para la app.
Ejecutar UNA SOLA VEZ antes de compilar la app.

Uso:
    cd backend && python scripts/download_models.py
"""

import urllib.request
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "frontend" / "app" / "src" / "main" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
}


def download(name: str, url: str) -> None:
    dest = ASSETS_DIR / name
    if dest.exists():
        print(f"  [OK] {name} ya existe ({dest.stat().st_size // 1024} KB)")
        return
    print(f"  Descargando {name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  [OK] {name} guardado ({dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    print("=" * 50)
    print("  Descargando modelos para LSEC Translator")
    print("=" * 50)
    for model_name, model_url in MODELS.items():
        download(model_name, model_url)
    print()
    print("Listo. Ahora compila la app en Android Studio.")
