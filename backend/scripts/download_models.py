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
    "pose_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}


def download(name: str, url: str) -> None:
    # 1. Download to frontend assets
    dest_frontend = ASSETS_DIR / name
    if not dest_frontend.exists():
        print(f"  Descargando {name} para frontend...")
        urllib.request.urlretrieve(url, dest_frontend)
        print(f"  [OK] {name} guardado en frontend ({dest_frontend.stat().st_size // 1024} KB)")
    else:
        print(f"  [OK] {name} ya existe en frontend ({dest_frontend.stat().st_size // 1024} KB)")

    # 2. Copy/download to backend models
    dest_backend = REPO_ROOT / "backend" / "models" / "mediapipe" / name
    dest_backend.parent.mkdir(parents=True, exist_ok=True)
    if not dest_backend.exists():
        import shutil
        print(f"  Copiando {name} a backend...")
        shutil.copy2(dest_frontend, dest_backend)
        print(f"  [OK] {name} guardado en backend")
    else:
        print(f"  [OK] {name} ya existe en backend")


if __name__ == "__main__":
    print("=" * 50)
    print("  Descargando modelos para LSEC Translator")
    print("=" * 50)
    for model_name, model_url in MODELS.items():
        download(model_name, model_url)
    print()
    print("Listo. Ahora compila la app en Android Studio.")
