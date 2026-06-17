"""Export the newly trained MLP and GRU checkpoints to ONNX and update
models/exports/registry.json so the backend serves the new versions.

Run from project root:
    python -m scripts.export_new_models
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.services.export_service import (  # noqa: E402
    export_classifier_onnx_only,
    export_sequence_onnx_only,
)

EXPORTS_DIR = PROJECT_ROOT / "models" / "exports"
REGISTRY_PATH = EXPORTS_DIR / "registry.json"

MLP_CKPT = PROJECT_ROOT / "models" / "classifier" / "sign_mlp.pth"
GRU_CKPT = PROJECT_ROOT / "models" / "sequence" / "sign_gru.pth"


def main() -> int:
    if not MLP_CKPT.exists():
        print(f"[!] MLP checkpoint missing: {MLP_CKPT}")
        return 1
    if not GRU_CKPT.exists():
        print(f"[!] GRU checkpoint missing: {GRU_CKPT}")
        return 1

    print("=" * 60)
    print("  Export new models to ONNX + update registry")
    print("=" * 60)
    print(f"MLP:  {MLP_CKPT.relative_to(PROJECT_ROOT)}")
    print(f"GRU:  {GRU_CKPT.relative_to(PROJECT_ROOT)}")
    print(f"Out:  {EXPORTS_DIR.relative_to(PROJECT_ROOT)}")
    print()

    # 1) Export MLP -> ONNX
    print("[1/3] Exporting MLP to ONNX ...")
    mlp_entry = export_classifier_onnx_only(str(MLP_CKPT), str(EXPORTS_DIR))
    mlp_entry["accuracy"] = 0.7925  # measured test accuracy from 2026-06-06
    print(f"      -> {mlp_entry['onnx_path']}  ({mlp_entry['size_kb']} KB)")
    print(f"      classes: {mlp_entry['num_classes']}")

    # 2) Export GRU -> ONNX
    print()
    print("[2/3] Exporting GRU to ONNX ...")
    gru_entry = export_sequence_onnx_only(str(GRU_CKPT), str(EXPORTS_DIR))
    gru_entry["accuracy"] = 0.7519  # measured test accuracy from 2026-06-07
    print(f"      -> {gru_entry['onnx_path']}  ({gru_entry['size_kb']} KB)")
    print(f"      classes: {gru_entry['num_classes']}")

    # 3) Update registry.json
    print()
    print("[3/3] Updating registry ...")
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {"classifier": [], "sequence": []}

    registry.setdefault("classifier", []).append(mlp_entry)
    registry.setdefault("sequence", []).append(gru_entry)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    # Show summary
    print()
    print("Registry summary:")
    print(f"  classifier: {len(registry.get('classifier', []))} versions")
    print(f"  sequence:   {len(registry.get('sequence', []))} versions")
    print()
    print("Latest entries:")
    print(f"  classifier: {mlp_entry['version']}  acc={mlp_entry['accuracy']}")
    print(f"  sequence:   {gru_entry['version']}  acc={gru_entry['accuracy']}")
    print()
    print("=" * 60)
    print("  Done. Restart the backend to pick up the new registry.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
