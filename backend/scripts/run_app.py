"""Launch the real-time LSEC sign language recognition application."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Real-time LSEC sign language recognition with audio output."
    )
    parser.add_argument(
        "--mode", choices=["static", "dynamic"], default="static",
        help="Recognition mode: 'static' for fingerspelling, 'dynamic' for words",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.7,
        help="Minimum confidence threshold (default: 0.7)",
    )
    parser.add_argument(
        "--confirm-frames", type=int, default=10,
        help="Consecutive frames required to confirm a sign (default: 10)",
    )
    parser.add_argument(
        "--classifier-model", default="models/classifier/sign_mlp.pth",
        help="Path to trained MLP classifier weights",
    )
    parser.add_argument(
        "--sequence-model", default="models/sequence/sign_gru.pth",
        help="Path to trained GRU sequence model weights",
    )
    args = parser.parse_args()

    from hand_model.config import AppConfig
    from hand_model.app import SignLanguageApp

    cfg = AppConfig(
        mode=args.mode,
        camera_index=args.camera,
        confidence_threshold=args.confidence,
        confirm_frames=args.confirm_frames,
        classifier_model=args.classifier_model,
        sequence_model=args.sequence_model,
    )

    app = SignLanguageApp(cfg)
    app.run()


if __name__ == "__main__":
    main()
