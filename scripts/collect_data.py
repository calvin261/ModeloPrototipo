"""Launch the webcam data collection tool for LSEC sign language."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Collect LSEC sign language landmark data via webcam."
    )
    parser.add_argument(
        "--output-dir",
        default="data/landmarks",
        help="Directory to save landmark CSVs (default: data/landmarks)",
    )
    parser.add_argument(
        "--sequences-dir",
        default="data/sequences",
        help="Directory to save sequence .npy files (default: data/sequences)",
    )
    parser.add_argument(
        "--model",
        default="models/mediapipe/hand_landmarker.task",
        help="Path to MediaPipe hand landmarker model",
    )
    args = parser.parse_args()

    from hand_model.collect import DataCollector

    collector = DataCollector(
        output_dir=args.output_dir,
        sequences_dir=args.sequences_dir,
        model_path=args.model,
    )
    collector.run()


if __name__ == "__main__":
    main()
