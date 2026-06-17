"""Service for managing training data uploads."""

import csv
from pathlib import Path

import numpy as np


class DataService:
    """Manages landmark data storage for training."""

    def __init__(self, landmarks_dir: str, sequences_dir: str):
        self.landmarks_dir = Path(landmarks_dir)
        self.sequences_dir = Path(sequences_dir)
        self.landmarks_dir.mkdir(parents=True, exist_ok=True)
        self.sequences_dir.mkdir(parents=True, exist_ok=True)

    def append_static_samples(
        self, samples: list[dict], csv_name: str = "mobile_upload.csv"
    ) -> int:
        """Append static landmark samples to a CSV file.

        Each sample has 'label' (str) and 'landmarks' (list of 63 floats).
        Returns number of samples added.
        """
        csv_path = self.landmarks_dir / csv_name
        file_exists = csv_path.exists()

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ["label"] + [f"x{i}" for i in range(63)]
                writer.writerow(header)
            for sample in samples:
                row = [sample["label"]] + [str(v) for v in sample["landmarks"]]
                writer.writerow(row)

        return len(samples)

    def save_sequence(self, label: str, sequence_data: np.ndarray) -> int:
        """Save a sequence of landmarks as .npy file.

        Returns the number of frames in the sequence.
        """
        label_dir = self.sequences_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        existing = list(label_dir.glob(f"{label}_*.npy"))
        idx = len(existing)
        filename = f"{label}_{idx:04d}.npy"
        np.save(label_dir / filename, sequence_data)

        return sequence_data.shape[0]

    def get_stats(self) -> dict:
        """Get statistics about stored training data."""
        static_labels = {}
        total_static = 0

        for csv_file in self.landmarks_dir.glob("*.csv"):
            with open(csv_file) as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    continue
                for row in reader:
                    if row:
                        label = row[0]
                        static_labels[label] = static_labels.get(label, 0) + 1
                        total_static += 1

        sequence_labels = {}
        total_sequences = 0
        for label_dir in sorted(self.sequences_dir.iterdir()):
            if label_dir.is_dir():
                count = len(list(label_dir.glob("*.npy")))
                if count > 0:
                    sequence_labels[label_dir.name] = count
                    total_sequences += count

        return {
            "static_samples": total_static,
            "sequence_samples": total_sequences,
            "static_labels": static_labels,
            "sequence_labels": sequence_labels,
        }

    def get_total_static(self) -> int:
        """Get total number of static samples across all CSVs."""
        total = 0
        for csv_file in self.landmarks_dir.glob("*.csv"):
            with open(csv_file) as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                total += sum(1 for _ in reader)
        return total

    def delete_static_label(self, label: str) -> int:
        """Remove all samples of a given label from all CSVs. Returns rows deleted."""
        deleted = 0
        for csv_file in self.landmarks_dir.glob("*.csv"):
            kept = []
            header = None
            with open(csv_file, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    continue
                for row in reader:
                    if row and row[0] == label:
                        deleted += 1
                    else:
                        kept.append(row)
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                if header:
                    writer.writerow(header)
                writer.writerows(kept)
        return deleted

    def delete_sequence_label(self, label: str) -> int:
        """Remove all .npy sequence files for a label. Returns files deleted."""
        label_dir = self.sequences_dir / label
        if not label_dir.exists():
            return 0
        files = list(label_dir.glob("*.npy"))
        for f in files:
            f.unlink()
        try:
            label_dir.rmdir()
        except OSError:
            pass
        return len(files)
