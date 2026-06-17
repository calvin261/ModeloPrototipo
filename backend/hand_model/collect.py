import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .landmarks import LandmarkExtractor


# MediaPipe hand landmark connections for skeleton drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]


def _draw_landmarks_on_frame(frame, raw_landmarks):
    """Draw hand skeleton on the frame using raw landmark positions."""
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in raw_landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
    for pt in points:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)


class DataCollector:
    """Webcam-based data collection tool for sign language landmarks.

    Controls:
        S         — Save current landmark frame with current label
        +/=       — Next letter
        -         — Previous letter
        R         — Toggle sequence recording (dynamic mode)
        ENTER     — Export collected data to CSV
        Q         — Quit
    """

    WORDS = [
        "hola",
        "gracias",
        "por_favor",
        "si",
        "no",
        "ayuda",
        "agua",
        "comida",
        "bano",
        "nombre",
        "como_estas",
        "bien",
        "mal",
        "perdon",
        "adios",
    ]
    ALPHABET = WORDS + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE"]

    def __init__(
        self,
        output_dir: str = "data/landmarks",
        sequences_dir: str = "data/sequences",
        model_path: str = "models/mediapipe/hand_landmarker.task",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sequences_dir = Path(sequences_dir)
        self.sequences_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = LandmarkExtractor(model_path=model_path)
        self.samples = []
        self.label_index = 0
        # Sequence recording state
        self.recording = False
        self.current_sequence = []
        self.sequence_label = ""

    @property
    def current_label(self):
        return self.ALPHABET[self.label_index]

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Data Collector started. Press Q to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            raw = self.extractor.extract_raw(frame)
            landmarks = None
            if raw is not None:
                from .landmarks import normalize_landmarks
                landmarks = normalize_landmarks(raw).astype(np.float32)
                _draw_landmarks_on_frame(frame, raw)

            # Sequence recording
            if self.recording and landmarks is not None:
                self.current_sequence.append(landmarks)

            self._draw_ui(frame, landmarks is not None)
            cv2.imshow("LSEC Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s") and landmarks is not None:
                self.samples.append([self.current_label] + landmarks.tolist())
                print(
                    f"  Saved '{self.current_label}' "
                    f"(total: {len(self.samples)})"
                )
            elif key in (ord("+"), ord("=")):
                self.label_index = min(
                    self.label_index + 1, len(self.ALPHABET) - 1
                )
            elif key == ord("-"):
                self.label_index = max(self.label_index - 1, 0)
            elif key == ord("r"):
                self._toggle_recording()
            elif key == 13:  # ENTER
                self._save_csv()

        cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()

    def _toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.current_sequence = []
            self.sequence_label = self.current_label
            print(f"  Recording sequence for '{self.sequence_label}'...")
        else:
            self.recording = False
            if len(self.current_sequence) > 5:
                self._save_sequence()
            else:
                print("  Sequence too short, discarded.")

    def _save_sequence(self):
        label_dir = self.sequences_dir / self.sequence_label
        label_dir.mkdir(parents=True, exist_ok=True)
        existing = list(label_dir.glob("*.npy"))
        idx = len(existing) + 1
        filename = f"{self.sequence_label}_{idx:03d}.npy"
        arr = np.array(self.current_sequence, dtype=np.float32)
        np.save(label_dir / filename, arr)
        print(
            f"  Saved sequence '{self.sequence_label}' "
            f"({arr.shape[0]} frames) -> {filename}"
        )

    def _save_csv(self):
        if not self.samples:
            print("  No samples to save.")
            return
        columns = ["label"] + [
            f"{c}{i}" for i in range(21) for c in ("x", "y", "z")
        ]
        df = pd.DataFrame(self.samples, columns=columns)
        output_path = self.output_dir / "collected_data.csv"
        if output_path.exists():
            existing = pd.read_csv(output_path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(output_path, index=False)
        count = len(self.samples)
        self.samples.clear()
        print(f"  Exported {count} samples -> {output_path}")

    def _draw_ui(self, frame, hand_detected: bool):
        h, w = frame.shape[:2]

        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        color = (0, 255, 0) if hand_detected else (0, 0, 255)
        status = "Hand detected" if hand_detected else "No hand"
        cv2.putText(
            frame, f"Label: {self.current_label}  |  {status}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )
        cv2.putText(
            frame,
            f"Samples: {len(self.samples)}  |  [S]ave [+/-]Letter [Enter]Export [Q]uit",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        if self.recording:
            rec_text = f"REC [{len(self.current_sequence)} frames] - {self.sequence_label}"
            cv2.putText(
                frame, rec_text,
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
