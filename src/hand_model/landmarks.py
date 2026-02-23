from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


def normalize_landmarks(landmarks_raw) -> np.ndarray:
    """Normalize 21 hand landmarks to wrist-relative, scale-invariant vector.

    Translates so wrist (landmark 0) is at origin, then scales so the
    distance from wrist to middle-finger MCP (landmark 9) equals 1.0.

    Returns a flat array of shape (63,).
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_raw])
    coords -= coords[0]
    ref_dist = np.linalg.norm(coords[9])
    if ref_dist > 1e-6:
        coords /= ref_dist
    return coords.flatten()


class LandmarkExtractor:
    """Wrapper around MediaPipe HandLandmarker for extracting normalized
    hand landmarks from BGR video frames."""

    def __init__(
        self,
        model_path: str = "models/mediapipe/hand_landmarker.task",
        num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.model_path = str(Path(model_path).resolve())
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._init_detector()

    def _init_detector(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_tracking_confidence,
        )
        self.detector = HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def extract(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract normalized landmark vector from a BGR frame.

        Returns a (63,) float32 array, or None if no hand is detected.
        """
        raw = self.extract_raw(frame)
        if raw is None:
            return None
        return normalize_landmarks(raw).astype(np.float32)

    def extract_raw(self, frame: np.ndarray):
        """Extract raw MediaPipe landmark list from a BGR frame.

        Returns the list of 21 NormalizedLandmark objects for the first
        detected hand, or None if no hand is found.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33  # ~30 fps
        result = self.detector.detect_for_video(mp_image, self._timestamp_ms)
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]

    def close(self):
        self.detector.close()
