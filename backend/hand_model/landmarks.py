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


def normalize_landmarks_invariant(landmarks_raw, handedness: str = "Right") -> np.ndarray:
    """Rotation- and handedness-invariant normalization for 21 hand landmarks.

    Differences from `normalize_landmarks`:
      1. Mirrors the hand if the detected hand is "Left", so every input is
         canonicalized to a right-hand frame of reference. This is critical
         because most signers use the dominant hand; treating it as the
         canonical orientation removes a major source of inter-user variance.
      2. Rotates the hand so the wrist→middle-finger-MCP vector aligns with
         the +Y axis (up). This removes in-plane rotation that depends on
         how the user holds the phone.
      3. Keeps the original wrist-relative + scale-invariant translation.

    The result has the same shape (63,) as `normalize_landmarks`, so it is a
    drop-in replacement for downstream code.

    Args:
        landmarks_raw: iterable of 21 MediaPipe NormalizedLandmark objects.
        handedness: "Right" or "Left" (MediaPipe handedness label).
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_raw])
    # 1. Mirror if left-handed: x = -x
    if handedness.lower().startswith("left"):
        coords[:, 0] = -coords[:, 0]
    # 2. Translate wrist to origin
    coords -= coords[0]
    # 3. Scale-invariant
    ref_dist = np.linalg.norm(coords[9])
    if ref_dist > 1e-6:
        coords /= ref_dist
    # 4. Rotate so wrist→MCP_9 aligns with +Y axis.
    # The XY vector from origin to landmark 9 currently has some angle theta
    # relative to +Y. We compute theta = atan2(dx, dy) (note the swap, so
    # that the rotation is around the Z axis, in the screen plane) and
    # rotate by -theta to bring (dx, dy) onto (0, length).
    v = coords[9, :2]
    theta = np.arctan2(v[0], v[1])
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    coords[:, :2] = coords[:, :2] @ rot.T
    return coords.flatten().astype(np.float32)


class LandmarkExtractor:
    """Wrapper around MediaPipe HandLandmarker for extracting normalized
    hand landmarks from BGR video frames.

    Args:
        model_path: path to hand_landmarker.task
        num_hands: max hands to detect (default 1)
        min_detection_confidence: MediaPipe threshold
        min_tracking_confidence: MediaPipe threshold
        rotation_invariant: if True, also applies rotation+handedness
            canonicalization on top of wrist-relative+scale normalization.
            Default False to keep the existing API behavior; turn on for
            training and inference of the new pipeline.
    """

    def __init__(
        self,
        model_path: str = "models/mediapipe/hand_landmarker.task",
        num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        rotation_invariant: bool = False,
    ):
        self.model_path = str(Path(model_path).resolve())
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.rotation_invariant = rotation_invariant
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
        Uses the simple wrist-relative + scale normalization (legacy behavior).
        """
        raw, handedness = self.extract_raw_with_handedness(frame)
        if raw is None:
            return None
        if self.rotation_invariant:
            return normalize_landmarks_invariant(raw, handedness or "Right")
        return normalize_landmarks(raw).astype(np.float32)

    def extract_raw(self, frame: np.ndarray):
        """Extract raw MediaPipe landmark list from a BGR frame.

        Returns the list of 21 NormalizedLandmark objects for the first
        detected hand, or None if no hand is found. Use this when you need
        access to the raw landmarks (e.g., to apply custom normalization).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33  # ~30 fps
        result = self.detector.detect_for_video(mp_image, self._timestamp_ms)
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]

    def extract_raw_with_handedness(self, frame: np.ndarray):
        """Like `extract_raw` but also returns the handedness label.

        Returns:
            (landmarks, handedness) tuple, or (None, None) if no hand found.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33  # ~30 fps
        result = self.detector.detect_for_video(mp_image, self._timestamp_ms)
        if not result.hand_landmarks:
            return None, None
        handedness = None
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name
        return result.hand_landmarks[0], handedness

    def close(self):
        self.detector.close()
