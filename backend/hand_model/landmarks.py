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


def normalize_holistic(left_hand, right_hand, pose_landmarks) -> np.ndarray:
    """Normalize left hand, right hand, and key pose landmarks to torso-relative, scale-invariant.

    Returns a flat array of shape (147,).
    """
    result = np.zeros(147, dtype=np.float32)
    origin = np.zeros(3, dtype=np.float32)
    scale = 1.0

    # If pose is available, we compute shoulder midpoint (origin) and shoulder distance (scale)
    if pose_landmarks is not None and len(pose_landmarks) > 12:
        l_sh = pose_landmarks[11]
        r_sh = pose_landmarks[12]
        l_sh_v = np.array([l_sh.x, l_sh.y, l_sh.z], dtype=np.float32)
        r_sh_v = np.array([r_sh.x, r_sh.y, r_sh.z], dtype=np.float32)
        origin = (l_sh_v + r_sh_v) / 2.0
        scale = float(np.linalg.norm(l_sh_v - r_sh_v))
        if scale < 1e-6:
            scale = 1.0

    # 1. Left Hand (indices 0 to 62)
    if left_hand is not None:
        coords = np.array([[lm.x, lm.y, lm.z] for lm in left_hand], dtype=np.float32)
        coords -= origin
        coords /= scale
        result[0:63] = coords.flatten()

    # 2. Right Hand (indices 63 to 125)
    if right_hand is not None:
        coords = np.array([[lm.x, lm.y, lm.z] for lm in right_hand], dtype=np.float32)
        coords -= origin
        coords /= scale
        result[63:126] = coords.flatten()

    # 3. Pose keypoints: [0, 11, 12, 13, 14, 15, 16] (indices 126 to 146)
    if pose_landmarks is not None:
        key_indices = [0, 11, 12, 13, 14, 15, 16]
        pose_coords = []
        for idx in key_indices:
            if idx < len(pose_landmarks):
                lm = pose_landmarks[idx]
                pose_coords.append([lm.x, lm.y, lm.z])
            else:
                pose_coords.append([0.0, 0.0, 0.0])
        coords = np.array(pose_coords, dtype=np.float32)
        coords -= origin
        coords /= scale
        result[126:147] = coords.flatten()

    return result


class LandmarkExtractor:
    """Wrapper around MediaPipe HandLandmarker and PoseLandmarker for extracting
    normalized holistic landmarks from BGR video frames.

    Args:
        model_path: path to hand_landmarker.task
        pose_model_path: path to pose_landmarker.task
        num_hands: max hands to detect (default 2)
        min_detection_confidence: MediaPipe threshold
        min_tracking_confidence: MediaPipe threshold
        rotation_invariant: if True, also applies rotation+handedness
            canonicalization on top of wrist-relative+scale normalization.
            (Used for legacy hand-only pipeline).
    """

    def __init__(
        self,
        model_path: str = "models/mediapipe/hand_landmarker.task",
        pose_model_path: str = "models/mediapipe/pose_landmarker.task",
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        rotation_invariant: bool = False,
    ):
        self.model_path = str(Path(model_path).resolve())
        self.pose_model_path = str(Path(pose_model_path).resolve())
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.rotation_invariant = rotation_invariant
        self._init_detector()

    def _init_detector(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Initialize HandLandmarker
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_tracking_confidence,
        )
        self.detector = HandLandmarker.create_from_options(options)

        # Initialize PoseLandmarker
        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.pose_model_path),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_tracking_confidence,
        )
        self.pose_detector = PoseLandmarker.create_from_options(pose_options)
        self._timestamp_ms = 0

    def extract(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract normalized landmark vector from a BGR frame (legacy single hand).

        Returns a (63,) float32 array, or None if no hand is detected.
        """
        raw, handedness = self.extract_raw_with_handedness(frame)
        if raw is None:
            return None
        if self.rotation_invariant:
            return normalize_landmarks_invariant(raw, handedness or "Right")
        return normalize_landmarks(raw).astype(np.float32)

    def extract_holistic(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract a 147-dimensional normalized holistic landmark vector.

        All coordinates are torso-relative and scaled by shoulder distance.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33
        
        # Detect hands and pose in the frame
        hand_result = self.detector.detect_for_video(mp_image, self._timestamp_ms)
        pose_result = self.pose_detector.detect_for_video(mp_image, self._timestamp_ms)

        has_hands = bool(hand_result.hand_landmarks)
        has_pose = bool(pose_result.pose_landmarks)

        if not has_hands and not has_pose:
            return None

        # Map hands by handedness
        left_hand = None
        right_hand = None
        if hand_result.hand_landmarks and hand_result.handedness:
            for i, hand_lms in enumerate(hand_result.hand_landmarks):
                handedness_label = hand_result.handedness[i][0].category_name
                if handedness_label.lower().startswith("left"):
                    left_hand = hand_lms
                else:
                    right_hand = hand_lms

        pose_landmarks = None
        if pose_result.pose_landmarks:
            pose_landmarks = pose_result.pose_landmarks[0]

        return normalize_holistic(left_hand, right_hand, pose_landmarks)

    def extract_raw(self, frame: np.ndarray):
        """Extract raw MediaPipe landmark list from a BGR frame (legacy first hand)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33
        result = self.detector.detect_for_video(mp_image, self._timestamp_ms)
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]

    def extract_raw_with_handedness(self, frame: np.ndarray):
        """Extract raw MediaPipe landmark list and handedness label (legacy first hand)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33
        result = self.detector.detect_for_video(mp_image, self._timestamp_ms)
        if not result.hand_landmarks:
            return None, None
        handedness = None
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name
        return result.hand_landmarks[0], handedness

    def close(self):
        self.detector.close()
        self.pose_detector.close()
