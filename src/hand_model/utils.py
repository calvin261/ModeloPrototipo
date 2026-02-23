from pathlib import Path

import numpy as np


def ensure_dir(path):
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def augment_landmarks(vector: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """Apply random augmentations to a 63-float normalized landmark vector.

    Augmentations applied: 2D rotation, scaling, translation, and Gaussian noise.
    The input vector should already be normalized (wrist at origin).
    """
    if rng is None:
        rng = np.random.default_rng()
    coords = vector.reshape(21, 3).copy()

    # Random 2D rotation around z-axis (+/- 15 degrees)
    angle = rng.uniform(-15, 15) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    coords[:, :2] = coords[:, :2] @ rot.T

    # Random uniform scaling (0.85 to 1.15)
    scale = rng.uniform(0.85, 1.15)
    coords *= scale

    # Random translation (small shift)
    shift = rng.uniform(-0.05, 0.05, size=3)
    coords += shift

    # Gaussian noise
    noise = rng.normal(0, 0.008, size=coords.shape)
    coords += noise

    return coords.flatten().astype(np.float32)


def mirror_landmarks(vector: np.ndarray) -> np.ndarray:
    """Mirror landmarks by flipping x-coordinates (simulate opposite hand)."""
    coords = vector.reshape(21, 3).copy()
    coords[:, 0] = -coords[:, 0]
    return coords.flatten().astype(np.float32)
