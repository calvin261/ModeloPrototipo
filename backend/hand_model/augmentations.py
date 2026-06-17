"""Augmentation pipeline for hand-landmark sequences.

Each function takes a (T, 63) numpy array (already wrist-relative + scale
normalized) and returns an augmented (T, 63) array. The augmentations are
designed to cover the realistic variance a new user introduces vs the
training distribution:

    1. gaussian_noise          – per-landmark Gaussian noise
    2. joint_jitter            – per-joint independent noise (each finger
                                 moves slightly independently)
    3. landmark_dropout        – randomly drop a fraction of landmarks to
                                 simulate MediaPipe tracking failures
    4. time_warp               – resample time to a different length
    5. frame_dropout           – drop random frames and re-stitch
    6. speed_change            – play back at 0.7-1.3x speed
    7. cutmix_temporal         – concatenate two halves of two different
                                 sequences of the SAME class (signer-mixup
                                 along the time axis)
    8. rotation_2d             – rotate all landmarks around wrist
    9. rotation_3d             – small roll/pitch around the wrist
   10. scale_jitter            – multiply all coordinates by a factor
   11. translation_xy          – shift the whole sequence in XY
   12. mirror_x                – flip X coordinates (simulate other hand)

All functions are deterministic given an integer seed, so the same input
will produce the same augmented output across calls (important for
debugging and reproducibility of the train run).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Individual augmentations
# ---------------------------------------------------------------------------

def gaussian_noise(seq: np.ndarray, rng: np.random.Generator, sigma: float = 0.02) -> np.ndarray:
    """Per-landmark Gaussian noise (all landmarks, all frames)."""
    return (seq + rng.normal(0, sigma, size=seq.shape)).astype(np.float32)


def joint_jitter(seq: np.ndarray, rng: np.random.Generator, sigma: float = 0.015) -> np.ndarray:
    """Per-joint Gaussian noise: each of the 21 landmarks gets its own
    per-frame random offset. This is more realistic than global noise
    because in real life, the wrist, fingers, and thumb move semi-
    independently."""
    T = seq.shape[0]
    coords = seq.reshape(T, 21, 3)
    noise = rng.normal(0, sigma, size=coords.shape)
    coords = coords + noise
    return coords.reshape(T, 63).astype(np.float32)


def landmark_dropout(
    seq: np.ndarray,
    rng: np.random.Generator,
    landmark_drop_prob: float = 0.10,
) -> np.ndarray:
    """Set a fraction of landmarks to the wrist (0,0,0) to simulate
    MediaPipe tracking failures for that joint."""
    T = seq.shape[0]
    coords = seq.reshape(T, 21, 3).copy()
    # mask shape (T, 21) — 1 means keep, 0 means dropped (set to 0)
    keep_mask = rng.random(size=(T, 21)) > landmark_drop_prob
    coords = coords * keep_mask[:, :, None]
    return coords.reshape(T, 63).astype(np.float32)


def time_warp(seq: np.ndarray, rng: np.random.Generator, factor_range: tuple = (0.7, 1.4)) -> np.ndarray:
    """Resample the time axis by a random factor in `factor_range`.

    Sequence shorter than 4 frames: return as-is (can't warp).
    Result is interpolated back to the original length using linear
    interpolation, so the model always sees the same T."""
    T, D = seq.shape
    if T < 4:
        return seq.copy()
    factor = rng.uniform(*factor_range)
    new_len = max(2, int(round(T * factor)))
    if new_len == T:
        return seq.copy()
    # 1. Resample from T -> new_len (stretch or compress)
    idx_in_orig = np.linspace(0, T - 1, new_len)
    out = np.zeros_like(seq)
    for d in range(D):
        warped = np.interp(idx_in_orig, np.arange(T), seq[:, d])
        # 2. Resample back from new_len -> T
        out[:, d] = np.interp(np.linspace(0, new_len - 1, T), np.arange(new_len), warped)
    return out.astype(np.float32)


def frame_dropout(seq: np.ndarray, rng: np.random.Generator, drop_prob: float = 0.10) -> np.ndarray:
    """Drop a random fraction of frames and resample back to T.

    Models robustness to MediaPipe frame failures."""
    T, D = seq.shape
    if T < 4:
        return seq.copy()
    keep = rng.random(size=T) > drop_prob
    if keep.sum() < 2:
        return seq.copy()
    kept_seq = seq[keep]
    idx_kept = np.arange(T)[keep].astype(np.float32)
    out = np.zeros_like(seq)
    query = np.arange(T, dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(query, idx_kept, kept_seq[:, d])
    return out.astype(np.float32)


def speed_change(seq: np.ndarray, rng: np.random.Generator, factor_range: tuple = (0.7, 1.3)) -> np.ndarray:
    """Same as time_warp but with a tighter range — only +/- 30% speed."""
    return time_warp(seq, rng, factor_range)


def cutmix_temporal(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    rng: np.random.Generator,
    max_seq_len: int | None = None,
) -> np.ndarray:
    """Concatenate the first half of seq_a with the second half of seq_b.

    Both sequences must be the same class (callers must enforce this).
    This teaches the model that the temporal mid-section of a sign is
    not always tightly coupled to its start/end.

    Result is padded/truncated to max_seq_len if provided.
    """
    Ta = seq_a.shape[0]
    Tb = seq_b.shape[0]
    mid_a = Ta // 2
    mid_b = Tb // 2
    out = np.concatenate([seq_a[:mid_a], seq_b[mid_b:]], axis=0)
    if max_seq_len is not None:
        if out.shape[0] > max_seq_len:
            out = out[:max_seq_len]
        elif out.shape[0] < max_seq_len:
            pad = np.zeros((max_seq_len - out.shape[0], out.shape[1]), dtype=out.dtype)
            out = np.concatenate([out, pad], axis=0)
    return out.astype(np.float32)


def rotation_2d(seq: np.ndarray, rng: np.random.Generator, max_deg: float = 35.0) -> np.ndarray:
    """Rotate every frame around the Z axis (image plane) by a random
    angle in [-max_deg, +max_deg]."""
    T = seq.shape[0]
    coords = seq.reshape(T, 21, 3).copy()
    angle = rng.uniform(-max_deg, max_deg) * np.pi / 180.0
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    xy = coords[:, :, :2]
    coords[:, :, 0] = xy[:, :, 0] * cos_a - xy[:, :, 1] * sin_a
    coords[:, :, 1] = xy[:, :, 0] * sin_a + xy[:, :, 1] * cos_a
    return coords.reshape(T, 63).astype(np.float32)


def rotation_3d(seq: np.ndarray, rng: np.random.Generator, max_deg: float = 25.0) -> np.ndarray:
    """Apply small roll and pitch rotations to simulate tilted hands.

    Implemented as separate rotations around X and Y axes applied to the
    relative coordinates (wrist at origin).
    """
    T = seq.shape[0]
    coords = seq.reshape(T, 21, 3).copy()
    roll = rng.uniform(-max_deg, max_deg) * np.pi / 180.0
    pitch = rng.uniform(-max_deg, max_deg) * np.pi / 180.0
    # Roll: rotate around X axis (affects Y, Z)
    cr, sr = np.cos(roll), np.sin(roll)
    y_old = coords[:, :, 1].copy()
    z_old = coords[:, :, 2].copy()
    coords[:, :, 1] = y_old * cr - z_old * sr
    coords[:, :, 2] = y_old * sr + z_old * cr
    # Pitch: rotate around Y axis (affects X, Z)
    cp, sp = np.cos(pitch), np.sin(pitch)
    x_old = coords[:, :, 0].copy()
    z_old = coords[:, :, 2].copy()
    coords[:, :, 0] = x_old * cp + z_old * sp
    coords[:, :, 2] = -x_old * sp + z_old * cp
    return coords.reshape(T, 63).astype(np.float32)


def scale_jitter(seq: np.ndarray, rng: np.random.Generator, factor_range: tuple = (0.7, 1.4)) -> np.ndarray:
    """Multiply all coordinates by a uniform factor in `factor_range`."""
    factor = rng.uniform(*factor_range)
    return (seq * factor).astype(np.float32)


def translation_xy(seq: np.ndarray, rng: np.random.Generator, max_shift: float = 0.15) -> np.ndarray:
    """Shift the whole sequence in XY by the same offset (per sample)."""
    T = seq.shape[0]
    coords = seq.reshape(T, 21, 3).copy()
    shift = rng.uniform(-max_shift, max_shift, size=2)
    coords[:, :, 0] += shift[0]
    coords[:, :, 1] += shift[1]
    return coords.reshape(T, 63).astype(np.float32)


def mirror_x(seq: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Flip X coordinates to simulate the opposite hand.

    `rng` is accepted but unused (kept for API symmetry with other augs
    and for future stochastic variants)."""
    coords = seq.copy()
    coords[:, 0::3] = -coords[:, 0::3]
    return coords.astype(np.float32)


# ---------------------------------------------------------------------------
# Composed pipelines
# ---------------------------------------------------------------------------

# Ordered list of "default" per-sample augmentations. Each is independent
# of the others, so they can be reordered without changing the semantics
# of the union (modulo the order in which they stack on the data).
DEFAULT_TECHNIQUES: tuple[Callable[[np.ndarray, np.random.Generator], np.ndarray], ...] = (
    gaussian_noise,
    joint_jitter,
    landmark_dropout,
    time_warp,
    frame_dropout,
    speed_change,
    rotation_2d,
    rotation_3d,
    scale_jitter,
    translation_xy,
    mirror_x,
)


def augment_sequence(
    seq: np.ndarray,
    rng: np.random.Generator,
    techniques: Sequence[Callable] = DEFAULT_TECHNIQUES,
) -> np.ndarray:
    """Apply every technique in `techniques` once, in order.

    Args:
        seq: (T, 63) float array.
        rng: numpy Generator.
        techniques: ordered tuple of callables (seq, rng) -> (T, 63).

    Returns:
        (T, 63) float32 augmented array.
    """
    out = seq.astype(np.float32).copy()
    for tech in techniques:
        out = tech(out, rng)
    return out


def augment_sequence_with_cutmix(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    rng: np.random.Generator,
    max_seq_len: int | None = None,
) -> np.ndarray:
    """Apply cutmix + the default per-sample pipeline.

    Both sequences must be from the same class. The cutmix step is
    applied first so the rest of the pipeline still operates on a
    (T, 63) sequence.
    """
    out = cutmix_temporal(seq_a, seq_b, rng, max_seq_len=max_seq_len)
    return augment_sequence(out, rng)
