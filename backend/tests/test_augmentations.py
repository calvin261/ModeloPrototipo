"""Smoke tests for `hand_model.augmentations.py`.

Each augmentation function is checked for:
  - input shape preserved: (T, 63) -> (T, 63)
  - dtype is float32
  - no NaN / no Inf in the output
  - the composed `augment_sequence` pipeline is stochastic
    (different calls with the same input produce different outputs)
  - `cutmix_temporal` correctly halves and concatenates two same-class seqs
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make `hand_model` importable when running pytest from anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hand_model import augmentations as A  # noqa: E402

T, D = 40, 63
RNG_SEED = 1234


def _make_seq(t: int = T, d: int = D, seed: int = RNG_SEED) -> np.ndarray:
    """Build a (T, 63) sequence with realistic non-zero values."""
    rng = np.random.default_rng(seed)
    seq = rng.normal(0.0, 0.3, size=(t, d)).astype(np.float32)
    return seq


def _assert_finite_float32(out: np.ndarray, expected_shape: tuple[int, int]) -> None:
    assert out.shape == expected_shape, f"shape {out.shape} != {expected_shape}"
    assert out.dtype == np.float32, f"dtype {out.dtype} != float32"
    assert np.isfinite(out).all(), "output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Per-augmentation smoke tests
# ---------------------------------------------------------------------------

def test_gaussian_noise_preserves_shape_and_dtype():
    seq = _make_seq()
    out = A.gaussian_noise(seq, np.random.default_rng(0))
    _assert_finite_float32(out, seq.shape)


def test_joint_jitter_preserves_shape_and_dtype():
    seq = _make_seq()
    out = A.joint_jitter(seq, np.random.default_rng(0))
    _assert_finite_float32(out, seq.shape)


def test_landmark_dropout_preserves_shape_and_dtype():
    seq = _make_seq()
    out = A.landmark_dropout(seq, np.random.default_rng(0))
    _assert_finite_float32(out, seq.shape)


def test_time_warp_preserves_shape_and_dtype():
    seq = _make_seq()
    out = A.time_warp(seq, np.random.default_rng(0))
    _assert_finite_float32(out, seq.shape)


def test_frame_dropout_preserves_shape_and_dtype():
    seq = _make_seq()
    out = A.frame_dropout(seq, np.random.default_rng(0))
    _assert_finite_float32(out, seq.shape)


def test_speed_change_preserves_shape_and_dtype():
    seq = _make_seq()
    out = A.speed_change(seq, np.random.default_rng(0))
    _assert_finite_float32(out, seq.shape)


def test_cutmix_temporal_length_and_dtype():
    a = _make_seq(T, D, seed=1)
    b = _make_seq(T, D, seed=2)
    out = A.cutmix_temporal(a, b, np.random.default_rng(0))
    # Length should be Ta//2 + Tb//2, still (L, 63) float32
    expected_len = a.shape[0] // 2 + b.shape[0] // 2
    assert out.shape == (expected_len, D)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_cutmix_temporal_with_max_seq_len_truncation():
    a = _make_seq(30, D, seed=1)
    b = _make_seq(30, D, seed=2)
    out = A.cutmix_temporal(a, b, np.random.default_rng(0), max_seq_len=20)
    assert out.shape == (20, D)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_cutmix_temporal_with_max_seq_len_padding():
    a = _make_seq(8, D, seed=1)
    b = _make_seq(8, D, seed=2)
    out = A.cutmix_temporal(a, b, np.random.default_rng(0), max_seq_len=20)
    assert out.shape == (20, D)
    assert out.dtype == np.float32
    # Padding rows should be exactly zero
    tail = out[8:, :]
    assert np.all(tail == 0.0)


def test_rotation_2d_preserves_shape_and_dtype():
    seq = _make_seq()
    out = A.rotation_2d(seq, np.random.default_rng(0))
    _assert_finite_float32(out, seq.shape)


def test_rotation_3d_preserves_shape_and_dtype():
    seq = _make_seq()
    out = A.rotation_3d(seq, np.random.default_rng(0))
    _assert_finite_float32(out, seq.shape)


def test_scale_jitter_preserves_shape_and_dtype():
    seq = _make_seq()
    out = A.scale_jitter(seq, np.random.default_rng(0))
    _assert_finite_float32(out, seq.shape)


def test_translation_xy_preserves_shape_and_dtype():
    seq = _make_seq()
    out = A.translation_xy(seq, np.random.default_rng(0))
    _assert_finite_float32(out, seq.shape)


def test_mirror_x_preserves_shape_and_dtype():
    seq = _make_seq()
    out = A.mirror_x(seq)
    _assert_finite_float32(out, seq.shape)
    # X-axis (every 3rd column starting at 0) must be sign-flipped
    assert np.allclose(out[:, 0::3], -seq[:, 0::3])


# ---------------------------------------------------------------------------
# Composed pipeline stochasticity
# ---------------------------------------------------------------------------

def test_augment_sequence_stochastic():
    """Two calls with the same input but different RNGs must produce
    different outputs (the pipeline is not constant)."""
    seq = _make_seq(seed=42)
    a = A.augment_sequence(seq, np.random.default_rng(0))
    b = A.augment_sequence(seq, np.random.default_rng(1))
    _assert_finite_float32(a, seq.shape)
    _assert_finite_float32(b, seq.shape)
    assert not np.allclose(a, b), "augment_sequence is deterministic across RNGs"


def test_augment_sequence_reproducible_with_same_seed():
    """Same seed must give the same output (the pipeline is deterministic
    given an integer seed — important for train-run reproducibility)."""
    seq = _make_seq(seed=42)
    a = A.augment_sequence(seq, np.random.default_rng(123))
    b = A.augment_sequence(seq, np.random.default_rng(123))
    np.testing.assert_array_equal(a, b)


def test_augment_sequence_with_cutmix_length_correct():
    """`augment_sequence_with_cutmix` should produce (max_seq_len, 63) when
    max_seq_len is provided, since cutmix truncates/pads to it BEFORE the
    default pipeline runs."""
    a = _make_seq(30, D, seed=1)
    b = _make_seq(30, D, seed=2)
    out = A.augment_sequence_with_cutmix(
        a, b, np.random.default_rng(0), max_seq_len=40
    )
    assert out.shape == (40, D)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_default_techniques_covers_all_augs():
    """Sanity check: the DEFAULT_TECHNIQUES tuple contains every per-sample
    aug except cutmix (cutmix is a pairwise op handled separately)."""
    expected = {
        A.gaussian_noise, A.joint_jitter, A.landmark_dropout,
        A.time_warp, A.frame_dropout, A.speed_change,
        A.rotation_2d, A.rotation_3d, A.scale_jitter,
        A.translation_xy, A.mirror_x,
    }
    assert set(A.DEFAULT_TECHNIQUES) == expected
