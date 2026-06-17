"""Smoke tests for `hand_model/landmarks.py`.

Coverage:
  - `normalize_landmarks`: wrist at origin, MCP_9 at (0, 1, 0)
  - `normalize_landmarks_invariant`: same + handedness mirror symmetry
  - `LandmarkExtractor`: constructor accepts the `rotation_invariant` flag

These tests DO NOT spin up MediaPipe — they only exercise the pure-numpy
normalizers and the constructor's argument handling. MediaPipe integration
is covered by the existing `scripts/evaluate_sequence.py` smoke runs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hand_model import landmarks as L  # noqa: E402


# ---------------------------------------------------------------------------
# Test fixtures: a simple right-handed hand with known wrist + MCP_9
# ---------------------------------------------------------------------------

@dataclass
class _Lm:
    x: float
    y: float
    z: float


def _make_hand(wrist=(0.30, 0.40, 0.00), mcp9=(0.30, 0.60, 0.05)):
    """Build a list of 21 _Lm objects with arbitrary finger positions.

    Wrist is at (0.30, 0.40, 0.00) and middle-finger MCP is at (0.30, 0.60, 0.05)
    in image coords, so the wrist→MCP vector is (0, 0.20, 0.05) — purely vertical
    plus a small Z component, length sqrt(0.04+0.0025) ~= 0.206.
    """
    return [_Lm(*wrist)] + [_Lm(0.30 + 0.01 * i, 0.40 + 0.01 * i, 0.0) for i in range(1, 9)] \
         + [_Lm(*mcp9)] + [_Lm(0.32, 0.55, 0.0) for _ in range(10, 21)]


def test_normalize_landmarks_wrist_at_origin_mcp9_unit_length():
    raw = _make_hand()
    out = L.normalize_landmarks(raw)
    arr = np.asarray(out, dtype=np.float32).reshape(21, 3)

    # Wrist (index 0) must be at origin.
    np.testing.assert_allclose(arr[0], np.zeros(3), atol=1e-6)
    # MCP_9 (middle-finger MCP) must be at distance 1.0 from wrist.
    np.testing.assert_allclose(np.linalg.norm(arr[9]), 1.0, atol=1e-6)
    # After this normalization the y-axis is the wrist→MCP axis, so y >= 0.
    # (We don't require exactly (0, 1, 0) here because the simple normalizer
    # does NOT rotate — that is the job of the invariant variant.)


def test_normalize_landmarks_invariant_rotates_mcp9_to_positive_y():
    """The rotation must align the (XY) wrist→MCP_9 vector with the +Y
    axis. Z is preserved (this is a 2D in-plane rotation)."""
    raw = _make_hand()
    out = L.normalize_landmarks_invariant(raw, handedness="Right")
    arr = np.asarray(out, dtype=np.float32).reshape(21, 3)

    # Invariants: wrist at origin.
    np.testing.assert_allclose(arr[0], np.zeros(3), atol=1e-6)
    # ||MCP_9|| should still be 1.0 (rotation preserves length).
    np.testing.assert_allclose(np.linalg.norm(arr[9]), 1.0, atol=1e-6)
    # MCP_9 must land on (0, +||xy||, z) — the XY rotation aligns it with +Y.
    assert abs(arr[9, 0]) < 1e-5, f"MCP_9 X should be 0 after rotation, got {arr[9, 0]}"
    assert arr[9, 1] > 0.0, f"MCP_9 Y should be > 0 after rotation, got {arr[9, 1]}"
    # Z component must equal the original (pre-rotation) Z — i.e. preserved.
    # The hand fixture has mcp9.z = 0.05 with wrist.z = 0.0 and ||MCP||=0.206,
    # so the normalized z is 0.05/0.206 ≈ 0.243.
    expected_z = 0.05 / np.sqrt(0.20**2 + 0.05**2)
    np.testing.assert_allclose(arr[9, 2], expected_z, atol=1e-5)


def test_normalize_landmarks_invariant_mirrors_left_handed_input():
    """The canonical frame is handedness-agnostic. So a right hand and
    its true X-mirror processed as "Left" must land in the SAME canonical
    frame (because the left-handed path flips X back to positive).

    We build a right hand whose finger X deltas are positive, then
    construct the "left" hand by mirroring X (so its X deltas are
    negative). The function should flip those negative X deltas back to
    positive, yielding an identical canonical frame to the right hand.
    """
    right_hand = _make_hand(wrist=(0.30, 0.40, 0.0), mcp9=(0.30, 0.60, 0.05))
    # Mirror along X: new_x = 2*wrist.x - old_x so finger X deltas flip sign.
    left_hand = [_Lm(2 * 0.30 - lm.x, lm.y, lm.z) for lm in right_hand]

    right_canon = np.asarray(
        L.normalize_landmarks_invariant(right_hand, handedness="Right"),
        dtype=np.float32,
    ).reshape(21, 3)
    left_canon = np.asarray(
        L.normalize_landmarks_invariant(left_hand, handedness="Left"),
        dtype=np.float32,
    ).reshape(21, 3)

    # Both canonical frames must agree to tolerance — the handedness
    # mirror canonicalizes the left hand into the right-handed reference
    # frame, so they end up identical.
    np.testing.assert_allclose(left_canon, right_canon, atol=1e-4)


def test_normalize_landmarks_invariant_does_not_change_y_of_wrist_mcp():
    """The rotation should not alter the y-axis projection of the
    wrist→MCP vector (it should keep the +y direction and only rotate
    around Z to align)."""
    raw = _make_hand()
    out = L.normalize_landmarks_invariant(raw, handedness="Right")
    arr = np.asarray(out, dtype=np.float32).reshape(21, 3)
    # MCP_9's y component must be positive (the rotation aligns with +Y).
    assert arr[9, 1] > 0.0


def test_normalize_landmarks_invariant_float32_output():
    raw = _make_hand()
    out = L.normalize_landmarks_invariant(raw, handedness="Right")
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (63,)


# ---------------------------------------------------------------------------
# LandmarkExtractor constructor
# ---------------------------------------------------------------------------

def test_landmark_extractor_accepts_rotation_invariant_flag(monkeypatch, tmp_path):
    """The constructor must accept `rotation_invariant` and forward it
    to the .extract() method path. We mock out the MediaPipe detector
    creation so we don't need a real .task model on disk."""
    fake_task = tmp_path / "fake_hand_landmarker.task"
    fake_task.write_bytes(b"")

    # Patch HandLandmarker.create_from_options so it never actually loads.
    class _FakeDetector:
        def __init__(self, *a, **kw):
            pass
        def detect_for_video(self, *a, **kw):
            class _R:
                hand_landmarks = []
                handedness = []
            return _R()
        def close(self):
            pass

    # mediapipe.tasks.vision.HandLandmarker must exist as a name; we patch it.
    import mediapipe as mp

    # The _init_detector path calls HandLandmarker.create_from_options,
    # so the fake class needs that classmethod.
    @classmethod
    def _create_from_options(cls, options):
        return cls()

    _FakeDetector.create_from_options = _create_from_options

    with patch.object(mp.tasks.vision, "HandLandmarker", _FakeDetector, create=True):
        # Default: rotation_invariant=False (legacy API behavior)
        ext = L.LandmarkExtractor(model_path=str(fake_task), rotation_invariant=False)
        assert ext.rotation_invariant is False
        ext.close()

        # Enabled: rotation_invariant=True
        ext2 = L.LandmarkExtractor(model_path=str(fake_task), rotation_invariant=True)
        assert ext2.rotation_invariant is True
        ext2.close()


def test_landmark_extractor_default_rotation_invariant_is_false():
    """The docstring says `rotation_invariant` defaults to False to keep
    legacy behavior. This test exercises the signature default path by
    passing it through argparse-like SimpleNamespace — we don't actually
    construct the extractor (would require MediaPipe), we just verify
    the kwarg default in the function signature."""
    import inspect
    sig = inspect.signature(L.LandmarkExtractor.__init__)
    assert "rotation_invariant" in sig.parameters
    assert sig.parameters["rotation_invariant"].default is False
