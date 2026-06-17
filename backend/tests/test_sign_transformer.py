"""Smoke tests for `hand_model/spoter_model.py::SignTransformer`.

Covers:
  - Default-config construction produces a model with the expected dims
  - Forward pass on (B, T, 21, 3) raw input gives (B, C) output
  - Forward pass on (B, T, 63) flat input also works
  - CLS-pooled vs mean-pooled give different outputs for the same input
  - ONNX export + onnxruntime inference matches PyTorch within 1e-4
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hand_model.spoter_model import SignTransformer  # noqa: E402

# Skip the ONNX round-trip test if onnxruntime is not available in the env.
ORT = pytest.importorskip("onnxruntime")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(num_classes: int = 12, max_seq_len: int = 32, use_cls: bool = True) -> SignTransformer:
    return SignTransformer(
        n_joints=21,
        n_channels=3,
        num_classes=num_classes,
        embed_dim=64,            # small to keep the test fast
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        max_seq_len=max_seq_len,
        use_cls=use_cls,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_default_construction_yields_subclass_of_nn_module():
    model = _build_model()
    assert isinstance(model, nn.Module)
    assert model.n_joints == 21
    assert model.n_channels == 3
    assert model.num_classes == 12
    assert model.use_cls is True


def test_default_construction_has_trainable_parameters():
    model = _build_model()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
    # Sanity bound: with embed=64, layers=2, ff=128, this should be well
    # under the full 565K production model.
    assert n_params < 200_000


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,T", [(1, 16), (3, 24)])
def test_forward_4d_input_returns_bc_output(B: int, T: int):
    model = _build_model(max_seq_len=T)
    model.eval()
    x = torch.randn(B, T, 21, 3, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (B, model.num_classes)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("B,T", [(1, 16), (3, 24)])
def test_forward_3d_flat_input_returns_bc_output(B: int, T: int):
    model = _build_model(max_seq_len=T)
    model.eval()
    x = torch.randn(B, T, 63, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (B, model.num_classes)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


def test_3d_and_4d_inputs_produce_same_output_for_same_data():
    """(B, T, 21, 3) reshaped back to (B, T, 63) must give the same result
    as the (B, T, 63) path through the linear projection."""
    model = _build_model(max_seq_len=20)
    model.eval()
    x4 = torch.randn(2, 20, 21, 3, dtype=torch.float32)
    x3 = x4.reshape(2, 20, 63).contiguous()
    with torch.no_grad():
        y4 = model(x4)
        y3 = model(x3)
    torch.testing.assert_close(y3, y4, rtol=0, atol=0)


def test_cls_pooled_vs_mean_pooled_produce_different_outputs():
    """Two SignTransformer instances with the same architecture but
    different pooling strategies (CLS vs mean) must give different
    outputs on the same input — otherwise `use_cls=False` is a no-op
    and the architectural choice is silent.

    Both models are built with the same seed; the second one's pos_embed
    is shorter (no CLS slot) and it has no cls_token. We copy the
    parameters that have matching shapes from the first model into the
    second so the rest of the architecture is identical.
    """
    torch.manual_seed(0)
    cls_model = SignTransformer(
        n_joints=21, n_channels=3, num_classes=12,
        embed_dim=64, nhead=4, num_layers=2, dim_feedforward=128,
        dropout=0.0, max_seq_len=24, use_cls=True,
    )
    torch.manual_seed(0)
    mean_model = SignTransformer(
        n_joints=21, n_channels=3, num_classes=12,
        embed_dim=64, nhead=4, num_layers=2, dim_feedforward=128,
        dropout=0.0, max_seq_len=24, use_cls=False,
    )

    # Copy all parameters that have matching names and shapes.
    cls_state = cls_model.state_dict()
    mean_state = mean_model.state_dict()
    transferred = 0
    for k, v in mean_state.items():
        if k in cls_state and cls_state[k].shape == v.shape:
            mean_state[k] = cls_state[k].clone()
            transferred += 1
    mean_model.load_state_dict(mean_state)
    assert transferred >= 4, f"Expected to copy several shared parameters, copied {transferred}"

    cls_model.eval()
    mean_model.eval()

    x = torch.randn(2, 24, 21, 3, dtype=torch.float32)
    with torch.no_grad():
        y_cls = cls_model(x)
        y_mean = mean_model(x)
    assert y_cls.shape == y_mean.shape == (2, 12)
    # Outputs must differ — otherwise pooling is not actually changing the model.
    assert not torch.allclose(y_cls, y_mean, atol=1e-5), (
        "CLS-pooled and mean-pooled outputs are identical; "
        "the use_cls flag is not affecting the forward pass."
    )


# ---------------------------------------------------------------------------
# ONNX export + onnxruntime inference parity
# ---------------------------------------------------------------------------

def test_onnx_export_matches_pytorch_within_1e_minus_4(tmp_path: Path):
    """Export to ONNX, run with onnxruntime, and verify that the output
    matches PyTorch within 1e-4 absolute tolerance (FP32 numerics)."""
    model = _build_model(max_seq_len=20)
    model.eval()

    B, T = 2, 20
    x = torch.randn(B, T, 21, 3, dtype=torch.float32)

    onnx_path = tmp_path / "transformer_smoke.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model, (x,), str(onnx_path),
            input_names=["sequence"], output_names=["logits"],
            dynamic_axes={"sequence": {0: "batch", 1: "time"}, "logits": {0: "batch"}},
            opset_version=18,
        )
    assert onnx_path.exists() and onnx_path.stat().st_size > 0

    sess = ORT.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_out = model(x).cpu().numpy()
    ort_out = sess.run(None, {"sequence": x.cpu().numpy()})[0]

    assert pt_out.shape == ort_out.shape == (B, model.num_classes)
    max_abs_diff = float(np.abs(pt_out - ort_out).max())
    assert max_abs_diff < 1e-4, (
        f"ONNX vs PyTorch max abs diff = {max_abs_diff:.6e}, expected < 1e-4"
    )
