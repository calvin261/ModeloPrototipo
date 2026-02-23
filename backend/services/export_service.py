"""Export PyTorch models to ONNX (and optionally TFLite) formats."""

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from hand_model.classifier import SignClassifierMLP
from hand_model.config import ClassifierConfig, SequenceConfig
from hand_model.sequence_model import SignSequenceModel


def _consolidate_onnx(output_path: str) -> None:
    """Re-save an ONNX file as single self-contained file (no external data).

    torch.onnx.export may create a separate .onnx.data file for large models.
    This function loads the exported model (with external data) and saves it
    back as a single file so Android ONNX Runtime can load it from one download.
    """
    import onnx

    model_proto = onnx.load(output_path)
    # Save back without external data (embeds tensors inside the protobuf)
    onnx.save_model(
        model_proto,
        output_path,
        save_as_external_data=False,
    )
    # Remove the companion .onnx.data file if torch created one
    data_file = Path(str(output_path) + ".data")
    if data_file.exists():
        data_file.unlink()


def export_classifier_to_onnx(
    checkpoint_path: str,
    output_path: str,
) -> str:
    """Export MLP classifier from .pth to .onnx."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]

    # Find output layer dynamically (last 2-D weight tensor = Linear output)
    linear_keys = [k for k in state_dict if k.endswith(".weight") and state_dict[k].dim() == 2]
    actual_num_classes = state_dict[linear_keys[-1]].shape[0]

    model = SignClassifierMLP(
        input_size=cfg.input_size,
        hidden_sizes=cfg.hidden_sizes,
        num_classes=actual_num_classes,
        dropout=0.0,
    )
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, cfg.input_size)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["landmarks"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes={"landmarks": {0: "batch"}},
        dynamo=False,
    )
    _consolidate_onnx(output_path)
    return output_path


def export_sequence_to_onnx(
    checkpoint_path: str,
    output_path: str,
) -> str:
    """Export GRU sequence model from .pth to .onnx."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]

    # Use actual output layer size from weights (training may differ from default config)
    actual_num_classes = state_dict["classifier.3.weight"].shape[0]

    model = SignSequenceModel(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        num_classes=actual_num_classes,
        dropout=0.0,
        bidirectional=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, cfg.max_seq_len, cfg.input_size)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["sequence"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes={"sequence": {0: "batch"}},
        dynamo=False,
    )
    _consolidate_onnx(output_path)
    return output_path


def onnx_to_tflite(
    onnx_path: str,
    tflite_path: str,
    quantize: bool = False,
) -> str:
    """Convert ONNX model to TFLite format.

    Requires onnx, onnx-tf, and tensorflow to be installed.
    """
    import onnx
    import tensorflow as tf
    from onnx_tf.backend import prepare

    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    # Save as TF SavedModel temporarily
    tf_saved_model_dir = onnx_path.replace(".onnx", "_tf_saved_model")
    tf_rep.export_graph(tf_saved_model_dir)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    Path(tflite_path).parent.mkdir(parents=True, exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    # Cleanup temp directory
    shutil.rmtree(tf_saved_model_dir, ignore_errors=True)

    return tflite_path


def export_classifier_pipeline(
    checkpoint_path: str,
    exports_dir: str,
    version: str = None,
    quantize: bool = False,
) -> dict:
    """Full pipeline: classifier .pth -> .onnx -> .tflite with metadata."""
    if version is None:
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    exports = Path(exports_dir)
    exports.mkdir(parents=True, exist_ok=True)

    onnx_path = str(exports / f"sign_mlp_{version}.onnx")
    tflite_path = str(exports / f"sign_mlp_{version}.tflite")

    # Export steps
    export_classifier_to_onnx(checkpoint_path, onnx_path)
    onnx_to_tflite(onnx_path, tflite_path, quantize=quantize)

    # Load metadata from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    label_map = checkpoint["label_map"]
    cfg = checkpoint["config"]

    # Save label map alongside the model
    labels_path = str(exports / f"classifier_labels_{version}.json")
    with open(labels_path, "w") as f:
        json.dump(label_map, f, indent=2)

    tflite_size = Path(tflite_path).stat().st_size / 1024

    return {
        "version": version,
        "tflite_path": tflite_path,
        "onnx_path": onnx_path,
        "labels_path": labels_path,
        "size_kb": round(tflite_size, 2),
        "num_classes": cfg.num_classes,
        "created_at": datetime.now().isoformat(),
    }


def export_sequence_pipeline(
    checkpoint_path: str,
    exports_dir: str,
    version: str = None,
    quantize: bool = False,
) -> dict:
    """Full pipeline: sequence model .pth -> .onnx -> .tflite with metadata."""
    if version is None:
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    exports = Path(exports_dir)
    exports.mkdir(parents=True, exist_ok=True)

    onnx_path = str(exports / f"sign_gru_{version}.onnx")
    tflite_path = str(exports / f"sign_gru_{version}.tflite")

    export_sequence_to_onnx(checkpoint_path, onnx_path)
    onnx_to_tflite(onnx_path, tflite_path, quantize=quantize)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocabulary = checkpoint["vocabulary"]
    cfg = checkpoint["config"]

    labels_path = str(exports / f"sequence_labels_{version}.json")
    with open(labels_path, "w") as f:
        json.dump(vocabulary, f, indent=2)

    tflite_size = Path(tflite_path).stat().st_size / 1024

    return {
        "version": version,
        "tflite_path": tflite_path,
        "onnx_path": onnx_path,
        "labels_path": labels_path,
        "size_kb": round(tflite_size, 2),
        "num_classes": cfg.num_classes,
        "created_at": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# ONNX-only pipelines (no TFLite / no tensorflow required)
# ---------------------------------------------------------------------------

def export_classifier_onnx_only(
    checkpoint_path: str,
    exports_dir: str,
    version: str = None,
) -> dict:
    """Export MLP classifier .pth -> .onnx. No tensorflow dependency."""
    if version is None:
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    exports = Path(exports_dir)
    exports.mkdir(parents=True, exist_ok=True)

    onnx_path = str(exports / f"sign_mlp_{version}.onnx")
    export_classifier_to_onnx(checkpoint_path, onnx_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    label_map = checkpoint["label_map"]

    labels_path = str(exports / f"classifier_labels_{version}.json")
    with open(labels_path, "w") as f:
        json.dump(label_map, f, indent=2)

    onnx_size = Path(onnx_path).stat().st_size / 1024

    return {
        "version": version,
        "format": "onnx",
        "onnx_path": onnx_path,
        "labels_path": labels_path,
        "size_kb": round(onnx_size, 2),
        "num_classes": len(label_map),
        "created_at": datetime.now().isoformat(),
    }


def export_sequence_onnx_only(
    checkpoint_path: str,
    exports_dir: str,
    version: str = None,
) -> dict:
    """Export GRU sequence model .pth -> .onnx. No tensorflow dependency."""
    if version is None:
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    exports = Path(exports_dir)
    exports.mkdir(parents=True, exist_ok=True)

    onnx_path = str(exports / f"sign_gru_{version}.onnx")
    export_sequence_to_onnx(checkpoint_path, onnx_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocabulary = checkpoint["vocabulary"]

    labels_path = str(exports / f"sequence_labels_{version}.json")
    with open(labels_path, "w") as f:
        json.dump(vocabulary, f, indent=2)

    onnx_size = Path(onnx_path).stat().st_size / 1024

    return {
        "version": version,
        "format": "onnx",
        "onnx_path": onnx_path,
        "labels_path": labels_path,
        "size_kb": round(onnx_size, 2),
        "num_classes": len(vocabulary),
        "created_at": datetime.now().isoformat(),
    }
