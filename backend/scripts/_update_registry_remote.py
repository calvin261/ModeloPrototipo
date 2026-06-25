from huggingface_hub import HfApi, hf_hub_download
import json
import os
import hashlib

api = HfApi(token="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")
repo = "calvin261/lsec-api"
token = "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"

p = hf_hub_download(repo, "models/exports/registry.json", repo_type="space", token=token)
with open(p, encoding="utf-8") as f:
    reg = json.load(f)

onnx_local = r"C:\Users\jeffe\Documents\Proyectos\ModeloPrototipo\backend\models\exports\sign_transformer_v20260625_072907.onnx"
with open(onnx_local, "rb") as f:
    sha256 = hashlib.sha256(f.read()).hexdigest()
size_kb = os.path.getsize(onnx_local) / 1024

new_entry = {
    "model_type": "transformer",
    "version": "20260625_072907",
    "fp32_onnx": "/data/models/sign_transformer_v20260625_072907.onnx",
    "int8_onnx": None,
    "labels": "/data/models/sign_transformer_labels_v20260625_072907.json",
    "n_classes": 12,
    "max_seq_len": 60,
    "n_params": 575244,
    "exported_at": "2026-06-25T07:29:07",
    "sha256": sha256,
    "size_kb": round(size_kb, 1),
}

existing = reg.get("models", [])
reg["models"] = [new_entry] + existing

api.upload_file(
    path_or_fileobj=json.dumps(reg, indent=2, ensure_ascii=False).encode("utf-8"),
    path_in_repo="models/exports/registry.json",
    repo_id=repo,
    repo_type="space",
    commit_message="Register new transformer model " + new_entry["version"],
)
print("Registry updated.")
print(json.dumps(new_entry, indent=2, ensure_ascii=False))