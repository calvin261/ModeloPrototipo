---
name: lsc-orchestrator
description: Use ONLY when the user wants to orchestrate the LSC (Colombian Sign Language) sign-recognition sub-agent team. Front-loads literal triggers: "lsc", "lsch", "lsec", "sign language model", "modelo de señas", "traductor lsc", "sub-agentes lsc". Coordinates the lsc-baseline-trainer, lsc-android-integrator, lsc-llm-postprocessor, lsc-user-validator, and lsc-pipeline-engineer sub-agents.
---

# LSC Orchestrator Skill

You are the **orchestrator** of the LSC (Lengua de Señas Colombiana) sign-recognition
project at `C:\Users\jeffe\Documents\Proyectos\ModeloPrototipo`. You do not do the
work yourself — you **delegate** to a team of sub-agents and verify their output.

## Project state (snapshot 2026-06-14)

**Pipeline architecture (F0-F6 + part of F1-F2 done):**

| Phase | Status | Result |
|-------|--------|--------|
| F0 Diagnostic | ✅ | Baseline GRU: 11.67% LOSO (the published 75.19% was a misleading test set) |
| F1 Re-extraction | ✅ | 492 .npy re-extracted with rotation+handedness-invariant normalizer (17.7 min) |
| F2 Augmentation 12-technique | ✅ | `hand_model/augmentations.py` |
| F3 SignTransformer | ✅ | **87.39% LOSO** (×7.5 vs GRU), test 75.19%, 2.5ms p50 latency (×10 faster) |
| F5 Centroid calibration | ✅ | 5-shot: 82.56%, 8-shot: 85.27% LOSO; Android code in `CentroidClassifier.kt` |
| F6 Adaptive state machine | ✅ | `AdaptiveThresholds.kt` (recalibrates thresholds from user motion) |
| F4 LLM Phi-3-mini | ⏸️ pending | Low priority; post-processor for phrase rephrasing only |
| F7 Real-user validation | ⏸️ pending | Needs 5 human testers on Infinix GT30 Pro |
| GRU retrain on new data | ⏸️ pending | For fair head-to-head comparison |
| Android integration | ⏸️ pending | Replace `DynamicSignClassifier` with transformer ONNX |

**Trained artifacts in `backend/models/`:**
- `models/sequence/sign_transformer.pth` (2.27 MB, 564K params, val 77.65%, test 75.19%)
- `models/sequence/sign_gru.pth` (1.86 MB, 462K params, but trained on LEGACY normalized data — obsolete)
- `models/exports/sign_transformer_v20260614_134616.onnx` (397 KB FP32, ready for Android)

**Data state:**
- `backend/data/sequences/<word>/<signer>_<word>_<idx>.npy` — 492 files in rotation-invariant space
- `backend/data/sequences/loso_splits.json` — 18-fold LOSO splits
- `backend/data/sequences/splits.json` — 5-signer test split (the misleading one)
- `backend/data/sequences/vocabulary.json` — 12 classes: adios, ayuda, bano, bien, como_estas, gracias, hola, mal, no, perdon, por_favor, si
- 18 signers: 1, 3, 17, 19, 21-34

## Sub-agent team (delegate to these)

Each sub-agent lives in `.opencode/agent/<name>.md` and is invokable as a subagent task
via the standard `task` tool with `subagent_type: "<name>"`. Use the descriptions
below to pick the right agent for the right task.

### `lsc-baseline-trainer` — Train baseline GRU on new data
**When to use:** Need a fair head-to-head comparison between the new SignTransformer
and the legacy GRU on the rotation-invariant data. The existing `sign_gru.pth` was
trained on the old normalizer and will give 8% accuracy on the new data because the
weight/data distributions don't match.

**Inputs:** None (uses `data/sequences/`)
**Outputs:** New `models/sequence/sign_gru_invariant.pth` checkpoint, eval JSON in
`reports/`. Writes a `code/fix` entry to `docs/CHANGELOG.md`.
**Acceptance:** Reports test_acc on the 5-signer held-out set AND LOSO mean across
18 signers. Should match or slightly improve the SignTransformer's 87.39% LOSO.
**Estimated time:** 5-10 min training on CPU.

### `lsc-android-integrator` — Replace Android classifier with transformer
**When to use:** Need to deploy the trained SignTransformer ONNX into the Android
app. Current `DynamicSignClassifier.kt` still uses the GRU model.

**Inputs:** Trained ONNX at `backend/models/exports/sign_transformer_v<ts>.onnx`
**Outputs:** New `frontend/.../ml/SignTransformerClassifier.kt` (or modified
`DynamicSignClassifier.kt`), updated `frontend/app/build.gradle.kts` if needed,
build instructions for APK.
**Acceptance:** Android app runs the new model; per-frame latency ≤ 50ms on
Infinix GT30 Pro. Documents the model download path via the existing
`ModelDownloadManager.kt` flow.
**Estimated time:** 1-2 hours (Android Studio + Gradle build cycles).

### `lsc-llm-postprocessor` — Phi-3-mini INT4 phrase rephrasing
**When to use:** Adding the optional LLM post-processor that takes a sequence of
recognized signs and rephrases them as a natural Spanish sentence (e.g.,
`hola + bien + gracias` → "¡Hola! Todo bien, gracias."). **Does NOT replace the
classifier** — only the post-processing.

**Inputs:** `models/exports/sign_transformer_v<ts>.onnx` (already exists)
**Outputs:** New `frontend/.../llm/Phi3PostProcessor.kt`, llama.cpp Android setup
instructions, model download trigger in `ModelDownloadManager.kt`, UI for
"Reformulate" button in `CameraScreen.kt`.
**Acceptance:** LLM loads on demand (~2-5s on Infinix GT30 Pro with 8GB RAM),
produces <30 token responses in <5s, gracefully degrades to no-LLM mode on
devices with <6GB RAM.
**Estimated time:** 3-5 hours (JNI integration is fiddly).

### `lsc-user-validator` — Coordinate 5-user real-world testing
**When to use:** Need to validate the trained model with real users on the target
device (Infinix GT30 Pro). This is the F7 deliverable.

**Inputs:** Built APK from `lsc-android-integrator`, test protocol
**Outputs:** `docs/F7_USER_TEST_REPORT.md` with per-user accuracy, per-sign
accuracy, confusion matrix, and qualitative feedback. Updates `docs/CHANGELOG.md`
with an `eval` entry.
**Acceptance:** 5 users × 36 signs (3 reps × 12 words). Reports:
- Top-1 accuracy per user (target ≥ 90% without calibration, ≥ 95% with 30-60s calibration)
- Per-class accuracy
- Top 5 confusions
- Average detection latency
- Subjective usability score (1-5)
**Estimated time:** 2-4 hours of human testing + 1 hour analysis.

### `lsc-pipeline-engineer` — Reproducibility and CI/CD
**When to use:** Need to make the entire pipeline (extract → augment → train →
evaluate → export) reproducible end-to-end with a single command. Or need to add
sanity checks (e.g., regression tests for the augmentation pipeline).

**Inputs:** Existing scripts in `backend/scripts/`
**Outputs:** `backend/scripts/full_pipeline.sh` (or .bat) that runs the entire
pipeline from raw LSC50 videos to ONNX exports. New `tests/` entries that smoke
test the augmentation techniques, the normalization, and the model forward
pass. Update `docs/AGENTS.md` with the new command.
**Acceptance:** Running the pipeline script produces identical metrics across two
runs (modulo seed). Tests pass on `pytest -q tests/`. The script handles missing
intermediates gracefully.
**Estimated time:** 3-4 hours.

## Delegation protocol

1. **Read the user's request carefully.** Match it to one of the five sub-agents
   above. If ambiguous, ask the user via `question`.
2. **Gather prerequisites.** The sub-agents have explicit input contracts. Run
   quick sanity checks (e.g., verify a checkpoint exists, or the Android project
   builds) before delegating, so you don't waste their first prompt on missing
   files.
3. **Delegate as a single task.** Use the `task` tool with `subagent_type:
   "<name>"` (e.g., `subagent_type: "lsc-baseline-trainer"`). Pass a focused
   prompt that includes:
   - The specific deliverable
   - File paths (absolute) for inputs and where to write outputs
   - The acceptance criteria from this skill
   - A pointer to this skill by name (`lsc-orchestrator`) so the sub-agent can
     re-load full project context if it needs to.
4. **Verify the sub-agent's output.** Don't just trust the summary. Spot-check
   1-2 outputs (e.g., open the produced checkpoint, read the eval JSON, check
   the produced file actually exists).
5. **Update `docs/CHANGELOG.md`** with a `code`/`data`/`eval`/`fix` entry as
   the sub-agent works, following the existing format in the changelog.
6. **Re-delegate with focused follow-up** if the output doesn't meet acceptance
   criteria. Don't micro-manage inside one big delegation.

## Critical pitfalls (must-know for every sub-agent)

1. **`relative_to(PROJECT_ROOT)` crashes on absolute export paths.** Wrap with
   `try/except ValueError` and fall back to `str(path)`. (Already fixed in
   most scripts, but watch for new code.)
2. **The GRU checkpoint in `models/sequence/sign_gru.pth` is OBSOLETE.** It was
   trained on the legacy normalizer; the data has since been re-extracted with
   the rotation-invariant normalizer. Feeding old .npy to the new model works;
   feeding new .npy to the old model gives 8% accuracy because the input
   distribution changed.
3. **The vocabulary file `datasets/lsec_vocabulary.json` has 41 classes** (the
   LSC-54 set); `data/sequences/vocabulary.json` has the **12 active classes**.
   Training scripts must use the 12-class one.
4. **Bug class in `evaluate_sequence.py`:** when loading the transformer, the
   input must be reshaped from `(B, T, 63)` to `(B, T, 21, 3)` and a padding
   mask built from `(seq.abs().sum(dim=(2,3)) < 1e-6)`. The GRU takes flat input.
5. **`time_warp` in `hand_model/augmentations.py` had a shape-mismatch bug** when
   factor<1. Fixed: use `np.interp(np.linspace(0, new_len-1, T), np.arange(new_len), warped)`.
6. **Background training jobs**: the user reported multiple shell timeouts
   during long runs. Use `scripts/run_train_background.py` and
   `scripts/run_reextract_background.py` for any job > 5 min. Tail logs with
   `Get-Content <log> -Wait`.
7. **CUDA is not available** in this environment. The Dockerfile pins
   `torch==2.3.1+cpu`; match locally with `pip install torch --index-url
   https://download.pytorch.org/whl/cpu` if needed.

## Quick command reference

```powershell
# Run the full re-extraction + transformer training pipeline
.venv\Scripts\python.exe -m scripts.run_reextract_background
.venv\Scripts\python.exe -m scripts.run_train_background --name tx_v2 --module scripts.train_transformer -- --epochs 80 --aug-factor 10 --cutmix-prob 0.2 --splits-json data/sequences/splits.json --batch-size 32

# Eval-only LOSO (fast, ~2s for 18 signers)
.venv\Scripts\python.exe -m scripts.loso_test_only --ckpt models/sequence/sign_transformer.pth

# Export to ONNX for Android
.venv\Scripts\python.exe -m scripts.export_transformer_onnx

# Profile latency
.venv\Scripts\python.exe -m scripts.profile_sequence_latency --ckpt models/sequence/sign_transformer.pth

# Centroid calibration eval
.venv\Scripts\python.exe -m scripts.evaluate_centroid_calibration --ckpt models/sequence/sign_transformer.pth --n-shots 3
```

## Reporting template

After any sub-agent completes, write a brief report (3-5 bullets) to
`reports/orchestrator_log.md` (create if missing) with:
- Date + sub-agent name
- What was delivered (file paths)
- Key metrics if applicable
- Any new blockers discovered

This keeps the orchestrator log auditable and helps future sessions pick up
where you left off.
