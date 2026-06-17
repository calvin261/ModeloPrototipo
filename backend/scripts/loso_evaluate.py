"""Run a Leave-One-Signer-Out (LOSO) evaluation over all signers.

This is the gold standard for signer-independent sign language recognition:
each signer is held out as the test set, the remaining signers train the
model, and we report the macro-average accuracy across all folds.

The script supports both the legacy GRU (`scripts/train_sequence.py`) and
the new Sign Transformer (`scripts/train_transformer.py`) via --model.

For a quick first pass we recommend:
    python -m scripts.loso_evaluate --model gru --folds 1 5 10 18

For a full LOSO (18 folds, ~30-60 min total):
    python -m scripts.loso_evaluate --model transformer

Usage:
    python -m scripts.loso_evaluate --model transformer
    python -m scripts.loso_evaluate --model gru --folds 0 1 2
    python -m scripts.loso_evaluate --export reports/loso_transformer.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _run_gru_fold(fold: dict, cfg_overrides: dict, splits_path: Path) -> dict:
    """Run a single LOSO fold using the legacy GRU pipeline.

    Returns a dict with: fold, test_signer, val_acc, test_acc, n_train, n_val, n_test, time_s.
    """
    from hand_model.config import SequenceConfig
    cfg = SequenceConfig()
    for k, v in cfg_overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.sequences_dir = str(PROJECT_ROOT / "data" / "sequences")
    cfg.vocabulary_path = str(PROJECT_ROOT / "data" / "sequences" / "vocabulary.json")
    cfg.model_save_path = str(
        PROJECT_ROOT / "models" / "sequence" / f"_loso_gru_{fold['test_signer']}.pth"
    )
    # Build a splits.json-equivalent for this fold
    fold_splits = {
        "train_signers": fold["train_signers"],
        "val_signers": fold["val_signers"],
        "test_signers": [fold["test_signer"]],
        "per_split_counts": {
            "train": fold["train_count"],
            "val": fold["val_count"],
            "test": fold["test_count"],
        },
    }
    fold_splits_path = PROJECT_ROOT / "data" / "sequences" / f"_loso_fold_{fold['fold']}.json"
    with open(fold_splits_path, "w", encoding="utf-8") as f:
        json.dump(fold_splits, f, indent=2)
    from scripts.train_sequence import train as train_gru
    t0 = time.time()
    result = train_gru(cfg, splits_json=str(fold_splits_path), evaluate_test=True,
                       progress_callback=None)
    result["time_s"] = time.time() - t0
    result["fold"] = fold["fold"]
    result["test_signer"] = fold["test_signer"]
    fold_splits_path.unlink(missing_ok=True)
    return result


def _run_transformer_fold(fold: dict, cfg_overrides: dict) -> dict:
    from hand_model.config import SequenceConfig
    cfg = SequenceConfig()
    for k, v in cfg_overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.sequences_dir = str(PROJECT_ROOT / "data" / "sequences")
    cfg.vocabulary_path = str(PROJECT_ROOT / "data" / "sequences" / "vocabulary.json")
    cfg.model_save_path = str(
        PROJECT_ROOT / "models" / "sequence" / f"_loso_transformer_{fold['test_signer']}.pth"
    )
    fold_splits = {
        "train_signers": fold["train_signers"],
        "val_signers": fold["val_signers"],
        "test_signers": [fold["test_signer"]],
        "per_split_counts": {
            "train": fold["train_count"],
            "val": fold["val_count"],
            "test": fold["test_count"],
        },
    }
    fold_splits_path = PROJECT_ROOT / "data" / "sequences" / f"_loso_fold_{fold['fold']}.json"
    with open(fold_splits_path, "w", encoding="utf-8") as f:
        json.dump(fold_splits, f, indent=2)
    from scripts.train_transformer import train as train_tx
    t0 = time.time()
    result = train_tx(cfg, splits_json=str(fold_splits_path), evaluate_test=True,
                       progress_callback=None)
    result["time_s"] = time.time() - t0
    result["fold"] = fold["fold"]
    result["test_signer"] = fold["test_signer"]
    fold_splits_path.unlink(missing_ok=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="LOSO cross-validation for sign models.")
    parser.add_argument("--model", choices=["gru", "transformer"], default="transformer")
    parser.add_argument(
        "--loso-splits", type=Path,
        default=PROJECT_ROOT / "data" / "sequences" / "loso_splits.json",
    )
    parser.add_argument(
        "--folds", type=int, nargs="+", default=None,
        help="Specific fold indices to run. Default: all.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--aug-factor", type=int, default=20)
    parser.add_argument("--cutmix-prob", type=float, default=0.3)
    parser.add_argument(
        "--export", type=Path, default=None,
        help="Save full LOSO results (per-fold + summary) to this JSON.",
    )
    args = parser.parse_args()

    if not args.loso_splits.exists():
        print(f"[!] LOSO splits missing: {args.loso_splits}")
        print("    Generate with: python -m scripts.build_loso_splits")
        return 1

    with open(args.loso_splits, encoding="utf-8") as f:
        loso = json.load(f)
    folds = loso["folds"]
    if args.folds is not None:
        folds = [f for f in folds if f["fold"] in args.folds]

    print("=" * 60)
    print(f"  LOSO evaluation — model: {args.model}")
    print("=" * 60)
    print(f"Total folds in splits: {len(loso['folds'])}")
    print(f"Folds to run:         {len(folds)}")
    print(f"Vocabulary:           {loso.get('vocabulary', {})}")
    print(f"Epochs per fold:      {args.epochs}")
    print(f"Augmentation factor:  {args.aug_factor}x  cutmix p={args.cutmix_prob}")
    print()

    cfg_overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "aug_factor": args.aug_factor,
        "cutmix_prob": args.cutmix_prob,
    }

    per_fold: list[dict] = []
    val_accs, test_accs = [], []
    t_start = time.time()
    for fold in folds:
        print(f"\n{'#' * 60}")
        print(f"# Fold {fold['fold']:>2}  test_signer={fold['test_signer']:>3}  "
              f"train={fold['train_count']:>3}  val={fold['val_count']:>2}  test={fold['test_count']:>2}")
        print(f"{'#' * 60}\n")
        try:
            if args.model == "gru":
                result = _run_gru_fold(fold, cfg_overrides, args.loso_splits)
            else:
                result = _run_transformer_fold(fold, cfg_overrides)
        except Exception as e:  # noqa: BLE001
            print(f"  Fold {fold['fold']} failed: {e}")
            result = {"fold": fold["fold"], "test_signer": fold["test_signer"], "error": str(e)}
        per_fold.append(result)
        if "test_acc" in result and result["test_acc"] is not None:
            val_accs.append(result.get("best_val_acc", 0))
            test_accs.append(result["test_acc"])
        elapsed = time.time() - t_start
        avg = sum(test_accs) / len(test_accs) if test_accs else 0
        print(f"\n  Fold {fold['fold']} done in {result.get('time_s', 0):.0f}s. "
              f"Test acc={result.get('test_acc', '?')}.  Running mean test acc={avg * 100:.2f}%  "
              f"({len(test_accs)}/{len(folds)} folds).  Wall={elapsed:.0f}s")

    # Summary
    if test_accs:
        import statistics
        mean_test = statistics.mean(test_accs)
        std_test = statistics.stdev(test_accs) if len(test_accs) > 1 else 0
        min_test = min(test_accs)
        max_test = max(test_accs)
        mean_val = statistics.mean(val_accs)
    else:
        mean_test = std_test = min_test = max_test = mean_val = 0

    print()
    print("=" * 60)
    print(f"  LOSO summary ({args.model})")
    print("=" * 60)
    print(f"Folds completed: {len(test_accs)}/{len(folds)}")
    print(f"Mean test acc:   {mean_test * 100:.2f}%  +/-{std_test * 100:.2f}%")
    print(f"Min test acc:    {min_test * 100:.2f}%")
    print(f"Max test acc:    {max_test * 100:.2f}%")
    print(f"Mean val acc:    {mean_val * 100:.2f}%")
    print(f"Total wall:      {time.time() - t_start:.0f}s")
    print()

    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": args.model,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "n_folds_run": len(test_accs),
                    "n_folds_total": len(folds),
                    "summary": {
                        "mean_test_acc": mean_test,
                        "std_test_acc": std_test,
                        "min_test_acc": min_test,
                        "max_test_acc": max_test,
                        "mean_val_acc": mean_val,
                    },
                    "per_fold": per_fold,
                },
                f, indent=2, ensure_ascii=False,
            )
        print(f"Results exported to: {args.export.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
