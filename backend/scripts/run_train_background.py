"""Train a model in a true detached background process.

Same approach as `run_reextract_background.py` but for any training
script. Logs to `logs/<name>.log` and survives the parent shell exit.

Usage:
    python -m scripts.run_train_background
    python -m scripts.run_train_background --name myexp --epochs 40 --aug-factor 5
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run training detached. All unrecognized args are forwarded to the training script.",
        allow_abbrev=False,
    )
    parser.add_argument("--name", default="train", help="Log file basename.")
    parser.add_argument(
        "--python", default=str(PROJECT_ROOT.parent / ".venv" / "Scripts" / "python.exe")
    )
    parser.add_argument("--module", default="scripts.train_transformer")
    # Use parse_known_args to allow forwarding extra args without the `--` separator.
    args, rest = parser.parse_known_args()

    log_path = PROJECT_ROOT / "logs" / f"{args.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [args.python, "-u", "-m", args.module] + [a for a in rest if a != "--"]
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write(f"Launching: {' '.join(cmd)}\n")
    log_file.write(f"CWD: {PROJECT_ROOT}\n")
    log_file.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.flush()

    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    log_file.write(f"Child PID: {proc.pid}\n")
    log_file.flush()
    print(f"Training started. PID={proc.pid}, log={log_path}")
    print(f"Tail: Get-Content {log_path} -Wait")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
