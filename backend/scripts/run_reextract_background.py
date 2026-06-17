"""Run LSC50 re-extraction as a true detached background process.

Uses `subprocess.Popen` with DETACHED_PROCESS so the process survives
the parent shell exiting, and redirects stdout/stderr to a log file.

Usage (from backend/):
    python -m scripts.run_reextract_background
    python -m scripts.run_reextract_background --limit 50
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run re-extraction detached.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--python", default=str(PROJECT_ROOT.parent / ".venv" / "Scripts" / "python.exe"))
    args = parser.parse_args()

    log_path = PROJECT_ROOT / "logs" / "reextract.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python, "-u", "-m", "scripts.reextract_sequences_invariant",
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]

    # Open log file (overwrite each run)
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write(f"Launching: {' '.join(cmd)}\n")
    log_file.write(f"CWD: {PROJECT_ROOT}\n")
    log_file.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.flush()

    # CREATE_NEW_PROCESS_GROUP makes the child survive parent shell exit on Windows.
    flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP
        if sys.platform == "win32"
        else 0
    )
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    log_file.write(f"Child PID: {proc.pid}\n")
    log_file.flush()

    print(f"Re-extraction started in background. PID={proc.pid}")
    print(f"Log: {log_path}")
    print()
    print("Tail the log with:")
    print(f"  Get-Content {log_path} -Wait")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
