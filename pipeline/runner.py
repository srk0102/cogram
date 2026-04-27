"""Entry point: run the full pipeline (sanitize → ingest → annotate → profile → readback).

Replaces run.bat for environments that prefer a Python entry point.
Console script: cogram-pipeline
"""
import asyncio
import subprocess
import sys


STEPS = [
    ("sanitize", "src.sanitize_episodes"),
    ("ingest", "src.ingest"),
    ("annotate", "src.annotate"),
    ("profile", "src.profile"),
    ("readback", "src.readback"),
]


def main() -> int:
    for label, module in STEPS:
        print(f"\n=== {label} ===")
        rc = subprocess.run([sys.executable, "-m", module]).returncode
        if rc != 0:
            print(f"[ERROR] {label} exited with code {rc}")
            return rc
    print("\n=== Pipeline complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
