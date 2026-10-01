"""Diagnose the SparkDiffusion runtime: python scripts/doctor.py [--backend wsl2 ...]"""

from __future__ import annotations

import argparse
import sys

import _bootstrap


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    _bootstrap.add_runtime_args(p)
    p.add_argument("--comfyui", default="", help="ComfyUI directory (optional check)")
    p.add_argument("--no-compile-smoke", action="store_true", help="skip the torch.compile smoke test")
    args = p.parse_args()
    doctor = _bootstrap.load("spark_runtime.doctor")
    cfg = _bootstrap.runtime_from_args(args)
    checks = doctor.run_doctor(
        cfg,
        comfyui_path=args.comfyui or None,
        compile_smoke=not args.no_compile_smoke,
        emit=lambda c: print(c.line(), flush=True),
    )
    worst = (
        "FAIL"
        if any(c.status == "FAIL" for c in checks)
        else "WARN"
        if any(c.status == "WARN" for c in checks)
        else "PASS"
    )
    print(f"\nOverall: {worst}")
    return 1 if worst == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
