"""Import this repository as a package the way ComfyUI does, without running its __init__.

The repository root is a ComfyUI custom node *package* (its folder name contains a
dash), so scripts register it under a stable alias before importing submodules.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "comfyui_sparkdiffusion"


def package():
    if PKG not in sys.modules:
        mod = types.ModuleType(PKG)
        mod.__path__ = [str(ROOT)]  # type: ignore[attr-defined]
        mod.__file__ = str(ROOT / "__init__.py")
        sys.modules[PKG] = mod
    return sys.modules[PKG]


def load(name: str):
    package()
    return importlib.import_module(f"{PKG}.{name}")


def add_runtime_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("runtime (defaults: SPARKDIFFUSION_* env vars, then sparkdiffusion_config.json)")
    g.add_argument("--backend", default="", choices=["", "auto", "windows-native", "wsl2", "linux"])
    g.add_argument("--repo", default="", help="SparkDiffusion checkout")
    g.add_argument("--python", default="", help="runtime interpreter (Linux path for wsl2)")
    g.add_argument("--model-root", default="", help="directory with Wan assets and SparkWan checkpoints")
    g.add_argument("--wsl-distro", default="")
    g.add_argument("--quantization", default="fp8", choices=["fp8", "bf16"])
    g.add_argument("--fp8-compat", default="auto", choices=["auto", "off", "force"])
    g.add_argument("--no-fused-kernels", action="store_true")
    g.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--sla-src", default="")
    g.add_argument("--timeout", type=int, default=3600)


def runtime_from_args(args):
    C = load("spark_runtime.config")
    command = load("spark_runtime.command")
    file_cfg = C.load_config_file()
    return command.RuntimeConfig(
        backend=C.resolve("backend", args.backend, file_cfg) or "auto",
        repo=C.resolve("repo", args.repo, file_cfg),
        python=C.resolve("python", args.python, file_cfg),
        model_root=C.resolve("model_root", args.model_root, file_cfg),
        wsl_distro=C.resolve("wsl_distro", args.wsl_distro, file_cfg),
        quantization=args.quantization,
        fp8_compat=args.fp8_compat,
        fused_kernels=not args.no_fused_kernels,
        torch_compile=args.torch_compile,
        sla_src=C.resolve("sla_src", args.sla_src, file_cfg),
        timeout_s=args.timeout,
    )
