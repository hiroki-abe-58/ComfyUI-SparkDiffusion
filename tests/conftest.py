"""Make the repository importable as a package, the way ComfyUI loads custom nodes."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = "comfyui_sparkdiffusion"


def _register_package() -> None:
    if PKG not in sys.modules:
        mod = types.ModuleType(PKG)
        mod.__path__ = [str(ROOT)]  # type: ignore[attr-defined]
        mod.__file__ = str(ROOT / "__init__.py")
        sys.modules[PKG] = mod


_register_package()


def load(name: str):
    return importlib.import_module(f"{PKG}.{name}")


@pytest.fixture
def pkg():
    return load


@pytest.fixture
def repo_root() -> Path:
    return ROOT
