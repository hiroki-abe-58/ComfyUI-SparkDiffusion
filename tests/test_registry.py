"""Comfy Registry metadata and package contents (https://docs.comfy.org/registry/specifications)."""

import fnmatch
import re
import shutil
import subprocess

import pytest

from conftest import ROOT, load

tomllib = pytest.importorskip("tomllib")  # Python 3.11+

PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT["project"]
COMFY = PYPROJECT["tool"]["comfy"]

VALID_OS_PREFIXES = ("Microsoft", "POSIX", "MacOS", "OS Independent")
VALID_ACCELERATORS = {
    "GPU :: NVIDIA CUDA",
    "GPU :: AMD ROCm",
    "GPU :: Intel Arc",
    "NPU :: Huawei Ascend",
    "GPU :: Apple Metal",
}


def test_node_id_follows_registry_rules():
    name = PROJECT["name"]
    assert name == "sparkdiffusion"
    assert len(name) <= 100
    assert re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*", name), "alphanumeric, '-', '_', '.'; must start with a letter"
    assert not re.search(r"[._-]{2}", name), "no consecutive special characters"
    assert not name.lower().startswith("comfyui"), "best practice: no ComfyUI prefix"


def test_version_is_semver_and_matches_runtime_metadata():
    assert re.fullmatch(r"\d+\.\d+\.\d+", PROJECT["version"])
    assert load("spark_runtime.metadata").package_version() == PROJECT["version"]


def test_publisher_and_urls():
    assert COMFY["PublisherId"] == "hiroki-abe-58"
    assert COMFY["DisplayName"]
    urls = PROJECT["urls"]
    assert urls["Repository"] == "https://github.com/hiroki-abe-58/ComfyUI-SparkDiffusion"
    assert urls["Issues"].startswith(urls["Repository"])
    assert PROJECT["license"] == {"file": "LICENSE"} and (ROOT / "LICENSE").is_file()
    assert "not an official AlibabaResearch project" in PROJECT["description"]


def test_nothing_is_installed_into_comfyui_python():
    """Registry dependencies and requirements.txt are pip-installed into ComfyUI's environment by the Manager."""
    assert PROJECT["dependencies"] == []
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")] == []


def test_classifiers_are_registry_valid():
    classifiers = PROJECT["classifiers"]
    os_values = [c.split(" :: ", 1)[1] for c in classifiers if c.startswith("Operating System :: ")]
    acc_values = [c.split(" :: ", 1)[1] for c in classifiers if c.startswith("Environment :: ")]
    assert os_values and all(v.startswith(VALID_OS_PREFIXES) for v in os_values)
    assert acc_values == ["GPU :: NVIDIA CUDA"] and set(acc_values) <= VALID_ACCELERATORS
    assert not any("MacOS" in v for v in os_values), "Apple Silicon is only on the roadmap"


def _comfyignore_patterns():
    lines = (ROOT / ".comfyignore").read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def _ignored(path: str, patterns) -> bool:
    """Small subset of gitwildmatch, enough for the patterns this repository uses."""
    for pat in patterns:
        if pat.endswith("/"):
            if path.startswith(pat) or f"/{pat}" in f"/{path}":
                return True
        elif "/" in pat:
            if fnmatch.fnmatch(path, pat):
                return True
        elif fnmatch.fnmatch(path.rsplit("/", 1)[-1], pat):
            return True
    return False


def _git_files():
    git = shutil.which("git")
    if not git or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout (e.g. installed from the Registry archive)")
    out = subprocess.run([git, "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


def test_registry_package_contents():
    patterns = _comfyignore_patterns()
    files = [f for f in _git_files() if not _ignored(f, patterns)]
    assert files, "empty package"
    for required in (
        "__init__.py",
        "LICENSE",
        "NOTICE",
        "README.md",
        "requirements.txt",
        "pyproject.toml",
        "nodes/__init__.py",
        "spark_runtime/launcher.py",
        "schedules/crossdistill.py",
        "scripts/doctor.py",
        "workflows/sparkdiffusion_1p3b_t2v.json",
    ):
        assert required in files, required
    assert not any(f.startswith(("tests/", ".github/")) for f in files)
    forbidden = re.compile(r"\.(pth|pt|ckpt|safetensors|gguf|bin|mp4|mov|webm|png|jpe?g|env)$|(^|/)\.env")
    assert not [f for f in files if forbidden.search(f)]
    assert "sparkdiffusion_config.json" not in files
