"""Path handling: host <-> runtime path mapping, asset discovery and redaction.

Everything here is pure string/``pathlib`` logic so it is unit-testable on any
OS. The only side-effecting helpers are ``wslpath_via_wsl`` (optional, calls
``wsl.exe``) and ``resolve_assets`` (stats files).
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, Iterable, List, Optional, Sequence

from . import profiles as _profiles

BACKEND_AUTO = "auto"
BACKEND_WINDOWS = "windows-native"
BACKEND_WSL2 = "wsl2"
BACKEND_LINUX = "linux"
BACKENDS = (BACKEND_AUTO, BACKEND_WINDOWS, BACKEND_WSL2, BACKEND_LINUX)

DEFAULT_WSL_AUTOMOUNT_ROOT = "/mnt/"

_DRIVE_RE = re.compile(r"^(?P<drive>[A-Za-z]):(?:[\\/](?P<rest>.*))?$")
_WSL_UNC_RE = re.compile(
    r"^(?:\\\\|//)(?:wsl\$|wsl\.localhost)[\\/](?P<distro>[^\\/]+)(?:[\\/](?P<rest>.*))?$", re.IGNORECASE
)


class PathConversionError(ValueError):
    pass


def is_windows_path(path: str) -> bool:
    return bool(_DRIVE_RE.match(path) or _WSL_UNC_RE.match(path) or path.startswith("\\\\"))


def windows_to_wsl(path: str, automount_root: str = DEFAULT_WSL_AUTOMOUNT_ROOT, distro: Optional[str] = None) -> str:
    """Convert a Windows path to the path WSL sees, without calling ``wsl.exe``.

    * ``C:\\models\\Spark`` -> ``/mnt/c/models/Spark`` (any drive letter, lower-cased)
    * ``\\\\wsl$\\Ubuntu\\home\\u\\x`` -> ``/home/u/x`` (must match ``distro`` if given)
    * POSIX paths are returned unchanged.

    Spaces and non-ASCII characters are preserved verbatim; quoting is never
    needed because commands are passed as argument lists.
    """
    if not path:
        raise PathConversionError("empty path")
    path = os.fspath(path)
    m = _WSL_UNC_RE.match(path)  # before the POSIX check: "//wsl.localhost/..." starts with "/"
    if m is None and path.startswith("/"):
        return path
    if m:
        if distro and m.group("distro").lower() != distro.lower():
            raise PathConversionError(
                f"{path!r} lives in WSL distro {m.group('distro')!r}, not the configured {distro!r}"
            )
        rest = (m.group("rest") or "").replace("\\", "/").strip("/")
        return "/" + rest if rest else "/"
    m = _DRIVE_RE.match(path)
    if m:
        root = automount_root if automount_root.endswith("/") else automount_root + "/"
        rest = (m.group("rest") or "").replace("\\", "/")
        rest = re.sub(r"/+", "/", rest).rstrip("/")
        base = f"{root}{m.group('drive').lower()}"
        return f"{base}/{rest}" if rest else base
    if path.startswith("\\\\"):
        raise PathConversionError(f"network share paths are not reachable from WSL automatically: {path!r}")
    raise PathConversionError(f"relative or unrecognised path cannot be mapped into WSL: {path!r}")


def wsl_to_windows(path: str, automount_root: str = DEFAULT_WSL_AUTOMOUNT_ROOT, distro: Optional[str] = None) -> str:
    """Inverse of :func:`windows_to_wsl` for ``/mnt/<drive>/...`` and distro paths."""
    root = automount_root if automount_root.endswith("/") else automount_root + "/"
    if path.startswith(root):
        rest = path[len(root) :]
        drive, _, tail = rest.partition("/")
        if len(drive) == 1 and drive.isalpha():
            return f"{drive.upper()}:\\" + tail.replace("/", "\\")
    if distro and path.startswith("/"):
        return f"\\\\wsl.localhost\\{distro}" + path.replace("/", "\\")
    raise PathConversionError(f"cannot map {path!r} back to Windows")


def wslpath_via_wsl(path: str, distro: Optional[str] = None, timeout: float = 20.0) -> str:
    """Ask ``wslpath -a`` inside the distro (authoritative, honours wsl.conf)."""
    cmd: List[str] = ["wsl.exe"]
    if distro:
        cmd += ["-d", distro]
    cmd += ["--exec", "wslpath", "-a", "-u", path]
    out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=True)
    return out.stdout.decode("utf-8", "replace").strip()


@functools.lru_cache(maxsize=8)
def detect_automount_root(distro: Optional[str] = None) -> str:
    """WSL automount root (``/mnt/`` unless ``wsl.conf`` changes it), asked once per distro via ``wslpath``."""
    try:
        drive = os.environ.get("SystemDrive", "C:") + "\\"
        mapped = wslpath_via_wsl(drive, distro)  # e.g. "/mnt/c"
    except (OSError, subprocess.SubprocessError):
        return DEFAULT_WSL_AUTOMOUNT_ROOT
    letter = drive[0].lower()
    if mapped.rstrip("/").endswith("/" + letter):
        root = mapped.rstrip("/")[: -len(letter)]
        return root if root.endswith("/") else root + "/"
    return DEFAULT_WSL_AUTOMOUNT_ROOT


def to_runtime_path(
    path: str, backend: str, automount_root: str = DEFAULT_WSL_AUTOMOUNT_ROOT, distro: Optional[str] = None
) -> str:
    """Map a host path into the path namespace of the runtime backend."""
    if backend == BACKEND_WSL2:
        return windows_to_wsl(path, automount_root=automount_root, distro=distro)
    return os.fspath(path)


def detect_backend(platform: Optional[str] = None) -> str:
    """Resolve ``auto`` to a concrete backend for this host."""
    plat = platform or os.name
    if plat in ("nt", "win32", "windows"):
        return BACKEND_WINDOWS
    return BACKEND_LINUX


def basename_any(path: str) -> str:
    """Basename that understands both Windows and POSIX separators."""
    if not path:
        return path
    if "\\" in path or _DRIVE_RE.match(path):
        return PureWindowsPath(path).name
    return PurePosixPath(path).name


_ABS_PATH_RE = re.compile(
    r"""(?x)
    (?:[A-Za-z]:[\\/][^\s"'<>|:*?]*)      # C:\... or C:/...
    | (?:\\\\[^\s"'<>|]+)                 # \\server\share or \\wsl$\...
    | (?:/(?:mnt|home|root|Users|opt|tmp|var|data|srv|media|workspace)/[^\s"']*)
    """
)


def _redact_match(m: re.Match[str]) -> str:
    return "<path:" + basename_any(m.group(0).rstrip("/\\")) + ">"


def redact_paths(text: str) -> str:
    """Replace absolute paths in free text with ``<path:basename>``."""
    return _ABS_PATH_RE.sub(_redact_match, text)


@dataclass
class ResolvedAssets:
    checkpoints: List[str]
    vae: Optional[str] = None
    t5: Optional[str] = None
    tokenizer: Optional[str] = None
    clip: Optional[str] = None
    missing: List[str] = field(default_factory=list)
    searched: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing

    def as_dict(self) -> Dict[str, object]:
        return {
            "checkpoints": self.checkpoints,
            "vae": self.vae,
            "t5": self.t5,
            "tokenizer": self.tokenizer,
            "clip": self.clip,
            "missing": self.missing,
        }


def _first_existing(candidates: Iterable[Path], want_dir: bool = False) -> Optional[Path]:
    for c in candidates:
        if c.is_dir() if want_dir else c.is_file():
            return c
    return None


def resolve_assets(model_root: str, profile: _profiles.Profile, extra_dirs: Sequence[str] = ()) -> ResolvedAssets:
    """Locate the checkpoint(s) and native Wan assets for ``profile``.

    Expected layout (each sub-directory is a ``hf download --local-dir`` target)::

        <model_root>/
          Wan2.1-T2V-1.3B/            Wan2.1_VAE.pth, models_t5_umt5-xxl-enc-bf16.pth, google/umt5-xxl/
          Wan2.1-I2V-14B-720P/        + models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth  (I2V only)
          SparkWan2.1-T2V-1.3B-480P-0.90Sparsity/SparkWan2.1-T2V-1.3B-480P-0.90Sparsity.pth

    Files placed directly in ``model_root`` are accepted too.
    """
    root = Path(model_root)
    res = ResolvedAssets(checkpoints=[])
    if not model_root or not root.is_dir():
        res.missing.append(f"model root directory not found: {model_root or '<empty>'}")
        return res

    for name in profile.checkpoint_files:
        found = _first_existing([root / profile.repo_dirname / name, root / name])
        if found is None:
            res.missing.append(f"checkpoint {name} (from https://huggingface.co/{profile.hf_repo})")
        else:
            res.checkpoints.append(str(found))

    base_dir = profile.base_repo.split("/", 1)[1]
    dirs: List[Path] = [root / d for d in (base_dir, *_profiles.SHARED_ASSET_DIRS)]
    dirs += [Path(d) for d in extra_dirs]
    dirs.append(root)
    seen = set()
    search_dirs = [d for d in dirs if not (str(d) in seen or seen.add(str(d)))]
    res.searched = [str(d) for d in search_dirs]

    vae = _first_existing(d / _profiles.VAE_FILE for d in search_dirs)
    t5 = _first_existing(d / _profiles.T5_FILE for d in search_dirs)
    tok = _first_existing((d / _profiles.TOKENIZER_DIR for d in search_dirs), want_dir=True)
    res.vae = str(vae) if vae else None
    res.t5 = str(t5) if t5 else None
    res.tokenizer = str(tok) if tok else None
    if vae is None:
        res.missing.append(f"{_profiles.VAE_FILE} (from https://huggingface.co/{profile.base_repo})")
    if t5 is None:
        res.missing.append(f"{_profiles.T5_FILE} (from https://huggingface.co/{profile.base_repo})")
    if tok is None:
        res.missing.append(f"{_profiles.TOKENIZER_DIR}/ (from https://huggingface.co/{profile.base_repo})")

    if profile.needs_clip:
        clip_dirs = [root / d for d in _profiles.CLIP_ASSET_DIRS] + search_dirs
        clip = _first_existing(d / _profiles.CLIP_FILE for d in clip_dirs)
        res.clip = str(clip) if clip else None
        if clip is None:
            res.missing.append(f"{_profiles.CLIP_FILE} (from https://huggingface.co/{profile.base_repo})")
    return res
