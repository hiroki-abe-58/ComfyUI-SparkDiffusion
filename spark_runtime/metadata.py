"""Run metadata: load the launcher's JSON, enrich it, validate the video, redact paths."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional

from . import paths as P

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def package_version() -> str:
    try:
        text = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return m.group(1) if m else "unknown"


def git_sha(repo: Path = PACKAGE_ROOT) -> Optional[str]:
    """HEAD of a git checkout without calling git (same logic as the launcher)."""
    git_dir = repo / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(" ", 1)[1].strip()
        ref_file = git_dir.joinpath(*ref.split("/"))
        if ref_file.is_file():
            return ref_file.read_text(encoding="utf-8").strip()
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split(" ")
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        return None
    return None


def load_metadata(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def redact(obj: Any) -> Any:
    """Recursively replace absolute paths inside strings with ``<path:basename>``."""
    if isinstance(obj, str):
        return P.redact_paths(obj)
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    return obj


def probe_video(path: str) -> Dict[str, Any]:
    """Codec / size / fps / frame count via ffprobe, falling back to PyAV."""
    info: Dict[str, Any] = {"file": os.path.basename(path), "exists": os.path.isfile(path)}
    if not info["exists"]:
        return info
    info["size_bytes"] = os.path.getsize(path)
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            out = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration",
                    "-of",
                    "json",
                    path,
                ],
                capture_output=True,
                timeout=120,
                check=True,
            )
            stream = json.loads(out.stdout.decode("utf-8", "replace"))["streams"][0]
            info.update(
                {
                    "probe": "ffprobe",
                    "codec": stream.get("codec_name"),
                    "width": int(stream["width"]),
                    "height": int(stream["height"]),
                    "fps": float(Fraction(stream.get("r_frame_rate", "0/1"))),
                    "frames": int(stream.get("nb_read_frames", 0)),
                    "duration_s": float(stream.get("duration", 0.0)),
                }
            )
            return info
        except Exception as exc:  # fall through to PyAV
            info["ffprobe_error"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        import av  # bundled with ComfyUI

        with av.open(path) as container:
            stream = container.streams.video[0]
            frames = sum(1 for _ in container.decode(stream))
            rate = float(stream.average_rate or 0)
            info.update(
                {
                    "probe": "pyav",
                    "codec": stream.codec_context.name,
                    "width": stream.codec_context.width,
                    "height": stream.codec_context.height,
                    "fps": rate,
                    "frames": frames,
                    "duration_s": round(frames / rate, 6) if rate else None,
                }
            )
    except Exception as exc:
        info["probe_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return info


def check_video(
    info: Dict[str, Any], expected_frames: int, expected_size: Optional[tuple] = None, fps: float = 16.0
) -> list:
    problems = []
    if not info.get("exists"):
        return ["output video was not written"]
    if info.get("size_bytes", 0) < 10_000:
        problems.append(f"output video is suspiciously small ({info.get('size_bytes')} bytes)")
    if "frames" in info and info["frames"] != expected_frames:
        problems.append(f"expected {expected_frames} frames, got {info['frames']}")
    if expected_size and "width" in info and (info["width"], info["height"]) != tuple(expected_size):
        problems.append(f"expected {expected_size[0]}x{expected_size[1]}, got {info['width']}x{info['height']}")
    if "fps" in info and abs(info["fps"] - fps) > 0.01:
        problems.append(f"expected {fps} fps, got {info['fps']}")
    return problems


def summarize(meta: Dict[str, Any]) -> str:
    """Compact multi-line summary for the node's metadata output and the console."""
    c = meta.get("checks", {})
    env = meta.get("environment", {})
    label = meta.get("profile", {}).get("label", "?")
    backend = meta.get("runtime", {}).get("backend", "?")
    lines = [
        f"SparkDiffusion | {label} | backend {backend}",
        f"RoLA active: {c.get('rola_active')} ({c.get('rola_layers')} layers) | top-k {c.get('topk_ratio')} "
        f"(sparsity {c.get('sparsity')})",
        f"CrossDistill: {c.get('crossdistill_active')} | timesteps {c.get('schedule_observed_net_timesteps')}",
        f"FP8 active: {c.get('fp8_active')} ({c.get('fp8_modules')} modules, {c.get('fp8_gemm_backend')})",
        f"fused kernels: {c.get('fused_kernels_active')} | torch.compile: {c.get('torch_compile_active')}",
        f"{env.get('gpu')} {env.get('compute_capability')} | torch {env.get('torch')} | CUDA {env.get('cuda_runtime')} "
        f"| triton {env.get('triton')}",
    ]
    for s in meta.get("samples", []):
        lines.append(f"sample {s['index']} ({s['label']}): denoise {s['denoise_s']}s, VAE decode {s['vae_decode_s']}s")
    mem = meta.get("memory", {})
    if mem:
        lines.append(f"peak VRAM (allocated): {mem.get('peak_allocated_gib')} GiB")
    host = meta.get("host", {})
    if host.get("wall_s") is not None:
        lines.append(f"total wall time (incl. load/compile): {host['wall_s']}s")
    return "\n".join(lines)
