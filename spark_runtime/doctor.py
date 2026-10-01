"""Environment diagnosis (PASS / WARN / FAIL), shared by scripts/doctor.py and the Doctor node."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, List, Optional

from . import paths as P
from .command import LAUNCHER_PATH, RuntimeConfig, child_env, host_view_of, runtime_path
from .profiles import PROFILES
from .process import run_process

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"


@dataclass
class Check:
    status: str
    item: str
    detail: str

    def line(self) -> str:
        return f"[{self.status:4s}] {self.item:28s} {self.detail}"


def list_wsl_distros(timeout: float = 20.0) -> Optional[List[str]]:
    if os.name != "nt" or not shutil.which("wsl.exe"):
        return None
    try:
        out = subprocess.run(
            [P.system_executable("wsl.exe"), "-l", "-q"], capture_output=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    raw = out.stdout
    # wsl.exe writes UTF-16LE to pipes
    text = raw.decode("utf-16-le", "replace") if b"\x00" in raw else raw.decode("utf-8", "replace")
    return [d.strip().strip("\x00") for d in text.splitlines() if d.strip().strip("\x00")]


def _version_tuple(v: str) -> tuple:
    out = []
    for part in v.split("+")[0].split("."):
        num = "".join(ch for ch in part if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out)


def probe_runtime(cfg: RuntimeConfig, compile_smoke: bool = True, timeout_s: float = 600) -> dict:
    backend = cfg.resolved_backend()
    repo_rt = (
        cfg.repo if (backend == P.BACKEND_WSL2 and cfg.repo.startswith("/")) else runtime_path(cfg.repo, cfg, backend)
    )
    args = ["-X", "utf8", runtime_path(str(LAUNCHER_PATH), cfg, backend), "--probe", "--repo", repo_rt]
    if compile_smoke:
        args.append("--probe-compile")
    if backend == P.BACKEND_WSL2:
        cmd = (
            [P.system_executable("wsl.exe")]
            + (["-d", cfg.wsl_distro] if cfg.wsl_distro else [])
            + ["--cd", "/", "--exec", cfg.python, *args]
        )
        env = None
    else:
        cmd = [cfg.python, *args]
        env = child_env(os.environ)
    captured: dict = {}

    def on_event(ev: dict) -> None:
        if ev.get("event") == "probe":
            captured.update(ev)

    result = run_process(cmd, env=env, timeout_s=timeout_s, on_event=on_event, on_line=lambda *_: None)
    if result.exit_code != 0 and not captured:
        captured["error"] = f"probe exited with code {result.exit_code}: " + " | ".join(result.log_tail[-5:])
    return captured


def run_doctor(
    cfg: RuntimeConfig,
    comfyui_path: Optional[str] = None,
    compile_smoke: bool = True,
    emit: Optional[Callable[[Check], None]] = None,
) -> List[Check]:
    checks: List[Check] = []

    def add(status: str, item: str, detail: str) -> None:
        c = Check(status, item, detail)
        checks.append(c)
        if emit:
            emit(c)

    add(INFO, "OS", f"{platform.system()} {platform.release()} ({platform.version()[:40]})")
    add(INFO, "Architecture", platform.machine())
    add(INFO, "Host Python", platform.python_version())
    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        add(
            PASS if found else WARN,
            tool,
            "found" if found else "not on PATH (only used for output validation; PyAV fallback)",
        )
    if comfyui_path:
        ok = os.path.isfile(os.path.join(comfyui_path, "main.py"))
        add(PASS if ok else WARN, "ComfyUI path", "main.py found" if ok else "main.py not found")
    else:
        add(INFO, "ComfyUI path", "not given (pass --comfyui to check it)")

    try:
        backend = cfg.resolved_backend()
    except ValueError as exc:
        add(FAIL, "Backend", str(exc))
        return checks
    add(INFO, "Backend", f"{cfg.backend} -> {backend}")
    distros = list_wsl_distros()
    if distros is None:
        add(INFO if backend != P.BACKEND_WSL2 else FAIL, "WSL", "not available on this host")
    else:
        detail = "distros: " + (", ".join(distros) or "<none>")
        status = PASS
        if backend == P.BACKEND_WSL2 and cfg.wsl_distro and cfg.wsl_distro not in distros:
            status, detail = FAIL, f"configured distro {cfg.wsl_distro!r} not installed ({detail})"
        add(status, "WSL", detail)

    if not cfg.repo or not cfg.python:
        add(FAIL, "Runtime config", "repo and python must be set (Runtime node, env vars or config file)")
        return checks

    repo_host = host_view_of(cfg.repo, cfg, backend)
    repo_ok = os.path.isfile(os.path.join(repo_host, "sparkdiffusion", "inference", "wan2pt1_t2v_distilled_infer.py"))
    add(
        PASS if repo_ok else FAIL,
        "SparkDiffusion path",
        "checkout found" if repo_ok else "inference entry points not found",
    )

    probe = probe_runtime(cfg, compile_smoke=compile_smoke)
    if "error" in probe or "torch" not in probe:
        add(FAIL, "Runtime interpreter", probe.get("error") or probe.get("torch_error") or "probe failed")
        return checks
    add(PASS, "Runtime Python", probe.get("python", "?"))
    add(PASS, "PyTorch", probe["torch"])
    add(PASS if probe.get("cuda_runtime") else FAIL, "CUDA runtime", str(probe.get("cuda_runtime")))
    add(PASS if probe.get("cuda_available") else FAIL, "CUDA available", str(probe.get("cuda_available")))
    if probe.get("cuda_available"):
        cc = probe.get("compute_capability") or [0, 0]
        add(PASS, "GPU", f"{probe.get('gpu')}")
        add(PASS if tuple(cc) >= (8, 0) else WARN, "Compute capability", f"sm_{cc[0]}{cc[1]}")
        vram = probe.get("vram_total_gib", 0)
        add(
            PASS if vram >= 24 else WARN,
            "VRAM",
            f"{vram} GiB" + ("" if vram >= 24 else " (14B profiles need ~24+ GiB with FP8)"),
        )
        smm = probe.get("scaled_mm") or {}
        if smm.get("rowwise"):
            add(PASS, "FP8 W8A8 (row-wise)", "torch._scaled_mm row-wise supported (upstream FP8 path)")
        elif smm.get("tensorwise"):
            add(
                WARN,
                "FP8 W8A8 (row-wise)",
                "row-wise _scaled_mm missing in this PyTorch build; "
                "fp8_compat=auto will emulate it (experimental) - WSL2/Linux runs the upstream path",
            )
        else:
            add(WARN, "FP8", "no FP8 _scaled_mm support: use quantization=bf16")
    triton = probe.get("triton")
    if triton:
        cc = probe.get("compute_capability") or [0, 0]
        too_old = cc and cc[0] == 12 and _version_tuple(triton) < (3, 4)
        add(
            WARN if too_old else PASS,
            "Triton",
            triton + (" (RTX 50-series FP8 needs Triton >= 3.4)" if too_old else ""),
        )
    else:
        add(FAIL, "Triton", "not importable: " + str(probe.get("triton_error", "")))
    add(
        PASS if probe.get("flash_attn") else INFO,
        "flash-attn",
        probe.get("flash_attn") or "not installed (optional; upstream falls back to PyTorch kernels)",
    )
    add(
        PASS if probe.get("fused_kernels_import") else FAIL,
        "Fused kernels",
        "sparkdiffusion.ops.fused_kernel imports"
        if probe.get("fused_kernels_import")
        else probe.get("fused_kernels_error", "?"),
    )
    if compile_smoke:
        add(
            PASS if probe.get("torch_compile_smoke") else WARN,
            "torch.compile",
            "smoke test OK" if probe.get("torch_compile_smoke") else probe.get("torch_compile_error", "failed"),
        )
    add(INFO, "SparkDiffusion SHA", str(probe.get("sparkdiffusion_sha")))
    add(INFO, "NVML telemetry", "available" if probe.get("pynvml") else "pynvml missing (GPU util/power not recorded)")

    if cfg.model_root:
        from .paths import resolve_assets

        root_host = host_view_of(cfg.model_root, cfg, backend)
        for profile in PROFILES.values():
            res = resolve_assets(root_host, profile)
            if res.ok:
                add(PASS, f"Assets {profile.key}", "checkpoint + Wan assets found")
            else:
                add(
                    INFO,
                    f"Assets {profile.key}",
                    "missing: " + "; ".join(P.basename_any(m.split(" (")[0]) for m in res.missing),
                )
    else:
        add(WARN, "Model root", "not set")
    return checks


def format_report(checks: List[Check]) -> str:
    worst = FAIL if any(c.status == FAIL for c in checks) else WARN if any(c.status == WARN for c in checks) else PASS
    return "\n".join(c.line() for c in checks) + f"\n\nOverall: {worst}"
