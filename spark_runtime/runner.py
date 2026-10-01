"""High-level orchestration shared by the ComfyUI nodes and the CLI scripts."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import platform
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from . import metadata as M
from . import paths as P
from .command import GenerationRequest, RuntimeConfig, prepare_run
from .process import SparkProcessError, run_process, wsl_kill_hook
from .profiles import Profile

LOG = logging.getLogger("ComfyUI-SparkDiffusion")

_SUGGESTIONS = [
    (
        "out of memory",
        "CUDA ran out of memory: close other GPU applications, use quantization=fp8, or pick a 480P / 1.3B profile",
    ),
    (
        "outofmemoryerror",
        "CUDA ran out of memory: close other GPU applications, use quantization=fp8, or pick a 480P / 1.3B profile",
    ),
    (
        "row-wise",
        "this PyTorch build lacks row-wise FP8: use backend=wsl2/linux, fp8_compat=auto, or quantization=bf16",
    ),
    ("rowwise scaling is not currently supported", "use backend=wsl2/linux or fp8_compat=auto"),
    ("no module named", "install the SparkDiffusion requirements into the runtime interpreter (see README)"),
    ("cuda is not available", "the runtime interpreter has no CUDA-enabled PyTorch; reinstall torch with CUDA"),
    ("sla_src", "upstream needs the external thu-ml/SLA kernel for this mode: set sla_src or enable fused kernels"),
    ("no such file or directory", "check the paths on the Runtime node (repo, python, model root)"),
    ("there is no distribution", "the WSL distro name is wrong; run `wsl -l -v` to list distros"),
    ("rola parameters are missing", "the checkpoint does not match the profile; re-download it from Hugging Face"),
    ("verification failed", "the run did not use the SparkDiffusion path it was configured for; see metadata"),
]


def suggest(log_tail: List[str], error: Optional[str]) -> str:
    text = ((error or "") + "\n" + "\n".join(log_tail)).lower()
    for needle, hint in _SUGGESTIONS:
        if needle in text:
            return hint
    return "run `python scripts/doctor.py` to check the runtime environment"


def _write_json(path: str, data: Dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


@dataclass
class RunResult:
    video_paths: List[str]
    metadata_path: str
    metadata: Dict
    summary: str


def generate(
    cfg: RuntimeConfig,
    profile: Profile,
    req: GenerationRequest,
    output_dir: str,
    *,
    filename_prefix: str = "sparkdiffusion",
    work_root: Optional[str] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    on_line: Optional[Callable[[str, str], None]] = None,
    strict_checks: bool = True,
) -> RunResult:
    t0 = time.time()
    if cfg.resolved_backend() == P.BACKEND_WSL2 and cfg.wsl_automount_root == P.DEFAULT_WSL_AUTOMOUNT_ROOT:
        cfg = dataclasses.replace(cfg, wsl_automount_root=P.detect_automount_root(cfg.wsl_distro or None))
    work_dir = tempfile.mkdtemp(prefix="sparkdiffusion-job-", dir=work_root)
    try:
        run = prepare_run(
            cfg, profile, req, output_dir, work_dir, filename_prefix=filename_prefix, strict_checks=strict_checks
        )
        extra_kill = wsl_kill_hook(cfg.wsl_distro, run.pid_file) if run.backend == P.BACKEND_WSL2 else None
        LOG.info(
            "[SparkDiffusion] %s | backend=%s quant=%s fused=%s compile=%s",
            profile.label,
            run.backend,
            cfg.quantization,
            cfg.fused_kernels,
            cfg.torch_compile,
        )
        result = run_process(
            run.command,
            cwd=run.cwd,
            env=run.env,
            timeout_s=cfg.timeout_s,
            cancel_check=cancel_check,
            cancel_file=run.cancel_file,
            extra_kill=extra_kill,
            on_event=on_event,
            on_line=on_line,
            backend=run.backend,
            profile=profile.key,
        )
        meta = M.load_metadata(run.metadata_path) or {}
        if result.exit_code != 0:
            error = meta.get("error") or (result.last_event("result") or {}).get("error")
            if meta:  # keep the failure record next to the outputs, without absolute paths
                _write_json(run.metadata_path, M.redact(meta))
            raise SparkProcessError(
                M.redact(error) if error else f"runtime exited with code {result.exit_code}",
                exit_code=result.exit_code,
                command=run.command,
                log_tail=result.error_tail(25),
                backend=run.backend,
                profile=profile.key,
                suggestion=suggest(result.log_tail + result.stderr_tail, error),
            )

        names = meta.get("outputs") or []
        video_paths = [os.path.join(output_dir, n) for n in names]
        if not video_paths:
            raise SparkProcessError(
                "runtime reported success but produced no video",
                exit_code=result.exit_code,
                command=run.command,
                log_tail=result.log_tail[-25:],
                backend=run.backend,
                profile=profile.key,
            )
        expected_size = profile.size(req.aspect_ratio) if profile.task == "t2v" else None
        videos, video_problems = [], []
        for vp in video_paths:
            info = M.probe_video(vp)
            videos.append(info)
            video_problems += [f"{info['file']}: {p}" for p in M.check_video(info, req.num_frames, expected_size)]

        meta.update(
            {
                "profile": profile.as_dict(),
                "runtime": cfg.as_public_dict(),
                "request": {
                    "prompt": req.prompt,
                    "seed": req.seed,
                    "num_frames": req.num_frames,
                    "aspect_ratio": req.aspect_ratio,
                    "num_samples": req.num_samples,
                    "topk_ratio": req.topk_ratio if req.topk_ratio is not None else profile.topk_ratio,
                    "input_image": os.path.basename(req.image_path) if req.image_path else None,
                },
                "videos": videos,
                "video_problems": video_problems,
                "host": {
                    "os": platform.system(),
                    "os_release": platform.release(),
                    "python": platform.python_version(),
                    "comfyui_sparkdiffusion_version": M.package_version(),
                    "comfyui_sparkdiffusion_sha": M.git_sha(),
                    "wall_s": round(time.time() - t0, 2),
                    "runtime_process_s": round(result.duration_s, 2),
                },
            }
        )
        meta = M.redact(meta)
        _write_json(run.metadata_path, meta)
        if video_problems and strict_checks:
            raise SparkProcessError(
                "output video failed validation: " + "; ".join(video_problems),
                exit_code=0,
                command=run.command,
                log_tail=result.log_tail[-10:],
                backend=run.backend,
                profile=profile.key,
            )
        summary = M.summarize(meta)
        LOG.info("[SparkDiffusion] done\n%s", summary)
        return RunResult(video_paths=video_paths, metadata_path=run.metadata_path, metadata=meta, summary=summary)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
