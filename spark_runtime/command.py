"""Build the job file and the argument-list command for one SparkDiffusion run.

Nothing here touches the GPU or imports torch. The host never formats a shell
command line: commands are argument lists, and the prompt travels inside a
UTF-8 job file rather than on the command line.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..schedules import crossdistill
from . import paths as P
from .profiles import Profile, validate_num_frames

JOB_SCHEMA = "comfyui-sparkdiffusion/job/v1"
QUANT_MODES = ("fp8", "bf16")
FP8_COMPAT_MODES = ("auto", "off", "force")

LAUNCHER_PATH = Path(__file__).resolve().with_name("launcher.py")


class ConfigError(ValueError):
    """Invalid user configuration detected before anything is launched."""


@dataclass
class RuntimeConfig:
    backend: str = P.BACKEND_AUTO
    repo: str = ""
    python: str = ""
    model_root: str = ""
    wsl_distro: str = ""
    wsl_automount_root: str = P.DEFAULT_WSL_AUTOMOUNT_ROOT
    quantization: str = "fp8"
    fp8_compat: str = "auto"
    fused_kernels: bool = True
    torch_compile: bool = True
    sla_src: str = ""
    timeout_s: int = 3600
    extra_env: Dict[str, str] = field(default_factory=dict)

    def resolved_backend(self) -> str:
        if self.backend not in P.BACKENDS:
            raise ConfigError(f"unknown backend {self.backend!r}; choose one of {list(P.BACKENDS)}")
        return P.detect_backend() if self.backend == P.BACKEND_AUTO else self.backend

    def validate(self) -> None:
        backend = self.resolved_backend()
        if self.quantization not in QUANT_MODES:
            raise ConfigError(f"quantization must be one of {QUANT_MODES}, got {self.quantization!r}")
        if self.fp8_compat not in FP8_COMPAT_MODES:
            raise ConfigError(f"fp8_compat must be one of {FP8_COMPAT_MODES}, got {self.fp8_compat!r}")
        if not self.repo:
            raise ConfigError(
                "SparkDiffusion repository path is empty (set it on the Runtime node or via SPARKDIFFUSION_REPO)"
            )
        if not self.python:
            raise ConfigError(
                "runtime Python interpreter is empty (set it on the Runtime node or via SPARKDIFFUSION_PYTHON)"
            )
        if not self.model_root:
            raise ConfigError("model root is empty (set it on the Runtime node or via SPARKDIFFUSION_MODEL_ROOT)")
        if backend == P.BACKEND_WSL2 and os.name != "nt":
            raise ConfigError("backend 'wsl2' is only meaningful on a Windows host")
        if backend == P.BACKEND_WINDOWS and os.name != "nt":
            raise ConfigError("backend 'windows-native' requires a Windows host")
        if self.timeout_s < 0:
            raise ConfigError("timeout must be >= 0 (0 disables it)")

    def as_public_dict(self) -> dict:
        """Settings safe to expose in metadata: no paths, no environment values."""
        return {
            "backend": self.resolved_backend(),
            "quantization": self.quantization,
            "fp8_compat": self.fp8_compat,
            "fused_kernels": self.fused_kernels,
            "torch_compile": self.torch_compile,
            "sla_src_set": bool(self.sla_src),
            "wsl_distro": self.wsl_distro or None,
            "timeout_s": self.timeout_s,
        }


@dataclass
class GenerationRequest:
    prompt: str
    seed: int = 0
    num_frames: int = 81
    aspect_ratio: str = "16:9"
    num_samples: int = 1
    topk_ratio: Optional[float] = None  # None -> profile default
    image_path: Optional[str] = None  # I2V only (host path)
    fixed_resolution: bool = False  # I2V only


@dataclass
class PreparedRun:
    command: List[str]
    job: dict
    job_path: str
    metadata_path: str
    video_path: str
    cancel_file: str
    pid_file: str
    backend: str
    env: Optional[Dict[str, str]]
    cwd: Optional[str]


def host_view_of(path: str, cfg: RuntimeConfig, backend: str) -> str:
    """Path the *host* can stat for a path given in runtime terms.

    For WSL2, a POSIX path such as ``/opt/models`` is reachable from Windows as
    ``\\\\wsl.localhost\\<distro>\\opt\\models``.
    """
    if backend == P.BACKEND_WSL2 and path.startswith("/"):
        root = cfg.wsl_automount_root if cfg.wsl_automount_root.endswith("/") else cfg.wsl_automount_root + "/"
        if path.startswith(root):
            return P.wsl_to_windows(path, cfg.wsl_automount_root)
        if not cfg.wsl_distro:
            return path
        return "\\\\wsl.localhost\\" + cfg.wsl_distro + path.replace("/", "\\")
    return path


def runtime_path(path: str, cfg: RuntimeConfig, backend: str) -> str:
    return P.to_runtime_path(path, backend, cfg.wsl_automount_root, cfg.wsl_distro or None)


def build_upstream_argv(
    profile: Profile, req: GenerationRequest, assets: Dict[str, object], save_path: str, cfg: RuntimeConfig
) -> List[str]:
    """Arguments for the official entry point, in runtime path terms."""
    topk = profile.topk_ratio if req.topk_ratio is None else profile.validate_topk(float(req.topk_ratio))
    argv: List[str] = ["--model_size", profile.upstream_model_size]
    if profile.family == "wan2.2":
        high_n, low_n = crossdistill.wan22_split_steps(profile.num_steps)
        ckpts = list(assets["checkpoints"])  # type: ignore[arg-type]
        argv += [
            "--high_noise_model_path",
            ckpts[0],
            "--low_noise_model_path",
            ckpts[-1],
            "--num_steps_high",
            str(high_n),
            "--num_steps_low",
            str(low_n),
        ]
    else:
        argv += ["--num_steps", str(profile.num_steps), "--dit_path", list(assets["checkpoints"])[0]]  # type: ignore[arg-type]
    argv += [
        "--sigma_max",
        repr(crossdistill.SIGMA_MAX_DEFAULT),
        "--rola_topk_ratio",
        repr(float(topk)),
        "--vae_path",
        str(assets["vae"]),
        "--text_encoder_path",
        str(assets["t5"]),
        "--tokenizer_path",
        str(assets["tokenizer"]),
        "--num_frames",
        str(validate_num_frames(int(req.num_frames))),
        "--resolution",
        profile.resolution,
        "--aspect_ratio",
        req.aspect_ratio,
        "--seed",
        str(int(req.seed)),
        "--num_samples",
        str(int(req.num_samples)),
        "--attn_precision",
        "bf16",
        "--save_path",
        save_path,
    ]
    if profile.task == "i2v":
        if not req.image_path:
            raise ConfigError(f"{profile.label} is an image-to-video profile and needs an input image")
        argv += ["--image_path", req.image_path, "--clip_encoder_path", str(assets["clip"])]
        if req.fixed_resolution:
            argv.append("--fixed_resolution")
    elif req.image_path:
        raise ConfigError(f"{profile.label} is a text-to-video profile; use an I2V profile for image input")
    if cfg.quantization == "fp8":
        argv += ["--quant_type", "fp8", "--quant_mode", "w8a8"]
    if not cfg.fused_kernels:
        argv.append("--disable_fused_kernels")
    # ``--prompt=<text>`` keeps prompts that start with "-" from being parsed as options.
    argv.append("--prompt=" + req.prompt)
    return argv


_SAFE_STEM = re.compile(r"[^\w\-. ]+", re.UNICODE)


def safe_stem(prefix: str) -> str:
    stem = _SAFE_STEM.sub("_", prefix).strip(" .") or "sparkdiffusion"
    return stem[:80]


def next_output_path(output_dir: str, prefix: str) -> str:
    """``<dir>/<prefix>_00001.mp4`` with the first unused counter."""
    os.makedirs(output_dir, exist_ok=True)
    stem = safe_stem(prefix)
    pattern = re.compile(re.escape(stem) + r"_(\d{5})(?:_sample_\d+_seed_\d+)?\.(?:mp4|json)$")
    used = [int(m.group(1)) for name in os.listdir(output_dir) if (m := pattern.match(name))]
    counter = max(used, default=0) + 1
    return os.path.join(output_dir, f"{stem}_{counter:05d}.mp4")


def prepare_run(
    cfg: RuntimeConfig,
    profile: Profile,
    req: GenerationRequest,
    output_dir: str,
    work_dir: str,
    filename_prefix: str = "sparkdiffusion",
    assets: Optional[P.ResolvedAssets] = None,
    strict_checks: bool = True,
    telemetry: bool = True,
) -> PreparedRun:
    """Validate everything and write the job file; return the command to run.

    ``output_dir`` and ``work_dir`` are host paths. ``work_dir`` must be
    reachable by the runtime (for WSL2 any local Windows drive is).
    """
    cfg.validate()
    backend = cfg.resolved_backend()
    if not req.prompt or not req.prompt.strip():
        raise ConfigError("prompt is empty")
    if req.aspect_ratio not in ("16:9", "9:16", "1:1", "4:3", "3:4"):
        raise ConfigError(f"unsupported aspect ratio {req.aspect_ratio!r}")
    if req.num_samples < 1:
        raise ConfigError("num_samples must be >= 1")
    if not cfg.fused_kernels and profile.upstream_model_size.endswith("_rola") and not cfg.sla_src:
        raise ConfigError(
            "fused kernels are disabled, but upstream's non-fused RoLA attention needs the external "
            "thu-ml/SLA kernel: set 'sla_src' to an SLA checkout or re-enable fused kernels"
        )

    if assets is None:
        assets = P.resolve_assets(host_view_of(cfg.model_root, cfg, backend), profile)
    if not assets.ok:
        raise ConfigError(
            f"missing model files for {profile.label} under the model root:\n  - "
            + "\n  - ".join(assets.missing)
            + "\nSee README 'Model placement' or run: python scripts/download_models.py --profile "
            + profile.key
        )

    video_path = next_output_path(output_dir, filename_prefix)
    metadata_path = os.path.splitext(video_path)[0] + ".json"
    os.makedirs(work_dir, exist_ok=True)
    job_path = os.path.join(work_dir, "job.json")
    cancel_file = os.path.join(work_dir, "CANCEL")
    pid_file = os.path.join(work_dir, "launcher.pid")

    rt = lambda p: runtime_path(p, cfg, backend)  # noqa: E731
    asset_rt: Dict[str, object] = {
        "checkpoints": [rt(c) for c in assets.checkpoints],
        "vae": rt(assets.vae or ""),
        "t5": rt(assets.t5 or ""),
        "tokenizer": rt(assets.tokenizer or ""),
        "clip": rt(assets.clip) if assets.clip else None,
    }
    req_rt = GenerationRequest(**{**req.__dict__})
    if req.image_path:
        if not os.path.isfile(req.image_path):
            raise ConfigError(f"input image not found: {req.image_path}")
        req_rt.image_path = rt(req.image_path)
    argv = build_upstream_argv(profile, req_rt, asset_rt, rt(video_path), cfg)
    topk = profile.topk_ratio if req.topk_ratio is None else float(req.topk_ratio)

    repo_rt = cfg.repo if (backend == P.BACKEND_WSL2 and cfg.repo.startswith("/")) else rt(cfg.repo)
    job = {
        "schema": JOB_SCHEMA,
        "repo": repo_rt,
        "entrypoint": profile.entrypoint,
        "family": profile.family,
        "task": profile.task,
        "profile_key": profile.key,
        "argv": argv,
        "quantization": cfg.quantization,
        "fp8_compat": cfg.fp8_compat,
        "torch_compile": cfg.torch_compile,
        "fused_kernels": cfg.fused_kernels,
        "expect_rola": profile.upstream_model_size.endswith("_rola"),
        "topk_ratio": topk,
        "expected_timesteps": list(profile.timesteps()),
        "t_scaling_factor": crossdistill.T_SCALING_FACTOR,
        "steps_per_sample": profile.num_steps,
        "num_samples": req.num_samples,
        "seed": req.seed,
        "sla_src": (cfg.sla_src if cfg.sla_src.startswith("/") else rt(cfg.sla_src)) if cfg.sla_src else None,
        "metadata_path": rt(metadata_path),
        "cancel_file": rt(cancel_file),
        "pid_file": rt(pid_file),
        "strict_checks": strict_checks,
        "telemetry": telemetry,
        "prewarm_einops": True,
        "allocator_tuning": True,
        "env": dict(cfg.extra_env),
    }
    with open(job_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(job, f, ensure_ascii=False, indent=1)

    launcher_rt = rt(str(LAUNCHER_PATH))
    py_args = ["-X", "utf8", launcher_rt, "--job", rt(job_path)]
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None
    if backend == P.BACKEND_WSL2:
        command = ["wsl.exe"]
        if cfg.wsl_distro:
            command += ["-d", cfg.wsl_distro]
        command += ["--cd", "/", "--exec", cfg.python, *py_args]
    else:
        command = [cfg.python, *py_args]
        env = child_env(os.environ)
        cwd = os.path.dirname(os.path.abspath(job_path))
    return PreparedRun(
        command=command,
        job=job,
        job_path=job_path,
        metadata_path=metadata_path,
        video_path=video_path,
        cancel_file=cancel_file,
        pid_file=pid_file,
        backend=backend,
        env=env,
        cwd=cwd,
    )


_STRIP_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONNOUSERSITE")


def child_env(base: Dict[str, str]) -> Dict[str, str]:
    """Environment for a host-native runtime: ComfyUI's Python settings must not leak in."""
    env = {k: v for k, v in base.items() if k.upper() not in _STRIP_ENV}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


_SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret|password)\s*[=:]\s*)\S+"),
]


def mask_secrets(text: str) -> str:
    for pat in _SECRET_PATTERNS:
        text = pat.sub(lambda m: (m.group(1) if m.lastindex else "") + "***", text)
    return text


def describe_command(command: Sequence[str]) -> str:
    """Short, redacted rendering of a command for error messages and logs."""
    shown = [P.basename_any(command[0])]
    for arg in command[1:]:
        shown.append(P.redact_paths(arg) if (os.sep in arg or "/" in arg) else arg)
    return mask_secrets(" ".join(shown))
