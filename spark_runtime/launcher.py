"""In-runtime launcher for the official SparkDiffusion inference entry points.

This file is executed by the *SparkDiffusion runtime* interpreter (a separate
venv, possibly inside WSL2), never imported by ComfyUI. It must therefore stay
self-contained: standard library + torch + the upstream repository only.

What it does:

1. Reads a UTF-8 job file written by ComfyUI-SparkDiffusion (prompt included,
   so no prompt ever travels through an OS command line).
2. Instruments, without modifying upstream source, the functions the official
   entry point imports: checkpoint loading, ``optimize_model_for_inference``,
   umT5 encoding and VAE decoding. This lets it *verify* rather than assume
   that RoLA sparse attention, the CrossDistill schedule, FP8 W8A8 and the fused
   kernels are active, and to time each phase.
3. Runs the unmodified upstream entry point with ``runpy`` and ``sys.argv`` set
   to the argument list prepared by the host.
4. Writes a metadata JSON (no absolute paths) and emits machine-readable
   ``@@SPARK@@ {json}`` event lines on stdout for progress reporting.

Usage::

    python launcher.py --job job.json
    python launcher.py --probe --repo /path/to/SparkDiffusion [--probe-compile]
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import runpy
import sys
import threading
import time
import traceback

EVENT_PREFIX = "@@SPARK@@ "
JOB_SCHEMA = "comfyui-sparkdiffusion/job/v1"
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_CANCELLED = 130

_ROLA_PARAM_MARKERS = ("proj_q", "proj_k", "gate_proj", "gate_bias")
_FP8_CLASSES = ("FP8Linear", "FP8LinearFusedGELU")


def _setup_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass


def emit(event: str, **payload) -> None:
    payload["event"] = event
    payload["t"] = round(time.time(), 3)
    sys.stdout.write(EVENT_PREFIX + json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    sys.stdout.write(f"[spark-launcher] {msg}\n")
    sys.stdout.flush()


def _basename(path) -> str:
    return os.path.basename(str(path).rstrip("/\\")) if path else path


def git_sha(repo: str) -> str | None:
    """Read HEAD without invoking git (git may be absent in the runtime)."""
    git_dir = os.path.join(repo, ".git")
    try:
        with open(os.path.join(git_dir, "HEAD"), encoding="utf-8") as f:
            head = f.read().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(" ", 1)[1].strip()
        ref_path = os.path.join(git_dir, *ref.split("/"))
        if os.path.isfile(ref_path):
            with open(ref_path, encoding="utf-8") as f:
                return f.read().strip()
        packed = os.path.join(git_dir, "packed-refs")
        if os.path.isfile(packed):
            with open(packed, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ")
                    if len(parts) == 2 and parts[1] == ref:
                        return parts[0]
    except OSError:
        return None
    return None


# --------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------


class CancelWatchdog(threading.Thread):
    """Exit promptly when the host creates the cancel file.

    Covers the WSL2 case where killing ``wsl.exe`` on the Windows side does not
    reliably terminate the Linux process tree.
    """

    def __init__(self, cancel_file: str | None, poll: float = 0.5):
        super().__init__(name="spark-cancel-watchdog", daemon=True)
        self.cancel_file = cancel_file
        self.poll = poll

    def run(self) -> None:
        if not self.cancel_file:
            return
        while True:
            if os.path.exists(self.cancel_file):
                emit("cancelled", reason="cancel file detected")
                sys.stdout.flush()
                sys.stderr.flush()
                if hasattr(os, "killpg"):
                    try:
                        import signal

                        os.killpg(os.getpgrp(), signal.SIGKILL)
                    except Exception:
                        pass
                os._exit(EXIT_CANCELLED)
            time.sleep(self.poll)


# --------------------------------------------------------------------------
# GPU telemetry (optional, NVML)
# --------------------------------------------------------------------------


class Telemetry(threading.Thread):
    def __init__(self, interval: float = 0.5):
        super().__init__(name="spark-telemetry", daemon=True)
        self.interval = interval
        self.samples: list[tuple[float, float, float, float, float]] = []  # t, util, power_w, temp_c, mem_used_mib
        self._halt = threading.Event()
        self.available = False
        self.error: str | None = None

    def run(self) -> None:
        try:
            import pynvml  # provided by nvidia-ml-py

            pynvml.nvmlInit()
            idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or 0)
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            self.available = True
        except Exception as exc:  # telemetry is best-effort
            self.error = f"{type(exc).__name__}: {exc}"
            return
        while not self._halt.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20
                self.samples.append((time.time(), float(util), float(power), float(temp), float(mem)))
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()

    def summarize(self, start: float | None = None, end: float | None = None) -> dict | None:
        rows = [s for s in self.samples if (start is None or s[0] >= start) and (end is None or s[0] <= end)]
        if not rows:
            return None
        n = len(rows)
        return {
            "samples": n,
            "gpu_util_avg_pct": round(sum(r[1] for r in rows) / n, 1),
            "gpu_util_max_pct": round(max(r[1] for r in rows), 1),
            "power_avg_w": round(sum(r[2] for r in rows) / n, 1),
            "power_max_w": round(max(r[2] for r in rows), 1),
            "temp_max_c": round(max(r[3] for r in rows), 1),
            "nvml_mem_used_max_mib": round(max(r[4] for r in rows), 0),
        }


# --------------------------------------------------------------------------
# FP8 compatibility shim (PyTorch builds without row-wise _scaled_mm)
# --------------------------------------------------------------------------


def probe_scaled_mm(torch) -> dict:
    d = torch.device("cuda", torch.cuda.current_device())
    out = {}
    a = torch.zeros((16, 64), device=d, dtype=torch.float8_e4m3fn)
    b = torch.zeros((64, 64), device=d, dtype=torch.float8_e4m3fn).t()
    for name, sa, sb in (
        ("rowwise", torch.ones((16, 1), device=d), torch.ones((1, 64), device=d)),
        ("tensorwise", torch.ones((), device=d), torch.ones((), device=d)),
    ):
        try:
            torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, use_fast_accum=True)
            out[name] = True
        except Exception as exc:
            out[name] = False
            out[name + "_error"] = str(exc).splitlines()[0][:200]
    return out


_EAGER_EPILOGUE_ROWS = 8192


def install_rowwise_fp8_compat(torch) -> None:
    """Emulate row-wise scaled FP8 GEMMs with a tensor-wise GEMM + epilogue.

    ``y = (A_fp8 @ B_fp8) * scale_a[M,1] * scale_b[1,N] + bias``. The FP8
    weights, activations and their row-wise scales are exactly the ones the
    upstream recipe produces; only where the scales are applied changes (a
    separate epilogue instead of inside the GEMM). Under ``torch.compile`` the
    epilogue fuses into one pointwise kernel; in eager mode it is applied in
    row chunks to bound the FP32 temporary.
    """
    original = torch._scaled_mm

    def _epilogue(y, scale_a, scale_b, bias, out_dtype):
        if torch.compiler.is_compiling():
            z = y.float() * scale_a * scale_b
            if bias is not None:
                z = z + bias.float()
            return z.to(out_dtype)
        out = y if y.dtype == out_dtype else torch.empty(y.shape, dtype=out_dtype, device=y.device)
        sa_rows = scale_a.shape[0] if scale_a.dim() == 2 and scale_a.shape[0] > 1 else None
        for i in range(0, y.shape[0], _EAGER_EPILOGUE_ROWS):
            j = min(i + _EAGER_EPILOGUE_ROWS, y.shape[0])
            sa = scale_a[i:j] if sa_rows else scale_a
            z = y[i:j].float() * sa * scale_b
            if bias is not None:
                z = z + bias.float()
            out[i:j] = z.to(out_dtype)
        return out

    def scaled_mm_compat(
        a, b, scale_a=None, scale_b=None, bias=None, scale_result=None, out_dtype=None, use_fast_accum=False, **kwargs
    ):
        # Inductor resolves its extern GEMM call through ``torch._scaled_mm`` and
        # may pass extra keywords such as ``out=``; forward them untouched.
        rowwise = (scale_a is not None and scale_a.numel() > 1) or (scale_b is not None and scale_b.numel() > 1)
        if not rowwise:
            return original(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                bias=bias,
                scale_result=scale_result,
                out_dtype=out_dtype,
                use_fast_accum=use_fast_accum,
                **kwargs,
            )
        out = kwargs.pop("out", None)
        one = torch.ones((), device=a.device, dtype=torch.float32)
        target = out_dtype or torch.bfloat16
        y = original(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16, use_fast_accum=use_fast_accum)
        result = _epilogue(y, scale_a.float(), scale_b.float(), bias, target)
        if out is not None:
            out.copy_(result)
            return out
        return result

    scaled_mm_compat.__wrapped__ = original  # type: ignore[attr-defined]
    torch._scaled_mm = scaled_mm_compat


# --------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------


class State:
    def __init__(self, job: dict):
        self.job = job
        self.checks: dict = {}
        self.timings: dict = {"samples": []}
        self.memory: dict = {}
        self.residency: list[dict] = []
        self.steps: list[dict] = []
        self.outputs: list[str] = []
        self.models: list[dict] = []
        self.warnings: list[str] = []
        self.fp8_backend: str | None = None
        self.compile_enabled: bool = bool(job.get("torch_compile", True))
        self.sdpa_fallback: str | None = None
        self.peaks: dict = {}
        self.overall_peak = {"allocated_gib": 0.0, "reserved_gib": 0.0}

    def peak_checkpoint(self, torch, phase: str | None) -> None:
        """Fold the CUDA peak since the last reset into ``phase`` and the overall peak, then reset."""
        alloc = _gib(torch.cuda.max_memory_allocated())
        reserved = _gib(torch.cuda.max_memory_reserved())
        self.overall_peak["allocated_gib"] = max(self.overall_peak["allocated_gib"], alloc)
        self.overall_peak["reserved_gib"] = max(self.overall_peak["reserved_gib"], reserved)
        if phase:
            prev = self.peaks.get(phase, {"allocated_gib": 0.0, "reserved_gib": 0.0})
            self.peaks[phase] = {
                "allocated_gib": max(prev["allocated_gib"], alloc),
                "reserved_gib": max(prev["reserved_gib"], reserved),
            }
        torch.cuda.reset_peak_memory_stats()


def _gib(nbytes: int) -> float:
    return round(nbytes / 2**30, 3)


def _cuda_mem(torch) -> dict:
    return {
        "allocated_gib": _gib(torch.cuda.memory_allocated()),
        "reserved_gib": _gib(torch.cuda.memory_reserved()),
        "peak_allocated_gib": _gib(torch.cuda.max_memory_allocated()),
        "peak_reserved_gib": _gib(torch.cuda.max_memory_reserved()),
    }


def _inspect_dit(net) -> dict:
    """Static inspection of a WanModel before optimization."""
    blocks = getattr(net, "blocks", [])
    attn_classes: dict[str, int] = {}
    topk: set = set()
    fused_flags: set = set()
    precisions: set = set()
    for blk in blocks:
        sa = blk.self_attn
        name = type(sa).__name__
        attn_classes[name] = attn_classes.get(name, 0) + 1
        if hasattr(sa, "rola_topk_ratio"):
            topk.add(float(sa.rola_topk_ratio))
        fused_flags.add(bool(getattr(sa, "_use_fused_inference", False)))
        if hasattr(sa, "attn_precision"):
            precisions.add(sa.attn_precision)
    return {
        "num_layers": len(blocks),
        "dim": getattr(net, "dim", None),
        "model_type": getattr(net, "model_type", None),
        "in_dim": getattr(net, "in_dim", None),
        "self_attention_classes": attn_classes,
        "rola_layers": attn_classes.get("WanSelfAttentionRoLa", 0),
        "rola_topk_ratios": sorted(topk),
        "fused_inference_flags": sorted(fused_flags),
        "rola_attn_precision": sorted(precisions),
        "params_billion": round(sum(p.numel() for p in net.parameters()) / 1e9, 3),
    }


def _count_fp8(net) -> dict:
    counts: dict[str, int] = {}
    for m in net.modules():
        n = type(m).__name__
        if n in _FP8_CLASSES:
            counts[n] = counts.get(n, 0) + 1
    return {"fp8_modules": sum(counts.values()), "fp8_module_classes": counts}


class StepRecorder:
    """Callable proxy around the (compiled) DiT that times every forward call.

    Upstream calls the returned object with keyword arguments only and may call
    ``.cuda()`` / ``.cpu()`` on it (Wan 2.2 expert swap); everything else is
    delegated to the wrapped module.
    """

    def __init__(self, inner, torch, state: State, name: str):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_torch", torch)
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_name", name)

    def __call__(self, *args, **kwargs):
        torch, state = self._torch, self._state
        ts = kwargs.get("timesteps_B_T")
        t_val = float(ts.flatten()[0].float().item()) if ts is not None else None
        per = int(state.job.get("steps_per_sample") or 0) or None
        idx = len(state.steps)
        torch.cuda.synchronize()
        if per and idx % per == 0:
            state.peak_checkpoint(torch, "between_phases")
        t0 = time.perf_counter()
        out = self._inner(*args, **kwargs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if per and idx % per == per - 1:
            state.peak_checkpoint(torch, "denoise")
        state.steps.append(
            {
                "model": self._name,
                "timestep": t_val,
                "seconds": round(dt, 4),
                "wall_start": time.time() - dt,
                "wall_end": time.time(),
            }
        )
        if per:
            emit(
                "progress",
                step=idx % per + 1,
                total=per,
                sample=idx // per + 1,
                samples=int(state.job.get("num_samples", 1)),
                seconds=round(dt, 3),
                timestep=t_val,
            )
        else:
            emit("progress", step=idx + 1, seconds=round(dt, 3), timestep=t_val)
        return out

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def __setattr__(self, key, value):
        setattr(self._inner, key, value)


def install_sdpa_fallback(torch, state: State) -> None:
    """Keep upstream's *eager* attention usable on PyTorch builds without FlashAttention.

    Upstream's ``_get_sdpa_config`` asks for ``[FLASH, CUDNN, EFFICIENT]`` with
    ``sdpa_kernel(..., set_priority_order=True)``. Newer PyTorch removed that
    keyword, so upstream falls back to ``[FLASH]`` only. Windows PyTorch builds
    ship without FlashAttention SDPA, which makes every eager attention call
    fail with "No available kernel". The compiled path is unaffected (Inductor
    picks the kernel). This shim substitutes memory-efficient SDPA (exact
    attention) only in that specific situation.
    """
    if torch.backends.cuda.is_flash_attention_available():
        return
    import sparkdiffusion.utils.attention as attn_mod
    from torch.nn.attention import SDPBackend, sdpa_kernel

    original = attn_mod._get_sdpa_config

    @torch.compiler.disable
    def _get_sdpa_config(device, is_half):
        cc, backends, kern = original(device, is_half)
        if list(backends) == [SDPBackend.FLASH_ATTENTION]:
            return cc, [SDPBackend.EFFICIENT_ATTENTION], sdpa_kernel
        return cc, backends, kern

    attn_mod._get_sdpa_config = _get_sdpa_config
    state.sdpa_fallback = (
        "installed for eager attention: flash-only -> memory-efficient "
        "(this PyTorch build has no FlashAttention SDPA; unused when torch.compile is on)"
    )


def install_instrumentation(torch, state: State) -> None:
    import sparkdiffusion.utils.inference_optimization as io_mod
    import sparkdiffusion.utils.model_utils as mu
    import sparkdiffusion.utils.umt5 as umt5
    import sparkdiffusion.tokenizers.wan2pt1 as vae_mod
    import imaginaire.utils.io as imio

    expect_rola = bool(state.job.get("expect_rola", True))

    # ---- checkpoint loading: refuse random-init RoLA weights ----
    orig_load = mu.load_checkpoint_auto

    def load_checkpoint_auto(path, net, *a, **kw):
        t0 = time.perf_counter()
        result = orig_load(path, net, *a, **kw)
        missing = list(getattr(result, "missing_keys", []) or [])
        rola_missing = [k for k in missing if any(m in k for m in _ROLA_PARAM_MARKERS)]
        info = {
            "checkpoint": _basename(path),
            "load_seconds": round(time.perf_counter() - t0, 2),
            "missing_keys": len(missing),
            "rola_params_missing": len(rola_missing),
            "unexpected_keys": len(getattr(result, "unexpected_keys", []) or []),
        }
        state.models.append(info)
        emit("phase", phase="checkpoint_loaded", **info)
        if expect_rola and rola_missing:
            raise RuntimeError(
                f"{len(rola_missing)} RoLA parameters are missing from {_basename(path)}; upstream would run "
                "them at random init. This checkpoint is not a SparkDiffusion RoLA checkpoint for this profile."
            )
        return result

    mu.load_checkpoint_auto = load_checkpoint_auto

    # ---- optimize_model_for_inference: inspect, toggle compile, count FP8 ----
    orig_opt = io_mod.optimize_model_for_inference

    def optimize_model_for_inference(model, *args, **kwargs):
        name = f"dit{len([m for m in state.models if 'inspect' in m])}"
        inspect = _inspect_dit(model)
        emit(
            "phase",
            phase="optimize_start",
            model=name,
            quant_type=kwargs.get("quant_type", ""),
            torch_compile=state.compile_enabled,
        )
        t0 = time.perf_counter()
        state.peak_checkpoint(torch, "pre_optimize")
        saved_compile = torch.compile
        if not state.compile_enabled:
            torch.compile = lambda m, *a, **k: m  # noqa: E731 - identity "compile"
        try:
            compiled, converted = orig_opt(model, *args, **kwargs)
        finally:
            torch.compile = saved_compile
        torch.cuda.synchronize()
        state.peak_checkpoint(torch, "dit_load_quantize")
        inner = getattr(compiled, "_orig_mod", compiled)
        fp8 = _count_fp8(inner)
        entry = None
        for m in state.models:
            if "inspect" not in m:
                entry = m
                break
        record = entry if entry is not None else {}
        record.update(
            {
                "name": name,
                "inspect": inspect,
                "quant_type": kwargs.get("quant_type", "") or "bf16",
                "converted_gemms": int(converted),
                **fp8,
                "optimize_seconds": round(time.perf_counter() - t0, 2),
                "torch_compile_wrapped": type(compiled).__name__ == "OptimizedModule",
                "memory_after_optimize": _cuda_mem(torch),
            }
        )
        if entry is None:
            state.models.append(record)
        if fp8["fp8_modules"]:
            log(f"FP8 active: {fp8['fp8_modules']} modules ({converted} GEMMs converted) on {name}")
        log(
            f"RoLA layers: {inspect['rola_layers']}/{inspect['num_layers']} top-k ratio "
            f"{inspect['rola_topk_ratios']} | fused inference flag {inspect['fused_inference_flags']} | "
            f"torch.compile {'ON' if record['torch_compile_wrapped'] else 'OFF'}"
        )
        emit(
            "phase",
            phase="optimize_done",
            model=name,
            fp8_modules=fp8["fp8_modules"],
            rola_layers=inspect["rola_layers"],
            torch_compile=record["torch_compile_wrapped"],
        )
        return StepRecorder(compiled, torch, state, name), converted

    io_mod.optimize_model_for_inference = optimize_model_for_inference

    # ---- umT5: timing + residency ----
    orig_embed = umt5.get_umt5_embedding
    orig_clear = umt5.clear_umt5_memory

    def get_umt5_embedding(*a, **kw):
        emit("phase", phase="text_encode_start")
        torch.cuda.synchronize()
        state.peak_checkpoint(torch, "pre_text_encode")
        t0 = time.perf_counter()
        out = orig_embed(*a, **kw)
        torch.cuda.synchronize()
        state.timings["text_encode_s"] = round(time.perf_counter() - t0, 3)
        state.peak_checkpoint(torch, "text_encode")
        state.residency.append({"event": "umt5_loaded", **_cuda_mem(torch)})
        return out

    def clear_umt5_memory(*a, **kw):
        out = orig_clear(*a, **kw)
        state.residency.append({"event": "umt5_unloaded", **_cuda_mem(torch)})
        return out

    umt5.get_umt5_embedding = get_umt5_embedding
    umt5.clear_umt5_memory = clear_umt5_memory

    # ---- VAE decode timing ----
    orig_decode = vae_mod.Wan2pt1VAEInterface.decode

    def decode(self, latent):
        emit("phase", phase="vae_decode_start")
        torch.cuda.synchronize()
        state.peak_checkpoint(torch, "pre_vae_decode")
        t0 = time.perf_counter()
        out = orig_decode(self, latent)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        state.timings.setdefault("vae_decode_s", []).append(round(dt, 3))
        state.peak_checkpoint(torch, "vae_decode")
        return out

    vae_mod.Wan2pt1VAEInterface.decode = decode

    # ---- video writer: record output names ----
    orig_save = imio.save_image_or_video

    def save_image_or_video(tensor, save_path, *a, **kw):
        t0 = time.perf_counter()
        out = orig_save(tensor, save_path, *a, **kw)
        state.outputs.append(_basename(save_path))
        state.timings.setdefault("video_write_s", []).append(round(time.perf_counter() - t0, 3))
        emit("output", file=_basename(save_path))
        return out

    imio.save_image_or_video = save_image_or_video


def _bf16_round(torch, values):
    return [float(v) for v in torch.tensor(values, dtype=torch.float32).to(torch.bfloat16).float().tolist()]


def finalize_checks(torch, state: State) -> dict:
    job = state.job
    expected_rf = [float(t) for t in job.get("expected_timesteps", [])]
    scale = float(job.get("t_scaling_factor", 1000.0))
    per = int(job.get("steps_per_sample") or len(expected_rf) - (2 if job.get("family") == "wan2.2" else 1))
    expected_net = [t * scale for t in expected_rf if t > 0.0]
    expected_bf16 = _bf16_round(torch, expected_net) if expected_net else []
    observed = [s["timestep"] for s in state.steps]
    sample_schedules = [observed[i : i + per] for i in range(0, len(observed), per)] if per else []
    schedule_ok = bool(sample_schedules) and all(s == expected_bf16 for s in sample_schedules)

    dit_models = [m for m in state.models if "inspect" in m]
    rola_layers = [m["inspect"]["rola_layers"] for m in dit_models]
    num_layers = [m["inspect"]["num_layers"] for m in dit_models]
    topk_observed = sorted({t for m in dit_models for t in m["inspect"]["rola_topk_ratios"]})
    expected_topk = job.get("topk_ratio")
    rola_active = (
        bool(dit_models)
        and all(r == n and r > 0 for r, n in zip(rola_layers, num_layers))
        and all(m.get("rola_params_missing", 0) == 0 for m in dit_models)
    )
    topk_ok = expected_topk is not None and topk_observed == [float(expected_topk)]
    fp8_modules = sum(m.get("fp8_modules", 0) for m in dit_models)
    quant_requested = job.get("quantization", "bf16")
    fused_flags = sorted({f for m in dit_models for f in m["inspect"]["fused_inference_flags"]})
    precisions = sorted({p for m in dit_models for p in m["inspect"]["rola_attn_precision"]})
    fused_active = fused_flags == [True] and "int8" not in precisions
    compile_active = bool(dit_models) and all(m.get("torch_compile_wrapped") for m in dit_models)

    checks = {
        "rola_active": rola_active,
        "rola_layers": rola_layers,
        "topk_ratio": topk_observed[0] if len(topk_observed) == 1 else topk_observed,
        "sparsity": round(1 - topk_observed[0], 4) if len(topk_observed) == 1 else None,
        "topk_matches_profile": topk_ok,
        "crossdistill_active": schedule_ok,
        "schedule_expected_rf": expected_rf,
        "schedule_observed_net_timesteps": sample_schedules[0] if sample_schedules else [],
        "schedule_expected_net_timesteps_bf16": expected_bf16,
        "quantization": quant_requested,
        "fp8_active": fp8_modules > 0,
        "fp8_modules": fp8_modules,
        "fp8_gemm_backend": state.fp8_backend if fp8_modules else None,
        "eager_sdpa_fallback": state.sdpa_fallback,
        "fused_kernels_active": fused_active,
        "torch_compile_active": compile_active,
    }
    problems = []
    if job.get("expect_rola", True) and not rola_active:
        problems.append("RoLA sparse attention is not active on every transformer block")
    if job.get("expect_rola", True) and not topk_ok:
        problems.append(f"RoLA top-k ratio {topk_observed} does not match the profile ({expected_topk})")
    if not schedule_ok:
        problems.append(f"observed timesteps {sample_schedules[:1]} != CrossDistill {expected_bf16}")
    if quant_requested == "fp8" and fp8_modules == 0:
        problems.append("FP8 was requested but no FP8 modules were created")
    if job.get("fused_kernels", True) and not fused_active:
        problems.append("fused kernels were requested but are not enabled on the RoLA layers")
    if state.compile_enabled and not compile_active:
        problems.append("torch.compile was requested but the model is not compiled")
    checks["problems"] = problems
    return checks


def build_samples(state: State, telemetry: Telemetry | None) -> list[dict]:
    per = int(state.job.get("steps_per_sample") or 0)
    if not per:
        return []
    out = []
    for i in range(0, len(state.steps), per):
        chunk = state.steps[i : i + per]
        start, end = chunk[0]["wall_start"], chunk[-1]["wall_end"]
        k = i // per
        vae = state.timings.get("vae_decode_s", [])
        out.append(
            {
                "index": k,
                "seed": int(state.job.get("seed", 0)) + k,
                "label": "warmup" if k == 0 else "after warmup",
                "denoise_s": round(end - start, 3),
                "model_forward_s": round(sum(s["seconds"] for s in chunk), 3),
                "step_s": [s["seconds"] for s in chunk],
                "vae_decode_s": vae[k] if k < len(vae) else None,
                "telemetry_denoise": telemetry.summarize(start, end) if telemetry else None,
            }
        )
    return out


def run_job(job_path: str) -> int:
    t_start = time.time()
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    if job.get("schema") != JOB_SCHEMA:
        log(f"unsupported job schema {job.get('schema')!r}")
        return EXIT_CONFIG

    if hasattr(os, "setpgrp"):
        try:
            os.setpgrp()
        except Exception:
            pass
    if job.get("pid_file"):
        with open(job["pid_file"], "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    CancelWatchdog(job.get("cancel_file")).start()

    repo = job["repo"]
    entry = os.path.join(repo, *job["entrypoint"].split("/"))
    metadata_path = job.get("metadata_path")
    state = State(job)
    meta: dict = {"schema": "comfyui-sparkdiffusion/metadata/v1", "status": "running"}

    defaults = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "PYTHONIOENCODING": "utf-8",
    }
    if job.get("allocator_tuning", True):
        # Upstream leaves a large reserved-but-unallocated pool (text encoder, FP8 staging,
        # activations). On 32 GB cards that pushes total usage past VRAM, and the Windows
        # driver (also under WSL2) then silently spills to system memory: a 14B 720P step
        # took ~2x longer in our tests. Allocator choice is numerically neutral.
        # Linux/WSL2: expandable segments. Windows: the CUDA async allocator, because the
        # native Windows allocator does not support expandable segments.
        if platform.system() == "Linux":
            defaults["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        elif platform.system() == "Windows":
            defaults["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    if job.get("sla_src"):
        os.environ["SLA_SRC"] = job["sla_src"]
    for key, value in (job.get("env") or {}).items():
        os.environ[str(key)] = str(value)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.chdir(repo)

    emit("start", entrypoint=job["entrypoint"], profile=job.get("profile_key"))
    telemetry = Telemetry() if job.get("telemetry", True) else None
    if telemetry:
        telemetry.start()

    exit_code = EXIT_OK
    error = None
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the SparkDiffusion runtime interpreter")
        props = torch.cuda.get_device_properties(0)
        try:
            import triton

            triton_version = triton.__version__
        except Exception:
            triton_version = None
        meta["environment"] = {
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "triton": triton_version,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": "sm_{}{}".format(*torch.cuda.get_device_capability(0)),
            "vram_total_gib": _gib(props.total_memory),
            "sparkdiffusion_sha": git_sha(repo),
            "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        }
        log(
            f"torch {torch.__version__} cuda {torch.version.cuda} triton {triton_version} | "
            f"{meta['environment']['gpu']} {meta['environment']['compute_capability']}"
        )

        if job.get("quantization") == "fp8":
            probe = probe_scaled_mm(torch)
            meta["environment"]["scaled_mm"] = probe
            compat = job.get("fp8_compat", "auto")
            if probe.get("rowwise") and compat != "force":
                state.fp8_backend = "torch._scaled_mm rowwise (native)"
            elif probe.get("tensorwise") and compat in ("auto", "force"):
                install_rowwise_fp8_compat(torch)
                state.fp8_backend = "torch._scaled_mm tensorwise + rowwise epilogue (compat)"
                msg = (
                    "this PyTorch build has no row-wise FP8 _scaled_mm (typical for Windows builds); "
                    "using the tensor-wise + epilogue compatibility path"
                )
                log("WARNING: " + msg)
                state.warnings.append(msg)
            else:
                raise RuntimeError(
                    "FP8 W8A8 needs row-wise torch._scaled_mm, which this PyTorch build lacks "
                    f"({probe.get('rowwise_error', 'unsupported')}). Use backend=wsl2/linux, "
                    "quantization=bf16, or fp8_compat=auto."
                )

        install_instrumentation(torch, state)
        install_sdpa_fallback(torch, state)
        if state.sdpa_fallback:
            log("eager SDPA fallback: " + state.sdpa_fallback)
        if job.get("prewarm_einops", True):
            # Dynamo guards on the size of einops' backend registry. Upstream's
            # video writer later registers the numpy backend, which would force
            # a full recompile of the DiT at the start of the next sample.
            # Registering both backends up front is numerically neutral.
            try:
                import einops
                import numpy as np

                einops.rearrange(torch.zeros(1, 1), "a b -> b a")
                einops.rearrange(np.zeros((1, 1)), "a b -> b a")
                state.timings["einops_prewarmed"] = True
            except Exception as exc:
                state.warnings.append(f"einops pre-registration skipped: {exc}")
        state.peak_checkpoint(torch, None)
        sys.argv = [entry, *[str(a) for a in job["argv"]]]
        log("exec " + job["entrypoint"])
        runpy.run_path(entry, run_name="__main__")
        torch.cuda.synchronize()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else EXIT_CONFIG)
        if code != 0:
            exit_code = code
            error = f"upstream entry point exited with code {code}"
    except BaseException as exc:  # noqa: BLE001 - report everything to the host
        exit_code = EXIT_ERROR
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        if telemetry:
            telemetry.stop()

    try:
        import torch  # noqa: F811

        meta["checks"] = finalize_checks(torch, state)
        state.peak_checkpoint(torch, "tail")
        meta["memory"] = {
            "peak_allocated_gib": state.overall_peak["allocated_gib"],
            "peak_reserved_gib": state.overall_peak["reserved_gib"],
            "phase_peaks": state.peaks,
        }
    except Exception as exc:  # never lose the original error
        meta["checks"] = {"problems": [f"check finalisation failed: {exc}"]}
    meta["models"] = state.models
    meta["residency"] = state.residency
    meta["samples"] = build_samples(state, telemetry)
    meta["timings"] = {k: v for k, v in state.timings.items() if k != "samples"}
    meta["timings"]["launcher_wall_s"] = round(time.time() - t_start, 2)
    meta["outputs"] = state.outputs
    meta["warnings"] = state.warnings
    if telemetry:
        meta["telemetry"] = {
            "available": telemetry.available,
            "error": telemetry.error,
            "overall": telemetry.summarize(),
        }
    if exit_code == EXIT_OK and not state.outputs:
        exit_code = EXIT_ERROR
        error = "upstream finished without writing a video"
    if exit_code == EXIT_OK and meta["checks"].get("problems") and job.get("strict_checks", True):
        exit_code = EXIT_ERROR
        error = "SparkDiffusion verification failed: " + "; ".join(meta["checks"]["problems"])
    meta["status"] = "ok" if exit_code == EXIT_OK else "error"
    meta["error"] = error
    if metadata_path:
        tmp = metadata_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, metadata_path)
    emit("result", status=meta["status"], error=error, metadata=_basename(metadata_path))
    if error:
        log("ERROR: " + error)
    return exit_code


def run_probe(repo: str | None, do_compile: bool) -> int:
    info: dict = {
        "python": platform.python_version(),
        "executable_name": os.path.basename(sys.executable),
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_runtime"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = torch.cuda.get_device_name(0)
            info["compute_capability"] = list(torch.cuda.get_device_capability(0))
            info["vram_total_gib"] = _gib(props.total_memory)
            info["scaled_mm"] = probe_scaled_mm(torch)
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    for mod in ("triton", "flash_attn", "pynvml", "imageio_ffmpeg"):
        try:
            m = importlib.import_module(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except Exception as exc:
            info[mod] = None
            info[mod + "_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    if repo:
        info["sparkdiffusion_sha"] = git_sha(repo)
        info["sparkdiffusion_repo_ok"] = os.path.isfile(
            os.path.join(repo, "sparkdiffusion", "inference", "wan2pt1_t2v_distilled_infer.py")
        )
        if repo not in sys.path:
            sys.path.insert(0, repo)
        try:
            import sparkdiffusion.ops.fused_kernel  # noqa: F401

            info["fused_kernels_import"] = True
        except Exception as exc:
            info["fused_kernels_import"] = False
            info["fused_kernels_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        info["sla_src"] = bool(os.environ.get("SLA_SRC"))
    if do_compile and info.get("cuda_available"):
        try:
            import torch

            fn = torch.compile(lambda a, b: torch.nn.functional.gelu(a @ b), dynamic=False)
            x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
            fn(x, x)
            torch.cuda.synchronize()
            info["torch_compile_smoke"] = True
        except Exception as exc:
            info["torch_compile_smoke"] = False
            info["torch_compile_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    emit("probe", **info)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    _setup_stdio()
    p = argparse.ArgumentParser(description="ComfyUI-SparkDiffusion runtime launcher")
    p.add_argument("--job", help="UTF-8 job JSON written by ComfyUI-SparkDiffusion")
    p.add_argument("--probe", action="store_true", help="print environment capabilities as an event")
    p.add_argument("--repo", help="SparkDiffusion checkout (probe mode)")
    p.add_argument("--probe-compile", action="store_true", help="also run a tiny torch.compile smoke test")
    args = p.parse_args(argv)
    if args.probe:
        return run_probe(args.repo, args.probe_compile)
    if not args.job:
        p.error("--job is required unless --probe is given")
    try:
        return run_job(args.job)
    except Exception as exc:  # malformed job / unreadable repo: fail loudly but cleanly
        traceback.print_exc()
        emit("result", status="error", error=f"{type(exc).__name__}: {exc}", metadata=None)
        return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
