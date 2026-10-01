"""Benchmark a SparkDiffusion profile end to end (fixed prompt, fixed seed).

Runs ``--samples`` sequential generations in ONE runtime process (the first is
the warmup that pays checkpoint loading, FP8 conversion, Triton autotuning and
torch.compile), samples ``nvidia-smi`` on the host, and writes a JSON + Markdown
record to ``benchmarks/local/`` (git-ignored). Absolute paths are not recorded.

    python scripts/benchmark.py --profile wan2.1-t2v-14b-720p-s95-3step --backend wsl2 --torch-compile
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import threading
import time

import _bootstrap

DEFAULT_PROMPT = (
    "A playful raccoon is seen playing an electronic guitar, strumming the strings with its front paws. "
    "The raccoon has distinctive black facial markings and a bushy tail. It sits comfortably on a small "
    "stool, its body slightly tilted as it focuses intently on the instrument. The setting is a cozy, "
    "dimly lit room with vintage posters on the walls, adding a retro vibe. The raccoon's expressive "
    "eyes convey a sense of joy and concentration. Medium close-up shot, focusing on the raccoon's face "
    "and hands interacting with the guitar."
)


class NvidiaSmi(threading.Thread):
    FIELDS = "timestamp,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu"

    def __init__(self, interval_ms: int = 500):
        super().__init__(daemon=True)
        self.interval_ms = interval_ms
        self.rows: list = []
        self.proc = None

    def run(self) -> None:
        exe = shutil.which("nvidia-smi")
        if not exe:
            return
        self.proc = subprocess.Popen(
            [exe, f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits", f"-lms={self.interval_ms}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        for raw in self.proc.stdout:  # type: ignore[union-attr]
            parts = [p.strip() for p in raw.decode("utf-8", "replace").split(",")]
            if len(parts) != 6:
                continue
            try:
                self.rows.append(
                    (time.time(), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
                )
            except ValueError:
                continue

    def stop(self) -> None:
        if self.proc:
            self.proc.terminate()

    def summary(self, start: float = 0, end: float = 1e20) -> dict:
        rows = [r for r in self.rows if start <= r[0] <= end]
        if not rows:
            return {}
        n = len(rows)
        return {
            "samples": n,
            "memory_used_max_mib": max(r[1] for r in rows),
            "memory_total_mib": rows[0][2],
            "gpu_util_avg_pct": round(sum(r[3] for r in rows) / n, 1),
            "power_avg_w": round(sum(r[4] for r in rows) / n, 1),
            "power_max_w": max(r[4] for r in rows),
            "temp_max_c": max(r[5] for r in rows),
        }


def main() -> int:
    profiles = _bootstrap.load("spark_runtime.profiles")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _bootstrap.add_runtime_args(p)
    p.add_argument("--profile", required=True, choices=sorted(profiles.PROFILES))
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--output-dir", default=str(_bootstrap.ROOT / "benchmarks" / "local" / "videos"))
    p.add_argument("--label", default="")
    args = p.parse_args()

    command = _bootstrap.load("spark_runtime.command")
    runner = _bootstrap.load("spark_runtime.runner")
    cfg = _bootstrap.runtime_from_args(args)
    profile = profiles.get_profile(args.profile)
    req = command.GenerationRequest(
        prompt=args.prompt, seed=args.seed, num_frames=args.num_frames, num_samples=args.samples
    )
    smi = NvidiaSmi()
    smi.start()
    t0 = time.time()
    try:
        result = runner.generate(
            cfg,
            profile,
            req,
            args.output_dir,
            filename_prefix="bench",
            on_line=lambda _s, line: print(line, flush=True),
        )
    finally:
        smi.stop()
    wall = time.time() - t0
    meta = result.metadata
    samples = meta.get("samples", [])
    for s in samples:
        s["nvidia_smi"] = smi.summary()  # whole run; per-sample NVML numbers are in telemetry_denoise
    record = {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "profile": profile.key,
        "runtime": cfg.as_public_dict(),
        "environment": meta.get("environment"),
        "host": meta.get("host"),
        "checks": {
            k: meta["checks"].get(k)
            for k in (
                "rola_active",
                "topk_ratio",
                "sparsity",
                "crossdistill_active",
                "fp8_active",
                "fp8_modules",
                "fp8_gemm_backend",
                "fused_kernels_active",
                "torch_compile_active",
                "eager_sdpa_fallback",
            )
        },
        "samples": samples,
        "text_encode_s": meta.get("timings", {}).get("text_encode_s"),
        "models": [
            {k: m.get(k) for k in ("checkpoint", "load_seconds", "optimize_seconds", "fp8_modules")}
            for m in meta.get("models", [])
        ],
        "memory": meta.get("memory"),
        "nvidia_smi_run": smi.summary(),
        "end_to_end_wall_s": round(wall, 2),
        "prompt": args.prompt,
        "seed": args.seed,
        "num_frames": args.num_frames,
    }
    out_dir = _bootstrap.ROOT / "benchmarks" / "local"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    mode = "compile" if cfg.torch_compile else "eager"
    stem = f"{stamp}-{profile.key}-{cfg.resolved_backend()}-{cfg.quantization}-{mode}"
    (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"## {profile.label} | {cfg.resolved_backend()} | {cfg.quantization} | "
        f"compile={'on' if cfg.torch_compile else 'off'}",
        "",
        "| sample | label | denoise (s) | model forward (s) | VAE decode (s) |",
        "|---|---|---|---|---|",
    ]
    for s in samples:
        lines.append(
            f"| {s['index']} | {s['label']} | {s['denoise_s']} | {s['model_forward_s']} | {s['vae_decode_s']} |"
        )
    lines += [
        "",
        f"end-to-end wall (load + compile + {len(samples)} samples): {record['end_to_end_wall_s']} s",
        f"peak VRAM allocated (torch): {meta.get('memory', {}).get('peak_allocated_gib')} GiB",
        f"nvidia-smi: {record['nvidia_smi_run']}",
    ]
    (out_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nrecord: benchmarks/local/{stem}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
