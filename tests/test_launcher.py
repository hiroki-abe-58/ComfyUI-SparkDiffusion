"""Model-free launcher tests. The launcher is also executed as a standalone script."""

import json
import subprocess
import sys
import types

import pytest

from conftest import ROOT, load

launcher = load("spark_runtime.launcher")
LAUNCHER = str(ROOT / "spark_runtime" / "launcher.py")


def run(*args, cwd=None):
    return subprocess.run([sys.executable, "-X", "utf8", LAUNCHER, *args], capture_output=True, cwd=cwd, timeout=120)


def events(stdout: bytes):
    out = []
    for line in stdout.decode("utf-8").splitlines():
        if line.startswith(launcher.EVENT_PREFIX):
            out.append(json.loads(line[len(launcher.EVENT_PREFIX) :]))
    return out


def test_launcher_is_self_contained():
    """It runs in another interpreter: only stdlib imports at module level."""
    src = (ROOT / "spark_runtime" / "launcher.py").read_text(encoding="utf-8")
    top = [line for line in src.splitlines() if line.startswith(("import ", "from "))]
    allowed = {"argparse", "json", "os", "platform", "runpy", "sys", "threading", "time", "traceback", "__future__"}
    for line in top:
        mod = line.split()[1].split(".")[0]
        assert mod in allowed, line


def test_probe_emits_event_without_torch_requirement(tmp_path):
    res = run("--probe", "--repo", str(tmp_path))
    assert res.returncode == 0
    ev = events(res.stdout)
    assert ev and ev[-1]["event"] == "probe"
    assert "python" in ev[-1] and ev[-1]["sparkdiffusion_repo_ok"] is False


def test_rejects_unknown_schema(tmp_path):
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"schema": "other"}), encoding="utf-8")
    res = run("--job", str(job))
    assert res.returncode == launcher.EXIT_CONFIG


def test_malformed_job_fails_cleanly(tmp_path):
    job = tmp_path / "job.json"
    job.write_text("{not json", encoding="utf-8")
    res = run("--job", str(job))
    assert res.returncode == launcher.EXIT_CONFIG
    assert events(res.stdout)[-1]["status"] == "error"


def test_missing_job_argument():
    res = run()
    assert res.returncode != 0


def test_git_sha_reads_loose_and_packed_refs(tmp_path):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "refs" / "heads" / "main").write_text("a" * 40 + "\n", encoding="utf-8")
    assert launcher.git_sha(str(tmp_path)) == "a" * 40
    (git / "refs" / "heads" / "main").unlink()
    (git / "packed-refs").write_text("# pack\n" + "b" * 40 + " refs/heads/main\n", encoding="utf-8")
    assert launcher.git_sha(str(tmp_path)) == "b" * 40
    (git / "HEAD").write_text("c" * 40, encoding="utf-8")
    assert launcher.git_sha(str(tmp_path)) == "c" * 40
    assert launcher.git_sha(str(tmp_path / "nope")) is None


class _FakeTensor(list):
    def to(self, *_a, **_k):
        return self

    def float(self):
        return self

    def tolist(self):
        return list(self)


def _fake_torch():
    """Just enough torch for finalize_checks (bf16 rounding is emulated)."""
    import struct

    def bf16(x: float) -> float:
        bits = struct.unpack("<I", struct.pack("<f", x))[0]
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000  # round-to-nearest-even
        return struct.unpack("<f", struct.pack("<I", bits))[0]

    t = types.SimpleNamespace(float32="f32", bfloat16="bf16")
    t.tensor = lambda values, dtype=None: _FakeTensor(bf16(v) for v in values)
    return t


def _state(job, steps, models):
    st = launcher.State(job)
    st.steps = [{"timestep": t, "seconds": 0.1, "wall_start": 0, "wall_end": 0.1, "model": "dit0"} for t in steps]
    st.models = models
    st.fp8_backend = "torch._scaled_mm rowwise (native)"
    return st


def _model(rola=30, layers=30, topk=0.1, fused=True, fp8=300, compiled=True, missing=0):
    return {
        "inspect": {
            "rola_layers": rola,
            "num_layers": layers,
            "rola_topk_ratios": [topk],
            "fused_inference_flags": [fused],
            "rola_attn_precision": ["bf16"],
        },
        "fp8_modules": fp8,
        "torch_compile_wrapped": compiled,
        "rola_params_missing": missing,
    }


JOB = {
    "expected_timesteps": [1600 / 1601, 0.933781, 0.852895, 0.608979, 0.0],
    "t_scaling_factor": 1000.0,
    "steps_per_sample": 4,
    "topk_ratio": 0.1,
    "quantization": "fp8",
    "fused_kernels": True,
    "torch_compile": True,
    "expect_rola": True,
    "family": "wan2.1",
}


def test_finalize_checks_all_good():
    checks = launcher.finalize_checks(_fake_torch(), _state(JOB, [1000.0, 932.0, 852.0, 608.0] * 2, [_model()]))
    assert checks["problems"] == []
    assert checks["rola_active"] and checks["crossdistill_active"] and checks["fp8_active"]
    assert checks["sparsity"] == pytest.approx(0.9) and checks["fp8_modules"] == 300


@pytest.mark.parametrize(
    "steps,model,needle",
    [
        ([1000.0, 932.0, 852.0], _model(), "CrossDistill"),
        ([1000.0, 900.0, 852.0, 608.0], _model(), "CrossDistill"),
        ([1000.0, 932.0, 852.0, 608.0], _model(rola=0), "RoLA sparse attention"),
        ([1000.0, 932.0, 852.0, 608.0], _model(missing=12), "RoLA sparse attention"),
        ([1000.0, 932.0, 852.0, 608.0], _model(topk=0.05), "top-k"),
        ([1000.0, 932.0, 852.0, 608.0], _model(fp8=0), "FP8"),
        ([1000.0, 932.0, 852.0, 608.0], _model(fused=False), "fused"),
        ([1000.0, 932.0, 852.0, 608.0], _model(compiled=False), "torch.compile"),
    ],
)
def test_finalize_checks_detects_problems(steps, model, needle):
    checks = launcher.finalize_checks(_fake_torch(), _state(JOB, steps, [model]))
    assert any(needle in p for p in checks["problems"]), checks["problems"]


def test_build_samples_groups_steps():
    st = _state(JOB, [1000.0, 932.0, 852.0, 608.0] * 3, [_model()])
    st.timings["vae_decode_s"] = [1.0, 2.0, 3.0]
    samples = launcher.build_samples(st, None)
    assert [s["label"] for s in samples] == ["warmup", "after warmup", "after warmup"]
    assert samples[2]["vae_decode_s"] == 3.0 and samples[1]["seed"] == 1


def test_telemetry_summary():
    t = launcher.Telemetry()
    t.samples = [(1.0, 50.0, 300.0, 60.0, 1000.0), (2.0, 100.0, 500.0, 70.0, 2000.0)]
    s = t.summarize()
    assert s["gpu_util_avg_pct"] == 75.0 and s["power_max_w"] == 500.0 and s["temp_max_c"] == 70.0
    assert t.summarize(start=1.5)["samples"] == 1
    assert t.summarize(start=5) is None


def test_step_recorder_delegates_attributes():
    class Inner:
        value = 3

        def cpu(self):
            return "moved"

    rec = launcher.StepRecorder(Inner(), None, None, "dit0")
    assert rec.value == 3 and rec.cpu() == "moved"
    rec.value = 5
    assert rec._inner.value == 5


def _cuda_fp8():
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (8, 9):
        return None
    return torch


@pytest.mark.skipif(_cuda_fp8() is None, reason="needs CUDA with FP8 tensor cores")
def test_rowwise_compat_matches_dequantized_reference():
    torch = _cuda_fp8()
    original = torch._scaled_mm
    try:
        launcher.install_rowwise_fp8_compat(torch)
        g = torch.Generator(device="cuda").manual_seed(0)
        a = torch.randn(300, 256, device="cuda", generator=g)
        w = torch.randn(512, 256, device="cuda", generator=g)
        sa = a.abs().amax(1, keepdim=True) / 448.0
        sw = w.abs().amax(1, keepdim=True) / 448.0
        a8, w8 = (a / sa).to(torch.float8_e4m3fn), (w / sw).to(torch.float8_e4m3fn)
        bias = torch.randn(512, device="cuda", generator=g).bfloat16()
        out = torch._scaled_mm(
            a8,
            w8.t(),
            scale_a=sa,
            scale_b=sw.t().contiguous(),
            bias=bias,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        ref = (a8.float() * sa) @ (w8.float() * sw).t() + bias.float()
        rel = (out.float() - ref).norm() / ref.norm()
        assert rel < 1e-2
    finally:
        torch._scaled_mm = original
