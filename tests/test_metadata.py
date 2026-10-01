import json

from conftest import load

M = load("spark_runtime.metadata")


def test_redact_recurses():
    data = {"a": [r"C:\Users\x\y.mp4", {"b": "/home/x/z.pth"}], "n": 3, "ok": "plain text"}
    out = M.redact(data)
    assert out == {"a": ["<path:y.mp4>", {"b": "<path:z.pth>"}], "n": 3, "ok": "plain text"}


def test_load_metadata(tmp_path):
    p = tmp_path / "m.json"
    assert M.load_metadata(str(p)) is None
    p.write_text("{bad", encoding="utf-8")
    assert M.load_metadata(str(p)) is None
    p.write_text("[1, 2]", encoding="utf-8")
    assert M.load_metadata(str(p)) is None
    p.write_text(json.dumps({"status": "ok", "プロンプト": "猫"}, ensure_ascii=False), encoding="utf-8")
    assert M.load_metadata(str(p))["プロンプト"] == "猫"


def test_check_video():
    good = {"exists": True, "size_bytes": 500_000, "frames": 81, "width": 832, "height": 480, "fps": 16.0}
    assert M.check_video(good, 81, (832, 480)) == []
    bad = dict(good, frames=80, width=640, size_bytes=10, fps=24.0)
    problems = M.check_video(bad, 81, (832, 480))
    assert len(problems) == 4
    assert M.check_video({"exists": False}, 81) == ["output video was not written"]


def test_probe_missing_video(tmp_path):
    info = M.probe_video(str(tmp_path / "none.mp4"))
    assert info["exists"] is False


def test_summarize():
    meta = {
        "profile": {"label": "Wan2.1 T2V 1.3B 480P 90% 4-step"},
        "runtime": {"backend": "wsl2"},
        "checks": {
            "rola_active": True,
            "rola_layers": [30],
            "topk_ratio": 0.1,
            "sparsity": 0.9,
            "crossdistill_active": True,
            "schedule_observed_net_timesteps": [1000.0],
            "fp8_active": True,
            "fp8_modules": 300,
            "fp8_gemm_backend": "native",
            "fused_kernels_active": True,
            "torch_compile_active": False,
        },
        "environment": {
            "gpu": "GPU",
            "compute_capability": "sm_120",
            "torch": "2",
            "cuda_runtime": "13",
            "triton": "3",
        },
        "samples": [{"index": 0, "label": "warmup", "denoise_s": 1.0, "vae_decode_s": 2.0}],
        "memory": {"peak_allocated_gib": 3.0},
        "host": {"wall_s": 10.0},
    }
    text = M.summarize(meta)
    for part in (
        "RoLA active: True",
        "FP8 active: True (300 modules",
        "CrossDistill: True",
        "wsl2",
        "sm_120",
        "denoise 1.0s",
        "3.0 GiB",
        "10.0s",
    ):
        assert part in text


def test_package_version_and_git_sha(tmp_path):
    assert M.package_version() != "unknown"
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("d" * 40, encoding="utf-8")
    assert M.git_sha(tmp_path) == "d" * 40
    assert M.git_sha(tmp_path / "missing") is None
