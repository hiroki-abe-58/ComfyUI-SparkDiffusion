import json
import os

import pytest

from conftest import load

command = load("spark_runtime.command")
paths = load("spark_runtime.paths")
profiles = load("spark_runtime.profiles")

T2V = profiles.get_profile("wan2.1-t2v-1.3b-480p-s90-4step")
T2V_720 = profiles.get_profile("wan2.1-t2v-14b-720p-s95-3step")
I2V = profiles.get_profile("wan2.1-i2v-14b-720p-s97-4step")
WAN22 = profiles.get_profile("wan2.2-t2v-a14b-480p-s95-4step")

ASSETS = {
    "checkpoints": ["/m/ck.pth"],
    "vae": "/m/vae.pth",
    "t5": "/m/t5.pth",
    "tokenizer": "/m/tok",
    "clip": "/m/clip.pth",
}


def argmap(argv):
    out, i = {}, 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--prompt="):
            out["--prompt"] = a[len("--prompt=") :]
            i += 1
        elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[a] = argv[i + 1]
            i += 2
        else:
            out[a] = True
            i += 1
    return out


def cfg(**kw):
    base = dict(
        backend="linux" if os.name != "nt" else "windows-native", repo="/repo", python="/py", model_root="/models"
    )
    base.update(kw)
    return command.RuntimeConfig(**base)


def test_t2v_argv_fp8():
    req = command.GenerationRequest(prompt="a cat", seed=7)
    a = argmap(command.build_upstream_argv(T2V, req, ASSETS, "/out/v.mp4", cfg()))
    assert a["--model_size"] == "1.3B_rola"
    assert a["--num_steps"] == "4"
    assert float(a["--rola_topk_ratio"]) == pytest.approx(0.1)
    assert float(a["--sigma_max"]) == 1600.0
    assert a["--dit_path"] == "/m/ck.pth"
    assert a["--quant_type"] == "fp8" and a["--quant_mode"] == "w8a8"
    assert a["--resolution"] == "480p" and a["--seed"] == "7" and a["--num_frames"] == "81"
    assert a["--attn_precision"] == "bf16"
    assert "--disable_fused_kernels" not in a and "--image_path" not in a
    assert a["--prompt"] == "a cat"


def test_t2v_argv_bf16_and_steps_and_topk_override():
    req = command.GenerationRequest(prompt="x", topk_ratio=0.1)
    a = argmap(command.build_upstream_argv(T2V_720, req, ASSETS, "/o.mp4", cfg(quantization="bf16")))
    assert "--quant_type" not in a
    assert a["--num_steps"] == "3" and a["--resolution"] == "720p"
    assert float(a["--rola_topk_ratio"]) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        command.build_upstream_argv(
            T2V_720, command.GenerationRequest(prompt="x", topk_ratio=0.03), ASSETS, "/o.mp4", cfg()
        )


def test_disable_fused_flag():
    argv = command.build_upstream_argv(
        T2V, command.GenerationRequest(prompt="x"), ASSETS, "/o.mp4", cfg(fused_kernels=False)
    )
    assert "--disable_fused_kernels" in argv


@pytest.mark.parametrize(
    "prompt",
    [
        "-starts with dash",
        "--model_size 14B",
        "日本語のプロンプト、猫が歩く",
        "quotes \" and ' and $(rm -rf /) && echo pwned",
        "multi\nline",
    ],
)
def test_prompt_is_single_literal_argument(prompt):
    argv = command.build_upstream_argv(T2V, command.GenerationRequest(prompt=prompt), ASSETS, "/o.mp4", cfg())
    assert argv[-1] == "--prompt=" + prompt
    assert argv.count("--model_size") == 1


def test_i2v_requires_image_and_t2v_rejects_image():
    with pytest.raises(command.ConfigError):
        command.build_upstream_argv(I2V, command.GenerationRequest(prompt="x"), ASSETS, "/o.mp4", cfg())
    a = argmap(
        command.build_upstream_argv(
            I2V,
            command.GenerationRequest(prompt="x", image_path="/in.png", fixed_resolution=True),
            ASSETS,
            "/o.mp4",
            cfg(),
        )
    )
    assert a["--image_path"] == "/in.png" and a["--clip_encoder_path"] == "/m/clip.pth"
    assert a["--fixed_resolution"] is True and a["--model_size"] == "14B_rola"
    with pytest.raises(command.ConfigError):
        command.build_upstream_argv(
            T2V, command.GenerationRequest(prompt="x", image_path="/in.png"), ASSETS, "/o.mp4", cfg()
        )


def test_wan22_argv():
    assets = dict(ASSETS, checkpoints=["/m/high.pth", "/m/low.pth"])
    a = argmap(command.build_upstream_argv(WAN22, command.GenerationRequest(prompt="x"), assets, "/o.mp4", cfg()))
    assert a["--high_noise_model_path"] == "/m/high.pth" and a["--low_noise_model_path"] == "/m/low.pth"
    assert a["--num_steps_high"] == "2" and a["--num_steps_low"] == "2"
    assert "--num_steps" not in a and "--dit_path" not in a
    assert a["--model_size"] == "A14B_rola"


def test_invalid_frames():
    with pytest.raises(ValueError):
        command.build_upstream_argv(T2V, command.GenerationRequest(prompt="x", num_frames=80), ASSETS, "/o", cfg())


def _model_tree(root, profile):
    d = root / "Wan2.1-T2V-1.3B"
    (d / "google" / "umt5-xxl").mkdir(parents=True)
    (d / "Wan2.1_VAE.pth").write_bytes(b"x")
    (d / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"x")
    (root / profile.repo_dirname).mkdir()
    (root / profile.repo_dirname / profile.checkpoint_files[0]).write_bytes(b"x")


def test_prepare_run_native(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/comfy/site-packages")
    models = tmp_path / "models dir"
    _model_tree(models, T2V)
    c = cfg(model_root=str(models), repo=str(tmp_path / "repo"), python="python")
    run = command.prepare_run(
        c, T2V, command.GenerationRequest(prompt="桜と猫 🐱", seed=3), str(tmp_path / "out put"), str(tmp_path / "work")
    )
    job = json.loads(open(run.job_path, encoding="utf-8").read())
    assert job["schema"] == command.JOB_SCHEMA
    assert job["argv"][-1] == "--prompt=桜と猫 🐱"
    assert job["expected_timesteps"][-1] == 0.0 and len(job["expected_timesteps"]) == 5
    assert job["topk_ratio"] == pytest.approx(0.1) and job["expect_rola"] is True
    assert job["steps_per_sample"] == 4
    assert run.command[0] == "python" and run.command[1:3] == ["-X", "utf8"]
    assert run.command[3].endswith("launcher.py") and run.command[-2:] == ["--job", run.job_path]
    assert "PYTHONPATH" not in run.env and run.env["PYTHONIOENCODING"] == "utf-8"
    assert run.video_path.endswith("sparkdiffusion_00001.mp4")
    assert run.metadata_path.endswith("sparkdiffusion_00001.json")
    assert isinstance(run.command, list)


@pytest.mark.skipif(os.name != "nt", reason="wsl2 backend is Windows-only")
def test_prepare_run_wsl2(tmp_path):
    models = tmp_path / "models dir"
    _model_tree(models, T2V)
    c = cfg(
        backend="wsl2",
        wsl_distro="Ubuntu-24.04",
        repo="/opt/sparkdiffusion/SparkDiffusion",
        python="/opt/sparkdiffusion/venv/bin/python",
        model_root=str(models),
    )
    run = command.prepare_run(
        c, T2V, command.GenerationRequest(prompt="x"), str(tmp_path / "出力"), str(tmp_path / "work")
    )
    assert run.command[:6] == ["wsl.exe", "-d", "Ubuntu-24.04", "--cd", "/", "--exec"]
    assert run.command[6] == "/opt/sparkdiffusion/venv/bin/python"
    assert run.command[9].startswith("/mnt/") and run.command[9].endswith("/spark_runtime/launcher.py")
    assert run.command[-1].startswith("/mnt/") and run.env is None
    job = json.loads(open(run.job_path, encoding="utf-8").read())
    assert job["repo"] == "/opt/sparkdiffusion/SparkDiffusion"
    a = argmap(job["argv"])
    for key in ("--dit_path", "--vae_path", "--text_encoder_path", "--tokenizer_path", "--save_path"):
        assert a[key].startswith("/mnt/"), key
    assert "/models dir/" in a["--vae_path"] and "/出力/" in a["--save_path"]
    assert job["metadata_path"].startswith("/mnt/") and job["cancel_file"].startswith("/mnt/")


def test_prepare_run_errors(tmp_path):
    with pytest.raises(command.ConfigError, match="missing model files"):
        command.prepare_run(
            cfg(model_root=str(tmp_path)),
            T2V,
            command.GenerationRequest(prompt="x"),
            str(tmp_path / "o"),
            str(tmp_path / "w"),
        )
    with pytest.raises(command.ConfigError, match="prompt"):
        command.prepare_run(
            cfg(model_root=str(tmp_path)),
            T2V,
            command.GenerationRequest(prompt="  "),
            str(tmp_path / "o"),
            str(tmp_path / "w"),
        )
    with pytest.raises(command.ConfigError, match="SLA"):
        command.prepare_run(
            cfg(model_root=str(tmp_path), fused_kernels=False),
            T2V,
            command.GenerationRequest(prompt="x"),
            str(tmp_path / "o"),
            str(tmp_path / "w"),
        )
    with pytest.raises(command.ConfigError, match="aspect"):
        command.prepare_run(
            cfg(model_root=str(tmp_path)),
            T2V,
            command.GenerationRequest(prompt="x", aspect_ratio="2:1"),
            str(tmp_path / "o"),
            str(tmp_path / "w"),
        )


@pytest.mark.parametrize("field,msg", [("repo", "repository"), ("python", "interpreter"), ("model_root", "model root")])
def test_missing_runtime_settings(field, msg):
    c = cfg(**{field: ""})
    with pytest.raises(command.ConfigError, match=msg):
        c.validate()


def test_invalid_modes():
    with pytest.raises(command.ConfigError):
        cfg(quantization="int4").validate()
    with pytest.raises(command.ConfigError):
        cfg(fp8_compat="maybe").validate()
    with pytest.raises(command.ConfigError):
        cfg(backend="cloud").validate()
    if os.name != "nt":
        with pytest.raises(command.ConfigError):
            cfg(backend="wsl2").validate()


def test_public_dict_has_no_paths():
    d = cfg(repo=r"C:\secret\repo", python=r"C:\secret\py.exe", model_root=r"C:\secret\m").as_public_dict()
    assert "secret" not in json.dumps(d)


def test_next_output_path_counter(tmp_path):
    first = command.next_output_path(str(tmp_path), "clip")
    assert first.endswith("clip_00001.mp4")
    open(first, "wb").close()
    open(os.path.join(tmp_path, "clip_00002_sample_01_seed_1.mp4"), "wb").close()
    assert command.next_output_path(str(tmp_path), "clip").endswith("clip_00003.mp4")


def test_safe_stem():
    stem = command.safe_stem("../../etc/passwd")
    assert "/" not in stem and "\\" not in stem and not stem.startswith(".")
    assert command.safe_stem("猫 video") == "猫 video"
    assert command.safe_stem("") == "sparkdiffusion"


def test_mask_and_describe():
    fake_hf, fake_gh = "hf_" + "x" * 30, "ghp_" + "x" * 30  # built at runtime: no token-like literals
    assert "hf_" not in command.mask_secrets("token " + fake_hf)
    assert "ghp_" not in command.mask_secrets(fake_gh)
    assert command.mask_secrets("api_key=abc123") == "api_key=***"
    shown = command.describe_command([r"C:\Users\me\venv\python.exe", "-X", "utf8", r"C:\Users\me\x\launcher.py"])
    assert "me" not in shown.replace("python.exe", "") and "launcher.py" in shown


def test_child_env_strips_python_settings():
    env = command.child_env({"PYTHONPATH": "x", "PYTHONHOME": "y", "PATH": "p", "VIRTUAL_ENV": "v"})
    assert env == {"PATH": "p", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
