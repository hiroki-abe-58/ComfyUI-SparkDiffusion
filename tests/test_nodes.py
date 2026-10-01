import json
import os
import subprocess
import sys
import textwrap

import pytest

from conftest import ROOT, load

nodes = load("nodes")
common = load("nodes.common")
profiles = load("spark_runtime.profiles")

HEAVY = ("torch", "triton", "flash_attn", "sparkdiffusion", "imaginaire", "safetensors", "transformers")


def test_comfyui_style_import_is_lightweight():
    """Load the package exactly like ComfyUI's custom-node loader and check nothing heavy is imported."""
    code = textwrap.dedent(f"""
        import importlib.util, json, sys
        root = {str(ROOT)!r}
        spec = importlib.util.spec_from_file_location("ComfyUI-SparkDiffusion", root + "/__init__.py",
                                                      submodule_search_locations=[root])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ComfyUI-SparkDiffusion"] = mod
        spec.loader.exec_module(mod)
        heavy = [m for m in {HEAVY!r} if m in sys.modules]
        print(json.dumps({{"nodes": sorted(mod.NODE_CLASS_MAPPINGS), "names": mod.NODE_DISPLAY_NAME_MAPPINGS,
                           "heavy": heavy}}))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout.strip().splitlines()[-1])
    assert data["heavy"] == []
    assert set(data["nodes"]) == set(nodes.NODE_CLASS_MAPPINGS)


def test_registration_contract():
    assert set(nodes.NODE_CLASS_MAPPINGS) == set(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        assert name.startswith("SparkDiffusion")
        spec = cls.INPUT_TYPES()
        assert "required" in spec
        assert isinstance(cls.RETURN_TYPES, tuple) and len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)
        assert callable(getattr(cls, cls.FUNCTION))
        assert cls.CATEGORY.startswith("SparkDiffusion")


def test_profile_node_select_and_override():
    node = nodes.NODE_CLASS_MAPPINGS["SparkDiffusionProfile"]()
    p, info = node.select("wan2.1-t2v-14b-720p-s97-4step")
    assert p.topk_ratio == pytest.approx(0.03)
    assert json.loads(info)["sparsity"] == pytest.approx(0.97)
    p2, _ = node.select("wan2.1-t2v-14b-720p-s97-4step", topk_ratio_override=0.05)
    assert p2.topk_ratio == pytest.approx(0.05) and p.topk_ratio == pytest.approx(0.03)
    with pytest.raises(ValueError):
        node.select("wan2.1-t2v-1.3b-480p-s90-4step", topk_ratio_override=0.05)
    assert set(node.INPUT_TYPES()["required"]["preset"][0]) == set(profiles.PROFILES)


def test_runtime_node_builds_config(monkeypatch):
    for k in ("SPARKDIFFUSION_REPO", "SPARKDIFFUSION_PYTHON", "SPARKDIFFUSION_MODEL_ROOT", "SPARKDIFFUSION_BACKEND"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SPARKDIFFUSION_CONFIG", os.devnull + "-missing")
    node = nodes.NODE_CLASS_MAPPINGS["SparkDiffusionRuntime"]()
    (cfg,) = node.build("auto", "fp8", True, False, sparkdiffusion_repo="/r", python="/p", model_root="/m")
    assert cfg.repo == "/r" and cfg.python == "/p" and cfg.quantization == "fp8" and not cfg.torch_compile
    monkeypatch.setenv("SPARKDIFFUSION_REPO", "/env/repo")
    (cfg,) = node.build("auto", "bf16", True, True, python="/p", model_root="/m")
    assert cfg.repo == "/env/repo" and cfg.torch_compile
    monkeypatch.delenv("SPARKDIFFUSION_REPO")
    with pytest.raises(ValueError, match="repository"):
        node.build("auto", "fp8", True, False, sparkdiffusion_repo="", python="/p", model_root="/m")


def test_t2v_rejects_i2v_profile():
    node = nodes.NODE_CLASS_MAPPINGS["SparkDiffusionTextToVideo"]()
    with pytest.raises(ValueError):
        node.generate(None, profiles.get_profile("wan2.1-i2v-14b-720p-s97-4step"), "x", 0, 81, "16:9", "p")
    i2v = nodes.NODE_CLASS_MAPPINGS["SparkDiffusionImageToVideo"]()
    with pytest.raises(ValueError):
        i2v.generate(None, profiles.get_profile("wan2.1-t2v-1.3b-480p-s90-4step"), None, "x", 0, 81, "p")


def test_split_prefix_and_traversal(tmp_path):
    full, sub, name = common.split_prefix("sparkdiffusion/run", str(tmp_path))
    assert sub == "sparkdiffusion" and name == "run" and full == os.path.join(str(tmp_path), "sparkdiffusion")
    full, sub, name = common.split_prefix("clip", str(tmp_path))
    assert sub == "" and name == "clip"
    with pytest.raises(ValueError):
        common.split_prefix("../../outside/x", str(tmp_path))


def test_ui_for(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "output_directory", lambda: str(tmp_path))
    ui = common.ui_for([str(tmp_path / "sub" / "a.mp4"), str(tmp_path / "b.mp4")])
    assert ui["images"] == [
        {"filename": "a.mp4", "subfolder": "sub", "type": "output"},
        {"filename": "b.mp4", "subfolder": "", "type": "output"},
    ]
    assert ui["animated"] == (True,)


def test_progress_callback_without_comfy():
    cb = common.make_progress(4)
    cb({"event": "progress", "step": 1})  # no ComfyUI: must be a no-op, not an error
    assert common.interrupted() is False


def test_sigmas_node():
    torch = pytest.importorskip("torch")
    node = nodes.NODE_CLASS_MAPPINGS["SparkDiffusionCrossDistillSigmas"]()
    (s,) = node.build("3", 1600.0)
    assert torch.allclose(
        s.double(), torch.tensor([1600 / 1601, 0.933781, 0.852895, 0.0], dtype=torch.float64), atol=1e-7
    )
