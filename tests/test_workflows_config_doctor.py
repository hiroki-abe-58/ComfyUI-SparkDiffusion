import json
import os

import pytest

from conftest import ROOT, load

nodes = load("nodes")
profiles = load("spark_runtime.profiles")
config = load("spark_runtime.config")
doctor = load("spark_runtime.doctor")

WORKFLOWS = sorted((ROOT / "workflows").glob("*.json"))
CORE_NODES = {"SaveVideo", "PreviewAny", "LoadImage", "Note", "MarkdownNote", "PreviewImage"}


def test_workflows_exist():
    assert WORKFLOWS, "no example workflows shipped"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_json_is_valid(path):
    wf = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(wf.get("nodes"), list) and wf["nodes"]
    ids = {n["id"] for n in wf["nodes"]}
    types = {n["type"] for n in wf["nodes"]}
    assert types <= set(nodes.NODE_CLASS_MAPPINGS) | CORE_NODES, types
    for link in wf.get("links", []):
        _, src, _, dst, _, _ = link
        assert src in ids and dst in ids
    for n in wf["nodes"]:
        if n["type"] == "SparkDiffusionProfile":
            assert n["widgets_values"][0] in profiles.PROFILES
        if n["type"] == "SparkDiffusionRuntime":
            # shared workflows must not embed machine-specific paths
            for v in n["widgets_values"]:
                if isinstance(v, str):
                    assert ":\\" not in v and not v.startswith("/"), v
    text = path.read_text(encoding="utf-8")
    assert "E:\\\\" not in text and "/mnt/" not in text


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_widgets_match_node_inputs(path):
    from workflow_utils import ui_to_api

    api = ui_to_api(json.loads(path.read_text(encoding="utf-8")), nodes.NODE_CLASS_MAPPINGS)
    gen = next(v for v in api.values() if v["class_type"].startswith("SparkDiffusion") and "prompt" in v["inputs"])
    assert gen["inputs"]["num_frames"] == 81 and gen["inputs"]["num_videos"] == 1
    assert gen["inputs"]["runtime"][1] == 0 and gen["inputs"]["profile"][1] == 0
    runtime = next(v for v in api.values() if v["class_type"] == "SparkDiffusionRuntime")["inputs"]
    assert runtime["backend"] == "auto" and runtime["sparkdiffusion_repo"] == ""


def test_config_precedence(tmp_path, monkeypatch):
    cfg_file = tmp_path / "c.json"
    cfg_file.write_text(json.dumps({"repo": "/from/file", "python": "/py/file"}), encoding="utf-8")
    monkeypatch.setenv("SPARKDIFFUSION_CONFIG", str(cfg_file))
    monkeypatch.delenv("SPARKDIFFUSION_REPO", raising=False)
    file_cfg = config.load_config_file()
    assert config.resolve("repo", "", file_cfg) == "/from/file"
    assert config.resolve("repo", "", file_cfg, environ={"SPARKDIFFUSION_REPO": "/from/env"}) == "/from/env"
    assert config.resolve("repo", "/from/node", file_cfg, environ={"SPARKDIFFUSION_REPO": "/from/env"}) == "/from/node"
    assert config.resolve("model_root", None, file_cfg, environ={}) == ""


def test_config_file_errors(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError):
        config.load_config_file(bad)
    assert config.load_config_file(tmp_path / "missing.json") == {}


def test_example_config_is_valid_and_generic():
    example = ROOT / "sparkdiffusion_config.example.json"
    data = json.loads(example.read_text(encoding="utf-8"))
    assert set(data) <= set(config.ENV_KEYS) | {"_comment"}
    assert "E:\\" not in example.read_text(encoding="utf-8")


def test_doctor_report_format():
    checks = [doctor.Check("PASS", "a", "x"), doctor.Check("WARN", "b", "y")]
    report = doctor.format_report(checks)
    assert "[PASS] a" in report and report.endswith("Overall: WARN")
    assert doctor.format_report([doctor.Check("FAIL", "c", "z")]).endswith("Overall: FAIL")


def test_version_tuple():
    assert doctor._version_tuple("3.4.0") >= (3, 4)
    assert doctor._version_tuple("3.3.1+git") < (3, 4)
    assert doctor._version_tuple("2.14.1+cu130") == (2, 14, 1)


def test_doctor_without_runtime_config(monkeypatch):
    command = load("spark_runtime.command")
    cfg = command.RuntimeConfig(backend="linux" if os.name != "nt" else "windows-native")
    checks = doctor.run_doctor(cfg, compile_smoke=False)
    assert any(c.status == "FAIL" and c.item == "Runtime config" for c in checks)
