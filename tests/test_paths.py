import pytest

from conftest import load

paths = load("spark_runtime.paths")
profiles = load("spark_runtime.profiles")


@pytest.mark.parametrize(
    "win,wsl",
    [
        (r"C:\models\SparkDiffusion", "/mnt/c/models/SparkDiffusion"),
        (r"E:\AI\custom_nodes\x", "/mnt/e/AI/custom_nodes/x"),
        (r"d:\AI\Wan", "/mnt/d/AI/Wan"),
        ("Z:/forward/slashes/ok", "/mnt/z/forward/slashes/ok"),
        (r"C:\path with spaces\モデル\ファイル.pth", "/mnt/c/path with spaces/モデル/ファイル.pth"),
        (r"C:\trailing\\", "/mnt/c/trailing"),
        ("C:\\", "/mnt/c"),
        ("C:", "/mnt/c"),
        (r"C:\double\\\\sep", "/mnt/c/double/sep"),
    ],
)
def test_windows_to_wsl(win, wsl):
    assert paths.windows_to_wsl(win) == wsl


def test_drive_letter_not_hardcoded():
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        assert paths.windows_to_wsl(f"{letter}:\\x") == f"/mnt/{letter.lower()}/x"


def test_custom_automount_root():
    assert paths.windows_to_wsl(r"E:\a b", automount_root="/") == "/e/a b"
    assert paths.windows_to_wsl(r"E:\a", automount_root="/win") == "/win/e/a"


@pytest.mark.parametrize(
    "unc",
    [
        r"\\wsl$\Ubuntu\home\user\models",
        r"\\wsl.localhost\Ubuntu\home\user\models",
        "//wsl.localhost/Ubuntu/home/user/models",
    ],
)
def test_wsl_unc_paths(unc):
    assert paths.windows_to_wsl(unc) == "/home/user/models"
    assert paths.windows_to_wsl(unc, distro="ubuntu") == "/home/user/models"
    with pytest.raises(paths.PathConversionError):
        paths.windows_to_wsl(unc, distro="Debian")


def test_posix_passthrough_and_errors():
    assert paths.windows_to_wsl("/opt/sparkdiffusion/venv/bin/python") == "/opt/sparkdiffusion/venv/bin/python"
    with pytest.raises(paths.PathConversionError):
        paths.windows_to_wsl("relative\\path")
    with pytest.raises(paths.PathConversionError):
        paths.windows_to_wsl(r"\\server\share\x")
    with pytest.raises(paths.PathConversionError):
        paths.windows_to_wsl("")


def test_wsl_to_windows_roundtrip():
    for win in [r"C:\models\a b\c", r"E:\データ\x.pth"]:
        assert paths.wsl_to_windows(paths.windows_to_wsl(win)) == win
    assert paths.wsl_to_windows("/opt/x", distro="Ubuntu") == r"\\wsl.localhost\Ubuntu\opt\x"
    with pytest.raises(paths.PathConversionError):
        paths.wsl_to_windows("/opt/x")


def test_to_runtime_path_by_backend():
    assert paths.to_runtime_path(r"C:\x", paths.BACKEND_WSL2) == "/mnt/c/x"
    assert paths.to_runtime_path(r"C:\x", paths.BACKEND_WINDOWS) == r"C:\x"
    assert paths.to_runtime_path("/x", paths.BACKEND_LINUX) == "/x"


def test_detect_backend():
    assert paths.detect_backend("nt") == paths.BACKEND_WINDOWS
    assert paths.detect_backend("posix") == paths.BACKEND_LINUX


def test_basename_any():
    assert paths.basename_any(r"C:\a\b\c.pth") == "c.pth"
    assert paths.basename_any("/mnt/e/a/b.mp4") == "b.mp4"
    assert paths.basename_any("") == ""


def test_redact_paths():
    text = r"loading C:\Users\someone\models\x.pth and /home/someone/w/y.pth and /mnt/e/out/z.mp4 ok"
    out = paths.redact_paths(text)
    assert "someone" not in out
    assert "<path:x.pth>" in out and "<path:y.pth>" in out and "<path:z.mp4>" in out
    assert out.endswith(" ok")
    assert paths.redact_paths(r"\\wsl.localhost\Ubuntu\opt\m") == "<path:m>"


def _make_assets(root, wan_dir="Wan2.1-T2V-1.3B", clip=False):
    d = root / wan_dir
    (d / "google" / "umt5-xxl").mkdir(parents=True)
    (d / "Wan2.1_VAE.pth").write_bytes(b"x")
    (d / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"x")
    if clip:
        (d / profiles.CLIP_FILE).write_bytes(b"x")


def test_resolve_assets_found(tmp_path):
    root = tmp_path / "モデル dir"
    p = profiles.get_profile("wan2.1-t2v-1.3b-480p-s90-4step")
    _make_assets(root)
    (root / p.repo_dirname).mkdir()
    (root / p.repo_dirname / p.checkpoint_files[0]).write_bytes(b"x")
    res = paths.resolve_assets(str(root), p)
    assert res.ok, res.missing
    assert res.checkpoints[0].endswith(p.checkpoint_files[0])
    assert res.tokenizer.replace("\\", "/").endswith("google/umt5-xxl")
    assert res.clip is None


def test_resolve_assets_shared_from_other_wan_repo_and_flat_checkpoint(tmp_path):
    p = profiles.get_profile("wan2.1-t2v-14b-720p-s95-3step")
    _make_assets(tmp_path, wan_dir="Wan2.1-T2V-1.3B")  # shared assets from the 1.3B repo
    (tmp_path / p.checkpoint_files[0]).write_bytes(b"x")  # checkpoint directly in the root
    res = paths.resolve_assets(str(tmp_path), p)
    assert res.ok, res.missing


def test_resolve_assets_missing(tmp_path):
    p = profiles.get_profile("wan2.1-i2v-14b-720p-s97-4step")
    res = paths.resolve_assets(str(tmp_path), p)
    assert not res.ok
    joined = "\n".join(res.missing)
    assert p.checkpoint_files[0] in joined and profiles.CLIP_FILE in joined and "Wan2.1_VAE.pth" in joined
    assert "huggingface.co/" in joined


def test_resolve_assets_i2v_clip(tmp_path):
    p = profiles.get_profile("wan2.1-i2v-14b-720p-s97-4step")
    _make_assets(tmp_path, wan_dir="Wan2.1-I2V-14B-720P", clip=True)
    (tmp_path / p.checkpoint_files[0]).write_bytes(b"x")
    res = paths.resolve_assets(str(tmp_path), p)
    assert res.ok and res.clip.endswith(profiles.CLIP_FILE)


def test_resolve_assets_bad_root(tmp_path):
    res = paths.resolve_assets(str(tmp_path / "missing"), profiles.get_profile("wan2.1-t2v-1.3b-480p-s90-4step"))
    assert not res.ok and "model root" in res.missing[0]


@pytest.mark.parametrize("mapped,root", [("/mnt/c", "/mnt/"), ("/c", "/"), ("/win/c/", "/win/"), ("weird", "/mnt/")])
def test_detect_automount_root(monkeypatch, mapped, root):
    monkeypatch.setenv("SystemDrive", "C:")
    monkeypatch.setattr(paths, "wslpath_via_wsl", lambda path, distro=None: mapped)
    paths.detect_automount_root.cache_clear()
    assert paths.detect_automount_root("Ubuntu") == root
    paths.detect_automount_root.cache_clear()


def test_detect_automount_root_falls_back(monkeypatch):
    def boom(*_a, **_k):
        raise OSError("wsl.exe not found")

    monkeypatch.setattr(paths, "wslpath_via_wsl", boom)
    paths.detect_automount_root.cache_clear()
    assert paths.detect_automount_root("Ubuntu") == paths.DEFAULT_WSL_AUTOMOUNT_ROOT
    paths.detect_automount_root.cache_clear()


def test_system_executable(monkeypatch):
    monkeypatch.setattr(paths.shutil, "which", lambda name: "/usr/bin/" + name)
    assert paths.system_executable("wsl.exe") == "/usr/bin/wsl.exe"
    monkeypatch.setattr(paths.shutil, "which", lambda name: None)
    monkeypatch.setattr(paths.os, "name", "posix")
    assert paths.system_executable("taskkill") == "taskkill"
