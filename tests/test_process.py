import os
import sys
import textwrap
import time

import pytest

from conftest import load

process = load("spark_runtime.process")


def script(tmp_path, body: str) -> str:
    path = tmp_path / "child.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


def py(path, *args):
    return [sys.executable, "-X", "utf8", path, *args]


def test_streams_lines_and_events(tmp_path):
    s = script(
        tmp_path,
        """
        import json, sys
        print("hello 世界", flush=True)
        print("@@SPARK@@ " + json.dumps({"event": "progress", "step": 1, "total": 2}), flush=True)
        print("warn", file=sys.stderr, flush=True)
        sys.stdout.write("tqdm 1\\rtqdm 2\\n"); sys.stdout.flush()
    """,
    )
    lines, events = [], []
    res = process.run_process(py(s), on_line=lambda name, line: lines.append((name, line)), on_event=events.append)
    assert res.exit_code == 0
    assert ("stdout", "hello 世界") in lines and ("stderr", "warn") in lines
    assert ("stdout", "tqdm 1") in lines and ("stdout", "tqdm 2") in lines
    assert events == [{"event": "progress", "step": 1, "total": 2}]
    assert res.last_event("progress")["step"] == 1


def test_nonzero_exit_is_returned_with_tail(tmp_path):
    s = script(
        tmp_path,
        """
        import sys
        for i in range(100):
            print(f"line {i}")
        print("Traceback: boom", file=sys.stderr)
        sys.exit(3)
    """,
    )
    res = process.run_process(py(s), on_line=lambda *_: None, tail_lines=10)
    assert res.exit_code == 3
    assert len(res.log_tail) == 10
    assert res.stderr_tail == ["Traceback: boom"]
    assert "Traceback: boom" in res.error_tail(5)


def test_tail_redacts_paths_and_secrets(tmp_path):
    s = script(
        tmp_path,
        r"""
        print(r"loading C:\Users\someone\secret\model.pth")
        print("token=" + "hf_" + "y" * 30)
    """,
    )
    res = process.run_process(py(s), on_line=lambda *_: None)
    joined = "\n".join(res.log_tail)
    assert "someone" not in joined and "<path:model.pth>" in joined
    assert "hf_yyy" not in joined


def test_string_command_rejected():
    with pytest.raises(TypeError):
        process.run_process("python -c 'print(1)'")


def test_shell_metacharacters_are_literal(tmp_path):
    s = script(
        tmp_path,
        """
        import json, sys
        print("@@SPARK@@ " + json.dumps({"event": "args", "argv": sys.argv[1:]}), flush=True)
    """,
    )
    evil = "x && echo pwned > pwned.txt; $(whoami) | more"
    events = []
    res = process.run_process(
        py(s, evil, "日本語 arg"), on_event=events.append, on_line=lambda *_: None, cwd=str(tmp_path)
    )
    assert res.exit_code == 0
    assert events[0]["argv"] == [evil, "日本語 arg"]
    assert not (tmp_path / "pwned.txt").exists()


def test_timeout_kills_and_raises(tmp_path):
    s = script(
        tmp_path,
        """
        import time
        print("started", flush=True)
        time.sleep(60)
    """,
    )
    t = time.time()
    with pytest.raises(process.SparkTimeout) as exc:
        process.run_process(py(s), timeout_s=1.5, on_line=lambda *_: None, grace_s=0.5)
    assert time.time() - t < 30
    assert "timed out" in str(exc.value)


def _alive(pid: int) -> bool:
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    if os.name == "nt":
        import subprocess

        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True).stdout
        return str(pid) in out.decode("ascii", "replace")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/status") as f:
            return "zombie" not in f.read().lower()
    except OSError:
        return True


def test_cancel_kills_process_tree(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    s = script(
        tmp_path,
        f"""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        open(r"{pidfile}", "w").write(str(child.pid))
        print("spawned", flush=True)
        time.sleep(120)
    """,
    )
    start = time.time()

    def cancel():
        return pidfile.exists() and time.time() - start > 1.0

    with pytest.raises(process.SparkCancelled):
        process.run_process(py(s), cancel_check=cancel, on_line=lambda *_: None, grace_s=0.5)
    grandchild = int(pidfile.read_text())
    deadline = time.time() + 15
    while time.time() < deadline and _alive(grandchild):
        time.sleep(0.2)
    assert not _alive(grandchild), "grandchild survived cancellation"


def test_cancel_file_allows_clean_exit(tmp_path):
    cancel_file = tmp_path / "CANCEL"
    s = script(
        tmp_path,
        f"""
        import os, sys, time
        print("ready", flush=True)
        while not os.path.exists(r"{cancel_file}"):
            time.sleep(0.05)
        print("clean exit", flush=True)
        sys.exit(130)
    """,
    )
    lines = []
    start = time.time()
    with pytest.raises(process.SparkCancelled) as exc:
        process.run_process(
            py(s),
            cancel_check=lambda: time.time() - start > 1.0,
            cancel_file=str(cancel_file),
            on_line=lambda _name, line: lines.append(line),
            grace_s=10,
        )
    assert exc.value.exit_code == 130
    assert "clean exit" in lines


def test_error_rendering():
    err = process.SparkProcessError(
        "boom",
        exit_code=2,
        command=[r"C:\Users\me\py.exe", "x"],
        log_tail=["a", "b"],
        backend="wsl2",
        profile="p",
        suggestion="do x",
    )
    text = str(err)
    for part in ("boom", "wsl2", "exit code : 2", "py.exe", "do x", "    b"):
        assert part in text
    assert r"C:\Users\me" not in text


def test_parse_event():
    assert process.parse_event('@@SPARK@@ {"event": "x"}') == {"event": "x"}
    assert process.parse_event("@@SPARK@@ not json") is None
    assert process.parse_event("@@SPARK@@ [1]") is None
    assert process.parse_event("plain") is None
