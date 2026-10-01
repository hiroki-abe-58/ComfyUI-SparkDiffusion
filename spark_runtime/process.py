"""Run the runtime launcher as a child process: streaming, timeout, cancellation.

* argument-list ``Popen`` only (never ``shell=True``)
* stdout/stderr are read by two threads and forwarded line by line
* ``@@SPARK@@ {json}`` lines become structured events (progress, phases, result)
* cancellation creates the launcher's cancel file first (clean exit, also inside
  WSL2), then kills the whole process tree after a grace period
"""

from __future__ import annotations

import json
import logging
import os
import queue
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Sequence

from .command import describe_command, mask_secrets
from .paths import redact_paths

EVENT_PREFIX = "@@SPARK@@ "
LOG = logging.getLogger("ComfyUI-SparkDiffusion")

EventCallback = Callable[[dict], None]
LineCallback = Callable[[str, str], None]


class SparkProcessError(RuntimeError):
    """The runtime process failed; carries everything needed for a useful report."""

    def __init__(
        self,
        message: str,
        *,
        exit_code: Optional[int],
        command: Sequence[str],
        log_tail: List[str],
        backend: str = "",
        profile: str = "",
        suggestion: str = "",
    ):
        self.exit_code = exit_code
        self.command = list(command)
        self.log_tail = log_tail
        self.backend = backend
        self.profile = profile
        self.suggestion = suggestion
        self.summary = message
        super().__init__(self._render())

    def _render(self) -> str:
        lines = [f"SparkDiffusion runtime failed: {self.summary}"]
        if self.backend:
            lines.append(f"  backend   : {self.backend}")
        if self.profile:
            lines.append(f"  profile   : {self.profile}")
        lines.append(f"  exit code : {self.exit_code}")
        lines.append(f"  command   : {describe_command(self.command)}")
        if self.suggestion:
            lines.append(f"  next step : {self.suggestion}")
        if self.log_tail:
            lines.append("  log tail  :")
            lines.extend("    " + line for line in self.log_tail)
        return "\n".join(lines)


class SparkCancelled(SparkProcessError):
    pass


class SparkTimeout(SparkProcessError):
    pass


@dataclass
class ProcessResult:
    exit_code: int
    duration_s: float
    events: List[dict] = field(default_factory=list)
    log_tail: List[str] = field(default_factory=list)
    stderr_tail: List[str] = field(default_factory=list)

    def error_tail(self, n: int = 25) -> List[str]:
        """Combined tail, plus stderr lines (tracebacks) that later stdout output pushed out of it."""
        tail = self.log_tail[-n:]
        missing = [line for line in self.stderr_tail if line not in tail]
        return tail + (["--- stderr ---"] + missing[-n:] if missing else [])

    def last_event(self, name: str) -> Optional[dict]:
        for ev in reversed(self.events):
            if ev.get("event") == name:
                return ev
        return None


def parse_event(line: str) -> Optional[dict]:
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        payload = json.loads(line[len(EVENT_PREFIX) :])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _split_lines(chunk: str) -> List[str]:
    # tqdm redraws with "\r"; treat each redraw as its own line.
    return [p for p in chunk.replace("\r\n", "\n").replace("\r", "\n").split("\n")]


def _reader(stream, name: str, out: queue.Queue[tuple[str, Optional[str]]]) -> None:
    pending = ""
    try:
        while True:
            data = stream.read1(65536) if hasattr(stream, "read1") else stream.read(65536)
            if not data:
                break
            pending += data.decode("utf-8", errors="replace")
            parts = _split_lines(pending)
            pending = parts.pop()
            for line in parts:
                out.put((name, line))
    except (OSError, ValueError):
        pass
    finally:
        if pending:
            out.put((name, pending))
        out.put((name, None))


def kill_process_tree(pid: int, timeout: float = 5.0) -> None:
    """Terminate ``pid`` and all descendants (best effort, cross-platform)."""
    try:
        import psutil  # ComfyUI depends on psutil; optional elsewhere
    except ImportError:
        psutil = None  # type: ignore[assignment]
    if psutil is not None:
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        procs = parent.children(recursive=True) + [parent]
        for p in procs:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(procs, timeout=timeout)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=timeout)
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_process(
    command: Sequence[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout_s: float = 0,
    cancel_check: Optional[Callable[[], bool]] = None,
    cancel_file: Optional[str] = None,
    extra_kill: Optional[Callable[[], None]] = None,
    on_event: Optional[EventCallback] = None,
    on_line: Optional[LineCallback] = None,
    tail_lines: int = 60,
    grace_s: float = 5.0,
    backend: str = "",
    profile: str = "",
) -> ProcessResult:
    """Run ``command`` to completion, streaming output.

    Raises :class:`SparkCancelled` / :class:`SparkTimeout` after the process
    tree is gone. A non-zero exit is *returned*, not raised, so callers can
    attach context (metadata, suggestions) before reporting it.
    """
    if isinstance(command, (str, bytes)):
        raise TypeError("command must be an argument list, not a string")
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = True
    start = time.time()
    proc = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        shell=False,
        **kwargs,
    )
    LOG.info("[SparkDiffusion] started runtime pid=%s: %s", proc.pid, describe_command(command))
    lines: queue.Queue[tuple[str, Optional[str]]] = queue.Queue()
    readers = [
        threading.Thread(target=_reader, args=(proc.stdout, "stdout", lines), daemon=True),
        threading.Thread(target=_reader, args=(proc.stderr, "stderr", lines), daemon=True),
    ]
    for t in readers:
        t.start()

    tail: Deque[str] = deque(maxlen=tail_lines)
    err_tail: Deque[str] = deque(maxlen=tail_lines)
    events: List[dict] = []
    open_streams = 2
    stop_reason: Optional[str] = None

    def handle(name: str, line: str) -> None:
        ev = parse_event(line)
        if ev is not None:
            events.append(ev)
            if on_event:
                try:
                    on_event(ev)
                except Exception:  # a UI callback must never kill the run
                    LOG.exception("[SparkDiffusion] event callback failed")
            return
        if not line.strip():
            return
        clean = mask_secrets(line)
        if not clean.lstrip().startswith("Sampling:"):
            tail.append(redact_paths(clean))
            if name == "stderr":
                err_tail.append(redact_paths(clean))
        if on_line:
            on_line(name, clean)
        else:
            LOG.info("[SparkDiffusion] %s", clean)

    try:
        while open_streams or proc.poll() is None:
            try:
                name, line = lines.get(timeout=0.2)
                if line is None:
                    open_streams -= 1
                else:
                    handle(name, line)
            except queue.Empty:
                pass
            if stop_reason is None:
                if cancel_check is not None and cancel_check():
                    stop_reason = "cancelled"
                elif timeout_s and time.time() - start > timeout_s:
                    stop_reason = "timeout"
                if stop_reason:
                    _stop(proc, cancel_file, extra_kill, grace_s)
    except BaseException:
        _stop(proc, cancel_file, extra_kill, grace_s)
        raise
    finally:
        for t in readers:
            t.join(timeout=2)
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass

    exit_code = proc.wait()
    duration = time.time() - start
    if stop_reason == "cancelled":
        raise SparkCancelled(
            "cancelled by user",
            exit_code=exit_code,
            command=command,
            log_tail=list(tail)[-15:],
            backend=backend,
            profile=profile,
        )
    if stop_reason == "timeout":
        raise SparkTimeout(
            f"timed out after {timeout_s:.0f}s",
            exit_code=exit_code,
            command=command,
            log_tail=list(tail),
            backend=backend,
            profile=profile,
            suggestion="increase the timeout on the Runtime node, or check the log for a hang",
        )
    return ProcessResult(
        exit_code=exit_code, duration_s=duration, events=events, log_tail=list(tail), stderr_tail=list(err_tail)
    )


def _stop(
    proc: subprocess.Popen, cancel_file: Optional[str], extra_kill: Optional[Callable[[], None]], grace_s: float
) -> None:
    if proc.poll() is not None:
        if extra_kill:
            _safe(extra_kill)
        return
    if cancel_file:
        try:
            with open(cancel_file, "w", encoding="utf-8") as f:
                f.write("cancel\n")
        except OSError:
            pass
        deadline = time.time() + grace_s
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.1)
    if extra_kill:
        _safe(extra_kill)
    if proc.poll() is None:
        kill_process_tree(proc.pid)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _safe(fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception:
        LOG.exception("[SparkDiffusion] cleanup hook failed")


def wsl_kill_hook(distro: str, pid_file: str) -> Callable[[], None]:
    """Kill the launcher's process group inside WSL (``wsl.exe`` dying does not do that)."""

    def _kill() -> None:
        try:
            with open(pid_file, encoding="utf-8") as f:
                pid = int(f.read().strip())
        except (OSError, ValueError):
            return
        cmd = ["wsl.exe"] + (["-d", distro] if distro else []) + ["--exec", "kill", "-KILL", f"-{pid}"]
        subprocess.run(cmd, capture_output=True, timeout=20, check=False)

    return _kill
