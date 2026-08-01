"""Background work the UI can poll: local subprocesses (build / prepare / train)
and in-process threads (ssh provisioning, uploads). Each task keeps a log
buffer and a state; the UI polls /api/tasks/<id>."""
from __future__ import annotations

import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from ..paths import ROOT, RUNS_DIR

_tasks: dict[str, "Task"] = {}
_lock = threading.Lock()


class Task:
    def __init__(self, kind: str, title: str):
        self.id = uuid.uuid4().hex[:10]
        self.kind = kind
        self.title = title
        self.state = "running"
        self.created = time.time()
        self.finished: float | None = None
        self.lines: deque[str] = deque(maxlen=4000)
        self.result = None
        self.error: str | None = None
        self.meta: dict = {}
        with _lock:
            _tasks[self.id] = self

    def log(self, line: str):
        self.lines.append(f"[{time.strftime('%H:%M:%S')}] {line}")

    def done(self, result=None):
        self.result = result
        self.state = "finished"
        self.finished = time.time()

    def fail(self, err: str):
        self.error = err
        self.state = "failed"
        self.finished = time.time()
        self.log("ERROR: " + err)

    def to_dict(self, tail: int = 400):
        lines = list(self.lines)
        return {"id": self.id, "kind": self.kind, "title": self.title, "state": self.state, "created": self.created,
                "finished": self.finished, "error": self.error, "result": self.result, "meta": self.meta,
                "lines": lines[-tail:], "n_lines": len(lines)}


def get(task_id: str) -> Task | None:
    return _tasks.get(task_id)


def list_tasks() -> list[dict]:
    return [t.to_dict(tail=1) for t in sorted(_tasks.values(), key=lambda t: -t.created)]


def run_thread(kind: str, title: str, fn, *args, **kwargs) -> Task:
    """fn(task, *args) runs in a thread, its return value becomes task.result"""
    t = Task(kind, title)

    def _go():
        try:
            res = fn(t, *args, **kwargs)
            if t.state == "running":
                t.done(res)
        except Exception as e:  # noqa: BLE001 - surfaced to the ui
            t.fail(f"{type(e).__name__}: {e}")

    threading.Thread(target=_go, daemon=True, name=f"task-{t.id}").start()
    return t


def run_subprocess(kind: str, title: str, args: list[str], log_file: Path | None = None, env: dict | None = None) -> Task:
    """python -m something as a child process, output streamed into the task log"""
    t = Task(kind, title)
    cmd = [sys.executable, "-m", *args]
    t.meta["cmd"] = " ".join(cmd)

    def _go():
        import os

        full_env = os.environ.copy()
        full_env.update(env or {})
        full_env.setdefault("PYTHONUNBUFFERED", "1")
        lf = open(log_file, "a") if log_file else None
        try:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=full_env)
            t.meta["pid"] = proc.pid
            for line in proc.stdout:
                line = line.rstrip("\n")
                t.lines.append(line)
                if lf:
                    lf.write(line + "\n")
                    lf.flush()
            rc = proc.wait()
            if rc == 0:
                t.done({"returncode": 0})
            else:
                t.fail(f"exited with {rc}")
        except Exception as e:  # noqa: BLE001
            t.fail(str(e))
        finally:
            if lf:
                lf.close()

    threading.Thread(target=_go, daemon=True, name=f"task-{t.id}").start()
    return t


def run_name_from(prefix: str) -> str:
    base = f"{prefix}-{time.strftime('%Y%m%d-%H%M')}"
    name, i = base, 2
    while (RUNS_DIR / name).exists():
        name = f"{base}-{i}"
        i += 1
    return name
