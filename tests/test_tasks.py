"""the background task registry: finished tasks are pruned, a child process can be cancelled
with its whole process group, and a cancelled task says so"""
import time

from fraudvae.server import tasks


def _wait(t, seconds=15):
    for _ in range(int(seconds * 20)):
        if t.state != "running":
            return t
        time.sleep(0.05)
    raise AssertionError(f"task {t.id} still running: {t.state}")


def test_finished_tasks_are_pruned(monkeypatch):
    monkeypatch.setattr(tasks, "_tasks", {})
    monkeypatch.setattr(tasks, "KEEP_FINISHED", 3)
    done = [tasks.run_thread("t", f"quick {i}", lambda t: i) for i in range(5)]
    for t in done:
        _wait(t)
    running = tasks.run_thread("t", "slow", lambda t: time.sleep(0.3))
    tasks.Task("t", "trigger")   # creating a task prunes: at most KEEP_FINISHED finished ones stay
    kept = set(tasks._tasks)
    assert running.id in kept and len([t for t in done if t.id in kept]) == 3
    assert all(t.id in kept for t in done[-3:])
    _wait(running)
    # an old finished task goes too
    monkeypatch.setattr(tasks, "KEEP_SECONDS", 0.1)
    time.sleep(0.2)
    tasks.Task("t", "trigger again")
    assert not any(t.id in tasks._tasks for t in done)


def test_cancel_signals_the_process_group(monkeypatch):
    import os

    monkeypatch.setattr(tasks, "_tasks", {})
    # a child that runs until told to stop (a server on a random local port)
    t = tasks.run_subprocess("train", "sleeper", ["http.server", "0", "--bind", "127.0.0.1"])
    for _ in range(200):
        if t.meta.get("pid") and t.lines:
            break
        time.sleep(0.05)
    pid = t.meta["pid"]
    assert os.getpgid(pid) == pid                      # its own session: the cancel reaches every descendant
    assert t.cancel() == "signalled"
    _wait(t)
    assert t.state == "failed" and t.error.startswith("cancelled")
    assert t.cancel() == "already finished"
    for _ in range(100):
        try:
            os.kill(pid, 0)
            time.sleep(0.05)
        except ProcessLookupError:
            break
    else:
        raise AssertionError("child survived the cancel")
