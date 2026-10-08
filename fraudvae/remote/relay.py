"""The laptop-off control channel: a private git repo as a mailbox between this side and the
box (scripts/relay_agent.sh runs there). `run` pushes a command script and waits for the
agent to push its output back; `put` ships a file. Latency is a poll interval or two, which
is fine for launching and watching jobs that take hours. Nothing in here is the project's
own history: the relay repo is a throwaway.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path


class RelayError(RuntimeError):
    pass


class Relay:
    def __init__(self, repo_dir: Path, remote_url: str | None = None, log=print):
        self.dir = Path(repo_dir)
        self.log = log
        if not (self.dir / ".git").exists():
            if not remote_url:
                raise RelayError(f"no clone at {self.dir} and no remote url to clone from")
            self._git(["clone", "-q", remote_url, str(self.dir)], cwd=self.dir.parent)
        for d in ("cmd", "out", "files"):
            (self.dir / d).mkdir(exist_ok=True)
        self._git(["config", "user.email", "relay@fraudvae.local"])
        self._git(["config", "user.name", "fraudvae relay"])
        self._git(["config", "pull.rebase", "true"])

    def _git(self, args, cwd=None, check=True) -> str:
        r = subprocess.run(["git", *args], cwd=str(cwd or self.dir), capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RelayError(f"git {' '.join(args)}: {r.stderr.strip()[-500:]}")
        return r.stdout

    def pull(self):
        try:
            self._git(["pull", "-q"])
        except RelayError as e:
            self.log(f"relay pull: {e}")
            self._git(["rebase", "--abort"], check=False)
            self._git(["reset", "-q", "--hard", "origin/main"], check=False)

    def _push(self, message: str):
        self._git(["add", "-A"])
        self._git(["commit", "-qm", message], check=False)
        for _ in range(4):
            r = subprocess.run(["git", "push", "-q"], cwd=str(self.dir), capture_output=True, text=True)
            if r.returncode == 0:
                return
            self.pull()
            time.sleep(2)
        raise RelayError("could not push to the relay repo")

    def next_id(self, name: str) -> str:
        n = len(list((self.dir / "cmd").glob("*.sh"))) + 1
        slug = "".join(c if c.isalnum() else "-" for c in name)[:40].strip("-") or "cmd"
        return f"{n:04d}-{slug}"

    def put(self, local: Path, rel: str) -> str:
        """ship a file to the box (lands under files/ in the box's clone after its next pull)"""
        dest = self.dir / "files" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dest)
        self._push(f"file {rel}")
        return f"files/{rel}"

    def submit(self, cmd: str, name: str = "cmd") -> str:
        self.pull()
        cid = self.next_id(name)
        (self.dir / "cmd" / f"{cid}.sh").write_text(cmd if cmd.endswith("\n") else cmd + "\n")
        self._push(cid)
        return cid

    def result(self, cid: str) -> dict | None:
        done = self.dir / "out" / f"{cid}.done"
        if not done.exists():
            return None
        rc_p = self.dir / "out" / f"{cid}.rc"
        log_p = self.dir / "out" / f"{cid}.log"
        return {"id": cid, "rc": int(rc_p.read_text().strip() or -1) if rc_p.exists() else -1,
                "log": log_p.read_text(errors="replace") if log_p.exists() else "", "finished": done.read_text().strip()}

    def wait(self, cid: str, timeout: float = 3600, poll: float = 15) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.pull()
            r = self.result(cid)
            if r is not None:
                return r
            time.sleep(poll)
        raise RelayError(f"{cid}: no result after {timeout:.0f}s (is the agent running on the box?)")

    def run(self, cmd: str, name: str = "cmd", timeout: float = 3600, poll: float = 15) -> dict:
        cid = self.submit(cmd, name)
        return self.wait(cid, timeout=timeout, poll=poll)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="fraudvae remote relay", description=__doc__)
    ap.add_argument("--repo", required=True, help="local clone directory")
    ap.add_argument("--url", default=None, help="remote url, for the first clone")
    ap.add_argument("--timeout", type=float, default=3600)
    sub = ap.add_subparsers(dest="what", required=True)
    p = sub.add_parser("run"); p.add_argument("cmd"); p.add_argument("--name", default="cmd")
    p = sub.add_parser("put"); p.add_argument("local"); p.add_argument("rel")
    a = ap.parse_args(argv)
    relay = Relay(Path(a.repo), a.url)
    if a.what == "run":
        r = relay.run(a.cmd, a.name, timeout=a.timeout)
        print(r["log"], end="")
        return r["rc"]
    print(relay.put(Path(a.local), a.rel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
