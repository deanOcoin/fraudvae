"""Thin paramiko wrapper: connect with the app key, run commands, move files.

Host keys: first connection to a host is trust-on-first-use, the fingerprint is
stored in ~/.fraudvae/ssh/known_hosts and shown in the UI. A changed key on a
later connect is an error, not a prompt.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

import paramiko

from ..paths import APP_HOME
from . import keys

HOSTS_FILE = APP_HOME / "remote_hosts.json"


class RemoteError(RuntimeError):
    pass


@dataclass
class RemoteHost:
    host: str
    port: int = 22
    user: str = "root"
    key_path: str | None = None       # None -> the app's generated key
    label: str = ""
    remote_dir: str = "~/fraudvae"
    python: str = ""                  # filled in by provision
    last_used: float = 0.0
    fingerprint: str = ""

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return RemoteHost(**{k: v for k, v in d.items() if k in RemoteHost.__dataclass_fields__})


# ---- remembered hosts -------------------------------------------------------

def load_hosts() -> list[RemoteHost]:
    if not HOSTS_FILE.exists():
        return []
    try:
        return [RemoteHost.from_dict(d) for d in json.loads(HOSTS_FILE.read_text())]
    except (json.JSONDecodeError, TypeError):
        return []


def remember_host(h: RemoteHost):
    hosts = [x for x in load_hosts() if not (x.host == h.host and x.port == h.port and x.user == h.user)]
    h.last_used = time.time()
    hosts.insert(0, h)
    HOSTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    HOSTS_FILE.write_text(json.dumps([x.to_dict() for x in hosts[:20]], indent=1))


def forget_host(host: str, port: int, user: str):
    hosts = [x for x in load_hosts() if not (x.host == host and x.port == port and x.user == user)]
    HOSTS_FILE.write_text(json.dumps([x.to_dict() for x in hosts], indent=1))


def ssh_config_hosts() -> list[dict]:
    """Host entries from ~/.ssh/config so the form can autofill. read only"""
    cfg = Path.home() / ".ssh" / "config"
    if not cfg.exists():
        return []
    out = []
    cur = None
    for line in cfg.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"(\w+)\s+(.*)", line)
        if not m:
            continue
        k, v = m.group(1).lower(), m.group(2).strip()
        if k == "host":
            if cur and "*" not in cur["alias"]:
                out.append(cur)
            cur = {"alias": v, "host": v, "port": 22, "user": "", "key_path": None}
        elif cur is not None:
            if k == "hostname":
                cur["host"] = v
            elif k == "port":
                try:
                    cur["port"] = int(v)
                except ValueError:
                    pass
            elif k == "user":
                cur["user"] = v
            elif k == "identityfile":
                cur["key_path"] = os.path.expanduser(v)
    if cur and "*" not in cur["alias"]:
        out.append(cur)
    return out


# ---- the session ----------------------------------------------------------------

class _RecordingPolicy(paramiko.MissingHostKeyPolicy):
    def __init__(self, session):
        self.session = session

    def missing_host_key(self, client, hostname, key):
        self.session.fingerprint = _fp(key)
        self.session.first_seen = True
        client.get_host_keys().add(hostname, key.get_name(), key)
        keys.KEY_DIR.mkdir(parents=True, exist_ok=True)
        client.save_host_keys(str(keys.KNOWN_HOSTS))


def _fp(key) -> str:
    import base64

    return "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")


class SSHSession:
    def __init__(self, host: RemoteHost, timeout: float = 20.0):
        self.host = host
        self.timeout = timeout
        self.client: paramiko.SSHClient | None = None
        self._sftp: paramiko.SFTPClient | None = None
        self.fingerprint = ""
        self.first_seen = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    def connect(self):
        c = paramiko.SSHClient()
        if keys.KNOWN_HOSTS.exists():
            c.load_host_keys(str(keys.KNOWN_HOSTS))
        c.set_missing_host_key_policy(_RecordingPolicy(self))
        key_path = self.host.key_path or str(keys.ensure_keypair())
        try:
            pkey = _load_key(key_path)
            c.connect(self.host.host, port=self.host.port, username=self.host.user, pkey=pkey,
                      timeout=self.timeout, banner_timeout=self.timeout, auth_timeout=self.timeout,
                      allow_agent=False, look_for_keys=False)
        except paramiko.BadHostKeyException as e:
            raise RemoteError(f"host key for {self.host.host} changed! ({e}). remove it from {keys.KNOWN_HOSTS} if that's expected") from e
        except paramiko.AuthenticationException as e:
            raise RemoteError(
                f"authentication failed for {self.host.user}@{self.host.host}:{self.host.port}. "
                f"is the app's public key in the server's authorized_keys? ({keys.fingerprint()})") from e
        except (paramiko.SSHException, OSError) as e:
            raise RemoteError(f"could not connect to {self.host.host}:{self.host.port}: {e}") from e
        self.client = c
        if not self.fingerprint:
            hk = c.get_transport().get_remote_server_key()
            self.fingerprint = _fp(hk)
        self.host.fingerprint = self.fingerprint
        return self

    def close(self):
        if self._sftp:
            self._sftp.close()
            self._sftp = None
        if self.client:
            self.client.close()
            self.client = None

    @property
    def sftp(self) -> paramiko.SFTPClient:
        if self._sftp is None:
            self._sftp = self.client.open_sftp()
        return self._sftp

    # -- commands ------------------------------------------------------------------

    def run(self, cmd: str, timeout: float | None = 120, check=False) -> tuple[int, str, str]:
        _, stdout, stderr = self.client.exec_command(cmd, timeout=timeout, get_pty=False)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        rc = stdout.channel.recv_exit_status()
        if check and rc != 0:
            raise RemoteError(f"`{cmd}` failed ({rc}): {err.strip() or out.strip()}")
        return rc, out, err

    def stream(self, cmd: str) -> Iterator[str]:
        # yields lines as they arrive, stderr merged. polling with a sleep is dumb but paramiko has no select friendly api here
        chan = self.client.get_transport().open_session()
        chan.set_combine_stderr(True)
        chan.exec_command(cmd)
        buf = b""
        while True:
            if chan.recv_ready():
                buf += chan.recv(65536)
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    yield line.decode("utf-8", "replace")
            elif chan.exit_status_ready():
                break
            else:
                time.sleep(0.05)
        buf += chan.recv(65536)
        if buf:
            yield buf.decode("utf-8", "replace")

    def expand(self, path: str) -> str:
        if path.startswith("~"):
            rc, out, _ = self.run(f"echo {path}")
            return out.strip() or path
        return path

    # -- files -----------------------------------------------------------------------

    def exists(self, path: str) -> bool:
        try:
            self.sftp.stat(path)
            return True
        except FileNotFoundError:
            return False

    def mkdirs(self, path: str):
        self.run(f"mkdir -p {shlex.quote(path)}", check=True)

    def remote_sha256(self, path: str) -> str | None:
        rc, out, _ = self.run(f"sha256sum {shlex.quote(path)} 2>/dev/null | cut -d' ' -f1", timeout=600)
        return out.strip() or None

    def upload(self, local: Path, remote: str, progress: Callable[[int, int], None] | None = None, skip_same_size=True) -> bool:
        """returns True if bytes were actually sent (False = remote already had it)"""
        local = Path(local)
        if skip_same_size and self.exists(remote):
            if self.sftp.stat(remote).st_size == local.stat().st_size:
                return False
        self.sftp.put(str(local), remote, callback=progress)
        return True

    def download(self, remote: str, local: Path, progress: Callable[[int, int], None] | None = None):
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        self.sftp.get(remote, str(local), callback=progress)

    def read_text(self, remote: str) -> str:
        with self.sftp.open(remote, "r") as f:
            return f.read().decode("utf-8", "replace")


def _load_key(path: str):
    for cls in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
        try:
            return cls.from_private_key_file(path)
        except paramiko.SSHException:
            continue
    raise RemoteError(f"could not load private key {path}")
