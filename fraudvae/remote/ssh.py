"""Thin paramiko wrapper: connect with the app key, run commands.

Host keys: first connection to a host is trust-on-first-use, the fingerprint is
stored in ~/.fraudvae/ssh/known_hosts and shown in the UI. A changed key on a
later connect is an error, not a prompt.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

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
    fingerprint: str = ""

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return RemoteHost(**d)


# ---- remembered hosts -------------------------------------------------------

def load_hosts() -> list[RemoteHost]:
    if not HOSTS_FILE.exists():
        return []
    return [RemoteHost.from_dict(d) for d in json.loads(HOSTS_FILE.read_text())]


def remember_host(h: RemoteHost):
    hosts = [x for x in load_hosts() if x.host != h.host]
    hosts.insert(0, h)
    HOSTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    HOSTS_FILE.write_text(json.dumps([x.to_dict() for x in hosts], indent=1))


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
            c.connect(self.host.host, port=self.host.port, username=self.host.user,
                      key_filename=key_path, timeout=self.timeout,
                      allow_agent=False, look_for_keys=False)
        except paramiko.BadHostKeyException as e:
            raise RemoteError(f"host key for {self.host.host} changed! ({e})") from e
        except Exception as e:
            raise RemoteError(f"could not connect to {self.host.host}:{self.host.port}: {e}") from e
        self.client = c
        if not self.fingerprint:
            hk = c.get_transport().get_remote_server_key()
            self.fingerprint = _fp(hk)
        self.host.fingerprint = self.fingerprint
        return self

    def close(self):
        if self.client:
            self.client.close()
            self.client = None

    # -- commands ------------------------------------------------------------------

    def run(self, cmd: str, timeout: float | None = 60) -> tuple[int, str, str]:
        _, stdout, stderr = self.client.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode()
        err = stderr.read().decode()
        rc = stdout.channel.recv_exit_status()
        return rc, out, err
