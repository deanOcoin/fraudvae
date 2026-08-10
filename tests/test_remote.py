"""ssh layer against an in-process asyncssh server. No real network, no real GPU."""
import asyncio
import os
import threading
import time
from pathlib import Path

import pytest

asyncssh = pytest.importorskip("asyncssh")

from fraudvae.remote import keys  # noqa: E402
from fraudvae.remote.ssh import RemoteHost, SSHSession, ssh_config_hosts  # noqa: E402


def test_keypair_is_created_once():
    p = keys.ensure_keypair()
    assert p.exists() and keys.PUBLIC.exists()
    assert keys.public_key_text().startswith("ssh-ed25519 ")
    assert keys.fingerprint().startswith("SHA256:")
    assert oct(p.stat().st_mode & 0o777) == "0o600"
    first = keys.public_key_text()
    keys.ensure_keypair()
    assert keys.public_key_text() == first


def test_ssh_config_parsing(tmp_path, monkeypatch):
    cfg = tmp_path / ".ssh" / "config"
    cfg.parent.mkdir()
    cfg.write_text("Host gpu1\n  HostName 10.0.0.5\n  Port 2222\n  User ubuntu\n  IdentityFile ~/.ssh/gpu\nHost *\n  ServerAliveInterval 30\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    hosts = ssh_config_hosts()
    assert hosts == [{"alias": "gpu1", "host": "10.0.0.5", "port": 2222, "user": "ubuntu", "key_path": os.path.expanduser("~/.ssh/gpu")}]


class _Server(asyncssh.SSHServer):
    def begin_auth(self, username):
        return True

    def public_key_auth_supported(self):
        return True

    def validate_public_key(self, username, key):
        allowed = asyncssh.import_public_key(keys.public_key_text())
        return key == allowed


def _handle(process):
    # a very small "shell": run the command through the real shell so echo/mkdir/sha256sum work
    import subprocess

    cmd = process.command or ""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    process.stdout.write(r.stdout)
    process.stderr.write(r.stderr)
    process.exit(r.returncode)


@pytest.fixture(scope="module")
def mock_server(tmp_path_factory):
    keys.ensure_keypair()
    host_key = asyncssh.generate_private_key("ssh-ed25519")
    loop = asyncio.new_event_loop()
    ready = {}

    stop = None

    async def start():
        nonlocal stop
        stop = asyncio.Event()
        srv = await asyncssh.create_server(_Server, "127.0.0.1", 0, server_host_keys=[host_key],
                                           process_factory=_handle, sftp_factory=True, allow_scp=False)
        ready["port"] = srv.sockets[0].getsockname()[1]
        await stop.wait()
        srv.close()
        await srv.wait_closed()

    t = threading.Thread(target=lambda: loop.run_until_complete(start()), daemon=True)
    t.start()
    for _ in range(100):
        if "port" in ready:
            break
        time.sleep(0.05)
    yield ready["port"]
    loop.call_soon_threadsafe(stop.set)
    t.join(timeout=5)


def test_session_run_upload_download(mock_server, tmp_path):
    host = RemoteHost("127.0.0.1", mock_server, os.environ.get("USER", "root"), remote_dir=str(tmp_path / "remote"))
    with SSHSession(host, timeout=10) as s:
        assert s.fingerprint.startswith("SHA256:")
        rc, out, _ = s.run("echo hello && exit 0")
        assert rc == 0 and out.strip() == "hello"
        rc, _, err = s.run("echo nope >&2; exit 3")
        assert rc == 3 and "nope" in err
        s.mkdirs(str(tmp_path / "remote" / "data"))
        local = tmp_path / "blob.bin"
        local.write_bytes(os.urandom(300_000))
        remote = str(tmp_path / "remote" / "data" / "blob.bin")
        assert s.upload(local, remote) is True
        assert s.upload(local, remote) is False          # same size -> skipped
        assert s.exists(remote)
        back = tmp_path / "back.bin"
        s.download(remote, back)
        assert back.read_bytes() == local.read_bytes()
        # v5: a partial remote file (dropped connection mid-upload) is finished, not restarted
        big = tmp_path / "big.bin"
        big.write_bytes(os.urandom(2_500_000))
        remote_big = str(tmp_path / "remote" / "data" / "big.bin")
        Path(remote_big).write_bytes(big.read_bytes()[:1_000_000])
        seen = []
        assert s.upload(big, remote_big, progress=lambda done, total: seen.append((done, total))) is True
        assert Path(remote_big).read_bytes() == big.read_bytes()
        assert seen and seen[0][0] > 1_000_000 and seen[-1] == (2_500_000, 2_500_000)
        lines = list(s.stream("for i in 1 2 3; do echo line$i; done"))
        assert lines == ["line1", "line2", "line3"]
        assert s.expand("~").startswith("/")
    # known_hosts got the pin
    assert keys.KNOWN_HOSTS.exists() and "127.0.0.1" in keys.KNOWN_HOSTS.read_text()


def test_probe_against_mock(mock_server, tmp_path):
    from fraudvae.remote.provision import probe

    host = RemoteHost("127.0.0.1", mock_server, os.environ.get("USER", "root"), remote_dir=str(tmp_path / "r"))
    with SSHSession(host, timeout=10) as s:
        info = probe(s)
    assert info["python"]
    assert "hostname" in info and isinstance(info["gpus"], list)
    assert info["torch"] is None or "torch" in info["torch"]


def test_wrong_key_is_a_clean_error(mock_server, tmp_path):
    from fraudvae.remote.ssh import RemoteError

    other = asyncssh.generate_private_key("ssh-ed25519")
    kp = tmp_path / "other_key"
    kp.write_bytes(other.export_private_key())
    host = RemoteHost("127.0.0.1", mock_server, os.environ.get("USER", "root"), key_path=str(kp))
    with pytest.raises(RemoteError, match="authentication failed"):
        SSHSession(host, timeout=10).connect()


def test_relay_round_trip(tmp_path):
    """v5: the git-mailbox channel. a bare repo stands in for github, the agent script runs
    against one clone, the client against another; a command goes out, its output comes back,
    a shipped file is visible to the command"""
    import subprocess

    from fraudvae.remote.relay import Relay

    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", "-q", str(bare), str(seed)], check=True, capture_output=True)
    (seed / "README.md").write_text("relay\n")
    subprocess.run(["git", "-C", str(seed), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(seed), "-c", "user.email=a@b", "-c", "user.name=x", "commit", "-qm", "init"], check=True)
    subprocess.run(["git", "-C", str(seed), "push", "-q", "origin", "HEAD:main"], check=True, capture_output=True)
    box = tmp_path / "box"
    subprocess.run(["git", "clone", "-q", str(bare), str(box)], check=True, capture_output=True)
    agent = subprocess.Popen(["bash", str(Path(__file__).resolve().parent.parent / "scripts" / "relay_agent.sh"), str(box)],
                             env={**os.environ, "POLL": "1"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        relay = Relay(tmp_path / "laptop", str(bare), log=lambda *_: None)
        relay.put(seed / "README.md", "code/README.md")
        out = relay.run("cat files/code/README.md; exit 3", name="hello", timeout=60, poll=1)
        assert out["rc"] == 3 and out["log"].strip() == "relay"
        out2 = relay.run("nohup sleep 1 >/dev/null 2>&1 & echo launched", name="bg", timeout=60, poll=1)
        assert out2["rc"] == 0 and out2["log"].strip() == "launched"
        assert out["id"].startswith("0001-hello") and out2["id"].startswith("0002-bg")
    finally:
        agent.terminate()
