"""The app's own ssh identity. Generated once into ~/.fraudvae/ssh/, the public
half is what you paste into vast.ai (or authorized_keys) so the app can log in
without a password. Private key never leaves this machine and is never in git."""
from __future__ import annotations

import os
import socket
import stat
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ..paths import APP_HOME

KEY_DIR = APP_HOME / "ssh"
PRIVATE = KEY_DIR / "id_ed25519"
PUBLIC = KEY_DIR / "id_ed25519.pub"
KNOWN_HOSTS = KEY_DIR / "known_hosts"


def ensure_keypair(comment: str | None = None) -> Path:
    if PRIVATE.exists() and PUBLIC.exists():
        return PRIVATE
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(KEY_DIR, 0o700)
    key = Ed25519PrivateKey.generate()
    priv = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.OpenSSH,
        serialization.NoEncryption(),
    )
    pub = key.public_key().public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH)
    comment = comment or f"fraudvae@{socket.gethostname()}"
    PRIVATE.write_bytes(priv)
    os.chmod(PRIVATE, stat.S_IRUSR | stat.S_IWUSR)
    PUBLIC.write_text(pub.decode() + " " + comment + "\n")
    return PRIVATE


def public_key_text() -> str:
    ensure_keypair()
    return PUBLIC.read_text().strip()


def fingerprint() -> str:
    import base64
    import hashlib

    blob = public_key_text().split()[1]
    digest = hashlib.sha256(base64.b64decode(blob)).digest()
    return "SHA256:" + base64.b64encode(digest).decode().rstrip("=")


def rotate() -> str:
    """throw the old key away and make a new one. you have to re-add the pubkey remotely"""
    for p in (PRIVATE, PUBLIC):
        if p.exists():
            p.unlink()
    ensure_keypair()
    return public_key_text()
