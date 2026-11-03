from importlib import import_module

from .. import private
from ..registry import ORDER


def get_adapter(key: str):
    """adapters are modules exposing load(raw_dir) -> unified DataFrame; a private one (v7,
    FRAUDVAE_ADAPTERS) is read from its own directory"""
    if key not in ORDER:
        raise KeyError(key)
    if key in private.scan():
        return private.module(key)
    return import_module(f"fraudvae.data.adapters.{key}")
