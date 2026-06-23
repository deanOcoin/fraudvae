from importlib import import_module

from ..registry import ORDER


def get_adapter(key: str):
    """adapters are modules exposing load(raw_dir) -> unified DataFrame"""
    if key not in ORDER:
        raise KeyError(key)
    return import_module(f"fraudvae.data.adapters.{key}")
