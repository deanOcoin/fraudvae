"""Training config. YAML in, dataclass out. Nothing fancy."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    latent_dim: int = 16
    hidden: list[int] = field(default_factory=lambda: [256, 256])
    cat_embed_dim: int = 8
    dropout: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 1024
    lr: float = 1e-3
    beta: float = 1.0
    seed: int = 1337


@dataclass
class DataConfig:
    max_rows_per_source: int | None = None
    val_fraction: float = 0.10
    test_fraction: float = 0.20


@dataclass
class Config:
    name: str = "base"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    device: str = "auto"


def _merge(dc, d: dict):
    for k, v in d.items():
        cur = getattr(dc, k)
        if dataclasses.is_dataclass(cur):
            _merge(cur, v)
        else:
            setattr(dc, k, v)
    return dc


def load_config(path: str | Path | None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    with open(path) as f:
        raw = yaml.safe_load(f)
    _merge(cfg, raw)
    return cfg


def to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)
