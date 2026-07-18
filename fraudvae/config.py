"""Training config. YAML in, dataclass out. Nothing fancy."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .paths import CONFIG_DIR


@dataclass
class ModelConfig:
    latent_dim: int = 20
    hidden: list[int] = field(default_factory=lambda: [512, 512, 256])
    cat_embed_dim: int = 8
    source_embed_dim: int = 6
    dropout: float = 0.05
    min_logvar: float = -7.0
    max_logvar: float = 4.0


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 1e-6
    beta: float = 1.0
    beta_warmup_epochs: int = 4   # linear ramp 0 -> beta
    free_bits: float = 0.0
    grad_clip: float = 5.0
    source_dropout: float = 0.1   # p of replacing the source token with "generic"
    num_workers: int = 0
    seed: int = 1337
    amp: bool = True              # bf16 autocast on cuda, no-op on cpu
    eval_every: int = 1
    patience: int = 3
    score_samples: int = 8


@dataclass
class DataConfig:
    max_rows_per_source: int | None = None   # subsample legit rows for training
    sources: list[str] | None = None         # None = everything that was built
    val_fraction: float = 0.10
    test_fraction: float = 0.20


@dataclass
class Config:
    name: str = "base"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    device: str = "auto"
    notes: str = ""


def _merge(dc, d: dict):
    for k, v in (d or {}).items():
        if not hasattr(dc, k):
            raise KeyError(f"unknown config key: {k}")
        cur = getattr(dc, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(dc, k, v)
    return dc


def load_config(path: str | Path | None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists() and (CONFIG_DIR / f"{path}.yaml").exists():
        p = CONFIG_DIR / f"{path}.yaml"
    with open(p) as f:
        raw = yaml.safe_load(f) or {}
    _merge(cfg, raw)
    if cfg.name == "base":
        cfg.name = p.stem
    return cfg


def to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)
