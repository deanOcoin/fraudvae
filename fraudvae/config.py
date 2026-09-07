"""Training config. YAML in, dataclass out. Nothing fancy."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .paths import CONFIG_DIR


@dataclass
class HistoryConfig:
    """v5: the customer-history transformer that conditions the VAE (model/vae.py HistoryEncoder)"""
    enabled: bool = False
    window: int = 32          # previous transactions of the same customer the model gets to see
    d_model: int = 128
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1


@dataclass
class ModelConfig:
    latent_dim: int = 20
    hidden: list[int] = field(default_factory=lambda: [512, 512, 256])
    cat_embed_dim: int = 8
    source_embed_dim: int = 6
    dropout: float = 0.05
    min_logvar: float = -7.0
    max_logvar: float = 4.0
    history: HistoryConfig = field(default_factory=HistoryConfig)


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
    history_dropout: float = 0.1  # v5: p of hiding a row's history window during training
    num_workers: int = 0
    seed: int = 1337
    amp: bool = True              # bf16 autocast on cuda, no-op on cpu
    eval_every: int = 1
    patience: int = 3
    score_samples: int = 8


@dataclass
class BoostConfig:
    """the supervised half: a LightGBM model on the observable fields + customer context.
    trained on the labelled train split after the VAE, early stopped on val auc."""
    enabled: bool = True
    num_leaves: int = 127
    learning_rate: float = 0.05
    max_rounds: int = 2000
    early_stopping: int = 100
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    lambda_l2: float = 1.0
    max_legit_per_source: int = 600_000   # legit train rows sampled per source (all fraud rows are kept)
    val_legit_per_source: int = 150_000   # for early stopping
    generic_fraction: float = 0.15        # share of rows duplicated with the source blanked
    num_threads: int = 0                  # 0 = all cores
    specialists: bool = True              # v5: one extra booster per source with enough fraud, its own "local" signal
    specialist_min_fraud: int = 300       # a source needs at least this many labelled fraud rows in train to get one
    fast_dropout: float = 0.3             # v5: share of training rows whose one-week-delay label block (wide__*_fast)
                                          # is blanked, so a deployment whose labels take a month is a regime the
                                          # booster has seen rather than a NaN it never met. 0 = off
    history_features: bool = False        # v5: the frozen history encoder's read-out as booster features ("hist_*")


@dataclass
class GraphConfig:
    """v5: the entity graph module. a GraphSAGE encoder over each source's entity graph
    (customers, counterparties, devices, emails, addresses, cards as nodes; transactions as
    weighted co-occurrence edges) with a transaction head on the entities' embeddings,
    trained in the post stage. see fraudvae/graph/."""
    enabled: bool = True
    hidden: int = 64
    out_dim: int = 32
    layers: int = 2
    dropout: float = 0.1
    epochs: int = 8
    batch_size: int = 8192
    lr: float = 2e-3
    min_degree: int = 1            # entities seen fewer times than this in the snapshot are "unknown" (1 = any)
    max_nodes: int = 4_000_000     # per source; the busiest nodes are kept past this
    fit_fraction: float = 0.2      # last share of the train period the transaction head is fitted on
    max_legit_fit: int = 300_000   # legit fit rows per source (fraud rows all kept)
    label_delay_days: int = 30


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
    boost: BoostConfig = field(default_factory=BoostConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
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
