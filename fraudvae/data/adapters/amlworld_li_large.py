"""IBM AMLworld LI-Large: the largest file of all: 16.7 GB of csv. The same generator and columns as HI-Small
(adapters/amlworld.py has the reading); its own source, because the files share no accounts.
Added for v6, where the transaction encoder pretrains on every row of every source."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from .amlworld import load_variant

KEY = "amlworld_li_large"
PREFIX = "LI-Large"


def load(raw_dir: Path, max_rows: int | None = None) -> Iterator[pd.DataFrame]:
    yield from load_variant(raw_dir, KEY, PREFIX, max_rows)
