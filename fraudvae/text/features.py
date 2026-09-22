"""Text -> a bag of hashed features, fastText style: word unigrams, word bigrams and character
trigrams (with word-boundary markers), each hashed into a fixed table. No vocabulary to ship,
no tokenizer to version, works the same on a url, a bio and a 3,000-word posting, and
crc32 is stable across processes (python's hash() is salted per run)."""
from __future__ import annotations

import re
import zlib

import numpy as np

BUCKETS = 1 << 18   # 262k rows x 64 dims = 67 MB per head in fp32 (shipped as fp16); 2^20 cost accuracy nothing but 800 MB of run
MAX_TOKENS = 4000
_WORD = re.compile(r"[a-z0-9]+")


def tokens(text: str, max_words: int = 1500) -> list[str]:
    """the feature strings of one text (before hashing)"""
    t = (text or "").lower()
    words = _WORD.findall(t)[:max_words]
    feats = []
    prev = None
    for w in words:
        feats.append("w:" + w)
        if prev is not None:
            feats.append("b:" + prev + "_" + w)
        prev = w
        padded = "<" + w + ">"
        if len(padded) > 3:
            feats.extend("c:" + padded[i : i + 3] for i in range(len(padded) - 2))
        else:
            feats.append("c:" + padded)
    return feats[:MAX_TOKENS]


def hash_ids(feats: list[str], buckets: int = BUCKETS) -> np.ndarray:
    if not feats:
        return np.zeros(1, dtype=np.int64)   # bucket 0 stands for "empty text"
    return np.fromiter((zlib.crc32(f.encode("utf-8")) % buckets for f in feats), dtype=np.int64, count=len(feats))


def encode_batch(texts: list[str], buckets: int = BUCKETS) -> tuple[np.ndarray, np.ndarray]:
    """flat ids + offsets, the EmbeddingBag calling convention"""
    ids, offsets = [], [0]
    for t in texts:
        h = hash_ids(tokens(t), buckets)
        ids.append(h)
        offsets.append(offsets[-1] + len(h))
    return np.concatenate(ids) if ids else np.zeros(0, np.int64), np.asarray(offsets[:-1], dtype=np.int64)
