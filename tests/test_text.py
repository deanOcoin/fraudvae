"""v5 text module: hashing is stable and fastText-shaped, platt scaling calibrates, the head
learns a planted vocabulary"""
import numpy as np
import torch

from fraudvae.text.features import encode_batch, hash_ids, tokens
from fraudvae.text.model import TextHead, fit_platt


def test_tokens_and_hashing_are_stable():
    t = tokens("Verify your PayPal account: http://pay-pal.secure-login.ru/x?id=1")
    assert "w:verify" in t and "b:verify_your" in t and "c:<ve" in t and "c:al>" in t
    assert tokens("") == [] and hash_ids([]).tolist() == [0]
    a, b = hash_ids(tokens("hello world")), hash_ids(tokens("hello world"))
    assert a.tolist() == b.tolist() and a.max() < (1 << 20)
    ids, offsets = encode_batch(["one two", "", "three"])
    assert offsets.tolist() == [0, len(hash_ids(tokens("one two"))), len(hash_ids(tokens("one two"))) + 1]
    assert len(ids) == offsets[-1] + len(hash_ids(tokens("three")))


def test_platt_scaling_recovers_a_shift():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.1).astype(np.float64)
    true_logit = np.where(y > 0, rng.normal(1.0, 1.0, 4000), rng.normal(-2.0, 1.0, 4000))
    raw = 2.0 * true_logit + 3.0          # a head that is both over-confident and shifted
    T, b = fit_platt(raw, y)
    p = 1 / (1 + np.exp(-(raw / T + b)))
    p_raw = 1 / (1 + np.exp(-raw))
    assert abs(p.mean() - y.mean()) < 0.02 and abs(p_raw.mean() - y.mean()) > 0.2
    # reliability: in the top-third bucket the hit rate matches the mean prediction within a few points
    top = np.argsort(p)[-1000:]
    assert abs(p[top].mean() - y[top].mean()) < 0.06


def test_text_head_learns_a_vocabulary():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    bad = ["verify account login fee wire", "urgent prize free verify", "wire the fee now account"]
    good = ["weather news today blog", "recipe team sports music", "docs about the shop team"]
    texts = [str(rng.choice(bad if i % 3 == 0 else good)) + f" w{rng.integers(0, 50)}" for i in range(600)]
    y = np.array([1.0 if i % 3 == 0 else 0.0 for i in range(600)], dtype=np.float32)
    head = TextHead("t", buckets=1 << 16)
    ids, off = encode_batch(texts, head.buckets)
    ids_t, off_t = torch.from_numpy(ids), torch.from_numpy(off)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(60):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(ids_t, off_t), torch.from_numpy(y))
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    out = head.score(["urgent: verify your account and wire the fee", "the weather blog has a new recipe"])
    assert out["prob"][0] > 0.8 and out["prob"][1] < 0.2 and out["llr"][0] > out["llr"][1]
