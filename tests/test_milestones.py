"""the curated timeline: what the model page, the score tab's picker, the story and the bundle
manifest all read"""
from fraudvae.milestones import MILESTONES


def test_timeline_is_ordered_and_whole():
    assert [m["order"] for m in MILESTONES.values()] == list(range(len(MILESTONES)))     # oldest first, no gaps, no ties
    for name, m in MILESTONES.items():
        assert {"version", "label", "hardware", "corpus", "what", "changed"} <= set(m), name
        # a booster-side refit names an earlier generation's checkpoint
        base = m.get("vae_from")
        assert base is None or (base in MILESTONES and MILESTONES[base]["order"] < m["order"]), name
    # one generation is what ships, and it is the newest
    assert [n for n, m in MILESTONES.items() if m["what"].endswith("What ships.")] == [list(MILESTONES)[-1]]
