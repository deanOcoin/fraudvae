"""The generations that ship, oldest first: what each was trained on and what changed, in the
words the model page, the score tab's picker, the story and the bundle manifest all use.
Kept apart from the server so `fraudvae models pack` can read it on a machine without torch."""
from __future__ import annotations

# hardware is what each was actually trained on; vae_from marks a booster-side refit of another run's checkpoint
MILESTONES = {
    "cpu_small-8src": {
        "version": "v3", "label": "on a laptop", "order": 0,
        "hardware": "2 x86-64 CPU cores, no GPU", "corpus": "8 sources, capped at 1.5M rows per source (4M rows)",
        "what": "The first complete model: a variational autoencoder over the shared schema, scoring novelty alone. "
                "No labels used at all. Calibration picks a score type and freezes per-bank thresholds on validation.",
        "changed": "Baseline.",
    },
    "gpu_full-4090": {
        "version": "v3", "label": "on the full corpus", "order": 1,
        "hardware": "1x NVIDIA RTX 4090 (rented), 16 CPU cores", "corpus": "8 sources, every row (29M)",
        "what": "The same VAE on seven times the rows. It did not change what the density model sees: 0.833 against 0.829 "
                "mean AUROC. The lesson that the feature set, not the row count, was the ceiling.",
        "changed": "More data, same features.",
    },
    "gpu_full-4090-v4": {
        "version": "v4", "label": "labels, context and the blend", "order": 2,
        "hardware": "1x NVIDIA RTX 4090 (rented), 16 CPU cores", "corpus": "9 sources (AMLworld added), every row (29M)",
        "what": "A gradient-boosted classifier on the labelled fraud next to the VAE, the customer-context block (velocity, "
                "typical amounts, first-time flags), the merchant block, and a per-bank logistic blend of the two scores. "
                "The step change: 0.83 to 0.92 mean AUROC.",
        "changed": "Booster + context features + blend.",
    },
    "v5-h200-e": {
        "version": "v5", "label": "history, entities, modules", "order": 3,
        "hardware": "1x NVIDIA H200 NVL (rented), 64 CPU cores, 188 GB RAM", "corpus": "11 sources (BAF, fraudecom added), every row (30M train / 4.3M val / 8.6M test)",
        "what": "The VAE reads the customer's previous transactions through a small transformer; the entity-graph block "
                "(counterparty / device / email / address / card history with 30-day-delayed fraud labels), the wide blocks "
                "of each source's raw columns, per-bank specialist boosters, a sequence head, a device head, GraphSAGE "
                "entity graphs, text heads and behaviour verifiers, all blended. 0.916 to 0.936 on v4's nine sources.",
        "changed": "Two sources, the entity and wide blocks, five modules, history in the VAE.",
    },
    "v5-h200-f": {
        "version": "v5", "label": "one-week fraud labels", "order": 4, "vae_from": "v5-h200-e",
        "hardware": "the VAE checkpoint of e; booster side refit on the H200 box (64 CPU cores)", "corpus": "as e",
        "what": "The same entity fraud history at a one-week label delay (the Fraud Detection Handbook's assumption), "
                "booster-only, next to the 30-day block. fdh 0.69 to 0.89, IEEE 0.94 to 0.96, 0.936 to 0.961 over the nine.",
        "changed": "Label delay 7 days for the booster.",
    },
    "v5-h200-h": {
        "version": "v5", "label": "recent-fraud windows, latency as a setting", "order": 5, "vae_from": "v5-h200-e",
        "hardware": "the VAE checkpoint of e; booster side refit on the H200 box (64 CPU cores)", "corpus": "as e",
        "what": "Fraud share over the last 7 / 30 days before the label horizon (tabformer 74% to 86% of its fraud at 1% FPR), "
                "and the booster trained for both label latencies, so a deployment whose labels take a month is a "
                "regime it has seen. 0.963 over the nine, 0.969 generic. What ships.",
        "changed": "Recent-fraud windows, latency dropout, both evaluations.",
    },
}
