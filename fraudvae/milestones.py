"""The generations that ship, oldest first: what each was trained on and what changed, in the
words the model page, the score tab's picker, the story and the bundle manifest all use.
Kept apart from the server so `fraudvae models pack` can read it on a machine without torch."""
from __future__ import annotations

# hardware is what each was actually trained on; vae_from marks a booster-side refit of another run's checkpoint
# (v6 and v7 also read a pretrained transaction encoder, txfm.pt beside the booster)
MILESTONES = {
    "cpu_small-8src": {
        "version": "v3", "label": "on a laptop", "order": 0,
        "hardware": "2 x86-64 CPU cores, no GPU", "corpus": "8 datasets, at most 500,000 training rows each (3.5M)",
        "what": "The first complete model: a variational autoencoder over the shared schema, scoring novelty alone. "
                "No labels used at all. Calibration picks a score type and freezes per-bank thresholds on validation.",
        "changed": "The starting point: the normal-payments model alone, no fraud labels used anywhere.",
    },
    "gpu_full-4090": {
        "version": "v3", "label": "on all the data", "order": 1,
        "hardware": "1x NVIDIA RTX 4090 (rented), 16 CPU cores", "corpus": "8 datasets, every row (25.6M training rows)",
        "what": "The same VAE on seven times the rows. It did not change what the density model sees: 0.833 against 0.829 "
                "mean AUROC. The lesson that the feature set, not the row count, was the ceiling.",
        "changed": "The same model on seven times the data. It barely moved: the fields it could see, not the number of rows, were the limit.",
    },
    "gpu_full-4090-v4": {
        "version": "v4", "label": "labels, context and the blend", "order": 2,
        "hardware": "1x NVIDIA RTX 4090 (rented), 16 CPU cores", "corpus": "9 datasets (AMLworld added), every row (29.2M training rows)",
        "what": "A gradient-boosted classifier on the labelled fraud next to the VAE, the customer-context block (velocity, "
                "typical amounts, first-time flags), the merchant block, and a per-bank logistic blend of the two scores. "
                "The step change: 0.83 to 0.92 mean AUROC.",
        "changed": "Added decision trees trained on the labelled fraud, the customer's and merchant's recent history as inputs, and a per-bank blend of the two models. The big step: 0.83 to 0.92.",
    },
    "v5-h200-e": {
        "version": "v5", "label": "customer history, entities, extra models", "order": 3,
        "hardware": "1x NVIDIA H200 NVL (rented), 64 CPU cores, 188 GB RAM", "corpus": "11 datasets (BAF, fraudecom added), every row (30M train / 4.3M validation / 8.6M test)",
        "what": "The VAE reads the customer's previous transactions through a small transformer; the entity-graph block "
                "(counterparty / device / email / address / card history with 30-day-delayed fraud labels), the wide blocks "
                "of each source's raw columns, per-bank specialist boosters, a sequence head, a device head, GraphSAGE "
                "entity graphs, text heads and behaviour verifiers, all blended. 0.916 to 0.936 on v4's nine sources.",
        "changed": "The model now reads the customer's previous payments; new facts about the merchant, card, device, e-mail and address around each payment; two more datasets; five extra models. 0.916 to 0.936.",
    },
    "v5-h200-f": {
        "version": "v5", "label": "fresher fraud history", "order": 4, "vae_from": "v5-h200-e",
        "hardware": "the VAE checkpoint of e; booster side refit on the H200 box (64 CPU cores)", "corpus": "as e",
        "what": "The same entity fraud history at a one-week label delay (the Fraud Detection Handbook's assumption), "
                "booster-only, next to the 30-day block. fdh 0.69 to 0.89, IEEE 0.94 to 0.96, 0.936 to 0.961 over the nine.",
        "changed": "Counts fraud confirmed a week ago instead of a month ago, where a bank's labels arrive that fast. The terminal-compromise dataset went from 0.69 to 0.89.",
    },
    "v5-h200-h": {
        "version": "v5", "label": "recent fraud, label timing as a setting", "order": 5, "vae_from": "v5-h200-e",
        "hardware": "the VAE checkpoint of e; booster side refit on the H200 box (64 CPU cores)", "corpus": "as e",
        "what": "Fraud share over the last 7 / 30 days before the label horizon (tabformer 74% to 86% of its fraud at 1% FPR), "
                "and the booster trained for both label latencies, so a deployment whose labels take a month is a "
                "regime it has seen. 0.963 over the nine, 0.969 generic. What shipped until v7.",
        "changed": "Looks at a merchant's fraud over the last 7 and 30 days rather than its whole lifetime, and copes whether or not fresh labels are available. 0.963 average over the nine shared datasets; what shipped until v7.",
    },
    "v6-base-e": {
        "version": "v6", "label": "a pretrained model of payment histories", "order": 6, "vae_from": "v5-h200-e",
        "hardware": "the encoder on a rented node of 8 NVIDIA B200 GPUs (60,000 steps, four and a half hours); the VAE checkpoint of e; booster side refit on one of them",
        "corpus": "the encoder: 16 datasets (five more AMLworld variants), 328M training rows, no labels; the booster as e",
        "what": "A 128M-parameter transformer pretrained without labels to predict a customer's next payment field by field. "
                "Its state after the customer's last 127 payments and its surprise at each field of this one, 888 columns, "
                "go to the booster for every bank. tabformer 85.9% to 94.3% of its fraud at 1% FPR (73.4% to 87.6% when "
                "labels take a month), fdh and amlworld better; capone, ieee and fraudecom a little worse. 0.9277 to 0.9282 "
                "mean AUROC over the eleven: a trade, not a step.",
        "changed": "A large model that read 328 million payments with no fraud labels and learnt what a customer's next payment looks like. What it makes of a customer's history goes to the decision trees. A big gain for the bank with long card histories (86% to 94% of its fraud caught), a small loss for three others.",
    },
    "v7-f": {
        "version": "v7", "label": "the history model where it helps", "order": 7, "vae_from": "v5-h200-e",
        "hardware": "the encoder of v6, unchanged; the VAE checkpoint of e; booster side refit on a rented box with 2x NVIDIA H200 NVL",
        "corpus": "as v6",
        "what": "v6's block for tabformer and fdh only (a control showed capone's and ieee's losses were the weight of 888 "
                "more columns on the trees), a supervised head on that block as one more signal for the blend, and the blend "
                "fit without the VAE's two columns. Against h on the same 8.6M test rows, paired: mean recall at 1% FPR "
                "+0.011 [+0.009, +0.013], +0.018 [+0.016, +0.021] when labels take a month; tabformer 85.9% to 95.4% "
                "(73.4% to 90.6%); no bank worse beyond a few millionths of AUROC on paysim. 0.964 over the nine, 0.971 "
                "generic. What ships.",
        "changed": "Uses the history model only for the two banks where it earned its place, adds a small model trained on fraud labels on top of it, and lets the verdict rest on the models that learnt fraud. More fraud caught at the same false-alarm rate on average, 86% to 95% for the bank with long card histories, and no bank worse; what ships.",
    },
}
