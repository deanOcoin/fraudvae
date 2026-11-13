"""The generations that ship, oldest first: what each was trained on and what changed, in the
words the model page, the score tab's picker, the story and the bundle manifest all use.
Kept apart from the server so `fraudvae models pack` can read it on a machine without torch."""
from __future__ import annotations

# hardware is what each was actually trained on; vae_from marks a booster-side refit of another run's checkpoint
# (v6 and v7 also read a pretrained transaction encoder, txfm.pt beside the booster)
MILESTONES = {
    "cpu_small-8src": {
        "version": "v3", "label": "on two CPU cores", "order": 0,
        "hardware": "2 x86-64 CPU cores, no GPU", "corpus": "8 datasets, at most 500,000 training rows each (3.5M)",
        "what": "The first complete model: a variational autoencoder over the shared schema, which scores how unusual a "
                "payment is and nothing else. It learns from legitimate rows only; the validation split's fraud labels "
                "choose the epoch to keep, the score type (raw or per-field normalised) and each bank's thresholds. "
                "0.829 mean test AUROC.",
        "changed": "The starting point: the model of normal payments (a VAE) alone. It learns only from legitimate payments and never from a fraud label.",
    },
    "gpu_full-4090": {
        "version": "v3", "label": "on all the data", "order": 1,
        "hardware": "1x NVIDIA RTX 4090 (rented), 16 CPU cores", "corpus": "8 datasets, every row (25.6M training rows)",
        "what": "The same design, 3.4 times larger, on seven times the rows: 0.833 mean test AUROC against 0.829. More rows "
                "gave the density model nothing new to see; the set of fields, not the number of rows, was the limit.",
        "changed": "The same design, 3.4 times larger, on seven times the data. The mean AUROC moved only from 0.829 to 0.833: the fields the model could see, not the number of rows, were the limit.",
    },
    "gpu_full-4090-v4": {
        "version": "v4", "label": "labels, context and the blend", "order": 2,
        "hardware": "1x NVIDIA RTX 4090 (rented), 16 CPU cores", "corpus": "9 datasets (AMLworld added), every row (29.2M training rows)",
        "what": "A gradient-boosted classifier trained on the labelled fraud beside the VAE, the customer-context block "
                "(bursts, typical amounts, tenure, first-time flags), the merchant block, and a logistic blend of the two "
                "scores with weights for each bank. On the eight datasets v3 had, the mean test AUROC rose from 0.833 to "
                "0.907 (0.916 over all nine).",
        "changed": "Added decision trees trained on the labelled fraud, facts about the customer's and the merchant's recent history as their inputs, and a blend of the two models with weights for each bank. On the eight datasets v3 had, the mean AUROC rose from 0.833 to 0.907.",
    },
    "v5-h200-e": {
        "version": "v5", "label": "history, entities, supporting models", "order": 3,
        "hardware": "1x NVIDIA H200 NVL (rented), 64 CPU cores, 188 GB RAM", "corpus": "11 datasets (BAF and fraudecom added), every row (30.0M training / 4.3M validation / 8.6M test rows)",
        "what": "The VAE reads the customer's previous 32 payments through a small transformer. New inputs: the entity block "
                "(the volume, age, distinct customers and confirmed fraud of the payment's counterparty, device, e-mail "
                "domain, address and card, with fraud labels counted after a 30-day delay) and each dataset's own raw "
                "columns for the booster. Six kinds of supporting model: a specialist booster per bank, a sequence head, a "
                "device head, GraphSAGE entity graphs, text heads and behaviour verifiers. On v4's nine datasets the mean "
                "test AUROC rose from 0.916 to 0.936.",
        "changed": "The VAE now reads the customer's previous 32 payments; new facts about the counterparty, card, device, e-mail domain and address of each payment; two more datasets; six kinds of supporting model. On v4's nine datasets the mean AUROC rose from 0.916 to 0.936.",
    },
    "v5-h200-f": {
        "version": "v5", "label": "one-week fraud labels", "order": 4, "vae_from": "v5-h200-e",
        "hardware": "the VAE of v5-h200-e; booster side refit on the same rented H200 box", "corpus": "the same as v5-h200-e",
        "what": "The entity block's fraud history counted a second time, after a one-week label delay (the Fraud Detection "
                "Handbook's assumption), for the booster only and beside the 30-day block. fdh 0.69 to 0.89, IEEE 0.94 "
                "to 0.96, and 0.936 to 0.961 mean test AUROC over the nine shared datasets.",
        "changed": "Counts the fraud confirmed against each merchant, card and device after a one-week delay as well as after a month, for banks whose labels arrive that fast. The terminal-compromise dataset went from 0.69 to 0.89.",
    },
    "v5-h200-h": {
        "version": "v5", "label": "recent fraud, label delay as a setting", "order": 5, "vae_from": "v5-h200-e",
        "hardware": "the VAE of v5-h200-e; booster side refit on the same rented H200 box", "corpus": "the same as v5-h200-e",
        "what": "Each entity's fraud count and share over the 7 and 30 days before the label horizon (tabformer 74% to 86% "
                "of its fraud caught at a 1% false-positive rate), and the booster trained with the one-week block blanked "
                "on a random 30% of its rows, so a deployment whose labels take a month is a case it was trained on. "
                "0.963 mean test AUROC over the nine shared datasets, 0.969 generic. What shipped until v7.",
        "changed": "Counts the fraud at each merchant, card and device over the last 7 and 30 days instead of over their whole history, and trains the trees both for banks whose labels take a week and for banks whose labels take a month. 0.963 mean AUROC over the nine shared datasets; shipped until v7.",
    },
    "v6-base-e": {
        "version": "v6", "label": "a pretrained model of payment histories", "order": 6, "vae_from": "v5-h200-e",
        "hardware": "the encoder pretrained on a rented node with 8 NVIDIA B200 GPUs (60,000 steps, 4.5 hours); the VAE of v5-h200-e; booster side refit on the same node",
        "corpus": "the encoder: 16 datasets (the 11 plus five AMLworld variants), 328M training rows, no labels; the booster: the same as v5-h200-e",
        "what": "A 128M-parameter transformer pretrained without labels to predict a customer's next payment field by field. "
                "Its state after the customer's last 127 payments and its surprise at each field of the new one, 888 "
                "columns, go to the booster for every bank. tabformer 85.9% to 94.3% of its fraud caught at a 1% "
                "false-positive rate (73.4% to 87.6% when labels take a month); fdh and amlworld slightly better; capone, "
                "ieee and fraudecom slightly worse. 0.9277 to 0.9282 mean test AUROC over the eleven: a trade, not an "
                "overall gain.",
        "changed": "Adds a 128-million-parameter model that learnt, from 328 million payments and without fraud labels, to predict a customer's next payment. Its reading of the customer's history goes to the decision trees as 888 extra inputs. One bank gained a lot (86% to 94% of its fraud caught at a 1% false-positive rate) and three lost slightly.",
    },
    "v7-f": {
        "version": "v7", "label": "the history model for two banks", "order": 7, "vae_from": "v5-h200-e",
        "hardware": "the encoder of v6 and the VAE of v5-h200-e, unchanged; booster side refit on a rented box with 2 NVIDIA H200 NVL GPUs",
        "corpus": "the same as v6",
        "what": "v6's block for tabformer and fdh only (a control with the block shuffled showed that capone's and ieee's "
                "losses came from adding 888 columns to the trees), a supervised head on that block as one more input to "
                "the blend, and the blend fit without the VAE's two columns. Against v5-h200-h on the same 8.6M test rows, "
                "paired: mean recall at a 1% false-positive rate +0.011 [+0.009, +0.013], and +0.018 [+0.016, +0.021] "
                "when labels take a month; tabformer 85.9% to 95.4% (73.4% to 90.6%); no bank worse except paysim, by a "
                "few millionths of AUROC. 0.964 mean test AUROC over the nine shared datasets, 0.971 generic. What ships.",
        "changed": "Gives the history model's inputs to the trees for two banks only (TabFormer and the Fraud Detection Handbook simulator), adds a small model trained on fraud labels on top of them, and fits the blend without the VAE. More fraud caught on average at a 1% false-positive rate (86% to 95% on TabFormer), and no bank worse by more than a few millionths of AUROC. The version that ships.",
    },
}
