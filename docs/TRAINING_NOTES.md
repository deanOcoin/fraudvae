# Training notes

Running log of what was tried and what it did. Newest at the bottom. Metrics are
AUROC on the held-out (newest 20%) test split of each source unless stated
otherwise.

## Scoring

The VAE gives every row a −ELBO: the sum of one negative log-likelihood per
field (Gaussian for numerics, softmax for categoricals) plus the KL term. Two
scores come out of that:

- `raw`: the −ELBO itself, averaged over 4 posterior samples.
- `norm`: each field's NLL z-scored against its mean and standard deviation
  on the legit validation rows, then averaged. A field the decoder is always
  unsure about stops dominating the sum, and the per-field z is what the
  receipt shows.

Calibration scores the validation split both ways, keeps whichever has the
higher mean per-source AUROC, and freezes the alert threshold at the chosen
score's 99th percentile on legit validation rows.

## Lessons from the first runs (smoke + quality probes, fdh + capone, 200k train rows / source)

- The raw −ELBO is dominated by the categorical hour/dow terms and by whatever
  numeric the decoder is least sure about. On fdh the amount term alone is a
  better detector than the sum. Hence the per-field normalised score (above);
  on the probe it beat raw by 0.02 on fdh and tied on capone, but calibration
  still checks both on every run.
- β warm-up matters: without it the first epoch collapses to KL≈0 and the
  decoder ignores z (train nll flat, val auroc 0.5). 2 epochs of ramp fixed
  it.
- Capital One is hard for anything unsupervised (~0.55 on the probe). The
  fraud there is mostly ordinary-looking purchases; the only strong signal is
  the CVV mismatch which is rare (0.9% of rows) and not fraud-exclusive.
- fdh scenario 3 (compromised card, amounts ×5) is caught by
  `amount_over_cust_mean`; scenario 1 (amount > 220) by `amount`; scenario 2
  (compromised terminal) is invisible without terminal features, which is
  by design here (unsupervised, no terminal risk scores).
- pandas 3 parses datetimes to microsecond resolution; `astype(int64)//1e9`
  silently gave timestamps 1000× too small for capone until the velocity
  features looked absurd. Now converted through `datetime64[s]`.

## Runs

### probes (2-core sandbox, fdh + capone, 200k train rows / source)

| config | epochs | source | raw AUROC | norm AUROC |
|---|---|---|---|---|
| smoke | 3 | fdh | 0.612 | 0.634 |
| smoke | 3 | capone | 0.541 | 0.538 |
| quality | 8 | fdh | 0.631 | 0.652 |
| quality | 8 | capone | 0.549 | 0.552 |

Nothing that ships yet. The first real run waits on the remaining six adapters;
the probes only size the epoch time (~40 s per 200k rows on 2 cores) and settle
the score question for now.
