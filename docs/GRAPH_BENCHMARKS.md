# Graph benchmarks

`fraudvae graph-bench --dataset <name> --out results.json` runs the same GraphSAGE
encoder the transaction graphs use (`fraudvae/graph/model.py`: mean aggregation over
row-normalised adjacency, one weight per relation, two layers, LayerNorm + SiLU between
them) as a node classifier on the public fraud graphs, with each dataset's standard
split. The point is to check the code against numbers the literature knows, not to
compete: no sampling tricks, no label-aware aggregation, early stopping on validation
AUROC, 300 epochs of full-batch Adam at most.

| dataset | nodes | edges | relations | split | test AUROC | test AUPRC | F1 at the base rate |
|---|---|---|---|---|---|---|---|
| DGraph-Fin | 3.70M | 4.30M | 1 | published train / valid / test masks | **0.784** | 0.045 | 0.076 |
| Elliptic | 204k | 234k | 1 | time steps 1-29 / 30-34 / 35-49 | **0.880** | 0.559 | 0.579 |
| YelpChi | 46k | 4.03M (3 relations, symmetrised) | 3 | stratified 40 / 20 / 40, seed 2 | **0.923** | 0.737 | 0.673 |
| Amazon | 11.9k | 4.78M | 3 | same, first 3,305 nodes unlabelled and left out | **0.978** | 0.915 | 0.857 |

Context: the DGraph-Fin paper reports 0.761 for GraphSAGE and 0.741 for GAT on this
split; Elliptic's illicit F1 is usually quoted around 0.6-0.8 for tree models on all
features and lower for plain GCNs because the dark-market shutdown at time step 43
shifts the distribution; CARE-GNN's YelpChi / Amazon AUROCs are 0.76 / 0.90 and the
later BWGNN's 0.90 / 0.98. The multi-relation encoder with standardised features
lands at the top of those ranges on the review graphs and a little above the published
SAGE on DGraph; Elliptic's F1 sits in the middle. F1 here is computed at the threshold
that flags exactly the true fraud share of the test set.

Run on the H200 (DGraph: 40 s for 275 epochs) and on the 2-core sandbox (the other
three, 2.5-4 min each). Results json: `runs/graph_bench.json` on the box,
the same file for the three small ones once copied back.
