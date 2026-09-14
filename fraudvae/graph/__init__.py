"""v5: the entity graph module. Entities (customers, counterparties, devices, emails,
addresses, cards) are nodes, transactions connect the entities they touch, a GraphSAGE
encoder turns each source's graph into node embeddings and a small head reads a
transaction's entities' embeddings into the "graph" signal. build.py makes the snapshots,
model.py the networks, train.py fits one encoder + head per source in the post stage,
scorer.py serves them, bench.py runs the public fraud-graph benchmarks on the same encoder."""
