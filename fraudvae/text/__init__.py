"""v5: the text module. A deployer's free text (a url on the order, a product or job
description, a profile bio, a memo) goes through small text heads trained on public labelled
text-fraud data - malicious urls, fraudulent job postings, bot profiles - and comes back as a
calibrated "text" likelihood ratio that is fused into the final probability. None of the nine
transaction sources carries text, so this module changes nothing in the held-out tables; it
is there for the deployment that has memos and descriptors. features.py hashes text into
fastText-style bags, model.py is the encoder, train.py fits and calibrates the heads,
scorer.py serves them."""
