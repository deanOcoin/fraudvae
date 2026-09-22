"""v5: behavioural biometrics. Mouse strokes, keystroke timing, touch swipes and device motion
are turned into fixed-length windows of hand-made dynamics features (features.py), an
embedding per modality is learned by telling the public datasets' users apart (model.py,
train.py: Balabit, BB-MAS, HMOG, BrainRun), and at serve time a session payload is compared
with the account holder's enrolled embedding: the cosine similarity, calibrated on genuine
and impostor trials, is the "behaviour" likelihood ratio the fusion step adds to the verdict
(scorer.py). No payment dataset carries sessions, so like the text module this never changes
a held-out table; its own numbers are equal-error rates on the biometric sets."""
