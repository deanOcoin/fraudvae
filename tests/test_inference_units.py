"""small pure helpers of the inference module"""
from fraudvae.server.inference import _percentile


def test_percentile_interpolates_the_stored_quantiles():
    q = {"0.5": 1.0, "0.9": 2.0, "0.99": 3.0, "0.999": 3.0}     # keys as calibration.json writes them
    assert _percentile(q, 0.5) == 0.0
    assert _percentile(q, 1.0) == 50.0
    assert abs(_percentile(q, 1.5) - 70.0) < 1e-9
    assert abs(_percentile(q, 2.5) - 94.5) < 1e-9
    assert _percentile(q, 3.0) == 99.0                           # a flat stretch resolves to its lower quantile
    assert _percentile(q, 10.0) == 100.0
    assert _percentile(q, 0.999) == 0.0
    # exponent-form keys, as json may write small quantiles
    assert abs(_percentile({"1e-3": 0.1, "5e-1": 1.0}, 0.55) - 100 * (0.001 + 0.5 * (0.5 - 0.001))) < 1e-9
