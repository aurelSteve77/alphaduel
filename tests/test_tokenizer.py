import numpy as np

from alphaduel.generative.constraints import MandateConstraints
from alphaduel.generative.tokenizer import PortfolioTokenizer, QuantBucketizer


def test_bucketizer_fit_transform_range():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(200, 3))
    b = QuantBucketizer(n_buckets=8).fit(feats)
    ids = b.transform(feats)
    assert ids.min() >= 0
    assert ids.max() < 8


def test_bucketizer_requires_fit():
    try:
        QuantBucketizer().transform(np.zeros((2, 3)))
    except RuntimeError:
        return
    raise AssertionError("transform should fail before fit (leakage guard).")


def test_book_round_trip():
    symbols = ["A", "B", "C", "D"]
    tok = PortfolioTokenizer(symbols, n_feature_buckets=8, n_weight_buckets=11)
    tok.fit_buckets(np.random.default_rng(0).normal(size=(50, 4, 2)))
    weights = np.array([0.5, 0.0, 0.3, 0.2])
    decoded = tok.decode_book(tok.encode_book(weights))
    # Weight buckets are lossy; check ordering/support is preserved.
    assert decoded[0] > decoded[2] >= decoded[3] > 0
    assert decoded[1] == 0.0


def test_vocab_layout_non_overlapping():
    tok = PortfolioTokenizer(["A", "B"], n_feature_buckets=4, n_weight_buckets=5)
    assert tok.security_base < tok.feature_base < tok.weight_base < tok.vocab_size


def test_constraints_position_cap():
    symbols = ["A", "B", "C"]
    c = MandateConstraints(max_position_weight=0.3)
    current = np.array([0.3, 0.1, 0.0])
    eligible = c.eligible_securities(symbols, current)
    assert not eligible[0]  # at cap
    assert eligible[1] and eligible[2]
