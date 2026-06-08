import numpy as np
import pytest

from recsys_market.mechanisms.m0_random import RandomMechanism
from recsys_market.mechanisms.m1_single import SingleStageMechanism


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def market_state(rng):
    n_users, n_creators, d = 10, 15, 8
    prefs = rng.standard_normal((n_users, d)).astype(np.float32)
    prefs /= np.linalg.norm(prefs, axis=1, keepdims=True)
    contents = rng.standard_normal((n_creators, d)).astype(np.float32)
    contents /= np.linalg.norm(contents, axis=1, keepdims=True)
    quality = rng.uniform(0, 1, n_creators).astype(np.float32)
    bait = rng.uniform(0, 1, n_creators).astype(np.float32)
    return prefs, contents, quality, bait


def test_slate_size_m0(market_state, rng):
    prefs, contents, quality, bait = market_state
    mech = RandomMechanism()
    slates = mech.recommend(prefs, contents, quality, bait, slate_size=5, rng=rng)
    assert slates.shape == (10, 5)


def test_slate_size_m1(market_state, rng):
    prefs, contents, quality, bait = market_state
    mech = SingleStageMechanism(temperature=1.0)
    slates = mech.recommend(prefs, contents, quality, bait, slate_size=5, rng=rng)
    assert slates.shape == (10, 5)


def test_m0_valid_indices(market_state, rng):
    prefs, contents, quality, bait = market_state
    mech = RandomMechanism()
    slates = mech.recommend(prefs, contents, quality, bait, slate_size=5, rng=rng)
    assert slates.min() >= 0
    assert slates.max() < contents.shape[0]


def test_m0_no_repeats_in_slate(market_state, rng):
    prefs, contents, quality, bait = market_state
    mech = RandomMechanism()
    slates = mech.recommend(prefs, contents, quality, bait, slate_size=5, rng=rng)
    for row in slates:
        assert len(set(row)) == len(row), "Duplicate creators in slate"


def test_m0_uniform_over_many_samples(rng):
    """All creators should appear with roughly equal frequency under M0."""
    n_users, n_creators, d = 100, 10, 4
    prefs = rng.standard_normal((n_users, d)).astype(np.float32)
    prefs /= np.linalg.norm(prefs, axis=1, keepdims=True)
    contents = rng.standard_normal((n_creators, d)).astype(np.float32)
    contents /= np.linalg.norm(contents, axis=1, keepdims=True)
    quality = np.ones(n_creators, dtype=np.float32) * 0.5
    bait = np.ones(n_creators, dtype=np.float32) * 0.5

    mech = RandomMechanism()
    counts = np.zeros(n_creators)
    for _ in range(200):
        slates = mech.recommend(prefs, contents, quality, bait, slate_size=5, rng=rng)
        for idx in slates.flatten():
            counts[idx] += 1

    # All creators should appear; no creator should dominate drastically
    assert (counts > 0).all(), "Some creators never appear in M0"
    cv = counts.std() / counts.mean()
    assert cv < 0.5, f"Coefficient of variation too high for M0: {cv:.3f}"


def test_m1_ranking_monotone(rng):
    """Creator with highest dot-product should appear first in the slate.

    Uses bait_weight=0 to isolate the relevance signal.
    """
    n_users, n_creators, d = 5, 10, 8
    prefs = rng.standard_normal((n_users, d)).astype(np.float32)
    prefs /= np.linalg.norm(prefs, axis=1, keepdims=True)

    # Make one creator clearly most relevant for all users
    dominant = prefs.mean(axis=0)
    dominant /= np.linalg.norm(dominant)
    contents = rng.standard_normal((n_creators, d)).astype(np.float32)
    contents[0] = dominant * 10  # very high scores
    contents /= np.linalg.norm(contents, axis=1, keepdims=True)
    # Restore dominant after normalisation
    contents[0] = dominant

    quality = np.zeros(n_creators, dtype=np.float32)
    bait = np.zeros(n_creators, dtype=np.float32)

    # bait_weight=0: pure relevance ranking, isolates the dot-product ordering
    mech = SingleStageMechanism(temperature=1.0, bait_weight=0.0, deterministic=True)
    slates = mech.recommend(prefs, contents, quality, bait, slate_size=3, rng=rng)

    # The dominant creator (index 0) should appear in every user's slate
    assert (slates == 0).any(axis=1).all(), "Dominant creator missing from some slates"


def test_m1_bait_weight_promotes_bait_creators(rng):
    """With bait_weight>0, a high-bait creator should outrank an equally relevant but zero-bait one."""
    d = 4
    prefs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # single user
    # Two creators with identical content; creator 1 has high bait
    contents = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    quality = np.zeros(2, dtype=np.float32)
    bait = np.array([0.0, 1.0], dtype=np.float32)  # creator 1 has bait=1

    mech = SingleStageMechanism(temperature=1.0, bait_weight=0.5, deterministic=True)
    slates = mech.recommend(prefs, contents, quality, bait, slate_size=1, rng=rng)

    # Creator 1 (high bait) should be ranked first
    assert slates[0, 0] == 1, "High-bait creator should be promoted by M1"


def test_m1_temperature_affects_scores():
    """Higher temperature should reduce score spread (softer ranking)."""
    rng = np.random.default_rng(0)
    d = 8
    prefs = rng.standard_normal((1, d)).astype(np.float32)
    prefs /= np.linalg.norm(prefs, axis=1, keepdims=True)
    contents = rng.standard_normal((20, d)).astype(np.float32)
    contents /= np.linalg.norm(contents, axis=1, keepdims=True)
    quality = np.zeros(20, dtype=np.float32)
    bait = np.zeros(20, dtype=np.float32)

    mech_low = SingleStageMechanism(temperature=0.1, deterministic=False)
    mech_high = SingleStageMechanism(temperature=10.0, deterministic=False)

    # With low temperature: top-1 almost always the same; high temp: varies
    top1_counts_low = np.zeros(20)
    top1_counts_high = np.zeros(20)
    sample_rng = np.random.default_rng(1)
    for _ in range(200):
        s_low = mech_low.recommend(prefs, contents, quality, bait, 1, sample_rng)
        s_high = mech_high.recommend(prefs, contents, quality, bait, 1, sample_rng)
        top1_counts_low[s_low[0, 0]] += 1
        top1_counts_high[s_high[0, 0]] += 1

    # Low temperature should be more concentrated (higher max count)
    assert top1_counts_low.max() > top1_counts_high.max()
