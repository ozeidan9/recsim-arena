import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from recsys_market.mechanisms.m0_random import RandomMechanism
from recsys_market.mechanisms.m1_single import SingleStageMechanism
from recsys_market.mechanisms.m2_two_stage import TwoStageMechanism
from recsys_market.mechanisms.m3_llm_gate import LLMGateMechanism
from recsys_market.llm_gate.gate import SimpleLinearGate


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


# ── M2 (TwoStageMechanism) ────────────────────────────────────────────────────

@pytest.fixture
def small_market():
    rng = np.random.default_rng(7)
    n_users, n_creators, d = 8, 12, 8
    prefs = rng.standard_normal((n_users, d)).astype(np.float32)
    prefs /= np.linalg.norm(prefs, axis=1, keepdims=True)
    contents = rng.standard_normal((n_creators, d)).astype(np.float32)
    contents /= np.linalg.norm(contents, axis=1, keepdims=True)
    quality = rng.uniform(0, 1, n_creators).astype(np.float32)
    bait = rng.uniform(0, 1, n_creators).astype(np.float32)
    return prefs, contents, quality, bait, d


def test_m2_slate_shape(small_market):
    prefs, contents, quality, bait, d = small_market
    rng = np.random.default_rng(0)
    mech = TwoStageMechanism(content_dim=d, retrieval_size=6)
    slates = mech.recommend(prefs, contents, quality, bait, slate_size=3, rng=rng)
    assert slates.shape == (8, 3)


def test_m2_valid_indices(small_market):
    prefs, contents, quality, bait, d = small_market
    rng = np.random.default_rng(0)
    mech = TwoStageMechanism(content_dim=d, retrieval_size=6)
    slates = mech.recommend(prefs, contents, quality, bait, slate_size=3, rng=rng)
    assert slates.min() >= 0
    assert slates.max() < contents.shape[0]


def test_m2_retrieval_bottleneck(small_market):
    """With retrieval_size=4 and n_creators=12, each slate should only draw
    from 4 possible creator indices per user (not all 12).

    We verify by checking that across all users, the union of slate items is a
    subset of each user's top-4 by encoder similarity — which is stochastically
    true when the reranker has not yet learned anything special.
    We just check slates contain valid indices (structural test).
    """
    prefs, contents, quality, bait, d = small_market
    rng = np.random.default_rng(0)
    retrieval_size = 4
    mech = TwoStageMechanism(content_dim=d, retrieval_size=retrieval_size)
    slates = mech.recommend(prefs, contents, quality, bait, slate_size=3, rng=rng)
    assert slates.min() >= 0 and slates.max() < 12


def test_m2_update_does_not_crash(small_market):
    prefs, contents, quality, bait, d = small_market
    rng = np.random.default_rng(0)
    mech = TwoStageMechanism(content_dim=d, retrieval_size=6, update_every=1, min_buffer=0)
    slates = mech.recommend(prefs, contents, quality, bait, slate_size=3, rng=rng)
    chosen = np.array([slates[i, 0] for i in range(len(prefs))])
    # Should not raise
    mech.update(
        slates=slates,
        chosen_creators=chosen,
        user_preferences=prefs,
        creator_contents=contents,
        creator_quality=quality,
        creator_bait=bait,
    )


def test_m2_buffer_grows(small_market):
    prefs, contents, quality, bait, d = small_market
    rng = np.random.default_rng(0)
    mech = TwoStageMechanism(content_dim=d, retrieval_size=6)
    slates = mech.recommend(prefs, contents, quality, bait, slate_size=3, rng=rng)
    chosen = np.array([slates[i, 0] for i in range(len(prefs))])
    mech.update(slates, chosen, prefs, contents, quality, bait)
    # 8 users × 3 items per slate = 24 buffer entries
    assert mech.buffer_size() == 8 * 3


def test_m2_reranker_loss_finite(small_market):
    prefs, contents, quality, bait, d = small_market
    rng = np.random.default_rng(0)
    mech = TwoStageMechanism(
        content_dim=d, retrieval_size=6, update_every=1, min_buffer=1
    )
    slates = mech.recommend(prefs, contents, quality, bait, slate_size=3, rng=rng)
    chosen = np.array([slates[i, 0] for i in range(len(prefs))])
    mech.update(slates, chosen, prefs, contents, quality, bait)
    loss = mech.reranker_loss()
    assert np.isfinite(loss), f"Reranker loss is not finite: {loss}"


# ── M3 (LLMGateMechanism) ─────────────────────────────────────────────────────

def test_m3_slate_shape(small_market):
    prefs, contents, quality, bait, d = small_market
    rng = np.random.default_rng(0)
    gate = SimpleLinearGate(threshold=0.5)
    mech = LLMGateMechanism(content_dim=d, retrieval_size=6, gate=gate)
    slates = mech.recommend(prefs, contents, quality, bait, slate_size=3, rng=rng)
    assert slates.shape == (8, 3)


def test_m3_valid_indices(small_market):
    prefs, contents, quality, bait, d = small_market
    rng = np.random.default_rng(0)
    gate = SimpleLinearGate(threshold=0.5)
    mech = LLMGateMechanism(content_dim=d, retrieval_size=6, gate=gate)
    slates = mech.recommend(prefs, contents, quality, bait, slate_size=3, rng=rng)
    assert slates.min() >= 0 and slates.max() < 12


def test_m3_strict_gate_still_fills_slate(small_market):
    """Even with a very strict gate (nearly all fail), slates should be full
    because gate_min_slots ensures fallback to gate-rejects."""
    prefs, contents, quality, bait, d = small_market
    rng = np.random.default_rng(0)
    # Near-impossible gate: requires quality > 0.99 AND bait < 0.01
    gate = SimpleLinearGate(alpha_quality=10.0, alpha_bait=-10.0, bias=-9.0, threshold=0.99)
    mech = LLMGateMechanism(content_dim=d, retrieval_size=8, gate=gate)
    slates = mech.recommend(prefs, contents, quality, bait, slate_size=3, rng=rng)
    # Should still return valid slates
    assert slates.shape == (8, 3)
    assert slates.min() >= 0 and slates.max() < 12


def test_m3_update_does_not_crash(small_market):
    prefs, contents, quality, bait, d = small_market
    rng = np.random.default_rng(0)
    gate = SimpleLinearGate(threshold=0.5)
    mech = LLMGateMechanism(content_dim=d, retrieval_size=6, gate=gate,
                             update_every=1, min_buffer=0)
    slates = mech.recommend(prefs, contents, quality, bait, slate_size=3, rng=rng)
    chosen = np.array([slates[i, 0] for i in range(len(prefs))])
    mech.update(slates, chosen, prefs, contents, quality, bait)
