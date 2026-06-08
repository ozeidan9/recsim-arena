import numpy as np
import pytest

from recsys_market.metrics.diversity import content_entropy, coverage, intra_list_diversity
from recsys_market.metrics.inequality import exposure_distribution, gini
from recsys_market.metrics.quality import mean_recommended_quality, quality_distribution
from recsys_market.metrics.welfare import total_welfare, user_welfare


# ─── Gini ──────────────────────────────────────────────────────────────────

def test_gini_uniform():
    exposures = np.ones(10)
    assert gini(exposures) == pytest.approx(0.0, abs=1e-6)


def test_gini_monopoly():
    exposures = np.zeros(10)
    exposures[0] = 100.0
    g = gini(exposures)
    assert g > 0.8  # approaches 1 for strong monopoly


def test_gini_zero_exposure():
    assert gini(np.zeros(5)) == 0.0


def test_gini_range():
    rng = np.random.default_rng(0)
    exposures = rng.exponential(1.0, 50)
    g = gini(exposures)
    assert 0.0 <= g <= 1.0


def test_exposure_distribution_monotone():
    exposures = np.array([1.0, 2.0, 3.0, 4.0])
    cum = exposure_distribution(exposures)
    assert (np.diff(cum) >= 0).all()
    assert cum[-1] == pytest.approx(1.0)


# ─── Diversity ─────────────────────────────────────────────────────────────

def test_entropy_bounds():
    rng = np.random.default_rng(42)
    contents = rng.standard_normal((20, 8)).astype(np.float32)
    contents /= np.linalg.norm(contents, axis=1, keepdims=True)
    n_clusters = 8
    h = content_entropy(contents, n_clusters=n_clusters)
    assert 0.0 <= h <= np.log(n_clusters) + 1e-6


def test_entropy_identical_contents_low():
    """Identical content vectors → all map to one cluster → low entropy."""
    base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    contents = np.tile(base, (20, 1))
    h = content_entropy(contents, n_clusters=4)
    assert h < 0.5


def test_intra_list_diversity_single_item():
    contents = np.eye(4, dtype=np.float32)  # orthogonal unit vectors
    slates = np.array([[0]])
    assert intra_list_diversity(slates, contents) == 0.0


def test_intra_list_diversity_orthogonal():
    contents = np.eye(4, dtype=np.float32)
    slates = np.array([[0, 1, 2, 3]])
    ild = intra_list_diversity(slates, contents)
    assert ild == pytest.approx(1.0, abs=1e-5)  # orthogonal → distance = 1


def test_coverage_all_covered():
    d = 4
    prefs = np.eye(d, dtype=np.float32)      # users at unit-basis vectors
    contents = np.eye(d, dtype=np.float32)   # creators at exact same positions
    cov = coverage(contents, prefs, threshold=0.99)
    assert cov == pytest.approx(1.0, abs=1e-5)


def test_coverage_none_covered():
    prefs = np.array([[1.0, 0.0]], dtype=np.float32)
    contents = np.array([[-1.0, 0.0]], dtype=np.float32)  # opposite direction
    cov = coverage(contents, prefs, threshold=0.5)
    assert cov == 0.0


# ─── Quality ───────────────────────────────────────────────────────────────

def test_mean_recommended_quality():
    qualities = np.array([0.0, 0.5, 1.0, 0.25])
    slates = np.array([[1, 2], [0, 3]])   # selected qualities: 0.5, 1.0, 0.0, 0.25
    expected = (0.5 + 1.0 + 0.0 + 0.25) / 4
    assert mean_recommended_quality(slates, qualities) == pytest.approx(expected)


def test_quality_distribution_bins():
    qualities = np.linspace(0, 1, 50)
    counts, edges = quality_distribution(qualities, bins=10)
    assert len(counts) == 10
    assert counts.sum() == 50


# ─── Welfare ───────────────────────────────────────────────────────────────

def test_welfare_positive_when_no_fatigue():
    clicks = np.array([1.0, 2.0, 0.0])
    fatigue = np.zeros(3)
    w = user_welfare(clicks, fatigue)
    np.testing.assert_array_equal(w, clicks)


def test_welfare_monotone_quality():
    """Higher click rate with same fatigue → higher welfare."""
    fatigue = np.array([0.1, 0.1])
    low_clicks = np.array([1.0, 1.0])
    high_clicks = np.array([2.0, 2.0])
    assert total_welfare(high_clicks, fatigue) > total_welfare(low_clicks, fatigue)


def test_total_welfare_sum():
    clicks = np.array([1.0, 2.0])
    fatigue = np.array([0.5, 0.5])
    expected = (1.0 - 0.5 * 0.5) + (2.0 - 0.5 * 0.5)
    assert total_welfare(clicks, fatigue, lambda_fatigue=0.5) == pytest.approx(expected)
