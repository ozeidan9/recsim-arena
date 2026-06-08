import numpy as np
import pytest

from recsys_market.users.user_model import UserPool


@pytest.fixture
def pool():
    return UserPool(n_users=20, n_populations=3, content_dim=8, seed=42)


def test_preferences_unit_norm(pool):
    norms = np.linalg.norm(pool.preferences, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_preferences_shape(pool):
    assert pool.preferences.shape == (20, 8)


def test_fatigue_initialised_zero(pool):
    assert (pool.fatigue == 0).all()


def test_fatigue_accumulates(pool):
    bait = np.ones(20, dtype=np.float32) * 0.5
    pool.update_fatigue(bait, gamma=0.95)
    assert (pool.fatigue > 0).all()


def test_fatigue_decay(pool):
    bait = np.ones(20, dtype=np.float32) * 1.0
    pool.update_fatigue(bait, gamma=0.0)  # gamma=0 → no carry-forward
    pool.update_fatigue(np.zeros(20), gamma=0.0)
    assert (pool.fatigue == 0).all()


def test_preferences_stable_across_reset(pool):
    prefs_before = pool.preferences.copy()
    pool.update_fatigue(np.ones(20) * 0.3)
    pool.reset()
    np.testing.assert_array_equal(pool.preferences, prefs_before)


def test_fatigue_reset_to_zero(pool):
    pool.update_fatigue(np.ones(20) * 0.5)
    pool.reset()
    assert (pool.fatigue == 0).all()
