"""Tests for LLM quality gate implementations."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from recsys_market.llm_gate.gate import DistilledGate, SimpleLinearGate


# ── SimpleLinearGate ──────────────────────────────────────────────────────────

def test_simple_gate_score_range():
    gate = SimpleLinearGate()
    quality = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    bait = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    scores = gate.score(quality, bait)
    assert scores.shape == (3,)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_simple_gate_high_quality_low_bait_scores_high():
    """High quality + zero bait should score above 0.5."""
    gate = SimpleLinearGate(alpha_quality=3.0, alpha_bait=-3.0, bias=0.0)
    scores = gate.score(
        quality=np.array([1.0], dtype=np.float32),
        bait=np.array([0.0], dtype=np.float32),
    )
    assert scores[0] > 0.9


def test_simple_gate_low_quality_high_bait_scores_low():
    """Zero quality + max bait should score below 0.5."""
    gate = SimpleLinearGate(alpha_quality=3.0, alpha_bait=-3.0, bias=0.0)
    scores = gate.score(
        quality=np.array([0.0], dtype=np.float32),
        bait=np.array([1.0], dtype=np.float32),
    )
    assert scores[0] < 0.1


def test_simple_gate_quality_monotone():
    """Increasing quality should monotonically increase score (bait fixed)."""
    gate = SimpleLinearGate()
    quality = np.linspace(0, 1, 10).astype(np.float32)
    bait = np.full(10, 0.3, dtype=np.float32)
    scores = gate.score(quality, bait)
    assert (np.diff(scores) > 0).all()


def test_simple_gate_bait_monotone():
    """Increasing bait should monotonically decrease score (quality fixed)."""
    gate = SimpleLinearGate()
    quality = np.full(10, 0.5, dtype=np.float32)
    bait = np.linspace(0, 1, 10).astype(np.float32)
    scores = gate.score(quality, bait)
    assert (np.diff(scores) < 0).all()


def test_simple_gate_passes_threshold():
    gate = SimpleLinearGate(threshold=0.5)
    quality = np.array([1.0, 0.0], dtype=np.float32)
    bait = np.array([0.0, 1.0], dtype=np.float32)
    mask = gate.passes(quality, bait)
    assert mask[0] and not mask[1]


# ── DistilledGate ─────────────────────────────────────────────────────────────

def test_distilled_gate_output_range():
    gate = DistilledGate()
    quality = np.random.rand(20).astype(np.float32)
    bait = np.random.rand(20).astype(np.float32)
    scores = gate.score(quality, bait)
    assert scores.shape == (20,)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_distilled_gate_fit_learns_direction():
    """After fitting to SimpleLinearGate labels, DistilledGate should agree
    directionally: item (q=1, b=0) scores higher than item (q=0, b=1)."""
    rng = np.random.default_rng(0)
    n = 200
    quality = rng.uniform(0, 1, n).astype(np.float32)
    bait = rng.uniform(0, 1, n).astype(np.float32)

    teacher = SimpleLinearGate(alpha_quality=3.0, alpha_bait=-3.0)
    labels = teacher.score(quality, bait)

    student = DistilledGate.fit(quality, bait, labels, n_epochs=400, lr=1e-2)

    high_q = student.score(np.array([1.0], np.float32), np.array([0.0], np.float32))[0]
    high_b = student.score(np.array([0.0], np.float32), np.array([1.0], np.float32))[0]
    assert high_q > high_b, f"Distilled gate direction wrong: {high_q:.3f} vs {high_b:.3f}"


def test_distilled_gate_save_load(tmp_path):
    gate = DistilledGate(threshold=0.6)
    path = str(tmp_path / "gate.pt")
    gate.save(path)
    loaded = DistilledGate.load(path)
    assert loaded.threshold == 0.6

    quality = np.array([0.5], dtype=np.float32)
    bait = np.array([0.5], dtype=np.float32)
    np.testing.assert_allclose(
        gate.score(quality, bait), loaded.score(quality, bait), atol=1e-5
    )
