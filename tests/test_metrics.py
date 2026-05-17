import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.metrics import (
    compute_best_action_match, compute_regret, compute_rank_accuracy,
    compute_residual_variance_ratio, compute_oracle_action_entropy,
    compute_bad_debt_ratio, compute_state_only_shortcut_index,
    compute_shuffled_action_gap, compute_seed_stability_ratio,
)


def test_best_action_match():
    y_pred = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert compute_best_action_match(y_pred, y_true) == 1.0


def test_best_action_match_partial():
    y_pred = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0]])
    y_true = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    assert compute_best_action_match(y_pred, y_true) == 1.0 / 3.0


def test_regret():
    y_pred = np.array([[0, 1, 0], [0, 0, 1]])
    y_true = np.array([[0, 1, 0], [0, 0, 1]])
    assert compute_regret(y_pred, y_true) == 0.0


def test_rank_accuracy_perfect():
    y_pred = np.array([[3, 2, 1], [1, 2, 3]])
    y_true = np.array([[3, 2, 1], [1, 2, 3]])
    assert compute_rank_accuracy(y_pred, y_true) == 1.0


def test_residual_variance_ratio():
    outcomes = []
    for i in range(10):
        base = np.random.randn(2)
        outcomes.extend([
            base + np.random.randn(2) * 0.1 + np.array([1.0, 0.0]),
            base + np.random.randn(2) * 0.1,
            base + np.random.randn(2) * 0.1 + np.array([-0.5, 0.5]),
        ])
    rvr = compute_residual_variance_ratio(outcomes)
    assert 0 <= rvr <= 1


def test_oracle_action_entropy():
    best_actions = np.array([0, 0, 0, 1, 1, 2])
    ent = compute_oracle_action_entropy(best_actions)
    assert 0 <= ent <= 1


def test_bad_debt_ratio():
    bdr = compute_bad_debt_ratio(0.5, 0.4, 0.45, 0.55)
    assert 0 <= bdr <= 1


def test_state_only_shortcut_index():
    idx = compute_state_only_shortcut_index(0.5, 0.55)
    assert idx > 0


def test_shuffled_action_gap():
    gap = compute_shuffled_action_gap(0.5, 0.4)
    assert abs(gap - 0.1) < 1e-9


def test_seed_stability_ratio():
    ssr = compute_seed_stability_ratio(0.1, 0.2)
    assert ssr == 0.5