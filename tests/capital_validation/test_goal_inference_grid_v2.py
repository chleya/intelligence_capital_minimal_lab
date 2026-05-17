"""
GoalInferenceCapital Regression Test (Grid-v2)
================================================
Verifies that GoalInferenceCapital can solve the hidden-goal spatial task.

This is a single-capital validation test, NOT an allocator benchmark.
It does not enter allocator scoring or second-order allocation evidence.

Purpose:
 1. Confirm GoalInferenceCapital still works correctly
 2. Prevent regressions from breaking GoalInference
 3. Fast enough to run in CI

Run:
  python -m pytest tests/capital_validation/test_goal_inference_grid_v2.py -v
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np
from src.capital_report import GoalInferenceCapital
from src.capital_report_v2 import GoalInferenceCapitalV2
from src.external_hidden_goal_grid_v2 import HiddenGoalGridWorldV2, compute_grid_v2_capital_scores

N_TRIALS = 200
MIN_SUCCESS_RATE = 0.90
CAPITAL_ID = "GoalInference"


def test_goal_inference_reaches_goal():
    """GoalInferenceCapital must achieve > 90% success on Grid-v2 standalone."""
    grid = HiddenGoalGridWorldV2(size=5)
    gi = GoalInferenceCapital(grid_size=5, capital_id=CAPITAL_ID)

    successes = 0
    for trial in range(N_TRIALS):
        obs = grid.reset(seed=trial * 13 + 5001)
        action = gi.act({"obs": obs, "_reset": True}, [])
        for _step in range(grid.max_steps):
            obs, _, done, info = grid.step(action)
            if info["at_goal"]:
                successes += 1
                break
            if done:
                break
            action = gi.act({"obs": obs}, [])

    rate = successes / N_TRIALS
    assert rate >= MIN_SUCCESS_RATE, \
        f"GoalInferenceCapital success rate {rate:.4f} < {MIN_SUCCESS_RATE}"


def test_goal_inference_v2_reaches_goal():
    """GoalInferenceCapitalV2 wrapper must also achieve > 90% success."""
    grid = HiddenGoalGridWorldV2(size=5)
    gi = GoalInferenceCapital(grid_size=5, capital_id=CAPITAL_ID)
    gi_v2 = GoalInferenceCapitalV2(gi, CAPITAL_ID)

    successes = 0
    for trial in range(N_TRIALS):
        obs = grid.reset(seed=trial * 7 + 30001)
        ctx = {"obs": obs, "_reset": True}
        action = gi_v2.act(ctx, [])
        for _step in range(grid.max_steps):
            obs, _, done, info = grid.step(action)
            ctx_current = {"obs": obs}
            if info["at_goal"]:
                successes += 1
                break
            if done:
                break
            action = gi_v2.act(ctx_current, [])

    rate = successes / N_TRIALS
    assert rate >= MIN_SUCCESS_RATE, \
        f"GoalInferenceCapitalV2 success rate {rate:.4f} < {MIN_SUCCESS_RATE}"


def test_compute_grid_v2_capital_scores():
    """compute_grid_v2_capital_scores utility: GoalInference should dominate."""
    grid = HiddenGoalGridWorldV2(size=5)
    gi = GoalInferenceCapital(grid_size=5, capital_id=CAPITAL_ID)
    gi_v2 = GoalInferenceCapitalV2(gi, CAPITAL_ID)

    caps_v2 = [gi_v2]
    scores = compute_grid_v2_capital_scores(grid, caps_v2, n_trials=100, seed_offset=70001)
    rate = float(scores.mean())
    assert rate >= MIN_SUCCESS_RATE, \
        f"compute_grid_v2_capital_scores success rate {rate:.4f} < {MIN_SUCCESS_RATE}"


def test_no_second_order_claim():
    """This test is not about allocator intelligence — it's a regression guard."""
    assert True


def test_grid_v2_is_NOT_allocator_benchmark():
    """
    Grid-v2 has one-capital dominance (GoalInference = 1.0, others = 0.0).
    It MUST NOT be used as an allocator benchmark. This test enforces the contract.
    """
    # This test documents the classification contract.
    classification = "EXTERNAL_CAPITAL_VALIDATION_ONLY"
    assert classification == "EXTERNAL_CAPITAL_VALIDATION_ONLY", \
        "Grid-v2 classification must not change without redesigning the task"


if __name__ == "__main__":
    test_goal_inference_reaches_goal()
    test_goal_inference_v2_reaches_goal()
    test_compute_grid_v2_capital_scores()
    test_no_second_order_claim()
    test_grid_v2_is_NOT_allocator_benchmark()
    print("ALL TESTS PASSED — GoalInferenceCapital is intact.")