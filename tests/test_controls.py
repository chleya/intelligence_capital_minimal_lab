import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.env_structured_volatility import StructuredVolatilityEnv
from src.throttling_mechanisms import StateOnlyMechanism, ActionOnlyMechanism
from src.audit import check_death, run_full_audit


def test_state_only_shortcut():
    env = StructuredVolatilityEnv(seed=42, autonomous_drift=0.3, autonomous_noise=0.01,
                                   action_gain=0.05, action_noise=0.01)
    env.reset()
    for _ in range(50):
        env.step(0)
    for _ in range(20):
        a = env.rng.choice([-1, 0, 1])
        env.step(a)

    obs_list = []
    best_acts = []
    for _ in range(100):
        env.step(env.rng.choice([-1, 0, 1]))
        outcomes = env.compute_outcomes(horizon=1)
        obs_list.append(env._get_obs())
        best_acts.append(max(outcomes, key=lambda a: np.sum(outcomes[a])))

    so_mech = StateOnlyMechanism(2, 8)
    assert so_mech.model is not None
    assert len(best_acts) == 100


def test_death_check_d1():
    metrics = {"residual_variance_ratio": 0.05}
    result = check_death("D1_residual_absent", metrics)
    assert result is not None
    assert result["triggered"] is True


def test_death_check_d1_pass():
    metrics = {"residual_variance_ratio": 0.30}
    result = check_death("D1_residual_absent", metrics)
    assert result["triggered"] is False


def test_death_check_d5():
    metrics = {"bad_debt_ratio": 0.7}
    result = check_death("D5_mostly_bad_debt", metrics)
    assert result["triggered"] is True


def test_death_check_d6():
    metrics = {"permuted_history_gap": 0.02}
    result = check_death("D6_temporal_broken", metrics)
    assert result is not None
    assert result["triggered"] is True


def test_full_audit():
    metrics = {
        "residual_variance_ratio": 0.30,
        "oracle_match": 0.75,
        "so_match": 0.50,
        "counterfactual_value": 0.15,
        "bad_debt_ratio": 0.3,
        "permuted_history_gap": 0.15,
        "seed_stability_ratio": 0.2,
        "transfer_premium": 0.10,
        "orr": 0.7,
        "state_only_shortcut_index": 0.6,
        "action_only_shortcut_index": 0.5,
        "shuffled_action_gap": 0.10,
    }
    results = run_full_audit(metrics)
    for r in results:
        if r and r["triggered"]:
            print(f"Unexpected death: {r['condition_id']} - {r['message']}")
    triggered = sum(1 for r in results if r and r["triggered"])
    assert triggered == 0, f"Expected 0 deaths, got {triggered}"