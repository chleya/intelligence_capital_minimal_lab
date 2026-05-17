import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.env_structured_volatility import StructuredVolatilityEnv


def test_env_reset():
    env = StructuredVolatilityEnv(seed=42)
    obs = env.reset()
    assert obs.shape == (8 * (2 + 1),), f"Expected obs shape {8*(2+1)}, got {obs.shape}"
    assert env.t == 0
    assert env.mode in (0, 1)


def test_env_step():
    env = StructuredVolatilityEnv(seed=42)
    env.reset()
    obs, state, mode = env.step(1)
    assert obs.shape == (8 * (2 + 1),)
    assert state.shape == (2,)
    assert mode in (0, 1)
    assert env.t == 1


def test_env_snapshot_restore():
    env = StructuredVolatilityEnv(seed=42)
    env.reset()
    env.step(1)
    snap = env.snapshot()
    env.step(-1)
    env.step(0)
    env.restore(snap)
    assert env.t == snap["t"]


def test_env_compute_outcomes():
    env = StructuredVolatilityEnv(seed=42)
    env.reset()
    outcomes = env.compute_outcomes(horizon=1)
    assert -1 in outcomes and 0 in outcomes and 1 in outcomes
    for a, v in outcomes.items():
        assert v.shape == (2,)


def test_env_oracle_residual():
    env = StructuredVolatilityEnv(seed=42)
    env.reset()
    residuals = env.get_oracle_residual(horizon=1)
    assert -1 in residuals and 0 in residuals and 1 in residuals
    assert np.allclose(residuals[0], np.zeros(2))
    assert not np.allclose(residuals[-1], np.zeros(2))
    assert not np.allclose(residuals[1], np.zeros(2))


def test_env_best_action():
    env = StructuredVolatilityEnv(seed=42)
    env.reset()
    best = env.get_best_action(horizon=1)
    assert best in (-1, 0, 1)


def test_mode_flip():
    env = StructuredVolatilityEnv(seed=42, mode_flip_prob=1.0)
    env.reset()
    initial_mode = env.mode
    modes = []
    for _ in range(20):
        _, _, m = env.step(0)
        modes.append(m)
    assert any(m != initial_mode for m in modes), "Mode should flip with prob 1.0"


def test_autonomous_vs_action_effect():
    env = StructuredVolatilityEnv(seed=42, autonomous_drift=0.0, autonomous_noise=0.0,
                                   action_gain=0.5, action_noise=0.0)
    env.reset()
    env.step(-1)
    state1 = env.get_current_state().copy()
    env.restore(env.snapshot())
    env.step(1)
    state2 = env.get_current_state()
    assert not np.allclose(state1, state2), "Action -1 and +1 should produce different states"