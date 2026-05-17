import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.env_structured_volatility import StructuredVolatilityEnv
from src.counterfactual_table import generate_trajectory, generate_counterfactual_table


def test_generate_trajectory():
    env = StructuredVolatilityEnv(seed=42)
    records = generate_trajectory(env, steps=100, seed=42)
    assert len(records) == 100
    for rec in records:
        assert "t" in rec
        assert "action" in rec
        assert "state_next" in rec
        assert "mode" in rec


def test_generate_counterfactual_table():
    env_kwargs = dict(state_dim=2, mode_flip_prob=0.1)
    cf_df = generate_counterfactual_table(
        StructuredVolatilityEnv, env_kwargs,
        n_train=100, n_val=20, n_test_id=30, n_ood=30,
        horizons=(1,), seeds=[0, 1],
    )
    assert len(cf_df) > 0
    expected_cols = ["seed", "state_idx", "split", "horizon", "outcome_m1", "outcome_0", "outcome_p1", "best_action"]
    for col in expected_cols:
        assert col in cf_df.columns, f"Missing column: {col}"


def test_cf_table_across_actions():
    env_kwargs = dict(state_dim=2, mode_flip_prob=0.1)
    cf_df = generate_counterfactual_table(
        StructuredVolatilityEnv, env_kwargs,
        n_train=20, n_val=5, n_test_id=5, n_ood=5,
        horizons=(1,), seeds=[0,],
    )
    sub = cf_df[(cf_df["seed"] == 0) & (cf_df["split"] == "train")]
    for _, row in sub.iterrows():
        assert row["outcome_m1"] is not None
        assert row["outcome_0"] is not None
        assert row["outcome_p1"] is not None