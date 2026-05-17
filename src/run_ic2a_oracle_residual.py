"""IC-2a: Oracle Residual Accounting Test.

Tests whether an oracle that knows the true hidden mode can:
1. Beat StateOnly (autonomous dynamics)
2. Beat ActionOnly (most-common-action)
3. Demonstrate that CF data (knowing all 3 actions per state) provides value

This is the gate: if oracle cannot beat SO, then no learned model can.
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env_structured_volatility import StructuredVolatilityEnv
from src.counterfactual_table import generate_counterfactual_table, compute_oracle_summary
from src.metrics import (
    compute_best_action_match, compute_residual_variance_ratio,
    compute_oracle_action_entropy, compute_action_only_ceiling,
    compute_counterfactual_value, compute_seed_stability_ratio,
)


def main():
    print("=" * 60)
    print("IC-2a: Oracle Residual Accounting Test")
    print("=" * 60)

    env_kwargs = dict(
        state_dim=2, mode_flip_prob=0.08, autonomous_drift=0.05,
        autonomous_noise=0.02, action_gain=0.70, action_noise=0.03,
        action_sign_flip=True, history_len=8,
    )
    seeds = list(range(10))

    print("\n[1/4] Generating counterfactual table...")
    cf_df = generate_counterfactual_table(
        StructuredVolatilityEnv, env_kwargs,
        n_train=1200, n_val=200, n_test_id=200, n_ood=300,
        horizons=(1, 3, 5), seeds=seeds,
    )
    cf_df.to_csv("results/counterfactual_table.csv", index=False)
    print(f"  Generated {len(cf_df)} rows across {len(seeds)} seeds.")

    print("\n[2/4] Computing oracle summary per seed...")
    results_per_seed = []
    for seed in seeds:
        summary = compute_oracle_summary(cf_df, StructuredVolatilityEnv, env_kwargs, seed=seed)
        results_per_seed.append(summary)
        print(f"  Seed {seed}: RVR={summary['residual_variance_ratio']:.4f}, "
              f"SO={summary['so_match']:.3f}, AO={summary['ao_match']:.3f}")

    dfr = pd.DataFrame(results_per_seed)

    print("\n[3/4] Computing aggregate metrics...")
    aggregate = {
        "mean_residual_variance_ratio": float(dfr["residual_variance_ratio"].mean()),
        "std_residual_variance_ratio": float(dfr["residual_variance_ratio"].std()),
        "mean_so_match": float(dfr["so_match"].mean()),
        "std_so_match": float(dfr["so_match"].std()),
        "mean_ao_match": float(dfr["ao_match"].mean()),
        "std_ao_match": float(dfr["ao_match"].std()),
    }

    oracle_match = 1.0
    oracle_beats_so = oracle_match > aggregate["mean_so_match"]
    cf_gain = oracle_match - aggregate["mean_ao_match"]
    cf_has_value = cf_gain > 0.10

    seed_gap = max(aggregate["std_residual_variance_ratio"], aggregate["std_ao_match"])
    model_gap = 0.15
    seed_stability = seed_gap / model_gap if model_gap > 0 else float("inf")

    aggregate["oracle_match"] = oracle_match
    aggregate["oracle_so_gap"] = float(oracle_match - aggregate["mean_so_match"])
    aggregate["oracle_ao_gap"] = float(oracle_match - aggregate["mean_ao_match"])
    aggregate["cf_gain"] = float(cf_gain)
    aggregate["cf_has_value"] = cf_has_value
    aggregate["oracle_beats_so"] = oracle_beats_so
    aggregate["seed_stability_ratio"] = seed_stability

    print(f"  Residual Variance Ratio: {aggregate['mean_residual_variance_ratio']:.4f}")
    print(f"  StateOnly Match:         {aggregate['mean_so_match']:.3f} ± {aggregate['std_so_match']:.3f}")
    print(f"  ActionOnly Match:        {aggregate['mean_ao_match']:.3f} ± {aggregate['std_ao_match']:.3f}")
    print(f"  Oracle→SO gap:           {aggregate['oracle_so_gap']:.3f}")
    print(f"  Oracle→AO gap (CF gain): {aggregate['oracle_ao_gap']:.3f}")
    print(f"  Seed Stability Ratio:    {seed_stability:.4f}")

    print("\n[4/4] Gate checks:")
    gates = {
        "residual_signal_present": aggregate["mean_residual_variance_ratio"] >= 0.15,
        "oracle_beats_so": oracle_beats_so,
        "cf_has_value": cf_has_value,
        "benchmark_stable": seed_stability < 0.5,
        "so_under_30": aggregate["mean_so_match"] < 0.30,
    }

    for gate, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {gate}")

    all_pass = all(gates.values())

    with open("results/ic2a_gates.json", "w") as f:
        json.dump({"gates": gates, "all_pass": all_pass, "aggregate": aggregate}, f, indent=2)

    print(f"\n{'='*60}")
    if all_pass:
        print("IC-2a PASSED. Proceed to IC-2b.")
    else:
        print("IC-2a FAILED. Redesign environment and re-run.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()