"""
Phase 6-C: Noise Scaling.
===========================
Tests whether Per-Action KMeans advantage persists under
increasing environmental noise (autonomous_noise + action_noise).

Design from unified research roadmap Section 1.2.

Usage:
  cd F:\\intelligence_capital_minimal_lab
  python src/run_c4b_noise_scaling.py
"""

import os, sys, json, time, copy
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import prepare_counterfactual_data
from src.counterfactual_table import generate_counterfactual_table
from src.env_structured_volatility import StructuredVolatilityEnv

RESULTS_DIR = "results/ic2c_scaling"
os.makedirs(RESULTS_DIR, exist_ok=True)

BASE_ENV = dict(
    state_dim=2, mode_flip_prob=0.08, autonomous_drift=0.05,
    autonomous_noise=0.02, action_gain=0.70, action_noise=0.03,
    action_sign_flip=True, history_len=8,
    action_cost=0.20, state_dependent_gain=True, saturation_k=0.5,
)

NOISE_LEVELS = {
    "N_low":   {"autonomous_noise": 0.03, "action_noise": 0.05,  "label": "Low noise"},
    "N_mid":   {"autonomous_noise": 0.10, "action_noise": 0.15,  "label": "Mid noise"},
    "N_high":  {"autonomous_noise": 0.30, "action_noise": 0.40,  "label": "High noise"},
}

N_SEEDS = 5
RANDOM_BASELINE = 1.0 / 3.0

from run_c4_stabilization_scaling import (
    KMeansBaseline, YAwareKMeans, PerActionKMeans,
    AdaptivePerActionKMeans, NoMemoryBaseline, best_action_match,
)

def generate_noise_data(env_kwargs, tag):
    cf_path = f"results/counterfactual_table_noise_{tag}.csv"
    if os.path.exists(cf_path):
        return pd.read_csv(cf_path)
    print(f"  Generating CF data (noise={tag})...")
    df = generate_counterfactual_table(
        StructuredVolatilityEnv, env_kwargs,
        n_train=1200, n_val=200, n_test_id=200, n_ood=300,
        horizons=(1, 3, 5), seeds=range(N_SEEDS),
    )
    df.to_csv(cf_path, index=False)
    return df

def run_strategies(cf_df, env_kwargs, label):
    strategies = {
        "kmeans_20": lambda: KMeansBaseline(n_prototypes=20),
        "y_aware_w5.0": lambda: YAwareKMeans(n_prototypes=20, y_weight=5.0),
        "per_action_kmeans": lambda: PerActionKMeans(n_prototypes_per_action=7),
        "nomemory": lambda: NoMemoryBaseline(),
    }

    per_seed_train = {}
    for seed in range(N_SEEDS):
        train_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "train") & (cf_df["horizon"] == 1)]
        X_tr, Y_tr, ba_tr = prepare_counterfactual_data(train_df, seed, env_kwargs)
        if X_tr is not None and len(X_tr) > 0:
            per_seed_train[seed] = (X_tr, Y_tr, ba_tr, train_df)

    test_df = cf_df[(cf_df["seed"] == 0) & (cf_df["split"] == "test_id") & (cf_df["horizon"] == 1)]
    X_test, Y_test, ba_test = prepare_counterfactual_data(test_df, 0, env_kwargs)

    results = {}
    for sname, factory in strategies.items():
        strategy = factory()
        for seed in range(N_SEEDS):
            if seed not in per_seed_train:
                continue
            X_tr, Y_tr, ba_tr, _ = per_seed_train[seed]
            strategy.update(X_tr, Y_tr, seed_label=seed)
        match = best_action_match(strategy.predict(X_test), Y_test)
        results[sname] = match
        print(f"  {sname:25s}  match={match:.4f}")
    return results

def main():
    print("Phase 6-C: Noise Scaling")
    print("=" * 60)
    print(f"Testing 3 noise levels with {N_SEEDS} seeds each.")
    print()

    unperturbed_cf = generate_noise_data(copy.deepcopy(BASE_ENV), "base")
    print("\n--- Baseline (unperturbed, noise=0.02/0.03) ---")
    baseline = run_strategies(unperturbed_cf, BASE_ENV, "baseline")
    print(f"  Delta(PA - NM) = +{baseline['per_action_kmeans'] - baseline['nomemory']:.4f}")

    all_results = []
    for level_tag, noise_overrides in NOISE_LEVELS.items():
        env = copy.deepcopy(BASE_ENV)
        label = noise_overrides.get("label", level_tag)
        env.update({k: v for k, v in noise_overrides.items() if k != "label"})
        print(f"\n--- {level_tag}: {label} (anoise={env['autonomous_noise']:.2f}, anoise={env['action_noise']:.2f}) ---")
        cf_df = generate_noise_data(env, level_tag)
        res = run_strategies(cf_df, env, label)
        delta = res['per_action_kmeans'] - res['nomemory']
        nm_decay = baseline['nomemory'] - res['nomemory']
        pa_decay = baseline['per_action_kmeans'] - res['per_action_kmeans']
        print(f"  Delta(PA - NM) = +{delta:.4f}")
        print(f"  PA decay: {pa_decay:.4f}, NM decay: {nm_decay:.4f}")
        res['level'] = level_tag
        res['label'] = label
        res['delta'] = delta
        res['pa_decay'] = pa_decay
        res['nm_decay'] = nm_decay
        all_results.append(res)

    print("\n" + "=" * 60)
    print("Noise Scaling Summary:")
    print("=" * 60)
    print(f"  Baseline:           PA={baseline['per_action_kmeans']:.4f}, NM={baseline['nomemory']:.4f}, d=+{baseline['per_action_kmeans']-baseline['nomemory']:.4f}")
    for r in all_results:
        print(f"  {r['label']:15s} PA={r['per_action_kmeans']:.4f}, NM={r['nomemory']:.4f}, d=+{r['delta']:.4f}, pa_decay={r['pa_decay']:.4f}")

    pa_decays = [r['pa_decay'] for r in all_results]
    if all(d > 0.05 for d in pa_decays):
        verdict = "PA degrades faster than NM under noise — NO robustness advantage"
    elif all(r['delta'] > 0.05 for r in all_results):
        verdict = "PA maintains advantage under noise — robustness confirmed"
    else:
        verdict = "MIXED — advantage narrows but survives at certain noise levels"
    print(f"\n  VERDICT: {verdict}")

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, "noise_scaling_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()