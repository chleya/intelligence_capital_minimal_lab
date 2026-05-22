"""r
Phase 6-A: Stabilization Scaling — Seed Scaling.
==================================================
Tests whether Per-Action KMeans (0.585) survives scaling from
5 seeds to 20, 50, 100 seeds.

Design from PHASE_6_7_8_PLAN.md Section 2.3.

Usage:
  cd F:\\intelligence_capital_minimal_lab
  python src/run_c4_stabilization_scaling.py --level S1     (20 seeds, ~2 min)
  python src/run_c4_stabilization_scaling.py --level S2     (50 seeds, ~5 min)
  python src/run_c4_stabilization_scaling.py --level S3     (100 seeds, ~12 min)
  python src/run_c4_stabilization_scaling.py --level S1 --fast   (skip data gen)
"""

import argparse
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import prepare_counterfactual_data
from src.counterfactual_table import generate_counterfactual_table
from src.env_structured_volatility import StructuredVolatilityEnv

RESULTS_DIR = "results/ic2c_scaling"
os.makedirs(RESULTS_DIR, exist_ok=True)

ENV_KWARGS = dict(
    state_dim=2, mode_flip_prob=0.08, autonomous_drift=0.05,
    autonomous_noise=0.02, action_gain=0.70, action_noise=0.03,
    action_sign_flip=True, history_len=8,
    action_cost=0.20, state_dependent_gain=True, saturation_k=0.5,
)
RANDOM_BASELINE = 1.0 / 3.0
N_PROTOTYPES = 20

SCALING_LEVELS = {
    "S1": {"n_seeds": 20, "label": "20 seeds"},
    "S2": {"n_seeds": 50, "label": "50 seeds"},
    "S3": {"n_seeds": 100, "label": "100 seeds"},
}

# ============================================================
# Strategy classes (imported from C3 for self-containment)
# ============================================================

class KMeansBaseline:
    def __init__(self, n_prototypes=20):
        self.n_prototypes = n_prototypes
        self._centroids = None
        self._Y_centroids = None
        self._X_all = []
        self._Y_all = []

    def update(self, X_new, Y_new, seed_label=None):
        self._X_all.extend(list(X_new))
        self._Y_all.extend(list(Y_new))
        X_arr = np.array(self._X_all)
        Y_arr = np.array(self._Y_all)
        nc = min(self.n_prototypes, len(X_arr))
        km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
        labels = km.fit_predict(X_arr)
        self._centroids = km.cluster_centers_
        self._Y_centroids = np.zeros((nc, 3))
        for i in range(nc):
            mask = labels == i
            if mask.sum() > 0:
                self._Y_centroids[i] = Y_arr[mask].mean(axis=0)

    def predict(self, X_query):
        if self._centroids is None:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids)
        return self._Y_centroids[idxs]


class YAwareKMeans:
    def __init__(self, n_prototypes=20, y_weight=5.0):
        self.n_prototypes = n_prototypes
        self.y_weight = y_weight
        self._centroids_X = None
        self._Y_centroids = None
        self._X_all = []
        self._Y_all = []

    def update(self, X_new, Y_new, seed_label=None):
        self._X_all.extend(list(X_new))
        self._Y_all.extend(list(Y_new))
        X_arr = np.array(self._X_all)
        Y_arr = np.array(self._Y_all)
        X_norm = X_arr / (np.std(X_arr, axis=0) + 1e-8)
        Y_norm = Y_arr / (np.std(Y_arr, axis=0) + 1e-8)
        joint = np.hstack([X_norm, self.y_weight * Y_norm])
        nc = min(self.n_prototypes, len(X_arr))
        km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
        labels = km.fit_predict(joint)
        self._centroids_X = np.zeros((nc, X_arr.shape[1]))
        self._Y_centroids = np.zeros((nc, 3))
        for i in range(nc):
            mask = labels == i
            if mask.sum() > 0:
                self._centroids_X[i] = X_arr[mask].mean(axis=0)
                self._Y_centroids[i] = Y_arr[mask].mean(axis=0)

    def predict(self, X_query):
        if self._centroids_X is None:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids_X)
        return self._Y_centroids[idxs]


class PerActionKMeans:
    def __init__(self, n_prototypes_per_action=7):
        self.n_prototypes_per_action = n_prototypes_per_action
        self._per_action = {0: {"X": [], "Y": []},
                            1: {"X": [], "Y": []},
                            2: {"X": [], "Y": []}}
        self._centroids = None
        self._Y_centroids = None

    def update(self, X_new, Y_new, seed_label=None):
        best_actions = np.argmax(Y_new, axis=1)
        for i in range(len(X_new)):
            a = int(best_actions[i])
            self._per_action[a]["X"].append(X_new[i])
            self._per_action[a]["Y"].append(Y_new[i])
        centroids_parts = []
        Y_parts = []
        for a in range(3):
            X_arr = np.array(self._per_action[a]["X"])
            Y_arr = np.array(self._per_action[a]["Y"])
            nc = min(self.n_prototypes_per_action, len(X_arr))
            if nc < 1:
                continue
            km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
            labels = km.fit_predict(X_arr)
            for i in range(nc):
                mask = labels == i
                if mask.sum() > 0:
                    centroids_parts.append(km.cluster_centers_[i])
                    Y_parts.append(Y_arr[mask].mean(axis=0))
        if len(centroids_parts) > 0:
            self._centroids = np.stack(centroids_parts)
            self._Y_centroids = np.stack(Y_parts)

    def predict(self, X_query):
        if self._centroids is None:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids)
        return self._Y_centroids[idxs]


class AdaptivePerActionKMeans:
    def __init__(self, n_max=20):
        self.n_max = n_max
        self._per_action = {0: {"X": [], "Y": []},
                            1: {"X": [], "Y": []},
                            2: {"X": [], "Y": []}}
        self._centroids = None
        self._Y_centroids = None

    def update(self, X_new, Y_new, seed_label=None):
        best_actions = np.argmax(Y_new, axis=1)
        for i in range(len(X_new)):
            a = int(best_actions[i])
            self._per_action[a]["X"].append(X_new[i])
            self._per_action[a]["Y"].append(Y_new[i])
        centroids_parts = []
        Y_parts = []
        for a in range(3):
            X_arr = np.array(self._per_action[a]["X"])
            Y_arr = np.array(self._per_action[a]["Y"])
            if len(X_arr) == 0:
                continue
            nc = max(3, min(self.n_max, int(np.sqrt(len(X_arr)))))
            km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
            labels = km.fit_predict(X_arr)
            for i in range(nc):
                mask = labels == i
                if mask.sum() > 0:
                    centroids_parts.append(km.cluster_centers_[i])
                    Y_parts.append(Y_arr[mask].mean(axis=0))
        if len(centroids_parts) > 0:
            self._centroids = np.stack(centroids_parts)
            self._Y_centroids = np.stack(Y_parts)

    def predict(self, X_query):
        if self._centroids is None:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids)
        return self._Y_centroids[idxs]


class NoMemoryBaseline:
    def __init__(self):
        self._action_counts = None

    def update(self, X_new, Y_new, seed_label=None):
        if self._action_counts is None:
            self._action_counts = np.zeros(3)
        best_actions = np.argmax(Y_new, axis=1)
        for ba in best_actions:
            self._action_counts[ba] += 1

    def predict(self, X_query):
        if self._action_counts is None or self._action_counts.sum() == 0:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE
        probs = self._action_counts / self._action_counts.sum()
        return np.tile(probs, (len(X_query), 1))


def best_action_match(preds, Y_true):
    pred_actions = np.argmax(preds, axis=1)
    true_actions = np.argmax(Y_true, axis=1)
    return np.mean(pred_actions == true_actions)


def generate_scaled_cf_data(n_seeds, cf_path):
    if os.path.exists(cf_path):
        df = pd.read_csv(cf_path)
        existing = df["seed"].nunique()
        if existing >= n_seeds:
            return df
    print(f"Generating counterfactual data for {n_seeds} seeds...")
    t0 = time.time()
    df = generate_counterfactual_table(
        StructuredVolatilityEnv, ENV_KWARGS,
        n_train=1200, n_val=200, n_test_id=200, n_ood=300,
        horizons=(1, 3, 5), seeds=range(n_seeds),
    )
    df.to_csv(cf_path, index=False)
    print(f"  Generated {len(df)} rows in {time.time()-t0:.0f}s, saved to {cf_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="S1", choices=["S1", "S2", "S3"])
    parser.add_argument("--fast", action="store_true", help="Skip counterfactual data generation")
    args = parser.parse_args()

    cfg = SCALING_LEVELS[args.level]
    n_seeds = cfg["n_seeds"]
    level = args.level
    label = cfg["label"]

    run_id = f"c4_{level}_{int(time.time())}"
    log_path = os.path.join(RESULTS_DIR, f"run_log_{level}.txt")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()

    log(f"Phase 6-A: Seed Scaling {level} ({label})")
    log("=" * 60)
    log(f"C3 baseline (5 seeds): per_action_kmeans=0.585, nomemory=0.445, kmeans_20=0.095")
    log(f"Now testing at {n_seeds} seeds.")
    log(f"Run ID: {run_id}")

    cf_path = f"results/counterfactual_table_{n_seeds}s.csv"
    cf_df = generate_scaled_cf_data(n_seeds, cf_path)

    per_seed_train = {}
    for seed in range(n_seeds):
        train_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "train") & (cf_df["horizon"] == 1)]
        X_tr, Y_tr, ba_tr = prepare_counterfactual_data(train_df, seed, ENV_KWARGS)
        if X_tr is not None and len(X_tr) > 0:
            per_seed_train[seed] = (X_tr, Y_tr, ba_tr, train_df)
    train_sizes = [(s, len(v[0])) for s, v in per_seed_train.items()]
    log(f"Per-seed train sizes (first 5): {train_sizes[:5]} ... total={sum(s[1] for s in train_sizes)}")

    test_df = cf_df[(cf_df["seed"] == 0) & (cf_df["split"] == "test_id") & (cf_df["horizon"] == 1)]
    X_test, Y_test, ba_test = prepare_counterfactual_data(test_df, 0, ENV_KWARGS)
    log(f"Test set: {len(X_test)} samples (seed=0, horizon=1, split=test_id)")

    strategies = {
        "kmeans_20": lambda: KMeansBaseline(n_prototypes=20),
        "y_aware_w5.0": lambda: YAwareKMeans(n_prototypes=20, y_weight=5.0),
        "per_action_kmeans": lambda: PerActionKMeans(n_prototypes_per_action=7),
        "adaptive_per_action": lambda: AdaptivePerActionKMeans(n_max=20),
        "nomemory": lambda: NoMemoryBaseline(),
    }

    all_records = []
    update_steps = n_seeds

    for sname, factory in strategies.items():
        log(f"\n[{sname}]")
        strategy = factory()
        for step in range(update_steps):
            seed = step
            if seed not in per_seed_train:
                continue
            X_step, Y_step, _, _ = per_seed_train[seed]
            strategy.update(X_step, Y_step, seed_label=seed)
            preds = strategy.predict(X_test)
            match = best_action_match(preds, Y_test)
            all_records.append({"step": step + 1, "strategy": sname,
                                "best_action_match": match, "n_seeds": n_seeds, "level": level})
        log(f"  Final match: {all_records[-1]['best_action_match']:.4f}")

    df = pd.DataFrame(all_records)
    csv_path = os.path.join(RESULTS_DIR, f"scaling_{level}_results.csv")
    df.to_csv(csv_path, index=False)
    log(f"\nResults saved to {csv_path}")

    final_step = df[df["step"] == update_steps]
    log(f"\n{'='*60}")
    log(f"Final Step ({update_steps}) Results ({label}):")
    log(f"{'='*60}")

    kmeans_final = final_step[final_step["strategy"] == "kmeans_20"]["best_action_match"].values[0]
    y_aware_final = final_step[final_step["strategy"] == "y_aware_w5.0"]["best_action_match"].values[0]
    pa_final = final_step[final_step["strategy"] == "per_action_kmeans"]["best_action_match"].values[0]
    adaptive_final = final_step[final_step["strategy"] == "adaptive_per_action"]["best_action_match"].values[0]
    nomemory_final = final_step[final_step["strategy"] == "nomemory"]["best_action_match"].values[0]

    log(f"  kmeans_20:            {kmeans_final:.4f}")
    log(f"  y_aware_w5.0:         {y_aware_final:.4f}")
    log(f"  per_action_kmeans:    {pa_final:.4f}")
    log(f"  adaptive_per_action:  {adaptive_final:.4f}")
    log(f"  nomemory:             {nomemory_final:.4f}")

    delta_pa = pa_final - nomemory_final
    log(f"\n  Delta(per_action - nomemory) = {delta_pa:+.4f}")

    c3_pa = 0.585
    c3_nm = 0.445
    log(f"\n  vs C3 (5 seeds): per_action={c3_pa:.3f}, nomemory={c3_nm:.3f}, delta={c3_pa-c3_nm:+.3f}")

    if pa_final > nomemory_final + 0.05:
        log("\n  VERDICT: STRONG PASS — Per-Action dominates at scale.")
    elif pa_final > nomemory_final:
        log(f"\n  VERDICT: CONDITIONAL PASS — Per-Action still ahead but delta={delta_pa:+.4f} (< 0.05).")
    elif pa_final >= nomemory_final - 0.02:
        log("\n  VERDICT: MARGINAL — Per-Action approximately equals NoMemory at this scale.")
    else:
        log("\n  VERDICT: FAIL — Per-Action falls below NoMemory at this scale.")

    if adaptive_final > pa_final:
        log(f"\n  Adaptive centroids ({adaptive_final:.4f}) > fixed ({pa_final:.4f}) — adaptive helps at scale.")
    else:
        log(f"\n  Adaptive ({adaptive_final:.4f}) <= fixed ({pa_final:.4f}) — fixed is sufficient.")

    log("\n" + "=" * 60)
    log("Phase 6-A scaling experiment complete.")
    log("=" * 60)


if __name__ == "__main__":
    main()