"""
Phase 6-B/C: Objective Scaling + Noise Scaling.
================================================
Two experiments in one script:

6-B: Objective Scaling (Multi-Action)
  -- Use 5-seed counterfactual data
  -- Quantize soft Y vectors into 3, 5, 10, 20 discrete actions
  -- Test: does Y-aware advantage grow with more actions?

6-C: Noise Scaling
  -- Add controlled noise to X during training
  -- Noise levels: 0.0, 0.03, 0.10, 0.30, 1.00 (x std(X))
  -- Test: does Y-aware degrade slower than X-only?

Hypotheses:
  H6.5: action越多，X-only越难（action粒度细 → X结构不足以区分）
  H6.6: Y-aware在高噪声下更鲁棒（Y信号补偿X噪声）

Usage:
  cd F:\intelligence_capital_minimal_lab
  python src/run_c4_objective_noise_scaling.py
"""

import os, sys, time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import prepare_counterfactual_data
from src.counterfactual_table import generate_counterfactual_table
from src.env_structured_volatility import StructuredVolatilityEnv

RESULTS_DIR = "results/ic2c_scaling"
os.makedirs(RESULTS_DIR, exist_ok=True)
CF_PATH = "results/counterfactual_table.csv"

ENV_KWARGS = dict(
    state_dim=2, mode_flip_prob=0.08, autonomous_drift=0.05,
    autonomous_noise=0.02, action_gain=0.70, action_noise=0.03,
    action_sign_flip=True, history_len=8,
    action_cost=0.20, state_dependent_gain=True, saturation_k=0.5,
)
N_SEEDS = 5


class PerActionKMeans:
    def __init__(self, n_actions, n_prototypes_per_action=3):
        self.n_actions = n_actions
        self.n_prototypes_per_action = n_prototypes_per_action
        self._per_action = {a: {"X": [], "Y": []} for a in range(n_actions)}
        self._centroids = None
        self._Y_centroids = None

    def update(self, X_new, Y_new, seed_label=None):
        best_actions = np.argmax(Y_new, axis=1)
        for i in range(len(X_new)):
            a = int(best_actions[i])
            if a < self.n_actions:
                self._per_action[a]["X"].append(X_new[i])
                self._per_action[a]["Y"].append(Y_new[i])
        self._rebuild()

    def _rebuild(self):
        centroids_parts, Y_parts = [], []
        for a in range(self.n_actions):
            X_arr = np.array(self._per_action[a]["X"])
            Y_arr = np.array(self._per_action[a]["Y"])
            nc = min(self.n_prototypes_per_action, len(X_arr), max(1, len(X_arr) // 3))
            if nc < 1:
                continue
            km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
            labels = km.fit_predict(X_arr)
            for i in range(nc):
                m = labels == i
                if m.sum() > 0:
                    centroids_parts.append(km.cluster_centers_[i])
                    Y_parts.append(Y_arr[m].mean(axis=0))
        if centroids_parts:
            self._centroids = np.stack(centroids_parts)
            self._Y_centroids = np.stack(Y_parts)

    def predict(self, X_query):
        if self._centroids is None:
            return np.ones((len(X_query), self.n_actions)) / self.n_actions
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids)
        return self._Y_centroids[idxs]


class XOnlyKMeans:
    def __init__(self, n_actions, n_prototypes=10):
        self.n_actions = n_actions
        self.n_prototypes = n_prototypes
        self._centroids = None
        self._Y_centroids = None
        self._X_all, self._Y_all = [], []

    def update(self, X_new, Y_new, seed_label=None):
        self._X_all.extend(list(X_new))
        self._Y_all.extend(list(Y_new))
        X_arr = np.array(self._X_all)
        Y_arr = np.array(self._Y_all)
        nc = min(self.n_prototypes, len(X_arr), max(1, len(X_arr) // 2))
        km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
        labels = km.fit_predict(X_arr)
        self._centroids = km.cluster_centers_
        self._Y_centroids = np.zeros((nc, self.n_actions))
        for i in range(nc):
            m = labels == i
            if m.sum() > 0:
                self._Y_centroids[i] = Y_arr[m].mean(axis=0)

    def predict(self, X_query):
        if self._centroids is None:
            return np.ones((len(X_query), self.n_actions)) / self.n_actions
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids)
        return self._Y_centroids[idxs]


class YAwareKMeans:
    def __init__(self, n_actions, n_prototypes=10, y_weight=5.0):
        self.n_actions = n_actions
        self.n_prototypes = n_prototypes
        self.y_weight = y_weight
        self._centroids_X = None
        self._Y_centroids = None
        self._X_all, self._Y_all = [], []

    def update(self, X_new, Y_new, seed_label=None):
        self._X_all.extend(list(X_new))
        self._Y_all.extend(list(Y_new))
        X_arr = np.array(self._X_all)
        Y_arr = np.array(self._Y_all)
        nc = min(self.n_prototypes, len(X_arr), max(1, len(X_arr) // 2))
        dX = np.std(X_arr, axis=0) + 1e-8
        dY = np.std(Y_arr, axis=0) + 1e-8
        joint = np.hstack([X_arr / dX, self.y_weight * Y_arr / dY])
        km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
        labels = km.fit_predict(joint)
        self._centroids_X = np.zeros((nc, X_arr.shape[1]))
        self._Y_centroids = np.zeros((nc, self.n_actions))
        for i in range(nc):
            m = labels == i
            if m.sum() > 0:
                self._centroids_X[i] = X_arr[m].mean(axis=0)
                self._Y_centroids[i] = Y_arr[m].mean(axis=0)

    def predict(self, X_query):
        if self._centroids_X is None:
            return np.ones((len(X_query), self.n_actions)) / self.n_actions
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids_X)
        return self._Y_centroids[idxs]


class NoMemoryBaseline:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self._counts = None

    def update(self, X_new, Y_new, seed_label=None):
        if self._counts is None:
            self._counts = np.zeros(self.n_actions)
        best_actions = np.argmax(Y_new, axis=1)
        for ba in best_actions:
            if ba < self.n_actions:
                self._counts[ba] += 1

    def predict(self, X_query):
        if self._counts is None or self._counts.sum() == 0:
            return np.ones((len(X_query), self.n_actions)) / self.n_actions
        probs = self._counts / self._counts.sum()
        return np.tile(probs, (len(X_query), 1))


def best_action_match(preds, Y_true):
    return np.mean(np.argmax(preds, axis=1) == np.argmax(Y_true, axis=1))


def quantize_actions(Y_soft, n_actions):
    scores = Y_soft.sum(axis=1)
    if n_actions == 3:
        return Y_soft.copy()
    bins = np.linspace(scores.min() - 1e-8, scores.max() + 1e-8, n_actions + 1)
    action_idx = np.digitize(scores, bins) - 1
    action_idx = np.clip(action_idx, 0, n_actions - 1)
    Y_hard = np.zeros((len(Y_soft), n_actions))
    Y_hard[np.arange(len(Y_soft)), action_idx] = 1.0
    return Y_hard


def run_objective_scaling(per_seed_train, X_test, Y_test_orig, log):
    log("")
    log("=" * 50)
    log("PHASE 6-B: OBJECTIVE SCALING (Multi-Action)")
    log("=" * 50)
    log(f"  Baseline: 3 actions, PA=0.585, NM=0.445, delta=+0.140")

    all_rows = []
    for n_actions in [3, 5, 10, 20]:
        log(f"\n  --- {n_actions} actions ---")
        random_baseline = 1.0 / n_actions

        Y_test = quantize_actions(Y_test_orig, n_actions)
        per_seed_quant = {}
        for seed, (X_tr, Y_tr, _, _) in per_seed_train.items():
            per_seed_quant[seed] = (X_tr, quantize_actions(Y_tr, n_actions))

        strategies = {
            f"kmeans": lambda: XOnlyKMeans(n_actions=n_actions, n_prototypes=10),
            f"y_aware_w5": lambda: YAwareKMeans(n_actions=n_actions, n_prototypes=10, y_weight=5.0),
            f"per_action": lambda: PerActionKMeans(n_actions=n_actions, n_prototypes_per_action=max(1, 10 // n_actions)),
            f"nomemory": lambda: NoMemoryBaseline(n_actions=n_actions),
        }

        row = {"n_actions": n_actions, "random_baseline": random_baseline}
        for sname, factory in strategies.items():
            strat = factory()
            for seed in sorted(per_seed_quant.keys()):
                X_s, Y_s = per_seed_quant[seed]
                strat.update(X_s, Y_s, seed_label=seed)
            match = best_action_match(strat.predict(X_test), Y_test)
            row[sname] = match
            log(f"    {sname:20s}: {match:.4f}")

        delta_pa_nm = row.get("per_action", 0) - row.get("nomemory", 0)
        delta_pa_km = row.get("per_action", 0) - row.get("kmeans", 0)
        row["delta_pa_nm"] = delta_pa_nm
        row["delta_pa_km"] = delta_pa_km
        log(f"    delta(PA-NM)={delta_pa_nm:+.4f}  delta(PA-KM)={delta_pa_km:+.4f}")
        all_rows.append(row)

    log(f"\n  Summary:")
    log(f"  {'Actions':>8s}  {'KMeans':>8s}  {'Y-aware':>8s}  {'PerAct':>8s}  {'NoMem':>8s}  {'PA-NM':>8s}  {'PA-KM':>8s}")
    for r in all_rows:
        log(f"  {r['n_actions']:8d}  {r.get('kmeans',0):8.4f}  {r.get('y_aware_w5',0):8.4f}  {r.get('per_action',0):8.4f}  {r.get('nomemory',0):8.4f}  {r.get('delta_pa_nm',0):+8.4f}  {r.get('delta_pa_km',0):+8.4f}")

    pa_nm_slope = all_rows[-1]["delta_pa_nm"] - all_rows[0]["delta_pa_nm"]
    log(f"\n  delta(PA-NM) slope (20act - 3act) = {pa_nm_slope:+.4f}")
    if pa_nm_slope > 0.02:
        log("  VERDICT: POSITIVE — Y-aware advantage GROWS with more actions")
    elif pa_nm_slope > -0.02:
        log("  VERDICT: STABLE — Y-aware advantage holds steady across actions")
    else:
        log("  VERDICT: NEGATIVE — Y-aware advantage SHRINKS with more actions")

    return all_rows


def run_noise_scaling(per_seed_train, X_test, Y_test, log):
    log("")
    log("=" * 50)
    log("PHASE 6-C: NOISE SCALING")
    log("=" * 50)

    X_train_all = np.concatenate([v[0] for v in per_seed_train.values()], axis=0)
    Y_train_all = np.concatenate([v[1] for v in per_seed_train.values()], axis=0)
    X_scale = np.std(X_train_all)
    n_actions = Y_train_all.shape[1]

    all_rows = []
    for noise_level in [0.0, 0.03, 0.10, 0.30, 1.00]:
        log(f"\n  --- noise={noise_level:.2f} x std(X) ---")

        strategies = {
            "kmeans": lambda: XOnlyKMeans(n_actions=n_actions, n_prototypes=10),
            "y_aware_w5": lambda: YAwareKMeans(n_actions=n_actions, n_prototypes=10, y_weight=5.0),
            "per_action": lambda: PerActionKMeans(n_actions=n_actions, n_prototypes_per_action=3),
            "nomemory": lambda: NoMemoryBaseline(n_actions=n_actions),
        }

        row = {"noise_level": noise_level}
        for sname, factory in strategies.items():
            best_matches = []
            for trial in range(3):
                rng = np.random.RandomState(42 + trial)
                strat = factory()
                for seed in sorted(per_seed_train.keys()):
                    X_s, Y_s, _, _ = per_seed_train[seed]
                    X_noisy = X_s + rng.randn(*X_s.shape) * X_scale * noise_level
                    strat.update(X_noisy, Y_s, seed_label=seed)
                match = best_action_match(strat.predict(X_test), Y_test)
                best_matches.append(match)
            avg_match = np.mean(best_matches)
            row[sname] = avg_match
            log(f"    {sname:20s}: {avg_match:.4f}")

        delta_pa_nm = row.get("per_action", 0) - row.get("nomemory", 0)
        row["delta_pa_nm"] = delta_pa_nm
        log(f"    delta(PA-NM)={delta_pa_nm:+.4f}")
        all_rows.append(row)

    pa_drop = all_rows[0].get("per_action", 0) - all_rows[-1].get("per_action", 0)
    km_drop = all_rows[0].get("kmeans", 0) - all_rows[-1].get("kmeans", 0)
    log(f"\n  PA drop (0 -> max noise): {pa_drop:.4f}")
    log(f"  KM drop (0 -> max noise): {km_drop:.4f}")
    if km_drop > 1e-6 and pa_drop / km_drop < 0.7:
        log(f"  VERDICT: POSITIVE — Y-aware is {km_drop/pa_drop:.1f}x more noise-robust than X-only")
    elif pa_drop < km_drop:
        log("  VERDICT: WEAK POSITIVE — Y-aware degrades slightly slower")
    else:
        log("  VERDICT: NO ADVANTAGE — Y-aware not more noise-robust")

    return all_rows


def main():
    run_id = f"c4_bc_{int(time.time())}"
    log_path = os.path.join(RESULTS_DIR, "run_log_objective_noise.txt")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()

    log("=" * 60)
    log("Phase 6-B/C: Objective Scaling + Noise Scaling")
    log("=" * 60)
    log(f"Run ID: {run_id}")
    log("")

    t0 = time.time()

    log("Step 1: Load counterfactual data")
    if not os.path.exists(CF_PATH):
        df = generate_counterfactual_table(
            StructuredVolatilityEnv, ENV_KWARGS,
            n_train=1200, n_val=200, n_test_id=200, n_ood=300,
            horizons=(1, 3, 5), seeds=range(N_SEEDS),
        )
        df.to_csv(CF_PATH, index=False)
    else:
        df = pd.read_csv(CF_PATH)
    log(f"  Loaded {len(df)} rows, {df.seed.nunique()} seeds")

    per_seed_train = {}
    for seed in range(N_SEEDS):
        train_df = df[(df["seed"] == seed) & (df["split"] == "train") & (df["horizon"] == 1)]
        X_tr, Y_tr, ba_tr = prepare_counterfactual_data(train_df, seed, ENV_KWARGS)
        if X_tr is not None and len(X_tr) > 0:
            per_seed_train[seed] = (X_tr, Y_tr, ba_tr, train_df)

    test_df = df[(df["seed"] == 0) & (df["split"] == "test_id") & (df["horizon"] == 1)]
    X_test, Y_test, ba_test = prepare_counterfactual_data(test_df, 0, ENV_KWARGS)
    log(f"  Train: {sum(len(v[0]) for v in per_seed_train.values())} samples")
    log(f"  Test: {len(X_test)} samples (seed=0, horizon=1)")

    obj_results = run_objective_scaling(per_seed_train, X_test, Y_test, log)
    noise_results = run_noise_scaling(per_seed_train, X_test, Y_test, log)

    df_obj = pd.DataFrame(obj_results)
    df_noise = pd.DataFrame(noise_results)
    df_obj.to_csv(os.path.join(RESULTS_DIR, "objective_scaling_results.csv"), index=False)
    df_noise.to_csv(os.path.join(RESULTS_DIR, "noise_scaling_results.csv"), index=False)

    elapsed = time.time() - t0
    log(f"\n{'='*60}")
    log(f"Phase 6-B/C complete. Total time: {elapsed:.0f}s")
    log("=" * 60)


if __name__ == "__main__":
    main()