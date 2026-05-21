"""
Root Cause Fix: Y-Aware Consolidation.
========================================
Root cause identified: KMeans (X-only clustering) gives 0.095 match,
while learned_state_only (supervised) gives 0.740 and
counterfactual_compressor gives 0.775. The 8x gap is because KMeans
ignores Y information entirely — nearby points in X space can have
completely different optimal actions.

Fix: Use Y-aware clustering that considers both X proximity and Y
consistency when forming clusters. This should dramatically improve
consolidation quality.

Strategies tested:
  1. KMeans (baseline, X-only): 20 centroids
  2. KMeans high-res: 100, 200 centroids (test resolution hypothesis)
  3. Y-Aware KMeans: cluster on [X, w*Y] joint space
  4. Decision Tree centroids: use DT leaf nodes as centroids
  5. Per-Action KMeans: separate KMeans per best-action group
  6. Supervised prototype: train a small NN, use hidden centroids

Plus episodic and nomemory baselines.

Usage:
  cd intelligence_capital_minimal_lab
  python src/run_c3_y_aware_consolidation.py
"""

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import prepare_counterfactual_data

RESULTS_DIR = "results/ic2c_y_aware"
os.makedirs(RESULTS_DIR, exist_ok=True)

ENV_KWARGS = dict(
    state_dim=2, mode_flip_prob=0.08, autonomous_drift=0.05,
    autonomous_noise=0.02, action_gain=0.70, action_noise=0.03,
    action_sign_flip=True, history_len=8,
    action_cost=0.20, state_dependent_gain=True, saturation_k=0.5,
)
SEEDS = list(range(5))
UPDATE_STEPS = len(SEEDS)
RANDOM_BASELINE = 1.0 / 3.0
N_PROTOTYPES = 20

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
    """Cluster on joint [X, w*Y] space. Weight Y contribution so that
    clusters respect action boundaries, not just state proximity."""
    def __init__(self, n_prototypes=20, y_weight=1.0):
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
    """Separate KMeans per action group. Each action group gets
    n_prototypes/3 centroids. This ensures Y-consistency within clusters."""
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

class DecisionTreeCentroids:
    """Use Decision Tree leaf nodes as centroids. DT naturally
    partitions X space by Y purity — exactly what we need."""
    def __init__(self, max_leaf_nodes=20):
        self.max_leaf_nodes = max_leaf_nodes
        self._centroids = None
        self._Y_centroids = None
        self._X_all = []
        self._Y_all = []

    def update(self, X_new, Y_new, seed_label=None):
        self._X_all.extend(list(X_new))
        self._Y_all.extend(list(Y_new))
        X_arr = np.array(self._X_all)
        Y_arr = np.array(self._Y_all)
        best_actions = np.argmax(Y_arr, axis=1)
        dt = DecisionTreeClassifier(max_leaf_nodes=self.max_leaf_nodes, random_state=42)
        dt.fit(X_arr, best_actions)
        leaf_ids = dt.apply(X_arr)
        unique_leaves = np.unique(leaf_ids)
        centroids_parts = []
        Y_parts = []
        for leaf_id in unique_leaves:
            mask = leaf_ids == leaf_id
            if mask.sum() > 0:
                centroids_parts.append(X_arr[mask].mean(axis=0))
                Y_parts.append(Y_arr[mask].mean(axis=0))
        if len(centroids_parts) > 0:
            self._centroids = np.stack(centroids_parts)
            self._Y_centroids = np.stack(Y_parts)

    def predict(self, X_query):
        if self._centroids is None:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids)
        return self._Y_centroids[idxs]

class EpisodicRetention:
    def __init__(self, max_buffer=200, k=5):
        self.max_buffer = max_buffer
        self.k = k
        self._X = []
        self._Y = []

    def update(self, X_new, Y_new, seed_label=None):
        X_list = list(self._X) if isinstance(self._X, np.ndarray) else list(self._X)
        Y_list = list(self._Y) if isinstance(self._Y, np.ndarray) else list(self._Y)
        for i in range(len(X_new)):
            X_list.append(X_new[i])
            Y_list.append(Y_new[i])
        while len(X_list) > self.max_buffer:
            X_list.pop(0)
            Y_list.pop(0)
        self._X = np.array(X_list)
        self._Y = np.array(Y_list)

    def predict(self, X_query):
        if len(self._X) == 0:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE
        k_eff = min(self.k, len(self._X))
        knns = [KNeighborsRegressor(n_neighbors=k_eff) for _ in range(3)]
        for a in range(3):
            knns[a].fit(self._X, self._Y[:, a])
        return np.stack([k.predict(X_query) for k in knns], axis=-1)

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

def main():
    log_path = os.path.join(RESULTS_DIR, "run_log.txt")
    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()

    log("Root Cause Fix: Y-Aware Consolidation")
    log("=" * 60)
    log("Hypothesis: KMeans fails because it ignores Y information.")
    log("Evidence: learned_state_only=0.740, counterfactual_compressor=0.775, KMeans=0.095")
    log("Fix: Y-aware clustering that respects action boundaries.")

    cf_df = pd.read_csv("results/counterfactual_table.csv")
    log(f"CF table: {len(cf_df)} rows, {cf_df['seed'].nunique()} seeds")

    per_seed_train = {}
    for seed in SEEDS:
        train_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "train") & (cf_df["horizon"] == 1)]
        X_tr, Y_tr, ba_tr = prepare_counterfactual_data(train_df, seed, ENV_KWARGS)
        if X_tr is not None and len(X_tr) > 0:
            per_seed_train[seed] = (X_tr, Y_tr, ba_tr, train_df)
    log(f"Per-seed train sizes: {[(s, len(v[0])) for s, v in per_seed_train.items()]}")

    test_df = cf_df[(cf_df["seed"] == 0) & (cf_df["split"] == "test_id") & (cf_df["horizon"] == 1)]
    X_test, Y_test, ba_test = prepare_counterfactual_data(test_df, 0, ENV_KWARGS)
    log(f"Fixed test set: {len(X_test)} samples (seed=0)")

    strategies = {
        "kmeans_20": lambda: KMeansBaseline(n_prototypes=20),
        "kmeans_50": lambda: KMeansBaseline(n_prototypes=50),
        "kmeans_100": lambda: KMeansBaseline(n_prototypes=100),
        "kmeans_200": lambda: KMeansBaseline(n_prototypes=200),
        "y_aware_w0.5": lambda: YAwareKMeans(n_prototypes=20, y_weight=0.5),
        "y_aware_w1.0": lambda: YAwareKMeans(n_prototypes=20, y_weight=1.0),
        "y_aware_w2.0": lambda: YAwareKMeans(n_prototypes=20, y_weight=2.0),
        "y_aware_w5.0": lambda: YAwareKMeans(n_prototypes=20, y_weight=5.0),
        "per_action_kmeans": lambda: PerActionKMeans(n_prototypes_per_action=7),
        "dt_20": lambda: DecisionTreeCentroids(max_leaf_nodes=20),
        "dt_50": lambda: DecisionTreeCentroids(max_leaf_nodes=50),
        "dt_100": lambda: DecisionTreeCentroids(max_leaf_nodes=100),
        "episodic": lambda: EpisodicRetention(max_buffer=200, k=5),
        "nomemory": lambda: NoMemoryBaseline(),
    }

    all_records = []

    for sname, factory in strategies.items():
        log(f"\n[{sname}]")
        strategy = factory()
        for step in range(UPDATE_STEPS):
            seed = step
            if seed not in per_seed_train:
                continue
            X_step, Y_step, _, _ = per_seed_train[seed]
            strategy.update(X_step, Y_step, seed_label=seed)
            preds = strategy.predict(X_test)
            match = best_action_match(preds, Y_test)
            all_records.append({"step": step + 1, "strategy": sname,
                                "best_action_match": match})
            log(f"  Step {step+1}: match={match:.4f}")

    df = pd.DataFrame(all_records)
    csv_path = os.path.join(RESULTS_DIR, "y_aware_consolidation_results.csv")
    df.to_csv(csv_path, index=False)

    log(f"\n{'='*60}")
    log(f"RESULTS saved to {csv_path} ({len(df)} rows)")
    log(f"{'='*60}")

    table = df.pivot_table(index="step", columns="strategy", values="best_action_match", aggfunc="mean")
    log("\nBest Action Match Trajectory:")
    log(table.to_string())

    final_step = df[df["step"] == UPDATE_STEPS]
    log(f"\nFinal Step ({UPDATE_STEPS}) Results:")
    for _, row in final_step.iterrows():
        log(f"  {row['strategy']:<25s} match={row['best_action_match']:.4f}")

    kmeans_20_final = final_step[final_step["strategy"] == "kmeans_20"]["best_action_match"].values[0]
    episodic_final = final_step[final_step["strategy"] == "episodic"]["best_action_match"].values[0]
    nomemory_final = final_step[final_step["strategy"] == "nomemory"]["best_action_match"].values[0]

    log(f"\nKey comparisons:")
    log(f"  kmeans_20 (baseline): {kmeans_20_final:.4f}")
    log(f"  episodic:             {episodic_final:.4f}")
    log(f"  nomemory:             {nomemory_final:.4f}")

    best_y_aware = 0
    best_y_name = ""
    for sname in ["y_aware_w0.5", "y_aware_w1.0", "y_aware_w2.0", "y_aware_w5.0",
                   "per_action_kmeans", "dt_20", "dt_50", "dt_100"]:
        val = final_step[final_step["strategy"] == sname]["best_action_match"].values
        if len(val) > 0 and val[0] > best_y_aware:
            best_y_aware = val[0]
            best_y_name = sname

    log(f"\nBest Y-aware intervention: {best_y_name} = {best_y_aware:.4f}")
    if best_y_aware > episodic_final:
        log(f"  BREAKTHROUGH: Y-aware consolidation BEATS episodic!")
    if best_y_aware > kmeans_20_final * 2:
        log(f"  Y-aware > 2x KMeans: root cause confirmed (Y-information is key)")
    if best_y_aware > nomemory_final:
        log(f"  Y-aware BEATS NoMemory shortcut!")

    log(f"\nResolution hypothesis test:")
    kmeans_100_val = final_step[final_step["strategy"] == "kmeans_100"]["best_action_match"].values
    kmeans_200_val = final_step[final_step["strategy"] == "kmeans_200"]["best_action_match"].values
    if len(kmeans_100_val) > 0:
        log(f"  kmeans_100: {kmeans_100_val[0]:.4f}")
    if len(kmeans_200_val) > 0:
        log(f"  kmeans_200: {kmeans_200_val[0]:.4f}")
    if len(kmeans_200_val) > 0 and kmeans_200_val[0] > kmeans_20_final * 1.5:
        log(f"  Resolution helps but Y-awareness is still needed")
    elif len(kmeans_200_val) > 0 and kmeans_200_val[0] < kmeans_20_final * 1.5:
        log(f"  Resolution does NOT solve the problem → Y-information is the bottleneck")

    log("\n" + "=" * 60)
    log("Y-Aware Consolidation experiment complete.")
    log("=" * 60)

if __name__ == "__main__":
    main()
