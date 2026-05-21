"""IC-2f: Structural Fidelity Metrics — Measuring Consolidation Information Loss.

Defines and measures three structural fidelity metrics during continual
consolidation:

  1. Centroid Drift Rate: how much prototype centroids move per update step
     (L2 norm of centroid displacement / update)

  2. Coverage Preservation: fraction of original prototype space still covered
     after new data is merged in (via convex hull or pairwise distance ratio)

  3. Information Loss Rate: how much old information is overwritten — measured
     as the change in nearest-prototype assignments for a fixed query set
     (fraction of queries whose nearest prototype changed, weighted by distance)

These metrics are measured for:
  - CrossSeedConsolidated (original IC-2c, known to create bad debt)
  - PerSeedConsolidated (IC-2e, distribution-aware, expected more stable)
  - Episodic-kNN (reference: exact retention, should have zero drift)
"""

import os, sys, json, time, numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import prepare_counterfactual_data

ENV_KWARGS = dict(
    state_dim=2, mode_flip_prob=0.08, autonomous_drift=0.05,
    autonomous_noise=0.02, action_gain=0.70, action_noise=0.03,
    action_sign_flip=True, history_len=8, action_cost=0.20,
    state_dependent_gain=True, saturation_k=0.5,
)
SEEDS = list(range(5))
OUT_DIR = "results/ic2f"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "log.txt")

RANDOM_BASELINE = 1.0 / 3.0


def _log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


class EpisodicKNN:
    def __init__(self, max_buffer=200, k=5):
        self.max_buffer = max_buffer
        self.k = k
        self._X = []
        self._Y = []
    def update(self, X_new, Y_new):
        for i in range(len(X_new)):
            self._X.append(X_new[i])
            self._Y.append(Y_new[i])
        while len(self._X) > self.max_buffer:
            self._X.pop(0)
            self._Y.pop(0)
    def predict(self, X_query):
        if len(self._X) == 0:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE
        X_buf, Y_buf = np.array(self._X), np.array(self._Y)
        k_eff = min(self.k, len(X_buf))
        preds = []
        for a in range(3):
            knn = KNeighborsRegressor(n_neighbors=k_eff)
            knn.fit(X_buf, Y_buf[:, a])
            preds.append(knn.predict(X_query))
        return np.stack(preds, axis=-1)
    def get_all_data(self):
        return np.array(self._X) if len(self._X) > 0 else np.zeros((0, 24))
    def get_all_Y(self):
        return np.array(self._Y) if len(self._Y) > 0 else np.zeros((0, 3))


class CrossSeedConsolidated:
    def __init__(self, n_prototypes=20):
        self.n_prototypes = n_prototypes
        self._centroids = None
        self._Y_centroids = None
        self._X_all = []
        self._Y_all = []

    def update(self, X_new, Y_new):
        old_centroids = self._centroids.copy() if self._centroids is not None else None

        if len(self._X_all) == 0:
            self._X_all = list(X_new)
            self._Y_all = list(Y_new)
        else:
            for i in range(len(X_new)):
                self._X_all.append(X_new[i])
                self._Y_all.append(Y_new[i])

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

        return old_centroids

    def predict(self, X_query):
        if self._centroids is None:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE
        from sklearn.metrics import pairwise_distances_argmin_min
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids)
        return self._Y_centroids[idxs]

    def get_centroids(self):
        return self._centroids if self._centroids is not None else np.zeros((0, 24))


class PerSeedConsolidated:
    def __init__(self, prototypes_per_seed=8, total_max=40):
        self.prototypes_per_seed = prototypes_per_seed
        self.total_max = total_max
        self._seed_centroids = []
        self._seed_Y = []

    def update(self, X_new, Y_new):
        old_all = np.concatenate(self._seed_centroids, axis=0) if self._seed_centroids else None

        X_arr = np.array(X_new)
        Y_arr = np.array(Y_new)
        nc = min(self.prototypes_per_seed, len(X_arr))
        km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
        labels = km.fit_predict(X_arr)
        centroids = km.cluster_centers_
        Y_centroids = np.zeros((nc, 3))
        for i in range(nc):
            mask = labels == i
            if mask.sum() > 0:
                Y_centroids[i] = Y_arr[mask].mean(axis=0)

        self._seed_centroids.append(centroids)
        self._seed_Y.append(Y_centroids)

        total_p = sum(len(c) for c in self._seed_centroids)
        while total_p > self.total_max and len(self._seed_centroids) > 1:
            self._seed_centroids.pop(0)
            self._seed_Y.pop(0)
            total_p = sum(len(c) for c in self._seed_centroids)

        return old_all

    def predict(self, X_query):
        if len(self._seed_centroids) == 0:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE
        all_centroids = np.concatenate(self._seed_centroids, axis=0)
        all_Y = np.concatenate(self._seed_Y, axis=0)
        from sklearn.metrics import pairwise_distances_argmin_min
        idxs, _ = pairwise_distances_argmin_min(X_query, all_centroids)
        return all_Y[idxs]

    def get_centroids(self):
        if not self._seed_centroids:
            return np.zeros((0, 24))
        return np.concatenate(self._seed_centroids, axis=0)


def compute_centroid_drift(old_centroids, new_centroids):
    """L2 drift: average L2 displacement of matched centroids."""
    if old_centroids is None or len(old_centroids) == 0:
        return 0.0
    if len(new_centroids) == 0:
        return 0.0

    from sklearn.metrics import pairwise_distances
    from scipy.optimize import linear_sum_assignment

    dists = pairwise_distances(new_centroids, old_centroids)
    if dists.shape[0] > dists.shape[1]:
        dists = dists.T
        row_ind, col_ind = linear_sum_assignment(dists)
        drift = dists[row_ind, col_ind].mean()
    else:
        row_ind, col_ind = linear_sum_assignment(dists)
        drift = dists[row_ind, col_ind].mean()

    return float(drift)


def compute_coverage_preservation(old_centroids, new_centroids):
    """Fraction of old centroid space covered by new centroids.

    Measured as the ratio of: min pairwise distance from old→new centroids
    vs max pairwise distance within old centroids.
    Higher = new centroids cover old space well.
    """
    if old_centroids is None or len(old_centroids) < 2:
        return 1.0
    if len(new_centroids) < 1:
        return 0.0

    from sklearn.metrics import pairwise_distances

    old_max_dist = pairwise_distances(old_centroids).max()

    old_to_new = pairwise_distances(old_centroids, new_centroids)
    nearest_dist = old_to_new.min(axis=1).mean()

    if old_max_dist < 1e-8:
        return 1.0

    coverage = 1.0 - (nearest_dist / old_max_dist)
    return float(max(0.0, min(1.0, coverage)))


def compute_routing_disruption(old_centroids, old_Y, new_centroids, new_Y, X_query):
    """Fraction of query samples whose nearest prototype class changed.

    Measures structural information loss: how many queries would get different
    action predictions after consolidation update.
    """
    if old_centroids is None or len(old_centroids) == 0:
        return 0.0
    if len(new_centroids) == 0:
        return 1.0

    from sklearn.metrics import pairwise_distances_argmin_min

    old_idx, _ = pairwise_distances_argmin_min(X_query, old_centroids)
    new_idx, _ = pairwise_distances_argmin_min(X_query, new_centroids)

    old_best = np.argmax(old_Y[old_idx], axis=1)
    new_best = np.argmax(new_Y[new_idx], axis=1)

    disruption = np.mean(old_best != new_best)
    return float(disruption)


def compute_routing_disruption_episodic(old_X, old_Y, X_query):
    """For episodic: fraction of queries whose nearest neighbor class changed."""
    if len(old_X) < 2:
        return 0.0
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(old_X)
    _, idx = nn.kneighbors(X_query)
    return float(np.mean(np.argmax(old_Y[idx[:, 0]], axis=1) != np.argmax(old_Y[idx[:, 0]], axis=1)))


def main():
    _log("=" * 60)
    _log("IC-2f: Structural Fidelity Metrics")
    _log("=" * 60)

    cf_df = pd.read_csv("results/counterfactual_table.csv")
    per_seed_train = {}
    for seed in SEEDS:
        train_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "train") & (cf_df["horizon"] == 1)]
        X, Y, ba = prepare_counterfactual_data(train_df, seed, ENV_KWARGS)
        if X is not None and len(X) > 0:
            per_seed_train[seed] = (X, Y, ba)

    test_df = cf_df[(cf_df["seed"] == 0) & (cf_df["split"] == "test_id") & (cf_df["horizon"] == 1)]
    X_test, Y_test, ba_test = prepare_counterfactual_data(test_df, 0, ENV_KWARGS)
    _log(f"Test samples: {len(X_test)}")

    strategies = {
        "Episodic-kNN":          EpisodicKNN(max_buffer=200, k=5),
        "CrossSeedConsolidated": CrossSeedConsolidated(n_prototypes=20),
        "PerSeedConsolidated":   PerSeedConsolidated(prototypes_per_seed=8, total_max=40),
    }

    all_records = []
    prev_state = {}

    for step in range(len(SEEDS)):
        seed = step
        if seed not in per_seed_train:
            continue
        X_step, Y_step, _ = per_seed_train[seed]
        _log(f"\nStep {step+1}: adding seed={seed} ({len(X_step)} samples)")

        for name, strat in strategies.items():
            old_centroids = prev_state.get(name + "_centroids", None)
            old_Y = prev_state.get(name + "_Y", None)
            old_X = prev_state.get(name + "_X", None)

            result = strat.update(X_step, Y_step)

            if name == "Episodic-kNN":
                new_centroids = strat.get_all_data()
                new_Y = strat.get_all_Y()
            else:
                result_centroids = result
                new_centroids = strat.get_centroids()
                if name == "CrossSeedConsolidated":
                    new_Y = strat._Y_centroids if strat._Y_centroids is not None else np.zeros((0, 3))
                    old_centroids = result_centroids
                else:
                    all_Y = np.concatenate(strat._seed_Y, axis=0) if strat._seed_Y else np.zeros((0, 3))
                    new_Y = all_Y
                    old_centroids = result

            drift = compute_centroid_drift(old_centroids, new_centroids)
            coverage = compute_coverage_preservation(old_centroids, new_centroids)

            if old_centroids is not None and old_Y is not None and len(old_centroids) > 0:
                disruption = compute_routing_disruption(
                    old_centroids, old_Y, new_centroids, new_Y, X_test)
            else:
                disruption = 0.0

            preds = strat.predict(X_test)
            match = float(np.mean(np.argmax(preds, axis=1) == ba_test))

            all_records.append({
                "step": step + 1, "strategy": name,
                "best_action_match": round(match, 4),
                "centroid_drift": round(drift, 4),
                "coverage_preservation": round(coverage, 4),
                "routing_disruption": round(disruption, 4),
                "n_centroids": len(new_centroids),
            })

            prev_state[name + "_centroids"] = new_centroids
            prev_state[name + "_Y"] = new_Y

            _log(f"  {name:<25} match={match:.4f}  drift={drift:.4f}  "
                 f"coverage={coverage:.4f}  disrupt={disruption:.4f}  n_cent={len(new_centroids)}")

    df = pd.DataFrame(all_records)
    df.to_csv(f"{OUT_DIR}/ic2f_fidelity_metrics.csv", index=False)
    _log(f"\nSaved to {OUT_DIR}/ic2f_fidelity_metrics.csv")

    for name in ["Episodic-kNN", "CrossSeedConsolidated", "PerSeedConsolidated"]:
        sub = df[df["strategy"] == name]
        _log(f"\n--- {name} Fidelity Summary ---")
        _log(f"  Avg centroid_drift:        {sub['centroid_drift'].mean():.4f}")
        _log(f"  Avg coverage_preservation:  {sub['coverage_preservation'].mean():.4f}")
        _log(f"  Avg routing_disruption:     {sub['routing_disruption'].mean():.4f}")
        _log(f"  Final best_action_match:    {sub['best_action_match'].iloc[-1]:.4f}")

    _log("\nDone.")


if __name__ == "__main__":
    main()