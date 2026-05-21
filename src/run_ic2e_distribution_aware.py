"""IC-2e: Distribution-Aware Consolidation — Per-Seed Independent Prototypes.

The original IC-2c consolidated strategy crosses seeds: KMeans clusters ALL
accumulated data from seeds 0→1→2→3→4 together, creating centroids that
average across different distributions.

This experiment tests: if we keep SEPARATE prototypes per seed (independent
KMeans within each seed's data), does consolidation avoid the "bad debt"
problem?

Strategies:
  - NoMemory         (action-frequency baseline)
  - Episodic (k-NN)  (same as IC-2c/d)
  - CrossSeedConsolidated (original: KMeans on ALL accumulated data)
  - PerSeedConsolidated   (new: independent KMeans per seed, maintained separately)
"""

import os, sys, json, numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import prepare_counterfactual_data
from src.metrics import compute_best_action_match

ENV_KWARGS = dict(
    state_dim=2, mode_flip_prob=0.08, autonomous_drift=0.05,
    autonomous_noise=0.02, action_gain=0.70, action_noise=0.03,
    action_sign_flip=True, history_len=8, action_cost=0.20,
    state_dependent_gain=True, saturation_k=0.5,
)
SEEDS = list(range(5))
OUT_DIR = "results/ic2e"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "log.txt")

RANDOM_BASELINE = 1.0 / 3.0


def _log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


class NoMemoryBaseline:
    def __init__(self):
        self._counts = None
    def update(self, X, Y):
        if self._counts is None:
            self._counts = np.zeros(3)
        bas = np.argmax(Y, axis=1)
        for ba in bas:
            self._counts[ba] += 1
    def predict(self, X):
        if self._counts is None:
            return np.ones((len(X), 3)) * RANDOM_BASELINE
        return np.tile(self._counts / self._counts.sum(), (len(X), 1))
    @property
    def memory_size(self):
        return 3


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
    @property
    def memory_size(self):
        return len(self._X)


class CrossSeedConsolidated:
    """Original: KMeans on ALL accumulated data from all seeds."""
    def __init__(self, n_prototypes=20):
        self.n_prototypes = n_prototypes
        self._centroids = None
        self._Y_centroids = None
        self._X_all = []
        self._Y_all = []

    def update(self, X_new, Y_new):
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

    def predict(self, X_query):
        if self._centroids is None:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE
        from sklearn.metrics import pairwise_distances_argmin_min
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids)
        return self._Y_centroids[idxs]

    @property
    def memory_size(self):
        return len(self._centroids) if self._centroids is not None else 0


class PerSeedConsolidated:
    """New: Independent KMeans per seed, maintained as separate prototype sets.

    Each seed gets its own prototypes_per_seed centroids. Prediction picks the
    nearest prototype from ANY seed's set (union of all prototype pools).
    This prevents cross-distribution averaging in the clustering step.
    """
    def __init__(self, prototypes_per_seed=8, total_max=40):
        self.prototypes_per_seed = prototypes_per_seed
        self.total_max = total_max
        self._seed_centroids = []
        self._seed_Y = []
        self._seed_weights = []

    def update(self, X_new, Y_new):
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
        self._seed_weights.append(1.0)

        total_p = sum(len(c) for c in self._seed_centroids)
        while total_p > self.total_max and len(self._seed_centroids) > 1:
            self._seed_centroids.pop(0)
            self._seed_Y.pop(0)
            self._seed_weights.pop(0)
            total_p = sum(len(c) for c in self._seed_centroids)

    def predict(self, X_query):
        if len(self._seed_centroids) == 0:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE

        all_centroids = np.concatenate(self._seed_centroids, axis=0)
        all_Y = np.concatenate(self._seed_Y, axis=0)

        from sklearn.metrics import pairwise_distances_argmin_min
        idxs, _ = pairwise_distances_argmin_min(X_query, all_centroids)
        return all_Y[idxs]

    @property
    def memory_size(self):
        return sum(len(c) for c in self._seed_centroids)


def best_action_match(preds, ba_true):
    return float(np.mean(np.argmax(preds, axis=1) == ba_true))


def main():
    _log("=" * 60)
    _log("IC-2e: Distribution-Aware Consolidation")
    _log("=" * 60)

    cf_df = pd.read_csv("results/counterfactual_table.csv")
    per_seed_train = {}
    for seed in SEEDS:
        train_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "train") & (cf_df["horizon"] == 1)]
        X, Y, ba = prepare_counterfactual_data(train_df, seed, ENV_KWARGS)
        if X is not None and len(X) > 0:
            per_seed_train[seed] = (X, Y, ba)
    _log(f"Per-seed train sizes: {[(s, len(v[0])) for s, v in per_seed_train.items()]}")

    test_df = cf_df[(cf_df["seed"] == 0) & (cf_df["split"] == "test_id") & (cf_df["horizon"] == 1)]
    X_test, Y_test, ba_test = prepare_counterfactual_data(test_df, 0, ENV_KWARGS)
    _log(f"Test samples: {len(X_test)}")

    strategies = {
        "NoMemory":             NoMemoryBaseline(),
        "Episodic-kNN":         EpisodicKNN(max_buffer=200, k=5),
        "CrossSeedConsolidated": CrossSeedConsolidated(n_prototypes=20),
        "PerSeedConsolidated":  PerSeedConsolidated(prototypes_per_seed=8, total_max=40),
    }

    all_records = []

    for step in range(len(SEEDS)):
        seed = step
        if seed not in per_seed_train:
            continue
        X_step, Y_step, _ = per_seed_train[seed]
        _log(f"\nStep {step+1}: adding seed={seed} ({len(X_step)} samples)")

        for name, strat in strategies.items():
            strat.update(X_step, Y_step)
            preds = strat.predict(X_test)
            match = best_action_match(preds, ba_test)
            all_records.append({
                "step": step + 1, "strategy": name,
                "best_action_match": round(match, 4),
                "memory_size": strat.memory_size,
            })
            _log(f"  {name:<25} match={match:.4f}  mem={strat.memory_size}")

    df = pd.DataFrame(all_records)
    df.to_csv(f"{OUT_DIR}/ic2e_results.csv", index=False)
    _log(f"\nSaved to {OUT_DIR}/ic2e_results.csv")

    table = df.pivot_table(index="step", columns="strategy", values="best_action_match", aggfunc="mean")
    _log("\nBest Action Match Trajectory:")
    _log(table.to_string())

    final = df[df["step"] == len(SEEDS)]
    _log("\nFinal Step Comparison:")
    for _, row in final.iterrows():
        _log(f"  {row['strategy']:<25} match={row['best_action_match']:.4f}")

    cross_final = final[final["strategy"] == "CrossSeedConsolidated"]["best_action_match"].values[0]
    per_final = final[final["strategy"] == "PerSeedConsolidated"]["best_action_match"].values[0]
    epis_final = final[final["strategy"] == "Episodic-kNN"]["best_action_match"].values[0]
    nm_final = final[final["strategy"] == "NoMemory"]["best_action_match"].values[0]

    _log(f"\n--- IC-2e Results ---")
    _log(f"   NoMemory:               {nm_final:.4f}")
    _log(f"   Episodic-kNN:           {epis_final:.4f}")
    _log(f"   CrossSeedConsolidated:  {cross_final:.4f}")
    _log(f"   PerSeedConsolidated:    {per_final:.4f}")
    _log(f"   PerSeed vs CrossSeed:   {per_final - cross_final:+.4f}")

    summary = {
        "experiment": "IC-2e Distribution-Aware Consolidation",
        "final_nomemory": float(nm_final),
        "final_episodic_knn": float(epis_final),
        "final_cross_seed": float(cross_final),
        "final_per_seed": float(per_final),
        "per_vs_cross_delta": float(per_final - cross_final),
    }
    with open(f"{OUT_DIR}/ic2e_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _log("\nDone.")


if __name__ == "__main__":
    main()