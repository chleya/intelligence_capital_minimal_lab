"""
Stronger Stabilization Remedy: Readout-Level Interventions.
============================================================
Proof C showed anchored centroids improve +8.7% over naive, but the
improvement is modest because the root cause is NOT centroid position
but READOUT contamination: Y_centroids are averaged across seeds in
impure clusters (Purity=0.261).

This script tests three readout-level interventions that directly
attack the purity problem:

  1. Seed-Conditioned Readout: Y_centroids computed per-seed,
     query uses the Y from the nearest seed's contribution.
  2. Purity-Gated Readout: only use centroids with purity > threshold.
  3. Combined: Anchored centroids + seed-conditioned + purity-gated.

Plus two additional baselines:
  - Per-Seed Consolidated: separate KMeans per seed (no cross-seed mixing)
  - Distribution-Aware Readout: weight Y by seed proximity

Design:
  - 5 seeds, 5 update steps
  - 20 prototypes per strategy
  - Test on seed-0 test set
  - Metrics: best_action_match, centroid purity, per-step trajectory

Success criterion:
  - Any readout intervention > anchored br=0.7 (0.125) by > 20%
  - or any intervention > episodic (0.195)

Usage:
  cd intelligence_capital_minimal_lab
  python src/run_c2_stronger_stabilization.py
"""

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import prepare_counterfactual_data

RESULTS_DIR = "results/ic2c_stronger"
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
PURITY_GATE = 0.5

class ConsolidatedSummary:
    def __init__(self, n_prototypes=20):
        self.n_prototypes = n_prototypes
        self._centroids = None
        self._Y_centroids = None
        self._X_all = []
        self._Y_all = []
        self._seed_all = []

    def update(self, X_new, Y_new, seed_label=None):
        X_list = list(self._X_all) if len(self._X_all) > 0 else []
        Y_list = list(self._Y_all) if len(self._Y_all) > 0 else []
        seed_list = list(self._seed_all) if len(self._seed_all) > 0 else []
        for i in range(len(X_new)):
            X_list.append(X_new[i])
            Y_list.append(Y_new[i])
            seed_list.append(seed_label if seed_label is not None else -1)
        self._X_all = X_list
        self._Y_all = Y_list
        self._seed_all = seed_list
        X_arr = np.array(self._X_all)
        Y_arr = np.array(self._Y_all)
        seed_arr = np.array(self._seed_all)
        nc = min(self.n_prototypes, len(X_arr))
        km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
        labels = km.fit_predict(X_arr)
        self._centroids = km.cluster_centers_
        self._Y_centroids = np.zeros((nc, 3))
        self._cluster_seeds = []
        self._cluster_purity = []
        for i in range(nc):
            mask = labels == i
            if mask.sum() > 0:
                self._Y_centroids[i] = Y_arr[mask].mean(axis=0)
                cluster_seed_labels = seed_arr[mask]
                self._cluster_seeds.append(cluster_seed_labels)
                unique, counts = np.unique(cluster_seed_labels, return_counts=True)
                purity = counts.max() / counts.sum() if len(counts) > 0 else 0
                self._cluster_purity.append(purity)
            else:
                self._cluster_seeds.append(np.array([]))
                self._cluster_purity.append(0.0)

    def predict(self, X_query):
        if self._centroids is None:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids)
        return self._Y_centroids[idxs]

    @property
    def memory_size(self):
        return len(self._centroids) if self._centroids is not None else 0

class SeedConditionedReadout:
    """KMeans centroids as usual, but Y is computed per-seed.
    At prediction time, use the Y from the dominant seed in the
    nearest centroid's cluster."""
    def __init__(self, n_prototypes=20):
        self.n_prototypes = n_prototypes
        self._centroids = None
        self._Y_per_seed = {}
        self._dominant_seed = {}
        self._X_all = []
        self._Y_all = []
        self._seed_all = []

    def update(self, X_new, Y_new, seed_label=None):
        X_list = list(self._X_all) if len(self._X_all) > 0 else []
        Y_list = list(self._Y_all) if len(self._Y_all) > 0 else []
        seed_list = list(self._seed_all) if len(self._seed_all) > 0 else []
        for i in range(len(X_new)):
            X_list.append(X_new[i])
            Y_list.append(Y_new[i])
            seed_list.append(seed_label if seed_label is not None else -1)
        self._X_all = X_list
        self._Y_all = Y_list
        self._seed_all = seed_list
        X_arr = np.array(self._X_all)
        Y_arr = np.array(self._Y_all)
        seed_arr = np.array(self._seed_all)
        nc = min(self.n_prototypes, len(X_arr))
        km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
        labels = km.fit_predict(X_arr)
        self._centroids = km.cluster_centers_
        self._Y_per_seed = {}
        self._dominant_seed = {}
        for i in range(nc):
            mask = labels == i
            cluster_seeds = seed_arr[mask]
            cluster_Y = Y_arr[mask]
            unique_seeds = np.unique(cluster_seeds)
            per_seed_Y = {}
            for s in unique_seeds:
                s_mask = cluster_seeds == s
                if s_mask.sum() > 0:
                    per_seed_Y[int(s)] = cluster_Y[s_mask].mean(axis=0)
            self._Y_per_seed[i] = per_seed_Y
            if len(unique_seeds) > 0:
                counts = np.array([(cluster_seeds == s).sum() for s in unique_seeds])
                self._dominant_seed[i] = int(unique_seeds[np.argmax(counts)])
            else:
                self._dominant_seed[i] = -1

    def predict(self, X_query, query_seed=None):
        if self._centroids is None:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids)
        results = np.zeros((len(X_query), 3))
        for j, idx in enumerate(idxs):
            per_seed = self._Y_per_seed.get(idx, {})
            if query_seed is not None and query_seed in per_seed:
                results[j] = per_seed[query_seed]
            elif self._dominant_seed.get(idx, -1) in per_seed:
                results[j] = per_seed[self._dominant_seed[idx]]
            elif len(per_seed) > 0:
                results[j] = list(per_seed.values())[0]
            else:
                results[j] = RANDOM_BASELINE
        return results

    @property
    def memory_size(self):
        return len(self._centroids) if self._centroids is not None else 0

class PurityGatedReadout:
    """Only use centroids with purity > threshold. For impure centroids,
    fall back to the nearest pure centroid's Y."""
    def __init__(self, n_prototypes=20, purity_gate=0.5):
        self.n_prototypes = n_prototypes
        self.purity_gate = purity_gate
        self._centroids = None
        self._Y_centroids = None
        self._cluster_purity = []
        self._pure_mask = None
        self._X_all = []
        self._Y_all = []
        self._seed_all = []

    def update(self, X_new, Y_new, seed_label=None):
        X_list = list(self._X_all) if len(self._X_all) > 0 else []
        Y_list = list(self._Y_all) if len(self._Y_all) > 0 else []
        seed_list = list(self._seed_all) if len(self._seed_all) > 0 else []
        for i in range(len(X_new)):
            X_list.append(X_new[i])
            Y_list.append(Y_new[i])
            seed_list.append(seed_label if seed_label is not None else -1)
        self._X_all = X_list
        self._Y_all = Y_list
        self._seed_all = seed_list
        X_arr = np.array(self._X_all)
        Y_arr = np.array(self._Y_all)
        seed_arr = np.array(self._seed_all)
        nc = min(self.n_prototypes, len(X_arr))
        km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
        labels = km.fit_predict(X_arr)
        self._centroids = km.cluster_centers_
        self._Y_centroids = np.zeros((nc, 3))
        self._cluster_purity = []
        for i in range(nc):
            mask = labels == i
            if mask.sum() > 0:
                self._Y_centroids[i] = Y_arr[mask].mean(axis=0)
                cluster_seeds = seed_arr[mask]
                unique, counts = np.unique(cluster_seeds, return_counts=True)
                self._cluster_purity.append(counts.max() / counts.sum())
            else:
                self._cluster_purity.append(0.0)
        self._pure_mask = np.array(self._cluster_purity) >= self.purity_gate

    def predict(self, X_query):
        if self._centroids is None:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids)
        results = np.zeros((len(X_query), 3))
        pure_centroids = self._centroids[self._pure_mask]
        pure_Y = self._Y_centroids[self._pure_mask]
        for j, idx in enumerate(idxs):
            if self._pure_mask[idx]:
                results[j] = self._Y_centroids[idx]
            elif len(pure_centroids) > 0:
                nearest_pure, _ = pairwise_distances_argmin_min(
                    X_query[j:j+1], pure_centroids)
                results[j] = pure_Y[nearest_pure[0]]
            else:
                results[j] = RANDOM_BASELINE
        return results

    @property
    def memory_size(self):
        return len(self._centroids) if self._centroids is not None else 0

class CombinedAnchoredSeedPurity:
    """Anchored centroids + seed-conditioned readout + purity gating."""
    def __init__(self, n_prototypes=20, blend_ratio=0.7, purity_gate=0.4):
        self.n_prototypes = n_prototypes
        self.blend_ratio = blend_ratio
        self.purity_gate = purity_gate
        self._centroids = None
        self._anchors = None
        self._Y_per_seed = {}
        self._dominant_seed = {}
        self._cluster_purity = []
        self._pure_mask = None
        self._X_all = []
        self._Y_all = []
        self._seed_all = []
        self._step = 0

    def update(self, X_new, Y_new, seed_label=None):
        self._step += 1
        X_list = list(self._X_all) if len(self._X_all) > 0 else []
        Y_list = list(self._Y_all) if len(self._Y_all) > 0 else []
        seed_list = list(self._seed_all) if len(self._seed_all) > 0 else []
        for i in range(len(X_new)):
            X_list.append(X_new[i])
            Y_list.append(Y_new[i])
            seed_list.append(seed_label if seed_label is not None else -1)
        self._X_all = X_list
        self._Y_all = Y_list
        self._seed_all = seed_list
        X_arr = np.array(self._X_all)
        Y_arr = np.array(self._Y_all)
        seed_arr = np.array(self._seed_all)
        nc = min(self.n_prototypes, len(X_arr))
        km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
        labels = km.fit_predict(X_arr)
        raw_centroids = km.cluster_centers_
        if self._step == 1 or self._anchors is None:
            self._anchors = raw_centroids.copy()
            self._centroids = raw_centroids
        else:
            nc_new = min(nc, len(self._anchors))
            blended = (self.blend_ratio * self._anchors[:nc_new] +
                       (1 - self.blend_ratio) * raw_centroids[:nc_new])
            self._centroids = blended
        self._Y_per_seed = {}
        self._dominant_seed = {}
        self._cluster_purity = []
        for i in range(len(self._centroids)):
            if i < len(labels):
                mask = labels == i
                cluster_seeds = seed_arr[mask]
                cluster_Y = Y_arr[mask]
                unique_seeds = np.unique(cluster_seeds)
                per_seed_Y = {}
                for s in unique_seeds:
                    s_mask = cluster_seeds == s
                    if s_mask.sum() > 0:
                        per_seed_Y[int(s)] = cluster_Y[s_mask].mean(axis=0)
                self._Y_per_seed[i] = per_seed_Y
                if len(unique_seeds) > 0:
                    counts = np.array([(cluster_seeds == s).sum() for s in unique_seeds])
                    self._dominant_seed[i] = int(unique_seeds[np.argmax(counts)])
                else:
                    self._dominant_seed[i] = -1
                if len(cluster_seeds) > 0:
                    unique, counts = np.unique(cluster_seeds, return_counts=True)
                    self._cluster_purity.append(counts.max() / counts.sum())
                else:
                    self._cluster_purity.append(0.0)
            else:
                self._Y_per_seed[i] = {}
                self._dominant_seed[i] = -1
                self._cluster_purity.append(0.0)
        self._pure_mask = np.array(self._cluster_purity) >= self.purity_gate

    def predict(self, X_query, query_seed=None):
        if self._centroids is None:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids)
        results = np.zeros((len(X_query), 3))
        pure_centroids = self._centroids[self._pure_mask] if self._pure_mask.any() else self._centroids
        for j, idx in enumerate(idxs):
            if not self._pure_mask[idx] and self._pure_mask.any():
                nearest_pure, _ = pairwise_distances_argmin_min(
                    X_query[j:j+1], pure_centroids)
                idx = np.where(self._pure_mask)[0][nearest_pure[0]]
            per_seed = self._Y_per_seed.get(idx, {})
            if query_seed is not None and query_seed in per_seed:
                results[j] = per_seed[query_seed]
            elif self._dominant_seed.get(idx, -1) in per_seed:
                results[j] = per_seed[self._dominant_seed[idx]]
            elif len(per_seed) > 0:
                results[j] = list(per_seed.values())[0]
            else:
                results[j] = RANDOM_BASELINE
        return results

    @property
    def memory_size(self):
        return len(self._centroids) if self._centroids is not None else 0

class Seed0OnlyReadout:
    """Only use seed-0 data for both clustering and readout.
    Ignores all other seeds — no cross-seed contamination at all."""
    def __init__(self, n_prototypes=20):
        self.n_prototypes = n_prototypes
        self._centroids = None
        self._Y_centroids = None
        self._X_seed0 = []
        self._Y_seed0 = []

    def update(self, X_new, Y_new, seed_label=None):
        if seed_label == 0:
            self._X_seed0.extend(list(X_new))
            self._Y_seed0.extend(list(Y_new))
        X_arr = np.array(self._X_seed0)
        Y_arr = np.array(self._Y_seed0)
        if len(X_arr) < self.n_prototypes:
            self._centroids = None
            return
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

    @property
    def memory_size(self):
        return len(self._centroids) if self._centroids is not None else 0

class WeightedSeedReadout:
    """KMeans on all data, but Y_centroids weighted by seed proximity
    to query seed. Closer seeds get more weight."""
    def __init__(self, n_prototypes=20, temperature=1.0):
        self.n_prototypes = n_prototypes
        self.temperature = temperature
        self._centroids = None
        self._Y_per_seed = {}
        self._cluster_seeds_raw = {}
        self._X_all = []
        self._Y_all = []
        self._seed_all = []

    def update(self, X_new, Y_new, seed_label=None):
        X_list = list(self._X_all) if len(self._X_all) > 0 else []
        Y_list = list(self._Y_all) if len(self._Y_all) > 0 else []
        seed_list = list(self._seed_all) if len(self._seed_all) > 0 else []
        for i in range(len(X_new)):
            X_list.append(X_new[i])
            Y_list.append(Y_new[i])
            seed_list.append(seed_label if seed_label is not None else -1)
        self._X_all = X_list
        self._Y_all = Y_list
        self._seed_all = seed_list
        X_arr = np.array(self._X_all)
        Y_arr = np.array(self._Y_all)
        seed_arr = np.array(self._seed_all)
        nc = min(self.n_prototypes, len(X_arr))
        km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
        labels = km.fit_predict(X_arr)
        self._centroids = km.cluster_centers_
        self._Y_per_seed = {}
        self._cluster_seeds_raw = {}
        for i in range(nc):
            mask = labels == i
            cluster_seeds = seed_arr[mask]
            cluster_Y = Y_arr[mask]
            unique_seeds = np.unique(cluster_seeds)
            per_seed_Y = {}
            per_seed_count = {}
            for s in unique_seeds:
                s_mask = cluster_seeds == s
                if s_mask.sum() > 0:
                    per_seed_Y[int(s)] = cluster_Y[s_mask].mean(axis=0)
                    per_seed_count[int(s)] = s_mask.sum()
            self._Y_per_seed[i] = per_seed_Y
            self._cluster_seeds_raw[i] = per_seed_count

    def predict(self, X_query, query_seed=0):
        if self._centroids is None:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids)
        results = np.zeros((len(X_query), 3))
        for j, idx in enumerate(idxs):
            per_seed = self._Y_per_seed.get(idx, {})
            if len(per_seed) == 0:
                results[j] = RANDOM_BASELINE
                continue
            weights = {}
            for s, y in per_seed.items():
                dist = abs(s - query_seed)
                weights[s] = np.exp(-dist / self.temperature)
            total_w = sum(weights.values())
            if total_w > 0:
                weighted_Y = sum(weights[s] * per_seed[s] for s in per_seed) / total_w
                results[j] = weighted_Y
            else:
                results[j] = list(per_seed.values())[0]
        return results

    @property
    def memory_size(self):
        return len(self._centroids) if self._centroids is not None else 0

class PerSeedConsolidated:
    """Separate KMeans per seed — no cross-seed mixing at all."""
    def __init__(self, n_prototypes_per_seed=4):
        self.n_prototypes_per_seed = n_prototypes_per_seed
        self._per_seed = {}

    def update(self, X_new, Y_new, seed_label=None):
        if seed_label is None:
            return
        if seed_label not in self._per_seed:
            self._per_seed[seed_label] = {"X": [], "Y": []}
        self._per_seed[seed_label]["X"].extend(list(X_new))
        self._per_seed[seed_label]["Y"].extend(list(Y_new))

    def _refit(self):
        centroids_parts = []
        Y_parts = []
        for seed in sorted(self._per_seed.keys()):
            X_arr = np.array(self._per_seed[seed]["X"])
            Y_arr = np.array(self._per_seed[seed]["Y"])
            nc = min(self.n_prototypes_per_seed, len(X_arr))
            if nc < 1:
                continue
            km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
            labels = km.fit_predict(X_arr)
            for i in range(nc):
                mask = labels == i
                if mask.sum() > 0:
                    centroids_parts.append(km.cluster_centers_[i])
                    Y_parts.append(Y_arr[mask].mean(axis=0))
        if len(centroids_parts) == 0:
            return None, None
        return np.stack(centroids_parts), np.stack(Y_parts)

    def predict(self, X_query):
        centroids, Y_c = self._refit()
        if centroids is None:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE
        idxs, _ = pairwise_distances_argmin_min(X_query, centroids)
        return Y_c[idxs]

    @property
    def memory_size(self):
        total = 0
        for seed_data in self._per_seed.values():
            total += len(seed_data["X"])
        return total

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

    @property
    def memory_size(self):
        return len(self._X)

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

    @property
    def memory_size(self):
        return 3

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

    log("Stronger Stabilization Remedy: Readout-Level Interventions")
    log("=" * 60)

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
        "naive_consolidated": lambda: ConsolidatedSummary(n_prototypes=N_PROTOTYPES),
        "seed_conditioned": lambda: SeedConditionedReadout(n_prototypes=N_PROTOTYPES),
        "seed0_only": lambda: Seed0OnlyReadout(n_prototypes=N_PROTOTYPES),
        "weighted_seed_T1": lambda: WeightedSeedReadout(n_prototypes=N_PROTOTYPES, temperature=1.0),
        "weighted_seed_T05": lambda: WeightedSeedReadout(n_prototypes=N_PROTOTYPES, temperature=0.5),
        "purity_gated_0.5": lambda: PurityGatedReadout(n_prototypes=N_PROTOTYPES, purity_gate=0.5),
        "purity_gated_0.3": lambda: PurityGatedReadout(n_prototypes=N_PROTOTYPES, purity_gate=0.3),
        "combined_anchor_seed_purity": lambda: CombinedAnchoredSeedPurity(
            n_prototypes=N_PROTOTYPES, blend_ratio=0.7, purity_gate=0.4),
        "per_seed_consolidated": lambda: PerSeedConsolidated(n_prototypes_per_seed=4),
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

            if hasattr(strategy, 'predict') and 'query_seed' in strategy.predict.__code__.co_varnames:
                preds = strategy.predict(X_test, query_seed=0)
            else:
                preds = strategy.predict(X_test)
            match = best_action_match(preds, Y_test)

            purity_info = ""
            if hasattr(strategy, '_cluster_purity') and strategy._cluster_purity:
                mean_purity = np.mean(strategy._cluster_purity)
                n_pure = sum(1 for p in strategy._cluster_purity if p >= 0.5)
                purity_info = f", purity={mean_purity:.3f}, n_pure={n_pure}/{len(strategy._cluster_purity)}"

            all_records.append({"step": step + 1, "strategy": sname,
                                "best_action_match": match})
            log(f"  Step {step+1}: match={match:.4f}{purity_info}")

    df = pd.DataFrame(all_records)
    csv_path = os.path.join(RESULTS_DIR, "stronger_stabilization_results.csv")
    df.to_csv(csv_path, index=False)

    log(f"\n{'='*60}")
    log(f"RESULTS saved to {csv_path} ({len(df)} rows)")
    log(f"{'='*60}")

    table = df.pivot_table(index="step", columns="strategy", values="best_action_match", aggfunc="mean")
    log("\nBest Action Match Trajectory:")
    log(table.to_string())

    final_step = df[df["step"] == UPDATE_STEPS]
    log(f"\nFinal Step ({UPDATE_STEPS}) Comparison:")
    baseline_match = final_step[final_step["strategy"] == "naive_consolidated"]["best_action_match"].values[0]
    episodic_match = final_step[final_step["strategy"] == "episodic"]["best_action_match"].values[0]
    nomemory_match = final_step[final_step["strategy"] == "nomemory"]["best_action_match"].values[0]
    log(f"  naive_consolidated: {baseline_match:.4f} (baseline)")
    log(f"  episodic:           {episodic_match:.4f}")
    log(f"  nomemory:           {nomemory_match:.4f}")

    log(f"\nIntervention Improvements over naive:")
    for sname in strategies:
        if sname in ("naive_consolidated", "episodic", "nomemory"):
            continue
        val = final_step[final_step["strategy"] == sname]["best_action_match"].values
        if len(val) > 0:
            delta = val[0] - baseline_match
            pct = delta / baseline_match * 100 if baseline_match > 0 else 0
            vs_episodic = val[0] / episodic_match * 100 if episodic_match > 0 else 0
            log(f"  {sname:<35s} match={val[0]:.4f}  Δ={delta:+.4f} ({pct:+.1f}%)  vs_episodic={vs_episodic:.1f}%")

    log(f"\n{'='*60}")
    log("VERDICT:")
    best_intervention = None
    best_match = 0
    for sname in strategies:
        if sname in ("naive_consolidated", "episodic", "nomemory"):
            continue
        val = final_step[final_step["strategy"] == sname]["best_action_match"].values
        if len(val) > 0 and val[0] > best_match:
            best_match = val[0]
            best_intervention = sname

    if best_match > baseline_match * 1.2:
        log(f"  STRONG improvement: {best_intervention} = {best_match:.4f} > naive*1.2 = {baseline_match*1.2:.4f}")
        log(f"  Stabilization is SIGNIFICANTLY correctable at readout level.")
    elif best_match > baseline_match:
        log(f"  Modest improvement: {best_intervention} = {best_match:.4f} > naive = {baseline_match:.4f}")
        log(f"  Stabilization is partially correctable but readout alone is insufficient.")
    else:
        log(f"  No improvement: best intervention = {best_match:.4f} <= naive = {baseline_match:.4f}")
        log(f"  Readout contamination is NOT the rate-limiting factor.")

    if best_match >= episodic_match:
        log(f"  BREAKTHROUGH: {best_intervention} matches or exceeds episodic ({episodic_match:.4f})!")
    else:
        gap = episodic_match - best_match
        log(f"  Gap to episodic: {gap:.4f} ({gap/episodic_match*100:.1f}% of episodic)")

    log("\n" + "=" * 60)
    log("Stronger Stabilization Remedy complete.")
    log("=" * 60)

if __name__ == "__main__":
    main()
