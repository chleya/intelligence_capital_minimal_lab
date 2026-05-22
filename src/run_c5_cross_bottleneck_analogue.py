"""
Phase 7 3.3A: Cross-Bottleneck Joint — Counterfactual Analogue.
=================================================================
Tests whether Y-aware stabilization (Per-Action KMeans) produces
centroids that are more robust to "organizational" perturbations
than X-only consolidation.

Analogue mapping:
  X-only centroids         → no stabilization
  Per-Action centroids     → stabilized structure
  Additive noise on centroids  → random steering intervention
  Directional shift        → structured steering intervention
  Centroid dropout         → gate failure / routing error

Hypotheses:
  H7.2: Y-aware reduces perturbation sensitivity
  H7.3: Y-aware advantage grows with perturbation level
  H7.4: Joint advantage > additive at high perturbation

Usage:
  cd F:\intelligence_capital_minimal_lab
  python src/run_c5_cross_bottleneck_analogue.py
"""

import os, sys, time, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import prepare_counterfactual_data
from src.counterfactual_table import generate_counterfactual_table
from src.env_structured_volatility import StructuredVolatilityEnv

RESULTS_DIR = "results/ic2c_cross_bottleneck"
os.makedirs(RESULTS_DIR, exist_ok=True)

ENV_KWARGS = dict(
    state_dim=2, mode_flip_prob=0.08, autonomous_drift=0.05,
    autonomous_noise=0.02, action_gain=0.70, action_noise=0.03,
    action_sign_flip=True, history_len=8,
    action_cost=0.20, state_dependent_gain=True, saturation_k=0.5,
)
RANDOM_BASELINE = 1.0 / 3.0
N_SEEDS = 5
N_PROTOTYPES = 20
CF_PATH = "results/counterfactual_table.csv"


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

    def get_centroids(self):
        return self._centroids, self._Y_centroids


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

    def get_centroids(self):
        return self._centroids_X, self._Y_centroids


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

    def get_centroids(self):
        return self._centroids, self._Y_centroids


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

    def get_centroids(self):
        return None, None


def best_action_match(preds, Y_true):
    return np.mean(np.argmax(preds, axis=1) == np.argmax(Y_true, axis=1))


def perturb_and_evaluate(strategy, X_test, Y_test, perturbation_type, levels):
    centroids_X, centroids_Y = strategy.get_centroids()
    if centroids_X is None:
        base_match = best_action_match(strategy.predict(X_test), Y_test)
        return {f"{perturbation_type}_{lvl}": base_match for lvl in levels}

    X_orig = centroids_X.copy()
    n_c, d = X_orig.shape
    X_scale = np.std(X_orig)
    results = {}

    for lvl in levels:
        X_perturbed = X_orig.copy()
        if perturbation_type == "additive_noise":
            X_perturbed += np.random.randn(n_c, d) * X_scale * lvl
        elif perturbation_type == "directional_shift":
            direction = np.random.randn(d)
            direction /= np.linalg.norm(direction) + 1e-8
            X_perturbed += direction * X_scale * lvl
        elif perturbation_type == "centroid_dropout":
            keep_mask = np.random.random(n_c) > lvl
            if keep_mask.sum() == 0:
                keep_mask[0] = True
            X_perturbed = X_orig[keep_mask]
            centroids_Y_pert = centroids_Y[keep_mask]
        else:
            raise ValueError(f"Unknown perturbation: {perturbation_type}")

        if perturbation_type == "centroid_dropout":
            preds = _predict_from_centroids(X_test, X_perturbed, centroids_Y_pert)
        else:
            preds = _predict_from_centroids(X_test, X_perturbed, centroids_Y)
        results[f"{perturbation_type}_{lvl}"] = best_action_match(preds, Y_test)

    return results


def _predict_from_centroids(X_query, centroids_X, centroids_Y):
    idxs, _ = pairwise_distances_argmin_min(X_query, centroids_X)
    return centroids_Y[idxs]


def main():
    run_id = f"c5_{int(time.time())}"
    log_path = os.path.join(RESULTS_DIR, "run_log.txt")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()

    log("=" * 60)
    log("Phase 7 3.3A: Cross-Bottleneck Joint — Counterfactual Analogue")
    log("=" * 60)
    log(f"Run ID: {run_id}")
    log(f"Hypothesis: Y-aware stabilization centroids are more robust to perturbations")
    log("")

    t0 = time.time()

    log("Step 1: Load counterfactual data")
    if not os.path.exists(CF_PATH):
        log(f"  Generating counterfactual data to {CF_PATH}...")
        df = generate_counterfactual_table(
            StructuredVolatilityEnv, ENV_KWARGS,
            n_train=1200, n_val=200, n_test_id=200, n_ood=300,
            horizons=(1, 3, 5), seeds=range(N_SEEDS),
        )
        df.to_csv(CF_PATH, index=False)
    else:
        df = pd.read_csv(CF_PATH)
    log(f"  Loaded {len(df)} rows, {df['seed'].nunique()} seeds")

    per_seed_train = {}
    for seed in range(N_SEEDS):
        train_df = df[(df["seed"] == seed) & (df["split"] == "train") & (df["horizon"] == 1)]
        X_tr, Y_tr, ba_tr = prepare_counterfactual_data(train_df, seed, ENV_KWARGS)
        if X_tr is not None and len(X_tr) > 0:
            per_seed_train[seed] = (X_tr, Y_tr, ba_tr)

    test_df = df[(df["seed"] == 0) & (df["split"] == "test_id") & (df["horizon"] == 1)]
    X_test, Y_test, ba_test = prepare_counterfactual_data(test_df, 0, ENV_KWARGS)
    log(f"  Train: {sum(len(v[0]) for v in per_seed_train.values())} samples")
    log(f"  Test: {len(X_test)} samples")

    log("")
    log("Step 2: Consolidate all strategies (unperturbed baseline)")

    strategies = {
        "kmeans_20": lambda: KMeansBaseline(n_prototypes=20),
        "y_aware_w5.0": lambda: YAwareKMeans(n_prototypes=20, y_weight=5.0),
        "per_action_kmeans": lambda: PerActionKMeans(n_prototypes_per_action=7),
        "nomemory": lambda: NoMemoryBaseline(),
    }

    baseline_results = {}
    for sname, factory in strategies.items():
        strategy = factory()
        for seed in range(N_SEEDS):
            if seed not in per_seed_train:
                continue
            strategy.update(per_seed_train[seed][0], per_seed_train[seed][1], seed_label=seed)
        base_match = best_action_match(strategy.predict(X_test), Y_test)
        baseline_results[sname] = base_match
        log(f"  {sname:25s}: {base_match:.4f} (unperturbed)")

    log("")
    log("Step 3: Apply perturbations and measure robustness")

    perturbation_types = {
        "additive_noise":   [0.01, 0.03, 0.06, 0.10, 0.20, 0.40, 0.80],
        "directional_shift": [0.01, 0.03, 0.06, 0.10, 0.20, 0.40, 0.80],
        "centroid_dropout":  [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90],
    }

    all_records = []
    for seed in range(5):
        if seed > 0:
            rng = np.random.RandomState(42 + seed)
        else:
            rng = np.random.RandomState(42)

        for sname, factory in strategies.items():
            strategy = factory()
            for sd in range(N_SEEDS):
                if sd not in per_seed_train:
                    continue
                strategy.update(per_seed_train[sd][0], per_seed_train[sd][1], seed_label=sd)

            for ptype, levels in perturbation_types.items():
                results = perturb_and_evaluate(strategy, X_test, Y_test, ptype, levels)
                for key, match in results.items():
                    lvl = float(key.split("_")[-1])
                    all_records.append({
                        "strategy": sname, "perturbation": ptype,
                        "level": lvl, "match": match, "run": seed,
                    })

    df_results = pd.DataFrame(all_records)

    log("")
    log("Step 4: Aggregate results")
    log("=" * 60)

    agg = df_results.groupby(["strategy", "perturbation", "level"])["match"].agg(["mean", "std"]).reset_index()

    summary = {}
    for ptype in perturbation_types:
        log(f"\n  {ptype}:")
        for sname in ["per_action_kmeans", "y_aware_w5.0", "kmeans_20", "nomemory"]:
            sub = agg[(agg["strategy"] == sname) & (agg["perturbation"] == ptype)]
            if len(sub) == 0:
                continue
            lvl_0 = sub[sub["level"] == sub["level"].min()]["mean"].values[0]
            lvl_max = sub[sub["level"] == sub["level"].max()]["mean"].values[0]
            drop = lvl_0 - lvl_max
            decay_pct = (drop / lvl_0 * 100) if lvl_0 > 0 else 0
            log(f"    {sname:25s}: {lvl_0:.4f} → {lvl_max:.4f} (drop {drop:.4f}, {decay_pct:.0f}%)")
            if sname not in summary:
                summary[sname] = {}
            summary[sname][ptype] = {"lvl_0": lvl_0, "lvl_max": lvl_max, "drop": drop, "decay_pct": decay_pct}

    pa = summary.get("per_action_kmeans", {})
    km = summary.get("kmeans_20", {})

    log(f"\n{'='*60}")
    log("Cross-Bottleneck Synergism Analysis:")
    log(f"{'='*60}")

    for ptype in perturbation_types:
        pa_decay = pa.get(ptype, {}).get("decay_pct", 0)
        km_decay = km.get(ptype, {}).get("decay_pct", 0)
        robustness_advantage = km_decay - pa_decay
        log(f"  {ptype}:")
        log(f"    Per-Action decay: {pa_decay:.1f}%, KMeans decay: {km_decay:.1f}%")
        log(f"    Robustness advantage (PA vs KMeans): {robustness_advantage:+.1f}%")
        if robustness_advantage > 5:
            log(f"    → SYNERGISM: Y-aware stabilization provides substantial robustness")
        elif robustness_advantage > 0:
            log(f"    → WEAK SYNERGISM: marginal robustness benefit")
        else:
            log(f"    → NO SYNERGISM: Y-aware does not improve perturbation robustness")

    csv_path = os.path.join(RESULTS_DIR, "cross_bottleneck_results.csv")
    df_results.to_csv(csv_path, index=False)
    log(f"\n  Results saved to {csv_path}")

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed:.0f}s")
    log(f"\n{'='*60}")
    log("Phase 7 3.3A complete.")
    log("=" * 60)


if __name__ == "__main__":
    main()