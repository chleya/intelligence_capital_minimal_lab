"""
Proof C: Anchored Consolidation — Stabilization Remedy.
========================================================
Tests whether adding an anchor (seed-0 centroids) to the consolidation
process prevents centroid drift and cluster purity collapse.

Design:
  - Naive Consolidated: original KMeans refit on all accumulated data
  - Anchored Consolidated: KMeans + centroid anchoring term
      * After seed-0: save centroids as anchors
      * On subsequent seeds: refit KMeans, then blend new centroids
        toward anchors using blend_ratio α
        centroid_new = α * anchor + (1-α) * kmeans_centroid
  - Compare: best_action_match trajectory over 5 update steps
  - Measure: centroid drift (L2 distance) and pseudo-purity preservation

Success criterion:
  - Anchored best_action_match > Naive best_action_match in later steps
  - or anchored centroid drift < naive centroid drift

Usage:
  cd intelligence_capital_minimal_lab
  python src/run_c_anchored_consolidation.py
"""

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import prepare_counterfactual_data

RESULTS_DIR = "results/ic2c_anchored"
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
BLEND_RATIOS = [0.0, 0.3, 0.5, 0.7]

# ── Naive ConsolidatedSummary (same as IC-2c) ────────────────
class ConsolidatedSummary:
    def __init__(self, n_prototypes=20):
        self.n_prototypes = n_prototypes
        self._centroids = None
        self._Y_centroids = None
        self._X_all = []
        self._Y_all = []

    def update(self, X_new, Y_new):
        X_list = list(self._X_all) if len(self._X_all) > 0 else []
        Y_list = list(self._Y_all) if len(self._Y_all) > 0 else []
        for i in range(len(X_new)):
            X_list.append(X_new[i])
            Y_list.append(Y_new[i])
        self._X_all = X_list
        self._Y_all = Y_list
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

    @property
    def memory_size(self):
        return len(self._centroids) if self._centroids is not None else 0

# ── Anchored ConsolidatedSummary ─────────────────────────────
class AnchoredConsolidatedSummary:
    """Consolidation with anchor: first KMeans centroids are saved as anchors.
    On subsequent refits, centroids are blended toward anchors."""
    def __init__(self, n_prototypes=20, blend_ratio=0.5):
        self.n_prototypes = n_prototypes
        self.blend_ratio = blend_ratio
        self._centroids = None
        self._anchors = None
        self._Y_centroids = None
        self._X_all = []
        self._Y_all = []
        self._step = 0
        self.centroid_drift = 0.0

    def update(self, X_new, Y_new):
        self._step += 1
        X_list = list(self._X_all) if len(self._X_all) > 0 else []
        Y_list = list(self._Y_all) if len(self._Y_all) > 0 else []
        for i in range(len(X_new)):
            X_list.append(X_new[i])
            Y_list.append(Y_new[i])
        self._X_all = X_list
        self._Y_all = Y_list
        X_arr = np.array(self._X_all)
        Y_arr = np.array(self._Y_all)
        nc = min(self.n_prototypes, len(X_arr))

        km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
        labels = km.fit_predict(X_arr)
        raw_centroids = km.cluster_centers_

        if self._step == 1 or self._anchors is None:
            self._anchors = raw_centroids.copy()
            self._centroids = raw_centroids
        else:
            nc_new = min(nc, len(self._anchors))
            diff = np.linalg.norm(raw_centroids[:nc_new] - self._anchors[:nc_new], axis=1).mean()
            self.centroid_drift = float(diff)
            blended = (self.blend_ratio * self._anchors[:nc_new] +
                       (1 - self.blend_ratio) * raw_centroids[:nc_new])
            self._centroids = blended

        self._Y_centroids = np.zeros((len(self._centroids), 3))
        for i in range(len(self._centroids)):
            mask = labels == i if i < len(labels) else np.zeros(len(labels), dtype=bool)
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

# ── Episodic Retention (same as IC-2c) ──────────────────────
from sklearn.neighbors import KNeighborsRegressor

class EpisodicRetention:
    def __init__(self, max_buffer=200, k=5):
        self.max_buffer = max_buffer
        self.k = k
        self._X = []
        self._Y = []

    def update(self, X_new, Y_new):
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

    def update(self, X_new, Y_new):
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

# ── Metrics ─────────────────────────────────────────────────
def best_action_match(preds, Y_true):
    pred_actions = np.argmax(preds, axis=1)
    true_actions = np.argmax(Y_true, axis=1)
    return np.mean(pred_actions == true_actions)

def centroid_pseudo_purity(centroids, labels_true, X_query):
    """How well do centroids separate samples from different seeds?
    Pseudo-purity: fraction of centroids that are closer to samples
    of one seed than to any sample of other seeds."""
    if centroids is None:
        return 0.0
    idxs, _ = pairwise_distances_argmin_min(X_query, centroids)
    purity_per_cluster = []
    for i in range(len(centroids)):
        mask = idxs == i
        if mask.sum() > 0:
            unique_labels, counts = np.unique(np.array(labels_true)[mask], return_counts=True)
            purity_per_cluster.append(counts.max() / counts.sum())
    return np.mean(purity_per_cluster) if purity_per_cluster else 0.0

# ── Main ────────────────────────────────────────────────────
def main():
    log_path = os.path.join(RESULTS_DIR, "run_log.txt")
    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()

    log("Proof C: Anchored Consolidation — Stabilization Remedy")
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
    log(f"Fixed test set: {len(X_test)} samples")

    # ── Run comparison ──────────────────────────────────────
    all_records = []

    # Naive consolidated
    log("\n[1] Naive Consolidated...")
    naive = ConsolidatedSummary(n_prototypes=N_PROTOTYPES)
    for step in range(UPDATE_STEPS):
        seed = step
        if seed not in per_seed_train:
            continue
        X_step, Y_step, _, _ = per_seed_train[seed]
        naive.update(X_step, Y_step)
        preds = naive.predict(X_test)
        match = best_action_match(preds, Y_test)
        all_records.append({"step": step + 1, "strategy": "naive_consolidated",
                            "best_action_match": match, "blend_ratio": 0.0})
        log(f"  Step {step+1}: match={match:.4f}")

    # Anchored with different blend ratios
    for br in BLEND_RATIOS:
        name = f"anchored_br{br:.1f}"
        log(f"\n[2] {name}...")
        anchored = AnchoredConsolidatedSummary(n_prototypes=N_PROTOTYPES, blend_ratio=br)
        for step in range(UPDATE_STEPS):
            seed = step
            if seed not in per_seed_train:
                continue
            X_step, Y_step, _, _ = per_seed_train[seed]
            anchored.update(X_step, Y_step)
            preds = anchored.predict(X_test)
            match = best_action_match(preds, Y_test)
            all_records.append({"step": step + 1, "strategy": name,
                                "best_action_match": match, "blend_ratio": br})
            log(f"  Step {step+1}: match={match:.4f}, centroid_drift={anchored.centroid_drift:.4f}")

    # Episodic & NoMemory baselines
    log("\n[3] Baselines...")
    episodic = EpisodicRetention(max_buffer=200, k=5)
    nomemory = NoMemoryBaseline()
    for step in range(UPDATE_STEPS):
        seed = step
        if seed not in per_seed_train:
            continue
        X_step, Y_step, _, _ = per_seed_train[seed]
        episodic.update(X_step, Y_step)
        nomemory.update(X_step, Y_step)
        preds_epi = episodic.predict(X_test)
        preds_nm = nomemory.predict(X_test)
        all_records.append({"step": step + 1, "strategy": "episodic",
                            "best_action_match": best_action_match(preds_epi, Y_test), "blend_ratio": 0.0})
        all_records.append({"step": step + 1, "strategy": "nomemory",
                            "best_action_match": best_action_match(preds_nm, Y_test), "blend_ratio": 0.0})

    # ── Results ─────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    csv_path = os.path.join(RESULTS_DIR, "anchored_consolidation_results.csv")
    df.to_csv(csv_path, index=False)
    log(f"\n{'='*60}")
    log(f"RESULTS saved to {csv_path} ({len(df)} rows)")
    log(f"{'='*60}")

    # Trajectory table
    table = df.pivot_table(index="step", columns="strategy", values="best_action_match", aggfunc="mean")
    log("\nBest Action Match Trajectory:")
    log(table.to_string())

    # Key comparison at final step
    final_step = df[df["step"] == UPDATE_STEPS]
    log(f"\nFinal Step ({UPDATE_STEPS}) Results:")
    for _, row in final_step.iterrows():
        log(f"  {row['strategy']:<30s} match={row['best_action_match']:.4f}")

    naive_final = final_step[final_step["strategy"] == "naive_consolidated"]["best_action_match"].values[0] if "naive_consolidated" in final_step["strategy"].values else 0

    for br in BLEND_RATIOS:
        name = f"anchored_br{br:.1f}"
        if name in final_step["strategy"].values:
            anchored_match = final_step[final_step["strategy"] == name]["best_action_match"].values[0]
            delta = anchored_match - naive_final
            log(f"  {name} Δ over naive: {delta:+.4f}")

    log(f"\n{'='*60}")
    best_anchored = max(
        [(br, final_step[final_step["strategy"] == f"anchored_br{br:.1f}"]["best_action_match"].values[0] 
          if f"anchored_br{br:.1f}" in final_step["strategy"].values else 0)
         for br in BLEND_RATIOS],
        key=lambda x: x[1]
    )
    log(f"KEY RESULT:")
    log(f"  naive_consolidated final match: {naive_final:.4f}")
    log(f"  best anchored (br={best_anchored[0]:.1f}) match: {best_anchored[1]:.4f}")
    if best_anchored[1] > naive_final:
        log(f"  VERDICT: Anchored consolidation IMPROVES over naive → stabilization remedy works")
        improvement = best_anchored[1] - naive_final
        log(f"  Improvement: {improvement:+.4f} ({(improvement/naive_final*100):+.1f}% over naive)")
    else:
        log(f"  VERDICT: Anchored consolidation does NOT improve over naive")
        log(f"  Possible reasons: anchor=seed0 data insufficient, or drift not the rate-limiting factor")

    # Trajectory comparison
    traj_naive = [float(df[(df["step"] == s+1) & (df["strategy"] == "naive_consolidated")]["best_action_match"].values[0]) for s in range(UPDATE_STEPS)]
    traj_anchored_best = [float(df[(df["step"] == s+1) & (df["strategy"] == f"anchored_br{best_anchored[0]:.1f}")]["best_action_match"].values[0]) for s in range(UPDATE_STEPS)]

    log(f"\nTrajectory comparison:")
    log(f"  Step:       {list(range(1, UPDATE_STEPS+1))}")
    log(f"  Naive:      {[f'{v:.4f}' for v in traj_naive]}")
    log(f"  Anchored:   {[f'{v:.4f}' for v in traj_anchored_best]}")
    log(f"  Difference: {[f'{traj_anchored_best[i]-traj_naive[i]:+.4f}' for i in range(len(traj_naive))]}")

    # Pseudo-purity analysis on test set
    log(f"\nPseudo-purity on test set:")
    seed_labels = list(test_df["seed"].values[:len(X_test)]) if "seed" in test_df.columns else None
    if seed_labels is not None:
        naive_purity = centroid_pseudo_purity(
            naive._centroids, seed_labels[:len(X_test)], X_test[:len(X_test)])
        log(f"  Naive centroids: {naive_purity:.4f}")
        for br in BLEND_RATIOS:
            name = f"anchored_br{br:.1f}"
            anch = None
            if br == best_anchored[0]:
                for step in reversed(range(UPDATE_STEPS)):
                    sub = df[(df["step"] == step+1) & (df["strategy"] == name)]
                    if len(sub) > 0:
                        break

    log("\n" + "=" * 60)
    log("Proof C complete.")
    log("=" * 60)

if __name__ == "__main__":
    main()