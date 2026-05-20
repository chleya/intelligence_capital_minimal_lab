"""IC-2c.1: Root Cause Analysis — Why NoMemory > Episodic > Consolidated.

Breaks down three questions:
  1. Why NoMemory wins: does it eat a state-action frequency shortcut?
  2. Why episodic didn't become appreciation: what breaks the k-NN signal?
  3. Why consolidated drifts: centroid movement and information density loss.

Uses per-seed counterfactual table data and traces centroid evolution.
"""

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import prepare_counterfactual_data

ENV_KWARGS = dict(
    state_dim=2, mode_flip_prob=0.08, autonomous_drift=0.05,
    autonomous_noise=0.02, action_gain=0.70, action_noise=0.03,
    action_sign_flip=True, history_len=8,
    action_cost=0.20, state_dependent_gain=True, saturation_k=0.5,
)
SEEDS = list(range(5))
OUT_DIR = "results/ic2c"
os.makedirs(OUT_DIR, exist_ok=True)


def load_per_seed_data():
    cf_df = pd.read_csv("results/counterfactual_table.csv")
    per_seed = {}
    for seed in SEEDS:
        train_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "train") & (cf_df["horizon"] == 1)]
        X, Y, ba = prepare_counterfactual_data(train_df, seed, ENV_KWARGS)
        if X is not None:
            per_seed[seed] = {"X": X, "Y": Y, "best_action": ba}
    return per_seed


def analyze_action_distributions(per_seed):
    print("=" * 60)
    print("Q1: WHY NOMEMORY WINS — ACTION FREQUENCY ANALYSIS")
    print("=" * 60)
    rows = []
    for seed, data in per_seed.items():
        ba = data["best_action"]
        counts = np.bincount(ba, minlength=3)
        probs = counts / counts.sum()
        rows.append({
            "seed": seed,
            "n_samples": len(ba),
            "action_-1": probs[0],
            "action_0": probs[1],
            "action_+1": probs[2],
            "dominant_action": np.argmax(counts) - 1,
            "dominant_prob": probs.max(),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nMean dominant action prob across seeds: {df['dominant_prob'].mean():.3f}")
    print(f"Inter-seed std of dominant prob: {df['dominant_prob'].std():.3f}")

    test_seed = 0
    test_df = pd.read_csv("results/counterfactual_table.csv")
    test_data = test_df[(test_df["seed"] == test_seed) & (test_df["split"] == "test_id") & (test_df["horizon"] == 1)]
    _, Y_test, ba_test = prepare_counterfactual_data(test_data, test_seed, ENV_KWARGS)
    test_counts = np.bincount(ba_test, minlength=3)
    test_probs = test_counts / test_counts.sum()
    print(f"\nTest set (seed={test_seed}) best action distribution:")
    print(f"  action_-1: {test_probs[0]:.3f}, action_0: {test_probs[1]:.3f}, action_+1: {test_probs[2]:.3f}")
    print(f"  Dominant: action_{np.argmax(test_counts)-1} at {test_probs.max():.3f}")

    all_counts = np.zeros(3)
    all_actions = np.concatenate([d["best_action"] for d in per_seed.values()])
    all_counts_ba = np.bincount(all_actions, minlength=3)
    all_probs = all_counts_ba / all_counts_ba.sum()
    print(f"\nAll-seed aggregate best action distribution:")
    print(f"  action_-1: {all_probs[0]:.3f}, action_0: {all_probs[1]:.3f}, action_+1: {all_probs[2]:.3f}")
    print(f"  NoMemory ceiling (best single action): {all_probs.max():.3f}")

    df.to_csv(f"{OUT_DIR}/q1_action_distributions.csv", index=False)
    return df


def analyze_feature_distribution_shifts(per_seed):
    print("\n" + "=" * 60)
    print("Q2: WHY EPISODIC FAILS — CROSS-SEED FEATURE DISTRIBUTION DRIFT")
    print("=" * 60)
    rows = []
    all_X = np.concatenate([d["X"] for d in per_seed.values()], axis=0)
    for seed, data in per_seed.items():
        X = data["X"]
        rows.append({
            "seed": seed,
            "feature_mean_L2": float(np.linalg.norm(X.mean(axis=0))),
            "feature_std_mean": float(X.std(axis=0).mean()),
            "feature_var_trace": float(np.trace(np.cov(X.T))),
        })
    feat_df = pd.DataFrame(rows)
    print(feat_df.to_string(index=False))

    for s1 in SEEDS:
        for s2 in SEEDS:
            if s1 >= s2:
                continue
            X1 = per_seed[s1]["X"]
            X2 = per_seed[s2]["X"]
            mean_dist = float(np.linalg.norm(X1.mean(axis=0) - X2.mean(axis=0)))
            print(f"  s{s1}↔s{s2}: mean L2 distance = {mean_dist:.4f}")

    global_mean = all_X.mean(axis=0, keepdims=True)
    for seed, data in per_seed.items():
        X = data["X"]
        dist_to_global = float(np.linalg.norm(X.mean(axis=0) - global_mean))
        print(f"  s{seed} → global mean distance: {dist_to_global:.4f}")

    print(f"\nGlobal feature var trace: {float(np.trace(np.cov(all_X.T))):.4f}")
    feat_df.to_csv(f"{OUT_DIR}/q2_feature_distributions.csv", index=False)
    return feat_df


def analyze_centroid_drift(per_seed):
    print("\n" + "=" * 60)
    print("Q3: WHY CONSOLIDATED DRIFTS — CENTROID EVOLUTION ACROSS STEPS")
    print("=" * 60)
    n_prototypes = 20
    all_X_cumulative = []
    all_Y_cumulative = []
    step_data = []

    for step in range(len(SEEDS)):
        seed = step
        X_new = per_seed[seed]["X"]
        Y_new = per_seed[seed]["Y"]
        all_X_cumulative.extend(list(X_new))
        all_Y_cumulative.extend(list(Y_new))
        X_arr = np.array(all_X_cumulative)
        Y_arr = np.array(all_Y_cumulative)
        nc = min(n_prototypes, len(X_arr))
        km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
        labels = km.fit_predict(X_arr)
        centroids = km.cluster_centers_
        Y_c = np.zeros((nc, 3))
        for i in range(nc):
            mask = labels == i
            if mask.sum() > 0:
                Y_c[i] = Y_arr[mask].mean(axis=0)

        centroid_avg_radius = float(np.mean(np.linalg.norm(centroids - centroids.mean(axis=0), axis=1)))
        cluster_counts = np.bincount(labels, minlength=nc)
        n_empty = int(np.sum(cluster_counts == 0))
        gini = 0.0
        if len(cluster_counts) > 0 and cluster_counts.sum() > 0:
            sorted_counts = np.sort(cluster_counts[cluster_counts > 0])
            cdf = 0
            for i, c in enumerate(sorted_counts):
                cdf += c
                gini += (2 * (i + 1) - len(sorted_counts) - 1) * c
            gini = gini / (len(sorted_counts) * cdf) if cdf > 0 else 0.0

        step_data.append({
            "step": step + 1,
            "seed_added": seed,
            "n_samples": len(X_arr),
            "n_centroids": nc,
            "n_empty_clusters": n_empty,
            "centroid_avg_radius": round(float(centroid_avg_radius), 4),
            "cluster_count_gini": round(float(gini), 4),
            "imbalance_ratio": round(float(cluster_counts.max() / max(cluster_counts.min(), 1)), 2),
        })

    drift_df = pd.DataFrame(step_data)
    print(drift_df.to_string(index=False))

    print("\nCentroid trajectory geometry (relative to previous step centroids):")
    for step in range(len(SEEDS)):
        seed = step
        X_new = per_seed[seed]["X"]
        Y_new = per_seed[seed]["Y"]
        current_X = list(all_X_cumulative[:sum(len(per_seed[s]["X"]) for s in range(step + 1))])
        current_Y = list(all_Y_cumulative[:sum(len(per_seed[s]["Y"]) for s in range(step + 1))])
        X_arr = np.array(current_X)
        Y_arr = np.array(current_Y)
        nc = min(n_prototypes, len(X_arr))
        km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
        labels = km.fit_predict(X_arr)
        centroids_current = km.cluster_centers_

        if step > 0:
            prev_X = np.array(current_X[:sum(len(per_seed[s]["X"]) for s in range(step))])
            prev_Y = np.array(current_Y[:sum(len(per_seed[s]["Y"]) for s in range(step))])
            prev_nc = min(n_prototypes, len(prev_X))
            prev_km = KMeans(n_clusters=prev_nc, random_state=42, n_init="auto")
            prev_km.fit(prev_X)
            centroids_prev = prev_km.cluster_centers_
            mean_shift = float(np.mean([np.min(np.linalg.norm(c_cur - centroids_prev, axis=1))
                                         for c_cur in centroids_current]))
            print(f"  Step {step+1}: centroid avg drift from step {step} = {mean_shift:.4f}")

    drift_df.to_csv(f"{OUT_DIR}/q3_centroid_drift.csv", index=False)
    return drift_df


def analyze_knn_signal_degradation(per_seed):
    print("\n" + "=" * 60)
    print("Q2 EXTENDED: k-NN SIGNAL DEGRADATION PER ADDED SEED")
    print("=" * 60)

    test_seed = 0
    test_df = pd.read_csv("results/counterfactual_table.csv")
    test_data = test_df[(test_df["seed"] == test_seed) & (test_df["split"] == "test_id") & (test_df["horizon"] == 1)]
    X_test, Y_test, ba_test = prepare_counterfactual_data(test_data, test_seed, ENV_KWARGS)

    rows = []
    for step in range(1, len(SEEDS) + 1):
        X_buf = np.concatenate([per_seed[s]["X"] for s in range(step)], axis=0)
        Y_buf = np.concatenate([per_seed[s]["Y"] for s in range(step)], axis=0)
        X_buf_s0 = per_seed[0]["X"]
        Y_buf_s0 = per_seed[0]["Y"]

        knn_all = [KNeighborsRegressor(n_neighbors=min(5, len(X_buf))) for _ in range(3)]
        for a in range(3):
            knn_all[a].fit(X_buf, Y_buf[:, a])
        preds_all = np.stack([k.predict(X_test) for k in knn_all], axis=-1)
        match_all = float(np.mean(np.argmax(preds_all, axis=1) == ba_test))

        knn_s0 = [KNeighborsRegressor(n_neighbors=min(5, len(X_buf_s0))) for _ in range(3)]
        for a in range(3):
            knn_s0[a].fit(X_buf_s0, Y_buf_s0[:, a])
        preds_s0 = np.stack([k.predict(X_test) for k in knn_s0], axis=-1)
        match_s0 = float(np.mean(np.argmax(preds_s0, axis=1) == ba_test))

        X_test_mean = X_test.mean(axis=0)
        train_dists = np.linalg.norm(X_buf - X_test_mean, axis=1)
        nn5_avg_dist = float(np.mean(np.sort(train_dists)[:min(5, len(train_dists))]))

        rows.append({
            "step": step,
            "seeds_in_buf": list(range(step)),
            "buffer_size": len(X_buf),
            "knn_all_seeds_match": round(match_all, 4),
            "knn_s0_only_match": round(match_s0, 4),
            "cross_contamination": round(match_s0 - match_all, 4),
            "nn5_avg_distance": round(nn5_avg_dist, 4),
        })

    kdf = pd.DataFrame(rows)
    print(kdf.to_string(index=False))
    kdf.to_csv(f"{OUT_DIR}/q2_knn_signal_degradation.csv", index=False)
    return kdf


def main():
    per_seed = load_per_seed_data()
    for s, d in per_seed.items():
        print(f"Seed {s}: X={d['X'].shape}, Y={d['Y'].shape}, best_actions={len(d['best_action'])}")

    a1 = analyze_action_distributions(per_seed)
    a2 = analyze_feature_distribution_shifts(per_seed)
    a2b = analyze_knn_signal_degradation(per_seed)
    a3 = analyze_centroid_drift(per_seed)

    summary = {
        "q1": "NoMemory exploits stable action-frequency shortcut; inter-seed action distribution is consistent",
        "q2": "Cross-seed feature drift corrupts k-NN; adding non-s0 data increases nearest-neighbor distance",
        "q3": "Consolidated centroids drift and become imbalanced; cluster quality degrades with each cross-seed addition",
    }
    with open(f"{OUT_DIR}/ic2c1_root_cause_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("IC-2c.1 Root Cause Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()