"""
IC-2: Consolidation Topology Audit — Relational Memory Hypothesis Experiment #2.

Computes topological fidelity metrics on episodic vs. consolidated memory:
  - TPR (Topology Preservation Ratio): correlation of episodic vs consolidated pairwise distances
  - RRP (Relational Recall Precision): fraction of nearest neighbors preserved after consolidation
  - MEC proxy (Multi-Entry Consistency): query robustness across seed subsets

Based on existing IC-2c.1 infrastructure. Fully CPU-executable.

Usage:
    python -m src.run_consolidation_topology_audit
"""

import os
import sys
import json
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
OUT_DIR = "results/ic2c_topology"
os.makedirs(OUT_DIR, exist_ok=True)


def _log(msg):
    print(msg, flush=True)


def load_per_seed_data():
    cf_df = pd.read_csv("results/counterfactual_table.csv")
    per_seed = {}
    for seed in SEEDS:
        train_df = cf_df[
            (cf_df["seed"] == seed)
            & (cf_df["split"] == "train")
            & (cf_df["horizon"] == 1)
        ]
        X, Y, ba = prepare_counterfactual_data(train_df, seed, ENV_KWARGS)
        if X is not None:
            per_seed[seed] = {"X": X, "Y": Y, "best_action": ba}
    return per_seed


def compute_pairwise_distances(X):
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def compute_topology_preservation_ratio(per_seed, n_prototypes=20):
    _log("=" * 60)
    _log("TPR: TOPOLOGY PRESERVATION RATIO")
    _log("=" * 60)

    all_X_cumulative = []
    step_records = []

    for step in range(len(SEEDS)):
        seed = step
        X_new = per_seed[seed]["X"]
        all_X_cumulative.extend(list(X_new))
        X_arr = np.array(all_X_cumulative)

        nc = min(n_prototypes, len(X_arr))
        km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
        labels = km.fit_predict(X_arr)
        centroids = km.cluster_centers_

        D_episodic = compute_pairwise_distances(X_arr)
        D_consolidated = compute_pairwise_distances(centroids)

        triu_idx = np.triu_indices(len(D_episodic), k=1)
        D_epi_flat = D_episodic[triu_idx]
        if len(D_epi_flat) > 0:
            pearson_r = float(np.corrcoef(D_epi_flat, np.random.rand(len(D_epi_flat)))[0, 1])
        else:
            pearson_r = 0.0

        n_samples = len(X_arr)
        sample_indices = np.arange(n_samples)
        np.random.shuffle(sample_indices)
        sampled_indices = sample_indices[:min(500, n_samples)]
        D_epi_sampled = D_episodic[np.ix_(sampled_indices, sampled_indices)]

        centroid_assignments = labels[sampled_indices]
        centroid_pairs = []
        for i in range(len(sampled_indices)):
            for j in range(i + 1, len(sampled_indices)):
                ci = centroid_assignments[i]
                cj = centroid_assignments[j]
                if ci < len(centroids) and cj < len(centroids):
                    centroid_pairs.append(
                        float(np.linalg.norm(centroids[ci] - centroids[cj]))
                    )

        if len(centroid_pairs) > 0:
            epi_triu = D_epi_sampled[np.triu_indices(len(sampled_indices), k=1)]
            if len(epi_triu) == len(centroid_pairs) and len(epi_triu) > 1:
                tpr = float(np.corrcoef(epi_triu, centroid_pairs)[0, 1])
            else:
                tpr = 0.0
        else:
            tpr = 0.0

        step_records.append({
            "step": step + 1,
            "seed_added": seed,
            "n_samples": n_samples,
            "n_centroids": nc,
            "tpr": round(tpr, 4),
            "pearson_baseline": round(pearson_r, 4),
            "tpr_excess": round(tpr - pearson_r, 4),
        })
        _log(f"  Step {step+1}: n={n_samples}, centroids={nc}, TPR={tpr:.4f}")

    tpr_df = pd.DataFrame(step_records)
    tpr_df.to_csv(f"{OUT_DIR}/topology_preservation_ratio.csv", index=False)

    final_tpr = step_records[-1]["tpr"]
    _log(f"\n  Final TPR (step {len(SEEDS)}): {final_tpr:.4f}")
    if final_tpr < 0.5:
        _log("  D10 TRIGGERED: TPR < 0.5 — consolidation is topology-destroying.")
    else:
        _log("  OK: TPR >= 0.5 — relational structure partially preserved.")

    return tpr_df


def compute_relational_recall_precision(per_seed, n_prototypes=20):
    _log("\n" + "=" * 60)
    _log("RRP: RELATIONAL RECALL PRECISION")
    _log("=" * 60)

    all_X_cumulative = []
    step_records = []

    for step in range(len(SEEDS)):
        seed = step
        X_new = per_seed[seed]["X"]
        all_X_cumulative.extend(list(X_new))
        X_arr = np.array(all_X_cumulative)

        nc = min(n_prototypes, len(X_arr))
        km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
        labels = km.fit_predict(X_arr)
        centroids = km.cluster_centers_

        nn_preserved = 0
        nn_total = min(100, len(X_arr))
        sample_indices = np.random.choice(len(X_arr), size=nn_total, replace=False)

        for si in sample_indices:
            point = X_arr[si]
            si_cluster = labels[si]

            D_epi = np.linalg.norm(X_arr - point, axis=1)
            D_epi[si] = 1e10
            nn_epi_idx = int(np.argmin(D_epi))
            nn_epi_cluster = labels[nn_epi_idx]

            D_con = np.linalg.norm(centroids - point, axis=1)
            nn_con_idx = int(np.argmin(D_con))

            if nn_epi_cluster == nn_con_idx:
                nn_preserved += 1

        rrp = nn_preserved / nn_total if nn_total > 0 else 0.0

        step_records.append({
            "step": step + 1,
            "seed_added": seed,
            "n_samples": len(X_arr),
            "n_centroids": nc,
            "nn_tested": nn_total,
            "nn_preserved": nn_preserved,
            "rrp": round(rrp, 4),
        })
        _log(f"  Step {step+1}: RRP={rrp:.4f} ({nn_preserved}/{nn_total})")

    rrp_df = pd.DataFrame(step_records)
    rrp_df.to_csv(f"{OUT_DIR}/relational_recall_precision.csv", index=False)

    final_rrp = step_records[-1]["rrp"]
    _log(f"\n  Final RRP (step {len(SEEDS)}): {final_rrp:.4f}")
    if final_rrp < 0.5:
        _log("  D12 TRIGGERED: RRP < 0.5 — nearest-neighbor structure destroyed.")
    else:
        _log("  OK: RRP >= 0.5 — local geometry partially preserved")

    return rrp_df


def compute_cluster_purity_degradation(per_seed, n_prototypes=20):
    _log("\n" + "=" * 60)
    _log("CLUSTER PURITY: SEED MIXING AUDIT")
    _log("=" * 60)

    all_X = []
    all_seed_labels = []

    for seed in SEEDS:
        X_seed = per_seed[seed]["X"]
        all_X.extend(list(X_seed))
        all_seed_labels.extend([seed] * len(X_seed))

    X_arr = np.array(all_X)
    seed_arr = np.array(all_seed_labels)

    nc = min(n_prototypes, len(X_arr))
    km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
    cluster_labels = km.fit_predict(X_arr)

    purity_rows = []
    for c in range(nc):
        mask = cluster_labels == c
        if mask.sum() == 0:
            continue
        seed_counts = np.bincount(seed_arr[mask], minlength=len(SEEDS))
        dominant_seed = int(np.argmax(seed_counts))
        purity = float(seed_counts[dominant_seed] / mask.sum())
        purity_rows.append({
            "cluster": c,
            "size": int(mask.sum()),
            "dominant_seed": dominant_seed,
            "purity": round(purity, 4),
            "seed_distribution": str([int(s) for s in seed_counts]),
        })

    purity_df = pd.DataFrame(purity_rows)
    purity_df.to_csv(f"{OUT_DIR}/cluster_purity.csv", index=False)

    mean_purity = float(purity_df["purity"].mean())
    n_mixed = int(np.sum(purity_df["purity"] < 0.8))

    _log(f"  Mean cluster purity: {mean_purity:.4f}")
    _log(f"  Clusters with purity < 0.8: {n_mixed}/{len(purity_df)}")
    _log(f"  This measures how much centroids mix different seeds' distributions.")

    return purity_df


def compute_knn_multi_entry_consistency(per_seed):
    _log("\n" + "=" * 60)
    _log("MEC PROXY: MULTI-ENTRY RECALL CONSISTENCY")
    _log("=" * 60)

    X_all = np.concatenate([d["X"] for d in per_seed.values()], axis=0)
    Y_all = np.concatenate([d["Y"] for d in per_seed.values()], axis=0)
    ba_all = np.concatenate([d["best_action"] for d in per_seed.values()])

    test_indices = np.random.choice(len(X_all), size=min(100, len(X_all)), replace=False)

    n_perturbations = 5
    noise_scale = 0.1
    consistency_records = []

    for idx in test_indices:
        query = X_all[idx].copy()
        true_ba = ba_all[idx]

        retrieved_bas = []
        for p in range(n_perturbations):
            perturbed = query + np.random.randn(*query.shape) * noise_scale * np.std(X_all, axis=0)
            distances = np.linalg.norm(X_all - perturbed, axis=1)
            nn_idx = int(np.argmin(distances))
            retrieved_bas.append(int(ba_all[nn_idx]))

        unique_bas = len(set(retrieved_bas))
        most_common_ba = max(set(retrieved_bas), key=retrieved_bas.count)
        consistency = float(retrieved_bas.count(most_common_ba) / n_perturbations)
        matches_true = 1.0 if most_common_ba == true_ba else 0.0

        consistency_records.append({
            "query_idx": int(idx),
            "true_ba": int(true_ba),
            "retrieved_bas": str(retrieved_bas),
            "unique_bas": unique_bas,
            "consistency": round(consistency, 4),
            "matches_true": int(matches_true),
        })

    mec_df = pd.DataFrame(consistency_records)
    mec_df.to_csv(f"{OUT_DIR}/multi_entry_consistency.csv", index=False)

    mean_consistency = float(mec_df["consistency"].mean())
    mean_match = float(mec_df["matches_true"].mean())

    _log(f"  Mean recall consistency (MEC proxy): {mean_consistency:.4f}")
    _log(f"  Mean match-to-true: {mean_match:.4f}")
    if mean_consistency < 0.5:
        _log("  D11 TRIGGERED: MEC proxy < 0.5 — memory not relationally robust.")
    else:
        _log("  OK: MEC proxy >= 0.5 — retrieval is query-robust.")

    return mec_df


def generate_report(tpr_df, rrp_df, purity_df, mec_df):
    report_path = f"{OUT_DIR}/IC2_TOPOLOGY_AUDIT_REPORT.md"
    lines = []

    lines.append("# IC-2: Consolidation Topology Audit Report")
    lines.append("")
    lines.append("**Experiment:** Relational Memory Hypothesis — Experiment #2  ")
    lines.append("**Date:** 2026-05-20  ")
    lines.append("**Status:** CPU-executed, no GPU required")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This experiment audits whether consolidation (KMeans centroids) preserves ")
    lines.append("the high-dimensional relational topology of episodic traces. The Relational ")
    lines.append("Memory Hypothesis predicts that centroid-based consolidation destroys ")
    lines.append("topological structure, explaining why consolidated match < random.")
    lines.append("")

    lines.append("## 2. Metrics Computed")
    lines.append("")
    lines.append("| Metric | Definition | Death Condition |")
    lines.append("|---|---|---|")
    lines.append("| **TPR** | Topology Preservation Ratio — corr(episodic pairwise distances, centroid distances) | D10: TPR < 0.5 |")
    lines.append("| **RRP** | Relational Recall Precision — fraction of nearest neighbors preserved after consolidation | D12: RRP < 0.5 |")
    lines.append("| **Purity** | Cluster purity — how much centroids mix different seeds' distributions | — |")
    lines.append("| **MEC proxy** | Multi-Entry Consistency — recall robustness to query perturbation | D11: MEC < 0.5 |")
    lines.append("")

    lines.append("## 3. TPR: Topology Preservation Ratio")
    lines.append("")
    lines.append("| Step | N Samples | N Centroids | TPR | Pearson Baseline | TPR Excess |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in tpr_df.iterrows():
        lines.append(f"| {int(row['step'])} | {int(row['n_samples'])} | {int(row['n_centroids'])} | {row['tpr']:.4f} | {row['pearson_baseline']:.4f} | {row['tpr_excess']:.4f} |")
    lines.append("")
    final_tpr = float(tpr_df.iloc[-1]["tpr"])
    lines.append(f"**Final TPR: {final_tpr:.4f}**")
    lines.append("")
    if final_tpr < 0.5:
        lines.append("**D10 TRIGGERED**: TPR < 0.5 — consolidation is topology-destroying. ")
        lines.append("Less than half of pairwise distance structure survives the centroid compression. ")
        lines.append("This directly supports the Relational Memory Hypothesis.")
    elif final_tpr < 0.7:
        lines.append("**WARNING**: TPR between 0.5-0.7 — significant topology loss.")
    else:
        lines.append("**OK**: TPR >= 0.7 — most relational structure preserved.")
    lines.append("")

    lines.append("## 4. RRP: Relational Recall Precision")
    lines.append("")
    lines.append("| Step | N Samples | N Centroids | NN Tested | NN Preserved | RRP |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in rrp_df.iterrows():
        lines.append(f"| {int(row['step'])} | {int(row['n_samples'])} | {int(row['n_centroids'])} | {int(row['nn_tested'])} | {int(row['nn_preserved'])} | {row['rrp']:.4f} |")
    lines.append("")
    final_rrp = float(rrp_df.iloc[-1]["rrp"])
    lines.append(f"**Final RRP: {final_rrp:.4f}**")
    lines.append("")
    if final_rrp < 0.5:
        lines.append("**D12 TRIGGERED**: RRP < 0.5 — nearest-neighbor structure is destroyed by compression.")
    else:
        lines.append("**OK**: RRP >= 0.5 — local geometry partially preserved.")
    lines.append("")

    lines.append("## 5. Cluster Purity: Seed Mixing")
    lines.append("")
    lines.append(f"**Mean purity: {float(purity_df['purity'].mean()):.4f}**")
    n_mixed = int(np.sum(purity_df["purity"] < 0.8))
    lines.append(f"**Clusters with purity < 0.8: {n_mixed}/{len(purity_df)}**")
    lines.append("")
    if n_mixed > len(purity_df) * 0.3:
        lines.append("Centroids significantly mix distributions from different seeds, ")
        lines.append("confirming that cross-distribution averaging creates structurally impure clusters.")
    lines.append("")

    lines.append("## 6. MEC Proxy: Multi-Entry Recall Consistency")
    lines.append("")
    mean_mec = float(mec_df["consistency"].mean())
    mean_match = float(mec_df["matches_true"].mean())
    lines.append(f"**Mean recall consistency: {mean_mec:.4f}**")
    lines.append(f"**Mean match-to-true: {mean_match:.4f}**")
    lines.append("")
    if mean_mec < 0.5:
        lines.append("**D11 TRIGGERED**: MEC proxy < 0.5 — retrieval not relationally robust.")
    lines.append("")

    lines.append("## 7. Death Condition Summary")
    lines.append("")
    lines.append("| Death Condition | Threshold | Actual | Triggered? |")
    lines.append("|---|---|---|---|")
    lines.append(f"| D10: TPR < 0.5 | < 0.5 | {final_tpr:.4f} | {'YES' if final_tpr < 0.5 else 'NO'} |")
    lines.append(f"| D11: MEC < 0.5 | < 0.5 | {mean_mec:.4f} | {'YES' if mean_mec < 0.5 else 'NO'} |")
    lines.append(f"| D12: RRP < 0.5 | < 0.5 | {final_rrp:.4f} | {'YES' if final_rrp < 0.5 else 'NO'} |")
    lines.append("")

    lines.append("## 8. Interpretation")
    lines.append("")
    if final_tpr < 0.5 and final_rrp < 0.5:
        lines.append("**Consolidation is confirmed as topology-destroying.** ")
        lines.append("The Relational Memory Hypothesis correctly predicted that centroid compression ")
        lines.append("breaks the high-dimensional relational structure that episodic traces maintain. ")
        lines.append("This explains why consolidated match (0.115) < random (0.33): the useful ")
        lines.append("relational information in episodic traces is destroyed by KMeans compression.")
    elif final_tpr < 0.5:
        lines.append("Global topology is destroyed but some local geometry survives. ")
        lines.append("Partial support for the Relational Memory Hypothesis.")
    else:
        lines.append("Topology is partially preserved. The Relational Memory Hypothesis prediction ")
        lines.append("is NOT strongly supported. Other mechanisms may explain the consolidation failure.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*IC-2 Consolidation Topology Audit — Relational Memory Hypothesis Experiment #2*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    _log(f"\nReport saved to {report_path}")


def main():
    _log("=" * 60)
    _log("IC-2: CONSOLIDATION TOPOLOGY AUDIT")
    _log("=" * 60)

    per_seed = load_per_seed_data()
    for s, d in per_seed.items():
        _log(f"  Seed {s}: X={d['X'].shape}, Y={d['Y'].shape}")

    tpr_df = compute_topology_preservation_ratio(per_seed)
    rrp_df = compute_relational_recall_precision(per_seed)
    purity_df = compute_cluster_purity_degradation(per_seed)
    mec_df = compute_knn_multi_entry_consistency(per_seed)

    generate_report(tpr_df, rrp_df, purity_df, mec_df)

    _log("\n" + "=" * 60)
    _log("IC-2 TOPOLOGY AUDIT COMPLETE")
    _log("=" * 60)


if __name__ == "__main__":
    main()