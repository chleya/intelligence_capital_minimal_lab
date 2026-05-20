"""IC-2c: Episodic vs Consolidated Capital — Continual Update Experiment.

Tests whether capital (compressed experience) appreciates or drifts into bad debt
under repeated consolidation/rewriting, compared to raw episodic retention.

Four strategies:
  1. EpisodicRetention  — fixed buffer, append new / evict old, k-NN predict
  2. ConsolidatedSummary — K prototypes, refit on all accumulated data each step
  3. MixedMemory         — half episodic buffer + half consolidated centroids
  4. NoMemoryBaseline    — simple action-frequency predictor (no history)

Update stream: train data from seeds 0→1→2→3→4 sequentially.
"""

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import prepare_counterfactual_data
from src.metrics import (
    compute_best_action_match, compute_regret, compute_rank_accuracy,
    compute_action_only_ceiling, compute_bad_debt_ratio,
    compute_state_only_shortcut_index, compute_action_only_shortcut_index,
    compute_shuffled_action_gap, compute_iar, compute_orr,
)

ENV_KWARGS = dict(
    state_dim=2, mode_flip_prob=0.08, autonomous_drift=0.05,
    autonomous_noise=0.02, action_gain=0.70, action_noise=0.03,
    action_sign_flip=True, history_len=8,
    action_cost=0.20, state_dependent_gain=True, saturation_k=0.5,
)
OBS_DIM = 2
HISTORY_LEN = 8
SEEDS = list(range(5))
UPDATE_STEPS = len(SEEDS)
RANDOM_BASELINE = 1.0 / 3.0


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
        preds = np.stack([k.predict(X_query) for k in knns], axis=-1)
        return preds

    @property
    def memory_size(self):
        return len(self._X)


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
        from sklearn.metrics import pairwise_distances_argmin_min
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids)
        return self._Y_centroids[idxs]

    @property
    def memory_size(self):
        return len(self._centroids) if self._centroids is not None else 0


class MixedMemory:
    def __init__(self, max_buffer=100, n_prototypes=10, k=5):
        self.episodic = EpisodicRetention(max_buffer=max_buffer, k=k)
        self.consolidated = ConsolidatedSummary(n_prototypes=n_prototypes)
        self.episodic_weight = 0.5

    def update(self, X_new, Y_new):
        self.episodic.update(X_new, Y_new)
        self.consolidated.update(X_new, Y_new)

    def predict(self, X_query):
        p_ep = self.episodic.predict(X_query)
        p_co = self.consolidated.predict(X_query)
        return self.episodic_weight * p_ep + (1 - self.episodic_weight) * p_co

    @property
    def memory_size(self):
        return self.episodic.memory_size + self.consolidated.memory_size


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


def evaluate_all_strategies(strategies, X_test, Y_test, ba_test, step_idx):
    records = []
    for name, strat in strategies.items():
        preds = strat.predict(X_test)
        match = compute_best_action_match(preds, Y_test)
        regret = compute_regret(preds, Y_test)
        rank_acc = compute_rank_accuracy(preds, Y_test)
        records.append({
            "step": step_idx,
            "strategy": name,
            "best_action_match": match,
            "regret": regret,
            "rank_accuracy": rank_acc,
            "memory_size": strat.memory_size,
        })
    return records


def main():
    os.makedirs("results/ic2c", exist_ok=True)
    print("=" * 60)
    print("IC-2c: Episodic vs Consolidated Capital")
    print("=" * 60)

    cf_df = pd.read_csv("results/counterfactual_table.csv")
    print(f"CF table: {len(cf_df)} rows, {cf_df['seed'].nunique()} seeds")

    per_seed_train = {}
    for seed in SEEDS:
        train_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "train") & (cf_df["horizon"] == 1)]
        X_tr, Y_tr, ba_tr = prepare_counterfactual_data(train_df, seed, ENV_KWARGS)
        if X_tr is not None and len(X_tr) > 0:
            per_seed_train[seed] = (X_tr, Y_tr, ba_tr)
    print(f"Per-seed train sizes: {[(s, len(v[0])) for s, v in per_seed_train.items()]}")

    test_df = cf_df[(cf_df["seed"] == 0) & (cf_df["split"] == "test_id") & (cf_df["horizon"] == 1)]
    X_test, Y_test, ba_test = prepare_counterfactual_data(test_df, 0, ENV_KWARGS)
    print(f"Fixed test set: {len(X_test)} samples")

    state_only_df = cf_df[(cf_df["seed"] == 0) & (cf_df["split"] == "test_id") & (cf_df["horizon"] == 1)]
    _, _, ba_so = prepare_counterfactual_data(state_only_df, 0, ENV_KWARGS)
    so_match = np.mean(ba_test == np.argmax(Y_test, axis=1)) if len(ba_test) > 0 else 0.0

    strategies = {
        "episodic": EpisodicRetention(max_buffer=200, k=5),
        "consolidated": ConsolidatedSummary(n_prototypes=20),
        "mixed": MixedMemory(max_buffer=100, n_prototypes=10, k=5),
        "nomemory": NoMemoryBaseline(),
    }
    print(f"\nStrategies: {list(strategies.keys())}")

    all_records = []
    cumulative_X = []
    cumulative_Y = []

    for step in range(UPDATE_STEPS):
        seed = step
        if seed not in per_seed_train:
            continue
        X_step, Y_step, _ = per_seed_train[seed]
        print(f"\nStep {step+1}/{UPDATE_STEPS}: adding seed={seed} data ({len(X_step)} samples)")

        if len(cumulative_X) == 0:
            cumulative_X = list(X_step)
            cumulative_Y = list(Y_step)
        else:
            for i in range(len(X_step)):
                cumulative_X.append(X_step[i])
                cumulative_Y.append(Y_step[i])
        cum_n = len(cumulative_X)
        print(f"  Cumulative data: {cum_n} samples")

        for name, strat in strategies.items():
            strat.update(X_step, Y_step)

        step_records = evaluate_all_strategies(strategies, X_test, Y_test, ba_test, step + 1)
        all_records.extend(step_records)

    df = pd.DataFrame(all_records)
    df.to_csv("results/ic2c/trajectory_metrics.csv", index=False)
    print(f"\nSaved trajectory metrics: {len(df)} rows")

    table = df.pivot_table(index="step", columns="strategy", values="best_action_match", aggfunc="mean")
    print("\nBest Action Match Trajectory:")
    print(table.to_string())

    regret_table = df.pivot_table(index="step", columns="strategy", values="regret", aggfunc="mean")
    print("\nRegret Trajectory:")
    print(regret_table.to_string())

    final_step = df[df["step"] == UPDATE_STEPS]
    print(f"\nFinal Step ({UPDATE_STEPS}) Results:")
    for _, row in final_step.iterrows():
        print(f"  {row['strategy']:<20} match={row['best_action_match']:.4f} "
              f"regret={row['regret']:.4f} rank_acc={row['rank_accuracy']:.4f} "
              f"mem={row['memory_size']}")

    epi_final = final_step[final_step["strategy"] == "episodic"]["best_action_match"].values[0]
    con_final = final_step[final_step["strategy"] == "consolidated"]["best_action_match"].values[0]
    mix_final = final_step[final_step["strategy"] == "mixed"]["best_action_match"].values[0]
    nm_final = final_step[final_step["strategy"] == "nomemory"]["best_action_match"].values[0]

    epi_traj = df[df["strategy"] == "episodic"]["best_action_match"].values
    con_traj = df[df["strategy"] == "consolidated"]["best_action_match"].values
    mix_traj = df[df["strategy"] == "mixed"]["best_action_match"].values
    nm_traj = df[df["strategy"] == "nomemory"]["best_action_match"].values

    epi_peak_idx = np.argmax(epi_traj)
    con_peak_idx = np.argmax(con_traj)
    mix_peak_idx = np.argmax(mix_traj)

    epi_peak = epi_traj[epi_peak_idx]
    con_peak = con_traj[con_peak_idx]
    mix_peak = mix_traj[mix_peak_idx]

    print(f"\nPeak Analysis:")
    print(f"  episodic peak at step {epi_peak_idx+1}: {epi_peak:.4f} → final: {epi_final:.4f} (Δ={epi_final-epi_peak:+.4f})")
    print(f"  consolidated peak at step {con_peak_idx+1}: {con_peak:.4f} → final: {con_final:.4f} (Δ={con_final-con_peak:+.4f})")
    print(f"  mixed peak at step {mix_peak_idx+1}: {mix_peak:.4f} → final: {mix_final:.4f} (Δ={mix_final-mix_peak:+.4f})")

    epi_trend = "appreciation" if epi_final >= epi_peak else "plateau/drift"
    con_trend = "appreciation" if con_final >= con_peak else "plateau/drift"
    mix_trend = "appreciation" if mix_final >= mix_peak else "plateau/drift"

    print(f"\nTrend Classification:")
    print(f"  episodic:     {epi_trend}")
    print(f"  consolidated: {con_trend}")
    print(f"  mixed:        {mix_trend}")

    bd_records = []
    for name in ["episodic", "consolidated", "mixed"]:
        sub = df[df["strategy"] == name]
        for _, row in sub.iterrows():
            step = int(row["step"])
            mm = row["best_action_match"]
            bdr = compute_bad_debt_ratio(1.0, 0.5, 0.4, mm)
            bd_records.append({
                "step": step,
                "strategy": name,
                "best_action_match": mm,
                "bad_debt_ratio": bdr,
            })
    bd_df = pd.DataFrame(bd_records)
    bd_df.to_csv("results/ic2c/bad_debt_trajectory.csv", index=False)

    summary = {
        "experiment": "IC-2c Episodic vs Consolidated Capital",
        "update_steps": UPDATE_STEPS,
        "test_samples": len(X_test),
        "final_episodic_match": float(epi_final),
        "final_consolidated_match": float(con_final),
        "final_mixed_match": float(mix_final),
        "final_nomemory_match": float(nm_final),
        "episodic_trend": epi_trend,
        "consolidated_trend": con_trend,
        "mixed_trend": mix_trend,
        "consolidated_peak_delta": float(con_final - con_peak),
        "episodic_peak_delta": float(epi_final - epi_peak),
    }
    with open("results/ic2c/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to results/ic2c/summary.json")

    print("\n" + "=" * 60)
    print("IC-2c complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()