"""IC-2d: Readout-Matched Episodic — Learned vs k-NN Readout.

Tests the IC-2c.1 hypothesis: episodic retention fails NOT because of insufficient
information, but because Euclidean k-NN is a wrong readout for 24-dim history features.

Replaces k-NN with a simple MLP regressor trained on the episodic buffer.
Compares:
  - NoMemory (action-frequency baseline)
  - Episodic-kNN (original, Euclidean)
  - Episodic-MLP (new, learned readout)
  - Episodic-RF  (RandomForest as another learned readout)
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import prepare_counterfactual_data

ENV_KWARGS = dict(
    state_dim=2, mode_flip_prob=0.08, autonomous_drift=0.05,
    autonomous_noise=0.02, action_gain=0.70, action_noise=0.03,
    action_sign_flip=True, history_len=8, action_cost=0.20,
    state_dependent_gain=True, saturation_k=0.5,
)
SEEDS = list(range(5))
OUT_DIR = "results/ic2d"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "log.txt")

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
            return np.ones((len(X), 3)) / 3.0
        probs = self._counts / self._counts.sum()
        return np.tile(probs, (len(X), 1))


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
            return np.ones((len(X_query), 3)) / 3.0
        X_buf = np.array(self._X)
        Y_buf = np.array(self._Y)
        k_eff = min(self.k, len(X_buf))
        knns = [KNeighborsRegressor(n_neighbors=k_eff) for _ in range(3)]
        for a in range(3):
            knns[a].fit(X_buf, Y_buf[:, a])
        return np.stack([k.predict(X_query) for k in knns], axis=-1)


class EpisodicMLP:
    def __init__(self, max_buffer=200):
        self.max_buffer = max_buffer
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
            return np.ones((len(X_query), 3)) / 3.0
        X_buf = np.array(self._X)
        Y_buf = np.array(self._Y)
        min_s = min(100, len(X_buf))
        mlps = []
        for a in range(3):
            m = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                             max_iter=500, random_state=42, early_stopping=False)
            m.fit(X_buf, Y_buf[:, a])
            mlps.append(m)
        return np.stack([m.predict(X_query) for m in mlps], axis=-1)


class EpisodicRF:
    def __init__(self, max_buffer=200):
        self.max_buffer = max_buffer
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
            return np.ones((len(X_query), 3)) / 3.0
        X_buf = np.array(self._X)
        Y_buf = np.array(self._Y)
        rfs = []
        for a in range(3):
            rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=1)
            rf.fit(X_buf, Y_buf[:, a])
            rfs.append(rf)
        return np.stack([rf.predict(X_query) for rf in rfs], axis=-1)


def best_action_match(preds, ba_true):
    return float(np.mean(np.argmax(preds, axis=1) == ba_true))


def main():
    _log("=" * 60)
    _log("IC-2d: Readout-Matched Episodic")
    _log("=" * 60)

    cf_df = pd.read_csv("results/counterfactual_table.csv")
    per_seed_train = {}
    for seed in SEEDS:
        train_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "train") & (cf_df["horizon"] == 1)]
        X, Y, ba = prepare_counterfactual_data(train_df, seed, ENV_KWARGS)
        if X is not None:
            per_seed_train[seed] = (X, Y, ba)

    test_df = cf_df[(cf_df["seed"] == 0) & (cf_df["split"] == "test_id") & (cf_df["horizon"] == 1)]
    X_test, Y_test, ba_test = prepare_counterfactual_data(test_df, 0, ENV_KWARGS)
    _log(f"Test samples: {len(X_test)}")

    strategies = {
        "NoMemory": NoMemoryBaseline(),
        "Episodic-kNN": EpisodicKNN(max_buffer=200, k=5),
        "Episodic-MLP": EpisodicMLP(max_buffer=200),
        "Episodic-RF": EpisodicRF(max_buffer=200),
    }

    all_records = []
    for step in range(len(SEEDS)):
        seed = step
        if seed not in per_seed_train:
            continue
        X_step, Y_step, _ = per_seed_train[seed]
        _log(f"\nStep {step+1}: adding seed={seed} ({len(X_step)} samples)")

        for name, strat in strategies.items():
            t0 = time.time()
            strat.update(X_step, Y_step)
            if name != "NoMemory":
                preds = strat.predict(X_test)
            else:
                preds = strat.predict(X_test)
            match = best_action_match(preds, ba_test)
            elapsed = time.time() - t0
            all_records.append({
                "step": step + 1, "strategy": name,
                "best_action_match": round(match, 4),
                "predict_time_s": round(elapsed, 3),
            })
            _log(f"  {name:<16} match={match:.4f}  ({elapsed:.2f}s)")

    df = pd.DataFrame(all_records)
    df.to_csv(f"{OUT_DIR}/ic2d_results.csv", index=False)
    _log(f"\nSaved to {OUT_DIR}/ic2d_results.csv")

    table = df.pivot_table(index="step", columns="strategy", values="best_action_match", aggfunc="mean")
    _log("\nBest Action Match Trajectory:")
    _log(table.to_string())

    final = df[df["step"] == len(SEEDS)]
    _log("\nFinal Step Comparison:")
    for _, row in final.iterrows():
        _log(f"  {row['strategy']:<20} match={row['best_action_match']:.4f}")

    summary = {
        "final_nomemory": float(final[final["strategy"] == "NoMemory"]["best_action_match"].values[0]),
        "final_episodic_knn": float(final[final["strategy"] == "Episodic-kNN"]["best_action_match"].values[0]),
        "final_episodic_mlp": float(final[final["strategy"] == "Episodic-MLP"]["best_action_match"].values[0]),
        "final_episodic_rf": float(final[final["strategy"] == "Episodic-RF"]["best_action_match"].values[0]),
        "mlp_vs_knn_ratio": float(
            final[final["strategy"] == "Episodic-MLP"]["best_action_match"].values[0] /
            max(final[final["strategy"] == "Episodic-kNN"]["best_action_match"].values[0], 0.001)
        ),
        "rf_vs_knn_ratio": float(
            final[final["strategy"] == "Episodic-RF"]["best_action_match"].values[0] /
            max(final[final["strategy"] == "Episodic-kNN"]["best_action_match"].values[0], 0.001)
        ),
    }
    with open(f"{OUT_DIR}/ic2d_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _log(f"\nMLP/k-NN ratio: {summary['mlp_vs_knn_ratio']:.2f}x")
    _log(f"RF/k-NN ratio:  {summary['rf_vs_knn_ratio']:.2f}x")
    _log("\nDone.")


if __name__ == "__main__":
    main()