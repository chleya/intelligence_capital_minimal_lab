import json
import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_trajectory(env, steps=3000, seed=0):
    env.reset(seed)
    records = []
    for _ in range(steps):
        obs = env._get_obs()
        action = env.rng.choice(env.actions)
        obs_next_full, state_next, mode = env.step(action)
        records.append({
            "t": env.t - 1,
            "obs": obs,
            "action": action,
            "state_next": state_next,
            "mode": mode,
            "history_obs": env.get_history_obs(),
            "history_act": env.get_history_act(),
        })
    return records


def sample_states_from_trajectory(records, n_states, rng):
    idxs = rng.choice(len(records), n_states, replace=False)
    return [records[i] for i in idxs]


def generate_counterfactual_table(env_cls, env_kwargs, n_train=1200, n_val=200,
                                   n_test_id=200, n_ood=300, horizons=(1, 3, 5),
                                   seeds=range(10)):
    all_tables = []
    for seed in seeds:
        env = env_cls(seed=seed, **env_kwargs)
        records = generate_trajectory(env, steps=3000, seed=seed + 1000)
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(records), n_train + n_val + n_test_id + n_ood, replace=False)
        train_idx = idxs[:n_train]
        val_idx = idxs[n_train:n_train + n_val]
        test_idx = idxs[n_train + n_val:n_train + n_val + n_test_id]
        ood_idx = idxs[n_train + n_val + n_test_id:]

        def build_rows(indices, split_label, ood_type="none"):
            rows = []
            for i in indices:
                rec = records[i]
                snap = env.snapshot()
                env.restore({
                    "state": rec["state_next"].copy(),
                    "mode": rec["mode"],
                    "t": rec["t"] + 1,
                    "history_obs": [o.copy() for o in rec["history_obs"]],
                    "history_act": list(rec["history_act"]),
                    "rng_state": env.rng.bit_generator.state,
                })
                for h in horizons:
                    outcomes = env.compute_outcomes(horizon=h)
                    residuals = {int(a): out - outcomes[0] for a, out in outcomes.items()}
                    best_action = max(outcomes, key=lambda a: float(np.sum(outcomes[a])))
                    rows.append({
                        "seed": seed,
                        "state_idx": int(i),
                        "split": split_label,
                        "ood_type": ood_type,
                        "horizon": h,
                        "outcome_m1": outcomes[-1].tolist() if isinstance(outcomes[-1], np.ndarray) else outcomes[-1],
                        "outcome_0": outcomes[0].tolist() if isinstance(outcomes[0], np.ndarray) else outcomes[0],
                        "outcome_p1": outcomes[1].tolist() if isinstance(outcomes[1], np.ndarray) else outcomes[1],
                        "residual_m1": residuals[-1].tolist() if isinstance(residuals[-1], np.ndarray) else residuals[-1],
                        "residual_p1": residuals[1].tolist() if isinstance(residuals[1], np.ndarray) else residuals[1],
                        "best_action": int(best_action),
                        "mode": int(rec["mode"]),
                        "history_obs": [o.tolist() if isinstance(o, np.ndarray) else o for o in rec["history_obs"]],
                        "history_act": list(rec["history_act"]),
                    })
            return rows

        rows = build_rows(train_idx, "train")
        rows += build_rows(val_idx, "val")
        rows += build_rows(test_idx, "test_id")
        rows += build_rows(ood_idx, "test_ood", "mixed")
        all_tables.extend(rows)

    df = pd.DataFrame(all_tables)

    for h in horizons:
        for a in [-1, 0, 1]:
            col = f"outcome_{'m1' if a==-1 else str(a)}"
            if col in df.columns:
                pass

    return df


def compute_oracle_summary(cf_df, env_cls, env_kwargs, seed=0):
    env = env_cls(seed=seed, **env_kwargs)
    sub = cf_df[(cf_df["seed"] == seed) & (cf_df["horizon"] == 1) & (cf_df["split"] == "train")]
    outcomes = []
    for _, row in sub.iterrows():
        out_m1 = np.array(json.loads(row["outcome_m1"])) if isinstance(row["outcome_m1"], str) else np.array(row["outcome_m1"])
        out_0 = np.array(json.loads(row["outcome_0"])) if isinstance(row["outcome_0"], str) else np.array(row["outcome_0"])
        out_p1 = np.array(json.loads(row["outcome_p1"])) if isinstance(row["outcome_p1"], str) else np.array(row["outcome_p1"])
        outcomes.append({"m1": out_m1, "0": out_0, "p1": out_p1})

    all_vals = np.concatenate([o["m1"] for o in outcomes] + [o["0"] for o in outcomes] + [o["p1"] for o in outcomes])
    total_var = float(np.var(all_vals))

    residuals = []
    for o in outcomes:
        nop = o["0"]
        residuals.append(o["m1"] - nop)
        residuals.append(o["p1"] - nop)
    all_residuals = np.concatenate(residuals)
    residual_var = float(np.var(all_residuals))
    residual_variance_ratio = residual_var / total_var if total_var > 0 else 0.0

    so_correct = 0
    ao_correct = 0
    oracle_correct = 0
    total = len(outcomes)
    global_best_counts = {a: 0 for a in [-1, 0, 1]}
    for o in outcomes:
        best = max([(np.sum(o["m1"]), -1), (np.sum(o["0"]), 0), (np.sum(o["p1"]), 1)], key=lambda x: x[0])[1]
        global_best_counts[best] += 1
        so_pred = 0
        if so_pred == best:
            so_correct += 1
        oracle_correct += 1
    ao_pred = max(global_best_counts, key=global_best_counts.get)
    ao_correct = global_best_counts[ao_pred]
    oracle_correct = total

    return {
        "residual_variance_ratio": residual_variance_ratio,
        "total_variance": total_var,
        "residual_variance": residual_var,
        "so_match": so_correct / total,
        "ao_match": ao_correct / total,
        "oracle_match": 1.0,
        "n_samples": total,
    }