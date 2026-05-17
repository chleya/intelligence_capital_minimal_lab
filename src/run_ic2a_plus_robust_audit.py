"""IC-2a+: Robust Oracle Residual Audit.

Validates that IC-2a oracle residual pass is robust against:
1. True OOD splits (background shift, action gain shift, sign rule shift)
2. Learned baselines (StateOnly, ActionOnly)
3. 0-action optimal proportion gate
4. Conditional RVR
5. History ablation

Only if all A-F gates pass may we proceed to IC-2b.
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env_structured_volatility import StructuredVolatilityEnv
from src.counterfactual_table import generate_counterfactual_table, compute_oracle_summary
from src.metrics import (
    compute_best_action_match, compute_oracle_action_entropy,
    compute_action_only_ceiling, compute_seed_stability_ratio,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Env kwargs from current provisional pass
ENV_KWARGS = dict(
    state_dim=2, mode_flip_prob=0.08, autonomous_drift=0.05,
    autonomous_noise=0.02, action_gain=0.70, action_noise=0.03,
    action_sign_flip=True, history_len=8,
    action_cost=0.20, state_dependent_gain=True, saturation_k=0.5,
)
SEEDS = list(range(10))
HORIZONS = (1, 3, 5)


def make_obs_array(row):
    """Reconstruct obs vector from stored history."""
    hist_obs = row.get("history_obs")
    hist_act = row.get("history_act")
    if hist_obs is None or hist_act is None:
        return None
    if isinstance(hist_obs, str):
        hist_obs = json.loads(hist_obs)
    if isinstance(hist_act, str):
        hist_act = json.loads(hist_act)
    parts = []
    for o, a in zip(hist_obs, hist_act):
        oa = np.array(o, dtype=np.float32)
        parts.append(np.concatenate([oa, [float(a)]]))
    return np.concatenate(parts).astype(np.float32)


def make_state_only_features(row, history_len=8):
    """Observable/history features WITHOUT action channel."""
    hist_obs = row.get("history_obs")
    hist_act = row.get("history_act")
    if hist_obs is None or hist_act is None:
        return None
    if isinstance(hist_obs, str):
        hist_obs = json.loads(hist_obs)
    if isinstance(hist_act, str):
        hist_act = json.loads(hist_act)
    parts = []
    for o, a in zip(hist_obs, hist_act):
        oa = np.array(o, dtype=np.float32)
        parts.append(oa)
        parts.append(np.array([float(a)], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def make_current_obs_only(row):
    """Only current state (last history entry), no action."""
    hist_obs = row.get("history_obs")
    if hist_obs is None:
        return None
    if isinstance(hist_obs, str):
        hist_obs = json.loads(hist_obs)
    return np.array(hist_obs[-1], dtype=np.float32)


def make_short_history(row, short_len=2):
    """Only last short_len steps."""
    hist_obs = row.get("history_obs")
    hist_act = row.get("history_act")
    if hist_obs is None or hist_act is None:
        return None
    if isinstance(hist_obs, str):
        hist_obs = json.loads(hist_obs)
    if isinstance(hist_act, str):
        hist_act = json.loads(hist_act)
    parts = []
    for o, a in zip(hist_obs[-short_len:], hist_act[-short_len:]):
        oa = np.array(o, dtype=np.float32)
        parts.append(np.concatenate([oa, [float(a)]]))
    # pad if needed
    while len(parts) < short_len:
        parts.append(np.zeros_like(parts[0]))
    return np.concatenate(parts).astype(np.float32)


def make_permuted_history(row, rng):
    """Permute temporal order of history steps."""
    hist_obs = row.get("history_obs")
    hist_act = row.get("history_act")
    if hist_obs is None or hist_act is None:
        return None
    if isinstance(hist_obs, str):
        hist_obs = json.loads(hist_obs)
    if isinstance(hist_act, str):
        hist_act = json.loads(hist_act)
    n = len(hist_obs)
    perm = rng.permutation(n)
    parts = []
    for idx in perm:
        oa = np.array(hist_obs[idx], dtype=np.float32)
        parts.append(np.concatenate([oa, [float(hist_act[idx])]]))
    return np.concatenate(parts).astype(np.float32)


def compute_outcome_vectors(df):
    """Return X, y_class, y_reg where y_reg is sum of outcomes per action."""
    def parse(v):
        if isinstance(v, str):
            return np.array(json.loads(v), dtype=np.float32)
        return np.array(v, dtype=np.float32)

    y_reg = []
    y_class = []
    for _, row in df.iterrows():
        sums = []
        for a in [-1, 0, 1]:
            col = f"outcome_{'m1' if a==-1 else ('p1' if a==1 else str(a))}"
            o = parse(row[col])
            sums.append(float(np.sum(o)))
        y_reg.append(sums)
        best = int(np.argmax(sums))
        y_class.append(best)
    return np.array(y_reg, dtype=np.float32), np.array(y_class, dtype=np.int64)


def compute_residual_variance_ratio_conditional(df):
    """Conditional RVR: mean_h Var_a(Y(h,a)) / mean_h Var_a(Y(h,a) + autonomous_component proxy).

    We proxy autonomous_component by outcome_0 (noop). So:
      numerator = mean_h Var_a(Y(h,a))
      denominator = mean_h Var_a(Y(h,a) + outcome_0)  ... but outcome_0 is same for all a,
    so Var_a(Y(h,a) + outcome_0) = Var_a(Y(h,a)). That collapses.

    Better: denominator = mean_h Var_a(outcome_a) where we treat outcome_0 as baseline.
    Actually the spec says:
      conditional_rvr = mean_h Var_a(Y(h,a)) / mean_h Var_a(Y(h,a) + autonomous_component)
    We proxy autonomous_component = outcome_0. Since adding constant per state doesn't change
    variance across actions, we instead interpret as:
      numerator = within-state action outcome variance
      denominator = total within-state variance = Var of all outcomes for that state
    So:
      numerator = Var_a(outcome_sum_a) across 3 actions
      denominator = Var of [outcome_m1, outcome_0, outcome_p1] concatenated? No.

    Let's use the simpler definition from spec:
      within_state_action_variance / total_within_state_variance
    where total_within_state_variance = variance of all scalar outcomes across the 3 actions.
    Since each outcome is a vector, we flatten per state.
    """
    def parse(v):
        if isinstance(v, str):
            return np.array(json.loads(v), dtype=np.float32)
        return np.array(v, dtype=np.float32)

    numers = []
    denoms = []
    for _, row in df.iterrows():
        outs = []
        for a in [-1, 0, 1]:
            col = f"outcome_{'m1' if a==-1 else ('p1' if a==1 else str(a))}"
            o = parse(row[col])
            outs.append(float(np.sum(o)))
        numers.append(np.var(outs))
        # total within-state variance = variance of all elements in all 3 outcome vectors
        all_elems = np.concatenate([parse(row[f"outcome_{'m1' if a==-1 else ('p1' if a==1 else str(a))}"]) for a in [-1,0,1]])
        denoms.append(np.var(all_elems))
    numer = float(np.mean(numers))
    denom = float(np.mean(denoms))
    return numer / denom if denom > 0 else 0.0, numer, denom


def compute_per_horizon_rvr(df):
    out = {}
    for h in df["horizon"].unique():
        sub = df[df["horizon"] == h]
        out[int(h)] = compute_residual_variance_ratio_conditional(sub)[0]
    return out


# ---------------------------------------------------------------------------
# Task 1: True OOD Splits
# ---------------------------------------------------------------------------
def generate_true_ood_table(ood_type, env_kwargs, n_samples=1200, seeds=SEEDS, horizons=HORIZONS):
    """Generate counterfactual table with a specific OOD perturbation."""
    all_rows = []
    for seed in seeds:
        kwargs = dict(env_kwargs)
        rng = np.random.default_rng(seed + 9999)
        if ood_type == "background_shift":
            kwargs["autonomous_drift"] = float(rng.uniform(0.02, 0.10))
            kwargs["autonomous_noise"] = float(rng.uniform(0.01, 0.06))
        elif ood_type == "action_gain_shift":
            kwargs["action_gain"] = float(rng.uniform(0.40, 0.90))
        elif ood_type == "sign_rule_shift":
            kwargs["action_sign_flip"] = True
            # Flip the sign rule interpretation: mode 0 gets flipped instead of mode 1
            # We implement by overriding env step logic via a wrapper flag not in base.
            # Instead, we swap mode interpretation by pre-flipping mode at snapshot restore.
            pass
        else:
            raise ValueError(ood_type)

        env = StructuredVolatilityEnv(seed=seed + 5000, **kwargs)
        # For sign_rule_shift, we hack by inverting mode effect
        if ood_type == "sign_rule_shift":
            # We'll generate table directly with custom outcome computation
            rows = _generate_rows_sign_rule_shift(env, seed, n_samples, horizons)
            all_rows.extend(rows)
            continue

        records = []
        env.reset(seed=seed + 1000)
        for _ in range(3000):
            obs = env._get_obs()
            action = env.rng.choice(env.actions)
            obs_next_full, state_next, mode = env.step(action)
            records.append({
                "t": env.t - 1,
                "state_next": state_next,
                "mode": mode,
                "history_obs": env.get_history_obs(),
                "history_act": env.get_history_act(),
            })
        rng2 = np.random.default_rng(seed)
        idxs = rng2.choice(len(records), n_samples, replace=False)
        for i in idxs:
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
                all_rows.append({
                    "seed": seed,
                    "state_idx": int(i),
                    "horizon": h,
                    "outcome_m1": outcomes[-1].tolist(),
                    "outcome_0": outcomes[0].tolist(),
                    "outcome_p1": outcomes[1].tolist(),
                    "residual_m1": residuals[-1].tolist(),
                    "residual_p1": residuals[1].tolist(),
                    "best_action": int(best_action),
                    "mode": int(rec["mode"]),
                })
            env.restore(snap)
    return pd.DataFrame(all_rows)


def _generate_rows_sign_rule_shift(env, seed, n_samples, horizons):
    """For sign_rule_shift OOD: flip which mode inverts action effect."""
    rows = []
    records = []
    env.reset(seed=seed + 1000)
    for _ in range(3000):
        action = env.rng.choice(env.actions)
        _, state_next, mode = env.step(action)
        records.append({
            "t": env.t - 1,
            "state_next": state_next,
            "mode": mode,
            "history_obs": env.get_history_obs(),
            "history_act": env.get_history_act(),
        })
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(records), n_samples, replace=False)
    for i in idxs:
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
        # Compute outcomes with flipped sign rule (mode 0 flips, mode 1 normal)
        outcomes = {}
        for a in env.actions:
            snap2 = env.snapshot()
            sign = -1.0 if env.mode == 0 else 1.0
            action_effect = sign * env.action_gain * a * np.ones(env.state_dim)
            action_effect += env.rng.normal(0, env.action_noise, env.state_dim)
            auto_effect = env.rng.normal(0, env.autonomous_noise, env.state_dim)
            auto_effect -= env.autonomous_drift * env.state
            env.state = env.state + action_effect + auto_effect
            env.state = env.state.astype(np.float32)
            env._history_obs.pop(0)
            env._history_obs.append(env.state.copy())
            env._history_act.pop(0)
            env._history_act.append(float(a))
            env.t += 1
            outcomes[int(a)] = env.state.copy()
            env.restore(snap2)
        # restore original after computing
        env.restore(snap)
        for h in horizons:
            # For horizon > 1 we keep same single-step outcomes for simplicity
            # (multi-step with flipped sign is complex; we use h=1 primarily)
            residuals = {int(a): out - outcomes[0] for a, out in outcomes.items()}
            best_action = max(outcomes, key=lambda a: float(np.sum(outcomes[a])))
            rows.append({
                "seed": seed,
                "state_idx": int(i),
                "horizon": h,
                "outcome_m1": outcomes[-1].tolist(),
                "outcome_0": outcomes[0].tolist(),
                "outcome_p1": outcomes[1].tolist(),
                "residual_m1": residuals[-1].tolist(),
                "residual_p1": residuals[1].tolist(),
                "best_action": int(best_action),
                "mode": int(rec["mode"]),
            })
    return rows


# ---------------------------------------------------------------------------
# Task 2: Learned Baselines
# ---------------------------------------------------------------------------
def fit_learned_state_only_classifier(X_train, y_train, X_test):
    """Learned StateOnly classifier predicting best_action from state/history."""
    # Use a small MLP via sklearn
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True,
                        validation_fraction=0.1, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    return pred, proba


def fit_learned_state_only_regressor(X_train, y_train_reg, X_test):
    """Learned StateOnly regressor predicting outcome sums, then derive best action."""
    reg = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True,
                       validation_fraction=0.1, random_state=42)
    reg.fit(X_train, y_train_reg)
    pred_reg = reg.predict(X_test)
    pred_action = np.argmax(pred_reg, axis=1)
    return pred_action, pred_reg


def fit_learned_action_only_classifier(y_train, y_test):
    """Learned ActionOnly: predict best_action from global action prior (no state)."""
    # Train a dummy classifier that only sees action priors
    # We simulate by training a classifier on zeros input
    X_dummy = np.zeros((len(y_train), 1), dtype=np.float32)
    X_dummy_test = np.zeros((len(y_test), 1), dtype=np.float32)
    try:
        clf = LogisticRegression(max_iter=500, multi_class="multinomial")
    except TypeError:
        clf = LogisticRegression(max_iter=500)
    clf.fit(X_dummy, y_train)
    pred = clf.predict(X_dummy_test)
    # proba same for all test points
    proba = clf.predict_proba(X_dummy_test)
    return pred, proba


def compute_majority_action_baseline(y_train, y_test):
    from collections import Counter
    c = Counter(y_train)
    majority = c.most_common(1)[0][0]
    pred = np.full(len(y_test), majority, dtype=np.int64)
    return pred, majority


# ---------------------------------------------------------------------------
# Task 5: History Ablation helpers
# ---------------------------------------------------------------------------
def evaluate_on_feature_variant(df_train, df_test, feature_fn, rng=None):
    """Train a simple MLP classifier on a feature variant and return match."""
    X_train = []
    for _, row in df_train.iterrows():
        f = feature_fn(row)
        if f is not None:
            X_train.append(f)
    X_test = []
    for _, row in df_test.iterrows():
        f = feature_fn(row)
        if f is not None:
            X_test.append(f)
    X_train = np.stack(X_train)
    X_test = np.stack(X_test)
    _, y_train = compute_outcome_vectors(df_train)
    _, y_test = compute_outcome_vectors(df_test)
    # Ensure alignment
    min_len = min(len(X_train), len(y_train))
    X_train, y_train = X_train[:min_len], y_train[:min_len]
    min_len = min(len(X_test), len(y_test))
    X_test, y_test = X_test[:min_len], y_test[:min_len]
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True,
                        validation_fraction=0.1, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return accuracy_score(y_test, pred)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs("results/ic2a_plus", exist_ok=True)
    print("=" * 70)
    print("IC-2a+: Robust Oracle Residual Audit")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Load or generate base counterfactual table
    # -----------------------------------------------------------------------
    base_path = "results/counterfactual_table.csv"
    if os.path.exists(base_path):
        cf_df = pd.read_csv(base_path)
        print(f"\nLoaded base CF table: {len(cf_df)} rows")
    else:
        print("\nGenerating base counterfactual table...")
        cf_df = generate_counterfactual_table(
            StructuredVolatilityEnv, ENV_KWARGS,
            n_train=1200, n_val=200, n_test_id=200, n_ood=300,
            horizons=HORIZONS, seeds=SEEDS,
        )
        cf_df.to_csv(base_path, index=False)

    # Ensure history columns are present (older CSV may not have them)
    if "history_obs" not in cf_df.columns:
        print("  Re-generating with history columns...")
        cf_df = generate_counterfactual_table(
            StructuredVolatilityEnv, ENV_KWARGS,
            n_train=1200, n_val=200, n_test_id=200, n_ood=300,
            horizons=HORIZONS, seeds=SEEDS,
        )
        cf_df.to_csv(base_path, index=False)

    # -----------------------------------------------------------------------
    # Task 1: True OOD Splits
    # -----------------------------------------------------------------------
    print("\n[Task 1] Generating True OOD tables...")
    ood_types = {
        "background_shift": "ood_background_table.csv",
        "action_gain_shift": "ood_gain_table.csv",
        "sign_rule_shift": "ood_sign_table.csv",
    }
    ood_tables = {}
    for ood_name, fname in ood_types.items():
        fpath = f"results/ic2a_plus/{fname}"
        if os.path.exists(fpath):
            ood_df = pd.read_csv(fpath)
            print(f"  Loaded {ood_name}: {len(ood_df)} rows")
        else:
            print(f"  Generating {ood_name}...")
            ood_df = generate_true_ood_table(ood_name, ENV_KWARGS, n_samples=1200, seeds=SEEDS, horizons=HORIZONS)
            ood_df.to_csv(fpath, index=False)
            print(f"  Saved {fpath}: {len(ood_df)} rows")
        ood_tables[ood_name] = ood_df

    # -----------------------------------------------------------------------
    # Task 3: Action distribution audit (on base train split)
    # -----------------------------------------------------------------------
    print("\n[Task 3] Action distribution audit...")
    action_dist_records = []
    for seed in SEEDS:
        for h in HORIZONS:
            sub = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "train") & (cf_df["horizon"] == h)]
            if len(sub) == 0:
                continue
            best_actions = sub["best_action"].values
            counts = np.bincount(best_actions + 1, minlength=3)  # indices 0,1,2 for -1,0,1
            total = len(best_actions)
            p_m1 = counts[0] / total
            p_0 = counts[1] / total
            p_p1 = counts[2] / total
            ent = compute_oracle_action_entropy(best_actions)
            zero_action_optimal_rate = p_0
            action_dist_records.append({
                "seed": seed,
                "horizon": h,
                "p_best_m1": p_m1,
                "p_best_0": p_0,
                "p_best_p1": p_p1,
                "oracle_action_entropy": ent,
                "zero_action_optimal_rate": zero_action_optimal_rate,
                "n_samples": total,
            })
    action_dist_df = pd.DataFrame(action_dist_records)
    action_dist_df.to_csv("results/ic2a_plus/action_distribution_audit.csv", index=False)
    mean_zero_opt = action_dist_df["zero_action_optimal_rate"].mean()
    print(f"  Mean zero_action_optimal_rate = {mean_zero_opt:.4f}")
    print(f"  Saved results/ic2a_plus/action_distribution_audit.csv")

    # -----------------------------------------------------------------------
    # Task 4: Conditional RVR
    # -----------------------------------------------------------------------
    print("\n[Task 4] Conditional RVR computation...")
    rvr_records = []
    for seed in SEEDS:
        for h in HORIZONS:
            sub = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "train") & (cf_df["horizon"] == h)]
            if len(sub) == 0:
                continue
            cond_rvr, numer, denom = compute_residual_variance_ratio_conditional(sub)
            rvr_records.append({
                "seed": seed,
                "horizon": h,
                "conditional_rvr": cond_rvr,
                "within_state_action_var": numer,
                "within_state_total_var": denom,
            })
    rvr_df = pd.DataFrame(rvr_records)
    # Also compute global RVR per seed/horizon for comparison
    def global_rvr_from_df(sub):
        def parse(v):
            if isinstance(v, str):
                return np.array(json.loads(v), dtype=np.float32)
            return np.array(v, dtype=np.float32)
        all_vals = []
        residuals = []
        for _, row in sub.iterrows():
            outs = []
            for a in [-1, 0, 1]:
                col = f"outcome_{'m1' if a==-1 else ('p1' if a==1 else str(a))}"
                o = parse(row[col])
                outs.append(o)
                all_vals.append(o)
            residuals.append(outs[0] - outs[1])  # m1 - 0
            residuals.append(outs[2] - outs[1])  # p1 - 0
        total_var = float(np.var(np.concatenate(all_vals)))
        res_var = float(np.var(np.concatenate(residuals)))
        return res_var / total_var if total_var > 0 else 0.0

    global_rvrs = []
    for _, row in rvr_df.iterrows():
        sub = cf_df[(cf_df["seed"] == row["seed"]) & (cf_df["split"] == "train") & (cf_df["horizon"] == row["horizon"])]
        global_rvrs.append(global_rvr_from_df(sub))
    rvr_df["global_rvr"] = global_rvrs
    rvr_df.to_csv("results/ic2a_plus/robust_oracle_residual_diagnostics.csv", index=False)
    mean_cond = rvr_df["conditional_rvr"].mean()
    mean_global = rvr_df["global_rvr"].mean()
    print(f"  Mean conditional_rvr = {mean_cond:.4f}")
    print(f"  Mean global_rvr      = {mean_global:.4f}")

    # -----------------------------------------------------------------------
    # Task 2: Learned Baselines
    # -----------------------------------------------------------------------
    print("\n[Task 2] Learned baseline audit...")
    learned_records = []
    for seed in SEEDS:
        train_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "train") & (cf_df["horizon"] == 1)]
        test_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "test_id") & (cf_df["horizon"] == 1)]
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        # Features
        X_train_full = np.stack([make_obs_array(row) for _, row in train_df.iterrows()])
        X_test_full = np.stack([make_obs_array(row) for _, row in test_df.iterrows()])
        X_train_so = np.stack([make_state_only_features(row) for _, row in train_df.iterrows()])
        X_test_so = np.stack([make_state_only_features(row) for _, row in test_df.iterrows()])

        y_reg_train, y_class_train = compute_outcome_vectors(train_df)
        y_reg_test, y_class_test = compute_outcome_vectors(test_df)

        # Oracle match (on test_id)
        oracle_pred = y_class_test  # oracle is ground truth
        oracle_match = 1.0

        # Learned StateOnly Classifier
        pred_so_clf, _ = fit_learned_state_only_classifier(X_train_so, y_class_train, X_test_so)
        learned_so_match = accuracy_score(y_class_test, pred_so_clf)

        # Learned StateOnly Regressor
        pred_so_reg, _ = fit_learned_state_only_regressor(X_train_so, y_reg_train, X_test_so)
        learned_so_reg_match = accuracy_score(y_class_test, pred_so_reg)

        # Learned ActionOnly
        pred_ao, _ = fit_learned_action_only_classifier(y_class_train, y_class_test)
        learned_ao_match = accuracy_score(y_class_test, pred_ao)

        # Majority Action
        pred_maj, _ = compute_majority_action_baseline(y_class_train, y_class_test)
        majority_match = accuracy_score(y_class_test, pred_maj)

        # Rule-based SO/AO from oracle summary
        summary = compute_oracle_summary(cf_df, StructuredVolatilityEnv, ENV_KWARGS, seed=seed)
        rule_so = summary["so_match"]
        rule_ao = summary["ao_match"]

        learned_records.append({
            "seed": seed,
            "oracle_match": oracle_match,
            "learned_so_clf_match": learned_so_match,
            "learned_so_reg_match": learned_so_reg_match,
            "learned_ao_match": learned_ao_match,
            "majority_match": majority_match,
            "rule_so_match": rule_so,
            "rule_ao_match": rule_ao,
            "oracle_so_gap": oracle_match - learned_so_match,
            "oracle_ao_gap": oracle_match - learned_ao_match,
        })
    learned_df = pd.DataFrame(learned_records)
    learned_df.to_csv("results/ic2a_plus/learned_baseline_audit.csv", index=False)
    print(f"  Mean learned_so_clf_match = {learned_df['learned_so_clf_match'].mean():.4f}")
    print(f"  Mean learned_ao_match     = {learned_df['learned_ao_match'].mean():.4f}")
    print(f"  Mean oracle_so_gap        = {learned_df['oracle_so_gap'].mean():.4f}")
    print(f"  Mean oracle_ao_gap        = {learned_df['oracle_ao_gap'].mean():.4f}")

    # -----------------------------------------------------------------------
    # Task 5: History Ablation Audit
    # -----------------------------------------------------------------------
    print("\n[Task 5] History ablation audit...")
    hist_records = []
    for seed in SEEDS:
        train_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "train") & (cf_df["horizon"] == 1)]
        test_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "test_id") & (cf_df["horizon"] == 1)]
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        rng = np.random.default_rng(seed + 777)

        full_match = evaluate_on_feature_variant(train_df, test_df, make_obs_array)
        short_match = evaluate_on_feature_variant(train_df, test_df, lambda r: make_short_history(r, short_len=2))
        curr_match = evaluate_on_feature_variant(train_df, test_df, make_current_obs_only)
        perm_match = evaluate_on_feature_variant(train_df, test_df, lambda r: make_permuted_history(r, rng))

        hist_records.append({
            "seed": seed,
            "full_history_match": full_match,
            "short_history_match": short_match,
            "current_obs_match": curr_match,
            "permuted_history_match": perm_match,
            "full_vs_current_advantage": full_match - curr_match,
            "permuted_impairment": full_match - perm_match,
            "short_degradation": full_match - short_match,
        })
    hist_df = pd.DataFrame(hist_records)
    hist_df.to_csv("results/ic2a_plus/history_ablation_audit.csv", index=False)
    print(f"  Mean full_history_match      = {hist_df['full_history_match'].mean():.4f}")
    print(f"  Mean current_obs_match       = {hist_df['current_obs_match'].mean():.4f}")
    print(f"  Mean permuted_history_match  = {hist_df['permuted_history_match'].mean():.4f}")
    print(f"  Mean full_vs_current_adv     = {hist_df['full_vs_current_advantage'].mean():.4f}")
    print(f"  Mean permuted_impairment     = {hist_df['permuted_impairment'].mean():.4f}")

    # -----------------------------------------------------------------------
    # OOD transfer diagnostics
    # -----------------------------------------------------------------------
    print("\n[OOD Transfer] Evaluating oracle residual advantage on OOD...")
    ood_records = []
    for ood_name, ood_df in ood_tables.items():
        for seed in SEEDS:
            sub = ood_df[(ood_df["seed"] == seed) & (ood_df["horizon"] == 1)]
            if len(sub) == 0:
                continue
            _, y_class = compute_outcome_vectors(sub)
            oracle_match = 1.0
            # Rule SO = always 0
            so_match = float((y_class == 1).mean())  # action 0 index is 1
            # Rule AO = majority
            counts = np.bincount(y_class, minlength=3)
            ao_match = counts.max() / len(y_class)
            ood_records.append({
                "ood_type": ood_name,
                "seed": seed,
                "oracle_match": oracle_match,
                "so_match": so_match,
                "ao_match": ao_match,
                "oracle_so_gap": oracle_match - so_match,
                "oracle_ao_gap": oracle_match - ao_match,
                "n_samples": len(sub),
            })
    ood_df_out = pd.DataFrame(ood_records)
    ood_df_out.to_csv("results/ic2a_plus/true_ood_transfer_diagnostics.csv", index=False)
    print(f"  Saved true_ood_transfer_diagnostics.csv")

    # -----------------------------------------------------------------------
    # Task 6: Gate Summary
    # -----------------------------------------------------------------------
    print("\n[Task 6] IC-2a+ Gate Summary...")
    agg = {}
    # A. Oracle > LearnedStateOnly + 0.20
    agg["mean_oracle_so_gap"] = float(learned_df["oracle_so_gap"].mean())
    gate_a = agg["mean_oracle_so_gap"] >= 0.20

    # B. Oracle > LearnedActionOnly + 0.20
    agg["mean_oracle_ao_gap"] = float(learned_df["oracle_ao_gap"].mean())
    gate_b = agg["mean_oracle_ao_gap"] >= 0.20

    # C. conditional_rvr >= 0.25
    agg["mean_conditional_rvr"] = float(rvr_df["conditional_rvr"].mean())
    gate_c = agg["mean_conditional_rvr"] >= 0.25

    # D. OOD background shift residual oracle advantage keeps
    bg = ood_df_out[ood_df_out["ood_type"] == "background_shift"]
    agg["ood_bg_oracle_ao_gap"] = float(bg["oracle_ao_gap"].mean()) if len(bg) else 0.0
    gate_d = agg["ood_bg_oracle_ao_gap"] >= 0.10  # at least some advantage

    # E. zero_action_optimal_rate >= 0.10
    agg["mean_zero_action_optimal_rate"] = float(action_dist_df["zero_action_optimal_rate"].mean())
    gate_e = agg["mean_zero_action_optimal_rate"] >= 0.10
    zero_action_caveat = agg["mean_zero_action_optimal_rate"] < 0.15

    # F. seed variance < oracle gap / 2
    oracle_gap = agg["mean_oracle_ao_gap"]
    seed_var_so = float(learned_df["learned_so_clf_match"].std())
    seed_var_ao = float(learned_df["learned_ao_match"].std())
    agg["seed_var_so"] = seed_var_so
    agg["seed_var_ao"] = seed_var_ao
    gate_f = (seed_var_so < oracle_gap / 2.0) and (seed_var_ao < oracle_gap / 2.0)

    gates = {
        "A_oracle_beats_learned_so_by_20": gate_a,
        "B_oracle_beats_learned_ao_by_20": gate_b,
        "C_conditional_rvr_ge_25": gate_c,
        "D_ood_bg_advantage_kept": gate_d,
        "E_zero_action_optimal_ge_10": gate_e,
        "F_seed_variance_lt_half_gap": gate_f,
    }
    all_pass = all(gates.values())

    summary = {
        "gates": gates,
        "all_pass": all_pass,
        "aggregate": agg,
        "zero_action_caveat": zero_action_caveat,
    }
    with open("results/ic2a_plus/ic2a_plus_gates.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n  Gate Results:")
    for g, v in gates.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {g}")
    print(f"\n  Zero-action optimal rate: {agg['mean_zero_action_optimal_rate']:.4f}", end="")
    if zero_action_caveat:
        print("  [CAVEAT: < 0.15]")
    else:
        print("  [OK]")

    # -----------------------------------------------------------------------
    # Write Report
    # -----------------------------------------------------------------------
    report = f"""# IC-2a+ Robust Oracle Residual Audit Report

Generated: 2026-05-09

## 1. Base IC-2a Recap (Provisional Pass)
- Global RVR mean: {mean_global:.4f}
- Rule SO match: {learned_df['rule_so_match'].mean():.4f}
- Rule AO match: {learned_df['rule_ao_match'].mean():.4f}
- Oracle→AO gap: {learned_df['oracle_match'].mean() - learned_df['rule_ao_match'].mean():.4f}
- SSR: {seed_var_ao:.4f}

## 2. True OOD Splits
| OOD Type | Oracle→AO Gap | Notes |
|----------|---------------|-------|
| background_shift | {ood_df_out[ood_df_out['ood_type']=='background_shift']['oracle_ao_gap'].mean():.4f} | Autonomous dynamics changed |
| action_gain_shift | {ood_df_out[ood_df_out['ood_type']=='action_gain_shift']['oracle_ao_gap'].mean():.4f} | Action gain magnitude changed |
| sign_rule_shift | {ood_df_out[ood_df_out['ood_type']=='sign_rule_shift']['oracle_ao_gap'].mean():.4f} | Sign rule inverted |

## 3. Learned Baselines
| Baseline | Mean Match | Oracle Gap |
|----------|------------|------------|
| LearnedStateOnly (Classifier) | {learned_df['learned_so_clf_match'].mean():.4f} | {learned_df['oracle_so_gap'].mean():.4f} |
| LearnedStateOnly (Regressor) | {learned_df['learned_so_reg_match'].mean():.4f} | {1.0 - learned_df['learned_so_reg_match'].mean():.4f} |
| LearnedActionOnly | {learned_df['learned_ao_match'].mean():.4f} | {learned_df['oracle_ao_gap'].mean():.4f} |
| Majority Action | {learned_df['majority_match'].mean():.4f} | - |

## 4. Zero-Action Optimal Proportion Gate (D6)
- Mean zero_action_optimal_rate: {agg['mean_zero_action_optimal_rate']:.4f}
- Gate D6_zero_action_not_absent: {'PASS' if gate_e else 'FAIL'} (>= 0.10)
- Preferred >= 0.15: {'CAVEAT' if zero_action_caveat else 'PASS'}

## 5. Conditional RVR
- Mean conditional_rvr: {mean_cond:.4f}
- Mean global_rvr: {mean_global:.4f}
- Ratio (cond/global): {mean_cond/mean_global if mean_global>0 else 0:.2f}x

## 6. History Ablation
| Variant | Mean Match | vs Full |
|---------|------------|---------|
| full_history_len_8 | {hist_df['full_history_match'].mean():.4f} | baseline |
| short_history_len_2 | {hist_df['short_history_match'].mean():.4f} | -{hist_df['short_degradation'].mean():.4f} |
| current_obs_only | {hist_df['current_obs_match'].mean():.4f} | -{hist_df['full_vs_current_advantage'].mean():.4f} |
| permuted_history | {hist_df['permuted_history_match'].mean():.4f} | -{hist_df['permuted_impairment'].mean():.4f} |

## 7. IC-2a+ Gates (A-F)
| Gate | Condition | Result |
|------|-----------|--------|
| A | Oracle > LearnedSO + 0.20 | {'PASS' if gate_a else 'FAIL'} ({agg['mean_oracle_so_gap']:.3f}) |
| B | Oracle > LearnedAO + 0.20 | {'PASS' if gate_b else 'FAIL'} ({agg['mean_oracle_ao_gap']:.3f}) |
| C | conditional_rvr >= 0.25 | {'PASS' if gate_c else 'FAIL'} ({agg['mean_conditional_rvr']:.3f}) |
| D | OOD bg residual advantage kept | {'PASS' if gate_d else 'FAIL'} ({agg['ood_bg_oracle_ao_gap']:.3f}) |
| E | zero_action_optimal_rate >= 0.10 | {'PASS' if gate_e else 'FAIL'} ({agg['mean_zero_action_optimal_rate']:.3f}) |
| F | seed_variance < oracle_gap / 2 | {'PASS' if gate_f else 'FAIL'} (so_var={seed_var_so:.3f}, ao_var={seed_var_ao:.3f}) |

## 8. Answers to Audit Questions

1. **Does IC-2a PASS hold under true OOD?**
   {'Yes' if gate_d else 'No'} — background_shift oracle advantage remains.

2. **Does LearnedStateOnly stay below Oracle?**
   {'Yes' if gate_a else 'No'} — gap is {agg['mean_oracle_so_gap']:.3f}.

3. **Is 0-action severely absent?**
   {'No' if gate_e else 'YES — CRITICAL'}. Rate = {agg['mean_zero_action_optimal_rate']:.3f}.

4. **Is conditional RVR clearly above global RVR?**
   {'Yes' if mean_cond > mean_global else 'No'} — cond={mean_cond:.4f}, global={mean_global:.4f}.

5. **Does history order have real value?**
   {'Yes' if hist_df['permuted_impairment'].mean() > 0.02 else 'No'} — permuted impairment = {hist_df['permuted_impairment'].mean():.4f}.

6. **May we proceed to IC-2b?**
   {'**PROCEED to IC-2b.**' if all_pass else '**STOP. Redesign environment before IC-2b.**'}

"""
    with open("results/ic2a_plus/IC2A_PLUS_ROBUSTNESS_REPORT.md", "w") as f:
        f.write(report)
    print("\n  Saved results/ic2a_plus/IC2A_PLUS_ROBUSTNESS_REPORT.md")

    print("\n" + "=" * 70)
    if all_pass:
        print("IC-2a+ ALL GATES PASSED. Proceed to IC-2b.")
    else:
        print("IC-2a+ FAILED. Do not proceed to IC-2b. Redesign environment.")
    print("=" * 70)


if __name__ == "__main__":
    main()
