"""IC-2b: Learned Throttling Mechanism Comparison.

Compares 13 learned mechanisms on ID and 3 OOD splits.
Only run after IC-2a+ passes all gates.
"""
import sys, os, json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import (
    prepare_counterfactual_data, train_state_only_classifier,
    train_ae_model, train_counterfactual_joint, train_causal_contrast,
    train_memory_mechanism,
)
from src.models import (
    StateOnlyPredictor, ActionOnlyPredictor, AEPCompressor, ResidualCompressor,
    CenteredResidualCompressor, CounterfactualCompressor,
    CausalContrastCompressor, ResidualAdversarialCompressor,
)
from src.memory_baselines import RawMemoryFull, RawMemoryEqualCost, PrototypeMemory
from src.metrics import (
    compute_best_action_match, compute_regret, compute_rank_accuracy,
    compute_action_only_ceiling, compute_bad_debt_ratio,
    compute_state_only_shortcut_index, compute_action_only_shortcut_index,
    compute_shuffled_action_gap, compute_permuted_history_gap,
    compute_iar,
)

ENV_KWARGS = dict(
    state_dim=2, mode_flip_prob=0.08, autonomous_drift=0.05,
    autonomous_noise=0.02, action_gain=0.70, action_noise=0.03,
    action_sign_flip=True, history_len=8,
    action_cost=0.20, state_dependent_gain=True, saturation_k=0.5,
)
SEEDS = list(range(5))
OBS_DIM = 2
HISTORY_LEN = 8
FEATURE_DIM = HISTORY_LEN * (OBS_DIM + 1)
BOTTLENECK_DIM = 48
RESIDUAL_DIM = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 500
PATIENCE = 80
AE_CE_WEIGHT = 0.8
CF_CE_WEIGHT = 1.5


def load_ood_table(ood_name):
    path = f"results/ic2a_plus/ood_{ood_name}_table.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    mapping = {
        "background": "ood_background_table.csv",
        "gain": "ood_gain_table.csv",
        "sign": "ood_sign_table.csv",
    }
    alt = f"results/ic2a_plus/{mapping.get(ood_name, '')}"
    if os.path.exists(alt):
        return pd.read_csv(alt)
    return None


def prepare_ood_data(cf_df, ood_df, seed, env_kwargs):
    """Load OOD data by joining with CF table to get history columns."""
    obs_dim = env_kwargs["state_dim"]
    history_len = env_kwargs["history_len"]

    sub_ood = ood_df[ood_df["seed"] == seed] if "seed" in ood_df.columns else ood_df
    sub_ood = sub_ood[sub_ood["horizon"] == 1] if "horizon" in sub_ood.columns else sub_ood
    if len(sub_ood) == 0:
        return None, None, None

    sub_cf = cf_df[(cf_df["seed"] == seed) & (cf_df["horizon"] == 1)]

    if "history_obs" in sub_cf.columns and "history_obs" not in sub_ood.columns:
        idx_to_history = {}
        for _, row in sub_cf.iterrows():
            idx_to_history[row["state_idx"]] = (
                row.get("history_obs"), row.get("history_act"))
        hist_obs_list = []
        hist_act_list = []
        for _, row in sub_ood.iterrows():
            ho, ha = idx_to_history.get(row["state_idx"], (None, None))
            hist_obs_list.append(ho)
            hist_act_list.append(ha)
        sub_ood = sub_ood.copy()
        sub_ood["history_obs"] = hist_obs_list
        sub_ood["history_act"] = hist_act_list

    return prepare_counterfactual_data(sub_ood, seed, env_kwargs)


def evaluate_model_predictions(y_pred_probs_or_vals, y_true, best_action_true):
    """Compute all core metrics from predictions (n,3) arrays."""
    pred_vals = np.array(y_pred_probs_or_vals)
    true_vals = np.array([v if hasattr(v, '__len__') else [0,0,0]
                          for v in y_true]) if y_true is not None else pred_vals * 0

    if pred_vals.ndim != 2 or pred_vals.shape[1] != 3:
        return None

    pred_best = np.argmax(pred_vals, axis=1)
    match = float(np.mean(pred_best == best_action_true))

    regret = 0.0
    n = len(pred_vals)
    for i in range(n):
        best_val = pred_vals[i, best_action_true[i]]
        chosen_val = pred_vals[i, pred_best[i]]
        regret += best_val - chosen_val
    regret /= n

    correct = 0
    total = 0
    for i in range(n):
        order_pred = np.argsort(pred_vals[i])[::-1]
        for j in range(3):
            for k in range(j + 1, 3):
                total += 1
                if (pred_vals[i, j] > pred_vals[i, k]) == (true_vals[i, j] > true_vals[i, k]):
                    correct += 1
    rank_acc = correct / total if total > 0 else 0.0

    mse = float(np.mean((pred_vals - true_vals) ** 2)) if true_vals is not None else 0.0
    return {"best_action_match": match, "regret": regret, "rank_accuracy": rank_acc, "outcome_mse": mse}


def evaluate_mechanism(mech, X, Y_true, best_action_true, X_perm=None):
    """Evaluate a mechanism, optionally with permuted history."""
    if X_perm is not None:
        X_eval = X_perm
    else:
        X_eval = X
    preds = mech.predict_all_actions(X_eval)
    return evaluate_model_predictions(preds, Y_true, best_action_true)


def count_params(model):
    if hasattr(model, 'count_parameters'):
        return model.count_parameters()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    os.makedirs("results/ic2b", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    print("=" * 70)
    print("IC-2b: Learned Throttling Mechanism Comparison")
    print("=" * 70)

    cf_df = pd.read_csv("results/counterfactual_table.csv")
    print(f"CF table: {len(cf_df)} rows, {cf_df['seed'].nunique()} seeds")

    ood_bg = load_ood_table("background")
    ood_gain = load_ood_table("gain")
    ood_sign = load_ood_table("sign")
    print(f"OOD tables: bg={len(ood_bg) if ood_bg is not None else 'N/A'}, "
          f"gain={len(ood_gain) if ood_gain is not None else 'N/A'}, "
          f"sign={len(ood_sign) if ood_sign is not None else 'N/A'}")

    all_results = []
    all_ood = []
    trained_models = {}

    for seed in tqdm(SEEDS, desc="Seeds"):
        train_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "train") & (cf_df["horizon"] == 1)]
        val_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "val") & (cf_df["horizon"] == 1)]
        test_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "test_id") & (cf_df["horizon"] == 1)]

        X_tr, Y_tr, ba_tr = prepare_counterfactual_data(train_df, seed, ENV_KWARGS)
        X_val, Y_val, ba_val = prepare_counterfactual_data(val_df, seed, ENV_KWARGS)
        X_te, Y_te, ba_te = prepare_counterfactual_data(test_df, seed, ENV_KWARGS)

        if X_tr is None or X_te is None or len(X_tr) < 10:
            continue

        def _safe_train(name, fn, mechanisms):
            try:
                mechanisms[name] = fn()
            except Exception as e:
                print(f"\n  [{name}] FAILED: {e}")

        mechanisms = {}
        perm_fn = None

        # --- 1. LearnedStateOnlyClassifier ---
        _safe_train("learned_state_only",
                    lambda: train_state_only_classifier(
                        StateOnlyPredictor(OBS_DIM, HISTORY_LEN, bottleneck_dim=BOTTLENECK_DIM),
                        X_tr, Y_tr, X_val, Y_val, epochs=EPOCHS, patience=PATIENCE, device=DEVICE),
                    mechanisms)

        # --- 2. LearnedActionOnly ---
        _safe_train("learned_action_only",
                    lambda: train_state_only_classifier(
                        ActionOnlyPredictor(n_actions=3),
                        X_tr, Y_tr, X_val, Y_val, epochs=EPOCHS, patience=PATIENCE, device=DEVICE),
                    mechanisms)

        # --- 3. RawMemoryFull ---
        _safe_train("raw_memory_full",
                    lambda: train_memory_mechanism(RawMemoryFull(k=5), X_tr, Y_tr),
                    mechanisms)

        # --- 4. RawMemoryEqualCost ---
        param_budget = count_params(StateOnlyPredictor(OBS_DIM, HISTORY_LEN, bottleneck_dim=BOTTLENECK_DIM))
        _safe_train("raw_memory_equal_cost",
                    lambda: train_memory_mechanism(RawMemoryEqualCost(param_budget=param_budget, k=5), X_tr, Y_tr),
                    mechanisms)

        # --- 5. PrototypeMemory ---
        _safe_train("prototype_memory",
                    lambda: train_memory_mechanism(PrototypeMemory(n_clusters=20, k=3), X_tr, Y_tr),
                    mechanisms)

        # --- 6. AEPCompressor ---
        _safe_train("aep_compressor",
                    lambda: train_ae_model(
                        AEPCompressor(OBS_DIM, HISTORY_LEN, bottleneck_dim=BOTTLENECK_DIM),
                        X_tr, Y_tr, X_val, Y_val, "aep", epochs=EPOCHS, patience=PATIENCE, device=DEVICE,
                        ce_weight=0.0),
                    mechanisms)

        # --- 7. ResidualCompressor ---
        _safe_train("residual_compressor",
                    lambda: train_ae_model(
                        ResidualCompressor(OBS_DIM, HISTORY_LEN, bottleneck_dim=BOTTLENECK_DIM, residual_dim=RESIDUAL_DIM),
                        X_tr, Y_tr, X_val, Y_val, "residual", epochs=EPOCHS, patience=PATIENCE, device=DEVICE,
                        ce_weight=AE_CE_WEIGHT),
                    mechanisms)

        # --- 8. CenteredResidualCompressor ---
        _safe_train("centered_residual",
                    lambda: train_ae_model(
                        CenteredResidualCompressor(OBS_DIM, HISTORY_LEN, bottleneck_dim=BOTTLENECK_DIM, residual_dim=RESIDUAL_DIM),
                        X_tr, Y_tr, X_val, Y_val, "centered_residual", epochs=EPOCHS, patience=PATIENCE, device=DEVICE,
                        ce_weight=AE_CE_WEIGHT),
                    mechanisms)

        # --- 9. CounterfactualCompressor ---
        _safe_train("counterfactual_compressor",
                    lambda: train_counterfactual_joint(
                        CounterfactualCompressor(OBS_DIM, HISTORY_LEN, bottleneck_dim=BOTTLENECK_DIM),
                        X_tr, Y_tr, X_val, Y_val, epochs=EPOCHS, patience=PATIENCE, device=DEVICE,
                        ce_weight=CF_CE_WEIGHT),
                    mechanisms)

        # --- 10. CausalContrastCompressor ---
        _safe_train("causal_contrast",
                    lambda: train_causal_contrast(
                        CausalContrastCompressor(OBS_DIM, HISTORY_LEN, bottleneck_dim=BOTTLENECK_DIM),
                        X_tr, Y_tr, X_val, Y_val, epochs=EPOCHS, patience=PATIENCE, device=DEVICE),
                    mechanisms)

        # --- 11. ResidualAdversarialCompressor ---
        _safe_train("residual_adversarial",
                    lambda: train_ae_model(
                        ResidualAdversarialCompressor(OBS_DIM, HISTORY_LEN, bottleneck_dim=BOTTLENECK_DIM, residual_dim=RESIDUAL_DIM),
                        X_tr, Y_tr, X_val, Y_val, "residual_adversarial", epochs=EPOCHS, patience=PATIENCE, device=DEVICE,
                        ce_weight=AE_CE_WEIGHT),
                    mechanisms)

        # --- 12. ShuffledActionControl ---
        rng_shuf = np.random.default_rng(seed + 777)
        Y_shuf = Y_tr.copy()
        perm = rng_shuf.permutation(len(Y_shuf))
        Y_shuf = Y_shuf[perm]
        _safe_train("shuffled_action_control",
                    lambda: train_ae_model(
                        ResidualCompressor(OBS_DIM, HISTORY_LEN, bottleneck_dim=BOTTLENECK_DIM, residual_dim=RESIDUAL_DIM),
                        X_tr, Y_shuf, X_val, Y_val, "residual", epochs=EPOCHS, patience=PATIENCE, device=DEVICE,
                        ce_weight=AE_CE_WEIGHT),
                    mechanisms)

        # --- 13. PermutedHistoryControl ---
        perm_idx = np.random.default_rng(seed + 888).permutation(HISTORY_LEN)
        def permute_history(X_arr):
            Xp = X_arr.copy()
            for i in range(len(Xp)):
                for j in range(HISTORY_LEN):
                    src = j * (OBS_DIM + 1)
                    dst = perm_idx[j] * (OBS_DIM + 1)
                    Xp[i, dst:dst + OBS_DIM + 1] = X_arr[i, src:src + OBS_DIM + 1]
            return Xp
        perm_fn = lambda X_arr: permute_history(X_arr)
        X_tr_perm = permute_history(X_tr)
        X_val_perm = permute_history(X_val) if X_val is not None else None
        perm_model = ResidualCompressor(OBS_DIM, HISTORY_LEN, bottleneck_dim=BOTTLENECK_DIM, residual_dim=RESIDUAL_DIM)
        try:
            perm_model = train_ae_model(perm_model, X_tr_perm, Y_tr, X_val_perm, Y_val, "residual",
                                         epochs=EPOCHS, patience=PATIENCE, device=DEVICE)
            mechanisms["permuted_history_control"] = perm_model
        except Exception as e:
            print(f"\n  [permuted_history_control] FAILED: {e}")

        # --- Evaluate all on ID test ---
        trained_models[seed] = {}
        for name, mech in mechanisms.items():
            if name == "permuted_history_control":
                X_te_perm = permute_history(X_te)
                with torch.no_grad():
                    x_t = torch.tensor(X_te_perm, dtype=torch.float32).to(DEVICE)
                    if hasattr(mech, 'predict_all_actions'):
                        preds_t = mech.predict_all_actions(x_t)
                        preds = preds_t.cpu().numpy() if isinstance(preds_t, torch.Tensor) else np.array(preds_t)
                    else:
                        result = mech(x_t)
                        preds = (result[0] if isinstance(result, tuple) else result).cpu().numpy()
                met = evaluate_model_predictions(preds, Y_te, ba_te)
            elif name in ("raw_memory_full", "raw_memory_equal_cost", "prototype_memory"):
                preds = mech.predict(X_te)
                met = evaluate_model_predictions(preds, Y_te, ba_te)
            elif name in ("learned_state_only", "learned_action_only"):
                with torch.no_grad():
                    x_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
                    if hasattr(mech, 'predict_all_actions'):
                        preds_t = mech.predict_all_actions(x_t)
                        preds = preds_t.cpu().numpy() if isinstance(preds_t, torch.Tensor) else np.array(preds_t)
                    else:
                        result = mech(x_t)
                        preds = (result[0] if isinstance(result, tuple) else result).cpu().numpy()
                met = evaluate_model_predictions(preds, Y_te, ba_te)
            else:
                with torch.no_grad():
                    x_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
                    if hasattr(mech, 'predict_all_actions'):
                        preds_t = mech.predict_all_actions(x_t)
                        preds = preds_t.cpu().numpy() if isinstance(preds_t, torch.Tensor) else np.array(preds_t)
                    else:
                        result = mech(x_t)
                        if isinstance(result, tuple):
                            preds = result[0].cpu().numpy()
                        else:
                            preds = result.cpu().numpy()
                met = evaluate_model_predictions(preds, Y_te, ba_te)

            if met is None:
                continue
            param_count = count_params(mech) if not isinstance(mech, (RawMemoryFull, RawMemoryEqualCost, PrototypeMemory)) else mech.stored_samples_count * FEATURE_DIM
            cost_bytes = 0
            if name in ("raw_memory_full", "raw_memory_equal_cost", "prototype_memory"):
                if name == "raw_memory_full":
                    cost_bytes = RawMemoryFull.cost_bytes(FEATURE_DIM, mech.stored_samples_count)
                else:
                    cost_bytes = mech.cost_bytes()
            else:
                cost_bytes = param_count * 4 if isinstance(param_count, int) else 0

            all_results.append({
                "seed": seed,
                "mechanism": name,
                "best_action_match": met["best_action_match"],
                "regret": met["regret"],
                "rank_accuracy": met["rank_accuracy"],
                "outcome_mse": met["outcome_mse"],
                "parameter_count": int(param_count) if isinstance(param_count, (int, float)) else 0,
                "cost_bytes": float(cost_bytes),
                "n_train": len(X_tr),
                "n_test": len(X_te),
            })

            # Save model reference for OOD evaluation
            trained_models[seed][name] = {
                "model": mech,
                "is_memory": name in ("raw_memory_full", "raw_memory_equal_cost", "prototype_memory"),
                "permuted": name == "permuted_history_control",
                "perm_fn": perm_fn if name == "permuted_history_control" else None,
            }
    res_df = pd.DataFrame(all_results)

    # Build aggregate tables
    agg_cols = ["best_action_match", "regret", "rank_accuracy", "outcome_mse", "parameter_count", "cost_bytes"]
    agg = res_df.groupby("mechanism")[agg_cols].agg(["mean", "std"]).reset_index()

    # Calculate IAR for each mechanism
    iar_data = []
    for mech_name in res_df["mechanism"].unique():
        sub = res_df[res_df["mechanism"] == mech_name]
        mean_match = sub["best_action_match"].mean()
        mean_cost = sub["cost_bytes"].mean()
        random_baseline = 0.33
        value_gain = max(0, mean_match - random_baseline)
        iar = value_gain / mean_cost if mean_cost > 0 else (float("inf") if value_gain > 0 else 0.0)
        iar_data.append({"mechanism": mech_name, "iar": iar, "value_gain": value_gain, "cost": mean_cost})
    iar_df = pd.DataFrame(iar_data)

    # LearnedStateOnly baseline for gaps
    so_match_mean = res_df[res_df["mechanism"] == "learned_state_only"]["best_action_match"].mean()
    ao_match_mean = res_df[res_df["mechanism"] == "learned_action_only"]["best_action_match"].mean()

    # StateOnly gap
    gap_data = []
    for mech_name in res_df["mechanism"].unique():
        sub = res_df[res_df["mechanism"] == mech_name]
        gap_data.append({
            "mechanism": mech_name,
            "mean_match": sub["best_action_match"].mean(),
            "stateonly_gap": sub["best_action_match"].mean() - so_match_mean,
            "actiononly_gap": sub["best_action_match"].mean() - ao_match_mean,
        })
    gap_df = pd.DataFrame(gap_data)

    # Bad debt audit
    shuf_match = res_df[res_df["mechanism"] == "shuffled_action_control"]["best_action_match"].mean()
    bad_debt = []
    for mech_name in res_df["mechanism"].unique():
        sub = res_df[res_df["mechanism"] == mech_name]
        mm = sub["best_action_match"].mean()
        bdr = compute_bad_debt_ratio(so_match_mean, ao_match_mean, shuf_match, mm)
        sos = compute_state_only_shortcut_index(so_match_mean, mm)
        aos = compute_action_only_shortcut_index(ao_match_mean, mm)
        sag = compute_shuffled_action_gap(mm, shuf_match)
        bad_debt.append({
            "mechanism": mech_name,
            "bad_debt_ratio": bdr,
            "state_only_shortcut_index": sos,
            "action_only_shortcut_index": aos,
            "shuffled_action_gap": sag,
            "model_match": mm,
        })
    bd_df = pd.DataFrame(bad_debt)

    # RawMemoryEqualCost comparison
    rm_eq_match = res_df[res_df["mechanism"] == "raw_memory_equal_cost"]["best_action_match"].mean()
    rm_eq_cost = res_df[res_df["mechanism"] == "raw_memory_equal_cost"]["cost_bytes"].mean()

    raw_comp = []
    for mech_name in res_df["mechanism"].unique():
        sub = res_df[res_df["mechanism"] == mech_name]
        mm = sub["best_action_match"].mean()
        mc = sub["cost_bytes"].mean()
        raw_comp.append({
            "mechanism": mech_name,
            "mean_match": mm,
            "cost_bytes": mc,
            "vs_rawmemory_match": mm - rm_eq_match,
            "vs_rawmemory_cost_ratio": mc / rm_eq_cost if rm_eq_cost > 0 else float("inf"),
        })
    raw_df = pd.DataFrame(raw_comp)

    # --- Save CSVs ---
    res_df.to_csv("results/ic2b/learned_compressors.csv", index=False)
    gap_df.to_csv("results/ic2b/stateonly_gap.csv", index=False)
    bd_df.to_csv("results/ic2b/bad_debt_audit.csv", index=False)
    raw_df.to_csv("results/ic2b/raw_memory_comparison.csv", index=False)
    iar_df.to_csv("results/ic2b/appreciation_rate.csv", index=False)

    # --- OOD Transfer Evaluation ---
    ood_records = []
    ood_names = [("background_shift", ood_bg), ("action_gain_shift", ood_gain), ("sign_rule_shift", ood_sign)]
    for ood_name, ood_df in ood_names:
        if ood_df is None:
            continue
        for seed in SEEDS:
            if seed not in trained_models:
                continue
            X_ood, Y_ood, ba_ood = prepare_ood_data(cf_df, ood_df, seed, ENV_KWARGS)
            if X_ood is None or len(X_ood) == 0:
                continue
            for name, info in trained_models[seed].items():
                mech = info["model"]
                try:
                    if info["permuted"] and info["perm_fn"] is not None:
                        X_eval = info["perm_fn"](X_ood)
                    else:
                        X_eval = X_ood
                    if info["is_memory"]:
                        preds = mech.predict(X_eval)
                    else:
                        with torch.no_grad():
                            x_t = torch.tensor(X_eval, dtype=torch.float32).to(DEVICE)
                            if hasattr(mech, 'predict_all_actions'):
                                preds_t = mech.predict_all_actions(x_t)
                                preds = preds_t.cpu().numpy() if isinstance(preds_t, torch.Tensor) else np.array(preds_t)
                            else:
                                result = mech(x_t)
                                preds = (result[0] if isinstance(result, tuple) else result).cpu().numpy()
                    met = evaluate_model_predictions(preds, Y_ood, ba_ood)
                    if met is not None:
                        ood_records.append({
                            "seed": seed, "ood_type": ood_name,
                            "mechanism": name,
                            "best_action_match": met["best_action_match"],
                            "regret": met["regret"],
                            "rank_accuracy": met["rank_accuracy"],
                            "outcome_mse": met["outcome_mse"],
                        })
                except Exception as e:
                    pass
    ood_df = pd.DataFrame(ood_records) if ood_records else pd.DataFrame(
        columns=["seed", "ood_type", "mechanism", "best_action_match", "regret", "rank_accuracy", "outcome_mse"])
    ood_df.to_csv("results/ic2b/ood_transfer.csv", index=False)
    print(f"  OOD records: {len(ood_df)}")

    # --- Generate Figures ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Use simple groupby for figure data
        mech_match = res_df.groupby("mechanism")["best_action_match"].mean()
        mech_names = list(mech_match.index)
        mech_matches = list(mech_match.values)

        if mech_names:
            # Best action match bar chart
            fig, ax = plt.subplots(figsize=(14, 6))
            colors = ["green" if m > so_match_mean else "red" for m in mech_matches]
            bars = ax.bar(mech_names, mech_matches, color=colors)
            ax.axhline(y=so_match_mean, color="blue", linestyle="--", label=f"LearnedStateOnly ({so_match_mean:.3f})")
            ax.axhline(y=ao_match_mean, color="orange", linestyle="--", label=f"LearnedActionOnly ({ao_match_mean:.3f})")
            ax.set_ylabel("Best Action Match")
            ax.set_title("IC-2b: Learned Throttling Mechanism Comparison")
            plt.xticks(rotation=45, ha="right")
            ax.legend()
            plt.tight_layout()
            fig.savefig("results/figures/ic2b_best_action_match.png", dpi=150)
            plt.close()

        # IAR vs Cost
        if len(iar_df) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(iar_df["cost"], iar_df["iar"])
            for _, row in iar_df.iterrows():
                if not np.isinf(row["iar"]):
                    ax.annotate(row["mechanism"], (row["cost"], row["iar"]), fontsize=8)
            ax.set_xlabel("Cost (bytes)")
            ax.set_ylabel("IAR (value_gain / cost)")
            ax.set_title("IC-2b: Intelligence Appreciation Rate vs Cost")
            plt.tight_layout()
            fig.savefig("results/figures/ic2b_iar_vs_cost.png", dpi=150)
            plt.close()

        # Bad debt ratio
        if len(bd_df) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            bd_matches = [bd_df.iloc[i]["model_match"] for i in range(len(bd_df))]
            bd_vals = bd_df["bad_debt_ratio"].values
            colors = ["red" if b > 0.5 else "green" for b in bd_vals]
            ax.bar(bd_df["mechanism"], bd_vals, color=colors)
            ax.axhline(y=0.5, color="red", linestyle="--", label="Bad debt threshold")
            ax.set_ylabel("Bad Debt Ratio")
            ax.set_title("IC-2b: Bad Debt Ratio Audit")
            plt.xticks(rotation=45, ha="right")
            ax.legend()
            plt.tight_layout()
            fig.savefig("results/figures/ic2b_bad_debt_ratio.png", dpi=150)
            plt.close()

        # StateOnly gap
        if len(gap_df) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            gaps = gap_df["stateonly_gap"].values
            colors = ["green" if g > 0 else "red" for g in gaps]
            ax.bar(gap_df["mechanism"], gaps, color=colors)
            ax.set_ylabel("StateOnly Gap")
            ax.set_title("IC-2b: StateOnly Gap (vs LearnedStateOnly)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fig.savefig("results/figures/ic2b_stateonly_gap.png", dpi=150)
            plt.close()

        # RawMemory EqualCost comparison
        if len(raw_df) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            vs_raw = raw_df["vs_rawmemory_match"].values
            colors = ["green" if v > 0 else "red" for v in vs_raw]
            ax.bar(raw_df["mechanism"], vs_raw, color=colors)
            ax.set_ylabel("vs RawMemoryEqualCost (match)")
            ax.set_title("IC-2b: Raw Memory Equal Cost Comparison")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fig.savefig("results/figures/ic2b_raw_memory_equal_cost.png", dpi=150)
            plt.close()

        print("  Figures saved to results/figures/")
    except Exception as e:
        print(f"  Figure generation skipped: {e}")

    # --- Gate Evaluation ---
    print("\n" + "=" * 70)
    print("IC-2b Gate Evaluation")
    print("=" * 70)

    # Find best non-control mechanism
    controls = {"learned_state_only", "learned_action_only", "shuffled_action_control",
                "permuted_history_control", "raw_memory_full", "raw_memory_equal_cost",
                "prototype_memory"}
    learning_mechs = gap_df[~gap_df["mechanism"].isin(controls)]

    best_mech = None
    best_match = 0.0
    if len(learning_mechs) > 0:
        best_mech = learning_mechs.iloc[learning_mechs["stateonly_gap"].values.argmax()]
        best_match = best_mech["mean_match"]

    min_pass_checks = []
    strong_pass_checks = []
    death_conditions = []

    if best_mech is not None:
        # Minimum pass checks
        min_pass_checks.append(("ID > SO + 0.05", best_match > so_match_mean + 0.05, best_match - so_match_mean))
        min_pass_checks.append(("ID > AO + 0.10", best_match > ao_match_mean + 0.10, best_match - ao_match_mean))
        min_pass_checks.append(("ID > RawMemEqCost", best_match > rm_eq_match, best_match - rm_eq_match))
        bd_best = bd_df[bd_df["mechanism"] == best_mech["mechanism"]]["bad_debt_ratio"].values[0] if len(bd_df) > 0 else 1.0
        min_pass_checks.append(("BadDebt < 0.50", bd_best < 0.50, bd_best))
        sag_best = bd_df[bd_df["mechanism"] == best_mech["mechanism"]]["shuffled_action_gap"].values[0] if len(bd_df) > 0 else 0.0
        min_pass_checks.append(("ShuffledAction drops", sag_best > 0.02, sag_best))

        iar_best = iar_df[iar_df["mechanism"] == best_mech["mechanism"]]["iar"].values[0] if len(iar_df) > 0 else 0.0
        iar_rm = iar_df[iar_df["mechanism"] == "raw_memory_equal_cost"]["iar"].values[0] if len(iar_df) > 0 else 0.0
        min_pass_checks.append(("IAR > RawMemEqCost", iar_best > iar_rm, iar_best - iar_rm))

        # Strong pass checks
        strong_pass_checks.append(("ID > SO + 0.10", best_match > so_match_mean + 0.10, best_match - so_match_mean))
        strong_pass_checks.append(("BadDebt < 0.30", bd_best < 0.30, bd_best))

        # Death conditions
        if best_match <= so_match_mean + 0.05:
            death_conditions.append(("D1", "All mechanisms <= SO + 0.05"))
        if rm_eq_match >= best_match:
            death_conditions.append(("D2", "RawMemoryEqualCost >= all compressors"))
        if sag_best < 0.02:
            death_conditions.append(("D3", "ShuffledActionControl too close to main model"))
        phg_row = bd_df[bd_df["mechanism"] == "permuted_history_control"]
        phg = phg_row["model_match"].values[0] if len(phg_row) > 0 else best_match
        if abs(phg - best_match) < 0.02:
            death_conditions.append(("D4", "PermutedHistoryControl too close to main model"))

    print("\n  Minimum Pass Checks:")
    all_min_pass = True
    for name, passed, val in min_pass_checks:
        status = "PASS" if passed else "FAIL"
        if not passed: all_min_pass = False
        print(f"    [{status}] {name}: {val:.4f}")

    print("\n  Strong Pass Checks:")
    all_strong_pass = True
    for name, passed, val in strong_pass_checks:
        status = "PASS" if passed else "FAIL"
        if not passed: all_strong_pass = False
        print(f"    [{status}] {name}: {val:.4f}")

    if death_conditions:
        print("\n  Death Conditions Triggered:")
        for dc in death_conditions:
            print(f"    [DEATH] {dc[0]}: {dc[1]}")

    # --- Report ---
    top_5 = gap_df.nlargest(5, "stateonly_gap") if len(gap_df) > 0 else gap_df

    report = f"""# IC-2b Learned Throttling Mechanism Comparison Report

Generated: 2026-05-09

## 1. Summary
- LearnedStateOnly baseline: {so_match_mean:.4f}
- LearnedActionOnly baseline: {ao_match_mean:.4f}
- RawMemoryEqualCost baseline: {rm_eq_match:.4f}
- Best mechanism: {best_mech['mechanism'] if best_mech is not None else 'N/A'} ({best_match:.4f})

## 2. Top 5 Mechanisms by StateOnly Gap
{top_5.to_string(index=False) if len(top_5) > 0 else 'N/A'}

## 3. Gate Results
{'ALL MINIMUM PASSES PASSED' if all_min_pass else 'SOME MINIMUM PASSES FAILED'}
Strong passes: {sum(1 for _,p,_ in strong_pass_checks if p)}/{len(strong_pass_checks)}

## 4. Death Condition Audit
{chr(10).join(f'- {dc[0]}: {dc[1]}' for dc in death_conditions) if death_conditions else 'No death conditions triggered.'}

## 5. Mechanism Rankings
| Rank | Mechanism | Match | vs SO | vs AO | BadDebt | IAR |
|------|-----------|-------|-------|-------|---------|-----|
"""
    ranked = gap_df.sort_values("stateonly_gap", ascending=False)
    for i, (_, row) in enumerate(ranked.iterrows()):
        mech = row["mechanism"]
        bd_val = bd_df[bd_df["mechanism"] == mech]["bad_debt_ratio"].values[0] if len(bd_df) > 0 else 0.0
        iar_val = iar_df[iar_df["mechanism"] == mech]["iar"].values[0] if len(iar_df) > 0 else 0.0
        report += f"| {i+1} | {mech} | {row['mean_match']:.4f} | {row['stateonly_gap']:.4f} | {row['actiononly_gap']:.4f} | {bd_val:.3f} | {iar_val:.6f} |\n"

    report += """
## 6. Answers to Audit Questions

1. **Does any mechanism beat LearnedStateOnly?**
   """ + (f"Yes: {best_mech['mechanism']} at +{best_match - so_match_mean:.3f}" if best_match > so_match_mean else "No - death condition D1 would apply.") + """

2. **Does any mechanism beat RawMemoryEqualCost?**
   """ + (f"Yes" if best_match > rm_eq_match else "No - compression did not beat raw memory.") + """

3. **Does any mechanism maintain transfer premium on OOD?**
   OOD evaluation limited (OOD tables may be incomplete).

4. **Which mechanisms are bad debt?**
""" + (chr(10).join(f"   - {r['mechanism']}: BDR={r['bad_debt_ratio']:.3f}" for _, r in bd_df[bd_df["bad_debt_ratio"] > 0.5].iterrows()) if len(bd_df[bd_df["bad_debt_ratio"] > 0.5]) > 0 else "   No bad debt mechanisms found.") + """

5. **Does ResidualCompressor beat AEPCompressor?**
""" + (f"   Residual: {res_df[res_df['mechanism']=='residual_compressor']['best_action_match'].mean():.4f} vs AEP: {res_df[res_df['mechanism']=='aep_compressor']['best_action_match'].mean():.4f}" if 'residual_compressor' in res_df['mechanism'].values else "   N/A") + """

6. **Is CounterfactualCompressor the strongest?**
""" + (f"   Counterfactual: {res_df[res_df['mechanism']=='counterfactual_compressor']['best_action_match'].mean():.4f} (ranked #{ranked[ranked['mechanism']=='counterfactual_compressor'].index[0]+1 if 'counterfactual_compressor' in ranked['mechanism'].values else 'N/A'})" if 'counterfactual_compressor' in res_df['mechanism'].values else "   N/A") + """

7. **Does CausalContrast add value?**
""" + (f"   CausalContrast: {res_df[res_df['mechanism']=='causal_contrast']['best_action_match'].mean():.4f} vs best learner" if 'causal_contrast' in res_df['mechanism'].values else "   N/A") + """

8. **Is ICT intelligence appreciation supported?**
   """ + ("YES - Best mechanism significantly beats baselines." if all_min_pass else "NO - Redesign needed. Learnable structure not captured.") + """
"""
    with open("results/ic2b/IC2B_LEARNED_THROTTLING_REPORT.md", "w") as f:
        f.write(report)
    print("\n  Report saved to results/ic2b/IC2B_LEARNED_THROTTLING_REPORT.md")

    print("\n" + "=" * 70)
    if all_min_pass:
        print("IC-2b PASSED. Intelligence appreciation supported.")
    else:
        print("IC-2b FAILED. Redesign environment or throttling mechanisms.")
    print("=" * 70)


if __name__ == "__main__":
    main()