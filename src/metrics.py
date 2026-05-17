import numpy as np
from scipy.stats import entropy as scipy_entropy


def compute_best_action_match(y_pred, y_true):
    a_pred = np.argmax(y_pred, axis=-1)
    a_true = np.argmax(y_true, axis=-1)
    return float((a_pred == a_true).mean())


def compute_regret(y_pred, y_true):
    best_pred = np.argmax(y_pred, axis=-1)
    best_true = np.argmax(y_true, axis=-1)
    val_pred = y_pred[np.arange(len(y_pred)), best_pred]
    val_best = y_pred[np.arange(len(y_pred)), best_true]
    return float((val_best - val_pred).mean())


def compute_rank_accuracy(y_pred, y_true):
    n = len(y_pred)
    correct = 0
    total = 0
    for i in range(n):
        rank_pred = np.argsort(y_pred[i])[::-1]
        rank_true = np.argsort(y_true[i])[::-1]
        for j in range(3):
            for k in range(j + 1, 3):
                total += 1
                if (y_pred[i, rank_true[j]] > y_pred[i, rank_true[k]]) == (y_true[i, rank_true[j]] > y_true[i, rank_true[k]]):
                    correct += 1
    return float(correct / total) if total > 0 else 0.0


def compute_residual_variance_ratio(outcomes_flat):
    all_vals = np.concatenate(outcomes_flat)
    total_var = float(np.var(all_vals))
    residuals = []
    for i in range(0, len(outcomes_flat), 3):
        if i + 3 <= len(outcomes_flat):
            noop = outcomes_flat[i + 1]
            residuals.append(outcomes_flat[i] - noop)
            residuals.append(outcomes_flat[i + 2] - noop)
    if len(residuals) == 0:
        return 0.0
    all_res = np.concatenate(residuals)
    res_var = float(np.var(all_res))
    return res_var / total_var if total_var > 0 else 0.0


def compute_oracle_action_entropy(best_actions):
    values, counts = np.unique(best_actions, return_counts=True)
    probs = counts / counts.sum()
    return float(scipy_entropy(probs)) / np.log(3)


def compute_action_only_ceiling(y_true):
    best_counts = np.bincount(np.argmax(y_true, axis=-1), minlength=3)
    most_common = np.argmax(best_counts)
    return float(best_counts[most_common]) / y_true.shape[0]


def compute_state_only_gap(model_match, so_match):
    return model_match - so_match


def compute_prediction_dividend(model_match, baseline_match):
    return model_match - baseline_match


def compute_control_dividend(model_regret, baseline_regret):
    return baseline_regret - model_regret


def compute_transfer_premium(ood_match, id_match):
    return ood_match - id_match


def compute_structural_dividend(match_new_context, match_old_context):
    return match_new_context - match_old_context


def compute_counterfactual_value(cf_match, traj_match):
    return cf_match - traj_match


def compute_20pct_cf_efficiency(gain_20pct, gain_100pct):
    if gain_100pct <= 0:
        return 0.0
    return gain_20pct / gain_100pct


def compute_iar(value_gain, cost_bytes):
    if cost_bytes <= 0:
        return float("inf") if value_gain > 0 else 0.0
    return value_gain / cost_bytes


def compute_orr(control_dividend, prediction_dividend):
    if prediction_dividend <= 0:
        return 0.0
    return control_dividend / prediction_dividend


def compute_bad_debt_ratio(so_match, ao_match, shuffled_match, claimed_match):
    shortcut_max = max(so_match, ao_match, shuffled_match)
    shortcut_gain = max(0, shortcut_max - 0.33)
    claimed_gain = max(0, claimed_match - 0.33)
    if claimed_gain <= 0:
        return 1.0
    return min(1.0, shortcut_gain / claimed_gain)


def compute_state_only_shortcut_index(so_match, model_match):
    if model_match <= 0:
        return 1.0
    return so_match / model_match


def compute_action_only_shortcut_index(ao_match, model_match):
    if model_match <= 0:
        return 1.0
    return ao_match / model_match


def compute_shuffled_action_gap(model_match, shuffled_match):
    return model_match - shuffled_match


def compute_permuted_history_gap(model_match, permuted_match):
    return model_match - permuted_match


def compute_compression_to_value_ratio(cost_bytes, value_gain):
    if value_gain <= 0:
        return float("inf")
    return cost_bytes / value_gain


def compute_capital_decay(match_t, match_t_plus_delta):
    if match_t <= 0:
        return 0.0
    return match_t_plus_delta / match_t


def compute_operational_leverage(control_gain, prediction_gain):
    if prediction_gain <= 0:
        return 0.0
    return control_gain / prediction_gain


def compute_seed_stability_ratio(model_std, model_gap):
    if model_gap <= 0:
        return float("inf")
    return model_std / model_gap


def compute_ood_kl_control(oracle_dist_train, oracle_dist_test):
    kl = 0.0
    for i in range(len(oracle_dist_train)):
        if oracle_dist_train[i] > 0 and oracle_dist_test[i] > 0:
            kl += oracle_dist_train[i] * np.log(oracle_dist_train[i] / oracle_dist_test[i])
    return float(kl)


METRICS_REGISTRY = {
    "best_action_match": compute_best_action_match,
    "regret": compute_regret,
    "rank_accuracy": compute_rank_accuracy,
    "residual_variance_ratio": compute_residual_variance_ratio,
    "oracle_action_entropy": compute_oracle_action_entropy,
    "action_only_ceiling": compute_action_only_ceiling,
    "prediction_dividend": compute_prediction_dividend,
    "control_dividend": compute_control_dividend,
    "transfer_premium": compute_transfer_premium,
    "structural_dividend": compute_structural_dividend,
    "counterfactual_value": compute_counterfactual_value,
    "20pct_cf_efficiency": compute_20pct_cf_efficiency,
    "iar": compute_iar,
    "orr": compute_orr,
    "bad_debt_ratio": compute_bad_debt_ratio,
    "state_only_shortcut_index": compute_state_only_shortcut_index,
    "action_only_shortcut_index": compute_action_only_shortcut_index,
    "shuffled_action_gap": compute_shuffled_action_gap,
    "permuted_history_gap": compute_permuted_history_gap,
    "compression_to_value_ratio": compute_compression_to_value_ratio,
    "capital_decay": compute_capital_decay,
    "operational_leverage": compute_operational_leverage,
    "seed_stability_ratio": compute_seed_stability_ratio,
    "ood_kl_control": compute_ood_kl_control,
}