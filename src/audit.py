import numpy as np


DEATH_CONDITIONS = {
    "D1_residual_absent": {
        "metric": "residual_variance_ratio",
        "threshold": 0.15,
        "op": "lt",
        "message": "Action-effect signal too weak. Redesign env.",
    },
    "D2_residual_le_stateonly": {
        "requires": ["oracle_match", "so_match"],
        "op": "le",
        "message": "Knowing true residual doesn't beat ignoring action. AEP-control impossible.",
    },
    "D3_cf_worthless": {
        "metric": "counterfactual_value",
        "threshold": 0.0,
        "op": "le",
        "message": "Counterfactual data provides no value.",
    },
    "D4_rawmemory_cheaper": {
        "requires": ["model_iar", "rawmemory_iar"],
        "op": "le",
        "message": "RawMemory is cheaper than throttling for same value.",
    },
    "D5_mostly_bad_debt": {
        "metric": "bad_debt_ratio",
        "threshold": 0.5,
        "op": "gt",
        "message": "More than half of claimed gain is shortcut-explained.",
    },
    "D6_temporal_broken": {
        "metric": "permuted_history_gap",
        "threshold": 0.10,
        "op": "lt",
        "message": "Temporal order provides zero benefit. Architecture is nominal.",
    },
    "D7_benchmark_unstable": {
        "metric": "seed_stability_ratio",
        "threshold": 0.5,
        "op": "gt",
        "message": "Seed variance > model gap. Results are noise.",
    },
    "D8_no_transfer": {
        "metric": "transfer_premium",
        "threshold": 0.0,
        "op": "le",
        "message": "No OOD benefit. Structure is memorization.",
    },
    "D9_prediction_not_control": {
        "metric": "orr",
        "threshold": 0.5,
        "op": "lt",
        "message": "Model predicts well but cannot control.",
    },
    "D10_stateonly_dominates": {
        "metric": "state_only_shortcut_index",
        "threshold": 0.90,
        "op": "gt",
        "message": "Model gain is almost entirely StateOnly.",
    },
    "D11_actiononly_dominates": {
        "metric": "action_only_shortcut_index",
        "threshold": 0.80,
        "op": "gt",
        "message": "Model gain is almost entirely ActionOnly.",
    },
    "D12_shuffled_irrelevant": {
        "metric": "shuffled_action_gap",
        "threshold": 0.05,
        "op": "lt",
        "message": "Action-outcome pairing doesn't matter. Using spurious correlations.",
    },
}


def check_death(condition_id, metrics_dict):
    cond = DEATH_CONDITIONS[condition_id]
    if "requires" in cond:
        vals = [metrics_dict.get(k) for k in cond["requires"]]
        if any(v is None for v in vals):
            return None
        if cond["op"] == "le":
            triggered = vals[0] <= vals[1]
        else:
            triggered = vals[0] <= vals[1]
    else:
        val = metrics_dict.get(cond["metric"])
        if val is None:
            return None
        th = cond["threshold"]
        if cond["op"] == "lt":
            triggered = val < th
        elif cond["op"] == "gt":
            triggered = val > th
        elif cond["op"] == "le":
            triggered = val <= th
        else:
            triggered = False

    return {
        "condition_id": condition_id,
        "triggered": triggered,
        "value": val if "metric" in cond else f"{metrics_dict.get(cond['requires'][0])} vs {metrics_dict.get(cond['requires'][1])}",
        "message": cond["message"],
    }


def run_full_audit(metrics_dict):
    results = []
    for cond_id in DEATH_CONDITIONS:
        r = check_death(cond_id, metrics_dict)
        if r is not None:
            results.append(r)
    return results


def compute_bad_debt_audit(mechanism_results, so_result, ao_result, shuffled_result):
    audit = {}
    for mech_name, res in mechanism_results.items():
        so_match = so_result.get("best_action_match", 0.33)
        ao_match = ao_result.get("best_action_match", 0.33)
        shuf_match = shuffled_result.get(mech_name, {}).get("best_action_match", so_match)
        model_match = res.get("best_action_match", 0.33)
        shortcut_max = max(so_match, ao_match, shuf_match)
        shortcut_gain = max(0, shortcut_max - 0.33)
        claimed_gain = max(0, model_match - 0.33)
        bdr = 1.0 if claimed_gain <= 0 else min(1.0, shortcut_gain / claimed_gain)
        audit[mech_name] = {
            "bad_debt_ratio": bdr,
            "so_match": so_match,
            "ao_match": ao_match,
            "shuffled_match": shuf_match,
            "model_match": model_match,
            "is_bad_debt": bdr > 0.5,
        }
    return audit