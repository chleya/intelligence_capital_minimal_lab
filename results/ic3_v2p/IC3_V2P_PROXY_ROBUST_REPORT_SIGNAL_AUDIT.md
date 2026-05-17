# IC-3-V2P: Proxy-Robust Report Signal Audit — Final Report

**Date**: 2026-05-11
**Phase**: IC-3-V2P (Proxy vs genuine-capital-signal diagnosis)
**Seeds**: 43..52 (10 seeds)

---

## Final Verdict: `IC3_V2P_TASK_PROXY_DEPENDENT`

---

## 1. Ablation Consistency Audit

| Configuration | Mean Reward | Std | Min | Max |
|---|---|---|---|---|
| full_v2 | 0.6950 | 0.0066 | 0.6817 | 0.7067 |
| only_counterfactual | 0.6732 | 0.0081 | 0.6600 | 0.6867 |
| only_expected_regret | 0.6732 | 0.0081 | 0.6600 | 0.6867 |
| only_margin | 0.6732 | 0.0081 | 0.6600 | 0.6867 |
| only_support | 0.6732 | 0.0081 | 0.6600 | 0.6867 |
| v1_only | 0.6732 | 0.0081 | 0.6600 | 0.6867 |
| only_disagreement | 0.6698 | 0.0080 | 0.6583 | 0.6867 |
| only_uncertainty | 0.6697 | 0.0080 | 0.6550 | 0.6833 |

**Key finding**: v1_only = 0.6732 vs full_v2 = 0.6950. v2 shows incremental value over v1.

---

## 2. Task-Heldout Generalization

| Scenario | Mean Reward | Mean Δ vs BS | Positive Seeds | Oracle Gap |
|---|---|---|---|---|
| train_ABC_test_D | 0.0187 | +0.0033 | 5/10 | 0.0473 |
| train_AB_test_C | 0.6780 | -0.2020 | 0/10 | 0.3187 |
| train_AC_test_B | 0.7360 | -0.1940 | 0/10 | 0.2573 |
| train_BC_test_A | 0.7847 | +0.0513 | 10/10 | 0.2133 |

**Heldout verdict**: Some held-out tasks show positive signal.

---

## 3. Within-Task Switching

| Task | Overall Δ vs BS | Oracle Gain |
|---|---|---|
| Task_A | +0.2147 | — |
| Task_B | -0.0307 | — |
| Task_C | +0.0753 | — |

**Within-task verdict**: PASS — selector shows within-task switching ability.

---

## 4. Task Proxy Suppression

| Configuration | Task Pred Acc | Selector Reward | Δ vs BS |
|---|---|---|---|
| full | 0.9872 | 0.7045 | +0.0648 |
| suppress_k10 | 0.2650 | 0.5102 | -0.1295 |
| suppress_k20 | 0.2500 | 0.3725 | -0.2672 |
| suppress_k5 | 0.3135 | 0.4842 | -0.1555 |

**Proxy suppression verdict**: TASK_PROXY_DEPENDENT — removing task-predictive features causes selector reward to collapse. Signal is largely task proxy.

---

## 5. Conservative Fallback Robustness

Best fallback per seed: mean Δ = +0.0645, positive ≥ 5% = 7/10
Beneficial switch rate = 0.081, Harmful = 0.015
Fallback robust (>5%): FAIL

---

## 6. External Task Recommendation

See `results/ic3_v2p/external_task_recommendation.md` for analysis and 4 replacement options.

---

## Generated Files (results/ic3_v2p/)

| # | File |
|---|---|
| 1 | `ablation_consistency_audit.csv` |
| 2 | `task_heldout_generalization.csv` |
| 3 | `within_task_switching_test.csv` |
| 4 | `proxy_suppression_test.csv` |
| 5 | `conservative_fallback_robustness.csv` |
| 6 | `external_task_recommendation.md` |
| 7 | `IC3_V2P_PROXY_ROBUST_REPORT_SIGNAL_AUDIT.md` |

---

*End of IC-3-V2P. No second-order intelligence claim made.*
