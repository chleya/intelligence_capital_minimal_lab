# IC-3-W: Within-Task Capital Switching Benchmark — Final Report

**Date**: 2026-05-11
**Seeds**: 43..52 (10 seeds)  |  **Task**: Task_W (unified, hidden sub-regimes)

---

## Final Verdict: `IC3_W_SUBREGIME_PROXY_DEPENDENT`

---

## 1. Subregime Capital Taxonomy

| Subregime | Expected | Actual Best | Margin | Match |
|---|---|---|---|---|
| W1_PolicyClone | PolicyClone | **PolicyClone** | 1.0000 | YES |
| W2_AEP | AEP | **AEP** | 1.0000 | YES |
| W3_PrototypeOutcome | PrototypeOutcome | **PrototypeOutcome** | 1.0000 | YES |
| W4_GoalInference | GoalInference | **GoalInference** | 0.9900 | YES |

| Capital | W1_PolicyClone | W2_AEP | W3_PrototypeOutcome | W4_GoalInference |
|---|---|---|---|---|
| PolicyClone | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| PrototypeOutcome | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| AEP | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| GoalInference | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| SafeFallback | 0.0000 | 0.0000 | 0.0000 | 0.0100 |

**Taxonomy valid**: YES — match 4/4, margin_ok 4/4

---

## 2. Within-Task Selector Test

| Seed | BS | Oracle | GB_v2 | Δ | LR_v2 | RF_v2 | MLP_v2 |
|---|---|---|---|---|---|---|---|
| 43 | 0.4017 | 0.9700 | 0.7050 | +0.3033 | 0.3950 | 0.4017 | 0.3167 |
| 44 | 0.4017 | 0.9867 | 0.7767 | +0.3750 | 0.4167 | 0.4183 | 0.3167 |
| 45 | 0.4017 | 0.9650 | 0.7217 | +0.3200 | 0.3667 | 0.4067 | 0.3167 |
| 46 | 0.4017 | 0.9867 | 0.7817 | +0.3800 | 0.3667 | 0.4267 | 0.3167 |
| 47 | 0.4017 | 0.9567 | 0.7483 | +0.3467 | 0.3500 | 0.4117 | 0.3167 |
| 48 | 0.4017 | 0.9583 | 0.7183 | +0.3167 | 0.3950 | 0.4217 | 0.3167 |
| 49 | 0.4017 | 0.9350 | 0.6733 | +0.2717 | 0.3600 | 0.4083 | 0.3167 |
| 50 | 0.4017 | 1.0000 | 0.7817 | +0.3800 | 0.3817 | 0.4250 | 0.3167 |
| 51 | 0.4017 | 0.9033 | 0.6550 | +0.2533 | 0.3700 | 0.3967 | 0.3167 |
| 52 | 0.4017 | 1.0000 | 0.7817 | +0.3800 | 0.3900 | 0.4217 | 0.3167 |

**Mean GB_v2 Δ**: +0.3327  |  **Positive**: 10/10  |  **95% CI**: [+0.2989, +0.3664]
**Within-task pass**: YES (need Δ>+0.05, ≥8/10, CI≥0)

---

## 3. Per-Subregime Selector Performance

| Subregime | GB_v2 | BestSingle | Δ | Positive |
|---|---|---|---|---|
| W1 | 0.8173 | 0.6467 | +0.1707 | 10/10 |
| W2 | 0.8693 | 0.3333 | +0.5360 | 10/10 |
| W3 | 0.9200 | 0.6267 | +0.2933 | 10/10 |
| W4 | 0.3307 | 0.0000 | +0.3307 | 10/10 |

---

## 4. Proxy Suppression (Subregime)

| Config | SR Pred Acc | Selector Reward | Δ vs BS |
|---|---|---|---|
| full | 0.9792 | 0.7343 | +0.3327 |
| suppress_k10 | 0.2510 | 0.3340 | -0.0677 |
| suppress_k20 | 0.2500 | 0.3167 | -0.0850 |
| suppress_k5 | 0.2995 | 0.3377 | -0.0640 |

**Proxy dependence**: SUBREGIME_PROXY_DEPENDENT — removing sr-predictive features crashes selector

---

## 5. Conservative Switcher

| Metric | Value |
|---|---|
| Test Δ vs BS | +0.2935 |
| Positive seeds (>3%) | 10/10 |
| Beneficial switch rate | 0.353 |
| Harmful switch rate | 0.009 |
| Robust | YES |

---

## 6. External Grid-v2 Diagnostic

| Capital | Score |
|---|---|
| PolicyClone | 0.0000 |
| PrototypeOutcome | 0.0000 |
| AEP | 0.0000 |
| GoalInference | 1.0000 |
| SafeFallback | 0.0067 |

**Grid-v2**: max=1.0000 oracle=1.0000 status=OK

---

## Generated Files (results/ic3_w/)

| # | File |
|---|---|
| 1 | `per_subregime_capital_matrix.csv` |
| 2 | `within_task_selector_test.csv` |
| 3 | `per_subregime_capital_matrix.csv` |
| 4 | `within_task_proxy_suppression.csv` |
| 5 | `conservative_switcher.csv` |
| 6 | `external_grid_v2_diagnostic.csv` |
| 7 | `IC3_W_WITHIN_TASK_CAPITAL_SWITCHING_REPORT.md` |

---

*End of IC-3-W. No second-order intelligence claim made.*
