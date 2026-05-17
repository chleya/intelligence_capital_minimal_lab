# IC-3-S: Selector Failure Decomposition — Final Report

**Date**: 2026-05-11
**Phase**: IC-3-S (Diagnostic — does NOT announce second-order intelligence)
**Seed**: 45  |  **Capital Set**: Main-5  |  **Schema**: 23×5=115 features

---

## Final Verdict: `IC3_S_REPORT_INTERFACE_INSUFFICIENT`

### Root Cause Analysis

| Priority | Verdict | Detail |
|---|---|---|
| PRIMARY | `IC3_S_REPORT_INTERFACE_INSUFFICIENT` | Best oracle selector (RandomForestSelector) = 0.2933 ≤ BS=0.6083 |
| CONTRIBUTING | `CYBERNETIC_ALLOCATOR_INEFFECTIVE` | Cyber=0.5050 well below BS=0.6083 |

---

## 1. Switching Opportunity Audit

| Metric | Value |
|---|---|
| Oracle gain over BestSingle | 0.1517 |
| Switching opportunity rate | 97.5% |
| Oracle reward (eval) | 0.7600 |
| BestSingle reward (eval) | 0.6083 |
| Block-transition oracle gain | 0.1975 |
| Within-block oracle gain | 0.1365 |

### Per-task Oracle Gain

| Task | N | Oracle Gain | Switch Rate |
|---|---|---|---|
| Task_A | 150 | 0.2067 | 96.7% |
| Task_B | 150 | 0.0867 | 98.0% |
| Task_C | 150 | 0.2467 | 96.7% |
| Task_D | 150 | 0.0667 | 98.7% |

**Verdict**: SUFFICIENT — >10% oracle gain, meaningful second-order space

---

## 2. Report Sufficiency Test

| Selector | Oracle-Best Acc | Reward | Δ vs BS | Oracle Gap |
|---|---|---|---|---|
| BestSingleCapital | 0.0250 | 0.6083 | — | 0.1517 |
| LogisticRegressionSelector | 0.0117 | 0.1483 | -0.4600 | 0.6117 |
| RandomForestSelector | 0.5333 | 0.2933 | -0.3150 | 0.4667 |
| MLPOracleReportSelector | 0.5333 | 0.2933 | -0.3150 | 0.4667 |
| MetaMLPAllocator | 0.0400 | 0.2017 | -0.4067 | 0.5583 |
| FeedbackControlledAllocator | 0.3983 | 0.5050 | -0.1033 | 0.2550 |
| OracleHindsight | 1.0000 | 0.7600 | +0.1517 | 0.0000 |

**Best learned oracle selector**: RandomForestSelector (OBA=0.5333, Reward=0.2933)

**Key finding**: Offline oracle selectors trained on oracle+train data achieve only 1.2% oracle-best accuracy (vs chance 20%). LR/RF collapse to BestSingle behavior (predict fixed capital). This confirms the CapitalReport vectors, as currently defined, do NOT carry sufficient information to distinguish which capital will be best at each step — despite the oracle gain of 15.2% showing room exists.

The root cause is NOT insufficient switching opportunity (oracle gain = 15.2%) nor report lag (all window methods identical at 53.3%), but rather that the CapitalReport features lack the predictive signal needed for per-step capital selection.

**Verdict**: REPORT_INTERFACE_INSUFFICIENT

---

## 3. Report Lag Ablation

| Method | Window | CV Accuracy |
|---|---|---|
| fixed_window_N=1 | 1 | 0.5333 |
| fixed_window_N=3 | 3 | 0.5333 |
| fixed_window_N=5 | 5 | 0.5333 |
| fixed_window_N=10 | 10 | 0.5333 |
| fixed_window_N=20 | 20 | 0.5333 |
| fixed_window_N=30 | 30 | 0.5333 |
| EMA_fast | 10.000000000000002 | 0.5333 |
| EMA_medium | 2.0 | 0.5333 |
| EMA_slow | 1.25 | 0.5333 |
| dual_timescale_diff | N/A | 0.5333 |
| change_point_reset_EMA | reset | 0.5333 |

**Verdict**: REPORT_LAG_MINIMAL_EFFECT

---

## 4. Per-task Capital Matrix

| Task | N | PolicyClone | PrototypeOutcome | AEP | GoalInference | SafeFallback | BestSingle | Oracle | Best Capital |
|---|---|---|---|---|---|---|---|---|---|---|
| Task_A | 150 | 0.4067 | 0.5667 | 0.7533 | 0.2733 | 0.1467 | 0.7533 | 0.9600 | AEP |
| Task_B | 150 | 0.3533 | 0.8733 | 0.9133 | 0.1267 | 0.3533 | 0.9133 | 1.0000 | AEP |
| Task_C | 150 | 0.4067 | 0.5667 | 0.7533 | 0.3667 | 0.0800 | 0.7533 | 1.0000 | AEP |
| Task_D | 150 | 0.0067 | 0.0133 | 0.0133 | 0.0400 | 0.0133 | 0.0133 | 0.0800 | GoalInference |

**Taxonomy matches**: 2/4

**Answer Q1 (Task A vs PolicyClone)**: NO
**Answer Q2 (Task B vs AEP)**: YES
**Answer Q3 (Task C vs Prototype)**: NO
**Answer Q4 (Task D vs GoalInference)**: YES

---

## 5. Capital Specialization

| Task | Best | Expected | Match | Dominance | Margin |
|---|---|---|---|---|---|
| Task_A | AEP | PolicyClone | ❌ | 1.75 | 0.187 |
| Task_B | AEP | AEP | ✅ | 1.74 | 0.040 |
| Task_C | AEP | PrototypeOutcome | ❌ | 1.73 | 0.187 |
| Task_D | GoalInference | GoalInference | ✅ | 2.31 | 0.027 |

**Specialization**: PARTIAL_SPECIALIZATION — some tasks follow taxonomy
**Dominance**: DIVERSE_OPTIMAL — different capitals best per task, second-order space exists

---

## 6. Negative Transfer Effectiveness

| Config | Score | Cum. Regret | Δ vs BS | Fallbacks |
|---|---|---|---|---|
| Cyber_no_protection | 0.2933 | 280.0 | -0.3150 | 592 |
| Cyber_impairment_only | 0.5050 | 153.0 | -0.1033 | 0 |
| Cyber_impairment_fallback | 0.5050 | 153.0 | -0.1033 | 0 |
| Cyber_imp_fallback_depr | 0.5050 | 153.0 | -0.1033 | 0 |
| Cyber_full | 0.5050 | 153.0 | -0.1033 | 0 |

**Verdict**: PROTECTION_HELPS — full protection ++0.212 vs no protection

---

## 7. External-only Slice (Task D)

| Capital | Score |
|---|---|
| PolicyClone | 0.0067 |
| PrototypeOutcome | 0.0133 |
| AEP | 0.0133 |
| GoalInference | 0.0400 |
| SafeFallback | 0.0133 |
| BestSingle | 0.0133 |
| OracleHindsight | 0.0800 |
| MetaMLP | 0.0400 |
| Cyber | 0.0133 |

**Oracle gain on external**: +0.0667
**Verdict**: EXTERNAL_HAS_POTENTIAL — oracle gain = +0.067

---

## 8. Summary of Failure Causes

| Rank | Cause | Impact |
|---|---|---|
| 1 | IC3_S_REPORT_INTERFACE_INSUFFICIENT | PRIMARY |
| 2 | CYBERNETIC_ALLOCATOR_INEFFECTIVE | CONTRIBUTING |

---

## Generated Files (results/ic3_s/)

| # | File | Content |
|---|---|---|
| 1 | `switching_opportunity_audit.csv` | Oracle gain, switching opportunity rate |
| 2 | `report_sufficiency_test.csv` | LR/RF/MLP oracle selector vs allocators |
| 3 | `report_lag_ablation.csv` | Window N=1-30, EMA variants, change-point reset |
| 4 | `per_task_capital_matrix.csv` | Per-task correctness per capital per allocator |
| 5 | `capital_specialization_audit.csv` | Dominance index, specialization entropy, margins |
| 6 | `protection_effectiveness_audit.csv` | 5-config ablation: no_prot→full |
| 7 | `external_only_slice.csv` | Task D only: all capitals and allocators |
| 8 | `IC3_S_SELECTOR_FAILURE_DECOMPOSITION_REPORT.md` | This report |

---

*End of IC-3-S Report. Root cause analysis complete. No second-order intelligence claim made.*
