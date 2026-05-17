# IC-3-V2: Current-Instance Capital Report Audit — Final Report

**Date**: 2026-05-11
**Phase**: IC-3-V2 (CapitalReport diagnostics — does NOT claim second-order intelligence)
**Seed**: 45  |  **Capital Set**: Main-5  |  **Schema**: v2 = 35 fields × 5 = 175 features (+12 vs v1)

---

## Final Verdict: `IC3_V2_WEAK_REPORT_SIGNAL_WITH_FALLBACK`

**External Task D (HiddenGoalGridWorld): UNINFORMATIVE** - all capitals near random (max=0.0533).

---

## 1. CapitalReport v2 — Implementation

35 fields per capital (12 new):
- v1 (23): historical performance — `recommended_action` ... `impairment_flag`
- v2 (+12): current-instance evidence — `action_margin`, `predicted_best_action_value`, `second_best_action_value`, `local_counterfactual_spread`, `self_consistency_score`, `local_support_quality`, `extrapolation_distance`, `goal_relevance_score`, `disagreement_with_portfolio`, `current_uncertainty`, `capital_specific_expected_regret`, `transition_sensitivity`

**Capitals with v2 specialization**:
- `PolicyCloneCapitalV2` — logits→action margin, entropy→uncertainty, perturbation→sensitivity
- `PrototypeOutcomeCapitalV2` — k-NN spread, support quality
- `AEPCapitalV2` — AE-compressed logits→margin/uncertainty
- `GoalInferenceCapitalV2` — belief propagation→goal relevance, belief entropy→uncertainty
- `SafeFallbackCapitalV2` — exploration rate→uncertainty

---

## 2. Feature Ban v2

| Category | Count |
|---|---|
| ALLOWED (v2 report fields) | 35 |
| FORBIDDEN (env metadata) | 11 |
| FORBIDDEN FOUND in input | 0 |

**Feature Ban**: ✅ PASS — 0 forbidden fields. No feature engineering regression.

---

## 3. Task Taxonomy Repair

| Task | PolicyClone | PrototypeOutcome | AEP | GoalInference | SafeFallback | Best | Expected | Margin | Match |
|---|---|---|---|---|---|---|---|---|---|
| Task_A | 1.0000 | 0.2000 | 0.5733 | 0.0067 | 0.0800 | **PolicyClone** | PolicyClone | 0.427 | ✅ |
| Task_B | 0.3467 | 0.8533 | 0.9200 | 0.1400 | 0.3333 | **AEP** | AEP | 0.067 | ✅ |
| Task_C | 0.1733 | 1.0000 | 0.9133 | 0.0000 | 0.1000 | **PrototypeOutcome** | PrototypeOutcome | 0.087 | ✅ |
| Task_D | 0.0267 | 0.0067 | 0.0267 | 0.0200 | 0.0133 | **PolicyClone** | GoalInference | 0.000 | ❌ |

**Taxonomy match rate**: 3/4
**Margin ≥ 0.05**: 3/4 tasks
**Verdict**: ✅ TAXONOMY PASSES

Taxonomy expectations:
- Task A (fixed-goal, low-entropy samples): PolicyClone should be best
- Task B (goal-transfer, cycling U): AEP should be best
- Task C (dense-support, dense-cluster samples): PrototypeOutcome should be best
- Task D (hidden-goal): GoalInference should be best

✅ Taxonomy repaired — capital specialization established.

---

## 4. Report Selector v2 Sufficiency

| Selector | Reward | Δ vs BS | Cum. Regret | Oracle Gap |
|---|---|---|---|---|
| BestSingle | 0.6083 | +0.0000 | 88.0 | 0.1467 |
| OracleHindsight | 0.7550 | +0.1467 | 0.0 | 0.0000 |
| LR_v1 | 0.5383 | -0.0700 | 130.0 | 0.2167 |
| LR_v2 | 0.5417 | -0.0667 | 128.0 | 0.2133 |
| RF_v2 | 0.5433 | -0.0650 | 127.0 | 0.2117 |
| GB_v2 | 0.6933 | +0.0850 | 37.0 | 0.0617 |
| MLP_v2 | 0.3867 | -0.2217 | 221.0 | 0.3683 |

**All metric invariants pass**: ✅

**Key comparison**:
- BestSingle (v2 env): 0.6083
- Best v2 selector: 0.6933
- v2 − v1 Δ: ...
- LR_v1 (baseline): 0.5383

---

## 5. BestSingle Fallback Selector

| Threshold | Reward | Δ vs BS | Switches |
|---|---|---|---|
| 0.5 | 0.6900 | +0.0817 | 498 |
| 0.6 | 0.6917 | +0.0833 | 405 |
| 0.7 | 0.6133 | +0.0050 | 196 |
| 0.8 | 0.6083 | +0.0000 | 0 |
| 0.9 | 0.6083 | +0.0000 | 0 |

**Best fallback config**: threshold=0.6 reward=0.6917

---

## 6. External Slice (Task D)

| Capital | Score |
|---|---|
| PolicyClone | 0.0267 |
| PrototypeOutcome | 0.0067 |
| AEP | 0.0267 |
| GoalInference | 0.0200 |
| SafeFallback | 0.0133 |
| BestSingle | 0.0267 |
| OracleHindsight | 0.0533 |

**Oracle gain on external**: +0.0267
**External task informative**: ❌ UNINFORMATIVE — all capitals near random

---

## 7. Answers

**Q1**: Did CapitalReport v2 recover observability?
**A**: IC3_V2_WEAK_REPORT_SIGNAL_WITH_FALLBACK. Best v2 selector = 0.6933 vs BestSingle = 0.6083 (Δ=+0.0850).

**Q2**: Which v2 selector performed best?
**A**: GB_v2 = 0.6933

**Q3**: Does fallback help?
**A**: YES — fallback at threshold=0.6 achieves 0.6916666626930237

**External Task D (HiddenGoalGridWorld): UNINFORMATIVE** - all capitals near random (max=0.0533).

---

## Generated Files (results/ic3_v2/)

| # | File | Content |
|---|---|---|
| 1 | `feature_ban_v2.csv` | 35 allowed + 11 forbidden, 0 found |
| 2 | `task_taxonomy_repair.csv` | Per-task per-capital reward, specialization indices |
| 3 | `report_v2_sufficiency_test.csv` | v1/v2 selectors with clean metrics |
| 4 | `metric_invariants_v2.csv` | I1/I2/OH invariants |
| 5 | `bestsingle_fallback_selector.csv` | 5 thresholds × confidence-based switching |
| 6 | `external_slice_v2.csv` | Task D only: all capitals + selectors |
| 7 | `IC3_V2_CURRENT_INSTANCE_REPORT_AUDIT.md` | This report |

---

*End of IC-3-V2. CapitalReport v2 audit complete. No second-order intelligence claim made.*
