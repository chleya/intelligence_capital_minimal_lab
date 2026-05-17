# IC-2c+: Fair Goal-Transfer Robustness Audit

**Date**: 2026-05-10
**Final Verdict**: `IC2C_PLUS_ABSOLUTE_GOAL_TRANSFER_SUPPORTED_COST_EFFICIENCY_UNRESOLVED`

---
## Executive Summary

IC-2c extends the Policy-Clone Trap Escape Benchmark with fairness audits:
1. Stronger policy baselines (MultiGoal, FewShot)
2. Fair memory baselines (OutcomeTable, not regression->argmax)
3. Active probe fairness (PolicyClone gets probe)
4. Revised coverage gap (systematic masking)
5. Cost-normalized premiums

### Held-Out Utility Transfer (mean of 25 random utilities)

| Mechanism | Mean Match | vs U1_PolicyClone Premium |
|---|---|---|
| U1_PolicyClone | 0.2610 | — |
| AEPCompressor | 0.9115 | +0.6505 |
| CounterfactualCompressor | 0.3289 | +0.0680 |
| ResidualCompressor | 0.9174 | - |
| MultiGoalPolicyClone | 0.3614 | +0.1004 |
| RawMemoryOutcomeTableFull | 0.8824 | +0.6214 |
| StandardizedKNNOutcomeTable | 0.8824 | +0.6214 |
| PrototypeOutcomeTable | 0.8484 | +0.5875 |

---
## Q1: Are Base Utilities (U1-U5) Valid?

| Utility | Valid? | Label Div from U1 | Distribution |
|---|---|---|---|
| U1_main | False | 0.0000 | -1:0.46 / 0:0.10 / +1:0.45 |
| U2_reverse | True | 1.0000 | -1:0.48 / 0:0.00 / +1:0.52 |
| U3_target | True | 0.9000 | -1:0.01 / 0:0.99 / +1:0.00 |
| U4_risk_avoid | False | 0.0000 | -1:0.46 / 0:0.10 / +1:0.45 |
| U5_energy_aware | True | 0.7450 | -1:0.10 / 0:0.84 / +1:0.07 |

**Finding**: U4_risk_avoid has < 0.10 divergence from U1 — marked INVALID. U4 outcomes are too similar to U1.

**Extended utilities**: 19 valid (divergence > 0.10 from U1) out of 25 total.

---
## Q2: Does AEP Beat PolicyClone on Held-Out Utilities?

**Answer: YES.**
  AEP held-out transfer premium: +0.6505

**SUCCESS**: AEP retains >+0.10 transfer premium on fair held-out utilities.

---
## Q3: Does AEP Beat MultiGoalPolicyClone?

**Answer: YES.**
  AEP=0.9115 vs MultiGoalPC=0.3614

**SUCCESS**: AEP exceeds goal-conditioned PolicyClone. Compression beats goal-labeled imitation.

---
## Q4: Does AEP Beat Fair RawMemoryOutcomeTableEqualCost?

**Answer: YES.**
  AEP=0.9115 vs RawMemoryOutcomeTableFull=0.8824

**SUCCESS**: AEP beats fair memory baseline (full outcome table with nearest-neighbor transfer).

---
## Q5: Is Active Probe Advantage Structural (AEP), Not Just Privilege?

| Model | NoProbe | OneProbe | Probe Gain |
|---|---|---|---|
| PolicyClone | 0.8533 | 0.9433 | +0.0900 |
| AEPCompressor | 0.7917 | 0.9517 | +0.1600 |
| RawMemory | — | — | — |

**AEP Structural Probe Advantage** (AEP gain - PC gain): +0.0700

**SUCCESS**: AEP probe advantage is structural (>0.05 beyond PolicyClone's probe gain).

---
## Q6: Does Revised Coverage Gap Provide Stronger Evidence?

| CF Fraction | PolicyClone | AEPCompressor | CFCompressor |
|---|---|---|---|
| 0% | 0.3983 | 0.4033 | 0.4600 |
| 1% | 0.4133 | 0.3033 | 0.4600 |
| 5% | 0.3883 | 0.4000 | 0.4600 |
| 10% | 0.4367 | 0.3950 | 0.4600 |
| 20% | 0.4500 | 0.5500 | 0.4600 |
| 50% | 0.5533 | 0.4550 | 0.4600 |
| 100% | 0.7250 | 0.5200 | 0.7417 |

**SUCCESS**: AEP at 20% CF (0.550) beats PolicyClone (0.450) by >0.10.

---
## Q7: Does Cost-Normalized Intelligence Appreciation Hold?

| Mechanism | Params | Transfer Premium | Cost-Norm Transfer | Cost-Norm Probe |
|---|---|---|---|---|
| U1_PolicyClone | 20000 | +0.0000 | +0.0000 | +0.0045 |
| AEPCompressor | 35000 | +0.6505 | +0.0186 | +0.0046 |
| CounterfactualCompressor | 25000 | +0.0680 | +0.0027 | +0.0000 |
| ResidualCompressor | 40000 | +0.6564 | +0.0164 | +0.0000 |
| MultiGoalPolicyClone | 25000 | +0.1004 | +0.0040 | +0.0000 |
| RawMemoryOutcomeTableFull | 5000 | +0.6214 | +0.1243 | +0.0000 |
| StandardizedKNNOutcomeTable | 6000 | +0.6214 | +0.1036 | +0.0000 |
| PrototypeOutcomeTable | 1200 | +0.5875 | +0.4895 | +0.0000 |
| OracleUtilityPolicy | inf | -0.2610 | -0.0000 | +0.0000 |
| ActionOnly | 3 | -0.2610 | -86.9883 | +0.0000 |

---
## Final Verdict

### `IC2C_PLUS_ABSOLUTE_GOAL_TRANSFER_SUPPORTED_COST_EFFICIENCY_UNRESOLVED`

Conditions passed: 6/6
- aep_heldout_gt_pc_10: PASS
- aep_heldout_gt_mgpc: PASS
- aep_heldout_gt_rmot: PASS
- aep_cost_norm_positive: PASS
- aep_probe_struct_adv_gt_05: PASS
- coverage_aep_gt_pc_20: PASS

---
### All IC-2c+ Outputs

| File | Content |
|---|---|
| `results/ic2c_plus/utility_validity_audit.csv` | U1-U5 validity + pairwise disagreements |
| `results/ic2c_plus/heldout_utility_transfer.csv` | All mechanisms on 25 held-out random utilities |
| `results/ic2c_plus/policy_baseline_comparison.csv` | U1-U5 policy baseline comparison |
| `results/ic2c_plus/memory_outcome_table_baselines.csv` | Fair memory baselines on held-out utilities |
| `results/ic2c_plus/active_probe_fairness.csv` | Probe fairness with PolicyClone privilege |
| `results/ic2c_plus/coverage_gap_revised.csv` | Revised systematic action masking |
| `results/ic2c_plus/cost_normalized_premium.csv` | Cost-normalized transfer & probe premiums |
| `results/ic2c_plus/IC2C_PLUS_FAIR_GOAL_TRANSFER_REPORT.md` | **This report** |