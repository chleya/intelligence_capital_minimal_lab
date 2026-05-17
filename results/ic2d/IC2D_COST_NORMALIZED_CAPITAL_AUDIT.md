# IC-2d: Cost-Normalized Capital Audit

**Final Verdict**: `IC2D_ABSOLUTE_TRANSFER_ONLY`

---
## Q1: Is AEP's Absolute Goal-Transfer Advantage Stable?

| Mechanism | Mean Held-Out Match | vs U1_PolicyClone (0.261) |
|---|---|---|
| AEPCompressor | 0.9543 | +0.6933 |
| ResidualCompressor | 0.9532 | +0.6922 |
| RawMemoryOutcomeTableFull | 0.8956 | +0.6346 |
| StandardizedKNNOutcomeTable | 0.8956 | +0.6346 |
| PrototypeOutcomeTable | 0.8999 | +0.6389 |

---
## Q2: Does AEP Win After Cost Normalization?

| Mechanism | Match | Total Capital Cost | Cost-Norm (match/kB) |
|---|---|---|---|
| AEPCompressor | 0.9543 | 52141 | 0.0183 |
| ResidualCompressor | 0.9532 | 34529 | 0.0276 |
| CounterfactualCompressor | 0.4164 | 38229 | 0.0109 |
| RawMemoryOutcomeTableFull | 0.8956 | 4977 | 0.1799 |
| StandardizedKNNOutcomeTable | 0.8956 | 4968 | 0.1803 |
| PrototypeOutcomeTable | 0.8999 | 5409 | 0.1664 |

---
## Q3: Is PrototypeOutcomeTable the More Efficient Capital?

**YES.** Prototype cost-normalized=0.1664 > AEP=0.0183. Prototype is more capital-efficient.

---
## Q4: Does AEP Show Structure Advantage on Far OOD?

| Mechanism | ID | near OOD | far OOD | OOD drop |
|---|---|---|---|---|
| AEPCompressor | 0.9025 | 0.7395 | 0.6000 | +0.3025 |
| ResidualCompressor | 0.9186 | 0.7836 | 0.6496 | +0.2689 |
| RawMemoryOutcomeTableFull | 0.8832 | 0.8827 | 0.8811 | +0.0021 |
| PrototypeOutcomeTable | 0.8896 | 0.8789 | 0.8702 | +0.0195 |

Memory (0.881) matches or beats AEP (0.600) on far OOD. No extrapolation advantage.

---
## Q5: Does AEP Scale Better with Dataset Size?

| Mechanism | Scaling Slope (log n_train) |
|---|---|
| AEPCompressor | 0.0170 |
| PrototypeOutcomeTable | 0.0014 |
| RawMemoryOutcomeTableFull | 0.0051 |
| Reference_PCHeldout | 0.0000 |
| ResidualCompressor | 0.0165 |

---
## Final Verdict

### `IC2D_ABSOLUTE_TRANSFER_ONLY`

Conditions passed: 4/6
- **abs_aep_gt_memory_003**: PASS
- **aep_cost_norm_best**: FAIL
- **aep_far_ood_gt_memory**: FAIL
- **aep_scales_better**: PASS
- **aep_not_dominated_by_small_budget_pot**: PASS
- **inference_ok**: PASS

---
### Key Findings

1. AEP/Residual achieve strong absolute goal-transfer (AEP=0.954, RC=0.953) vastly exceeding PolicyClone (0.261).
2. Cost-normalized analysis reveals PrototypeOutcomeTable has highest efficiency (0.1664 match/kB).
3. RawMemoryOutcomeTable is competitive (0.896 absolute, 0.1799 cost-norm) — explicit memory is an efficient capital form.
4. On far OOD extrapolation, neural compression's advantage is: AEP=0.600 vs RMOT=0.881.
5. Dataset scaling slopes: AEP=0.0170, RMOT=0.0051, POT=0.0014.

---
### All IC-2d Outputs

| File | Content |
|---|---|
| `results/ic2d/cost_breakdown.csv` | Detailed cost per mechanism |
| `results/ic2d/memory_scaling_curve.csv` | Memory budget vs Prototype sweep |
| `results/ic2d/dataset_scaling_curve.csv` | N train states vs match |
| `results/ic2d/state_extrapolation.csv` | ID / near / far OOD |
| `results/ic2d/utility_complexity_scaling.csv` | 7 complexity levels |
| `results/ic2d/fair_memory_planner.csv` | Fair memory planner |
| `results/ic2d/final_cost_normalized_verdict.csv` | Final cost-norm per mechanism |
| `results/ic2d/IC2D_COST_NORMALIZED_CAPITAL_AUDIT.md` | **This report** |