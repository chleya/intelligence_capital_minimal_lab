# IC-3A: Minimal Performance-Reported Capital Allocator Report

**Final Verdict**: `IC3A_SECOND_ORDER_ALLOCATOR_SUPPORTED`

---

## Evaluation Results

| Allocator | Mean Correct | vs BestSingle | vs Uniform | vs Random |
|---|---|---|---|---|
| OracleHindsight | 0.7500 | ++0.2060 | ++0.3500 | ++0.3900 |
| **MetaMLPAllocator** | **0.5880** | +0.0440 | +0.1880 | +0.2280 |
| LearnedBanditAllocator | 0.3400 | -0.2040 | -0.0600 | -0.0200 |
| BestSingleCapital | 0.5440 | — | ++0.1440 | ++0.1840 |
| UniformPortfolio | 0.4000 | — | — | ++0.0400 |
| RandomAllocator | 0.3600 | — | — | — |

## Q&A

**Q1: Does Allocator beat BestSingleCapital?**
YES — MetaMLP=0.5880 vs BestSingle=0.5440 (Δ=+0.0440)

**Q2: Cost-normalized?**
Cost-normalized regret: MetaMLP=0.4120 vs BestSingle=0.4560

**Q3: Beats Uniform?**
YES — MetaMLP=0.5880 vs Uniform=0.4000

**Q4: Uses GoalInferenceCapital on hidden-goal?**
GoalInference chosen on Task D: 156.0/4 steps (3900.0%)

**Q5: Biases toward PolicyCapital on fixed-goal?**
See eval capital choice distribution

**Q6: Biases toward ParametricCompression on goal-transfer?**
See eval capital choice distribution

**Q7: Biases toward PrototypeMemory on dense-support?**
See eval capital choice distribution

**Q8: Reduces weight of impaired capital after task switch?**
Impairment detector: 2/4 impaired, 2/4 healthy

**Q9: Can this be called a second-order intelligence prototype?**
YES — exceeds BestSingleCapital by Δ=+0.0440

## Death Conditions

| Condition | Status |
|---|---|| D1_allocator_lte_best_single | OK |
| D2_only_beats_uniform | OK |
| D3_feature_engineering_regression | OK |
| D4_goal_inference_failure | OK |
| D5_negative_transfer_failed | OK |
| D6_external_validation_failed | OK |

## Files Generated

| File |
|---|
| results/ic3a/capital_reports.csv |
| results/ic3a/allocator_performance.csv |
| results/ic3a/capital_weight_traces.csv |
| results/ic3a/regret_curves.csv |
| results/ic3a/cost_normalized_regret.csv |
| results/ic3a/external_validation_split.csv |
| results/ic3a/death_conditions.csv |
| IC3A_MINIMAL_PERFORMANCE_REPORTED_ALLOCATOR_REPORT.md |
