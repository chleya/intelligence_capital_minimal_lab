# IC-2a Experiment Plan: Oracle Residual Accounting

## Objective

Prove that a hidden-mode environment exists where oracle access to action-effect residuals beats all shortcuts (StateOnly, ActionOnly).

## Design

1. Generate StructuredVolatilityEnv with hidden binary mode and action-sign-flip
2. Generate counterfactual table: for each state, record outcomes for all 3 actions
3. Compute oracle metrics:
   - Residual variance ratio (M3)
   - Oracle residual match vs StateOnly match (M7)
   - CF value (M12)
   - Seed stability (M24)

## Gate Conditions

| Metric | Threshold | Death If |
|--------|-----------|----------|
| Residual Variance Ratio | ≥ 0.15 | D1 |
| Oracle beats StateOnly | oracle_match > so_match | D2 |
| CF has value | counterfactual_value > 0 | D3 |
| Benchmark stable | seed_stability_ratio < 0.5 | D7 |
| AO < 40% | action_only_ceiling < 0.40 | env not hard |

## Execution

```bash
python -m src.run_ic2a_oracle_residual
```

Output: `results/ic2a_gates.json`, `results/counterfactual_table.csv`

## Decision

- ALL gates pass → proceed to IC-2b
- ANY gate fails → redesign environment and re-run