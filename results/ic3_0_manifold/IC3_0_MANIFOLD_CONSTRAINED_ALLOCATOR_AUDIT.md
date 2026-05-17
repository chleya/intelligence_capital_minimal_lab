# IC-3-0: Manifold-Constrained Capital Allocator Audit

## Verdict: IC3_0_MANIFOLD_CONSTRAINED_ALLOCATOR_SUPPORTED

## 1. Performance Comparison

| Allocator | Mean Reward | Regret |
|---|---|---|
| Unconstrained | 0.4260 | -0.0519 |
| SimplexWeight | 0.3778 | -0.0037 |
| BirkhoffTransition | 0.3994 | -0.0254 |
| EqualWeight | 0.3964 | -0.0224 |
| BestSingle | 0.3752 | -0.0012 |
| OracleHindsight | 0.3740 | 0.0000 |

## 2. Stability Diagnostics

| Metric | Unconstrained | SimplexWeight | BirkhoffTransition |
|---|---|---|---|
| weight_entropy | 0.9980 | 0.5778 | 1.0000 |
| max_weight | 0.2734 | 0.7660 | 0.2500 |
| weight_turnover_rate | 0.0021 | 0.0002 | 0.0000 |
| oscillation_energy | 0.0000 | 0.0000 | 0.0000 |
| weight_collapse_ratio | 0.0000 | 0.0000 | 0.0000 |
| capital_trust_explosion_score | 0.0134 | 0.0118 | 0.0000 |
| transition_norm | 0.0000 | 0.0000 | 1.0000 |
| row_sum_error | 0.0000 | 0.0000 | 0.0000 |
| col_sum_error | 0.0000 | 0.0000 | 0.0000 |

## 3. Death Conditions

| Condition | Status | Detail |
|---|---|---|
| D1_weight_collapse | ✅ OK | collapse_ratio simplex=0.000 birkhoff=0.000 |
| D2_transition_sum_error | ✅ OK | row_err=0.000001 col_err=0.000001 (tol=0.05) |
| D3_weight_oscillation | ✅ OK | turnover simplex=0.0002 birkhoff=0.0000 |
| D4_unconstrained_vs_constrained | ✅ OK | unconstr_perf=0.4260 simplex=0.3778 birkhoff=0.3994 | turnover unconstrained=0.0021 simplex=0.0002 |
| D5_constrained_lt_best_single | ✅ OK | constrained_best=0.3994 BestSingle=0.3752 |

## 4. Research Questions

**Q1: Does manifold constraint reduce weight oscillation?**
  Unconstrained turnover: 0.00214
  Simplex turnover:       0.00019
  Birkhoff turnover:      0.00000
  Answer: YES — constrained reduces weight churn

**Q2: Does it prevent bad-capital amplification?**
  Unconstrained collapse ratio: 0.000
  Simplex collapse ratio:       0.000
  Birkhoff collapse ratio:      0.000
  Answer: NO — both near zero in this regime

**Q3: Does it maintain or improve cost-normalized regret?**
  Simplex regret:       -0.0037
  Birkhoff regret:      -0.0254
  Unconstrained regret: -0.0519
  Answer: See comparison table in Section 1

**Q4: Is Birkhoff transition more stable than simple softmax?**
  Birkhoff turnover: 0.00000
  Simplex turnover:  0.00019
  Answer: YES — Birkhoff is more stable (lower turnover)

**Q5: Does it qualify for formal IC-3 entry?**
  Answer: YES — ALL death conditions passed. Manifold-constrained allocation is ready for IC-3.
