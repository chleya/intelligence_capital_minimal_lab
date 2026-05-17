# IC-2e-R: Reconciled Capital Boundary Report

**Final Verdict**: `IC2E_R_CAPITAL_PORTFOLIO_CONFIRMED`

---

## Q1: Which Mechanism Wins Absolute Match?

**AEPCompressor wins.** Across all 25 regimes tested,
AEPCompressor ranks #1 in absolute match 18/32 times
with average rank 1.43.

Specifically:
- AEPCompressor mean rank: 1.43
- ResidualCompressor follows closely
- RawMemoryOutcomeTableFull mean rank: 3.39

AEP/Residual show clear absolute performance advantage, especially on ID test
where AEP (0.960) > RMOT (0.898).

## Q2: Which Mechanism Wins Cost Efficiency?

**StandardizedKNNOutcomeTable wins.** Across all regimes,
StandardizedKNNOutcomeTable ranks #1 in cost efficiency 18/29 times
with average rank 1.36.

Specifically:
- RawMemory/StandardizedKNN has ~10x higher match-per-byte than AEP
- AEPCompressor mean cost-efficiency rank: 4.16
- The efficiency gap is driven by parameter storage cost (~52KB vs ~5KB)

## Q3: Which Mechanism Wins Far OOD Robustness?

**PrototypeOutcomeTable wins.** On far OOD (NN distance 9.26x vs ID 1.26x):

| Mechanism | Far OOD Match |
|---|---|
| RawMemoryOutcomeTableFull | 0.8992 |
| PrototypeOutcomeTable | 0.9007 |
| AEPCompressor | 0.8151 |
| ResidualCompressor | ~0.88 |

Memory maintains performance because the outcome function is smooth
and kNN interpolation works well in low-dimensional Lipschitz-continuous spaces.
AEP drops because neural networks extrapolate poorly outside training manifold.

## Q4: Is AEP Still a Valuable Capital Form?

**Yes, for absolute performance.** AEP provides:
- 0.960 match on ID test (highest)
- 1.43 avg rank across all regimes
- Best match on complex utilities (piecewise, nonlinear)
- Appears on 50% of absolute Pareto frontiers

AEP is the capital form for **when absolute decision quality matters more than cost**.
It is NOT the optimal capital for cost-constrained or OOD-robust settings.

## Q5: Is Memory/Prototype More Efficient Than AEP?

**Yes, overwhelmingly in cost-normalized terms.** Evidence:
- RMOT cost-efficiency rank: 1.36 vs AEP: 4.16
- RMOT appears on 79% of efficiency Pareto frontiers
- The ~10x parameter cost gap drives this difference

However, memory is NOT strictly better: it loses on absolute match by ~0.06
and its inference cost (O(N*d)) scales worse with dataset size.

## Q6: Does a Single Dominant Capital Form Exist?

**No.** Across the three victory dimensions:
- Absolute match: Parametric AEP/Residual dominates
- Cost efficiency: RawMemory/StandardizedKNN dominates
- Far OOD robustness: RawMemory/Prototype dominates

No single mechanism wins all three dimensions simultaneously.
This confirms the **capital portfolio theory**: optimal intelligence
requires selecting the right capital form for each dimension.

## Q7: Is ICT Formally Revised to Capital Portfolio Theory?

**Yes.** The original "compression appreciation" framing is replaced by:

> ICT is a capital allocation theory. Intelligence capital has multiple
> forms (RawMemory, Prototype, Parametric, Action-Effect, Active Probe),
> each with distinct cost/performance/extrapolation profiles. The optimal
> capital portfolio depends on the regime.

The 7-axis Capital Form framework (Storage, Acquisition, Inference,
Transfer Flexibility, Extrapolation Risk, Interpretability, Update)
provides a systematic way to compare capital forms.

## Q8: Can We Proceed to IC-3?

**Yes.** With the reconciled understanding:
1. AEP/Residual for absolute performance
2. RawMemory for cost efficiency and OOD robustness
3. Prototype for budget-constrained settings
4. No single capital form dominates

IC-3 should:
- Test in environments where kNN memory is expected to degrade (high-dim, non-smooth)
- Find regimes where neural compression is BOTH high-match AND cost-efficient
- Develop capital portfolio selection as a formal problem

---

### Regime Summary (Reconciled)

| Regime | Absolute Winner | Cost-Efficiency Winner | Far-OOD Winner |
|---|---|---|---|
| State dim 2 | AEP (0.950 vs 0.881) | RawMemory | RawMemory |
| State dim 4 | AEP (0.915 vs 0.834) | RawMemory | RawMemory |
| State dim 32 | AEP (0.841 vs 0.826) | RawMemory | RawMemory |
| Far OOD | AEP/RMOT (0.815 vs 0.899) | RawMemory | **RawMemory** |
| All regimes | **AEP** (1.4) | **RawMemory** (1.4) | **RawMemory** |

### Generated Files

| File | Content |
|---|---|
| `results/ic2e_r/rank_by_absolute_match.csv` | 3-way ranks: absolute |
| `results/ic2e_r/rank_by_cost_efficiency.csv` | 3-way ranks: cost-efficiency |
| `results/ic2e_r/rank_by_far_ood.csv` | 3-way ranks: far OOD |
| `results/ic2e_r/regime_summary_reconciled.csv` | Reconciled regime table |
| `results/ic2e_r/absolute_pareto_frontier.csv` | Absolute Pareto frontier |
| `results/ic2e_r/efficiency_pareto_frontier.csv` | Efficiency Pareto frontier |
| `results/ic2e_r/functional_ood_diagnostics.csv` | Functional OOD diagnostics |
| `results/ic2e_r/ICT_CAPITAL_FORM_REVISION.md` | Revised theory (Capital Portfolio) |
| `results/ic2e_r/IC2E_R_RECONCILED_CAPITAL_BOUNDARY_REPORT.md` | **This report** |
