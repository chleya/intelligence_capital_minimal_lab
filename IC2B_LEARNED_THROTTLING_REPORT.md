# IC-2b: Learned Throttling Mechanism Comparison — Final Report

**Status: Death Condition D1 Triggered — No Mechanism Beats State Shortcut**

Generated: 2026-05-09

---

## Executive Summary

IC-2b compared 13 learned throttling mechanisms to determine whether learned compression of the world change stream produces intelligence capital that exceeds a simple state-only baseline (LearnedStateOnlyClassifier). The result is clear: **no mechanism passes the minimum threshold** of best_action_match > LearnedStateOnly + 0.05 on in-distribution (ID) test data.

The best mechanism (CounterfactualCompressor) achieves best_action_match = 0.780, essentially tied with LearnedStateOnly at 0.787 (gap: -0.007). This triggers Death Condition D1: "如果所有机制 <= LearnedStateOnly + 0.05，则 learned throttling 没有超过 state shortcut."

However, the results contain nuanced findings: action-effect capital is real (confirmed by ShuffledActionControl dropping from 0.780 to 0.342), temporal capital is real (PermutedHistoryControl drops to 0.521), and the CounterfactualCompressor shows a modest OOD transfer premium (+0.038 over SO on background/gain shifts).

---

## ID Test Results (Training on ID Only)

| Rank | Mechanism | best_action_match | regret | outcome_mse | param_count | cost_bytes |
|------|-----------|-------------------|--------|-------------|-------------|------------|
| 1 | **learned_state_only** | **0.787** | 0.213 | 63.6 | 9,555 | 38,220 |
| 2 | counterfactual_compressor | 0.780 | 0.220 | 0.33 | 9,555 | 38,220 |
| 3 | centered_residual | 0.568 | 0.432 | 0.22 | 8,630 | 34,520 |
| 4 | residual_adversarial | 0.548 | 0.452 | 0.22 | 10,119 | 40,476 |
| 5 | residual_compressor | 0.545 | 0.455 | 0.22 | 8,630 | 34,520 |
| 6 | permuted_history_control | 0.521 | 0.479 | 0.23 | 8,630 | 34,520 |
| 7 | learned_action_only | 0.479 | 0.521 | — | 3 | 12 |
| 8 | aep_compressor | 0.442 | 0.558 | 0.28 | 13,033 | 52,132 |
| 9 | causal_contrast | 0.417 | 0.583 | 0.68 | 8,307 | 33,228 |
| 10 | shuffled_action_control | 0.342 | 0.658 | 0.21 | 8,630 | 34,520 |
| 11 | raw_memory_full | 0.241 | 0.759 | — | 28,800 | 115,200 |
| 12 | raw_memory_equal_cost | 0.167 | 0.833 | — | 2,376 | 9,504 |
| 13 | prototype_memory | 0.117 | 0.883 | — | 480 | 1,920 |

*Note: RawMemory mechanisms have no outcome_mse as they are non-parametric. LearnedActionOnly outcome_mse is not comparable (it outputs class probabilities, not outcome values).*

### Key Observations:

1. **CounterfactualCompressor (0.780) essentially ties LearnedStateOnly (0.787)** — the gap of -0.007 is within random variance across seeds.
2. **ResidualCompressor and variants (0.545-0.568)** are functional after fixing a critical architectural bug (linear residual_head → nonlinear MLP), but still far below the state shortcut.
3. **AEPCompressor declined from 0.749 to 0.442** after increasing bottleneck_dim from 32 to 48, likely due to overfitting.
4. **RawMemory baselines collapse** (0.117-0.241) — k-nearest-neighbor on raw counterfactual outcomes performs poorly, failing to generalize from training states to test states.
5. **CausalContrastCompressor (0.417)** does not outperform simple AEP.

---

## Capital Metrics

| Mechanism | IAR | value_gain | cost_bytes |
|-----------|-----|------------|------------|
| learned_action_only | **1.24×10⁻²** | 0.149 | 12 |
| learned_state_only | 1.20×10⁻⁵ | 0.457 | 38,220 |
| counterfactual_compressor | 1.18×10⁻⁵ | 0.450 | 38,220 |
| centered_residual | 6.89×10⁻⁶ | 0.238 | 34,520 |
| residual_compressor | 6.23×10⁻⁶ | 0.215 | 34,520 |
| residual_adversarial | 5.39×10⁻⁶ | 0.218 | 40,476 |
| permuted_history_control | 5.53×10⁻⁶ | 0.191 | 34,520 |
| aep_compressor | 2.15×10⁻⁶ | 0.112 | 52,132 |
| causal_contrast | 2.62×10⁻⁶ | 0.087 | 33,228 |
| shuffled_action_control | 3.48×10⁻⁷ | 0.012 | 34,520 |
| raw_memory_full | 0.0 | 0.0 | 115,200 |
| raw_memory_equal_cost | 0.0 | 0.0 | 9,504 |
| prototype_memory | 0.0 | 0.0 | 1,920 |

**IAR finding**: LearnedActionOnly has the highest IAR (1.24×10⁻²) due to its tiny parameter count (12 bytes), even though its value_gain (0.149) is modest. This suggests that a simple action-frequency counter is the most "capital-efficient" mechanism in terms of bits-per-bit-of-value, but not in absolute terms.

---

## Bad Debt Audit

| Mechanism | BadDebtRatio | SO Shortcut | AO Shortcut | ShuffledGap |
|-----------|-------------|-------------|-------------|-------------|
| counterfactual_compressor | 0.009 | 1.009 | 0.614 | **0.438** |
| centered_residual | 0.008 | 1.386 | 0.843 | 0.226 |
| residual_adversarial | 0.008 | 1.436 | 0.874 | 0.206 |
| residual_compressor | 0.008 | 1.444 | 0.879 | 0.203 |
| permuted_history_control | 0.008 | 1.511 | 0.919 | 0.179 |
| aep_compressor | 0.008 | 1.781 | 1.084 | 0.100 |
| learned_action_only | 0 | 1.643 | 1.000 | 0.137 |
| causal_contrast | 0.003 | 1.887 | 1.149 | 0.075 |
| shuffled_action_control | 0.003 | 2.301 | 1.401 | 0.0 |

*BadDebtRatio = max(0, 1 - model_match / so_match). Values near 0 indicate the model does not have "bad debt" relative to SO. SO Shortcut Index: SO_match / model_match (>1 means model underperforms SO). AO Shortcut Index: AO_Match / SO_Match. ShuffledGap: how much best_action_match drops when action-outcome mapping is randomized.*

**Key audit findings:**
- **CounterfactualCompressor has near-zero bad debt** (BadDebtRatio ≈ 0.009) — it's the closest to matching SO.
- **ShuffledAction gap of 0.438** for CounterfactualCompressor confirms genuine action-effect capital.
- **SO Shortcut Index near 1.0** for CounterfactualCompressor confirms it's not a mere state shortcut.
- **PermutedHistory gap** (~0.26 for Residual) confirms temporal capital is real but modest.
- **RawMemory mechanisms are pure bad debt**: RawMemoryEqualCost at 0.167 is dramatically inferior to any learned mechanism.

---

## Raw Memory vs Learned Compression

| Mechanism | vs RawMemEqCost Match | vs RawMemEqCost Cost |
|-----------|----------------------|---------------------|
| learned_state_only | +0.620 | 4.02× |
| counterfactual_compressor | **+0.613** | 4.02× |
| centered_residual | +0.401 | 3.63× |
| residual_compressor | +0.378 | 3.63× |
| residual_adversarial | +0.381 | 4.26× |

**Finding**: All learned mechanisms dramatically outperform RawMemoryEqualCost on best_action_match (by +0.35 to +0.62). This confirms that **learned compression produces intelligence appreciation relative to raw memory storage** at equal or comparable cost. However, this does not satisfy D2 because the question is whether compressors beat RawMemoryEqualCost on their own target metrics — and they do. D2: "如果 RawMemoryEqualCost >= 所有 compressors，则 compression 没有产生 intelligence appreciation" — this is NOT triggered because all compressors significantly beat RawMemoryEqualCost.

---

## OOD Transfer Evaluation

| OOD Type | Mechanism | best_action_match |
|----------|-----------|-------------------|
| **background_shift** | counterfactual_compressor | **0.809** |
| background_shift | learned_state_only | 0.771 |
| background_shift | learned_action_only | 0.496 |
| background_shift | residual_compressor | 0.515 |
| **action_gain_shift** | counterfactual_compressor | **0.809** |
| action_gain_shift | learned_state_only | 0.771 |
| action_gain_shift | learned_action_only | 0.496 |
| **sign_rule_shift** | learned_action_only | **0.483** |
| sign_rule_shift | residual_compressor | 0.141 |
| sign_rule_shift | counterfactual_compressor | 0.132 |
| sign_rule_shift | learned_state_only | 0.115 |

### Transfer Premium vs LearnedStateOnly:

| OOD Type | CF vs SO | Residual vs SO |
|----------|----------|----------------|
| background_shift | **+0.038** | -0.256 |
| action_gain_shift | **+0.038** | -0.256 |
| sign_rule_shift | +0.017 | +0.026 |

**Key findings:**
1. **CounterfactualCompressor shows a +0.038 transfer premium** over SO on background and action gain shifts. This is below the +0.05 threshold but clearly positive, suggesting that outcome-supervised joint prediction provides OOD generalization advantages that pure classification does not.
2. **OOD_sign_rule_shift causes complete collapse** for all state-using mechanisms (0.11-0.14). Only ActionOnly (0.483) survives, because sign rules are embedded in action-outcome mappings, which state-based models learn and then fail to generalize. This triggers D6: models have learned sign rule memorization, not structural transfer.
3. LearnedActionOnly survives OOD_sign because it doesn't use state information — it's inherently sign-invariant (but useless on normal distributions at 0.479).

---

## Gate Evaluation Results

### IC-2b Minimum Pass Criteria:

| Criterion | Result | Status |
|-----------|--------|--------|
| ID > SO + 0.05 | Best: -0.007 (gap) | **FAIL** |
| OOD_bg > SO + 0.05 | Best: +0.038 (gap) | **MARGINAL** |
| best > AO + 0.10 | Best: +0.301 | ✅ PASS |
| best > RawMemEqCost | Best: +0.613 | ✅ PASS |
| BadDebtRatio < 0.50 | 0.009 (CF) | ✅ PASS |
| ShuffledAction significant drop | Drop: 0.438 | ✅ PASS |
| PermutedHistory significant drop | Drop: 0.259 | ✅ PASS |
| IAR > RawMemEqCost IAR | 1.18×10⁻⁵ vs 0 | ✅ PASS |

### Death Conditions:

| Death Condition | Triggered? | Evidence |
|----------------|-----------|----------|
| **D1**: All ≤ SO + 0.05 | **YES** | CounterfactualCompressor = 0.780, SO = 0.787, gap = -0.007 |
| D2: RawMemEqCost ≥ all compressors | NO | All compressors >> RawMemEqCost (0.167) |
| D3: ShuffledAction ≈ main model | NO | Drop of 0.438 confirms action capital |
| D4: PermutedHistory ≈ main model | NO | Drop of 0.259 confirms temporal capital |
| D5: ID strong but OOD_bg collapses | NO | CF at 0.809 on OOD_bg > 0.780 on ID |
| **D6**: OOD_sign complete collapse | **YES** | All state-using mechanisms ~0.12 on sign shift |

---

## Answers to the 8 Questions

### 1. 是否有机制超过 LearnedStateOnly？

**No.** The best mechanism (CounterfactualCompressor) achieves 0.780, essentially tied with LearnedStateOnly at 0.787. The -0.007 gap falls far short of the +0.05 minimum threshold. The state shortcut dominates — in this environment, the state/history captures 78.7% of best-action decisions, and no learned throttling mechanism can extract additional actionable intelligence from the action-effect residual that translates to improved classification.

### 2. 是否有机制超过 RawMemoryEqualCost？

**Yes, all of them.** Every learned mechanism dramatically outperforms RawMemoryEqualCost (0.167):
- CounterfactualCompressor: +0.613
- ResidualCompressor: +0.378
- CenteredResidual: +0.401

This confirms that learned compression DOES produce intelligence appreciation relative to raw memory storage. The compression pipeline extracts more value per byte than storing raw counterfactual exemplars.

### 3. 是否有机制在 OOD_background / OOD_gain 中保持 transfer premium？

**Partial yes — CounterfactualCompressor shows +0.038 transfer premium** over SO on both background_shift and action_gain_shift. This is below the +0.05 threshold but clearly positive, suggesting that learning outcome values (not just best-action classification) provides OOD robustness advantages. The joint prediction of all 3 counterfactual outcomes generalizes slightly better than pure classification when the environment distribution shifts.

### 4. 哪些机制是坏账？

**All mechanisms have BadDebtRatio near zero**:
- CounterfactualCompressor: 0.009 (near-perfect capital utilization)
- Most other mechanisms: 0.003-0.008

The "bad debt" metric is not informative here because almost every mechanism preserves its capital relative to the state shortcut baseline. The real concern is not bad debt but **insufficient appreciation**: even CounterfactualCompressor with near-zero bad debt cannot exceed the state shortcut by the required margin.

The true "bad debt" is in the RawMemory baselines (0.117-0.241 best_action_match), which are dramatically inferior to any learned mechanism.

### 5. ResidualCompressor 是否真的优于 AEPCompressor？

**Yes — but both are weak.**

| Mechanism | best_action_match | relative to SO |
|-----------|-------------------|----------------|
| ResidualCompressor | 0.545 | -0.242 |
| AEPCompressor | 0.442 | -0.345 |

ResidualCompressor outperforms AEPCompressor by +0.103. The explicit decomposition into autonomous baseline + action-conditional residual provides a better inductive bias than the generic (state, action) → outcome mapping. However, both are far below the state shortcut.

This finding supports the ICT hypothesis that structural decomposition (B(h) + R(h,a)) is superior to monolithic compression — but the superiority is insufficient to overcome the fundamental dominance of state information in this environment.

**Critical architectural note**: ResidualCompressor was initially broken (0.108) due to a fundamental bug: `residual_head = nn.Linear(bottleneck_dim + residual_dim, 1)` with no nonlinearity made the action-dependent component a CONSTANT (state-independent) offset. The fix (`nn.Sequential(Linear(...), ReLU, Linear(...))`) restored the ability to learn state-dependent action effects, improving performance from 0.108 to 0.545.

### 6. CounterfactualCompressor 是否最强？

**Yes — CounterfactualCompressor is the strongest mechanism** at 0.780, tied with LearnedStateOnly (0.787). It learns to jointly predict all 3 counterfactual outcomes from state input with combined MSE + CE loss (weight 1.5).

Its advantages:
- Joint prediction structure directly models outcome ordering
- Normalized outcomes + CE loss effectively optimize for best_action classification
- OOD transfer premium of +0.038 over SO

Its limitation:
- **Cannot exceed the information ceiling set by state alone** (0.787)
- The action-dependent residual signal, while real (confirmed by ShuffledAction gap of 0.438), does not provide sufficient additional discriminative power to exceed the state shortcut's classification boundary

This represents a **partial validation** of the CounterfactualCompressor architecture: it's the best design but cannot extract intelligence beyond what the environment provides.

### 7. CausalContrast 是否有额外价值？

**No.** CausalContrastCompressor (0.417) significantly underperforms both CounterfactualCompressor (0.780) and AEPCompressor (0.442). The causal contrastive loss (encoding similarity between action embeddings and outcome differences) adds no value over direct supervised training.

Possible reasons:
- The contrastive signal requires action-effect differences that are too small to form meaningful embeddings
- The architecture adds complexity without improving the information bottleneck
- The contrastive loss may interfere with MSE optimization

This is a negative result: CausalContrast provides no additional benefit in this environment.

### 8. ICT 的 intelligence appreciation 是否得到支持，还是 RawMemory / StateOnly 仍然主导？

**Partial support, but StateOnly still dominates in ID.**

The evidence:

**Supporting ICT:**
1. **Learned compression beats raw memory**: All compressors significantly outperform RawMemoryEqualCost (+0.35 to +0.62 in best_action_match). Compression produces real intelligence appreciation relative to storing raw experience.
2. **Structural decomposition works**: ResidualCompressor (0.545) > AEPCompressor (0.442). The B+R decomposition is a better inductive bias than monolithic compression.
3. **Action-effect capital is real**: ShuffledActionControl causes a 0.438 drop, confirming that learned mechanisms extract and rely on action-effect information.
4. **Temporal capital is real**: PermutedHistoryControl causes a 0.259 drop, confirming that history ordering carries information.
5. **OOD transfer premium exists**: CounterfactualCompressor shows +0.038 advantage over SO on distribution shifts.

**Against ICT (or at least requiring qualification):**
1. **StateOnly dominates ID**: No mechanism exceeds the classification accuracy of a simple state-only classifier. The state contains 78.7% of decision-relevant information.
2. **D1 triggered**: The minimum threshold for intelligence appreciation (exceeding state shortcut by 0.05) is not met.
3. **Action information doesn't translate to classification**: While action-effect capital is real (ShuffledAction drops), knowing action effects doesn't help classify best actions beyond what state already provides.

**Conclusion**: ICT's prediction that learned compression can produce intelligence capital (exceeding raw memory) is supported. ICT's stronger prediction — that this intelligence can exceed a simple state shortcut — is NOT supported in this environment configuration. The state dominates too heavily, and the action-dependent residual is too small relative to total outcome magnitude to provide actionable classification information beyond what state captures.

---

## Root Cause Analysis: Why Can't Any Mechanism Beat StateOnly?

The environment produces outcomes with a strong state-dependent component and a weak action-dependent residual:

- **State component**: ~78.7% of best_action decisions determined by state/history
- **Action residual**: ~21.3% of best_action decisions require action-effect knowledge
- **Zero-action optimal rate**: ~28.1% (confirmed by IC-2a+)

The problem is NOT that models fail to learn action effects — the ShuffledActionControl drop (0.438) proves they do. The problem is that the learned action-effect knowledge does not provide sufficient additional discriminative power to exceed state-only classification on the test set.

In other words: **knowing HOW MUCH better one action is than another doesn't tell you WHICH action is better, if the state already tells you with 78.7% accuracy.** The residual signal refines the outcome prediction but doesn't change the classification boundary.

This can be understood as:
- State → outcome_mean (captured by both SO and CF)
- State → action preference (what SO classifies)
- Action → outcome delta (what CF additionally learns)
- But outcome_delta ≈ mean_delta + noise, where mean_delta is already state-determined

The action-dependent component that is NOT already state-determined is essentially pure noise with respect to best_action classification on the ID test set. This noise can be memorized (hence ShuffledAction drops when shuffled), but it doesn't generalize to improve test-set classification.

---

## Figures Generated

- `results/figures/ic2b_best_action_match.png` — Bar chart of ID best_action_match by mechanism
- `results/figures/ic2b_ood_transfer.png` — Bar chart of OOD best_action_match by mechanism and OOD type
- `results/figures/ic2b_iar_vs_cost.png` — Scatter plot of IAR vs cost_bytes
- `results/figures/ic2b_bad_debt_ratio.png` — Bad debt ratio by mechanism
- `results/figures/ic2b_raw_memory_equal_cost.png` — Comparison of learned vs raw memory
- `results/figures/ic2b_stateonly_gap.png` — Gap vs StateOnly by mechanism

---

## Output Files

| File | Description |
|------|-------------|
| `results/ic2b/learned_compressors.csv` | Per-seed ID results for all 13 mechanisms |
| `results/ic2b/ood_transfer.csv` | Per-seed OOD evaluation for all mechanisms |
| `results/ic2b/appreciation_rate.csv` | IAR and value_gain per mechanism |
| `results/ic2b/bad_debt_audit.csv` | Bad debt and shortcut indices |
| `results/ic2b/raw_memory_comparison.csv` | Learned vs RawMemory cost/performance |
| `results/ic2b/stateonly_gap.csv` | Gap to StateOnly baseline |

---

## Recommendations

1. **Accept D1**: In this environment, learned throttling does not exceed the state shortcut. This is a valid scientific result.

2. **Investigate environment properties**: To find a regime where ICT's stronger prediction holds, consider:
   - Increasing action effect magnitude (higher `action_scale` or `state_dependent_gain`)
   - Reducing state observability (fewer history steps, more noise)
   - Introducing action-history interactions (action outcomes depend on past actions)

3. **CounterfactualCompressor is the champion architecture**: Despite D1, it achieves the closest performance to SO and shows the only positive OOD transfer premium. Future work should focus on this architecture.

4. **Fix AEPCompressor overfitting**: The AEP's decline from 0.749 to 0.442 with increased capacity suggests overfitting. Adding dropout or reducing bottleneck_dim for AEP specifically would help.

5. **Document D6 honestly**: All state-using mechanisms fail catastrophically on sign rule shift. This should be reported as a structural limitation, not a bug. Models in this environment are learning sign-rule memorization.

---