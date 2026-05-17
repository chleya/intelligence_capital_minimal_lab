# Metrics & Death Conditions

**Status:** Intelligence Capital Theory IC-0  
**Date:** 2026-05-09  

---

## Master Table: All Metrics with Death Conditions

| Metric ID | Metric | Measures | Formula Sketch | Baseline | Death Condition |
|---|---|---|---|---|---|
| M1 | effective_change_events | Count of non-trivial state transitions | Σ 1[\|o_{t+1} - o_t\| > ε] | — | If count < 100, not enough signal |
| M2 | retained_change_ratio | Compression degree | \|S_stored\| / \|D_stream\| | 1.0 (RawMemory) | If > 0.9, no meaningful throttling |
| M3 | residual_variance_ratio | Action-effect signal strength | Var(Y(h,a) - Y(h,0)) / Var(Y(h,a)) | 0 (no residual) | **< 0.15 → DEATH (D1)** |
| M4 | oracle_action_entropy | Diversity of optimal actions | -Σ p(action_a*) log p(action_a*) | — | **< 0.85 → gates fail** |
| M5 | action_only_ceiling | Trivial action predictor match | match(best_action_rule) | — | **> 0.40 → env not hard enough** |
| M6 | state_only_ceiling | Autonomous dynamics predictor match | match(SO_predictor) | — | **> model_match → DEATH (D2)** |
| M7 | residual_oracle_match | Oracle knows true residual | match(argmax_a R_oracle(h,a)) | SO, AO | **≤ SO → DEATH (D2)** |
| M8 | prediction_dividend | Raw prediction gain over baseline | match(model) - match(baseline) | 0 (no gain) | **≤ 0 → no prediction value** |
| M9 | control_dividend | Action-selection gain | regret(baseline) - regret(model) | 0 (no gain) | **≤ 0 → no control value** |
| M10 | transfer_premium | OOD benefit beyond expected | OOD_match - predicted_from_ID_model | 0 | **≤ 0 → no transfer** |
| M11 | structural_dividend | Same structure, new context gain | V(S, C_new) - V(S, C_old) | 0 | **≈ 0 → structure is memorization** |
| M12 | counterfactual_value | CF data marginal benefit | match(FullCF) - match(traj_only) | 0 | **≤ 0 → CF data worthless (D3)** |
| M13 | 20pct_cf_efficiency | Partial CF efficiency | gain(20%CF) / gain(100%CF) | 0.2 (proportional) | **< 0.70 → CF inefficient** |
| M14 | IAR (Intelligence Appreciation Rate) | Value per unit cost | [V(S,C_new) - V(Baseline,C_new)] / Cost(S) | 0 | **≤ RawMemoryEqualCost → memory cheaper (D4)** |
| M15 | ORR (Operational Realization Rate) | How much prediction → control | control_dividend / prediction_dividend | 1.0 (full realization) | **< 0.5 → prediction ≠ control** |
| M16 | bad_debt_ratio | Shortcut-explained fraction | max(SO_gain, AO_gain, Shuff_gain) / claimed_gain | 1.0 (all bad debt) | **> 0.5 → mostly bad debt (D5)** |
| M17 | state_only_shortcut_index | SO explains model performance | match(SO) / match(model) | 1.0 (SO ≥ model) | **> 0.90 → mostly autonomous shortcut** |
| M18 | action_only_shortcut_index | AO explains model performance | match(AO) / match(model) | 1.0 (AO ≥ model) | **> 0.80 → mostly action-main-effect shortcut** |
| M19 | shuffled_action_gap | Causal usage of action info | match(model) - match(shuffled_action_model) | 0 | **< 0.05 → not using action causally** |
| M20 | permuted_history_gap | Temporal order usage | match(model) - match(permuted_history_model) | 0 | **< 0.10 → temporal model broken (D6)** |
| M21 | compression_to_value_ratio | Cost efficiency | Cost(S) / V(S, C) | — | If increasing over time, capital decaying |
| M22 | capital_decay | Value erosion over time | V(S, t+δ) / V(S, t) | 1.0 (stable) | < 0.5 after δ steps → unstable |
| M23 | operational_leverage | Prediction amplifies control | control_dividend / cost | 0 | ≤ 0 → negative leverage |
| M24 | seed_stability_ratio | Benchmark reliability | std(model_match) / mean(model_gap) | 0 (perfect) | **> 0.5 → benchmark unstable (D7)** |
| M25 | ood_kl_control | Distribution shift robustness | KL(oracle_dist_train, oracle_dist_test) | 0 | High KL + low match → fake OOD |

---

## Death Conditions Master Table

| ID | Name | Threshold | Phase | Meaning if triggered |
|---|---|---|---|---|
| D1 | Residual signal absent | M3 < 0.15 | IC-2a | Action-effect signal too weak to learn. Redesign env. |
| D2 | Residual <= StateOnly | M7 ≤ M6 | IC-2a | Knowing true residual doesn't beat ignoring action. AEP-control impossible in this env. |
| D3 | CF data worthless | M12 ≤ 0 | IC-2a | Counterfactual data provides no value. Passive-only approach confirmed insufficient. |
| D4 | RawMemory cheaper | M14 ≤ RawMemory_IAR | IC-2b | Throttling costs more than it's worth. Memory is better than compression. |
| D5 | Mostly bad debt | M16 > 0.5 | IC-2b | More than half of claimed gain is shortcut-explained. Model is bad debt. |
| D6 | Temporal model broken | M20 < 0.10 | IC-2b | Temporal order provides zero benefit. Temporal architecture is nominal, not actual. |
| D7 | Benchmark unstable | M24 > 0.5 | All phases | Seed variance > model gap. Results are noise. Increase seeds or redesign env. |
| D8 | No transfer | M10 ≤ 0 | IC-2b | No OOD benefit. Structure is not transferable capital — it's memorization. |
| D9 | Prediction ≠ Control | M15 < 0.5 | IC-5 | Model predicts well but can't control. Realization gap confirmed. |
| D10 | StateOnly dominates | M17 > 0.90 | All phases | Model's claimed gain is almost entirely StateOnly. No genuine operational capital. |
| D11 | ActionOnly dominates | M18 > 0.80 | All phases | Model's claimed gain is almost entirely ActionOnly. Env is action main-effect task. |
| D12 | Shuffled action irrelevant | M19 < 0.05 | All phases | Causal action-outcome pairing doesn't matter. Model uses spurious correlations. |

---

## Minimum Pass Conditions for Intelligence Appreciation Claim

A model/throttling mechanism must satisfy ALL of:

1. **M3 (residual_variance_ratio) ≥ 0.15** — residual signal exists
2. **M7 (residual_oracle_match) > M6 (state_only_ceiling)** — oracle AEP beats SO
3. **M12 (counterfactual_value) > 0.10** — CF data provides meaningful value
4. **M14 (IAR) > RawMemoryEqualCost_IAR** — throttling beats equal-cost memory
5. **M16 (bad_debt_ratio) < 0.5** — most gain is not shortcut-explained
6. **M20 (permuted_history_gap) > 0.10** — temporal order matters
7. **M24 (seed_stability_ratio) < 0.5** — benchmark is reliable
8. **M10 (transfer_premium) > 0** — OOD transfer works

**If ANY condition fails → intelligence appreciation NOT supported → do not proceed to next phase.**

---

## Operational Execution Flow

```
Phase IC-2a:
  Run env → compute M3 (residual_variance_ratio)
    M3 < 0.15? → D1 DEATH. Redesign env.
  Run oracle residual → compute M7
    M7 ≤ M6 (SO)? → D2 DEATH. AEP-control impossible in this env.
  Compute M12 (CF value)
    M12 ≤ 0? → D3 DEATH. No CF value.
  Compute M24 (seed stability)
    M24 > 0.5? → D7 DEATH. Increase seeds.

  ALL PASS → Proceed to IC-2b.

Phase IC-2b:
  Train all throttling mechanisms.
  For each mechanism:
    Compute M14 (IAR) → D4 check
    Compute M16 (bad_debt_ratio) → D5 check
    Compute M20 (permuted_history_gap) → D6 check
    Compute M10 (transfer_premium) → D8 check
    Compute M15 (ORR) → D9 check

  At least ONE mechanism passes ALL:
    → Intelligence appreciation SUPPORTED.
    → Proceed to IC-3 (throttling taxonomy).

  ZERO mechanisms pass ALL:
    → All throttling approaches fail.
    → Return to env design.
```

---

## Summary

**14 metrics, 12 death conditions, 8 minimum pass criteria.** Every claim about intelligence capital must survive this entire audit before being reported as a result. The metrics are designed to detect bad debt, shortcuts, and seed artifacts — the failure modes that killed AEP-control in Phase 8.0.