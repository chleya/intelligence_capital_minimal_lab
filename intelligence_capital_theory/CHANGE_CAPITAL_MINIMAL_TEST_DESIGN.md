# Change Capital Minimal Test Design

**Status:** Intelligence Capital Theory IC-2  
**Date:** 2026-05-09  

---

## 0. Design Principle

This is NOT an AEP-control experiment. It is a **capital audit experiment**: testing whether different throttling mechanisms can extract transferable, realizable value from the same world-change stream.

**Rule:** If a mechanism cannot beat RawMemory (same cost), StateOnly, ActionOnly, ShuffledAction, and PermutedHistory simultaneously, it is bad debt — not intelligence capital.

---

## 1. Experiment Objective

> Given the same world-change stream, which throttling mechanism produces the highest: transfer premium, intelligence appreciation rate (IAR), operational realization rate (ORR), and lowest bad debt ratio?

---

## 2. Environment Design

### StructuredVolatilityEnv

Not random noise. Structured volatility containing:

| Mechanism | Implementation |
|---|---|
| **Autonomous dynamics** | Background drift + noise that moves state regardless of action. Creates StateOnly shortcut. |
| **Action-effect sign flip** | Hidden mode m ∈ {A,B}. In mode A: +1 → positive effect, -1 → negative. In mode B: reversed. Mode not visible to model. |
| **Delayed consequence** | Some outcomes only visible at horizon H=3,5. Requires temporal credit assignment. |
| **Counterfactual fork** | For each history h, generate outcomes for all 3 actions separately: snapshot state → apply a=-1 → record outcome; snapshot → a=0 → record; snapshot → a=+1 → record. |
| **Structured OOD** | Three OOD types: (a) background_drift changed (autonomous-only shift); (b) action_gain changed (magnitude shift); (c) sign_rule changed (structure shift). |

### State space

Continuous 1D or 2D with interpretable dynamics. Not gridworld — must have continuous change-capital to throttle.

### Action space

{-1, 0, +1} — three actions with potentially different effects per mode.

### Mode

Hidden binary m ∈ {0,1}. Toggles with probability p_flip at each step. Mode determines action-effect sign. Mode is used for evaluation only — never provided to model input.

### Horizon

Outcomes measured at H ∈ {1, 3, 5}. Short horizon tests immediate action-effect. Long horizon tests temporal credit.

---

## 3. Data Generation Protocol

### Trajectory generation
```
For each seed (10 seeds):
  env.reset(seed)
  For t = 1 to T_max:
    a = random_policy(env.state)
    obs = env.step(a)
    record (obs_t, a_t, obs_{t+1}, mode_t)
```

### Counterfactual table generation
```
For each sampled state h:
  For action in {-1, 0, +1}:
    snapshot = env.snapshot()
    env.restore(snapshot)
    outcome = env.step_forward(action, horizon=H)
    record (h, action, outcome_H)
```

### Splits
- Train: 60% of states (ID)
- Val: 10% of states (ID)
- Test: 10% of states (ID)
- OOD_background: 7% of states (different drift)
- OOD_gain: 7% of states (different action gain)
- OOD_sign: 6% of states (different sign rule)

### Data format
CSV with fields: state_id, split, horizon, obs_features, history_features, action, outcome, noop_outcome, oracle_residual, centered_residual, best_action, latent_mode

---

## 4. Mechanisms Under Test (IC-2b)

| # | Mechanism | Input | Output | Throttling Strategy |
|---|---|---|---|---|
| 1 | StateOnlyPredictor | obs, history | Y_hat (no action) | Predict autonomous dynamics, ignore action |
| 2 | ActionOnlyPredictor | action | Y_hat (per action) | Learn per-action average outcome |
| 3 | AEPCompressor | obs, history, action | Y_hat(h,a) | Joint state+action → outcome |
| 4 | ResidualCompressor | obs, history, action | B_hat(h) + R_hat(h,a) | Separate autonomous + action-effect |
| 5 | CenteredResidualCompressor | obs, history, action | mean_a(Y) + CR_hat(h,a) | Separate centered structure |
| 6 | CounterfactualCompressor | obs, history | Y_hat(-1), Y_hat(0), Y_hat(+1) | Output all 3 actions simultaneously |
| 7 | CausalContrastCompressor | obs, history | contrastive embedding | Pull similar histories, push different |
| 8 | ResidualAdversarialCompressor | obs, history, action | B_hat + R_hat + adv_penalty | Adversarially strip SO shortcut from residual |
| 9 | RawMemoryFull | all train data | kNN lookup | Store everything, query nearest |
| 10 | RawMemoryEqualCost | subset of train data | kNN lookup | Store up to parameter budget of neural model |
| 11 | PrototypeMemory | clustered samples | nearest cluster center | Cluster → store centroids |
| 12 | ShuffledActionControl | shuffled action→outcome | Y_hat(h,a_shuffled) | Test: does action-outcome pairing matter? |
| 13 | PermutedHistoryControl | permuted history order | Y_hat(h_permuted, a) | Test: does temporal order matter? |

---

## 5. Evaluation Metrics

### ID Test (same distribution)
- outcome_mse
- residual_mse (M3)
- best_action_match
- rank_accuracy
- prediction_dividend (M8)
- control_dividend (M9)

### OOD Test (per split)
- transfer_premium (M10)
- structural_dividend (M11)
- counterfactual_value (M12)

### Cost Accounting
- parameter_count
- stored_samples_count (for memory baselines)
- cost_bytes (uniform quantization)
- IAR (M14)
- compression_to_value_ratio (M21)

### Debt Audit
- bad_debt_ratio (M16)
- state_only_shortcut_index (M17)
- action_only_shortcut_index (M18)
- shuffled_action_gap (M19)
- permuted_history_gap (M20)

### Robustness
- seed_stability_ratio (M24)
- ood_kl_control (M25)

---

## 6. Success Criteria

A throttling mechanism must pass ALL of:

1. **prediction_dividend > 0** — predicts better than baselines
2. **IAR > RawMemoryEqualCost_IAR** — cheaper than equal-cost memory
3. **best_action_match > StateOnly + 0.10** — beats autonomous shortcut
4. **best_action_match > ActionOnly + 0.10** — beats action-main-effect shortcut
5. **bad_debt_ratio < 0.5** — most gain is genuine, not shortcut
6. **shuffled_action_gap > 0.05** — uses action causally
7. **permuted_history_gap > 0.10** (in delay tasks) — uses temporal order
8. **seed_stability_ratio < 0.5** — benchmark stable
9. **transfer_premium > 0** in at least 2 OOD splits

**If NO mechanism passes:** intelligence appreciation is NOT supported by any current throttling approach. STOP.

---

## 7. Experiment Flow

```
IC-2a:
  1. Generate StructuredVolatilityEnv
  2. Generate counterfactual table (all 3 actions per state)
  3. Compute oracle residual accounting:
     - M3 (residual_variance_ratio)
     - M7 (oracle residual match vs SO/AO)
     - M12 (CF value)
     - M24 (seed stability)
  4. Output: IC2A_ORACLE_RESIDUAL_REPORT.md
  5. Gate check:
     - M3 ≥ 0.15 AND M7 > SO AND M12 > 0 AND M24 < 0.5?
       YES → proceed to IC-2b
       NO  → redesign env, re-run IC-2a
  
IC-2b (only if IC-2a passes):
  1. Train all 13 mechanisms
  2. Evaluate all on ID + 3 OOD splits
  3. Compute full metrics table for each
  4. Compute bad debt audit for each
  5. Compare against all baselines and controls
  6. Output: IC2B_LEARNED_COMPRESSOR_REPORT.md
  7. Verdict:
     - At least 1 mechanism passes all 9 criteria:
       → Intelligence appreciation SUPPORTED
       → Proceed to IC-3
     - Zero mechanisms pass:
       → All throttling approaches fail
       → Return to IC-2a env design
```

---

## 8. What This Experiment Does NOT Do

- Does NOT test closed-loop control (that's IC-5)
- Does NOT test active probing (that's IC-6)
- Does NOT claim any mechanism is "the right way"
- Does NOT optimize for benchmark scores
- Does NOT claim intelligence if shortcuts pass

**This experiment is an audit. Its purpose is to detect failure, not to produce impressive numbers.**