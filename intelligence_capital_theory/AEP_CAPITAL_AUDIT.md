# AEP Capital Audit: Phase 1–8 Re-audited as Capital Flow

**Status:** Intelligence Capital Theory IC-1  
**Date:** 2026-05-09  

---

## 0. Audit Principle

This document re-audits every major claim from AEP Phases 1–8.0 through the lens of Intelligence Capital Theory. Instead of listing experiments chronologically, we ask:

1. Was capital (change-information structure) **discovered**?
2. Was the discovered capital properly **throttled** (retained at acceptable cost)?
3. Was the throttled capital successfully **realized** (converted to prediction/control value)?
4. Does any claimed capital turn out to be **bad debt** (non-transferable, shortcut-driven, seed-dependent)?

---

## 1. Capital Discovered

### 1.1 AEP > Observation Similarity for Operational State
- **Phase:** 7.5–7.9
- **Capital type:** Operational state representation capital
- **What was found:** AEP-based state representation predicts real outcomes better than observation-similarity-based clustering
- **Evidence:** Factorized match (0.50 in CFTrapV3) >> ActionOnly match (0.14), showing the representation captures structure beyond trivial action-heuristics
- **ICT classification:** ✓ Genuine change capital. The representation encodes action-conditioned consequence patterns, not just surface similarity.
- **Limitation:** Prediction capital only. Does not automatically convert to control capital.

### 1.2 CTD/ECTD as Diagnostic Instruments
- **Phase:** 6.2–6.6
- **Capital type:** World-structure diagnostic capital
- **What was found:** CTD quantifies mode-conditioned action-effect topology; ECTD measures agent's actual exposure to differentiated action-effects
- **Evidence:** CTD differentiability metrics correlate with learnability across envs; ECTD explains variance in model performance
- **ICT classification:** ✓ Diagnostic capital. CTD/ECTD tell us "is there structure to learn?" and "does the agent experience it?" — essential for capital audit.
- **Limitation:** Descriptive only. High CTD ≠ learnable structure. High ECTD ≠ learned structure.

### 1.3 Novelty Exposure Improves Structure Discovery
- **Phase:** 6.7–6.8
- **Capital type:** Data-collection strategy capital
- **What was found:** novelty_balanced sampling reduces learnability variance; promotes exposure to mode-differentiated states
- **Evidence:** Lower std in learnability metrics under balanced sampling; robustness maintained across proxy changes
- **ICT classification:** ✓ Valid throttling input strategy. Novelty_balanced determines *which change events enter the system* — a pre-throttling selection mechanism.
- **Limitation:** Improves exposure, doesn't guarantee realization.

### 1.4 Factorized Architecture Predicts RealAEP
- **Phase:** 7.6–7.9
- **Capital type:** Architectural compression capital
- **What was found:** FactorizedAEPModel_v2 (separate state encoder + action encoder → effect decoder) predicts RealAEP outcomes better than monolithic encoder-decoder
- **Evidence:** Lower prediction MSE across all envs vs BaselineAEPModel; match improvement over ActionOnly in all envs
- **ICT classification:** ✓ Valid throttling architecture. The factorization decomposes change events into state-conditioned and action-conditioned components — a theoretically motivated compression.
- **Limitation:** The decomposition may be structural rather than causal: the model may learn autonomous dynamics in the "action" branch.

### 1.5 ActionOnly Suppression Demonstrated
- **Phase:** 8.0
- **Capital type:** Environment-hardening proof
- **What was found:** In CFTrapV3 (seeds 40, 42) and StrongDelay, ActionOnly match < 0.30, proving the environment is NOT an action main-effect task
- **Evidence:** AO=0.144 (CFTrapV3 seed=40), AO=0.139 (seed=42), AO=0.267 (StrongDelay seed=41)
- **ICT classification:** ✓ Environment-design achievement. Proves that "picking the same action always" fails → environment requires some structural knowledge.
- **Limitation:** Not stable across seeds (AO=0.711 in seed=41). Seed-sensitive environments are unreliable capital-formation grounds.

---

## 2. Throttling Failures

### 2.1 Latent Geometry Not Grounded
- **Phase:** 7.0–7.4
- **Failure type:** Compression without realization
- **What happened:** Autoencoder bottleneck produced "nice" latent geometry, but the geometry was an internal model artifact — not a reflection of grounded operational structure
- **ICT classification:** Throttling mechanism produced *internal structure* without *external realization*. The latent space "looked like" an operational manifold but was operationally empty.
- **Root cause:** The compression objective (reconstruction) optimized internal consistency, not operational utility. The geometry encoded observation similarity, not change-capital.
- **Lesson:** Latent structure ≠ operational capital. Geometry must be validated by grounded counterfactual realization, not by visualization.

### 2.2 GRU Not Using Temporal Order
- **Phase:** 8.0
- **Failure type:** Architecture not providing its claimed function
- **What happened:** TemporalFactorized ≈ StaticFactorized in 5/6 envs; PermutedHistory ≈ TemporalFactorized in StrongDelay (diff=-0.002)
- **ICT classification:** The GRU encoder is a *nominal* throttling mechanism that failed to *actually* throttle temporal information. It passes through (obs,act) pairs but does not extract temporal credit structure.
- **Root cause:** Either (a) no temporal credit signal exists in these envs, or (b) the GRU architecture cannot separate temporal credit from autonomous drift.
- **Lesson:** Adding a temporal architecture does not automatically extract temporal capital. Must prove temporal signal exists before attempting to throttle it.

### 2.3 Factorized Not Exceeding StateOnly
- **Phase:** 8.0
- **Failure type:** Capital accumulation but zero realization advantage
- **What happened:** All 6 envs: Factorized match (0.30–0.56) ≤ StateOnly match (0.37–0.73). In AmbiguousRev: F=SO identically. In BalancedThree: F > SO (+0.094), but T = SO.
- **ICT classification:** The Factorized architecture accumulates action-effect capital, but this capital provides no advantage over *ignoring action entirely* and predicting autonomous dynamics. The accumulated capital has zero realization premium.
- **Root cause:** Autonomous dynamics (Var[p_auto]) >> action-effect signal (Var[p_action]) in all envs. Throttling p_action is costly and yields less value than simply memorizing p_auto.
- **Lesson:** Capital that provides no advantage over ignoring the action dimension is bad debt — it costs more without yielding more.

---

## 3. Realization Failures

### 3.1 Steering Changed Action Choice, Not Outcome
- **Phase:** 7.3–7.4
- **What happened:** Latent steering shifted the planner's action distribution, but fixed-action rollouts showed the steering did not change *what actually happens given the same action*. The steering changed *what the agent does*, not *how the world responds*.
- **ICT classification:** Nominal realization (action changed) without operational realization (world outcome unchanged). The "control" was an illusion.
- **Lesson:** Agent-side output change ≠ world-side consequence change. Operational realization requires ground-truth outcome difference.

### 3.2 Prediction Correlation ≠ Control Value
- **Phase:** 7.7–8.0
- **What happened:** All models evaluated by prediction match, but match improvement over baselines did not translate to closed-loop return improvement. OOD transfer match (0.31–0.56) was near ActionOnly ceiling.
- **ICT classification:** Prediction capital accumulated but never realized as control capital. The entire evaluation pipeline measured the wrong realization target.
- **Lesson:** Separate metrics for prediction dividend and control dividend. A model can have high prediction dividend and zero control dividend simultaneously.

### 3.3 Closed-Loop Planner Deceived by Shortcuts
- **Phase:** 7.7–7.8
- **What happened:** ActionOnly and StateOnly planners achieved competitive or better closed-loop performance than Factorized planners, because: (a) the environment's optimal action was state-dependent (StateOnly shortcut), or (b) a single action dominated (ActionOnly shortcut)
- **ICT classification:** The "hard" environments were not hard *in the right way*. They tested action-effect learning but the optimal policy was learnable through cheaper shortcuts.
- **Lesson:** An environment must be proven to block all shortcuts (SO, AO, ShuffledAction, PermutedHistory) before it can test operational realization.

---

## 4. Bad Debt Cases

### 4.1 Goodfire-Style Operational Manifold Overclaim
- **Claim:** "Latent geometry forms an operational manifold supporting causal steering"
- **Reality:** Latent geometry was a model-internal compression artifact, not grounded operational structure
- **Bad debt ratio:** ~1.0 (100% of the "steering gain" was attributable to confounders)
- **Lesson:** Visualization ≠ validation. Geometry must pass grounded counterfactual audit.

### 4.2 Action-Only Trivial Planner
- **Claim:** Environments are hard and require action-effect learning
- **Reality:** StateDependent: AO=SO=0.55; CFTrapV3 seed=41: AO=0.71. In these cases, picking the most common action works fine.
- **Bad debt ratio:** Variable (0.5–1.0 depending on env/seed). The "control advantage" of Factorized over ActionOnly is entirely seed-dependent.
- **Lesson:** Verify ActionOnly ceiling before claiming control advantage over it.

### 4.3 Fake OOD Transfer
- **Claim:** Factorized model transfers structure across environments
- **Reality:** OOD transfer match (0.31–0.56) can be achieved by learning "the most common optimal action" in the source env and applying it blindly to the target env.
- **Bad debt ratio:** >0.7 (most OOD gain explainable by trivial action-distribution matching)
- **Lesson:** OOD transfer must control for action-distribution-matching baselines.

### 4.4 Seed-Sensitive Benchmarks
- **Claim:** CFTrapV3 CF_gap = +0.367 (Phase 7.9 single seed)
- **Reality:** CFTrapV3 CF_gap disappears in multi-seed Phase 8.0 (CF curve non-monotonic, FullCF not consistently > trajectory-only)
- **Bad debt ratio:** ~1.0 for single-seed claim. The claimed "discovery" was a seed artifact.
- **Lesson:** Never report single-seed results as discoveries. Multi-seed mean/std mandatory.

### 4.5 StateOnly Shortcut Everywhere
- **Claim:** Factorized learns action-effect structure
- **Reality:** StateOnly achieves equal or better match in 6/6 environments. Whatever Factorized learned, it's not providing advantage over "predict what happens regardless of action."
- **Bad debt ratio:** >0.9 (the Factorized gain beyond StateOnly is near-zero everywhere)
- **Lesson:** StateOnly is the mandatory baseline. If Factorized ≤ StateOnly, the "action-effect" claim is invalid.

---

## 5. Salvageable Assets

| Asset | Type | Preserve as | Next use |
|---|---|---|---|
| AEP definition | Concept | Operational state representation framework | IC-2: as one throttling mechanism |
| CTD | Metric | World-structure differentiability measure | IC-2: environment audit |
| ECTD | Metric | Agent's structural exposure measure | IC-2: trajectory quality audit |
| RealAEP | Data format | Ground-truth asset table | IC-2: oracle residual accounting |
| Factorized architecture | Model design | Separated state/action encoding | IC-2b: one learned compressor |
| Environment gate methodology | Evaluation design | Entry-gate system | IC-2: environment validation |
| Death-condition culture | Evaluation design | Rigorous falsification protocol | All IC phases |
| Counterfactual table | Data methodology | Structured CF data generation | IC-2a: oracle residual test |
| PermutedHistoryControl | Control baseline | Temporal credit validation check | All IC experiments |
| novelty_balanced | Data strategy | Structural exposure boosting | IC-2: trajectory generation |

---

## 6. Written Conclusion

**AEP-control 的失败不是 AEP 思想失败，而是过早把 diagnostic capital 当作 control capital 的失败。**

The AEP program discovered real change-capital (action-effect structure), built a reasonable throttling mechanism (Factorized architecture), but then:
1. Evaluated the throttled structure on prediction (not control)
2. Never verified whether the throttled structure could be realized as control
3. Found (in Phase 8.0) that it could not

This is not a failure of the AEP concept. It is a failure to distinguish between:
- **Capital accumulation** (discovering and storing structure) — ✓ achieved
- **Capital realization** (converting stored structure to operational value) — ✗ failed

The lesson is not that AEP is wrong. The lesson is that the entire AI field conflates these two stages. Intelligence Capital Theory exists to separate them and to build audit mechanisms that detect when realization fails.

**Next: Proceed to IC-2a (Oracle Residual Accounting) — do not resume AEP-control model development until residual signal is proven to exist and exceed baselines.**