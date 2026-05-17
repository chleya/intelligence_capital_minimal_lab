# Cross-Phase Claim Ledger

**Status:** Intelligence Capital Theory IC-0  
**Date:** 2026-05-09  

---

## A. Retained Claims

| # | Claim | Evidence | Current Status | Next Required Test |
|---|---|---|---|---|
| R1 | AEP is a better operational state representation than observation similarity | Phase 7.5: grounded CF alignment shows AEP-based clustering ↑ prediction; Phase 7.8: Factorized > Baseline in all real-window envs | ✅ RETAINED | IC-2a: verify oracle residual exists |
| R2 | CTD quantifies mode-conditioned action-effect topology | Phase 6.2: CTD differentiability metrics correlate with learnability gaps; Phase 6.3: CTD sufficiency tests confirm | ✅ RETAINED | IC-2: use CTD as env audit tool |
| R3 | ECTD is closer to learnability than global CTD | Phase 6.5: ECTD explains more learnability variance than CTD; Phase 6.6: ECTD-policy coupling stronger | ✅ RETAINED | IC-2: compute ECTD for each env |
| R4 | novelty_balanced sampling improves structural exposure | Phase 6.7: lower learnability variance under balanced; Phase 6.8: proxy robustness maintained | ✅ RETAINED | IC-2: use novelty_balanced for trajectory gen |
| R5 | Factorized state-action architecture predicts RealAEP better than baseline encoder-decoder | Phase 7.6: Fact MSE < Baseline MSE across all envs; Phase 7.9: Fact match >> AO match in hard envs | ✅ RETAINED (prediction only) | IC-2b: test if Fact > SO in residual-present env |
| R6 | ActionOnly can be suppressed below 0.20 in properly designed envs | Phase 8.0: CFTrapV3 AO=0.14 (seed=40), AO=0.14 (seed=42); StrongDelay AO=0.27 | ✅ RETAINED (seed-dependent) | IC-2a: verify AO ceiling across 10 seeds |
| R7 | Environment entry gates (entropy, 0-action%, MRR) can detect structural quality | Phase 7.9: BalancedThree first env to pass action balance; Phase 8.0: StrongDelay consistently passes | ✅ RETAINED | IC-2: mandate gates before any model training |
| R8 | Death condition protocol forces rigorous falsification | Phase 7.1–8.0: death conditions caught latent steering artifacts, single-seed overclaims, GRU failure | ✅ RETAINED | All IC phases: define death conditions before each experiment |

---

## B. Withdrawn Claims

| # | Claim | Why Withdrawn | Evidence of Failure | Replacement Framework |
|---|---|---|---|---|
| W1 | Goodfire-style operational manifold is established and supports steering | Steering gains are confounds, not causal (Phase 7.2–7.4) | Shuffled audit: steering path gains from correlated confounders; Mediation: steering effect disappears under fixed-action control | ICT: latent geometry = internal artifact, not operational capital |
| W2 | Latent steering provides genuine causal control | Steering changes agent action choice, not world outcome (Phase 7.3–7.4) | Fixed-action rollouts: world outcome unchanged; Closed-loop: steering ≤ linear baseline | ICT: realization rate = 0; bad debt |
| W3 | GRU temporal encoder can solve delayed AEP | TemporalFactorized ≤ StaticFactorized everywhere (Phase 8.0) | 6/6 envs: T ≤ F; StrongDelay: T=0.296, F=0.324 (-0.028) | ICT: GRU is a throttling failure — does not extract temporal capital |
| W4 | Counterfactual data stably improves control | CF fraction curve non-monotonic, FullCF not > trajectory-only (Phase 8.0) | Multi-seed: CF_gap from Phase 7.9 disappeared; cf0.00 > cf1.00 in some runs | ICT: CF data value must be proven via oracle residual before learned |
| W5 | Factorized/TemporalFactorized exceeds StateOnly | StateOnly > Factorized in 6/6 envs (Phase 8.0) | CFTrapV3: SO=0.73, F=0.50; StrongDelay: SO=0.46, T=0.30 | ICT: autonomous dynamics domination prevents realization |
| W6 | Current benchmark is sufficient for Phase 8 architecture expansion | Only 7/18 env×seed combos pass entry gates; seed variance > model gap | 5/6 envs fail entry gates; CFTrapV3 AO std=0.27 > any model gap | ICT: stop model dev; return to oracle residual first |
| W7 | Closed-loop AEP planner provides control advantage | Planner match = prediction match, both ≤ SO (Phase 7.7) | ActionOnly/StateOnly planners match or exceed Factorized; OOD near AO ceiling | ICT: prediction capital ≠ control capital |

---

## C. Uncertain Claims (Requires Further Investigation)

| # | Claim | Why Uncertain | Evidence Gap | Next Required Test |
|---|---|---|---|---|
| U1 | Operational state representation can be converted to control advantage | Phase 8.0 shows prediction ↑ but control not ↑ | No env where Factorized > StateOnly > ActionOnly simultaneously | IC-2a: if oracle residual exists AND SO fails, proof of concept |
| U2 | RealAEP requires active intervention (not passive trajectory) to learn | Passive + partial CF failed; active never attempted | No head-to-head active vs passive comparison | IC-6: test ActiveProbe vs PassiveTrajectory after IC-2b |
| U3 | Why StateOnly dominates Factorized in all environments | Hypothesis: autonomous dynamics >> action-effect signal in current envs | No env with controlled action-effect/autonomous SNR ratio | IC-2a: measure residual_variance_ratio; design env with ratio > 0.3 |
| U4 | Whether a minimal AEP-requiring environment can be constructed | All construction attempts so far produced envs where SO > F | No formal definition of "AEP-requiring" exists | IC-2a: define minimal conditions + verify with oracle |
| U5 | Whether temporal information order matters at all in these envs | PermutedHistory ≈ TemporalFactorized everywhere | GRU may fail, or temporal signal may not exist — can't distinguish | IC-2a: test if oracle residual differs when history order is permuted |
| U6 | Whether ECTD/CTD diagnostic value extends beyond these specific envs | Only tested on AEP program's env suite | No third-party envs tested | IC-3: test on standard RL benchmarks |

---

## Summary Counts

| Category | Count |
|---|---|
| **Retained** | 8 claims |
| **Withdrawn** | 7 claims |
| **Uncertain** | 6 claims |
| **Total** | 21 claims |

---

## Key Insight

The retained claims are all **descriptive** (characterizing what structure exists and how to expose it). The withdrawn claims are all **prescriptive** (claiming that the structure directly enables control). The uncertain claims all concern the **realization gap**: whether and how descriptive capital can be converted to prescriptive value.

**This pattern is the core finding of the AEP program and the motivation for Intelligence Capital Theory.**