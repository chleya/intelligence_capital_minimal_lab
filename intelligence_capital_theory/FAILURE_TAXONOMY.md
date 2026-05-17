# Failure Taxonomy

**Status:** Intelligence Capital Theory IC-0  
**Date:** 2026-05-09  

---

## 1. Environment Failure

| # | Failure | Phase | Specific Case | Fix Condition |
|---|---|---|---|---|
| E1 | **Autonomous dynamics domination** | 8.0 | All 6 envs: SO match (0.37–0.73) > F match (0.30–0.56). State-change component independent of action dominates action-effect component. | Design env where Var(action_effect) / Var(autonomous_dynamics) > 0.30 |
| E2 | **0-action underrepresentation** | 8.0 | BalancedThree 0-action=9–13% (target 25–35%); StateDependent/Partition 0-action=7–9%. "Do nothing" almost never optimal. | Redesign reward/dynamics so 0-action is optimal ≥20% of states |
| E3 | **Seed-sensitive oracle distribution** | 8.0 | CFTrapV3: AO varies 0.14 → 0.71 across 3 seeds. Oracle action entropy varies 0.69 → 0.80. Single seed gives misleading picture. | At least 10 seeds; auto-tune env params to stabilize oracle distribution |
| E4 | **Entry gate instability** | 8.0 | Only 7/18 env×seed combos pass entry gates (entropy>0.85, 0-action>0.15, MRR>0.25). 5/6 envs fail in at least 2/3 seeds. | Must pass entry gates before any model training; auto-tune to pass |
| E5 | **Action-main-effect environments** | 8.0 | StateDependent: AO=SO. Optimal action is state-determinable — no operational ambiguity. | Diagnose and exclude from "hard" benchmark |
| E6 | **Counterfactual signal too weak** | 8.0 | Residual variance ratio (Var[R]/Var[Y]) appears very low in all empirically tested envs — action-effect differences are small relative to total outcome variance. | IC-2a: compute residual_variance_ratio; if <0.15, redesign env |

---

## 2. Model Failure

| # | Failure | Phase | Specific Case | Fix Condition |
|---|---|---|---|---|
| M1 | **Temporal encoder does not use temporal order** | 8.0 | StrongDelay: PermutedHistory (0.298) ≈ TemporalFactorized (0.296), diff=-0.002. GRU produces same output regardless of (obs,act) pair ordering. | IC-2a: verify oracle residual sensitive to history order; IC-2b: redesign encoder with explicit temporal credit detection |
| M2 | **Factorized architecture ≤ StateOnly baseline** | 8.0 | 6/6 envs: Factorized provides zero advantage over ignoring action entirely and predicting autonomous dynamics. | Factorized must prove advantage over SO in at least 3 envs |
| M3 | **Residual decomposition provides no control gain** | 8.0 | TemporalResidual and StaticResidual both ≤ Factorized ≤ StateOnly. Residual head does not separate autonomous from action-effect components. | IC-2a: verify oracle residual accounting works before training learned residual compressor |
| M4 | **Rank/pairwise models are unstable** | 8.0 | TemporalRankV2 match = SO (degenerates to state prediction). Pairwise loss adds no signal beyond MSE regression. | Only use rank objectives when oracle action distribution has high entropy; verify rank improvement over regression in ablated test |
| M5 | **Hybrid multi-task model underperforms simpler models** | 8.0 | HybridTemporalRankReg (uncertainty-weighted multi-task) achieves match 0.21–0.44, below Factorized (0.30–0.56). Multi-task complexity adds noise. | Simplify; don't add tasks unless single-task baseline works first |
| M6 | **Model capacity may exceed useful signal** | 7.9–8.0 | scale="large", bottleneck=32, gru_hidden=64, gru_layers=2 on 280 train samples. Overfit to noise patterns rather than action-effect structure. | Match model capacity to available signal; use signal-to-parameter ratio as design constraint |

---

## 3. Objective Failure

| # | Failure | Phase | Specific Case | Fix Condition |
|---|---|---|---|---|
| O1 | **best_action_match is too coarse** | 7.5–8.0 | Match = argmax(prediction) == argmax(truth). When action-effect differences are small (Δ=0.02), random guessing already achieves 33%. Metric provides no discrimination in low-SNR regime. | Use continuous ranking metric (weighted AUC, pairwise accuracy, soft regret) |
| O2 | **CF fraction experiment confounded by state distribution** | 8.0 | CF sampling: different CF fractions correspond to different state distributions. traj-only uses states where agent happened to be; FullCF uses all states with all actions. Cannot attribute match differences to CF data vs state coverage. | Same-state counterfactual sampling: for each state, select CF fraction of additional actions uniformly |
| O3 | **Trajectory-only vs FullCF indistinguishable** | 8.0 | cf0.00 and cf1.00 performance gap < 0.10 in multiple env/seed pairs. The CF data provides effectively zero marginal signal. | IC-2a: oracle residual must show significant gap before training learned models |
| O4 | **MSE optimization ≠ action-ranking optimization** | 7.5–8.0 | Model minimizes MSE on outcome prediction, but evaluation is on action ranking. A model can have low MSE but rank actions poorly (and vice versa). | Multi-objective or ranking-oriented loss; decouple prediction and ranking metrics |
| O5 | **No closed-loop objective ever used** | 7.7–8.0 | All evaluation is static: predict best_action on held-out states. No agent ever walks through the environment using the learned model. Match ≠ return. | IC-5: closed-loop evaluation with cumulative regret/return as primary metric |

---

## 4. Evaluation Failure

| # | Failure | Phase | Specific Case | Fix Condition |
|---|---|---|---|---|
| V1 | **3 seeds are insufficient** | 8.0 | CFTrapV3 AO std = 0.27, mean = 0.33. With 3 seeds, mean estimate is unreliable; cannot distinguish model gap from seed variance. | 10 seeds minimum; compute 95% CI; require CI < gap/2 |
| V2 | **StateOnly baseline introduced too late** | 7.7 | StateOnly model (predicts outcome from state/history alone, ignoring action) not introduced until Phase 7.7. Previous 7 phases compared Factorized only to Baseline encoder-decoder or ActionOnly — missing the simplest and strongest baseline. | StateOnly must be baseline #1 in every experiment from the start |
| V3 | **Held-out states too few** | 8.0 | 180 held-out states = 540 action-triples. For 3-way action distribution, effective sample per action < 200. Std on match estimates is high. | 500+ held-out states (1500+ action-triples) |
| V4 | **OOD transfer not controlled for trivial distribution matching** | 8.0 | OOD match 0.31–0.56. A model that learns "always pick +1 in source env" achieves nontrivial match in target env if +1 happens to be good there too. | OOD transfer must control for action-distribution-matching baseline |
| V5 | **No seed-stratified OOD evaluation** | 8.0 | OOD transfer uses same seed for both envs. If seed produces specific oracle distribution, transfer may reflect seed correlation, not structural transfer. | Cross-seed transfer: train on env_A/seed_i, test on env_B/seed_j where i≠j |
| V6 | **Prediction evaluation only; no control evaluation** | 7.5–8.0 | The entire project evaluates models on prediction match, then claims conclusions about control. The realization gap (prediction → control) was never tested. | IC-5: mandatory closed-loop control evaluation |

---

## 5. Theory-Overreach Failure

| # | Failure | Phase | Specific Case | Fix Condition |
|---|---|---|---|---|
| T1 | **AEP-control assumed, not tested** | 7.0–7.9 | The program's implicit assumption: "If we can predict action-effects, we can control." Phase 8.0 proved this assumption false, but it drove 8 phases of development. | Separate prediction and control hypotheses; test independently |
| T2 | **Action-effect topology ≠ policy utility** | 6.2–7.9 | CTD shows mode-conditioned action ranking differences, but knowing the ranking ≠ knowing what action to pick. Topology is descriptive; utility is prescriptive. | Don't claim policy implications from topology descriptions |
| T3 | **ECTD/CTD correlation ≠ causation for learnability** | 6.5–7.9 | ECTD correlates with learnability, but correlation may be driven by third factors (env complexity, noise level, action-main-effect strength). | Ablation studies: vary ECTD independently from other env factors |
| T4 | **Passive trajectory assumed sufficient for action-effect discovery** | 7.0–8.0 | The program assumes passive observation of agent-chosen actions is enough to discover action-effect structure. But passive sampling biases state-action coverage. | IC-6: test active probing vs passive; if active >> passive, passive assumption is invalid |
| T5 | **"Operational" used too loosely** | 7.0–8.0 | "Operational state" used to mean both "useful for prediction" and "useful for control." Phase 8.0 demonstrated these are NOT the same. | Strict definitions: operational_prediction ≠ operational_control |
| T6 | **Latent geometry reified as causal structure** | 7.0–7.4 | Visualized latent spaces interpreted as "operational manifolds" without causal validation. The geometry was a compression artifact. | No geometric claims without fixed-action counterfactual validation |

---

## Summary Severity Ranking

| Rank | Category | Failure | Severity |
|---|---|---|---|
| 1 | Theory | AEP-control assumed, not tested | **Fatal** |
| 2 | Evaluation | Prediction only, never control | **Fatal** |
| 3 | Environment | Autonomous dynamics domination | **Critical** |
| 4 | Evaluation | StateOnly baseline too late | **Critical** |
| 5 | Evaluation | 3 seeds insufficient | **Critical** |
| 6 | Model | Temporal encoder does not use order | Critical |
| 7 | Objective | CF experiment confounded | Critical |
| 8 | Objective | No closed-loop objective | Critical |
| 9 | Theory | Passive trajectory assumption | High |
| 10 | Environment | Seed-sensitive oracle | High |

**This taxonomy defines what Intelligence Capital Theory's experimental program must solve first.**