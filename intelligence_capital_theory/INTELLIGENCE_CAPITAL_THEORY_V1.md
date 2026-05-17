# Intelligence Capital Theory V1：智能资本论

**Version:** V1  
**Date:** 2026-05-09  
**Status:** Theory construction, not validation  
**Origin:** Post-AEP Phase 8.0 postmortem upgrade

---

## 0. One-sentence thesis

**智能不是即时推理，而是世界变动信息被压缩、节流、结构化保存后，在新情境中的增值兑现。**

Intelligence is not real-time inference, but the appreciation of world-change information that has been compressed, throttled, structurally retained, and then realized as value in new contexts.

---

## 1. Why this theory emerged

### 1.1 AEP did not die

The Action-Effect Profile (AEP) program ran Phases 1–8.0 and produced genuine descriptive insights:
- AEP > observation similarity as an operational state definition (Phases 7.5–7.9)
- CTD/ECTD quantify mode-conditioned action-effect topology (Phases 6.2–6.6)
- Factorized state-action architecture predicts RealAEP better than monolithic models (Phases 7.6–7.9)
- novelty_balanced sampling improves structural exposure (Phases 6.7–6.8)

These are **retained claims** with strong evidence. The descriptive value of AEP is real.

### 1.2 AEP-control paused

Phase 8.0 triggered all death conditions:
- TemporalFactorized ≤ StaticFactorized in 6/6 environments
- StateOnly > Factorized in 6/6 environments
- PermutedHistory ≈ TemporalFactorized (GRU does not use temporal order)
- CF data provides no stable marginal control value
- Seed variance > model gap in most comparisons

The attempt to convert AEP (action-effect prediction) into control (action selection advantage) failed systematically.

### 1.3 The real problem exposed

The AEP-control failure is not a methodological failure. It exposes something deeper:

**Prediction ≠ Realization. Representation ≠ Control. Geometry ≠ Utility.**

The entire program assumed an implicit chain:
```
world change → learn structure → structure → control
```

Phase 8.0 proved this chain breaks:
- Factorized learns action-effect structure (✓ prediction)
- But this structure does not translate to better action selection (✗ control)
- StateOnly, which ignores action structure entirely, achieves better control

This is not a bug. It is a **fundamental insight**: information about the world's action-effect structure is a form of capital that can be accumulated, but capital accumulation ≠ capital realization. You can own a factory (AEP capital) that produces nothing (no control realization).

### 1.4 The upgrade

The old question: *"Can a model learn operational state?"*

The new question: *"How does a system accumulate, throttle, transfer, and realize change-capital from a continuous world-change stream?"*

This repositions AEP from a control architecture to a **sub-asset class** within a broader theory of intelligence as capital formation and realization.

---

## 2. Core concepts

### 2.1 Change Capital

**Definition:** The effective information a system has absorbed and structurally retained about *how the world changes* — not about what the world is at a snapshot.

**Change Capital includes:**

| Type | Description | Example |
|------|-------------|---------|
| State-transition capital | Knowledge of p(s'\|s) regardless of action | Autonomous dynamics |
| Action-effect capital | Knowledge of p(s'\|s,a) - p(s'\|s,0) | AEP, RealAEP |
| Delayed-consequence capital | Knowledge of effects at H>1 | Temporal credit |
| Event-unfolding capital | Knowledge of multi-step trajectories | Rollout simulation |
| Object-permanence capital | Knowledge of entity identity across changes | Tracking |
| Tool-amplification capital | Knowledge of how actions multiply effects | Leverage dynamics |
| Risk-accumulation capital | Knowledge of compounding consequences | Safety modeling |
| Social-feedback capital | Knowledge of how others respond to changes | Multi-agent |

**Key property:** Change capital is *counterfactual* — it's about "what would happen if," not about "what happened." A snapshot observation contains no change capital. Change capital requires at minimum a pair (before, after) or a counterfactual triple (before, action= -1/0/+1, after@H).

### 2.2 Throttled Structure

**Definition:** The subset of change-capital that a system selectively retains under finite capacity constraints.

**Why throttling is essential:**
- The world-change stream is infinite: every timestep produces new change events
- No system can store all change events
- Throttling is the **selection function** that chooses which change events are worth retaining

**Throttling is not compression:**
- Compression reduces bit-cost of already-chosen information
- Throttling *chooses what to retain at all*
- A high-compression but poorly-throttled system stores irrelevant changes
- A well-throttled but poorly-compressed system stores high-value changes expensively

**Forms of throttling:**
- Latent geometry (autoencoder bottlenecks)
- AEP (action-conditioned consequence profiles)
- Symbols (named categories of situations/actions)
- Rules (if-then templates)
- World models (predictive simulators)
- Attention (selective input gating)
- Memory replay (priority-weighted retention)

**All these are throttling mechanisms.** They differ in: retention policy, cost per retained bit, transferability, and realizability.

### 2.3 Intelligence Appreciation

**Definition:** Throttled change-capital generates more value in a new context than its original storage cost.

**Analogy to capital appreciation:**
- Buy asset at cost C
- Asset generates returns V in new contexts
- Appreciation rate = (V - C_baseline) / C

This is *not* a metaphor. It's a measurement protocol:
- Cost(S) = storage cost (bytes, parameters, samples stored) of structure S
- V(S, C_new) = value generated by S in new context C_new
- Baseline = RawMemory at equal cost, or trivial predictor
- IAR = [V(S, C_new) - V(Baseline, C_new)] / Cost(S)

**A structure that generates value but costs more than RawMemory at equal performance is not intelligence — it is expensive redundancy.**

### 2.4 Operational Realization

**Definition:** Intelligence capital is *realized* when it produces actual downstream value — not just predicted value.

**The realization gap:**
| Prediction capital | Control capital |
|---|---|
| "I can predict what happens" | "I can make what I want happen" |
| Match metric | Regret/return metric |
| Passive evaluation | Closed-loop evaluation |
| State-conditioned | Policy-conditioned |

**Phase 8.0's core empirical finding:** Prediction capital ≠ Control capital. FactorizedAEP holds prediction capital (better RealAEP prediction than baseline), but this does not realize as control capital (best_action_match ≤ StateOnly).

**Operational realization requires:**
1. The structure enables action selection that changes real outcomes
2. The realized value exceeds the value of simpler baselines (StateOnly, ActionOnly)
3. The realization persists under OOD shifts
4. The realization is not attributable to confounders (shuffled/permuted controls fail)

### 2.5 Bad Debt / False Capital

**Definition:** Structure that appears to be intelligence capital but either cannot transfer, cannot realize, or is fully explained by trivial shortcuts.

**Taxonomy of bad debt:**

| Bad debt type | Appearance | Reality | Phase Example |
|---|---|---|---|
| **Latent steering artifact** | "Moving along latent direction changes output" | Correlated confounder, not causal | Phase 7.2–7.4 |
| **Observation leakage** | "Model predicts well from obs" | Future obs leaked into input | Phase 6.8 |
| **Action-main-effect shortcut** | "ActionOnly predicts well" | Environment is action-trivial | StateDependent (AO=SO=0.55) |
| **StateOnly shortcut** | "Prediction is good" | Autonomous dynamics dominates, not action-effect | All 6 Phase 8.0 envs |
| **Trivial OOD transfer** | "Works on different env" | Trivial common-best-action transfers | Phase 8.0 OOD |
| **Memorized trajectory** | "Low MSE on test" | Memorized specific trajectory patterns, not structure | — |
| **Seed-dependent "gain"** | "+0.367 CF gap" | Single seed artifact, disappears multi-seed | Phase 7.9 → 8.0 CFTrapV3 |

**Bad debt ratio:** BDR = shortcut_explainable_gain / total_reported_gain. If BDR > 0.5, more than half the claimed gain is bad debt.

---

## 3. Relation to AEP

### 3.1 AEP reclassified

Under Intelligence Capital Theory:
- **AEP** = one specific form of Action-Effect Capital
- **CTD** = Counterfactual Topology Differentiability — measures whether different modes produce different action-effect structures
- **ECTD** = Experienced CTD — measures whether the agent's trajectory actually exposes it to mode-differentiated action-effects
- **RealAEP** = verifiable, realizable action-effect asset table — the ground-truth record of "what happens if I take action a in state s"

### 3.2 AEP is not intelligence

AEP is a **sub-asset class** within change capital. It answers:
> "Given the world's action-effect structure, what would happen if I take action a?"

It does NOT answer:
> "Should I take action a?"

That requires control capital, which is a *different* asset class. Phase 8.0 proved these are distinct: you can own AEP capital without owning any control capital.

### 3.3 The AEP-control failure is a capital realization failure

The program accumulated AEP capital (descriptive action-effect knowledge) but failed to realize it as control capital (action-selection advantage). This is not that AEP capital is worthless — it's that the conversion mechanism (Factorized → best_action_match) failed to realize the stored value.

---

## 4. Relation to LLMs

### 4.1 LLMs as language-capital banks

LLMs are **high-leverage compression banks of language capital**:

| Capital type | LLM has it? | Why |
|---|---|---|
| Language capital | ✓ Massive | Trained on all text |
| Cultural capital | ✓ Massive | Encoded in linguistic patterns |
| Analogy capital | ✓ Significant | Pattern recombination |
| Symbolic recombination capital | ✓ Significant | Symbol manipulation |
| Embodied capital | ✗ None | No body, no physics |
| Intervention capital | ✗ None | No ability to act and observe effects |
| Risk-bearing capital | ✗ None | No consequences for errors |
| Real-time operational feedback | ✗ None | Static training, no online update |
| Persistent self-updating capital | ✗ Partial | No continuous learning from consequences |

### 4.2 LLM bad debt

| Bad debt type | Mechanism |
|---|---|
| Hallucination debt | Generated text with no ground-truth verification |
| Linguistic surface correlation | Learning "what sounds right" not "what is true" |
| Unverifiable semantic debt | Claims that cannot be operationally verified |
| Operational non-realization | Text output is not action in the world |

**ICT insight:** An LLM has enormous stored language capital with near-zero operational realization rate. Its appreciation rate (value per unit stored) may be lower than a simple rule-based system in bounded domains, because its stored capital is poorly throttled for specific operational contexts.

---

## 5. Relation to human intelligence

Human judgment is not pure real-time rationality. It is **multi-layer historically compressed structure realizing in current context**.

| Layer | Compression timescale | What it provides |
|---|---|---|
| Evolution | Millions of years | Body, instincts, perceptual priors |
| Body | Lifetime | Motor skills, proprioception, embodied physics |
| Childhood experience | Years | Causal intuitions, social norms, language |
| Cultural categories | Decades/centuries | Shared concepts, tools, practices |
| Social feedback | Ongoing | Reputation, trust, coordination signals |
| Professional practice | Years | Domain expertise, pattern recognition |
| Failure memory | Lifetime | Risk avoidance, negative examples |
| Tool-use history | Years/decades | Extended agency, amplification knowledge |

**Summary:** So-called "judgment" is nothing more than historically compressed structure appreciating in the current context.

---

## 6. New research question

**Old question (AEP Phases 1–8):**
> "Can a model learn operational state?"

**New question (ICT):**
> "How does a system accumulate, throttle, transfer, and realize intelligence capital from a continuous world-change stream?"

This shifts from engineering a specific architecture to auditing the entire capital formation and realization pipeline.

---

## 7. Minimal formalism

### 7.1 World change stream

```
D = {(o_t, a_t, o_{t+1}, r_t, c_t)}
```

where `o_t` is observation, `a_t` is action, `r_t` is reward/outcome, `c_t` is context/mode.

### 7.2 Change event

```
e_t = Δ(o_t, a_t, o_{t+1})
```

A change event is the triple (before-state, action, after-state). A single observation contains zero change events.

### 7.3 Throttling function

```
T(D) → S
```

T maps the infinite change stream D to a finite retained structure S. The choice of T is the central design problem.

### 7.4 Capital value in context

```
V(S, C_new)
```

The value that structure S generates when deployed in new context C_new. Value can be: prediction accuracy, control advantage, transfer performance, or decision quality.

### 7.5 Intelligence Appreciation Rate

```
IAR = [V(S, C_new) - V(Baseline, C_new)] / Cost(S)
```

Positive IAR means: the retained structure generates more value than a trivial baseline, per unit cost. Negative IAR means: storing the structure is worse than not storing it.

### 7.6 Operational Realization Rate

```
ORR = realized_value / predicted_or_potential_value
```

Measures what fraction of predicted capital value actually realizes as operational value. Low ORR means "the model predicts well but doesn't help action."

### 7.7 Bad Debt Ratio

```
BDR = non_transferable_or_shortcut_explainable_gain / total_reported_gain
```

A high BDR system claims intelligence but is really exploiting shortcuts.

**These formulas are precise but not yet theoretically proven. They define what must be measured, not what has been proven.**

---

## 8. Metrics

Each metric must be: defined, intuitive, computable, and falsifiable.

| # | Metric | Definition | Intuition | Min computation | Failure |
|---|--------|-----------|-----------|-----------------|---------|
| 1 | effective_change_events | Count of (o_t, a_t, o_{t+1}) with |o_{t+1} - o_t| > ε | How many "real changes" happened | Thresholded delta count | If →0, nothing to learn |
| 2 | retained_change_ratio | |S| / |D|: stored structures per change event | Compression ratio | Structure count / event count | If →1, no throttling |
| 3 | compression_to_value_ratio | Cost(S) / V(S, C) | Cost per unit of value generated | Bytes / match gain | If high, expensive capital |
| 4 | structural_dividend | V(S, C_new) - V(S, C_old) | How much the same structure yields in a new context | Match difference across contexts | If →0, no transfer |
| 5 | transfer_premium | OOD_match - ID_match_predicted_from_baseline | Unexpected OOD benefit | OOD gain over baseline prediction | If ≤0, no transfer value |
| 6 | realization_rate | control_match / prediction_match | How much prediction → control | Ratios of match metrics | If <0.5, prediction ≠ control |
| 7 | capital_decay | V(S, C_t+δ) / V(S, C_t) | How fast does value decay over time | Perf at t+δ / perf at t | If →0.5 fast, unstable capital |
| 8 | bad_debt_ratio | shortcut_explained_gain / total_gain | How much "gain" is from cheap shortcuts | (Max of SO/AO/Shuffled gain) / claimed gain | If >0.5, mostly bad debt |
| 9 | operational_leverage | control_gain / prediction_gain | How much prediction amplifies control | Ratio of gains | If <1, prediction > control |
| 10 | prediction_dividend | match(model) - match(baseline) | Raw prediction gain | Match difference | If ≤0, no prediction value |
| 11 | control_dividend | regret(baseline) - regret(model) | Action-selection gain | Regret reduction | If ≤0, no control value |
| 12 | counterfactual_value | match(FullCF) - match(traj_only) | Value of seeing alternative actions | Match gap | If ≤0, CF data is worthless |
| 13 | state_only_shortcut_index | match(SO) / match(model) | How much model gain is StateOnly-doable | Match ratio | If >0.9, mostly autonomous shortcut |
| 14 | action_only_shortcut_index | match(AO) / match(model) | How much model gain is ActionOnly-doable | Match ratio | If >0.8, mostly action-main-effect |

---

## 9. Death conditions

A throttling mechanism must pass the following to claim *intelligence appreciation*:

| # | Death condition | Meaning |
|---|---|---|
| D1 | RawMemoryEqualCost > ThrottledStructure in all metrics | The structure adds cost without adding value — it is *worse than remembering* |
| D2 | transfer_premium ≈ 0 across all OOD splits | The structure is *memorization*, not structure — it does not generalize |
| D3 | realization_rate ≈ 0 | The structure is *book capital*, not operational — it predicts but doesn't control |
| D4 | StateOnly achieves equal or better match | The claimed capital is *autonomous dynamics*, not action-effect |
| D5 | ActionOnly achieves equal or better match | The claimed capital is *action-main-effect*, not mode-conditioned |
| D6 | ShuffledAction ≈ model | The action-outcome pairing is *not causal* — shuffled gives same result |
| D7 | PermutedHistory ≈ model in delay tasks | The temporal order is *not used* — temporal model is a static model in disguise |
| D8 | seed variance > model gap / 2 | The *benchmark is unstable* — results are noise, not signal |

**If all death conditions trigger simultaneously:** the structure is bad debt, the throttling mechanism fails, and the experiment design must be revised before claiming anything.

---

## 10. Research roadmap

| Phase | Name | Description | Gates |
|---|---|---|---|
| IC-0 | Theory ledger | This document + claim ledger + failure taxonomy | — |
| IC-1 | AEP capital audit | Re-audit Phases 1–8 as capital flow, not experiment list | — |
| IC-2a | Oracle residual accounting | Prove residual signal exists before training models | D1–D5 |
| IC-2b | Learned throttling comparison | Compare RawMemory, AEP, Residual, Counterfactual, CausalContrast | D1–D8 |
| IC-3 | Throttling mechanism taxonomy | Systematic comparison across throttling forms | IC-2b results required |
| IC-4 | Transfer premium benchmark | Quantify OOD transfer premium per mechanism | IC-2b results required |
| IC-5 | Operational realization benchmark | Closed-loop control evaluation | IC-2b results required |
| IC-6 | Active capital accumulation | Active probing vs passive trajectory | Only if IC-5 passes |
| IC-7 | Return to AEP-control | ONLY IF all IC-2b gates pass | Re-evaluate at that time |

**Current phase: IC-0 (completed by this document). Next: IC-1 (AEP capital audit).**

---

*Intelligence Capital Theory V1. This is a research program, not a validated theory. All claims require experimental audit.*