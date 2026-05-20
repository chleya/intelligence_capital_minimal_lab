# IC-2c.1: Root Cause Analysis — Why NoMemory > Episodic > Consolidated

**Status**: Complete | **Date**: 2026-05-19 | **Predecessor**: [IC2C_EPISODIC_VS_CONSOLIDATED_REPORT.md](../IC2C_EPISODIC_VS_CONSOLIDATED_REPORT.md)

---

## Executive Summary

IC-2c found a clear performance ordering: **NoMemory (0.445) ≫ Episodic (0.195) > Consolidated (0.115) > Mixed (0.095)**. IC-2c.1 decomposes exactly why by examining per-seed data distributions, feature-space geometry, and centroid evolution.

**Three root causes identified**:

| Question | Root Cause | Mechanism |
|---|---|---|
| Why NoMemory wins? | Stable action-frequency shortcut across seeds | inter-seed dominant action prob = 0.468 ± 0.028 |
| Why episodic fails? | History features ≈ noise for k-NN; cross-seed feature drift amplifies | k-NN caps at 0.22 vs random 0.33; feature means shift by 0.18–0.93 between seeds |
| Why consolidated drifts? | Centroid rewriting + cluster imbalance → information dilution | imbalance ratio grows 2.86→7.27; per-step centroid drift ~1.0–1.3 |

---

## 1. Q1: Why NoMemory Wins — The Action-Frequency Shortcut

### 1.1 Per-Seed Action Distributions

| seed | action_-1 | action_0 | action_+1 | dominant |
|---|---|---|---|---|
| 0 | 0.457 | 0.104 | 0.439 | -1 |
| 1 | 0.439 | 0.103 | 0.458 | +1 |
| 2 | 0.433 | 0.049 | 0.518 | +1 |
| 3 | 0.446 | 0.107 | 0.448 | +1 |
| 4 | 0.431 | 0.110 | 0.459 | +1 |

**Key finding**: The inter-seed dominant action probability is **0.468 ± 0.028** — extremely stable. NoMemory simply learns "always pick `action_+1`" (or `action_-1` for seed=0), and gets ~46.4% accuracy.

**Test set (seed=0)**: action_-1 dominates at 0.460. NoMemory achieves 0.460 at step 1 and stabilizes at 0.445 — essentially at the frequency ceiling.

### 1.2 Is this a "state shortcut"?

**No — it's even simpler.** NoMemory doesn't use state features at all. It exploits a task-level regularity: the best action distribution is bimodal (action_-1 and action_+1 each ~45%, action_0 ~10%) and stable across seeds.

This is not a sophisticated shortcut — it's a **trivial frequency baseline**. The fact that it beats all memory strategies (0.445 vs best 0.195) tells us: **the memory strategies are making things WORSE than doing nothing.**

### 1.3 Lesson

> **NoMemory doesn't "win" because it's smart. It wins because the memory strategies are actively harmful — they take a simple problem (pick the most common action) and inject noise.**

---

## 2. Q2: Why Episodic Didn't Become Appreciation

### 2.1 k-NN Signal Quality

| Buffer | match vs test(seed=0) |
|---|---|
| s0 only (1200 samples) | **0.220** |
| s0+s1 (2400) | 0.215 |
| s0+s1+s2 (3600) | 0.230 |
| s0-s3 (4800) | 0.240 |
| s0-s4 (6000) | 0.235 |

**Shocking result**: k-NN with all 6000 samples achieves only 0.235 best_action_match — **below the 0.33 random baseline for 3 actions.**

Even using ONLY seed=0 data (which is the same seed as the test set), k-NN gets only 0.220. The history features (24-dim: 8 steps × (observation state + action)) simply do not contain enough Euclidean-distance structure for k-NN to distinguish the best action.

### 2.2 Cross-Seed Feature Drift

Inter-seed feature mean distances:

| Seed pair | L2 distance |
|---|---|
| s0↔s1 | 0.223 |
| s0↔s2 | **0.471** |
| s0↔s3 | **0.482** |
| s2↔s3 | **0.931** |
| s3↔s4 | 0.593 |

Seeds 2 and 3 are especially different from each other (L2=0.93) and from seed=0 (L2=0.47-0.48). When we mix all seeds into one k-NN buffer, the effective nearest-neighbor set increasingly pulls from different distributions.

### 2.3 Cross-Contamination

Adding cross-seed data to k-NN actually *slightly improves* accuracy (from 0.220→0.235, +0.015). But this is negligible — k-NN on history features is fundamentally capped around 0.22-0.24, well below random.

**The real problem is not cross-seed contamination — it's that the feature representation itself is too weak for k-NN.**

Compare with IC-2b: learned compressors (ResidualCompressor, CounterfactualCompressor) achieve ~0.60 best_action_match on the same data. The features DO contain useful information — but it requires MLP-based learned extraction, not Euclidean k-NN.

### 2.4 Lesson

> **Episodic retention fails not because of buffer pollution, but because raw history features + k-NN is a fundamentally insufficient readout for this task. Appreciation requires learned feature extraction, not just raw storage.**

---

## 3. Q3: Why Consolidated Drifts — Centroid Evolution

### 3.1 Cluster Quality Metrics

| Step | n_samples | imbalance_ratio | gini | centroid_drift_from_prev |
|---|---|---|---|---|
| 1 (s0) | 1200 | 2.86 | 0.137 | — |
| 2 (+s1) | 2400 | **4.94** | 0.136 | 1.29 |
| 3 (+s2) | 3600 | 2.94 | 0.111 | 1.14 |
| 4 (+s3) | 4800 | 4.43 | 0.093 | 0.99 |
| 5 (+s4) | 6000 | **7.27** | 0.107 | 1.19 |

Three degradation signatures:

1. **Per-step centroid drift ~1.0–1.3**: Each new seed causes centroids to move by ~1.0-1.3 in feature space (relative to a centroid-to-centroid radius of ~2.9). This is a substantial fraction of the cluster radius — centroids are being *rewritten* each step, not refined.

2. **Cluster imbalance grows 2.86→7.27**: The largest cluster has 7× more samples than the smallest. Some centroids become "mega-clusters" that cover multiple different seeds' distributions, losing specificity.

3. **No empty clusters, but extreme imbalance**: All 20 prototypes stay active, but their quality degrades. The gini coefficient (0.09-0.14) doesn't change much because imbalance is structural, not growing uniformly.

### 3.2 The Drift Mechanism

```
Step 1: KMeans on seed=0 → 20 centroids learn seed=0's distribution
Step 2: +seed=1 → KMeans refits → centroids shift to accommodate seed=1
  → old seed=0-specific structure is overwritten
Step 3: +seed=2 → more shift → centroids now average over 3 distributions
  → cluster assignments become less meaningful per-distribution
Step 4-5: imbalance accelerates → some centroids become "catch-all" for mixed seeds
```

The core issue: **KMeans doesn't know about seed boundaries.** It just minimizes within-cluster variance on the combined data, which pulls centroids toward the cross-seed average. For a fixed-capacity prototype system (K=20), this means each prototype becomes increasingly non-specific as more diverse data is added.

### 3.3 Lesson

> **Consolidated centroids drift because the rewriting process (refit KMeans on all data) treats different distributions as one — creating "average" prototypes that represent no single distribution well. The imbalance ratio of 7.27 at step 5 confirms that information is concentrating into a few mega-clusters while the rest become near-useless.**

---

## 4. Synthesis: The Complete Causal Chain

```
                         ┌─────────────────────────┐
                         │   Environment has stable │
                         │   action-frequency prior │
                         │   (action_±1 ~45% each)  │
                         └───────────┬─────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
     ┌────────────────┐   ┌─────────────────┐   ┌─────────────────┐
     │   NoMemory     │   │    Episodic     │   │  Consolidated   │
     │                │   │                 │   │                 │
     │ reads prior    │   │ k-NN on 24-dim  │   │ KMeans on mixed │
     │ directly       │   │ history → noise │   │ seeds → average │
     │ match=0.445    │   │ match=0.220     │   │ match=0.115     │
     └────────────────┘   └─────────────────┘   └─────────────────┘
              │                      │                      │
              ▼                      ▼                      ▼
        "Just pick the         "History features       "Centroids become
        most common            are below noise         non-specific
        action"                for raw k-NN"           mega-clusters"
```

**All three strategies are responding to the same underlying reality**: the task has a strong action-frequency prior. NoMemory exploits it directly. Episodic and Consolidated try to use history features that are too noisy for their respective readout mechanisms (Euclidean k-NN and centroid-based), making them actively worse than ignoring history entirely.

---

## 5. What This Means for THEORY.md

### 5.1 Bad Debt / False Capital — Now With Mechanism

IC-2c showed that consolidated memory produces bad debt. IC-2c.1 reveals the **mechanism**:

> **Bad debt is not just "poor performance" — it is a specific structural failure where compressed representations lose distribution-specific information through cross-distribution averaging.**

This maps directly to the LLM memory finding ("Useful Memories Become Faulty When Continuously Updated"): the degradation mechanism is the same — repeated rewriting across heterogeneous experiences creates averaged representations that are worse than no memory at all.

### 5.2 Shortcut vs Capital — Clarified

NoMemory is a **shortcut** (it bypasses the state → action mapping entirely). The key lesson is not that shortcuts exist (they always do), but that:

> **A capital mechanism can be worse than the simplest shortcut if its readout method is mismatched to its representation format.**

Episodic has more information (6000 raw traces) than NoMemory (3 action counts), but its readout (Euclidean k-NN) can't extract it. This is a **realization gap**: appreciation exists (the information is there, as IC-2b's learned compressors prove), but it's not operationally realizable through the chosen readout.

### 5.3 Over-Consolidation — Quantified

The centroid imbalance (2.86→7.27) and per-step drift (~1.0-1.3) give us quantitative signatures for "over-consolidation":
- **Imbalance ratio > 5**: prototypes have lost per-distribution specificity
- **Per-step centroid drift > 30% of cluster radius**: rewriting is structural, not refinement

---

## 6. Verdict

```
VERDICT: IC2C1_TRIVIAL_SHORTCUT_BEATS_NOISY_CAPITAL
```

### Two One-Sentence Conclusions

**For NoMemory**:  
> NoMemory wins because the task has a stable action-frequency prior (0.468 ± 0.028 across seeds), and a 3-count frequency model captures it perfectly — this is a trivial shortcut, not a sophisticated strategy.

**For episodic/consolidated**:  
> Episodic retention fails because raw 24-dim history features + Euclidean k-NN is too weak a readout (caps at 0.235, below random 0.33); consolidated rewrites amplify this failure by averaging across distributions, creating non-specific mega-clusters (imbalance 7.27×) that predict worse than random.

### The Bigger Picture

> **In this environment, the capital failure mode is not "not enough compression" — it's "wrong readout for the representation type." Raw traces contain useful structure (proven by learned compressors hitting 0.60), but Euclidean k-NN and centroid-based readouts cannot extract it. The shortcut wins because it bypasses the readout problem entirely, not because it understands more.**

---

## 7. Data Files

| File | Content |
|---|---|
| `results/ic2c/q1_action_distributions.csv` | Per-seed action frequency analysis |
| `results/ic2c/q2_feature_distributions.csv` | Cross-seed feature drift metrics |
| `results/ic2c/q2_knn_signal_degradation.csv` | k-NN match degradation per step |
| `results/ic2c/q3_centroid_drift.csv` | Centroid quality metrics per step |
| `results/ic2c/ic2c1_root_cause_summary.json` | Three-question summary |
| `src/run_ic2c1_root_cause.py` | Analysis script |