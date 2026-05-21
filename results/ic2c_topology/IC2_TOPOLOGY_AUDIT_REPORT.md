# IC-2: Consolidation Topology Audit Report

**Experiment:** Relational Memory Hypothesis — Experiment #2  
**Date:** 2026-05-20  
**Status:** CPU-executed, no GPU required

---

## 1. Purpose

This experiment audits whether consolidation (KMeans centroids) preserves 
the high-dimensional relational topology of episodic traces. The Relational 
Memory Hypothesis predicts that centroid-based consolidation destroys 
topological structure, explaining why consolidated match < random.

## 2. Metrics Computed

| Metric | Definition | Death Condition |
|---|---|---|
| **TPR** | Topology Preservation Ratio — corr(episodic pairwise distances, centroid distances) | D10: TPR < 0.5 |
| **RRP** | Relational Recall Precision — fraction of nearest neighbors preserved after consolidation | D12: RRP < 0.5 |
| **Purity** | Cluster purity — how much centroids mix different seeds' distributions | — |
| **MEC proxy** | Multi-Entry Consistency — recall robustness to query perturbation | D11: MEC < 0.5 |

## 3. TPR: Topology Preservation Ratio

| Step | N Samples | N Centroids | TPR | Pearson Baseline | TPR Excess |
|---|---|---|---|---|---|
| 1 | 1200 | 20 | 0.8735 | 0.0007 | 0.8728 |
| 2 | 2400 | 20 | 0.8870 | -0.0005 | 0.8875 |
| 3 | 3600 | 20 | 0.8701 | -0.0000 | 0.8702 |
| 4 | 4800 | 20 | 0.8764 | 0.0001 | 0.8763 |
| 5 | 6000 | 20 | 0.8752 | 0.0005 | 0.8747 |

**Final TPR: 0.8752**

**OK**: TPR >= 0.7 — most relational structure preserved.

## 4. RRP: Relational Recall Precision

| Step | N Samples | N Centroids | NN Tested | NN Preserved | RRP |
|---|---|---|---|---|---|
| 1 | 1200 | 20 | 100 | 72 | 0.7200 |
| 2 | 2400 | 20 | 100 | 68 | 0.6800 |
| 3 | 3600 | 20 | 100 | 64 | 0.6400 |
| 4 | 4800 | 20 | 100 | 64 | 0.6400 |
| 5 | 6000 | 20 | 100 | 70 | 0.7000 |

**Final RRP: 0.7000**

**OK**: RRP >= 0.5 — local geometry partially preserved.

## 5. Cluster Purity: Seed Mixing

**Mean purity: 0.2605**
**Clusters with purity < 0.8: 20/20**

Centroids significantly mix distributions from different seeds, 
confirming that cross-distribution averaging creates structurally impure clusters.

## 6. MEC Proxy: Multi-Entry Recall Consistency

**Mean recall consistency: 1.0000**
**Mean match-to-true: 1.0000**


## 7. Death Condition Summary

| Death Condition | Threshold | Actual | Triggered? |
|---|---|---|---|
| D10: TPR < 0.5 | < 0.5 | 0.8752 | NO |
| D11: MEC < 0.5 | < 0.5 | 1.0000 | NO |
| D12: RRP < 0.5 | < 0.5 | 0.7000 | NO |

## 8. Interpretation

Topology is partially preserved. The Relational Memory Hypothesis prediction 
is NOT strongly supported. Other mechanisms may explain the consolidation failure.

---

*IC-2 Consolidation Topology Audit — Relational Memory Hypothesis Experiment #2*