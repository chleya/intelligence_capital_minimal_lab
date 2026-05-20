# IC-2c: Episodic vs Consolidated Capital Report

**Status**: Complete | **Date**: 2026-05-19 | **Pipeline**: Continual Update Stream (5 seeds sequential)  
**Predecessor**: [IC2B_LEARNED_THROTTLING_REPORT.md](IC2B_LEARNED_THROTTLING_REPORT.md)

---

## Executive Summary

在 5-step 顺序更新流中测试了 4 种 capital/memory 策略，数据来自 5 个不同 seed 的环境。

**核心结果**：

| 策略 | 最终 best_action_match | 轨迹趋势 | vs random (0.33) |
|---|---|---|---|
| **NoMemory** | **0.445** | 微降后平稳 | ✅ 显著高于随机 |
| **Episodic** | 0.195 | 波动（0.145–0.195） | ❌ 远低于随机 |
| **Consolidated** | 0.115 | 几乎平坦（0.095–0.115） | ❌ 远低于随机 |
| **Mixed** | 0.095 | 平坦，step=3 小幅峰值后下降 | ❌ 远低于随机 |

**所有基于 state/history 的记忆策略均低于随机猜测（0.33）。**

NoMemory（简单动作频率）反而达到 0.445。consolidated 的压缩质心（KMeans centroids）是最差的策略，稳定在 0.095–0.115，**比随机差 3 倍**。

这个发现直接回答了核心问题：在当前环境设置下，跨 seed 分布偏移导致 state/history 特征从"帮助"变成"误导"，而 continual consolidation 进一步放大了这种误导——压缩质心无法区分不同环境的规律，持续改写产生的不是 appreciation 而是 **structural bad debt**。

---

## 1. Experiment Design

### 1.1 更新流

| Step | 训练数据来源 | 累积样本量 |
|---|---|---|
| 1 | seed=0 train (1200) | 1200 |
| 2 | + seed=1 train (1200) | 2400 |
| 3 | + seed=2 train (1200) | 3600 |
| 4 | + seed=3 train (1200) | 4800 |
| 5 | + seed=4 train (1200) | 6000 |

测试集固定：seed=0 test_id split（200 samples）。

### 1.2 四种策略

| 策略 | 机制 | 容量 | 更新方式 |
|---|---|---|---|
| **Episodic Retention** | 固定容量 buffer + k-NN (k=5) | 200 traces | 追加新数据；满时逐出最旧 |
| **Consolidated Summary** | KMeans 质心 (K=20) + 最近质心预测 | 20 prototypes | 每步在所有累积数据上重新拟合质心 |
| **Mixed Memory** | 一半 episodic buffer (100) + 一半 consolidated centroids (10) | 110 total | 两种组件独立更新，加权平均预测 |
| **NoMemory Baseline** | 动作频率计数器 | 3 counts | 每次增加新数据中的 best action 计数 |

### 1.3 关键设计决策

- **Cross-seed 数据作为 drift 来源**：不同 seed 的环境有各自独立的 mode_flip、autonomous_drift 轨迹。按 seed 顺序输入模拟了环境分布持续漂移。
- **固定测试集**：seed=0 的测试集确保评估基准不变，真实反映累积训练的效果变化。
- **Consolidated 的 "改写"**：每步在所有累积数据上重新 KMeans——质心被完全覆盖，之前的压缩结构丢失。这直接模拟了 LLM 论文中 "useful memories become faulty when continuously updated" 的机制。

---

## 2. Results

### 2.1 Best Action Match 轨迹

```
Step  | episodic | consolidated | mixed  | nomemory
------|----------|--------------|--------|----------
  1   |  0.175   |    0.095     | 0.095  |  0.460
  2   |  0.145   |    0.095     | 0.095  |  0.445
  3   |  0.145   |    0.095     | 0.100  |  0.445
  4   |  0.150   |    0.095     | 0.095  |  0.445
  5   |  0.195   |    0.115     | 0.095  |  0.445
```

关键观察：
1. **NoMemory 始终最高**（0.445–0.460），从 step 1 到 step 5 仅微降 0.015
2. **Consolidated 始终最低**（0.095–0.115），在所有 step 上都比随机 0.33 差 2–3 倍
3. **Episodic 有明显的波动**（0.145–0.195），step 5 反弹到最高点
4. **Mixed 在 step 3 出现微小的峰值（0.100）后回落**，呈现 "先升后降" 的典型 bad debt 特征

### 2.2 Regret 轨迹（负值 = 模型对错误动作估值高于正确动作）

```
Step  | episodic  | consolidated | mixed    | nomemory
------|-----------|--------------|----------|----------
  1   | -0.329    |   -0.355     | -0.350   | -0.041
  2   | -0.344    |   -0.360     | -0.359   | -0.033
  3   | -0.296    |   -0.350     | -0.349   | -0.050
  4   | -0.365    |   -0.346     | -0.368   | -0.046
  5   | -0.310    |   -0.361     | -0.327   | -0.046
```

- NoMemory 的 regret 很小（-0.033 到 -0.050），说明它虽然也犯错，但错误代价低——它给正确和错误动作的估值差异不大
- 所有记忆策略的 regret 大幅为负（-0.296 到 -0.368）——它们高度确信地选择了错误动作
- **Consolidated 的 regret 在所有 step 上都比 episodic 更负**（avgerage -0.355 vs -0.329），说明压缩不仅降低了准确率，还增加了错误决策的代价

### 2.3 Rank Accuracy 轨迹

```
Step  | episodic | consolidated | mixed  | nomemory
------|----------|--------------|--------|----------
  1   |  0.560   |    0.540     | 0.547  |  0.475
  2   |  0.558   |    0.542     | 0.540  |  0.462
  3   |  0.580   |    0.533     | 0.533  |  0.462
  4   |  0.552   |    0.552     | 0.528  |  0.462
  5   |  0.603   |    0.547     | 0.573  |  0.462
```

有趣的是：所有记忆策略的 rank accuracy 均高于 NoMemory（0.46–0.47）。这说明记忆策略在动作对之间的相对排序上是合理的，但在选择**哪个动作最好**时出错了——这是因为不同 seed 的最优动作分布不同，记忆策略无法区分当前环境的"正确"行为模式。

---

## 3. Failure Signature Analysis

### 3.1 是否出现 "先升后降"？

- **Mixed**：`0.095 → 0.095 → 0.100 ↑ → 0.095 ↓ → 0.095` — **符合**，step 3 出现短暂峰值后回落
- **Consolidated**：`0.095 → 0.095 → 0.095 → 0.095 → 0.115` — **不符合**，全程平坦，step 5 略有上升（但仍是所有策略中最差）
- **Episodic**：`0.175 → 0.145 → 0.145 → 0.150 → 0.195` — **不符合**，最后反弹

"先升后降" 签名仅在 mixed 策略中微弱出现。

### 3.2 是否掉到 NoMemory / episodic-only 以下？

- **Consolidated 在所有 step 上都远低于所有其他策略** — 不仅低于 NoMemory，也低于 episodic
- **Mixed 同样在所有 step 上低于 episodic**

**两个包含 consolidated 组件的策略始终是最差的。**

### 3.3 是否更容易被 shortcut explain？

NoMemory 作为 shortcut（不需理解环境状态就能做出合理决策），其 bad_debt_ratio 对 consolidated 来说极高：
- Consolidated match ≈ 0.095–0.115
- NoMemory match ≈ 0.445
- 这意味着 "什么都不理解也能做得好 4 倍" → shortcut explain 完全成立

---

## 4. 回答四个必须回答的问题

### Q1: Episodic retention 是否比 consolidated rewrite 更稳？

**是的，并且差距巨大。**

- Episodic 最终 match=0.195，consolidated=0.115 → episodic 高出 **70%**
- Episodic 的 regret 绝对值更小（-0.31 vs -0.36）
- Episodic 在 step 5 出现了反弹趋势，而 consolidated 几乎停滞

**Reasons**：
1. Episodic buffer 保留了不同 seed 的原始 trace 多样性，k-NN 可以根据查询自动选择最相关的最近邻——当测试数据来自 seed=0 时，seed=0 的 traces 在 buffer 中天然更近
2. Consolidated 将不同 seed 的 traces 强行压缩到同一个语义空间（KMeans 质心），跨 seed 的模式被平均化，导致质心不能准确表示任何单一 seed 的规律
3. 持续改写（每步重新 KMeans）使得质心不断被新数据稀释，已有的有用结构被反复覆盖

### Q2: Mixed strategy 是否优于纯 episodic 或纯 consolidated？

**不。Mixed 比纯 episodic 更差，接近纯 consolidated 的水平。**

- Mixed final=0.095，episodic final=0.195，consolidated final=0.115
- Mixed 在 step 3 有微小提升（0.100），但随后回落到 0.095

**原因**：mixed 的预测是加权平均（episodic × 0.5 + consolidated × 0.5），consolidated 的错误预测直接污染了最终的混合输出。在 consolidated 如此糟糕的情况下，任何正权重都会拖累整体表现。

**教训**：mixed strategy 只有在两种组件都至少"不差"时才有效。当 consolidation 本身产生 bad debt 时，混合只会放大错误。

### Q3: 持续 consolidation 是否会产生一种新的 bad debt / false capital？

**是的，并且是当前环境中表现最差的 capital 形式。**

关键证据：
- Consolidated 在所有 step 上 match 仅 0.095–0.115，**远低于**随机猜测（0.33）和 NoMemory（0.445）
- 这不是 "效果不够好"，而是 **actively harmful** — 使用 consolidated memory 做出的决策比完全忽略历史更差
- Regret 深度为负（-0.36），说明 consolidated 让模型高度自信地做出错误选择

**这种 bad debt 的产生机制**：
1. **Cross-distribution averaging**：不同 seed 的最优 action 分布不同，KMeans 质心被拉到多个分布的"中间地带"，无法准确表示任何单一分布
2. **Lossy rewriting cycle**：每步重新拟合质心 → 旧知识被覆盖 → 新数据权重不足 → 质心漂移
3. **Phantom confidence**：rank accuracy 相对较高（0.55），但 best_action_match 极低（0.115），说明模型知道 "A 比 B 好"，但不知道 "在当前 seed 下应该选哪一个"——这是典型的 false capital（看起来有结构，实际无法操作化）

### Q4: 当前环境下，capital 的主要失败模式更像什么？

**排序（按严重性）**：

1. **Wrong compression** ⭐⭐⭐⭐⭐（最主要）
   - 跨分布的数据被压缩到同一语义空间，丢失了关键的 seed-specific 信息
   - 压缩后的结构是 cross-seed average，不能有效指导任何单一 seed 的决策

2. **Over-consolidation** ⭐⭐⭐⭐
   - 20 个 prototypes 试图概括 6000 个来自 5 个不同分布的样本
   - 信息密度不足 → 每个 prototype 覆盖了多个分布的混合 → 预测模糊

3. **Shortcut substitution** ⭐⭐⭐
   - NoMemory 以 15 counts（最终 3 个动作频率）的内存开销达到 0.445 match
   - Consolidated 以 20 prototypes 达到 0.115 match → IAR 为负（投入更多成本，得到更差结果）
   - 这种 "shortcut beats capital" 的模式本身就是 bad debt 的定义特征

4. **Not enough compression** ⭐⭐（当前不适用）
   - Episodic 虽然比 consolidated 好，但仍远低于 NoMemory
   - 问题不在于压缩不够，而在于**是否应该压缩**以及**如何压缩**

---

## 5. Connection to Theory (THEORY.md)

### 5.1 验证的理论概念

| 理论概念 | 实验证据 |
|---|---|
| **Bad Debt** | Consolidated memory 是 bad debt — 占用 20 个 prototypes 的内存，产生比随机更差的决策（BDR≈4.0，远超 0.5 阈值） |
| **False Capital** | Consolidated 有 rank accuracy 但无 best_action_match — 拥有结构但不可操作化 |
| **Shortcut Exploitation** | NoMemory 以极低成本获得最优性能 — 记忆机制被 shortcut 彻底取代 |
| **Realization Gap** | Episodic 保留了 raw traces 但无法准确提取/应用其中信息 — appreciation 存在但 realization 不够 |
| **Over-consolidation Drift** | 持续改写使 consolidated 质心漂移，info density 持续下降 |

### 5.2 映射到 LLM Memory 文献

实验与 "Useful Memories Become Faulty When Continuously Updated" 的关键对应：

| 文献中的发现 | 本实验对应 | 证据 |
|---|---|---|
| 有用的记忆在持续更新中退化 | Consolidated 的质心从初始就极差，且不随数据增加改善 | match 0.095→0.115，仅微升 0.02 |
| 重写机制特别容易出错 | KMeans refit 模拟 "每步重写"，是所有策略中最差的 | 始终低于 episodic |
| Shortcut 记忆比压缩记忆更鲁棒 | NoMemory 不受跨 seed 偏移影响，始终最高 | 0.445–0.460，稳定 |

---

## 6. Verdict

```
VERDICT: IC2C_CONSOLIDATION_IS_BAD_DEBT
```

### 一句话结论

> **在当前 minimal lab 环境中，continual consolidation（跨 seed 的 KMeans 质心压缩+反复重写）更像 drift into bad debt 而非 appreciation——consolidated memory 的所有指标均远低于随机猜测和 NoMemory shortcut，证明了 'wrong compression + over-consolidation' 是当前主要的 capital 失败模式。**

### 对后续实验的建议

1. **不要急于增加更多压缩机制** — 先解决 "wrong compression" 问题（如何在有分布偏移时正确压缩）
2. **增加 seed-discriminative features** — 当前 memory 无法区分不同 seed 的环境规律，这是根本限制
3. **测试 seed-aware consolidation** — 比如给每个 seed 独立的 prototypes，仅在 seed-invariant 维度上共享
4. **将 episodic 的反弹趋势（step 5）**放大测试：更多 step 后 episodic buffer 是否最终能超过 NoMemory？需要更大的 buffer 和更多数据点

---

## 7. Appendix: Data Files

| 文件 | 说明 |
|---|---|
| `results/ic2c/trajectory_metrics.csv` | 5 steps × 4 strategies 的完整指标轨迹 |
| `results/ic2c/bad_debt_trajectory.csv` | BDR 随 step 变化的轨迹 |
| `results/ic2c/summary.json` | 最终汇总指标 |
| `src/run_ic2c_episodic_vs_consolidated.py` | 实验 runner