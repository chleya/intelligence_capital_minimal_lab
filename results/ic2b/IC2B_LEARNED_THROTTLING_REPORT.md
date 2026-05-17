# IC-2b Learned Throttling Mechanism Comparison Report

Generated: 2026-05-09

## 1. Summary
- LearnedStateOnly baseline: 0.7870
- LearnedActionOnly baseline: 0.4790
- RawMemoryEqualCost baseline: 0.1670
- Best mechanism: counterfactual_compressor (0.7800)

## 2. Top 5 Mechanisms by StateOnly Gap
                mechanism  mean_match  stateonly_gap  actiononly_gap
       learned_state_only       0.787          0.000           0.308
counterfactual_compressor       0.780         -0.007           0.301
        centered_residual       0.568         -0.219           0.089
     residual_adversarial       0.548         -0.239           0.069
      residual_compressor       0.545         -0.242           0.066

## 3. Gate Results
SOME MINIMUM PASSES FAILED
Strong passes: 0/2

## 4. Death Condition Audit
- D1: All mechanisms <= SO + 0.05

## 5. Mechanism Rankings
| Rank | Mechanism | Match | vs SO | vs AO | BadDebt | IAR |
|------|-----------|-------|-------|-------|---------|-----|
| 1 | learned_state_only | 0.7870 | 0.0000 | 0.3080 | 1.000 | 0.000012 |
| 2 | counterfactual_compressor | 0.7800 | -0.0070 | 0.3010 | 1.000 | 0.000012 |
| 3 | centered_residual | 0.5680 | -0.2190 | 0.0890 | 1.000 | 0.000007 |
| 4 | residual_adversarial | 0.5480 | -0.2390 | 0.0690 | 1.000 | 0.000005 |
| 5 | residual_compressor | 0.5450 | -0.2420 | 0.0660 | 1.000 | 0.000006 |
| 6 | permuted_history_control | 0.5210 | -0.2660 | 0.0420 | 1.000 | 0.000006 |
| 7 | learned_action_only | 0.4790 | -0.3080 | 0.0000 | 1.000 | 0.012417 |
| 8 | aep_compressor | 0.4420 | -0.3450 | -0.0370 | 1.000 | 0.000002 |
| 9 | causal_contrast | 0.4170 | -0.3700 | -0.0620 | 1.000 | 0.000003 |
| 10 | shuffled_action_control | 0.3420 | -0.4450 | -0.1370 | 1.000 | 0.000000 |
| 11 | raw_memory_full | 0.2410 | -0.5460 | -0.2380 | 1.000 | 0.000000 |
| 12 | raw_memory_equal_cost | 0.1670 | -0.6200 | -0.3120 | 1.000 | 0.000000 |
| 13 | prototype_memory | 0.1170 | -0.6700 | -0.3620 | 1.000 | 0.000000 |

## 6. Answers to Audit Questions

1. **Does any mechanism beat LearnedStateOnly?**
   No - death condition D1 would apply.

2. **Does any mechanism beat RawMemoryEqualCost?**
   Yes

3. **Does any mechanism maintain transfer premium on OOD?**
   OOD evaluation limited (OOD tables may be incomplete).

4. **Which mechanisms are bad debt?**
   - learned_state_only: BDR=1.000
   - learned_action_only: BDR=1.000
   - raw_memory_full: BDR=1.000
   - raw_memory_equal_cost: BDR=1.000
   - prototype_memory: BDR=1.000
   - aep_compressor: BDR=1.000
   - residual_compressor: BDR=1.000
   - centered_residual: BDR=1.000
   - counterfactual_compressor: BDR=1.000
   - causal_contrast: BDR=1.000
   - residual_adversarial: BDR=1.000
   - shuffled_action_control: BDR=1.000
   - permuted_history_control: BDR=1.000

5. **Does ResidualCompressor beat AEPCompressor?**
   Residual: 0.5450 vs AEP: 0.4420

6. **Is CounterfactualCompressor the strongest?**
   Counterfactual: 0.7800 (ranked #9)

7. **Does CausalContrast add value?**
   CausalContrast: 0.4170 vs best learner

8. **Is ICT intelligence appreciation supported?**
   NO - Redesign needed. Learnable structure not captured.
