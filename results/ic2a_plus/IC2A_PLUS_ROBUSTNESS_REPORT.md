# IC-2a+ Robust Oracle Residual Audit Report

Generated: 2026-05-09

## 1. Base IC-2a Recap (Provisional Pass)
- Global RVR mean: 0.4045
- Rule SO match: 0.0948
- Rule AO match: 0.4705
- Oracle¡úAO gap: 0.5295
- SSR: 0.0383

## 2. True OOD Splits
| OOD Type | Oracle¡úAO Gap | Notes |
|----------|---------------|-------|
| background_shift | 0.4745 | Autonomous dynamics changed |
| action_gain_shift | 0.4745 | Action gain magnitude changed |
| sign_rule_shift | 0.4745 | Sign rule inverted |

## 3. Learned Baselines
| Baseline | Mean Match | Oracle Gap |
|----------|------------|------------|
| LearnedStateOnly (Classifier) | 0.6910 | 0.3090 |
| LearnedStateOnly (Regressor) | 0.3020 | 0.6980 |
| LearnedActionOnly | 0.4570 | 0.5430 |
| Majority Action | 0.4570 | - |

## 4. Zero-Action Optimal Proportion Gate (D6)
- Mean zero_action_optimal_rate: 0.2810
- Gate D6_zero_action_not_absent: PASS (>= 0.10)
- Preferred >= 0.15: PASS

## 5. Conditional RVR
- Mean conditional_rvr: 3.8000
- Mean global_rvr: 0.4045
- Ratio (cond/global): 9.39x

## 6. History Ablation
| Variant | Mean Match | vs Full |
|---------|------------|---------|
| full_history_len_8 | 0.6910 | baseline |
| short_history_len_2 | 0.7340 | --0.0430 |
| current_obs_only | 0.4655 | -0.2255 |
| permuted_history | 0.4945 | -0.1965 |

## 7. IC-2a+ Gates (A-F)
| Gate | Condition | Result |
|------|-----------|--------|
| A | Oracle > LearnedSO + 0.20 | PASS (0.309) |
| B | Oracle > LearnedAO + 0.20 | PASS (0.543) |
| C | conditional_rvr >= 0.25 | PASS (3.800) |
| D | OOD bg residual advantage kept | PASS (0.475) |
| E | zero_action_optimal_rate >= 0.10 | PASS (0.281) |
| F | seed_variance < oracle_gap / 2 | PASS (so_var=0.079, ao_var=0.038) |

## 8. Answers to Audit Questions

1. **Does IC-2a PASS hold under true OOD?**
   Yes ¡ª background_shift oracle advantage remains.

2. **Does LearnedStateOnly stay below Oracle?**
   Yes ¡ª gap is 0.309.

3. **Is 0-action severely absent?**
   No. Rate = 0.281.

4. **Is conditional RVR clearly above global RVR?**
   Yes ¡ª cond=3.8000, global=0.4045.

5. **Does history order have real value?**
   Yes ¡ª permuted impairment = 0.1965.

6. **May we proceed to IC-2b?**
   **PROCEED to IC-2b.**

