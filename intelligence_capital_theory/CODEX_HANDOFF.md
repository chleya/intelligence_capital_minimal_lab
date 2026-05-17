# Codex Handoff

**To:** Next agent / developer  
**From:** Phase R0 postmortem session  
**Date:** 2026-05-09  

---

## 1. Current Project Status

**Intelligence Capital Theory (ICT)** is a theoretical upgrade of the AEP (Action-Effect Profile) program (Phases 1–8.0). The AEP program discovered descriptive capital (action-effect structure characterization tools) but failed to convert this into control advantage (Phase 8.0: all death conditions triggered).

ICT reframes intelligence as: _world-change information that is compressed, throttled, structurally retained, and then realized as value in new contexts_.

**The project is in IC-0 (theory construction). No new experiments have been run under ICT.**

---

## 2. What NOT to Do

- ❌ Do NOT resume AEP Phase 8.1 (new model architectures)
- ❌ Do NOT run manifold steering experiments
- ❌ Do NOT add more environments to the Phase 8.0 benchmark suite
- ❌ Do NOT claim intelligence from latent geometry without grounded counterfactual audit
- ❌ Do NOT report single-seed results as discoveries
- ❌ Do NOT skip IC-2a (oracle residual accounting) — learned compressor comes AFTER oracle passes
- ❌ Do NOT let hidden_mode leak into model inputs
- ❌ Do NOT evaluate prediction and call it control

---

## 3. What You CAN Do

- ✅ Implement `F:\intelligence_capital_minimal_lab\` from scratch (no dependency on old AEP code)
- ✅ Run IC-2a: oracle residual accounting test
- ✅ Redesign StructuredVolatilityEnv if IC-2a fails
- ✅ Write or refine theory documents
- ✅ Add tests
- ✅ Question any retained claim

---

## 4. File Locations

### Theory documents (read-only reference)
```
F:\intelligence_capital_theory\
  INTELLIGENCE_CAPITAL_THEORY_V1.md    — Main theory
  AEP_CAPITAL_AUDIT.md                  — Phase 1–8 re-audited as capital
  CLAIM_LEDGER.md                       — Retained / withdrawn / uncertain
  FAILURE_TAXONOMY.md                   — 5 categories of failure
  METRICS_AND_DEATH_CONDITIONS.md       — Executable metric table
  CHANGE_CAPITAL_MINIMAL_TEST_DESIGN.md — IC-2 experiment design
  NEXT_ACTIONS.md                       — Step-by-step plan
  CODEX_HANDOFF.md                      — This file
  立项文件.md                           — Original project spec
  立项文件1.md                          — Original lab spec
```

### Experimental project (to be created)
```
F:\intelligence_capital_minimal_lab\
  README.md
  THEORY.md
  EXPERIMENT_PLAN.md
  requirements.txt
  configs\default.yaml
  src\
    __init__.py
    env_structured_volatility.py
    counterfactual_table.py
    memory_baselines.py
    models.py
    throttling_mechanisms.py
    metrics.py
    audit.py
    train.py
    run_ic2a_oracle_residual.py
    run_ic2b_learned_compressors.py
    visualize.py
  tests\
    test_env.py
    test_counterfactual_table.py
    test_metrics.py
    test_controls.py
  results\
    .gitkeep
```

### AEP project (historical reference only)
```
F:\aep_operational_state_v01\
  src\phase*.py                          — All Phase 1–8.0 source
  results\phase*\                        — All Phase 1–8.0 results
  results\phaseR0\                       — Postmortem documents
```

---

## 5. Minimum Next Command

```bash
# Create and enter the lab
mkdir F:\intelligence_capital_minimal_lab
cd F:\intelligence_capital_minimal_lab

# Create project according to 立项文件1.md specification
# Then:
pip install -r requirements.txt
pytest -q

# First mandatory experiment:
python -m src.run_ic2a_oracle_residual --config configs/default.yaml

# ONLY if IC-2a passes:
python -m src.run_ic2b_learned_compressors --config configs/default.yaml
```

---

## 6. Acceptance Criteria

### IC-2a passes if:
- [ ] residual_variance_ratio ≥ 0.15
- [ ] residual_oracle_match > StateOnly match
- [ ] counterfactual_value > 0
- [ ] seed_stability_ratio < 0.5
- [ ] 10 seeds tested (not 3)
- [ ] IC2A_ORACLE_RESIDUAL_REPORT.md answers all 5 required questions

### IC-2b passes if at least ONE throttling mechanism:
- [ ] OOD transfer premium > 0
- [ ] IAR > RawMemoryEqualCost
- [ ] best_action_match > StateOnly + 0.10
- [ ] best_action_match > ActionOnly + 0.10
- [ ] bad_debt_ratio < 0.5
- [ ] shuffled_action_gap > 0.05
- [ ] permuted_history_gap > 0.10 in delay tasks
- [ ] seed_stability_ratio < 0.5

### Overall project passes if:
- [ ] IC-2b has ≥1 passing mechanism
- [ ] Intelligence appreciation is SUPPORTED (not just "promising")
- [ ] CI passes (pytest -q)
- [ ] All reports generated

---

## 7. Key Contacts / Context

The AEP program ran for 8 major phases. Key empirical findings:
- AEP beats observation similarity for state representation (retained)
- CTD/ECTD quantify action-effect topology (retained)
- Factorized architecture predicts RealAEP (retained)
- StateOnly beats Factorized in all 6 environments (withdrawn claim about control)
- GRU temporal encoder does not use temporal order (withdrawn claim about temporal models)
- CF data provides zero stable marginal value (withdrawn claim about CF data)

The core lesson: **Prediction capital ≠ Control capital.** The realization gap is the central research problem. Intelligence Capital Theory exists to formalize and eventually bridge this gap — but it must first survive its own audit system.

---

## 8. Final Note

This project's highest goal is not to prove ICT is correct. It is to build an audit system where ICT can be proven wrong. If all throttling mechanisms fail against RawMemory + StateOnly + ActionOnly + Shuffled/Permuted controls, then intelligence appreciation — in this formulation — does not exist. That is a valid and valuable result.

**Good luck. Be critical. Kill your darlings.**