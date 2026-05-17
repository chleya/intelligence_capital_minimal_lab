# Intelligence Capital Minimal Lab

Minimal experimental framework for testing Intelligence Capital Theory.

## Quick Start

```bash
pip install -r requirements.txt
pytest -q
python -m src.run_ic2a_oracle_residual
```

## Structure

- `src/env_structured_volatility.py` — StructuredVolatilityEnv with hidden mode
- `src/counterfactual_table.py` — Counterfactual data generation
- `src/memory_baselines.py` — RawMemory and prototype baselines
- `src/models.py` — AEP/Residual/Counterfactual/CausalContrast compressors
- `src/throttling_mechanisms.py` — All 13 mechanisms
- `src/metrics.py` — All 25 metrics + death conditions
- `src/audit.py` — Bad debt audit + shortcut detection
- `src/train.py` — Training loop
- `src/run_ic2a_oracle_residual.py` — IC-2a experiment
- `src/run_ic2b_learned_compressors.py` — IC-2b experiment
- `src/visualize.py` — Results visualization
- `tests/` — Unit tests for env, CF table, metrics, controls

## Theory

See `THEORY.md` and `EXPERIMENT_PLAN.md` in this directory.
Full theory documents: `intelligence_capital_theory/` (ICTV1, Audit, Claim Ledger, Failure Taxonomy, Metrics, Test Design, Next Actions, Codex Handoff)

## Verification

```bash
pytest -q -p no:asyncio  # 27 tests, all must pass
```