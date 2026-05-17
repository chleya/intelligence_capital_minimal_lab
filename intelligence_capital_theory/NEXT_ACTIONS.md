# Next Actions

**Status:** Intelligence Capital Theory IC-0  
**Date:** 2026-05-09  

---

## Immediate (IC-0: Complete — this document)

✅ INTELLIGENCE_CAPITAL_THEORY_V1.md  
✅ AEP_CAPITAL_AUDIT.md  
✅ CLAIM_LEDGER.md  
✅ FAILURE_TAXONOMY.md  
✅ METRICS_AND_DEATH_CONDITIONS.md  
✅ CHANGE_CAPITAL_MINIMAL_TEST_DESIGN.md  
✅ NEXT_ACTIONS.md (this file)  
✅ CODEX_HANDOFF.md  

---

## Next 3 Steps

### Step 1: IC-1 — AEP Capital Audit Review

**What:** Review the capital audit against original Phase 1–8 results. Verify all retained/withdrawn/uncertain claims are properly traced to evidence files.

**Output:** Validation that the claim ledger is evidence-grounded.

**Time:** 1–2 hours (review only, no new experiments).

---

### Step 2: IC-2a — Oracle Residual Accounting Test

**What:** Execute the oracle residual accounting experiment defined in `CHANGE_CAPITAL_MINIMAL_TEST_DESIGN.md` and `立项文件1.md`. This is the critical gate.

**Implementation location:** `F:\intelligence_capital_minimal_lab\`

**Key actions:**
1. Create the full project structure (立项文件1.md spec)
2. Implement StructuredVolatilityEnv
3. Generate counterfactual tables
4. Run oracle residual audit
5. Compute: M3, M7, M12, M24
6. Check: D1, D2, D3, D7
7. Output: IC2A_ORACLE_RESIDUAL_REPORT.md

**Gate:** Only proceed to IC-2b if ALL of:
- residual_variance_ratio ≥ 0.15
- residual_oracle_match > StateOnly
- counterfactual_value > 0
- seed_stability_ratio < 0.5

**Time:** 1–3 days (implementation + execution).

---

### Step 3: IC-2b — Learned Throttling Mechanism Test

**What:** Train and compare all 13 throttling mechanisms against baselines.

**Gate:** Only if IC-2a passes.

**Key actions:**
1. Train all mechanisms on same data
2. Evaluate ID + OOD
3. Compute full metrics (M8–M25)
4. Run bad debt audit
5. Compare IAR across mechanisms
6. Output: IC2B_LEARNED_COMPRESSOR_REPORT.md

**Time:** 2–5 days (training + evaluation).

---

## What Is Explicitly FORBIDDEN Right Now

- ❌ Phase 8.1 (new model architectures)
- ❌ Manifold steering experiments
- ❌ 2D environments (until 1D passes)
- ❌ AGI grand claims
- ❌ Metaphor-as-result ("the model learned operational structure")
- ❌ Results without death condition checks
- ❌ Single-seed claims
- ❌ "Promising" without cost comparison to RawMemory

---

## What IS Allowed

- ✅ IC-2a implementation and execution
- ✅ Environment redesign if IC-2a fails
- ✅ Documentation and theory refinement
- ✅ Code quality and test coverage
- ✅ Critical re-evaluation of any retained claim

---

## Decision Tree

```
Complete IC-2a
  │
  ├── IC-2a PASSES → Proceed to IC-2b
  │     │
  │     ├── IC-2b: ≥1 mechanism passes → Proceed to IC-3
  │     └── IC-2b: 0 mechanisms pass → Return to env design
  │
  └── IC-2a FAILS → Redesign env
        │
        ├── After redesign, re-run IC-2a
        └── If 3 redesigns all fail → STOP. This environment class cannot support the claim.
              Archive results. Write postmortem.
```

---

## Summary

The AEP program accumulated diagnostic capital. Phase 8.0 proved it cannot be realized as control capital under current designs. Intelligence Capital Theory is the framework for understanding why. The Change Capital Minimal Test is the experiment that will determine whether ANY throttling mechanism can produce genuine intelligence appreciation — or whether even under optimized conditions, the realization gap is fundamental.

**Next immediate action: Create `F:\intelligence_capital_minimal_lab\` and implement IC-2a.**