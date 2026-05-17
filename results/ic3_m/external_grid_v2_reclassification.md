# External Grid-v2 Reclassification

## Final Status: `EXTERNAL_CAPITAL_VALIDATION_ONLY` | `NOT_ALLOCATOR_VALIDATION`

The current HiddenGoalGridWorld-v2 shows:

| Capital | Score |
|---|---|
| GoalInference | 1.0000 |
| PolicyClone | 0.0000 |
| PrototypeOutcome | 0.0000 |
| AEP | 0.0000 |
| SafeFallback | 0.0000 |

- OracleHindsight = BestSingle = GoalInference = 1.0000
- No capital-switching pressure exists.
- No meaningful allocator decision to make.

### What Grid-v2 Proves

It proves exactly one thing: **GoalInferenceCapital correctly solves the hidden-goal spatial task.**

It cannot prove anything about the allocator, because there is nothing to allocate — one capital is perfect, all others are useless.

### Decision

Grid-v2 is **retained as a GoalInferenceCapital validation benchmark** (single-capital regression test).
It is **excluded from allocator validation** because it contains no meaningful capital-switching pressure.
It does **not contribute to second-order allocation evidence**.

### Grid-v2 Future Role

Grid-v2 is relocated to:

```
tests/capital_validation/test_goal_inference_grid_v2.py
```

Its purposes are:
1. Verify that GoalInferenceCapital still works correctly
2. Prevent future modifications from breaking GoalInference
3. Not included in allocator scoring
4. Not counted as second-order allocation evidence

### Requirements for Future External Allocator Benchmarks

Any external task that aspires to be an **allocator benchmark** (Grid-v3, Construction-Site Scheduling Toy, etc.) must satisfy:

1. **At least 3 capitals nontrivially useful**: each with score > 0.10
2. **OracleHindsight - BestSingle ≥ 0.10**: real capital-switching pressure
3. **BestSingle < 0.85**: no single capital solves all steps
4. **Each capital best on ≥ 10% of steps**: diverse capital utility regions
5. **Proxy suppression resilient**: selector does not trivially win via region labels

### Grid-v3 Specific Requirements (if Grid-based allocator benchmark is pursued)

PolicyClone useful on familiar route segments.
PrototypeOutcome useful on landmarks / repeated layouts.
AEP useful on dynamic obstacle / local action-effect changes.
GoalInference useful on hidden-goal / partial observation.
SafeFallback useful in high-risk zones.

The task must create heterogeneous local conditions where different capitals dominate, rather than a single global condition where one capital wins everything.

### Naming Convention

- **IC-3**: Capital Allocator / external capital configuration
- **IC-4**: Internal Circuit Capital / LLM internal circuit capitalization

For future IC-3 deployable allocator validation, use tasks where multiple capitals are simultaneously competent and capital switching adds value.

For future allocator validation, this task (Grid-v2) should not be included in the main allocator score unless it is redesigned to create multi-capital switching pressure.