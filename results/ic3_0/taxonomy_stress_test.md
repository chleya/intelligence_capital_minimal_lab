# Taxonomy Stress Test (IC-3-0)

## Q1: Does Memory / Prototype / AEP / Residual / Probe / Policy remain meaningful for external tasks?

**Answer: Largely YES, with caveats.**

- **Memory-based (RawMemory, Prototype)**: Meaningful — external tasks like HiddenGoalGridWorld
  benefit from experience replay / nearest-neighbor lookup
- **AEP/Residual**: Meaningful — any task requiring outcome prediction benefits from parametric compression
- **PolicyClone**: **NOT meaningful** in external tasks where goal is hidden
  (PolicyClone only outputs fixed best_action for training utility)
- **Probe**: Meaningful — active information gathering is universally useful

## Q2: Are any two capitals indistinguishable in performance and function?

**Answer: Currently AEP and Residual are close (Δ ~0.005).**
However they differ in interpretability (Residual decomposes autonomous vs action effects).
This is a moderate concern — for IC-3 allocation, AEP/Residual may be treated as one cluster.

## Q3: Does the external task require a new capital form?

**Answer: YES. A "Goal-Inference Capital" is needed.**
Hidden-goal gridworld requires inferring the goal location from partial observations
and reward signals. None of the current 6 capital forms explicitly model
latent-goal inference. This suggests adding a 7th form.

## Q4: Are any capital forms completely useless in external tasks?

**Answer: PolicyClone is useless in hidden-goal tasks.**
PolicyClone memorizes a fixed policy for a known utility function.
When the utility/goal is hidden and must be inferred, PolicyClone
has zero transfer value.

## Q5: Does ICT capital taxonomy need revision?

**Answer: YES, minor revision needed.**

Current: Memory / Prototype / AEP / Residual / Probe / PolicyClone
Proposed: Memory / Prototype / Parametric / Action-Effect / Probe / PolicyClone / Goal-Inference

- Merge AEP + Residual into "Parametric Compression Capital" (they're nearly indistinguishable)
- Add "Goal-Inference Capital" for latent-goal tasks
- Consider removing PolicyClone from second-order capital portfolio
  (it has zero transfer value for multi-goal settings)
