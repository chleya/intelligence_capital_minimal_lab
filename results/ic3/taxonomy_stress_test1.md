# IC-3 Capital Taxonomy Stress Test

## 6-Capital Set in Combined Pipeline

1. **PolicyCloneCapital** (fixed-goal, dense behavior cloning)
2. **PrototypeOutcomeCapital** (dense-support, k-NN outcome prediction)
3. **AEPCapital** (goal-transfer, learned compression)
4. **ResidualCapital** (action-effect, residual prediction)
5. **SafeFallbackCapital** (low-risk, experience-weighted random)

## Task Coverage
- Task A (fixed-goal): PolicyClone expected strongest
- Task B (goal-transfer): AEP/Residual expected strongest
- Task C (dense-support): PrototypeOutcome expected strongest
- Task D (hidden-goal): All struggle, GoalInference-like behaviors needed

## Disturbance Test Results
- Sudden goal shift: tracking error spikes, recovery in ~10 steps
- Gradual drift: smooth adaptation via confidence decay
- One capital failure: impairment detected, weight reduced
- Memory aging: confidence gradually decays
- AEP extrapolation failure: OOD score triggers weight reduction
- Probe cost spike: cost-normalized reliability drops

## External Validation
HiddenGoalGridWorld serves as semi-real benchmark. Allocator has NO access to env identity.
Allocator input: 115 CapitalReport-derived features.
