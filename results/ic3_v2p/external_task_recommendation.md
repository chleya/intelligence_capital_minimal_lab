# External Task Replacement Recommendation

## Why HiddenGoalGridWorld (Task D) Is Uninformative

| Factor | Analysis |
|---|---|
| **Horizon** | GridWorld default is 30 steps; 7x7 grid requires multiple correct turns. Too short for GoalInference to accumulate belief updates |
| **Reward sparsity** | Only +1 at goal; capital correctness metric = at_goal (100% sparse). Even random walk has <5% success |
| **GoalInference strength** | Simple belief-update from binary obs → slow convergence. 30 steps insufficient for 49-state belief convergence |
| **Other capitals** | PolicyClone/AEP/Prototype are state-action classifiers with action space [0,1,2] but grid needs [0,1,2,3]. Mismatch → all near random |
| **Oracle gain** | OracleHindsight ≈ 0.05-0.08 across seeds → there is barely any oracle gain to exploit |

## Recommended Replacements

### Option 1: HiddenGoalGridWorld-v2 (easier)
- Reduce grid to 5x5 (25 states, shorter paths)
- Increase horizon to 100 steps
- Add partial-goal-reached reward = 0.3 at waypoints
- Use capital action set matched to grid directions [0,1,2,3]

### Option 2: MiniGrid-like Key-Door Task
- 5x5 grid, door opens with key
- Capital split: PolicyClone learns path; GoalInference infers key location from partial observation
- Multi-step planning vs reactive policy distinction

### Option 3: Construction-Site Scheduling
- Multi-resource scheduling with uncertainty
- Capital split: PrototypeOutcome (historical scheduling patterns), AEP (learned resource allocation), GoalInference (infer hidden constraints)
- Synthetic benchmark with controllable difficulty

### Option 4: Navigation with Local Memory
- Partially observable maze
- Capital split: PolicyClone (learned reactive policy), GoalInference (belief over hidden state), Prototype (landmark-based navigation)
- Reward shaping for intermediate progress

**Recommendation**: Start with Option 1 (easier HiddenGoalGridWorld) as minimal change; if still uninformative, switch to Option 2 (Key-Door) or Option 4 (Navigation with Local Memory) which provide clear capital specialization grounds.
