"""
CapitalReport — Unified Capital Interface (IC-3-0)
====================================================
All capital forms expose a standard CapitalReport to the Allocator.
The Allocator is FORBIDDEN from accessing capital internals directly.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class CapitalReport:
    capital_id: str
    capital_type: str
    timestamp: int

    recommended_action: int = -1
    predicted_utility: float = 0.0
    recent_prediction_error: float = 0.0
    recent_regret: float = 0.0
    confidence: float = 1.0
    calibration_error: float = 0.0
    realized_utility: float = 0.0
    realization_rate: float = 1.0
    capital_local_ood_score: float = 0.0
    nearest_support_distance: float = 0.0
    inference_cost: float = 0.0
    update_cost: float = 0.0
    storage_cost: float = 0.0
    probe_cost: float = 0.0
    goal_shift_score: float = 0.0
    transfer_success_rate: float = 1.0
    recent_transfer_regret: float = 0.0
    expected_probe_value: float = 0.0
    uncertainty_reduction_if_probe: float = 0.0
    capital_age: int = 0
    depreciation_score: float = 0.0
    bad_debt_score: float = 0.0

    impaired: bool = False
    impairment_flag: float = 0.0
    last_validated_timestamp: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "extra"}

    def to_vector(self):
        return np.array([
            float(self.recommended_action), self.predicted_utility,
            self.recent_prediction_error, self.recent_regret, self.confidence,
            self.calibration_error, self.realized_utility, self.realization_rate,
            self.capital_local_ood_score, self.nearest_support_distance,
            self.inference_cost, self.update_cost, self.storage_cost,
            self.probe_cost, self.goal_shift_score,
            self.transfer_success_rate, self.recent_transfer_regret,
            self.expected_probe_value, self.uncertainty_reduction_if_probe,
            float(self.capital_age),
            self.depreciation_score, self.bad_debt_score,
            self.impairment_flag,
        ], dtype=np.float32)


class Capital:
    def __init__(self, capital_id: str, capital_type: str):
        self.capital_id = capital_id
        self.capital_type = capital_type
        self.timestep = 0
        self._history: List[CapitalReport] = []

    def generate_report(self, context: Dict[str, Any], history: List[Dict[str, Any]]) -> CapitalReport:
        raise NotImplementedError

    def act(self, context: Dict[str, Any], history: List[Dict[str, Any]]) -> int:
        raise NotImplementedError

    def update(self, feedback: Dict[str, Any]) -> None:
        self.timestep += 1

    def _base_report(self) -> CapitalReport:
        return CapitalReport(capital_id=self.capital_id, capital_type=self.capital_type, timestamp=self.timestep)


class PolicyCloneCapital(Capital):
    def __init__(self, model, capital_id="PolicyClone"):
        super().__init__(capital_id, "PolicyClone")
        self.model = model
        self.model.eval()
        self._recent_correct = []
        self._recent_utilities = []
        self._last_report = None

    def generate_report(self, context, history):
        r = self._base_report()
        r.confidence = self._decay_confidence()
        if len(self._recent_correct) >= 5:
            r.recent_prediction_error = 1.0 - np.mean(self._recent_correct[-20:])
            r.recent_regret = r.recent_prediction_error
            r.calibration_error = np.std(self._recent_correct[-20:]) if len(self._recent_correct) >= 20 else 0.0
        r.realized_utility = float(np.mean(self._recent_utilities[-20:])) if self._recent_utilities else 0.0
        r.inference_cost = 1.0
        r.storage_cost = sum(p.numel() for p in self.model.parameters()) * 4
        r.update_cost = 5000.0
        self._last_report = r
        return r

    def act(self, context, history):
        import torch
        x = torch.tensor(context.get("X", np.zeros(24)), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            return int(torch.argmax(logits, dim=-1).item())

    def update(self, feedback):
        super().update(feedback)
        correct = feedback.get("correct", 0)
        utility = feedback.get("utility", 0.0)
        self._recent_correct.append(int(correct))
        self._recent_utilities.append(float(utility))
        if len(self._recent_correct) > 200:
            self._recent_correct = self._recent_correct[-100:]
            self._recent_utilities = self._recent_utilities[-100:]

    def _decay_confidence(self):
        return max(0.1, 1.0 - 0.005 * self.timestep)


class PrototypeOutcomeCapital(Capital):
    def __init__(self, prototype_table, capital_id="PrototypeOutcome"):
        super().__init__(capital_id, "PrototypeOutcome")
        self.prototype_table = prototype_table
        self._recent_correct = []
        self._recent_utilities = []
        self._recent_ood = []
        self._last_report = None

    def generate_report(self, context, history):
        r = self._base_report()
        r.confidence = self._decay_confidence()
        if len(self._recent_correct) >= 5:
            r.recent_prediction_error = 1.0 - np.mean(self._recent_correct[-20:])
            r.recent_regret = r.recent_prediction_error
        r.realized_utility = float(np.mean(self._recent_utilities[-20:])) if self._recent_utilities else 0.0
        r.capital_local_ood_score = float(np.mean(self._recent_ood[-20:])) if self._recent_ood else 0.0
        r.inference_cost = self.prototype_table.inference_ops
        r.storage_cost = self.prototype_table.stored_bytes
        r.update_cost = 100.0
        r.nearest_support_distance = r.capital_local_ood_score
        self._last_report = r
        return r

    def act(self, context, history):
        X = np.array(context.get("X", np.zeros(24))).reshape(1, -1)
        utility_fn = context.get("utility_fn", lambda Y: np.argmax(Y, axis=1))
        Y_pred = self.prototype_table.predict(X)
        return int(utility_fn(Y_pred)[0])

    def update(self, feedback):
        super().update(feedback)
        correct = feedback.get("correct", 0)
        utility = feedback.get("utility", 0.0)
        ood_score = feedback.get("ood_distance", 0.0)
        self._recent_correct.append(int(correct))
        self._recent_utilities.append(float(utility))
        self._recent_ood.append(float(ood_score))
        if len(self._recent_correct) > 200:
            self._recent_correct = self._recent_correct[-100:]
            self._recent_utilities = self._recent_utilities[-100:]
            self._recent_ood = self._recent_ood[-100:]

    def _decay_confidence(self):
        return max(0.1, 1.0 - 0.003 * self.timestep)


class AEPCapital(Capital):
    def __init__(self, model, capital_id="AEP"):
        super().__init__(capital_id, "AEP")
        self.model = model
        self.model.eval()
        self._recent_correct = []
        self._recent_utilities = []
        self._recent_ood = []
        self._last_report = None

    def generate_report(self, context, history):
        r = self._base_report()
        r.confidence = self._decay_confidence()
        if len(self._recent_correct) >= 5:
            r.recent_prediction_error = 1.0 - np.mean(self._recent_correct[-20:])
            r.recent_regret = r.recent_prediction_error
        r.realized_utility = float(np.mean(self._recent_utilities[-20:])) if self._recent_utilities else 0.0
        r.capital_local_ood_score = float(np.mean(self._recent_ood[-20:])) if self._recent_ood else 0.0
        r.inference_cost = 1.0
        r.storage_cost = sum(p.numel() for p in self.model.parameters()) * 4
        r.update_cost = 20000.0
        r.nearest_support_distance = r.capital_local_ood_score
        r.expected_probe_value = max(0, 0.05 - r.recent_prediction_error)
        self._last_report = r
        return r

    def act(self, context, history):
        import torch
        x = torch.tensor(context.get("X", np.zeros(24)), dtype=torch.float32).unsqueeze(0)
        utility_fn = context.get("utility_fn", lambda Y: np.argmax(Y, axis=1))
        with torch.no_grad():
            Y_pred = self.model.predict_all_actions(x).cpu().numpy()
            return int(utility_fn(Y_pred)[0])

    def update(self, feedback):
        super().update(feedback)
        correct = feedback.get("correct", 0)
        utility = feedback.get("utility", 0.0)
        ood_score = feedback.get("ood_distance", 0.0)
        self._recent_correct.append(int(correct))
        self._recent_utilities.append(float(utility))
        self._recent_ood.append(float(ood_score))
        if len(self._recent_correct) > 200:
            self._recent_correct = self._recent_correct[-100:]
            self._recent_utilities = self._recent_utilities[-100:]
            self._recent_ood = self._recent_ood[-100:]

    def _decay_confidence(self):
        return max(0.1, 1.0 - 0.005 * self.timestep)


class ResidualCapital(Capital):
    def __init__(self, model, capital_id="Residual"):
        super().__init__(capital_id, "ResidualCapital")
        self.model = model
        self.model.eval()
        self._recent_correct = []
        self._recent_utilities = []
        self._recent_ood = []
        self._last_report = None

    def generate_report(self, context, history):
        r = self._base_report()
        r.confidence = self._decay_confidence()
        if len(self._recent_correct) >= 5:
            r.recent_prediction_error = 1.0 - np.mean(self._recent_correct[-20:])
            r.recent_regret = r.recent_prediction_error
            r.calibration_error = np.std(self._recent_correct[-20:]) if len(self._recent_correct) >= 20 else 0.0
        r.realized_utility = float(np.mean(self._recent_utilities[-20:])) if self._recent_utilities else 0.0
        r.capital_local_ood_score = float(np.mean(self._recent_ood[-20:])) if self._recent_ood else 0.0
        r.inference_cost = 1.0
        r.storage_cost = sum(p.numel() for p in self.model.parameters()) * 4
        r.update_cost = 15000.0
        r.expected_probe_value = max(0, 0.03 - r.recent_prediction_error)
        r.transfer_success_rate = float(np.mean(self._recent_correct[-10:])) if len(self._recent_correct) >= 10 else 1.0
        r.recent_transfer_regret = r.recent_regret
        r.capital_age = self.timestep
        r.depreciation_score = max(0.0, 0.001 * r.capital_age)
        r.impairment_flag = 0.0
        self._last_report = r
        return r

    def act(self, context, history):
        import torch
        x = torch.tensor(context.get("X", np.zeros(24)), dtype=torch.float32).unsqueeze(0)
        utility_fn = context.get("utility_fn", lambda Y: np.argmax(Y, axis=1))
        with torch.no_grad():
            Y_pred = self.model.predict_all_actions(x).cpu().numpy()
            return int(utility_fn(Y_pred)[0])

    def update(self, feedback):
        super().update(feedback)
        correct = feedback.get("correct", 0)
        utility = feedback.get("utility", 0.0)
        ood_score = feedback.get("ood_distance", 0.0)
        self._recent_correct.append(int(correct))
        self._recent_utilities.append(float(utility))
        self._recent_ood.append(float(ood_score))
        if len(self._recent_correct) > 200:
            self._recent_correct = self._recent_correct[-100:]
            self._recent_utilities = self._recent_utilities[-100:]
            self._recent_ood = self._recent_ood[-100:]

    def _decay_confidence(self):
        return max(0.1, 1.0 - 0.004 * self.timestep)


class SafeFallbackCapital(Capital):
    def __init__(self, capital_id="SafeFallback"):
        super().__init__(capital_id, "SafeFallbackCapital")
        self._action_counts = [0, 0, 0]
        self._correct_counts = [0, 0, 0]
        self._last_report = None

    def generate_report(self, context, history):
        r = self._base_report()
        r.confidence = 0.5
        r.recent_prediction_error = 0.5
        r.recent_regret = 0.5
        r.realized_utility = 0.5
        total = sum(self._action_counts) or 1
        correct_total = sum(self._correct_counts)
        r.realization_rate = correct_total / total if total > 0 else 0.5
        r.inference_cost = 0.01
        r.update_cost = 1.0
        r.storage_cost = 12
        r.capital_local_ood_score = 0.0
        r.goal_shift_score = 0.0
        r.transfer_success_rate = 0.5
        r.capital_age = self.timestep
        r.depreciation_score = 0.0
        r.bad_debt_score = 0.0
        r.impairment_flag = 0.0
        r.recommended_action = self._most_successful_action()
        r.predicted_utility = 0.5
        self._last_report = r
        return r

    def _most_successful_action(self):
        ratios = [
            self._correct_counts[i] / max(1, self._action_counts[i])
            for i in range(3)
        ]
        return int(np.argmax(ratios))

    def act(self, context, history):
        if np.random.rand() < 0.3:
            return int(np.random.randint(0, 3))
        return self._most_successful_action()

    def update(self, feedback):
        super().update(feedback)
        action = feedback.get("action", feedback.get("last_action", 0))
        correct = feedback.get("correct", 0)
        if 0 <= action < 3:
            self._action_counts[action] += 1
            if correct:
                self._correct_counts[action] += 1


class GoalInferenceCapital(Capital):
    def __init__(self, grid_size=7, capital_id="GoalInference"):
        super().__init__(capital_id, "GoalInferenceCapital")
        self.grid_size = grid_size
        self.goal_belief = np.ones((grid_size, grid_size), dtype=np.float32) / (grid_size * grid_size)
        self.last_obs = None; self.goal_observed = False; self.known_goal = None
        self._last_dist = None
        self._recent_reward = []; self._recent_goal_correct = []; self._recent_regret = []

    def generate_report(self, context, history):
        r = self._base_report()
        r.confidence = self._goal_confidence()
        if self._recent_goal_correct:
            r.recent_prediction_error = 1.0 - np.mean(self._recent_goal_correct[-20:])
            r.recent_regret = np.mean(self._recent_regret[-20:]) if self._recent_regret else 0.5
        r.realized_utility = float(np.mean(self._recent_reward[-20:])) if self._recent_reward else 0.0
        r.inference_cost = self.grid_size * self.grid_size * 2
        r.storage_cost = self.grid_size * self.grid_size * 4
        r.update_cost = self.grid_size * self.grid_size
        r.goal_shift_score = 1.0 - r.confidence
        r.expected_probe_value = 0.5 * (1.0 - r.confidence)
        r.capital_local_ood_score = 0.0 if self.goal_observed else 0.5 * (1.0 - r.confidence)
        r.transfer_success_rate = float(np.mean(self._recent_goal_correct[-10:])) if len(self._recent_goal_correct) >= 10 else 0.5
        r.capital_age = self.timestep
        r.depreciation_score = max(0.0, 0.001 * self.timestep)
        r.impairment_flag = 0.0
        return r

    def _goal_confidence(self):
        if self.known_goal is not None: return 0.95
        if self.goal_observed: return max(0.1, 0.5 - 0.05 * self.timestep)
        return max(0.05, np.max(self.goal_belief))

    def _reset_belief(self):
        self.goal_belief = np.ones((self.grid_size, self.grid_size), dtype=np.float32) / (self.grid_size * self.grid_size)
        self.known_goal = None
        self.goal_observed = False
        self._last_dist = None

    def _update_belief_from_obs(self, obs_flat):
        n = self.grid_size * self.grid_size
        obs_2d = obs_flat[:n].reshape(self.grid_size, self.grid_size)
        goal_pos = np.argwhere(obs_2d == 2.0)
        if len(goal_pos) > 0:
            gy, gx = goal_pos[0]
            new_belief = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            new_belief[gy, gx] = 1.0; self.goal_belief = new_belief
            self.known_goal = (gy, gx); self.goal_observed = True; return True

        dist_hint = None
        if len(obs_flat) > n:
            dist_hint = float(obs_flat[n]) * (2 * self.grid_size)

        agent_positions = np.argwhere(obs_2d == 1.0)
        if len(agent_positions) > 0:
            ay, ax = int(agent_positions[0][0]), int(agent_positions[0][1])

            if hasattr(self, '_last_dist') and self._last_dist is not None and dist_hint is not None:
                dist_change = self._last_dist - dist_hint
                for gy in range(self.grid_size):
                    for gx in range(self.grid_size):
                        md = abs(ay - gy) + abs(ax - gx)
                        likelihood = np.exp(-0.5 * ((md - dist_hint) ** 2) / max(1.0, dist_hint + 0.5))
                        self.goal_belief[gy, gx] *= max(0.01, likelihood)
                total = self.goal_belief.sum()
                if total > 1e-8:
                    self.goal_belief /= total

            if dist_hint is not None:
                self._last_dist = dist_hint

        return False

    def act(self, context, history):
        obs = context.get("obs", np.zeros(25, dtype=np.float32))
        if context.get("_reset", False):
            self._reset_belief()
        self._update_belief_from_obs(obs); self.last_obs = obs

        n = self.grid_size * self.grid_size
        obs_2d = obs[:n].reshape(self.grid_size, self.grid_size)
        agent_positions = np.argwhere(obs_2d >= 0.9)
        if len(agent_positions) > 0:
            ay, ax = int(agent_positions[0][0]), int(agent_positions[0][1])
        else:
            ay, ax = self.grid_size // 2, self.grid_size // 2

        if self.known_goal is not None:
            gy, gx = self.known_goal
            dy = gy - ay; dx = gx - ax
            if abs(dy) > abs(dx): return 1 if dy > 0 else 0
            else: return 3 if dx > 0 else 2
        best_pos = np.unravel_index(np.argmax(self.goal_belief), (self.grid_size, self.grid_size))
        dy = best_pos[0] - ay; dx = best_pos[1] - ax
        if abs(dy) > abs(dx): return 1 if dy > 0 else 0
        else: return 3 if dx > 0 else 2

    def update(self, feedback):
        super().update(feedback)
        reward = feedback.get("reward", 0.0); correct = feedback.get("goal_reached", 0)
        self._recent_reward.append(float(reward))
        self._recent_goal_correct.append(int(correct))
        self._recent_regret.append(1.0 - float(correct))
        if feedback.get("at_goal", False): self.goal_observed = True
        if len(self._recent_reward) > 200:
            self._recent_reward = self._recent_reward[-100:]
            self._recent_goal_correct = self._recent_goal_correct[-100:]


ALLOWED_REPORT_FIELDS = {
    "recommended_action", "predicted_utility",
    "recent_prediction_error", "recent_regret", "confidence",
    "calibration_error", "realized_utility", "realization_rate",
    "capital_local_ood_score", "nearest_support_distance",
    "inference_cost", "update_cost", "storage_cost",
    "probe_cost", "goal_shift_score",
    "transfer_success_rate", "recent_transfer_regret",
    "expected_probe_value", "uncertainty_reduction_if_probe",
    "capital_age", "depreciation_score", "bad_debt_score",
    "impairment_flag",
}

FORBIDDEN_REPORT_FIELDS = {
    "env_name", "env_id", "state_dim", "utility_type",
    "mode_type", "friction", "delay_strength",
    "action_effect_rule_name", "hand_written_regime_label",
    "manually_computed_global_coverage", "task_id", "task_type",
}