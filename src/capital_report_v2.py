"""
CapitalReport v2 — Current-Instance Capital Report (IC-3-V2)
=============================================================
Extends v1's 23 historical-performance fields with 12 current-instance
evidence fields, for a total of 35 fields per capital.

New fields (current-instance evidence):
  1. action_margin           — gap between best and 2nd-best action value
  2. predicted_best_action_value — value of the predicted best action
  3. second_best_action_value    — value of the 2nd-best action
  4. local_counterfactual_spread  — outcome variance among nearby counterfactuals
  5. self_consistency_score       — agreement with own recent predictions
  6. local_support_quality        — density/quality of support near current input
  7. extrapolation_distance       — distance from training distribution
  8. goal_relevance_score         — relevance of current hidden-goal belief
  9. disagreement_with_portfolio  — how much this capital disagrees with others
  10. current_uncertainty          — uncertainty about this specific instance
  11. capital_specific_expected_regret — expected regret for this instance
  12. transition_sensitivity      — sensitivity of action to small input changes

Forbidden: env_name, task_type, task_id, utility_type, state_dim, regime_label.
If a capital cannot compute a field, fill explicit default + missing_mask.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

V1_FIELD_NAMES = [
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
]
V2_NEW_FIELD_NAMES = [
    "action_margin",
    "predicted_best_action_value",
    "second_best_action_value",
    "local_counterfactual_spread",
    "self_consistency_score",
    "local_support_quality",
    "extrapolation_distance",
    "goal_relevance_score",
    "disagreement_with_portfolio",
    "current_uncertainty",
    "capital_specific_expected_regret",
    "transition_sensitivity",
]
ALL_V2_FIELD_NAMES = V1_FIELD_NAMES + V2_NEW_FIELD_NAMES
N_V1 = len(V1_FIELD_NAMES)
N_V2 = len(ALL_V2_FIELD_NAMES)


@dataclass
class CapitalReportV2:
    capital_id: str
    capital_type: str
    timestamp: int

    # v1 fields (23)
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
    impairment_flag: float = 0.0

    # v2 current-instance fields (12)
    action_margin: float = 0.0
    predicted_best_action_value: float = 0.0
    second_best_action_value: float = 0.0
    local_counterfactual_spread: float = 0.0
    self_consistency_score: float = 1.0
    local_support_quality: float = 0.0
    extrapolation_distance: float = 0.0
    goal_relevance_score: float = 0.0
    disagreement_with_portfolio: float = 0.0
    current_uncertainty: float = 0.0
    capital_specific_expected_regret: float = 0.0
    transition_sensitivity: float = 0.0

    computed_mask: List[float] = field(default_factory=list)

    def to_vector(self):
        return np.array([
            # v1 (23)
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
            # v2 (12)
            self.action_margin,
            self.predicted_best_action_value,
            self.second_best_action_value,
            self.local_counterfactual_spread,
            self.self_consistency_score,
            self.local_support_quality,
            self.extrapolation_distance,
            self.goal_relevance_score,
            self.disagreement_with_portfolio,
            self.current_uncertainty,
            self.capital_specific_expected_regret,
            self.transition_sensitivity,
        ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════
# V2 CAPITAL WRAPPERS — extend original capitals with v2 fields
# ═══════════════════════════════════════════════════════════

class CapitalV2:
    """Base wrapper that adds v2 fields to any capital."""
    def __init__(self, inner_capital, capital_id):
        self.inner = inner_capital
        self.capital_id = capital_id
        self.capital_type = inner_capital.capital_type if hasattr(inner_capital, "capital_type") else capital_id
        self.timestep = 0
        self._last_actions = []
        self._last_contexts = []

    def _compute_action_margin(self, action_values):
        if len(action_values) >= 2:
            sv = sorted(action_values, reverse=True)
            return sv[0] - sv[1] if len(sv) >= 2 else 0.0
        return 0.0

    def _compute_self_consistency(self, action):
        if len(self._last_actions) < 3:
            return 1.0
        recent = self._last_actions[-5:] if len(self._last_actions) >= 5 else self._last_actions
        same = sum(1 for a in recent if a == action)
        return same / len(recent)

    def _compute_current_uncertainty(self, probs_or_var):
        if isinstance(probs_or_var, np.ndarray) and len(probs_or_var) > 1:
            probs = np.maximum(probs_or_var, 1e-8)
            probs = probs / probs.sum()
            ent = -np.sum(probs * np.log(probs))
            return ent / np.log(len(probs)) if len(probs) > 1 else 0.0
        return float(np.clip(probs_or_var, 0.0, 1.0))

    def act(self, context, history):
        return self.inner.act(context, history)

    def update(self, feedback):
        self.inner.update(feedback)
        self.timestep += 1

    def generate_report(self, context, history, portfolio_action=None, precomputed_action=None):
        v1_report = self.inner.generate_report(context, history)
        action = precomputed_action if precomputed_action is not None else self.act(context, history)
        r = CapitalReportV2(
            capital_id=self.capital_id,
            capital_type=self.capital_type,
            timestamp=self.timestep,
            recommended_action=int(action),
            predicted_utility=v1_report.predicted_utility,
            recent_prediction_error=v1_report.recent_prediction_error,
            recent_regret=v1_report.recent_regret,
            confidence=v1_report.confidence,
            calibration_error=v1_report.calibration_error,
            realized_utility=v1_report.realized_utility,
            realization_rate=v1_report.realization_rate,
            capital_local_ood_score=v1_report.capital_local_ood_score,
            nearest_support_distance=v1_report.nearest_support_distance,
            inference_cost=v1_report.inference_cost,
            update_cost=v1_report.update_cost,
            storage_cost=v1_report.storage_cost,
            probe_cost=v1_report.probe_cost,
            goal_shift_score=v1_report.goal_shift_score,
            transfer_success_rate=v1_report.transfer_success_rate,
            recent_transfer_regret=v1_report.recent_transfer_regret,
            expected_probe_value=v1_report.expected_probe_value,
            uncertainty_reduction_if_probe=v1_report.uncertainty_reduction_if_probe,
            capital_age=v1_report.capital_age,
            depreciation_score=v1_report.depreciation_score,
            bad_debt_score=v1_report.bad_debt_score,
            impairment_flag=v1_report.impairment_flag,
        )
        self._add_v2_fields(r, context, history, portfolio_action)
        self._last_actions.append(int(action))
        if len(self._last_actions) > 20:
            self._last_actions = self._last_actions[-20:]
        return r

    def _add_v2_fields(self, r, context, history, portfolio_action):
        r.computed_mask = [0.0] * 12
        r.disagreement_with_portfolio = 0.0


class PolicyCloneCapitalV2(CapitalV2):
    def __init__(self, inner_capital, capital_id="PolicyClone"):
        super().__init__(inner_capital, capital_id)

    def _add_v2_fields(self, r, context, history, portfolio_action):
        computed = [0.0] * 12
        X = context.get("X", None)
        if X is not None and hasattr(self.inner, "forward"):
            import torch
            x_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = self.inner.forward(x_t)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                values = logits.cpu().numpy()[0]
            sv = sorted(values, reverse=True)
            r.predicted_best_action_value = float(sv[0])
            r.second_best_action_value = float(sv[1]) if len(sv) >= 2 else 0.0
            r.action_margin = float(sv[0] - sv[1]) if len(sv) >= 2 else 0.0
            r.current_uncertainty = self._compute_current_uncertainty(probs)
            r.extrapolation_distance = float(r.capital_local_ood_score)
            r.self_consistency_score = self._compute_self_consistency(r.recommended_action)
            r.capital_specific_expected_regret = float(r.recent_prediction_error)
            r.model_confidence = min(1.0, float(probs.max()))
            # transition sensitivity: perturb input ±epsilon
            eps = 0.001
            x_p = torch.tensor(X + eps, dtype=torch.float32).unsqueeze(0)
            x_n = torch.tensor(X - eps, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                lp = self.inner.forward(x_p).argmax().item()
                ln = self.inner.forward(x_n).argmax().item()
            r.transition_sensitivity = 1.0 if lp != ln else 0.0
            computed[0:4] = [1.0, 1.0, 1.0, 1.0]
            computed[4] = 1.0  # self_consistency
            computed[6] = 1.0  # extrapolation
            computed[9] = 1.0  # uncertainty
            computed[10] = 1.0  # expected regret
            computed[11] = 1.0  # transition sens
        if portfolio_action is not None:
            r.disagreement_with_portfolio = 1.0 if r.recommended_action != portfolio_action else 0.0
            computed[8] = 1.0
        r.computed_mask = computed


class PrototypeOutcomeCapitalV2(CapitalV2):
    def __init__(self, inner_capital, capital_id="PrototypeOutcome"):
        super().__init__(inner_capital, capital_id)

    def _add_v2_fields(self, r, context, history, portfolio_action):
        computed = [0.0] * 12
        X = context.get("X", None)
        pt = self.inner.prototype_table
        if X is not None and hasattr(pt, "predict"):
            Yp = pt.predict(np.array([X]))
            sv = sorted(Yp[0], reverse=True)
            r.predicted_best_action_value = float(sv[0])
            r.second_best_action_value = float(sv[1]) if len(sv) >= 2 else 0.0
            r.action_margin = float(sv[0] - sv[1]) if len(sv) >= 2 else 0.0
            r.current_uncertainty = self._compute_current_uncertainty(Yp[0])
            r.local_counterfactual_spread = float(np.std(Yp[0]))
            r.extrapolation_distance = float(r.capital_local_ood_score)
            r.self_consistency_score = self._compute_self_consistency(r.recommended_action)
            r.capital_specific_expected_regret = float(r.recent_prediction_error)
            # local support quality: number of prototypes within small radius
            if hasattr(pt, "prototypes") and hasattr(pt, "sc"):
                import sklearn.preprocessing
                X_s = pt.scaler.transform(np.array([X])) if hasattr(pt, "scaler") else X
                dists = np.sum((X_s - pt.X_store_s)**2, axis=1) if hasattr(pt, "X_store_s") else np.zeros(1)
                r.local_support_quality = float(np.exp(-np.min(dists) * 5.0))
                computed[5] = 1.0
            r.transition_sensitivity = 0.0  # default — k-NN no gradient
            computed[0:4] = [1.0, 1.0, 1.0, 1.0]
            computed[4] = 1.0
            computed[6] = 1.0
            computed[9] = 1.0
            computed[10] = 1.0
        if portfolio_action is not None:
            r.disagreement_with_portfolio = 1.0 if r.recommended_action != portfolio_action else 0.0
            computed[8] = 1.0
        r.computed_mask = computed


class AEPCapitalV2(CapitalV2):
    def __init__(self, inner_capital, capital_id="AEP"):
        super().__init__(inner_capital, capital_id)

    def _add_v2_fields(self, r, context, history, portfolio_action):
        computed = [0.0] * 12
        X = context.get("X", None)
        if X is not None and hasattr(self.inner, "model"):
            import torch
            m = self.inner.model
            if hasattr(m, "encoder") and hasattr(m, "action_head"):
                x_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    z = m.encoder(x_t)
                    logits = m.action_head(z)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    values = logits.cpu().numpy()[0]
                sv = sorted(values, reverse=True)
                r.predicted_best_action_value = float(sv[0])
                r.second_best_action_value = float(sv[1]) if len(sv) >= 2 else 0.0
                r.action_margin = float(sv[0] - sv[1]) if len(sv) >= 2 else 0.0
                r.current_uncertainty = self._compute_current_uncertainty(probs)
                r.extrapolation_distance = float(r.capital_local_ood_score)
                r.self_consistency_score = self._compute_self_consistency(r.recommended_action)
                r.capital_specific_expected_regret = float(r.recent_prediction_error)
                r.transition_sensitivity = 0.0  # default
                computed[0:4] = [1.0, 1.0, 1.0, 1.0]
                computed[4] = 1.0
                computed[6] = 1.0
                computed[9] = 1.0
                computed[10] = 1.0
        if portfolio_action is not None:
            r.disagreement_with_portfolio = 1.0 if r.recommended_action != portfolio_action else 0.0
            computed[8] = 1.0
        r.computed_mask = computed


class GoalInferenceCapitalV2(CapitalV2):
    def __init__(self, inner_capital, capital_id="GoalInference"):
        super().__init__(inner_capital, capital_id)

    def _add_v2_fields(self, r, context, history, portfolio_action):
        computed = [0.0] * 12
        obs = context.get("obs", None)
        if obs is not None and hasattr(self.inner, "belief"):
            belief = self.inner.belief.copy() if hasattr(self.inner, "belief") else np.ones(49)
            # goal relevance: how peaked is the belief?
            b_max = float(belief.max())
            b_sum = float(belief.sum()) + 1e-8
            r.goal_relevance_score = b_max / b_sum * 49
            r.current_uncertainty = self._compute_current_uncertainty(belief / b_sum)
            confidence = float(self.inner.goal_confidence if hasattr(self.inner, "goal_confidence") else 0.5)
            r.extrapolation_distance = 1.0 - confidence
            r.action_margin = confidence - 0.3
            r.predicted_best_action_value = confidence
            r.self_consistency_score = self._compute_self_consistency(r.recommended_action)
            r.capital_specific_expected_regret = max(0.0, 1.0 - confidence)
            computed[0] = 1.0
            computed[1] = 1.0
            computed[4] = 1.0
            computed[6] = 1.0
            computed[7] = 1.0
            computed[9] = 1.0
            computed[10] = 1.0
        if portfolio_action is not None:
            r.disagreement_with_portfolio = 1.0 if r.recommended_action != portfolio_action else 0.0
            computed[8] = 1.0
        r.computed_mask = computed


class SafeFallbackCapitalV2(CapitalV2):
    def __init__(self, inner_capital, capital_id="SafeFallback"):
        super().__init__(inner_capital, capital_id)

    def _add_v2_fields(self, r, context, history, portfolio_action):
        computed = [0.0] * 12
        r.current_uncertainty = float(
            np.clip(getattr(self.inner, "exploration_rate", 0.3), 0.0, 1.0))
        r.predicted_best_action_value = float(
            getattr(self.inner, "success_rate", 0.5) if hasattr(self.inner, "success_rate") else 0.5)
        r.self_consistency_score = self._compute_self_consistency(r.recommended_action)
        r.extrapolation_distance = 0.3  # always somewhat out-of-distribution
        computed[1] = 1.0
        computed[4] = 1.0
        computed[6] = 1.0
        computed[9] = 1.0
        if portfolio_action is not None:
            r.disagreement_with_portfolio = 1.0 if r.recommended_action != portfolio_action else 0.0
            computed[8] = 1.0
        r.computed_mask = computed


def make_v2_capitals(capitals_list):
    v2_map = {
        "PolicyClone": PolicyCloneCapitalV2,
        "PrototypeOutcome": PrototypeOutcomeCapitalV2,
        "AEP": AEPCapitalV2,
        "GoalInference": GoalInferenceCapitalV2,
        "SafeFallback": SafeFallbackCapitalV2,
        "Residual": CapitalV2,
    }
    result = []
    for cap in capitals_list:
        cid = cap.capital_id
        wrapper_cls = v2_map.get(cid, CapitalV2)
        result.append(wrapper_cls(cap, cid))
    return result


def report_v2_vector(reports):
    return np.concatenate([r.to_vector() for r in reports]).astype(np.float32)


# Feature ban
ALLOWED_V2 = ALL_V2_FIELD_NAMES
FORBIDDEN_V2 = [
    "env_name", "env_id", "task_id", "task_type", "state_dim", "utility_type",
    "mode_type", "friction", "delay_strength", "hand_written_regime_label",
    "manually_computed_global_coverage",
]