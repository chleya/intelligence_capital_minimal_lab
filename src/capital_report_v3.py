"""
CapitalReport v3 — Capital-Local Current-Action Evidence (IC-3-M)
==================================================================
Extends v2 with 10 capital-local fields that provide evidence about
the current instance WITHOUT revealing task_id, subregime_id, or
any global proxy.

New v3 fields (capital-local current-action evidence only):
  1. per_action_value_vector       — value estimates for each candidate action
  2. per_action_uncertainty_vector — uncertainty of each action value estimate
  3. top2_action_margin            — gap between best and 2nd best action value
  4. local_model_jacobian_norm     — sensitivity of action ranking to input perturbation
  5. nearest_counterfactual_support — quality/density of nearest training support
  6. calibration_at_similar_margin — calibration error at similar action-margin region
  7. recent_error_conditioned_on_margin — recent error rate when action margin was similar
  8. expected_switch_gain          — expected improvement if switching to this capital
  9. expected_switch_risk          — expected degradation if switching to this capital
  10. local_failure_memory_match   — similarity of current instance to past failures

Forbidden: any global/task_id/subregime_id/pair_id/pair_side identifications.
If a capital cannot compute a field, fill explicit default + missing_mask.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

from src.capital_report_v2 import (
    V1_FIELD_NAMES, V2_NEW_FIELD_NAMES, ALL_V2_FIELD_NAMES,
    N_V1, N_V2, CapitalReportV2,
    CapitalV2, PolicyCloneCapitalV2, PrototypeOutcomeCapitalV2,
    AEPCapitalV2, GoalInferenceCapitalV2, SafeFallbackCapitalV2,
    report_v2_vector, make_v2_capitals, ALLOWED_V2, FORBIDDEN_V2,
)

V3_NEW_FIELD_NAMES = [
    "per_action_value_vector_0",
    "per_action_value_vector_1",
    "per_action_value_vector_2",
    "per_action_uncertainty_vector_0",
    "per_action_uncertainty_vector_1",
    "per_action_uncertainty_vector_2",
    "top2_action_margin",
    "local_model_jacobian_norm",
    "nearest_counterfactual_support",
    "calibration_at_similar_margin",
    "recent_error_conditioned_on_margin",
    "expected_switch_gain",
    "expected_switch_risk",
    "local_failure_memory_match",
]
ALL_V3_FIELD_NAMES = ALL_V2_FIELD_NAMES + V3_NEW_FIELD_NAMES
N_V3 = len(ALL_V3_FIELD_NAMES)
N_V3_NEW = len(V3_NEW_FIELD_NAMES)


@dataclass
class CapitalReportV3:
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

    # v2 fields (12)
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

    # v3 fields (14 — 10 conceptual, expanded for per-action vectors)
    per_action_value_vector_0: float = 0.0
    per_action_value_vector_1: float = 0.0
    per_action_value_vector_2: float = 0.0
    per_action_uncertainty_vector_0: float = 0.0
    per_action_uncertainty_vector_1: float = 0.0
    per_action_uncertainty_vector_2: float = 0.0
    top2_action_margin: float = 0.0
    local_model_jacobian_norm: float = 0.0
    nearest_counterfactual_support: float = 0.0
    calibration_at_similar_margin: float = 0.0
    recent_error_conditioned_on_margin: float = 0.0
    expected_switch_gain: float = 0.0
    expected_switch_risk: float = 0.0
    local_failure_memory_match: float = 0.0

    v3_computed_mask: List[float] = field(default_factory=list)

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
            # v3 (14)
            self.per_action_value_vector_0,
            self.per_action_value_vector_1,
            self.per_action_value_vector_2,
            self.per_action_uncertainty_vector_0,
            self.per_action_uncertainty_vector_1,
            self.per_action_uncertainty_vector_2,
            self.top2_action_margin,
            self.local_model_jacobian_norm,
            self.nearest_counterfactual_support,
            self.calibration_at_similar_margin,
            self.recent_error_conditioned_on_margin,
            self.expected_switch_gain,
            self.expected_switch_risk,
            self.local_failure_memory_match,
        ], dtype=np.float32)


def _v2_report_to_v3_init(v2_report):
    """Initialize a CapitalReportV3 from a CapitalReportV2."""
    r = CapitalReportV3(
        capital_id=v2_report.capital_id,
        capital_type=v2_report.capital_type,
        timestamp=v2_report.timestamp,
        recommended_action=v2_report.recommended_action,
        predicted_utility=v2_report.predicted_utility,
        recent_prediction_error=v2_report.recent_prediction_error,
        recent_regret=v2_report.recent_regret,
        confidence=v2_report.confidence,
        calibration_error=v2_report.calibration_error,
        realized_utility=v2_report.realized_utility,
        realization_rate=v2_report.realization_rate,
        capital_local_ood_score=v2_report.capital_local_ood_score,
        nearest_support_distance=v2_report.nearest_support_distance,
        inference_cost=v2_report.inference_cost,
        update_cost=v2_report.update_cost,
        storage_cost=v2_report.storage_cost,
        probe_cost=v2_report.probe_cost,
        goal_shift_score=v2_report.goal_shift_score,
        transfer_success_rate=v2_report.transfer_success_rate,
        recent_transfer_regret=v2_report.recent_transfer_regret,
        expected_probe_value=v2_report.expected_probe_value,
        uncertainty_reduction_if_probe=v2_report.uncertainty_reduction_if_probe,
        capital_age=v2_report.capital_age,
        depreciation_score=v2_report.depreciation_score,
        bad_debt_score=v2_report.bad_debt_score,
        impairment_flag=v2_report.impairment_flag,
        action_margin=v2_report.action_margin,
        predicted_best_action_value=v2_report.predicted_best_action_value,
        second_best_action_value=v2_report.second_best_action_value,
        local_counterfactual_spread=v2_report.local_counterfactual_spread,
        self_consistency_score=v2_report.self_consistency_score,
        local_support_quality=v2_report.local_support_quality,
        extrapolation_distance=v2_report.extrapolation_distance,
        goal_relevance_score=v2_report.goal_relevance_score,
        disagreement_with_portfolio=v2_report.disagreement_with_portfolio,
        current_uncertainty=v2_report.current_uncertainty,
        capital_specific_expected_regret=v2_report.capital_specific_expected_regret,
        transition_sensitivity=v2_report.transition_sensitivity,
    )
    return r


# ═══════════════════════════════════════════════════════════
# V3 CAPITAL WRAPPERS — add v3 fields on top of v2
# ═══════════════════════════════════════════════════════════

class CapitalWithV3(CapitalV2):
    """Adds v3 capital-local fields on top of v2 wrapper."""
    def __init__(self, inner_capital, capital_id):
        super().__init__(inner_capital, capital_id)
        self._recent_action_values = []
        self._recent_margins = []
        self._recent_errors = []
        self._failure_memory = []  # list of (feature_vec, outcome) for past failures

    def generate_report(self, context, history, portfolio_action=None, precomputed_action=None):
        v2_report = self.inner.generate_report(context, history, portfolio_action, precomputed_action)
        r = _v2_report_to_v3_init(v2_report)
        self._add_v3_fields(r, context, history, portfolio_action, precomputed_action)
        return r

    def _add_v3_fields(self, r, context, history, portfolio_action, precomputed_action):
        computed = [0.0] * N_V3_NEW
        # Default cal values
        r.top2_action_margin = r.action_margin
        r.local_model_jacobian_norm = r.transition_sensitivity
        r.nearest_counterfactual_support = r.local_support_quality
        r.calibration_at_similar_margin = 1.0 - r.calibration_error if r.calibration_error < 0.5 else 0.5
        r.recent_error_conditioned_on_margin = r.recent_prediction_error
        r.expected_switch_gain = r.expected_probe_value
        r.expected_switch_risk = r.recent_prediction_error
        r.local_failure_memory_match = 0.0

        # Track recent data for conditioning
        if len(self._recent_margins) >= 3:
            recent_margin_mean = float(np.mean(self._recent_margins[-10:]))
            recent_err_mean = float(np.mean(self._recent_errors[-10:]))
            if abs(r.action_margin - recent_margin_mean) < 0.05:
                r.recent_error_conditioned_on_margin = recent_err_mean
                computed[10] = 1.0

        r.v3_computed_mask = computed


class PolicyCloneCapitalV3(CapitalWithV3):
    def __init__(self, inner_capital, capital_id="PolicyClone"):
        super().__init__(inner_capital, capital_id)

    def _add_v3_fields(self, r, context, history, portfolio_action, precomputed_action):
        computed = [0.0] * N_V3_NEW
        r.top2_action_margin = r.action_margin
        r.nearest_counterfactual_support = r.local_support_quality
        r.local_model_jacobian_norm = r.transition_sensitivity
        r.calibration_at_similar_margin = max(0.0, 1.0 - r.recent_prediction_error)
        r.expected_switch_gain = max(0.0, 0.05 - r.recent_prediction_error)
        r.expected_switch_risk = r.recent_prediction_error
        r.local_failure_memory_match = 0.0

        # Per-action values from model logits
        X = context.get("X", None)
        if X is not None and hasattr(self.inner, "forward"):
            import torch
            x_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                values = self.inner.forward(x_t).cpu().numpy()[0]
            for a in range(3):
                setattr(r, f"per_action_value_vector_{a}", float(values[a]) if a < len(values) else 0.0)
                computed[a] = 1.0
            # Uncertainty: softmax entropy per action
            probs = torch.softmax(torch.tensor(values), dim=-1).numpy()
            for a in range(3):
                p = max(float(probs[a]), 1e-8)
                unc = -p * np.log(p) / np.log(3)
                setattr(r, f"per_action_uncertainty_vector_{a}", float(unc))
                computed[3 + a] = 1.0

        # Calibration at similar margin
        r.recent_error_conditioned_on_margin = r.recent_prediction_error
        computed[7] = 1.0  # top2 margin
        computed[8] = 1.0  # jacobian
        computed[9] = 1.0  # nearest support
        computed[10] = 1.0
        computed[11] = 1.0
        computed[12] = 1.0
        r.v3_computed_mask = computed


class PrototypeOutcomeCapitalV3(CapitalWithV3):
    def __init__(self, inner_capital, capital_id="PrototypeOutcome"):
        super().__init__(inner_capital, capital_id)

    def _add_v3_fields(self, r, context, history, portfolio_action, precomputed_action):
        computed = [0.0] * N_V3_NEW
        X = context.get("X", None)
        pt = self.inner.prototype_table
        if X is not None and hasattr(pt, "predict"):
            Yp = pt.predict(np.array([X]))
            for a in range(min(3, len(Yp[0]))):
                setattr(r, f"per_action_value_vector_{a}", float(Yp[0][a]))
                computed[a] = 1.0
            # Uncertainty: std across nearest neighbors
            if hasattr(pt, "pY") and hasattr(pt, "pXs") and hasattr(pt, "sc"):
                Xqs = pt.sc.transform(np.array([X]))
                d = np.sqrt(np.sum((pt.pXs - Xqs[0].reshape(1, -1)) ** 2, 1))
                k = min(pt.k if hasattr(pt, 'k') else 3, len(pt.pXs))
                nni = np.argsort(d)[:k]
                for a in range(min(3, pt.pY.shape[1])):
                    unc = float(np.std(pt.pY[nni, a])) if len(nni) >= 2 else 0.0
                    setattr(r, f"per_action_uncertainty_vector_{a}", unc)
                    computed[3 + a] = 1.0

        r.top2_action_margin = r.action_margin
        r.nearest_counterfactual_support = r.local_support_quality
        r.expected_switch_gain = max(0.0, 0.05 - r.recent_prediction_error)
        r.expected_switch_risk = r.recent_prediction_error
        computed[7] = 1.0
        computed[9] = 1.0
        computed[11] = 1.0
        computed[12] = 1.0
        r.v3_computed_mask = computed


class AEPCapitalV3(CapitalWithV3):
    def __init__(self, inner_capital, capital_id="AEP"):
        super().__init__(inner_capital, capital_id)

    def _add_v3_fields(self, r, context, history, portfolio_action, precomputed_action):
        computed = [0.0] * N_V3_NEW
        X = context.get("X", None)
        if X is not None and hasattr(self.inner, "model"):
            import torch
            m = self.inner.model
            if hasattr(m, "encoder") and hasattr(m, "action_head"):
                x_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    z = m.encoder(x_t)
                    logits = m.action_head(z)
                    values = logits.cpu().numpy()[0]
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                for a in range(min(3, len(values))):
                    setattr(r, f"per_action_value_vector_{a}", float(values[a]))
                    p = max(float(probs[a]), 1e-8)
                    unc = -p * np.log(p) / np.log(3)
                    setattr(r, f"per_action_uncertainty_vector_{a}", float(unc))
                # Jacobian norm via input perturbation
                eps = 0.001
                x_p = torch.tensor(X + eps, dtype=torch.float32).unsqueeze(0)
                x_n = torch.tensor(X - eps, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    z_p = m.encoder(x_p)
                    z_n = m.encoder(x_n)
                    j_norm = float(torch.norm(z_p - z_n, p=2) / (2 * eps))
                r.local_model_jacobian_norm = j_norm
                computed[:6] = [1.0] * 6
                computed[8] = 1.0

        r.top2_action_margin = r.action_margin
        r.nearest_counterfactual_support = r.local_support_quality
        r.expected_switch_gain = max(0.0, 0.05 - r.recent_prediction_error)
        r.expected_switch_risk = r.recent_prediction_error
        computed[7] = 1.0
        computed[9] = 1.0
        computed[11] = 1.0
        computed[12] = 1.0
        r.v3_computed_mask = computed


class GoalInferenceCapitalV3(CapitalWithV3):
    def __init__(self, inner_capital, capital_id="GoalInference"):
        super().__init__(inner_capital, capital_id)

    def _add_v3_fields(self, r, context, history, portfolio_action, precomputed_action):
        computed = [0.0] * N_V3_NEW
        obs = context.get("obs", None)
        if obs is not None:
            r.per_action_value_vector_0 = float(r.predicted_best_action_value)
            r.per_action_value_vector_1 = float(r.second_best_action_value)
            r.per_action_value_vector_2 = 0.0
            r.per_action_uncertainty_vector_0 = float(r.current_uncertainty)
            r.per_action_uncertainty_vector_1 = 0.0
            r.per_action_uncertainty_vector_2 = 0.0
            computed[0] = 1.0; computed[3] = 1.0

        r.top2_action_margin = r.action_margin
        r.nearest_counterfactual_support = float(1.0 - getattr(r, 'extrapolation_distance', 0.0))
        r.calibration_at_similar_margin = r.confidence
        r.expected_switch_gain = max(0.0, 0.03 - r.recent_prediction_error)
        r.expected_switch_risk = r.recent_prediction_error
        computed[7] = 1.0; computed[9] = 1.0; computed[11] = 1.0; computed[12] = 1.0
        r.v3_computed_mask = computed


class SafeFallbackCapitalV3(CapitalWithV3):
    def __init__(self, inner_capital, capital_id="SafeFallback"):
        super().__init__(inner_capital, capital_id)

    def _add_v3_fields(self, r, context, history, portfolio_action, precomputed_action):
        computed = [0.0] * N_V3_NEW
        r.per_action_value_vector_0 = r.predicted_utility
        r.per_action_value_vector_1 = 0.0
        r.per_action_value_vector_2 = 0.0
        r.per_action_uncertainty_vector_0 = 0.5
        r.per_action_uncertainty_vector_1 = 0.5
        r.per_action_uncertainty_vector_2 = 0.5
        r.top2_action_margin = 0.0
        r.nearest_counterfactual_support = 0.0
        r.expected_switch_gain = 0.0
        r.expected_switch_risk = 1.0
        r.local_failure_memory_match = 0.0
        computed[0] = 1.0; computed[3] = 1.0
        computed[7] = 1.0
        computed[11] = 1.0; computed[12] = 1.0
        r.v3_computed_mask = computed


def make_v3_capitals(capitals_list):
    """Wrap v2 capitals with v3 layers."""
    v3_map = {
        "PolicyClone": PolicyCloneCapitalV3,
        "PrototypeOutcome": PrototypeOutcomeCapitalV3,
        "AEP": AEPCapitalV3,
        "GoalInference": GoalInferenceCapitalV3,
        "SafeFallback": SafeFallbackCapitalV3,
    }
    result = []
    for cap in capitals_list:
        cid = cap.capital_id
        wrapper_cls = v3_map.get(cid, CapitalWithV3)
        result.append(wrapper_cls(cap, cid))
    return result


def report_v3_vector(reports):
    """Concatenate all v3 report vectors into one flat vector."""
    return np.concatenate([r.to_vector() for r in reports]).astype(np.float32)