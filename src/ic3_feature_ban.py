"""
IC-3-0 Feature Ban Gate
========================
Audits the allocator input schema and enforces that only
CapitalReport-derived fields are allowed as input features.
"""
from dataclasses import dataclass, field
from typing import List, Set, Dict


FORBIDDEN_FEATURES: Set[str] = {
    "env_name", "env_id", "state_dim", "utility_type",
    "mode_type", "friction", "delay_strength",
    "action_effect_rule_name", "hand_written_regime_label",
    "manually_computed_global_coverage",
}

ALLOWED_FEATURES: Set[str] = {
    "recent_prediction_error", "recent_regret", "confidence",
    "calibration_error", "realized_utility", "realization_rate",
    "capital_local_ood_score", "nearest_support_distance",
    "inference_cost", "update_cost", "storage_cost",
    "goal_shift_score", "expected_probe_value",
    "depreciation_score", "bad_debt_score",
}


@dataclass
class FeatureBanReport:
    passed: bool = True
    violations: List[str] = field(default_factory=list)
    allowed_count: int = 0
    forbidden_count: int = 0
    warnings: List[str] = field(default_factory=list)
    clean_input_schema: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "passed": self.passed,
            "violations": self.violations,
            "allowed_count": self.allowed_count,
            "forbidden_count": self.forbidden_count,
            "warnings": self.warnings,
            "clean_input_schema": self.clean_input_schema,
        }


def audit_allocator_inputs(input_schema: List[str]) -> FeatureBanReport:
    report = FeatureBanReport()
    report.clean_input_schema = [f for f in input_schema if f not in FORBIDDEN_FEATURES]

    for feat in input_schema:
        if feat in FORBIDDEN_FEATURES:
            report.violations.append(feat)
            report.forbidden_count += 1
        elif feat in ALLOWED_FEATURES:
            report.allowed_count += 1
        else:
            report.warnings.append(f"Unknown feature '{feat}' — not in allowed list")

    if report.forbidden_count > 0:
        report.passed = False

    return report


def build_clean_input_schema(n_capitals: int, prefix: str = "cap") -> List[str]:
    schema = []
    for i in range(n_capitals):
        for field in sorted(ALLOWED_FEATURES):
            schema.append(f"{prefix}_{i}_{field}")
    schema.extend(["step_index", "total_capitals"])
    return schema