"""
Capital Impairment Detection & Negative Transfer Protection
=============================================================
Implements three mechanisms:
  1. Capital Impairment Detection
  2. Fallback Mechanism
  3. Depreciation Schedule
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ImpairmentState:
    capital_id: str
    recent_regrets: List[float] = field(default_factory=list)
    impaired: bool = False
    impairment_steps: int = 0
    last_validated_timestamp: int = 0
    confidence: float = 1.0
    depreciation_rate: float = 0.001


class CapitalImpairmentDetector:
    def __init__(self, window_size: int = 20, impairment_threshold_steps: int = 10,
                 random_baseline_regret: float = 0.5, depreciation_rate: float = 0.002):
        self.window_size = window_size
        self.impairment_threshold_steps = impairment_threshold_steps
        self.random_baseline_regret = random_baseline_regret
        self.depreciation_rate = depreciation_rate
        self.states: Dict[str, ImpairmentState] = {}
        self.global_timestep = 0

    def register_capital(self, capital_id: str):
        if capital_id not in self.states:
            self.states[capital_id] = ImpairmentState(
                capital_id=capital_id, depreciation_rate=self.depreciation_rate
            )

    def update(self, capital_id: str, regret: float):
        self.global_timestep += 1
        self.register_capital(capital_id)
        state = self.states[capital_id]

        state.recent_regrets.append(regret)
        if len(state.recent_regrets) > self.window_size * 3:
            state.recent_regrets = state.recent_regrets[-self.window_size:]

        state.last_validated_timestamp = self.global_timestep

        if len(state.recent_regrets) >= self.window_size:
            recent_mean = np.mean(state.recent_regrets[-self.window_size:])
            if recent_mean > self.random_baseline_regret:
                state.impairment_steps += 1
            else:
                state.impairment_steps = max(0, state.impairment_steps - 1)

            state.impaired = state.impairment_steps >= self.impairment_threshold_steps

    def apply_depreciation(self):
        for cid, state in self.states.items():
            steps_since_valid = self.global_timestep - state.last_validated_timestamp
            if steps_since_valid > 0:
                state.confidence = max(0.05, state.confidence * (1.0 - state.depreciation_rate) ** steps_since_valid)

    def get_state(self, capital_id: str) -> ImpairmentState:
        self.register_capital(capital_id)
        self.apply_depreciation()
        return self.states[capital_id]

    def all_impaired(self) -> bool:
        if not self.states:
            return False
        return all(s.impaired for s in self.states.values())

    def any_healthy(self) -> bool:
        return any(not s.impaired for s in self.states.values())

    def healthy_capitals(self) -> List[str]:
        return [cid for cid, s in self.states.items() if not s.impaired]


class FallbackController:
    def __init__(self, safe_action: int = 1):
        self.safe_action = safe_action
        self.fallback_activated = False
        self.consecutive_fallbacks = 0

    def decide(self, detector: CapitalImpairmentDetector, capital_weights: Dict[str, float]) -> Optional[int]:
        if detector.all_impaired() or len(capital_weights) == 0:
            self.fallback_activated = True
            self.consecutive_fallbacks += 1
            return self.safe_action
        self.fallback_activated = False
        self.consecutive_fallbacks = max(0, self.consecutive_fallbacks - 1)
        return None

    def get_fallback_confidence(self) -> float:
        return max(0.1, 1.0 - 0.1 * self.consecutive_fallbacks)