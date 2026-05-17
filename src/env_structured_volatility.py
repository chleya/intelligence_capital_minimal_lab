import numpy as np


class StructuredVolatilityEnv:
    def __init__(self, seed=0, state_dim=2, mode_flip_prob=0.08,
                 autonomous_drift=0.05, autonomous_noise=0.1,
                 action_gain=0.25, action_noise=0.05, action_sign_flip=True,
                 history_len=8,
                 action_cost=0.15, state_dependent_gain=True, saturation_k=0.5):
        self.rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        self.mode_flip_prob = mode_flip_prob
        self.autonomous_drift = autonomous_drift
        self.autonomous_noise = autonomous_noise
        self.action_gain = action_gain
        self.action_noise = action_noise
        self.action_sign_flip = action_sign_flip
        self.history_len = history_len
        self.action_cost = action_cost
        self.state_dependent_gain = state_dependent_gain
        self.saturation_k = saturation_k
        self.actions = np.array([-1, 0, 1], dtype=np.float32)
        self.state = None
        self.mode = None
        self.t = 0
        self._history_obs = []
        self._history_act = []

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.state = self.rng.normal(0, 0.5, self.state_dim).astype(np.float32)
        self.mode = self.rng.integers(0, 2)
        self._history_obs = [self.state.copy() for _ in range(self.history_len)]
        self._history_act = [0.0 for _ in range(self.history_len)]
        return self._get_obs()

    def _get_obs(self):
        hist = []
        for o, a in zip(self._history_obs, self._history_act):
            hist.append(np.concatenate([o, [a]]))
        return np.concatenate(hist).astype(np.float32)

    def _effective_gain(self, action):
        """Compute action gain, optionally state-dependent with saturation."""
        gain = self.action_gain
        if self.state_dependent_gain and action != 0:
            # Saturation: when |state| is large, action effect diminishes
            state_magnitude = np.linalg.norm(self.state)
            saturation = 1.0 / (1.0 + self.saturation_k * state_magnitude)
            gain = gain * saturation
        return gain

    def step(self, action):
        sign = 1.0
        if self.action_sign_flip and self.mode == 1:
            sign = -1.0
        eff_gain = self._effective_gain(action)
        action_effect = sign * eff_gain * action * np.ones(self.state_dim)
        action_effect += self.rng.normal(0, self.action_noise, self.state_dim)
        # Action cost subtracted from outcome (only for non-zero actions)
        if action != 0:
            action_effect -= self.action_cost * np.ones(self.state_dim)
        auto_effect = self.rng.normal(0, self.autonomous_noise, self.state_dim)
        auto_effect -= self.autonomous_drift * self.state
        self.state = self.state + action_effect + auto_effect
        self.state = self.state.astype(np.float32)
        if self.rng.random() < self.mode_flip_prob:
            self.mode = 1 - self.mode
        self._history_obs.pop(0)
        self._history_obs.append(self.state.copy())
        self._history_act.pop(0)
        self._history_act.append(float(action))
        self.t += 1
        return self._get_obs(), self.state.copy(), float(self.mode)

    def get_current_state(self):
        return self.state.copy()

    def get_history_obs(self):
        return list(self._history_obs)

    def get_history_act(self):
        return list(self._history_act)

    def snapshot(self):
        return {
            "state": self.state.copy(),
            "mode": self.mode,
            "t": self.t,
            "history_obs": [o.copy() for o in self._history_obs],
            "history_act": list(self._history_act),
            "rng_state": self.rng.bit_generator.state,
        }

    def restore(self, snap):
        self.state = snap["state"].copy()
        self.mode = snap["mode"]
        self.t = snap["t"]
        self._history_obs = [o.copy() for o in snap["history_obs"]]
        self._history_act = list(snap["history_act"])
        self.rng.bit_generator.state = snap["rng_state"]

    def step_forward(self, action, horizon=1):
        snap = self.snapshot()
        for _ in range(horizon):
            _, outcome, _ = self.step(action)
        self.restore(snap)
        return outcome.copy()

    def compute_outcomes(self, horizon=1):
        outcomes = {}
        for a in self.actions:
            outcomes[int(a)] = self.step_forward(a, horizon)
        return outcomes

    def get_oracle_residual(self, horizon=1, action=None):
        outcomes = self.compute_outcomes(horizon)
        noop = outcomes[0]
        residuals = {}
        for a_val, out in outcomes.items():
            residuals[a_val] = out - noop
        if action is not None:
            return residuals[int(action)]
        return residuals

    def get_best_action(self, horizon=1):
        outcomes = self.compute_outcomes(horizon)
        d = {a: float(np.sum(out)) for a, out in outcomes.items()}
        return max(d, key=d.get)
