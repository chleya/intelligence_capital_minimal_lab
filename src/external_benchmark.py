"""
External Benchmark — MiniGrid-like Hidden-Goal Gridworld
==========================================================
A minimal externally-validating benchmark distinct from
the StructuredVolatilityEnv synthetic environment.

Goal: a hidden-goal gridworld where an agent must navigate
to a target location given partial observation.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class GridWorldConfig:
    grid_size: int = 7
    n_actions: int = 4
    obs_radius: int = 2
    max_steps: int = 50
    n_goals: int = 3
    seed: int = 0


class HiddenGoalGridWorld:
    """Minimal gridworld with hidden goal location.
    Agent sees a local (2*obs_radius+1)x(2*obs_radius+1) window.
    Goal location must be inferred from reward signal.
    """

    def __init__(self, config: GridWorldConfig = None):
        self.cfg = config or GridWorldConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.grid_size = self.cfg.grid_size
        self.obs_radius = self.cfg.obs_radius
        self.max_steps = self.cfg.max_steps
        self.n_actions = self.cfg.n_actions
        self.agent_pos = np.zeros(2, dtype=int)
        self.goal_pos = np.zeros(2, dtype=int)
        self.goal_id = 0
        self.steps = 0
        self.done = False

    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent_pos = self.rng.integers(0, self.grid_size, 2)
        goals = []
        for _ in range(self.cfg.n_goals):
            g = self.rng.integers(0, self.grid_size, 2)
            goals.append(g)
        self.goal_id = self.rng.integers(0, self.cfg.n_goals)
        self.goal_pos = goals[self.goal_id]
        self.steps = 0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        r = self.obs_radius
        obs = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.float32)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                gy, gx = self.agent_pos[0] + dy, self.agent_pos[1] + dx
                if 0 <= gy < self.grid_size and 0 <= gx < self.grid_size:
                    if gy == self.goal_pos[0] and gx == self.goal_pos[1]:
                        obs[dy + r, dx + r] = 2.0
                    elif gy == self.agent_pos[0] and gx == self.agent_pos[1]:
                        obs[dy + r, dx + r] = 1.0
                    else:
                        obs[dy + r, dx + r] = -0.1
                else:
                    obs[dy + r, dx + r] = -1.0
        return obs.flatten()

    def step(self, action: int):
        old_dist = np.sum(np.abs(self.agent_pos - self.goal_pos))
        if action == 0:
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        new_dist = np.sum(np.abs(self.agent_pos - self.goal_pos))
        self.steps += 1
        at_goal = (self.agent_pos[0] == self.goal_pos[0] and self.agent_pos[1] == self.goal_pos[1])
        if at_goal:
            reward = 1.0
            self.done = True
        else:
            reward = 0.1 * (old_dist - new_dist) - 0.01
        if self.steps >= self.max_steps and not at_goal:
            self.done = True
            reward = -0.5
        return self._get_obs(), reward, self.done, {"dist": new_dist, "at_goal": at_goal}

    def oracle_action(self):
        dy = self.goal_pos[0] - self.agent_pos[0]
        dx = self.goal_pos[1] - self.agent_pos[1]
        if abs(dy) > abs(dx):
            return 1 if dy > 0 else 0
        else:
            return 3 if dx > 0 else 2


class GridWorldBenchmark:
    """Run gridworld benchmark with a given policy."""

    def __init__(self, n_episodes: int = 50, seed: int = 0):
        self.n_episodes = n_episodes
        self.seed = seed
        self.env = HiddenGoalGridWorld(GridWorldConfig(seed=seed))

    def evaluate(self, policy_fn, label: str = "unknown"):
        """policy_fn(obs) -> action"""
        results = []
        for ep in range(self.n_episodes):
            obs = self.env.reset(seed=self.seed + ep * 100)
            total_reward = 0.0
            reached = False
            for _ in range(self.env.max_steps):
                action = policy_fn(obs)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                if info["at_goal"]:
                    reached = True
                if done:
                    break
            results.append({"episode": ep, "reward": total_reward, "reached_goal": reached, "label": label})
        return results


def random_policy(obs):
    return np.random.default_rng().integers(0, 4)