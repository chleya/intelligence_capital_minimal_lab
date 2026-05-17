"""
External HiddenGoalGridWorld-v2
=================================
Improved 5x5 grid environment for Task_W/W4:
  - 5x5 grid (25 states)
  - horizon=100
  - action set [up,down,left,right] = [0,1,2,3]
  - distance shaping reward
  - waypoint reward = 0.3
  - goal reward = 1.0
  - observation includes Manhattan-distance-to-goal hint
"""
import numpy as np


class HiddenGoalGridWorldV2:
    """
    5x5 partially observable grid with hidden goal.

    Observation: flattened 5x5 grid (25-d) with:
      - -1 = unknown
      - 0 = empty/no-goal
      - 1 = agent position (with increasingly confident shade)
      - 2 = goal (only revealed if agent steps on it)

    Actions: 0=up, 1=down, 2=left, 3=right
    """
    def __init__(self, size=5, config=None):
        self.size = size
        self.max_steps = 15
        self.action_space = 4
        self.goal_reward = 1.0
        self.waypoint_reward = 0.3
        self.step_penalty = 0.0
        self.goal = None
        self.agent_pos = None
        self.steps = 0
        self.visited = None
        self.waypoint = None
        self.rng = np.random.default_rng(0)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        # Place goal at random non-edge position
        self.goal = (self.rng.integers(1, self.size - 1),
                     self.rng.integers(1, self.size - 1))
        # Place agent at far corner from goal
        if self.goal[0] < self.size // 2:
            ax = self.size - 2
        else:
            ax = 1
        if self.goal[1] < self.size // 2:
            ay = self.size - 2
        else:
            ay = 1
        self.agent_pos = (ax, ay)
        self.visited = set()
        self.visited.add(self.agent_pos)

        # Set waypoint halfway between agent and goal
        self.waypoint = ((ax + self.goal[0]) // 2, (ay + self.goal[1]) // 2)

        return self._get_obs()

    def _get_obs(self):
        """Build partial observation: 25-d grid with distance hint."""
        obs = np.zeros(self.size * self.size + 2, dtype=np.float32)
        # Mark agent position with distance-weighted confidence
        manhattan = abs(self.agent_pos[0] - self.goal[0]) + abs(self.agent_pos[1] - self.goal[1])
        # Distance-to-goal hint (normalized)
        obs[self.size * self.size] = manhattan / (2 * self.size)
        # Visited cells count
        obs[self.size * self.size + 1] = len(self.visited) / (self.size * self.size)

        for r in range(self.size):
            for c in range(self.size):
                idx = r * self.size + c
                if (r, c) == self.agent_pos:
                    obs[idx] = 1.0
                elif (r, c) in self.visited:
                    obs[idx] = 0.5  # visited but not goal
                # else: 0.0 = unknown
        return obs

    def _manhattan(self, pos):
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    def step(self, action):
        """Execute action, return (obs, reward, done, info)."""
        action = int(action) % 4
        r, c = self.agent_pos
        if action == 0:  # up
            r = max(0, r - 1)
        elif action == 1:  # down
            r = min(self.size - 1, r + 1)
        elif action == 2:  # left
            c = max(0, c - 1)
        elif action == 3:  # right
            c = min(self.size - 1, c + 1)

        old_dist = self._manhattan(self.agent_pos)
        self.agent_pos = (r, c)
        self.visited.add(self.agent_pos)
        new_dist = self._manhattan(self.agent_pos)
        self.steps += 1

        at_goal = (self.agent_pos == self.goal)
        at_waypoint = (self.agent_pos == self.waypoint)
        done = at_goal or self.steps >= self.max_steps

        # Reward shaping
        reward = self.step_penalty
        if at_goal:
            reward += self.goal_reward
        elif at_waypoint and self.waypoint not in getattr(self, '_waypoint_collected', set()):
            if not hasattr(self, '_waypoint_collected'):
                self._waypoint_collected = set()
            self._waypoint_collected.add(self.waypoint)
            reward += self.waypoint_reward
        else:
            # Small distance improvement reward
            dist_improvement = old_dist - new_dist
            reward += max(0, dist_improvement * 0.02)

        info = {
            "at_goal": at_goal,
            "at_waypoint": at_waypoint,
            "steps": self.steps,
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal,
            "distance": new_dist,
            "visited_count": len(self.visited),
        }
        return self._get_obs(), reward, done, info


def compute_grid_v2_capital_scores(grid_env, caps_v2, n_trials=200, seed_offset=500):
    """Compute per-capital success rate on grid-v2 (full episodes)."""
    NC = len(caps_v2)
    scores = np.zeros((n_trials, NC), dtype=np.float32)
    for s in range(n_trials):
        for ci in range(NC):
            obs = grid_env.reset(seed=s * 13 + ci * 7 + seed_offset)
            a = caps_v2[ci].act({"obs": obs, "_reset": True}, [])
            reached = False
            for _step in range(grid_env.max_steps):
                obs, _, done, info = grid_env.step(a)
                if info["at_goal"]:
                    reached = True
                    break
                if done:
                    break
                a = caps_v2[ci].act({"obs": obs}, [])
            scores[s, ci] = float(reached)
    return scores