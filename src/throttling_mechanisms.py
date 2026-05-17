import numpy as np
from .models import (
    StateOnlyPredictor, ActionOnlyPredictor,
    AEPCompressor, ResidualCompressor, CenteredResidualCompressor,
    CounterfactualCompressor, CausalContrastCompressor, ResidualAdversarialCompressor,
)
from .memory_baselines import RawMemoryFull, RawMemoryEqualCost, PrototypeMemory


MECHANISM_REGISTRY = {}


def register(name):
    def decorator(cls):
        MECHANISM_REGISTRY[name] = cls
        return cls
    return decorator


class BaseMechanism:
    name = "base"
    is_memory = False

    def fit(self, X, outcomes, action_indices):
        raise NotImplementedError

    def predict(self, X, action_indices=None):
        raise NotImplementedError

    def predict_all_actions(self, X):
        raise NotImplementedError

    def count_parameters(self):
        return 0

    def cost_bytes(self):
        return self.count_parameters() * 4


@register("state_only")
class StateOnlyMechanism(BaseMechanism):
    name = "state_only"

    def __init__(self, obs_dim, history_len, bottleneck_dim=16):
        self.model = StateOnlyPredictor(obs_dim, history_len, bottleneck_dim=bottleneck_dim)

    def fit(self, X, outcomes, action_indices):
        pass

    def predict_all_actions(self, X):
        import torch
        x_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model(x_t).numpy()

    def count_parameters(self):
        return self.model.count_parameters()


@register("action_only")
class ActionOnlyMechanism(BaseMechanism):
    name = "action_only"

    def __init__(self):
        self.model = ActionOnlyPredictor()

    def fit(self, X, outcomes, action_indices):
        pass

    def predict_all_actions(self, X):
        import torch
        x_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model(x_t).numpy()

    def count_parameters(self):
        return 3


@register("aep_compressor")
class AEPCompressorMechanism(BaseMechanism):
    name = "aep_compressor"

    def __init__(self, obs_dim, history_len, bottleneck_dim=16):
        self.model = AEPCompressor(obs_dim, history_len, bottleneck_dim=bottleneck_dim)

    def predict_all_actions(self, X):
        import torch
        x_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model.predict_all_actions(x_t).numpy()

    def count_parameters(self):
        return self.model.count_parameters()


@register("residual_compressor")
class ResidualCompressorMechanism(BaseMechanism):
    name = "residual_compressor"

    def __init__(self, obs_dim, history_len, bottleneck_dim=16, residual_dim=8):
        self.model = ResidualCompressor(obs_dim, history_len, bottleneck_dim=bottleneck_dim, residual_dim=residual_dim)

    def predict_all_actions(self, X):
        import torch
        x_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model.predict_all_actions(x_t).numpy()

    def count_parameters(self):
        return self.model.count_parameters()


@register("centered_residual")
class CenteredResidualMechanism(BaseMechanism):
    name = "centered_residual"

    def __init__(self, obs_dim, history_len, bottleneck_dim=16, residual_dim=8):
        self.model = CenteredResidualCompressor(obs_dim, history_len, bottleneck_dim=bottleneck_dim, residual_dim=residual_dim)

    def predict_all_actions(self, X):
        import torch
        x_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model.predict_all_actions(x_t).numpy()

    def count_parameters(self):
        return self.model.count_parameters()


@register("counterfactual_compressor")
class CounterfactualCompressorMechanism(BaseMechanism):
    name = "counterfactual_compressor"

    def __init__(self, obs_dim, history_len, bottleneck_dim=16):
        self.model = CounterfactualCompressor(obs_dim, history_len, bottleneck_dim=bottleneck_dim)

    def predict_all_actions(self, X):
        import torch
        x_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model(x_t).numpy()

    def count_parameters(self):
        return self.model.count_parameters()


@register("causal_contrast")
class CausalContrastMechanism(BaseMechanism):
    name = "causal_contrast"

    def __init__(self, obs_dim, history_len, bottleneck_dim=16, temperature=0.1):
        self.model = CausalContrastCompressor(obs_dim, history_len, bottleneck_dim=bottleneck_dim, temperature=temperature)

    def predict_all_actions(self, X):
        import torch
        x_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y, _ = self.model(x_t)
            return y.numpy()

    def count_parameters(self):
        return self.model.count_parameters()


@register("residual_adversarial")
class ResidualAdversarialMechanism(BaseMechanism):
    name = "residual_adversarial"

    def __init__(self, obs_dim, history_len, bottleneck_dim=16, residual_dim=8):
        self.model = ResidualAdversarialCompressor(obs_dim, history_len, bottleneck_dim=bottleneck_dim, residual_dim=residual_dim)

    def predict_all_actions(self, X):
        import torch
        x_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model.predict_all_actions(x_t).numpy()

    def count_parameters(self):
        return self.model.count_parameters()


@register("raw_memory_full")
class RawMemoryFullMechanism(BaseMechanism):
    name = "raw_memory_full"
    is_memory = True

    def __init__(self, k=5):
        self.memory = RawMemoryFull(k=k)

    def fit(self, X, outcomes, action_indices):
        outcomes_list = []
        for i in range(len(outcomes)):
            outcomes_list.append([outcomes[i][:, 0], outcomes[i][:, 1], outcomes[i][:, 2]])
        self.memory.fit(X, outcomes_list)

    def predict_all_actions(self, X):
        return self.memory.predict(X)

    def cost_bytes(self):
        return self.memory.cost_bytes(X.shape[1] if hasattr(X, "shape") else len(X[0]), self.memory.stored_samples_count)

    @property
    def stored_samples_count(self):
        return self.memory.stored_samples_count


@register("raw_memory_equal_cost")
class RawMemoryEqualCostMechanism(BaseMechanism):
    name = "raw_memory_equal_cost"
    is_memory = True

    def __init__(self, param_budget=5000, k=5):
        self.memory = RawMemoryEqualCost(param_budget=param_budget, k=k)

    def fit(self, X, outcomes, action_indices):
        outcomes_list = []
        for i in range(len(outcomes)):
            outcomes_list.append([outcomes[i][:, 0], outcomes[i][:, 1], outcomes[i][:, 2]])
        self.memory.fit(X, outcomes_list)

    def predict_all_actions(self, X):
        return self.memory.predict(X)

    def cost_bytes(self):
        return self.memory.cost_bytes()

    @property
    def stored_samples_count(self):
        return self.memory.stored_samples_count


@register("prototype_memory")
class PrototypeMemoryMechanism(BaseMechanism):
    name = "prototype_memory"
    is_memory = True

    def __init__(self, n_clusters=20, k=3):
        self.memory = PrototypeMemory(n_clusters=n_clusters, k=k)

    def fit(self, X, outcomes, action_indices):
        outcomes_list = []
        for i in range(len(outcomes)):
            outcomes_list.append([outcomes[i][:, 0], outcomes[i][:, 1], outcomes[i][:, 2]])
        self.memory.fit(X, outcomes_list)

    def predict_all_actions(self, X):
        return self.memory.predict(X)

    def cost_bytes(self):
        return self.memory.cost_bytes()

    @property
    def stored_samples_count(self):
        return self.memory.stored_samples_count


class ShuffledActionControl(BaseMechanism):
    name = "shuffled_action"

    def __init__(self, base_mechanism):
        self.base = base_mechanism
        self._shuffle_map = None

    def fit(self, X, outcomes, action_indices):
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(X))
        shuffled_outcomes = []
        for p in perm:
            shuffled_outcomes.append(outcomes[p])
        self.base.fit(X, shuffled_outcomes, action_indices)

    def predict_all_actions(self, X):
        return self.base.predict_all_actions(X)

    def count_parameters(self):
        return self.base.count_parameters()

    def cost_bytes(self):
        return self.base.cost_bytes()


class PermutedHistoryControl(BaseMechanism):
    name = "permuted_history"

    def __init__(self, base_mechanism, obs_dim, history_len):
        self.base = base_mechanism
        self.obs_dim = obs_dim
        self.history_len = history_len
        self._perm = None

    def _permute_history(self, X):
        if self._perm is None:
            rng = np.random.default_rng(42)
            n_pairs = self.history_len
            self._perm = rng.permutation(n_pairs)
        X_perm = X.copy()
        for i in range(len(X)):
            for j in range(self.history_len):
                src = j * (self.obs_dim + 1)
                dst = self._perm[j] * (self.obs_dim + 1)
                X_perm[i, dst:dst + self.obs_dim + 1] = X[i, src:src + self.obs_dim + 1]
        return X_perm

    def fit(self, X, outcomes, action_indices):
        self.base.fit(self._permute_history(X), outcomes, action_indices)

    def predict_all_actions(self, X):
        return self.base.predict_all_actions(self._permute_history(X))

    def count_parameters(self):
        return self.base.count_parameters()

    def cost_bytes(self):
        return self.base.cost_bytes()