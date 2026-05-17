"""
IC-3-0 Sub-task: Manifold-Constrained Capital Allocation
=========================================================
Reference: mHC (Manifold-Constrained Hyper-Connections)
  Constrain the capital allocation to a doubly stochastic manifold (Birkhoff polytope)
  to prevent trust explosion, weight oscillation, and bad-debt amplification.

Allocators:
  1. UnconstrainedAllocator    — raw weights, no manifold constraint
  2. SimplexWeightAllocator     — softmax projection onto probability simplex
  3. BirkhoffTransitionAllocator — Sinkhorn projection onto Birkhoff polytope
  4. EqualWeightPortfolio       — uniform 1/N weights
  5. BestSingleCapital          — hindsight static best
  6. OracleHindsightAllocator   — per-step oracle

Stability Diagnostics:
  - weight_entropy, max_weight
  - transition_norm (Frobenius)
  - row_sum_error, col_sum_error
  - capital_trust_explosion_score
  - weight_turnover_rate
"""
import os, sys, json, warnings, math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.capital_report import CapitalReport
from src.capital_impairment import CapitalImpairmentDetector

OUTDIR = "results/ic3_0_manifold"
FIGDIR = "results/figures"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# SINKHORN PROJECTION
# ═══════════════════════════════════════════════════════════

def sinkhorn_projection(raw_matrix: np.ndarray, n_iter: int = 20,
                        epsilon: float = 1e-6) -> np.ndarray:
    raw = np.maximum(raw_matrix, epsilon)
    for _ in range(n_iter):
        raw /= (raw.sum(axis=1, keepdims=True) + epsilon)
        raw /= (raw.sum(axis=0, keepdims=True) + epsilon)
    return raw


# ═══════════════════════════════════════════════════════════
# ALLOCATORS
# ═══════════════════════════════════════════════════════════

class UnconstrainedAllocator:
    def __init__(self, n_capitals: int = 4, lr: float = 0.05):
        self.n = n_capitals
        self.lr = lr
        self.raw_values = np.ones(n_capitals, dtype=np.float32) / n_capitals
        self._weight_history: List[np.ndarray] = []

    def get_weights(self, reports: List[CapitalReport]) -> np.ndarray:
        w = np.maximum(self.raw_values, 0.0)
        w /= (w.sum() + 1e-8)
        self._weight_history.append(w.copy())
        return w

    def select(self, reports: List[CapitalReport]) -> int:
        w = self.get_weights(reports)
        return int(np.random.choice(self.n, p=w))

    def update(self, capital_idx: int, reward: float):
        self.raw_values[capital_idx] += self.lr * (reward - self.raw_values[capital_idx])

    @property
    def weight_history(self) -> np.ndarray:
        return np.array(self._weight_history) if self._weight_history else np.zeros((0, self.n))


class SimplexWeightAllocator:
    def __init__(self, n_capitals: int = 4, lr: float = 0.05, temperature: float = 1.0,
                 entropy_reg: float = 0.01, max_weight_cap: float = 0.85):
        self.n = n_capitals
        self.lr = lr
        self.temperature = temperature
        self.entropy_reg = entropy_reg
        self.max_weight_cap = max_weight_cap
        self.logits = np.zeros(n_capitals, dtype=np.float32)
        self._weight_history: List[np.ndarray] = []

    def get_weights(self, reports: List[CapitalReport]) -> np.ndarray:
        logits_scaled = self.logits / max(self.temperature, 0.01)
        logits_scaled -= logits_scaled.max()
        exp_vals = np.exp(logits_scaled)

        # Entropy regularisation: push weights toward uniformity
        if self.entropy_reg > 0:
            w_raw = exp_vals / (exp_vals.sum() + 1e-8)
            entropy = -np.sum(w_raw * np.log(w_raw + 1e-8))
            uniform = np.ones(self.n) / self.n
            alpha = min(self.entropy_reg / max(entropy, 0.01), 0.3)
            w = (1.0 - alpha) * w_raw + alpha * uniform
        else:
            w = exp_vals / (exp_vals.sum() + 1e-8)

        # Hard cap: no single weight may exceed max_weight_cap
        if w.max() > self.max_weight_cap:
            excess = w.max() - self.max_weight_cap
            w = np.where(w == w.max(), self.max_weight_cap, w)
            others = (w != self.max_weight_cap)
            if others.any():
                w[others] += excess / others.sum()
            w = np.maximum(w, 0.0)

        w = np.maximum(w, 1e-8)
        w /= w.sum()
        self._weight_history.append(w.copy())
        return w

    def select(self, reports: List[CapitalReport]) -> int:
        w = self.get_weights(reports)
        return int(np.random.choice(self.n, p=w))

    def update(self, capital_idx: int, reward: float):
        self.logits[capital_idx] += self.lr * reward

    @property
    def weight_history(self) -> np.ndarray:
        return np.array(self._weight_history) if self._weight_history else np.zeros((0, self.n))


class BirkhoffTransitionAllocator:
    def __init__(self, n_capitals: int = 4, lr: float = 0.05,
                 sinkhorn_iters: int = 20, sinkhorn_eps: float = 1e-6,
                 momentum: float = 0.9):
        self.n = n_capitals
        self.lr = lr
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_eps = sinkhorn_eps
        self.momentum = momentum

        self.raw_T = np.ones((n_capitals, n_capitals), dtype=np.float32) / n_capitals
        self.raw_T_smoothed = self.raw_T.copy()
        self.weights = np.ones(n_capitals, dtype=np.float32) / n_capitals
        self._weight_history: List[np.ndarray] = []
        self._transition_history: List[np.ndarray] = []
        self._row_err_history: List[float] = []
        self._col_err_history: List[float] = []

    def get_transition(self, reports: List[CapitalReport]) -> np.ndarray:
        raw_pos = np.maximum(self.raw_T_smoothed, self.sinkhorn_eps)
        T = sinkhorn_projection(raw_pos, self.sinkhorn_iters, self.sinkhorn_eps)
        self._transition_history.append(T.copy())
        self._row_err_history.append(float(np.abs(T.sum(axis=1) - 1.0).mean()))
        self._col_err_history.append(float(np.abs(T.sum(axis=0) - 1.0).mean()))
        return T

    def get_weights(self, reports: List[CapitalReport]) -> np.ndarray:
        T = self.get_transition(reports)
        self.weights = T @ self.weights
        self.weights /= (self.weights.sum() + 1e-8)
        self._weight_history.append(self.weights.copy())
        return self.weights

    def select(self, reports: List[CapitalReport]) -> int:
        w = self.get_weights(reports)
        return int(np.random.choice(self.n, p=w))

    def update(self, capital_idx: int, reward: float):
        self.raw_T[:, capital_idx] += self.lr * reward
        self.raw_T_smoothed = (self.momentum * self.raw_T_smoothed +
                               (1.0 - self.momentum) * self.raw_T)

    @property
    def weight_history(self) -> np.ndarray:
        return np.array(self._weight_history) if self._weight_history else np.zeros((0, self.n))

    @property
    def transition_history(self) -> np.ndarray:
        return np.array(self._transition_history) if self._transition_history else np.zeros((0, self.n, self.n))

    @property
    def row_errors(self) -> List[float]:
        return self._row_err_history

    @property
    def col_errors(self) -> List[float]:
        return self._col_err_history


class EqualWeightPortfolio:
    def __init__(self, n_capitals: int = 4):
        self.n = n_capitals
        self._weight_history: List[np.ndarray] = []

    def get_weights(self, reports: List[CapitalReport]) -> np.ndarray:
        w = np.ones(self.n, dtype=np.float32) / self.n
        self._weight_history.append(w.copy())
        return w

    def select(self, reports: List[CapitalReport]) -> int:
        return int(np.random.choice(self.n, p=np.ones(self.n) / self.n))

    def update(self, capital_idx: int, reward: float):
        pass

    @property
    def weight_history(self) -> np.ndarray:
        return np.array(self._weight_history) if self._weight_history else np.zeros((0, self.n))


class BestSingleCapital:
    def __init__(self, best_idx: int, n_capitals: int = 4):
        self.best_idx = best_idx
        self.n = n_capitals
        self._weight_history: List[np.ndarray] = []

    def get_weights(self, reports: List[CapitalReport]) -> np.ndarray:
        w = np.zeros(self.n, dtype=np.float32)
        w[self.best_idx] = 1.0
        self._weight_history.append(w.copy())
        return w

    def select(self, reports: List[CapitalReport]) -> int:
        return self.best_idx

    def update(self, capital_idx: int, reward: float):
        pass

    @property
    def weight_history(self) -> np.ndarray:
        return np.array(self._weight_history) if self._weight_history else np.zeros((0, self.n))


class OracleHindsightAllocator:
    def __init__(self, oracle_correct: np.ndarray, n_capitals: int = 4):
        self.oracle_best = np.argmax(oracle_correct, axis=1)
        self.n = n_capitals
        self._weight_history: List[np.ndarray] = []

    def get_weights(self, reports: List[CapitalReport]) -> np.ndarray:
        w = np.zeros(self.n, dtype=np.float32)
        w[self.oracle_best[self._step]] = 1.0
        self._weight_history.append(w.copy())
        return w

    def select(self, reports: List[CapitalReport]) -> int:
        return int(self.oracle_best[self._step])

    def set_step(self, step: int):
        self._step = step

    def update(self, capital_idx: int, reward: float):
        pass

    @property
    def weight_history(self) -> np.ndarray:
        return np.array(self._weight_history) if self._weight_history else np.zeros((0, self.n))


# ═══════════════════════════════════════════════════════════
# STABILITY DIAGNOSTICS
# ═══════════════════════════════════════════════════════════

@dataclass
class StabilityDiagnostics:
    weight_entropy: float = 0.0
    max_weight: float = 0.0
    transition_norm: float = 0.0
    row_sum_error: float = 0.0
    col_sum_error: float = 0.0
    capital_trust_explosion_score: float = 0.0
    weight_turnover_rate: float = 0.0
    weight_collapse_ratio: float = 0.0
    oscillation_energy: float = 0.0

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def compute_stability_diagnostics(
    weight_history: np.ndarray,
    transition_history: Optional[np.ndarray] = None,
    row_errors: Optional[List[float]] = None,
    col_errors: Optional[List[float]] = None,
    regret_series: Optional[np.ndarray] = None,
) -> StabilityDiagnostics:
    d = StabilityDiagnostics()
    if weight_history is None or len(weight_history) < 2:
        return d

    wh = np.array(weight_history, dtype=np.float32)
    n_steps = wh.shape[0]
    n_caps = wh.shape[1]

    avg_weights = wh.mean(axis=0)
    d.max_weight = float(avg_weights.max())
    avg_weights_safe = np.maximum(avg_weights, 1e-8)
    d.weight_entropy = float(-np.sum(avg_weights_safe * np.log(avg_weights_safe)))
    d.weight_entropy /= max(np.log(float(n_caps)), 1e-8)

    d.weight_turnover_rate = float(np.abs(wh[1:] - wh[:-1]).mean())
    d.oscillation_energy = float(np.var(np.diff(wh, axis=0), axis=0).mean())
    d.capital_trust_explosion_score = float(np.max(np.diff(wh, axis=0)))

    collapse_count = 0
    for i in range(n_steps):
        if wh[i].max() > 0.9:
            collapse_count += 1
    d.weight_collapse_ratio = float(collapse_count / max(1, n_steps))

    if transition_history is not None and len(transition_history) > 0:
        th = np.array(transition_history)
        d.transition_norm = float(np.mean([np.linalg.norm(t, 'fro') for t in th]))
    if row_errors is not None:
        d.row_sum_error = float(np.mean(row_errors)) if row_errors else 0.0
    if col_errors is not None:
        d.col_sum_error = float(np.mean(col_errors)) if col_errors else 0.0
    if regret_series is not None and len(regret_series) > 0:
        d.capital_trust_explosion_score = float(
            d.capital_trust_explosion_score * np.mean(regret_series)
        )

    return d


# ═══════════════════════════════════════════════════════════
# DEATH CONDITION CHECKER
# ═══════════════════════════════════════════════════════════

def check_death_conditions(
    unconstrained_diag: StabilityDiagnostics,
    simplex_diag: StabilityDiagnostics,
    birkhoff_diag: StabilityDiagnostics,
    unconstrained_regret: float,
    simplex_regret: float,
    birkhoff_regret: float,
    best_single_regret: float,
    eval_performance: Dict[str, float],
    tolerance: float = 0.05,
) -> Dict[str, Tuple[bool, str]]:
    results = {}

    # D1: weight collapse (single capital > 0.95 persistently AND regret rising)
    D1 = (simplex_diag.weight_collapse_ratio > 0.5 or
          birkhoff_diag.weight_collapse_ratio > 0.5)
    results["D1_weight_collapse"] = (D1,
        f"collapse_ratio simplex={simplex_diag.weight_collapse_ratio:.3f} "
        f"birkhoff={birkhoff_diag.weight_collapse_ratio:.3f}")

    # D2: transition matrix row/col sum error > tolerance
    D2 = (birkhoff_diag.row_sum_error > tolerance or
          birkhoff_diag.col_sum_error > tolerance)
    results["D2_transition_sum_error"] = (D2,
        f"row_err={birkhoff_diag.row_sum_error:.6f} col_err={birkhoff_diag.col_sum_error:.6f} (tol={tolerance})")

    # D3: weight turnover too high (oscillation)
    D3 = (simplex_diag.weight_turnover_rate > 0.15 or
          birkhoff_diag.weight_turnover_rate > 0.15)
    results["D3_weight_oscillation"] = (D3,
        f"turnover simplex={simplex_diag.weight_turnover_rate:.4f} "
        f"birkhoff={birkhoff_diag.weight_turnover_rate:.4f}")

    # D4: unconstrained better but significantly less stable than constrained
    unconstrained_better = eval_performance.get("Unconstrained", 0.0) > max(
        eval_performance.get("SimplexWeight", 0.0),
        eval_performance.get("BirkhoffTransition", 0.0))
    best_constrained_turnover = min(
        simplex_diag.weight_turnover_rate,
        birkhoff_diag.weight_turnover_rate)
    unconstrained_less_stable = (
        unconstrained_diag.weight_turnover_rate > 0.003 and
        unconstrained_diag.weight_turnover_rate > 1.5 * best_constrained_turnover
    )
    D4 = unconstrained_better and unconstrained_less_stable
    results["D4_unconstrained_vs_constrained"] = (D4,
        f"unconstr_perf={eval_performance.get('Unconstrained', 0):.4f} "
        f"simplex={eval_performance.get('SimplexWeight', 0):.4f} "
        f"birkhoff={eval_performance.get('BirkhoffTransition', 0):.4f} | "
        f"turnover unconstrained={unconstrained_diag.weight_turnover_rate:.4f} "
        f"simplex={simplex_diag.weight_turnover_rate:.4f}")

    # D5: constrained allocator cannot beat BestSingle
    constrained_best = max(
        eval_performance.get("SimplexWeight", 0.0),
        eval_performance.get("BirkhoffTransition", 0.0))
    best_single_perf = eval_performance.get("BestSingle", 0.0)
    D5 = constrained_best <= best_single_perf
    results["D5_constrained_lt_best_single"] = (D5,
        f"constrained_best={constrained_best:.4f} BestSingle={best_single_perf:.4f}")

    return results


# ═══════════════════════════════════════════════════════════
# SYNTHETIC REPORT GENERATOR
# ═══════════════════════════════════════════════════════════

def generate_synthetic_report(capital_idx: int, timestep: int,
                              base_perf: List[float], noise_scale: float = 0.1,
                              impairment_prob: float = 0.05,
                              task_type: str = "mixed") -> CapitalReport:
    perf_mean = base_perf[capital_idx]
    perf = np.clip(perf_mean + np.random.randn() * noise_scale, 0.0, 1.0)

    r = CapitalReport(
        capital_id=f"capital_{capital_idx}",
        capital_type=f"type_{capital_idx}",
        timestamp=timestep,
        recent_prediction_error=1.0 - perf,
        recent_regret=1.0 - perf,
        confidence=1.0 - 0.001 * timestep,
        realized_utility=perf,
        capital_local_ood_score=np.random.beta(2, 5),
        inference_cost=float(capital_idx + 1),
        update_cost=float((capital_idx + 1) * 100),
        storage_cost=float((capital_idx + 1) * 1000),
        goal_shift_score=np.random.beta(1, 10),
        expected_probe_value=max(0.0, perf - np.random.beta(5, 2) * 0.5),
        impaired=np.random.rand() < impairment_prob,
    )
    r.depreciation_score = max(0.0, 1.0 - r.confidence)
    r.bad_debt_score = r.recent_regret * (1.0 - r.confidence)
    r.calibration_error = abs(r.recent_prediction_error - r.recent_regret)
    r.realization_rate = max(0.1, 1.0 - r.recent_regret)
    r.nearest_support_distance = r.capital_local_ood_score
    return r


# ═══════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════

def run_manifold_experiment(
    n_capitals: int = 4,
    n_steps: int = 2000,
    base_perfs: Optional[List[float]] = None,
    regime_change_every: int = 250,
    sinkhorn_iters: int = 20,
    sinkhorn_eps: float = 1e-6,
    seed: int = 42,
) -> Dict[str, Any]:
    np.random.seed(seed)
    if base_perfs is None:
        base_perfs = [0.35, 0.42, 0.58, 0.25]

    n = n_capitals

    # Allocators
    unconstr = UnconstrainedAllocator(n, lr=0.08)
    simplex = SimplexWeightAllocator(n, lr=0.08, temperature=1.0)
    birkhoff = BirkhoffTransitionAllocator(n, lr=0.08,
                                           sinkhorn_iters=sinkhorn_iters,
                                           sinkhorn_eps=sinkhorn_eps)
    equal_wt = EqualWeightPortfolio(n)
    best_single = BestSingleCapital(int(np.argmax(base_perfs)), n)

    # Oracle (compute true best at each step)
    oracle_per_step = []
    current_perfs = list(base_perfs)
    for step in range(n_steps):
        if step > 0 and step % regime_change_every == 0:
            perm = np.random.permutation(n)
            current_perfs = [base_perfs[p] for p in perm]
        noise = np.random.randn(n) * 0.12
        step_perfs = [np.clip(current_perfs[i] + noise[i], 0.0, 1.0) for i in range(n)]
        oracle_per_step.append(step_perfs)

    oracle_arr = np.array(oracle_per_step)
    oracle = OracleHindsightAllocator(oracle_arr, n)

    allocators: Dict[str, Any] = {
        "Unconstrained":     unconstr,
        "SimplexWeight":      simplex,
        "BirkhoffTransition":  birkhoff,
        "EqualWeight":        equal_wt,
        "BestSingle":         best_single,
        "OracleHindsight":    oracle,
    }

    # Accumulators
    perf_accum = defaultdict(list)
    report_history: Dict[str, List[List[CapitalReport]]] = defaultdict(list)

    current_perfs = list(base_perfs)
    for step in range(n_steps):
        if step > 0 and step % regime_change_every == 0:
            perm = np.random.permutation(n)
            current_perfs = [base_perfs[p] for p in perm]
        noise = np.random.randn(n) * 0.12
        step_perfs = [np.clip(current_perfs[i] + noise[i], 0.0, 1.0) for i in range(n)]

        reports = [generate_synthetic_report(i, step, current_perfs,
                                              noise_scale=0.12,
                                              impairment_prob=0.02)
                   for i in range(n)]

        for name, allo in allocators.items():
            if name == "OracleHindsight":
                allo.set_step(step)
            chosen = allo.select(reports)
            reward = step_perfs[chosen]
            allo.update(chosen, reward)
            perf_accum[name].append(reward)
            report_history[name].append([CapitalReport(
                capital_id=r.capital_id, capital_type=r.capital_type,
                timestamp=r.timestamp, recent_prediction_error=r.recent_prediction_error,
                recent_regret=r.recent_regret, confidence=r.confidence,
                realized_utility=r.realized_utility,
                capital_local_ood_score=r.capital_local_ood_score,
                inference_cost=r.inference_cost, update_cost=r.update_cost,
                storage_cost=r.storage_cost, goal_shift_score=r.goal_shift_score,
                expected_probe_value=r.expected_probe_value,
                depreciation_score=r.depreciation_score,
                bad_debt_score=r.bad_debt_score, impaired=r.impaired,
                calibration_error=r.calibration_error,
                realization_rate=r.realization_rate,
                nearest_support_distance=r.nearest_support_distance,
            ) for r in reports])

    # Diagnostics
    diags: Dict[str, StabilityDiagnostics] = {}
    for name in ["Unconstrained", "SimplexWeight", "BirkhoffTransition"]:
        allo = allocators[name]
        wh = allo.weight_history
        th = allo.transition_history if hasattr(allo, 'transition_history') else None
        re = allo.row_errors if hasattr(allo, 'row_errors') else None
        ce = allo.col_errors if hasattr(allo, 'col_errors') else None
        regrets = 1.0 - np.array(perf_accum[name]) if name in perf_accum else None
        diags[name] = compute_stability_diagnostics(
            wh, transition_history=th, row_errors=re, col_errors=ce,
            regret_series=regrets)

    # Evaluation performance (mean reward)
    eval_perf = {name: float(np.mean(perfs)) for name, perfs in perf_accum.items()}
    bs_perf = eval_perf.get("BestSingle", 0.0)

    # Death conditions
    death_results = check_death_conditions(
        diags.get("Unconstrained", StabilityDiagnostics()),
        diags.get("SimplexWeight", StabilityDiagnostics()),
        diags.get("BirkhoffTransition", StabilityDiagnostics()),
        1.0 - eval_perf.get("Unconstrained", 0.0),
        1.0 - eval_perf.get("SimplexWeight", 0.0),
        1.0 - eval_perf.get("BirkhoffTransition", 0.0),
        1.0 - bs_perf,
        eval_perf,
    )

    triggered = [k for k, (v, _) in death_results.items() if v]

    # Regret curves
    regret_curves = {}
    oracle_mean = eval_perf.get("OracleHindsight", 1.0)
    for name in allocators:
        p = np.array(perf_accum.get(name, []))
        regret_curves[name] = float(np.mean(oracle_mean - p))

    return {
        "eval_performance": eval_perf,
        "diagnostics": diags,
        "death_results": death_results,
        "triggered_deaths": triggered,
        "regret_curves": regret_curves,
        "weight_histories": {name: allo.weight_history
                            for name, allo in allocators.items()
                            if hasattr(allo, 'weight_history')},
        "transition_history": birkhoff.transition_history if hasattr(birkhoff, 'transition_history') else None,
        "oracle_upper_bound": oracle_mean,
        "base_perfs": base_perfs,
        "n_steps": n_steps,
        "regime_change_every": regime_change_every,
    }


# ═══════════════════════════════════════════════════════════
# REPORT GENERATOR
# ═══════════════════════════════════════════════════════════

def generate_report(results: Dict[str, Any], verdict: str) -> str:
    diags = results["diagnostics"]
    death = results["death_results"]
    perf = results["eval_performance"]

    lines = []
    lines.append("# IC-3-0: Manifold-Constrained Capital Allocator Audit\n")
    lines.append(f"## Verdict: {verdict}\n")

    lines.append("## 1. Performance Comparison\n")
    lines.append("| Allocator | Mean Reward | Regret |")
    lines.append("|---|---|---|")
    oracle = perf.get("OracleHindsight", 1.0)
    for name in ["Unconstrained", "SimplexWeight", "BirkhoffTransition",
                 "EqualWeight", "BestSingle", "OracleHindsight"]:
        if name not in perf:
            continue
        p = perf[name]
        reg = oracle - p
        lines.append(f"| {name} | {p:.4f} | {reg:.4f} |")
    lines.append("")

    lines.append("## 2. Stability Diagnostics\n")
    lines.append("| Metric | Unconstrained | SimplexWeight | BirkhoffTransition |")
    lines.append("|---|---|---|---|")
    metrics = ["weight_entropy", "max_weight", "weight_turnover_rate",
               "oscillation_energy", "weight_collapse_ratio",
               "capital_trust_explosion_score", "transition_norm",
               "row_sum_error", "col_sum_error"]
    for m in metrics:
        vals = []
        for name in ["Unconstrained", "SimplexWeight", "BirkhoffTransition"]:
            d = diags.get(name, StabilityDiagnostics())
            v = getattr(d, m, 0.0)
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append(f"| {m} | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines.append("")

    lines.append("## 3. Death Conditions\n")
    lines.append("| Condition | Status | Detail |")
    lines.append("|---|---|---|")
    for k, (triggered, detail) in death.items():
        status = "❌ TRIGGERED" if triggered else "✅ OK"
        lines.append(f"| {k} | {status} | {detail} |")
    lines.append("")

    lines.append("## 4. Research Questions\n")
    q1_uc_turnover = diags.get("Unconstrained", StabilityDiagnostics()).weight_turnover_rate
    q1_sw_turnover = diags.get("SimplexWeight", StabilityDiagnostics()).weight_turnover_rate
    q1_bf_turnover = diags.get("BirkhoffTransition", StabilityDiagnostics()).weight_turnover_rate
    q1 = q1_sw_turnover < q1_uc_turnover or q1_bf_turnover < q1_uc_turnover
    lines.append(f"**Q1: Does manifold constraint reduce weight oscillation?**")
    lines.append(f"  Unconstrained turnover: {q1_uc_turnover:.5f}")
    lines.append(f"  Simplex turnover:       {q1_sw_turnover:.5f}")
    lines.append(f"  Birkhoff turnover:      {q1_bf_turnover:.5f}")
    lines.append(f"  Answer: {'YES — constrained reduces weight churn' if q1 else 'NO — further investigation needed'}\n")

    uc_collapse = diags.get("Unconstrained", StabilityDiagnostics()).weight_collapse_ratio
    sw_collapse = diags.get("SimplexWeight", StabilityDiagnostics()).weight_collapse_ratio
    bf_collapse = diags.get("BirkhoffTransition", StabilityDiagnostics()).weight_collapse_ratio
    q2 = sw_collapse < uc_collapse or bf_collapse < uc_collapse
    lines.append(f"**Q2: Does it prevent bad-capital amplification?**")
    lines.append(f"  Unconstrained collapse ratio: {uc_collapse:.3f}")
    lines.append(f"  Simplex collapse ratio:       {sw_collapse:.3f}")
    lines.append(f"  Birkhoff collapse ratio:      {bf_collapse:.3f}")
    lines.append(f"  Answer: {'YES — collapse ratio reduced' if q2 else 'NO — both near zero in this regime'}\n")

    lines.append(f"**Q3: Does it maintain or improve cost-normalized regret?**")
    lines.append(f"  Simplex regret:       {oracle - perf.get('SimplexWeight', 0.0):.4f}")
    lines.append(f"  Birkhoff regret:      {oracle - perf.get('BirkhoffTransition', 0.0):.4f}")
    lines.append(f"  Unconstrained regret: {oracle - perf.get('Unconstrained', 0.0):.4f}")
    lines.append(f"  Answer: See comparison table in Section 1\n")

    bf_turnover = diags.get("BirkhoffTransition", StabilityDiagnostics()).weight_turnover_rate
    sw_turnover = diags.get("SimplexWeight", StabilityDiagnostics()).weight_turnover_rate
    q4 = bf_turnover <= sw_turnover
    lines.append(f"**Q4: Is Birkhoff transition more stable than simple softmax?**")
    lines.append(f"  Birkhoff turnover: {bf_turnover:.5f}")
    lines.append(f"  Simplex turnover:  {sw_turnover:.5f}")
    lines.append(f"  Answer: {'YES — Birkhoff is more stable (lower turnover)' if q4 else 'NO'}\n")

    triggered = results.get("triggered_deaths", [])
    lines.append(f"**Q5: Does it qualify for formal IC-3 entry?**")
    if not triggered:
        lines.append(f"  Answer: YES — ALL death conditions passed. Manifold-constrained allocation is ready for IC-3.\n")
    else:
        lines.append(f"  Answer: NO — {len(triggered)} death conditions triggered: {triggered}\n")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("IC-3-0: Manifold-Constrained Capital Allocation Audit")
    print("=" * 60)

    print("\nRunning experiment...")
    results = run_manifold_experiment(
        n_capitals=4,
        n_steps=3000,
        base_perfs=[0.35, 0.42, 0.58, 0.25],
        regime_change_every=100,
        sinkhorn_iters=20,
        sinkhorn_eps=1e-6,
        seed=42,
    )

    perf = results["eval_performance"]
    diags = results["diagnostics"]
    death = results["death_results"]
    triggered = results["triggered_deaths"]

    print("\n=== EVALUATION RESULTS ===")
    for name, p in sorted(perf.items(), key=lambda x: x[1], reverse=True):
        marker = " ←" if name == "BirkhoffTransition" else ""
        print(f"  {name}: {p:.4f}{marker}")

    print("\n=== STABILITY DIAGNOSTICS ===")
    for name in ["Unconstrained", "SimplexWeight", "BirkhoffTransition"]:
        d = diags.get(name, StabilityDiagnostics())
        print(f"  {name}:")
        print(f"    entropy={d.weight_entropy:.3f}  max_w={d.max_weight:.3f}  "
              f"turnover={d.weight_turnover_rate:.4f}  oscillation={d.oscillation_energy:.4f}  "
              f"collapse={d.weight_collapse_ratio:.3f}")

    print("\n=== DEATH CONDITIONS ===")
    all_ok = True
    for k, (trig, detail) in death.items():
        status = "❌ TRIGGERED" if trig else "✅ OK"
        if trig:
            all_ok = False
        print(f"  {k}: {status} — {detail}")

    verdict = "IC3_0_MANIFOLD_CONSTRAINED_ALLOCATOR_SUPPORTED" if all_ok else \
              "IC3_0_MANIFOLD_CONSTRAINED_ALLOCATOR_HAS_DEATH_CONDITIONS"

    print(f"\n  VERDICT: {verdict}")

    # Generate report
    report = generate_report(results, verdict)
    report_path = os.path.join(OUTDIR, "IC3_0_MANIFOLD_CONSTRAINED_ALLOCATOR_AUDIT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report written to {report_path}")

    # Save CSV
    diag_rows = []
    for name in ["Unconstrained", "SimplexWeight", "BirkhoffTransition"]:
        d = diags.get(name, StabilityDiagnostics())
        row = {"allocator": name}
        row.update(d.to_dict())
        diag_rows.append(row)
    pd.DataFrame(diag_rows).to_csv(os.path.join(OUTDIR, "stability_diagnostics.csv"), index=False)

    perf_rows = [{"allocator": name, "mean_reward": p, "regret": results["oracle_upper_bound"] - p}
                 for name, p in perf.items()]
    pd.DataFrame(perf_rows).to_csv(os.path.join(OUTDIR, "allocator_performance.csv"), index=False)

    death_rows = [{"condition": k, "triggered": v, "detail": d}
                  for k, (v, d) in death.items()]
    pd.DataFrame(death_rows).to_csv(os.path.join(OUTDIR, "death_conditions.csv"), index=False)

    # Save weight traces
    for name, wh in results["weight_histories"].items():
        if wh is not None and len(wh) > 0:
            df = pd.DataFrame(wh, columns=[f"capital_{i}" for i in range(wh.shape[1])])
            df["step"] = range(len(df))
            safe_name = name.replace(" ", "_")
            df.to_csv(os.path.join(OUTDIR, f"weight_trace_{safe_name}.csv"), index=False)

    print("\n" + "=" * 60)
    print(f"IC-3-0 COMPLETE — Verdict: {verdict}")
    print("=" * 60)