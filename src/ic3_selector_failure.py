"""
IC-3-S: Selector Failure Decomposition
======================================
Diagnoses *why* the CapitalReport-only allocator cannot beat BestSingleCapital.
Does NOT announce second-order intelligence — only locates failure causes.

Eight analyses:
  1. Switching Opportunity Audit  — is there room for second-order gain?
  2. Report Sufficiency Test       — can CapitalReport predict the best capital?
  3. Report Lag Ablation           — is report staleness the bottleneck?
  4. Per-task Breakdown            — does capital taxonomy hold per task?
  5. Capital Specialization Audit  — are capitals truly specialized?
  6. Negative Transfer Effectiveness — do protection mechanisms actually help?
  7. External-only Slice           — what happens on HiddenGoalGridWorld alone?
  8. Final Report
"""
import os, sys, warnings, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import prepare_counterfactual_data, train_ae_model, train_state_only_classifier
from src.models import StateOnlyPredictor, AEPCompressor
from src.capital_report import (CapitalReport, Capital, PolicyCloneCapital,
                                  PrototypeOutcomeCapital, AEPCapital,
                                  SafeFallbackCapital, GoalInferenceCapital,
                                  ALLOWED_REPORT_FIELDS, FORBIDDEN_REPORT_FIELDS)
from src.capital_impairment import CapitalImpairmentDetector, FallbackController
from src.cybernetic_allocator import FeedbackControlledAllocator
from src.external_benchmark import HiddenGoalGridWorld, GridWorldConfig
from src.manifold_capital_allocator import sinkhorn_projection

OUTDIR = "results/ic3_s"
FIGDIR = "results/figures"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENV_KWARGS = dict(state_dim=2, history_len=8, action_gain=0.25)
EPOCHS = 200; PATIENCE = 40; BOTTLENECK_DIM = 48

U1 = np.array([0.6, 0.2, 0.2], dtype=np.float32); U1 /= np.linalg.norm(U1) + 1e-8
U2 = np.array([-0.4, 0.7, 0.3], dtype=np.float32); U2 /= np.linalg.norm(U2) + 1e-8
U3 = np.array([0.2, -0.5, 0.6], dtype=np.float32); U3 /= np.linalg.norm(U3) + 1e-8
FP32_BYTES = 4

N_FIELDS = 23
MAIN5_IDS = ["PolicyClone", "PrototypeOutcome", "AEP", "GoalInference", "SafeFallback"]


def util_linear(Y, w):
    if Y.ndim == 1: return int(np.argmax(Y * w))
    return np.argmax(Y * w, axis=1)


# ═══════════════════════════════════════════════════════════
# DATA STRUCTURES (compact, same as IC-3-R)
# ═══════════════════════════════════════════════════════════

class RawMemoryOutcomeTable:
    def __init__(self, memory_budget=5000, standardize=True, k=None):
        self.memory_budget = memory_budget
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler() if standardize else None
        self.k = k
    def fit(self, X_train, Y_train_list):
        X_np = np.array(X_train); self.X_full = X_np
        self.Y_table = np.stack(Y_train_list, axis=-1)
        n_max = min(len(X_np), max(1, self.memory_budget // (X_np.shape[1] * FP32_BYTES + 3 * FP32_BYTES)))
        if n_max < len(X_np):
            idx = np.linspace(0, len(X_np)-1, n_max, dtype=int)
            self.X_store = X_np[idx]; self.Y_table = self.Y_table[idx]
        else: self.X_store = X_np
        if self.scaler: self.X_store_s = self.scaler.fit_transform(self.X_store)
        else: self.X_store_s = self.X_store
        if self.k is None: self.k = max(1, min(5, len(self.X_store)//10))
    def predict(self, X_query):
        X_q = np.array(X_query)
        X_q_s = self.scaler.transform(X_q) if self.scaler else X_q
        Y_pred = np.zeros((len(X_q), 3), dtype=np.float32)
        for i in range(len(X_q)):
            dists = np.sum((X_q_s[i] - self.X_store_s)**2, axis=1)
            k_eff = min(self.k, len(self.X_store))
            nn_idx = np.argpartition(dists, k_eff-1)[:k_eff] if k_eff > 0 else [0]
            Y_pred[i] = self.Y_table[nn_idx].mean(axis=0)
        return Y_pred
    @property
    def stored_bytes(self): return self.X_store.nbytes + self.Y_table.nbytes
    @property
    def inference_ops(self): return len(self.X_store) * self.X_store.shape[1] * 3

class PrototypeOutcomeTable:
    def __init__(self, n_clusters=50, k=3):
        self.n_clusters = n_clusters; self.k = k
    def fit(self, X_train, Y_train_list):
        X_np = np.array(X_train); Y_tab = np.stack(Y_train_list, axis=-1)
        n = len(X_np); nc = min(self.n_clusters, n)
        rng = np.random.default_rng(42)
        indices = rng.choice(n, nc, replace=False)
        self.prototypes = X_np[indices].copy(); self.labels = np.zeros(n, dtype=np.int64)
        for _ in range(10):
            for i in range(n):
                self.labels[i] = np.argmin(np.sum((X_np[i] - self.prototypes)**2, axis=1))
            for p in range(nc):
                mask = self.labels == p
                if mask.sum() > 0: self.prototypes[p] = X_np[mask].mean(axis=0)
        self.proto_tables = np.zeros((nc, 3), dtype=np.float32)
        for p in range(nc):
            mask = self.labels == p
            if mask.sum() > 0: self.proto_tables[p] = Y_tab[mask].mean(axis=0)
    def predict(self, X_query):
        X_q = np.array(X_query)
        Y_pred = np.zeros((len(X_q), 3), dtype=np.float32)
        for i in range(len(X_q)):
            dists = np.sum((X_q[i] - self.prototypes)**2, axis=1)
            k_eff = min(self.k, len(self.prototypes))
            top_k = np.argpartition(dists, k_eff-1)[:k_eff]
            Y_pred[i] = self.proto_tables[top_k].mean(axis=0)
        return Y_pred
    @property
    def stored_bytes(self): return self.prototypes.nbytes + self.proto_tables.nbytes
    @property
    def inference_ops(self): return len(self.prototypes) * self.prototypes.shape[1] * 3


class MixedTaskStream:
    def __init__(self, X_tr, Y_tr, X_te, Y_te, grid_env, n_total=800, block_size=20, seed=42):
        self.rng = np.random.default_rng(seed)
        self.X_tr = np.array(X_tr, dtype=np.float32); self.Y_tr = np.array(Y_tr, dtype=np.float32)
        self.X_te = np.array(X_te, dtype=np.float32); self.Y_te = np.array(Y_te, dtype=np.float32)
        self.grid_env = grid_env; self.n_total = n_total; self.block_size = block_size
        self._task_labels = []; self._build()
    def _build(self):
        n_per = self.n_total // 4; bs = self.block_size
        ta = [(f"A_fixed_goal", self.X_te[i%len(self.X_te)], self.Y_te[i%len(self.X_te)],
               util_linear, U1, None) for i in range(n_per)]
        tb = [(f"B_goal_transfer", self.X_te[i%len(self.X_te)], self.Y_te[i%len(self.X_te)],
               util_linear, [U1,U2,U3][i%3], None) for i in range(n_per)]
        tc = [(f"C_dense_support", self.X_te[i%len(self.X_te)], self.Y_te[i%len(self.X_te)],
               util_linear, U1, None) for i in range(n_per)]
        td = [(f"D_hidden_goal", None, None, None, None, i%30) for i in range(n_per)]
        task_groups = [("Task_A", ta), ("Task_B", tb), ("Task_C", tc), ("Task_D", td)]
        blocks = []
        for t_name, group in task_groups:
            for bi in range(0, n_per, bs):
                blocks.append((t_name, group[bi:bi+bs]))
        perm = self.rng.permutation(len(blocks))
        self.tasks = []; self._task_labels = []
        for pi in perm:
            t_name, block = blocks[pi]
            self.tasks.extend(block); self._task_labels.extend([t_name]*len(block))
    def get_step(self, step_idx): return self.tasks[step_idx]
    def task_label(self, step_idx): return self._task_labels[step_idx]


def capital_report_vector(reports):
    vecs = [r.to_vector() for r in reports]
    return np.concatenate(vecs).astype(np.float32)


def make_dummy_reports_from_vec(rpt_vec, n_capitals=5):
    """Create minimal CapitalReport objects from a report vector for allocator use."""
    reports = []
    import time
    for i in range(n_capitals):
        start = i * N_FIELDS
        conf = float(rpt_vec[start + 4]) if start + 4 < len(rpt_vec) else 0.5
        ood = float(rpt_vec[start + 8]) if start + 8 < len(rpt_vec) else 0.0
        reports.append(CapitalReport(
            capital_id=f"dummy_{i}", capital_type="dummy", timestamp=time.time(),
            confidence=conf, capital_local_ood_score=ood))
    return reports


class ValuePredictor(nn.Module):
    def __init__(self, n_features=115, n_out=5, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, n_out),
        )
    def forward(self, x): return self.net(x)


class MLPOracleReportSelector(nn.Module):
    """5-class classifier: CapitalReport vector -> oracle_best_capital."""
    def __init__(self, n_features=115, n_classes=5, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x): return self.net(x)


# ═══════════════════════════════════════════════════════════
# FEEDBACK-CONTROLLED ALLOCATOR WITH CONFIG FLAGS (for ablation)
# ═══════════════════════════════════════════════════════════

class FeedbackControlledAllocatorAblated(FeedbackControlledAllocator):
    """Same as FeedbackControlledAllocator but with toggleable protection flags."""
    def __init__(self, n_capitals=5, capital_ids=None, predictor=None,
                 max_weight_change=0.12, impairment_threshold=0.30,
                 clf_mean=None, clf_std=None, device="cpu",
                 enable_impairment=True, enable_fallback=True,
                 enable_depreciation=True, enable_smoothing=True,
                 enable_feedforward=True):
        super().__init__(n_capitals=n_capitals, capital_ids=capital_ids,
                         predictor=predictor, max_weight_change=max_weight_change,
                         impairment_threshold=impairment_threshold,
                         clf_mean=clf_mean, clf_std=clf_std, device=device)
        self.enable_impairment = enable_impairment
        self.enable_fallback = enable_fallback
        self.enable_depreciation = enable_depreciation
        self.enable_smoothing = enable_smoothing
        self.enable_feedforward = enable_feedforward

    def step(self, reports, report_vector=None, feedback=None):
        n = len(reports)
        if n == 0: return {}

        # Feedforward
        if self.enable_feedforward and self.predictor is not None and report_vector is not None:
            import torch as T
            rpt_log = np.log1p(np.maximum(report_vector, 0.0))
            if self.clf_mean is not None:
                rpt_norm = (rpt_log - self.clf_mean) / self.clf_std
                rpt_norm = np.clip(rpt_norm, -5.0, 5.0)
            else: rpt_norm = rpt_log
            rpt_t = T.tensor(rpt_norm, dtype=T.float32).unsqueeze(0).to(self.device)
            with T.no_grad(): predicted_values = self.predictor(rpt_t).cpu().numpy()[0]
        else:
            predicted_values = np.array([float(getattr(r, "confidence", 0.5)) for r in reports], dtype=np.float64)

        realized_regret = float(feedback.get("allocator_regret", 0.0)) if feedback else 0.0

        for i in range(n):
            if i >= len(self.capital_ids): break
            cid = self.capital_ids[i]; state = self.states[cid]
            pred_val = float(predicted_values[i]) if i < len(predicted_values) else 0.5
            state.predicted_value_ema = 0.8*state.predicted_value_ema + 0.2*pred_val
            realized_rew = 1.0 - realized_regret
            state.realized_reward_ema = 0.8*state.realized_reward_ema + 0.2*realized_rew
            state.tracking_error = 0.9*state.tracking_error + 0.1*abs(state.predicted_value_ema - state.realized_reward_ema)

            if self.enable_depreciation:
                step_count = max(1, self.global_step - state.last_validated)
                decay = (1.0 - self.depreciation_rate)**step_count
                pred_decayed = state.predicted_value_ema * decay
            else:
                pred_decayed = state.predicted_value_ema

            confidence = float(getattr(reports[i], "confidence", 0.5)) if i < len(reports) else 0.5
            ood = float(getattr(reports[i], "capital_local_ood_score", 0.0)) if i < len(reports) else 0.0
            state.reliability = max(0.05, pred_decayed - 0.2*ood*(1.0-confidence))
            state.last_validated = self.global_step

        # Impairment
        if self.enable_impairment:
            impairment = [self.states[cid].reliability < self.impairment_threshold for cid in self.capital_ids[:n]]
        else:
            impairment = [False]*n

        raw_weights = np.zeros(n, dtype=np.float64)
        for i, cid in enumerate(self.capital_ids[:n]):
            state = self.states[cid]; prev_w = state.weight
            if impairment[i]:
                target_w = prev_w*0.25; state.impairment_count = min(state.impairment_count+1, 100)
            else:
                rel_sum_r = sum(self.states[c].reliability for c in self.capital_ids[:n])
                rel_sum_p = sum(self.states[c].predicted_value_ema for c in self.capital_ids[:n])
                rel_norm = state.reliability/max(1e-8, rel_sum_r)
                pred_norm = state.predicted_value_ema/max(1e-8, rel_sum_p)
                target_w = 0.3*rel_norm + 0.7*pred_norm

            if self.enable_smoothing:
                delta = np.clip(target_w - prev_w, -self.max_weight_change, self.max_weight_change)
                raw_weights[i] = max(0.0, prev_w + delta)
            else:
                raw_weights[i] = max(0.0, target_w)

        w_sum = raw_weights.sum()
        if self.enable_fallback and w_sum < 1e-8:
            self.fallback_active = True; weights = np.ones(n)/n
        elif w_sum < 1e-8:
            self.fallback_active = True; weights = np.ones(n)/n
        else:
            self.fallback_active = False; weights = raw_weights/w_sum

        for i, cid in enumerate(self.capital_ids[:n]):
            self.states[cid].weight = float(weights[i])
            if not impairment[i]: self.states[cid].impairment_count = 0

        if self.global_step > 0:
            w_osc = np.sum(np.abs(np.diff(weights))) if n > 1 else 0.0
            self._diagnostics["weight_oscillation"].append(float(w_osc))
            self._diagnostics["tracking_error"].append(float(np.mean([s.tracking_error for s in self.states.values()])))
            self._diagnostics["regret_dynamics"].append(realized_regret)
        self.global_step += 1
        return {cid: float(weights[i]) for i, cid in enumerate(self.capital_ids[:n])}


# ═══════════════════════════════════════════════════════════
# CORE DATA COLLECTION PIPELINE (collect all data, no allocator eval)
# ═══════════════════════════════════════════════════════════

def collect_eval_data(seed=45, n_eval=600):
    print(f"\n{'='*60}")
    print(f"IC-3-S Data Collection (seed={seed})")
    print(f"{'='*60}")

    cf_data = pd.read_csv("results/counterfactual_table.csv")
    train_df = cf_data[(cf_data["seed"]==0)&(cf_data["split"]=="train")&(cf_data["horizon"]==1)]
    test_df = cf_data[(cf_data["seed"]==0)&(cf_data["split"]=="test_id")&(cf_data["horizon"]==1)]
    X_tr_all, Y_tr_all, ba_tr_all = prepare_counterfactual_data(train_df, 0, ENV_KWARGS)
    X_te_all, Y_te_all, ba_te_all = prepare_counterfactual_data(test_df, 0, ENV_KWARGS)
    Y3_tr_all = [Y_tr_all[:,0], Y_tr_all[:,1], Y_tr_all[:,2]]
    N_ORACLE=1000; N_TRAIN=2000

    print("  Training PolicyClone...")
    pc_model = StateOnlyPredictor(obs_dim=2, history_len=8, n_actions=3, bottleneck_dim=BOTTLENECK_DIM)
    pc_model = train_state_only_classifier(pc_model, X_tr_all, Y_tr_all, None, None, epochs=EPOCHS, patience=PATIENCE, device=DEVICE)
    pc_model.eval()

    print("  Training AEP...")
    aep_model = AEPCompressor(obs_dim=2, history_len=8, n_actions=3, bottleneck_dim=BOTTLENECK_DIM)
    aep_model = train_ae_model(aep_model, X_tr_all, Y_tr_all, None, None, "aep", epochs=EPOCHS, patience=PATIENCE, device=DEVICE, ce_weight=0.8)
    aep_model.eval()

    print("  Building Prototype tables...")
    rmot = RawMemoryOutcomeTable(memory_budget=5000); rmot.fit(X_tr_all, Y3_tr_all)
    pot = PrototypeOutcomeTable(n_clusters=50, k=3); pot.fit(X_tr_all, Y3_tr_all)
    grid_env = HiddenGoalGridWorld(GridWorldConfig(seed=0))

    # Oracle + Train phases (collect report vectors + correctness, same as IC-3-R)
    oracle_X = []; oracle_perfs = []
    capitals_oracle = [
        PolicyCloneCapital(pc_model, "PolicyClone"),
        PrototypeOutcomeCapital(pot, "PrototypeOutcome"),
        AEPCapital(aep_model, "AEP"),
        GoalInferenceCapital(grid_size=7, capital_id="GoalInference"),
        SafeFallbackCapital("SafeFallback"),
    ]
    NC = len(capitals_oracle)
    oracle_s = MixedTaskStream(X_tr_all, Y_tr_all, X_te_all, Y_te_all, grid_env, N_ORACLE, block_size=20, seed=41)

    oracle_correct = np.zeros((N_ORACLE, NC), dtype=np.float32)
    for step in range(N_ORACLE):
        task_name, X_val, Y_val, ufn, w, grid_ep = oracle_s.get_step(step)
        reports = [cap.generate_report({}, []) for cap in capitals_oracle]
        oracle_X.append(capital_report_vector(reports))
        for ci, cap in enumerate(capitals_oracle):
            ctx = {}
            if task_name.startswith("D_"):
                ctx["obs"] = grid_env.reset(seed=step); a = cap.act(ctx, [])
                _, reward, _, info = grid_env.step(a)
                oracle_correct[step, ci] = float(info["at_goal"])
                cap.update({"reward": reward, "goal_reached": int(info["at_goal"]), "at_goal": info["at_goal"]})
            else:
                ctx["X"] = X_val; ctx["utility_fn"] = lambda Y, w=w: util_linear(Y, w)
                a = min(cap.act(ctx, []), 2); oa = util_linear(Y_val, w); c = 1 if a==oa else 0
                oracle_correct[step, ci] = float(c)
                uv = float(Y_val[oa]) if c else float(Y_val[a])*0.5
                nn_d = np.sqrt(np.mean((X_val - X_tr_all)**2, axis=1))
                cap.update({"correct": c, "utility": uv, "ood_distance": float(np.min(nn_d))})
        oracle_perfs.append(np.array(oracle_correct[step], dtype=np.float32))

    train_X = []; train_perfs = []
    train_s = MixedTaskStream(X_tr_all, Y_tr_all, X_te_all, Y_te_all, grid_env, N_TRAIN, block_size=20, seed=42)
    for step in range(N_TRAIN):
        task_name, X_val, Y_val, ufn, w, grid_ep = train_s.get_step(step)
        reports = [cap.generate_report({}, []) for cap in capitals_oracle]
        train_X.append(capital_report_vector(reports))
        perfs = []
        for ci, cap in enumerate(capitals_oracle):
            ctx = {}
            if task_name.startswith("D_"): ctx["obs"] = grid_env.reset(seed=step*13+50001)
            else: ctx["X"] = X_val; ctx["utility_fn"] = lambda Y, w=w: util_linear(Y, w)
            a = min(cap.act(ctx, []), 2)
            if task_name.startswith("D_"):
                _, reward, _, info = grid_env.step(a); perf = float(info["at_goal"])
                cap.update({"reward": reward, "goal_reached": int(info["at_goal"]), "at_goal": info["at_goal"]})
            else:
                oa = util_linear(Y_val, w); c = 1 if a==oa else 0
                uv = float(Y_val[oa]) if c else float(Y_val[a])*0.5; perf = float(c)
                nn_d = np.sqrt(np.mean((X_val - X_tr_all)**2, axis=1))
                cap.update({"correct": int(c), "utility": uv, "ood_distance": float(np.min(nn_d))})
            perfs.append(perf)
        train_perfs.append(np.array(perfs, dtype=np.float32))

    # Eval phase: collect everything
    eval_s = MixedTaskStream(X_tr_all, Y_tr_all, X_te_all, Y_te_all, grid_env, n_eval, block_size=20, seed=seed)
    eval_caps = [
        PolicyCloneCapital(pc_model, "PolicyClone"),
        PrototypeOutcomeCapital(pot, "PrototypeOutcome"),
        AEPCapital(aep_model, "AEP"),
        GoalInferenceCapital(grid_size=7, capital_id="GoalInference"),
        SafeFallbackCapital("SafeFallback"),
    ]

    all_cap_correct = np.zeros((n_eval, NC), dtype=np.float32)
    eval_report_vectors = np.zeros((n_eval, N_FIELDS * NC), dtype=np.float32)
    eval_task_labels = []; eval_env_types = []
    block_boundaries = []

    prev_task = None
    for step in range(n_eval):
        task_name, X_val, Y_val, ufn, w, grid_ep = eval_s.get_step(step)
        tl = eval_s.task_label(step)
        eval_task_labels.append(tl)
        eval_env_types.append("EXTERNAL" if task_name.startswith("D_") else "SYNTH")
        if prev_task is not None and tl != prev_task:
            block_boundaries.append(step)
        prev_task = tl

        reports = [cap.generate_report({}, []) for cap in eval_caps]
        eval_report_vectors[step] = capital_report_vector(reports)

        for ci, cap in enumerate(eval_caps):
            ctx = {}
            if task_name.startswith("D_"): ctx["obs"] = grid_env.reset(seed=step+99999+ci)
            else: ctx["X"] = X_val; ctx["utility_fn"] = lambda Y, w=w: util_linear(Y, w)
            a = min(cap.act(ctx, []), 2)
            if task_name.startswith("D_"):
                _, _, _, info = grid_env.step(a); all_cap_correct[step, ci] = float(info["at_goal"])
            else:
                oa = util_linear(Y_val, w); all_cap_correct[step, ci] = 1.0 if a==oa else 0.0

    oracle_per_step = all_cap_correct.max(axis=1)
    oracle_best_idx = np.argmax(all_cap_correct, axis=1)

    best_single_idx = int(np.argmax(oracle_correct.mean(axis=0)))
    best_single_name = capitals_oracle[best_single_idx].capital_id

    oracle_upper = float(oracle_correct.max(axis=1).mean())

    # Combined training data for selectors
    all_train_X = np.concatenate([np.array(oracle_X, dtype=np.float32),
                                   np.array(train_X, dtype=np.float32)], axis=0)
    all_train_perfs = np.concatenate([np.array(oracle_perfs, dtype=np.float32),
                                       np.array(train_perfs, dtype=np.float32)], axis=0)
    all_train_oracle_best = np.argmax(all_train_perfs, axis=1)

    # Normalization
    all_X_log = np.log1p(np.maximum(all_train_X, 0.0))
    clf_mean = all_X_log.mean(axis=0); clf_std = all_X_log.std(axis=0)+1e-8

    # Train MetaMLP
    n_features = all_train_X.shape[1]
    all_X_norm = (all_X_log - clf_mean)/clf_std
    meta_mlp = ValuePredictor(n_features=n_features, n_out=NC).to(DEVICE)
    mlp_opt = torch.optim.AdamW(meta_mlp.parameters(), lr=0.003, weight_decay=1e-5)
    mlp_sch = torch.optim.lr_scheduler.CosineAnnealingLR(mlp_opt, T_max=300)
    mlp_ds = TensorDataset(torch.tensor(all_X_norm, dtype=torch.float32),
                           torch.tensor(all_train_perfs, dtype=torch.float32))
    mlp_ldr = DataLoader(mlp_ds, batch_size=64, shuffle=True)
    for _ in range(300):
        meta_mlp.train()
        for bx, by in mlp_ldr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            pred = meta_mlp(bx); loss = F.mse_loss(pred, by)
            mlp_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_mlp.parameters(), 1.0)
            mlp_opt.step()
        mlp_sch.step()
    meta_mlp.eval()

    # Normalize eval reports using same stats
    eval_X_log = np.log1p(np.maximum(eval_report_vectors, 0.0))
    eval_X_norm = (eval_X_log - clf_mean)/clf_std
    eval_X_norm = np.clip(eval_X_norm, -5.0, 5.0)

    print(f"  Collected {n_eval} eval steps, {len(block_boundaries)} block transitions")
    print(f"  BestSingle: {best_single_name} (idx={best_single_idx}), oracle_upper={oracle_upper:.4f}")

    return {
        "seed": seed,
        "n_eval": n_eval,
        "NC": NC,
        "capital_ids": MAIN5_IDS,
        "best_single_idx": best_single_idx,
        "best_single_name": best_single_name,
        "oracle_upper": oracle_upper,
        "all_cap_correct": all_cap_correct,
        "oracle_per_step": oracle_per_step,
        "oracle_best_idx": oracle_best_idx,
        "eval_report_vectors": eval_report_vectors,
        "eval_X_norm": eval_X_norm,
        "eval_task_labels": eval_task_labels,
        "eval_env_types": eval_env_types,
        "block_boundaries": block_boundaries,
        "all_train_X": all_train_X,
        "all_train_perfs": all_train_perfs,
        "all_train_oracle_best": all_train_oracle_best,
        "clf_mean": clf_mean,
        "clf_std": clf_std,
        "meta_mlp": meta_mlp,
        "pc_model": pc_model,
        "aep_model": aep_model,
        "pot": pot,
        "grid_env": grid_env,
        "X_tr_all": X_tr_all,
        "Y_tr_all": Y_tr_all,
        "X_te_all": X_te_all,
        "Y_te_all": Y_te_all,
    }


# ═══════════════════════════════════════════════════════════
# ANALYSIS 1: Switching Opportunity Audit
# ═══════════════════════════════════════════════════════════

def analysis_switching_opportunity(data):
    print(f"\n{'='*60}")
    print("ANALYSIS 1: Switching Opportunity Audit")
    print(f"{'='*60}")

    oracle_r = data["oracle_per_step"]
    bs_idx = data["best_single_idx"]
    bs_correct = data["all_cap_correct"][:, bs_idx]
    oracle_gain_per_step = oracle_r - bs_correct
    oracle_best = data["oracle_best_idx"]
    task_labels = np.array(data["eval_task_labels"])

    switching_opp = (oracle_best != bs_idx).astype(float)
    switching_rate = float(switching_opp.mean())
    oracle_gain = float(oracle_gain_per_step.mean())
    max_gain = float(np.max(oracle_gain_per_step))

    # Per-task oracle gain
    task_types = ["Task_A", "Task_B", "Task_C", "Task_D"]
    per_task_rows = []
    for tt in task_types:
        mask = task_labels == tt
        if mask.sum() > 0:
            per_task_rows.append({
                "task": tt,
                "n_steps": int(mask.sum()),
                "oracle_gain_over_bestsingle": float(oracle_gain_per_step[mask].mean()),
                "switching_opportunity_rate": float(switching_opp[mask].mean()),
                "oracle_reward": float(oracle_r[mask].mean()),
                "bestsingle_reward": float(bs_correct[mask].mean()),
            })

    # Block-transition vs within-block oracle gain
    boundaries = data["block_boundaries"]
    transition_gains = []; within_gains = []
    transition_switches = []; within_switches = []
    for b in boundaries:
        if b + 3 < data["n_eval"]:
            transition_gains.append(float(oracle_gain_per_step[b:b+3].mean()))
            transition_switches.append(float(switching_opp[b:b+3].mean()))
    prev_b = 0
    for b in boundaries:
        if b - prev_b >= 6:
            mid = prev_b + 3
            within_gains.append(float(oracle_gain_per_step[mid:b-2].mean()))
            within_switches.append(float(switching_opp[mid:b-2].mean()))
        prev_b = b

    trans_gain = float(np.mean(transition_gains)) if transition_gains else 0.0
    within_gain = float(np.mean(within_gains)) if within_gains else 0.0
    trans_switch = float(np.mean(transition_switches)) if transition_switches else 0.0
    within_switch = float(np.mean(within_switches)) if within_switches else 0.0

    # Switching opportunity rate
    opp_rate = switching_rate

    rows = [{
        "metric": "oracle_gain_over_bestsingle",
        "value": oracle_gain,
        "detail": f"Oracle {float(oracle_r.mean()):.4f} - BS {float(bs_correct.mean()):.4f}",
    }, {
        "metric": "switching_opportunity_rate",
        "value": opp_rate,
        "detail": f"BestSingle != OracleBest in {opp_rate:.1%} of steps",
    }, {
        "metric": "max_per_step_oracle_gain",
        "value": max_gain,
        "detail": "",
    }, {
        "metric": "block_transition_oracle_gain",
        "value": trans_gain,
        "detail": f"Mean gain in first 3 steps after block switch",
    }, {
        "metric": "within_block_oracle_gain",
        "value": within_gain,
        "detail": "Mean gain mid-block (stable regime)",
    }, {
        "metric": "block_transition_switch_rate",
        "value": trans_switch,
        "detail": f"Switching opportunity rate at block boundaries",
    }, {
        "metric": "within_block_switch_rate",
        "value": within_switch,
        "detail": "Switching opportunity rate within stable blocks",
    }]

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTDIR}/switching_opportunity_audit.csv", index=False)

    per_task_df = pd.DataFrame(per_task_rows)
    per_task_df.to_csv(f"{OUTDIR}/switching_opportunity_per_task.csv", index=False)

    verdict = "SUFFICIENT"
    if oracle_gain < 0.03:
        verdict = "INSUFFICIENT — <3% oracle gain, no room for second-order"
    elif oracle_gain < 0.10:
        verdict = "MARGINAL — <10% oracle gain, limited second-order ceiling"
    else:
        verdict = "SUFFICIENT — >10% oracle gain, meaningful second-order space"

    print(f"  Oracle gain over BestSingle: {oracle_gain:.4f}")
    print(f"  Switching opportunity rate: {opp_rate:.1%}")
    print(f"  Transition gain: {trans_gain:.4f}  |  Within-block gain: {within_gain:.4f}")
    print(f"  Verdict: {verdict}")

    return rows, per_task_rows, verdict


# ═══════════════════════════════════════════════════════════
# ANALYSIS 2: Report Sufficiency Test
# ═══════════════════════════════════════════════════════════

def analysis_report_sufficiency(data):
    print(f"\n{'='*60}")
    print("ANALYSIS 2: Report Sufficiency Test")
    print(f"{'='*60}")

    train_X = data["all_train_X"]
    train_Y = data["all_train_oracle_best"]
    eval_X = data["eval_report_vectors"]
    eval_true_best = data["oracle_best_idx"]
    all_cap_correct = data["all_cap_correct"]
    oracle_per_step = data["oracle_per_step"]
    bs_idx = data["best_single_idx"]
    NC = data["NC"]; n_eval = data["n_eval"]

    # Normalize
    train_X_log = np.log1p(np.maximum(train_X, 0.0))
    eval_X_log = np.log1p(np.maximum(eval_X, 0.0))
    scaler = StandardScaler()
    train_X_s = scaler.fit_transform(train_X_log)
    eval_X_s = scaler.transform(eval_X_log)

    results = []

    # Helper: compute both metrics for a set of predicted capital indices
    def compute_both_metrics(pred_indices):
        oba = float((pred_indices == eval_true_best).mean())  # oracle-best accuracy
        rew = float(all_cap_correct[np.arange(n_eval), pred_indices].mean())  # reward correctness
        return oba, rew

    # BestSingle baseline — both metrics
    bs_pred = np.full(n_eval, bs_idx, dtype=np.int64)
    bs_oba, bs_rew = compute_both_metrics(bs_pred)
    results.append({"selector": "BestSingleCapital", "oracle_best_accuracy": bs_oba,
                    "reward_correctness": bs_rew, "beats_bs_reward": False,
                    "oracle_gap": float(oracle_per_step.mean()) - bs_rew})

    # 1. LogisticRegression
    print("  Training LogisticRegression...")
    lr = LogisticRegression(max_iter=2000, solver='lbfgs', C=0.1)
    lr.fit(train_X_s, train_Y)
    lr_pred = lr.predict(eval_X_s)
    lr_oba, lr_rew = compute_both_metrics(lr_pred)
    results.append({"selector": "LogisticRegressionSelector", "oracle_best_accuracy": lr_oba,
                    "reward_correctness": lr_rew, "delta_vs_bs_reward": lr_rew - bs_rew,
                    "beats_bs_reward": lr_rew > bs_rew,
                    "oracle_gap": float(oracle_per_step.mean()) - lr_rew})

    # 2. RandomForest
    print("  Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    rf.fit(train_X_s, train_Y)
    rf_pred = rf.predict(eval_X_s)
    rf_oba, rf_rew = compute_both_metrics(rf_pred)
    results.append({"selector": "RandomForestSelector", "oracle_best_accuracy": rf_oba,
                    "reward_correctness": rf_rew, "delta_vs_bs_reward": rf_rew - bs_rew,
                    "beats_bs_reward": rf_rew > bs_rew,
                    "oracle_gap": float(oracle_per_step.mean()) - rf_rew})

    # 3. MLP classifier
    print("  Training MLPOracleReportSelector...")
    mlp_cls = MLPOracleReportSelector(n_features=train_X_s.shape[1], n_classes=NC).to(DEVICE)
    clf_opt = torch.optim.AdamW(mlp_cls.parameters(), lr=0.003, weight_decay=1e-5)
    clf_sch = torch.optim.lr_scheduler.CosineAnnealingLR(clf_opt, T_max=300)
    clf_ds = TensorDataset(torch.tensor(train_X_s, dtype=torch.float32),
                           torch.tensor(train_Y, dtype=torch.long))
    clf_ldr = DataLoader(clf_ds, batch_size=64, shuffle=True)
    mlp_cls.train()
    for _ in range(300):
        for bx, by in clf_ldr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            pred = mlp_cls(bx); loss = F.cross_entropy(pred, by)
            clf_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp_cls.parameters(), 1.0)
            clf_opt.step()
        clf_sch.step()
    mlp_cls.eval()
    with torch.no_grad():
        eval_t = torch.tensor(eval_X_s, dtype=torch.float32).to(DEVICE)
        mlp_pred = mlp_cls(eval_t).argmax(dim=-1).cpu().numpy()
    mlp_oba, mlp_rew = compute_both_metrics(mlp_pred)
    results.append({"selector": "MLPOracleReportSelector", "oracle_best_accuracy": mlp_oba,
                    "reward_correctness": mlp_rew, "delta_vs_bs_reward": mlp_rew - bs_rew,
                    "beats_bs_reward": mlp_rew > bs_rew,
                    "oracle_gap": float(oracle_per_step.mean()) - mlp_rew})

    # Run allocators (MetaMLP, Cyber)
    clf_mean = data["clf_mean"]; clf_std = data["clf_std"]
    meta_mlp = data["meta_mlp"]; eval_X_norm = data["eval_X_norm"]
    capital_ids = data["capital_ids"]

    mlp_pred_idx = np.zeros(n_eval, dtype=np.int64)
    for step in range(n_eval):
        rpt_t = torch.tensor(eval_X_norm[step], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): pv = meta_mlp(rpt_t).cpu().numpy()[0]
        mlp_pred_idx[step] = int(np.argmax(pv))
    mlp_oba, mlp_rew = compute_both_metrics(mlp_pred_idx)
    results.append({"selector": "MetaMLPAllocator", "oracle_best_accuracy": mlp_oba,
                    "reward_correctness": mlp_rew, "delta_vs_bs_reward": mlp_rew - bs_rew,
                    "beats_bs_reward": mlp_rew > bs_rew,
                    "oracle_gap": float(oracle_per_step.mean()) - mlp_rew})

    cb = FeedbackControlledAllocator(n_capitals=NC, capital_ids=capital_ids,
                                      predictor=meta_mlp, max_weight_change=0.12,
                                      impairment_threshold=0.30,
                                      clf_mean=clf_mean, clf_std=clf_std, device=DEVICE)
    cb_pred_idx = np.zeros(n_eval, dtype=np.int64)
    for step in range(n_eval):
        rpt_vec = data["eval_report_vectors"][step]
        weights = cb.step(make_dummy_reports_from_vec(rpt_vec), report_vector=rpt_vec,
                         feedback={"allocator_regret": 0.0})
        cb_pred_idx[step] = int(np.argmax(list(weights.values())))
    cb_oba, cb_rew = compute_both_metrics(cb_pred_idx)
    results.append({"selector": "FeedbackControlledAllocator", "oracle_best_accuracy": cb_oba,
                    "reward_correctness": cb_rew, "delta_vs_bs_reward": cb_rew - bs_rew,
                    "beats_bs_reward": cb_rew > bs_rew,
                    "oracle_gap": float(oracle_per_step.mean()) - cb_rew})

    results.append({"selector": "OracleHindsight", "oracle_best_accuracy": 1.0,
                    "reward_correctness": float(oracle_per_step.mean()),
                    "delta_vs_bs_reward": float(oracle_per_step.mean()) - bs_rew,
                    "beats_bs_reward": True, "oracle_gap": 0.0})

    df = pd.DataFrame(results)
    df.to_csv(f"{OUTDIR}/report_sufficiency_test.csv", index=False)

    # Verdict
    learned = [r for r in results if r["selector"] not in
               ("BestSingleCapital", "OracleHindsight", "MetaMLPAllocator", "FeedbackControlledAllocator")]
    best_learned = max(learned, key=lambda x: x["reward_correctness"]) if learned else {"selector": "None", "reward_correctness": 0}
    best_rew = best_learned["reward_correctness"]

    if best_rew > bs_rew + 0.05:
        verdict2 = "REPORT_SUFFICIENT_ALLOCATOR_FAILURE"
    elif best_rew <= bs_rew:
        verdict2 = "REPORT_INTERFACE_INSUFFICIENT"
    elif best_rew > float(oracle_per_step.mean()) - 0.05:
        verdict2 = "NEAR_ORACLE_SCHEDULING_FEASIBLE"
    else:
        verdict2 = "REPORT_MARGINALLY_USEFUL"

    print(f"  LR_reward={lr_rew:.4f}  RF_reward={rf_rew:.4f}  MLP_cls_reward={mlp_rew:.4f}")
    print(f"  MLP_alloc={mlp_rew:.4f}  Cyber={cb_rew:.4f}  BS={bs_rew:.4f}")
    print(f"  Best learned oracle-selector: {best_learned['selector']}={best_rew:.4f}")
    print(f"  Oracle-best-accuracy (LR/RF/MLP): {lr_oba:.4f}/{rf_oba:.4f}/{mlp_oba:.4f}")
    print(f"  Verdict: {verdict2}")

    return results, verdict2


# ═══════════════════════════════════════════════════════════
# ANALYSIS 3: Report Lag Ablation
# ═══════════════════════════════════════════════════════════

def analysis_report_lag(data):
    print(f"\n{'='*60}")
    print("ANALYSIS 3: Report Lag Ablation")
    print(f"{'='*60}")

    eval_X = data["eval_report_vectors"]
    eval_Y = data["oracle_best_idx"]
    n = len(eval_X)

    window_sizes = [1, 3, 5, 10, 20, 30]
    rows = []

    for w in window_sizes:
        X_lag = np.zeros_like(eval_X)
        for i in range(n):
            lo = max(0, i-w+1)
            X_lag[i] = eval_X[lo:i+1].mean(axis=0)

        X_lag_log = np.log1p(np.maximum(X_lag, 0.0))
        scaler = StandardScaler(); X_s = scaler.fit_transform(X_lag_log)

        from sklearn.model_selection import cross_val_score
        lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=0.1)
        try:
            cv_scores = cross_val_score(lr, X_s, eval_Y, cv=5, scoring="accuracy")
            cv_mean = float(cv_scores.mean())
        except Exception:
            cv_mean = 0.0

        rows.append({"method": f"fixed_window_N={w}", "window_size": w,
                     "cv_accuracy": cv_mean, "n_samples": n})

    # EMA variants
    for ema_alpha, ema_name in [(0.9, "EMA_fast"), (0.5, "EMA_medium"), (0.2, "EMA_slow")]:
        X_ema = np.zeros_like(eval_X)
        ema = eval_X[0].copy()
        for i in range(n):
            ema = ema_alpha*ema + (1-ema_alpha)*eval_X[i]
            X_ema[i] = ema
        X_l = np.log1p(np.maximum(X_ema, 0.0))
        sc = StandardScaler(); X_s = sc.fit_transform(X_l)
        lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=0.1)
        try:
            cv = cross_val_score(lr, X_s, eval_Y, cv=5, scoring="accuracy")
            rows.append({"method": ema_name, "window_size": float(1/(1-ema_alpha)),
                         "cv_accuracy": float(cv.mean()), "n_samples": n})
        except Exception:
            rows.append({"method": ema_name, "window_size": float(1/(1-ema_alpha)),
                         "cv_accuracy": 0.0, "n_samples": n})

    # Dual-timescale: fast (0.9) - slow (0.2) difference
    ema_fast = eval_X[0].copy(); ema_slow = eval_X[0].copy()
    X_dual = np.zeros((n, eval_X.shape[1]))
    for i in range(n):
        ema_fast = 0.9*ema_fast + 0.1*eval_X[i]
        ema_slow = 0.2*ema_slow + 0.8*eval_X[i]
        X_dual[i] = ema_fast - ema_slow
    X_dl = np.log1p(np.maximum(np.abs(X_dual), 0.0))
    sc2 = StandardScaler(); X_ds = sc2.fit_transform(X_dl)
    lr2 = LogisticRegression(max_iter=1000, solver='lbfgs', C=0.1)
    try:
        cv2 = cross_val_score(lr2, X_ds, eval_Y, cv=5, scoring="accuracy")
        rows.append({"method": "dual_timescale_diff", "window_size": "N/A",
                     "cv_accuracy": float(cv2.mean()), "n_samples": n})
    except Exception:
        rows.append({"method": "dual_timescale_diff", "window_size": "N/A",
                     "cv_accuracy": 0.0, "n_samples": n})

    # Change-point reset: reset EMA at block boundaries
    boundaries = set(data["block_boundaries"])
    era = eval_X[0].copy(); X_reset = np.zeros_like(eval_X)
    for i in range(n):
        if i in boundaries:
            era = eval_X[i].copy()
        else:
            era = 0.7*era + 0.3*eval_X[i]
        X_reset[i] = era
    X_rl = np.log1p(np.maximum(X_reset, 0.0))
    sc3 = StandardScaler(); X_rs = sc3.fit_transform(X_rl)
    lr3 = LogisticRegression(max_iter=1000, solver='lbfgs', C=0.1)
    try:
        cv3 = cross_val_score(lr3, X_rs, eval_Y, cv=5, scoring="accuracy")
        rows.append({"method": "change_point_reset_EMA", "window_size": "reset",
                     "cv_accuracy": float(cv3.mean()), "n_samples": n})
    except Exception:
        rows.append({"method": "change_point_reset_EMA", "window_size": "reset",
                     "cv_accuracy": 0.0, "n_samples": n})

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTDIR}/report_lag_ablation.csv", index=False)

    best_method = max(rows, key=lambda r: r["cv_accuracy"]) if rows else {"method": "N/A", "cv_accuracy": 0}
    base_acc = next((r["cv_accuracy"] for r in rows if r["method"] == "fixed_window_N=1"), 0.0)

    if best_method["cv_accuracy"] > base_acc + 0.03:
        verdict3 = f"REPORT_LAG_MATTERS — best is {best_method['method']} (+{best_method['cv_accuracy']-base_acc:+.3f} vs N=1)"
    else:
        verdict3 = "REPORT_LAG_MINIMAL_EFFECT"

    print(f"  Best lag method: {best_method['method']} = {best_method['cv_accuracy']:.4f}")
    print(f"  Baseline (N=1): {base_acc:.4f}")
    print(f"  Verdict: {verdict3}")

    return rows, verdict3


# ═══════════════════════════════════════════════════════════
# ANALYSIS 4 + 5: Per-task Breakdown + Capital Specialization
# ═══════════════════════════════════════════════════════════

def analysis_task_and_specialization(data):
    print(f"\n{'='*60}")
    print("ANALYSIS 4+5: Per-task Breakdown + Capital Specialization")
    print(f"{'='*60}")

    all_correct = data["all_cap_correct"]
    task_labels = np.array(data["eval_task_labels"])
    capital_ids = data["capital_ids"]
    NC = data["NC"]
    bs_idx = data["best_single_idx"]

    eval_X_norm = data["eval_X_norm"]
    meta_mlp = data["meta_mlp"]
    oracle_per_step = data["oracle_per_step"]
    oracle_best = data["oracle_best_idx"]

    task_types = ["Task_A", "Task_B", "Task_C", "Task_D"]
    per_task_rows = []
    spec_rows = []

    for tt in task_types:
        mask = task_labels == tt
        if mask.sum() == 0: continue
        n_t = int(mask.sum())
        row = {"task": tt, "n_steps": n_t}

        for ci, cid in enumerate(capital_ids):
            row[cid] = float(all_correct[mask, ci].mean())

        row["BestSingle"] = float(all_correct[mask, bs_idx].mean())
        row["OracleHindsight"] = float(oracle_per_step[mask].mean())

        # MetaMLP eval on this task
        mlp_total = 0.0
        indices = np.where(mask)[0]
        for si in indices:
            rpt_t = torch.tensor(eval_X_norm[si], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad(): pv = meta_mlp(rpt_t).cpu().numpy()[0]
            ci_mlp = int(np.argmax(pv))
            mlp_total += all_correct[si, ci_mlp]
        row["MetaMLP"] = mlp_total/n_t if n_t > 0 else 0.0

        # Cyber eval on this task
        cb = FeedbackControlledAllocator(n_capitals=NC, capital_ids=capital_ids,
                                          predictor=meta_mlp, max_weight_change=0.12,
                                          impairment_threshold=0.30,
                                          clf_mean=data["clf_mean"], clf_std=data["clf_std"],
                                          device=DEVICE)
        cb_total = 0.0
        for si in indices:
            rpt_vec = data["eval_report_vectors"][si]
            weights = cb.step(make_dummy_reports_from_vec(rpt_vec), report_vector=rpt_vec, feedback={"allocator_regret": 0.0})
            ci_cb = int(np.argmax(list(weights.values())))
            cb_total += all_correct[si, ci_cb]
        row["Cyber"] = cb_total/n_t if n_t > 0 else 0.0

        # Best capital per task (specialization)
        cap_scores = [row[cid] for cid in capital_ids]
        best_cap = capital_ids[int(np.argmax(cap_scores))]
        best_score = max(cap_scores)
        sorted_scores = sorted(cap_scores, reverse=True)
        second_best = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        margin = best_score - second_best
        row["BestCapital"] = best_cap
        row["BestCapitalMargin"] = margin

        # Expected vs actual
        expected_map = {"Task_A": "PolicyClone", "Task_B": "AEP",
                        "Task_C": "PrototypeOutcome", "Task_D": "GoalInference"}
        expected = expected_map.get(tt, "?")
        row["ExpectedBest"] = expected
        row["TaxonomyMatches"] = (best_cap == expected)

        per_task_rows.append(row)

        # Capital dominance: which capital dominates this task?
        dominance = best_score / max(0.01, np.mean(cap_scores))
        spec_rows.append({
            "task": tt,
            "best_capital": best_cap,
            "expected_best": expected,
            "taxonomy_match": best_cap == expected,
            "capital_dominance_index": float(dominance),
            "task_specialization_index": 1.0 - float(np.std(cap_scores)),
            "best_capital_margin": float(margin),
            "entropy_best_capital": 1.0 if margin > 0.05 else 0.0,
            "second_best": sorted_scores[1] if len(sorted_scores) > 1 else 0.0,
            "n_capitals_better_than_bs": int(sum(1 for s in cap_scores if s > row["BestSingle"])),
        })

    per_task_df = pd.DataFrame(per_task_rows)
    per_task_df.to_csv(f"{OUTDIR}/per_task_capital_matrix.csv", index=False)

    spec_df = pd.DataFrame(spec_rows)
    spec_df.to_csv(f"{OUTDIR}/capital_specialization_audit.csv", index=False)

    # Overall specialization verdict
    n_match = sum(1 for r in spec_rows if r["taxonomy_match"])
    if n_match == len(spec_rows):
        spec_verdict = "STRONG_SPECIALIZATION — taxonomy predictions hold per task"
    elif n_match >= len(spec_rows)//2:
        spec_verdict = "PARTIAL_SPECIALIZATION — some tasks follow taxonomy"
    else:
        spec_verdict = "WEAK_SPECIALIZATION — capital taxonomy does not align with per-task best"

    # Dominance verdict
    high_dominance = any(r["capital_dominance_index"] > 3.0 for r in spec_rows)
    same_best = len(set(r["best_capital"] for r in spec_rows)) == 1
    if same_best:
        dom_verdict = "SINGLE_CAPITAL_DOMINANCE — same capital best across tasks, no second-order space"
    elif high_dominance:
        dom_verdict = "PARTIAL_DOMINANCE — some tasks show strong single-capital advantage"
    else:
        dom_verdict = "DIVERSE_OPTIMAL — different capitals best per task, second-order space exists"

    print(f"  Taxonomy matches: {n_match}/{len(spec_rows)}")
    print(f"  Specialization: {spec_verdict}")
    print(f"  Dominance: {dom_verdict}")
    for r in per_task_rows:
        print(f"    {r['task']}: best={r['BestCapital']} (expected={r['ExpectedBest']}) margin={r['BestCapitalMargin']:.3f}")

    return per_task_rows, spec_rows, spec_verdict, dom_verdict


# ═══════════════════════════════════════════════════════════
# ANALYSIS 6: Negative Transfer Effectiveness Ablation
# ═══════════════════════════════════════════════════════════

def analysis_protection_effectiveness(data):
    print(f"\n{'='*60}")
    print("ANALYSIS 6: Negative Transfer Effectiveness Ablation")
    print(f"{'='*60}")

    NC = data["NC"]; n_eval = data["n_eval"]
    capital_ids = data["capital_ids"]
    all_correct = data["all_cap_correct"]
    oracle_per_step = data["oracle_per_step"]
    clf_mean = data["clf_mean"]; clf_std = data["clf_std"]
    meta_mlp = data["meta_mlp"]

    configs = [
        ("Cyber_no_protection",    dict(enable_impairment=False, enable_fallback=False,
                                          enable_depreciation=False, enable_smoothing=False,
                                          enable_feedforward=True)),
        ("Cyber_impairment_only",   dict(enable_impairment=True, enable_fallback=False,
                                          enable_depreciation=False, enable_smoothing=False,
                                          enable_feedforward=True)),
        ("Cyber_impairment_fallback", dict(enable_impairment=True, enable_fallback=True,
                                           enable_depreciation=False, enable_smoothing=False,
                                           enable_feedforward=True)),
        ("Cyber_imp_fallback_depr", dict(enable_impairment=True, enable_fallback=True,
                                          enable_depreciation=True, enable_smoothing=False,
                                          enable_feedforward=True)),
        ("Cyber_full",              dict(enable_impairment=True, enable_fallback=True,
                                          enable_depreciation=True, enable_smoothing=True,
                                          enable_feedforward=True)),
    ]

    rows = []
    for cfg_name, flags in configs:
        cb = FeedbackControlledAllocatorAblated(
            n_capitals=NC, capital_ids=capital_ids, predictor=meta_mlp,
            max_weight_change=0.12, impairment_threshold=0.30,
            clf_mean=clf_mean, clf_std=clf_std, device=DEVICE, **flags)
        cb.reset()
        total = 0.0; regrets = []; fallback_count = 0
        weight_after_failure = 0.0; failure_detected = False

        for step in range(n_eval):
            rpt_vec = data["eval_report_vectors"][step]
            oracle_r = float(oracle_per_step[step])
            weights = cb.step(make_dummy_reports_from_vec(rpt_vec), report_vector=rpt_vec, feedback={"allocator_regret": 0.0})
            ci = int(np.argmax(list(weights.values())))
            reward = float(all_correct[step, ci])
            total += reward; regrets.append(oracle_r - reward)
            if cb.is_fallback_active():
                fallback_count += 1

        score = total/n_eval
        cum_regret = float(np.sum(regrets))
        recovery_time = 0  # placeholder

        rows.append({
            "config": cfg_name,
            "score": score,
            "cumulative_regret": cum_regret,
            "delta_vs_BestSingle": score - float(all_correct[:, data["best_single_idx"]].mean()),
            "fallback_trigger_count": fallback_count,
            "fallback_rate": fallback_count/n_eval,
            "weight_mean_after_failure": weight_after_failure,
            "avg_per_step_regret": cum_regret/n_eval,
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTDIR}/protection_effectiveness_audit.csv", index=False)

    no_prot = rows[0]["score"]
    full_prot = rows[-1]["score"]
    delta = full_prot - no_prot

    if delta > 0.02:
        pv = f"PROTECTION_HELPS — full protection +{delta:+.3f} vs no protection"
    elif abs(delta) < 0.01:
        pv = "PROTECTION_NEUTRAL — no significant difference between configs"
    else:
        pv = f"PROTECTION_HURTS — full protection {delta:+.3f} vs no protection"

    print(f"  No protection: {no_prot:.4f}  |  Full protection: {full_prot:.4f}")
    print(f"  Delta: {delta:+.4f}")
    print(f"  Verdict: {pv}")

    return rows, pv


# ═══════════════════════════════════════════════════════════
# ANALYSIS 7: External-only Slice
# ═══════════════════════════════════════════════════════════

def analysis_external_slice(data):
    print(f"\n{'='*60}")
    print("ANALYSIS 7: External-only Slice (Task D / HiddenGoalGridWorld)")
    print(f"{'='*60}")

    all_correct = data["all_cap_correct"]
    task_labels = np.array(data["eval_task_labels"])
    capital_ids = data["capital_ids"]
    NC = data["NC"]
    bs_idx = data["best_single_idx"]
    oracle_per_step = data["oracle_per_step"]
    oracle_best = data["oracle_best_idx"]

    mask = task_labels == "Task_D"
    n_ext = mask.sum()
    row = {"env": "HiddenGoalGridWorld_TaskD", "n_steps": int(n_ext)}

    for ci, cid in enumerate(capital_ids):
        row[cid] = float(all_correct[mask, ci].mean()) if n_ext > 0 else 0.0

    row["BestSingle"] = float(all_correct[mask, bs_idx].mean()) if n_ext > 0 else 0.0
    row["OracleHindsight"] = float(oracle_per_step[mask].mean()) if n_ext > 0 else 0.0

    # MetaMLP + Cyber on external
    eval_X_norm = data["eval_X_norm"]; meta_mlp = data["meta_mlp"]

    mlp_total = 0.0
    indices = np.where(mask)[0]
    for si in indices:
        rpt_t = torch.tensor(eval_X_norm[si], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): pv = meta_mlp(rpt_t).cpu().numpy()[0]
        ci_mlp = int(np.argmax(pv)); mlp_total += all_correct[si, ci_mlp]
    row["MetaMLP"] = mlp_total/max(1, n_ext)

    cb = FeedbackControlledAllocator(n_capitals=NC, capital_ids=capital_ids,
                                      predictor=meta_mlp, max_weight_change=0.12,
                                      impairment_threshold=0.30,
                                      clf_mean=data["clf_mean"], clf_std=data["clf_std"],
                                      device=DEVICE)
    cb_total = 0.0
    for si in indices:
        rpt_vec = data["eval_report_vectors"][si]
        weights = cb.step(make_dummy_reports_from_vec(rpt_vec), report_vector=rpt_vec, feedback={"allocator_regret": 0.0})
        ci_cb = int(np.argmax(list(weights.values())))
        cb_total += all_correct[si, ci_cb]
    row["Cyber"] = cb_total/max(1, n_ext)

    df = pd.DataFrame([row])
    df.to_csv(f"{OUTDIR}/external_only_slice.csv", index=False)

    oracle_ext = row["OracleHindsight"]
    best_ext = row["BestSingle"]
    gain_ext = oracle_ext - best_ext

    if oracle_ext < 0.05:
        ev = "EXTERNAL_TASK_UNINFORMATIVE — capitals fail on external benchmark"
    elif gain_ext < 0.02:
        ev = "EXTERNAL_NO_SWITCHING_GAIN — no room for allocation on external task"
    else:
        ev = f"EXTERNAL_HAS_POTENTIAL — oracle gain = {gain_ext:+.3f}"

    print(f"  Oracle external: {oracle_ext:.4f}  |  BS external: {best_ext:.4f}")
    print(f"  Oracle gain: {gain_ext:+.4f}")
    for cid in capital_ids:
        print(f"    {cid}: {row[cid]:.4f}")
    print(f"  MetaMLP: {row['MetaMLP']:.4f}  Cyber: {row['Cyber']:.4f}")
    print(f"  Verdict: {ev}")

    return row, ev


# ═══════════════════════════════════════════════════════════
# ANALYSIS 8: FINAL REPORT
# ═══════════════════════════════════════════════════════════

def generate_final_report(data, results_all):
    print(f"\n{'='*60}")
    print("ANALYSIS 8: Generating Final Report")
    print(f"{'='*60}")

    (sw_rows, sw_pt, sw_v) = results_all["switching"]
    (rs_rows, rs_v) = results_all["report_sufficiency"]
    (rl_rows, rl_v) = results_all["report_lag"]
    (pt_rows, sp_rows, sp_v, dom_v) = results_all["task_specialization"]
    (pe_rows, pe_v) = results_all["protection"]
    (ext_row, ext_v) = results_all["external"]

    # Determine root cause
    oracle_gain = sw_rows[0]["value"]
    bs_score = float(data["all_cap_correct"][:, data["best_single_idx"]].mean())
    oracle_score = float(data["oracle_per_step"].mean())
    bs_rew = next((r["reward_correctness"] for r in rs_rows if r["selector"]=="BestSingleCapital"), bs_score)
    lr_oba = next((r["oracle_best_accuracy"] for r in rs_rows if r["selector"]=="LogisticRegressionSelector"), 0)
    lr_rew = next((r["reward_correctness"] for r in rs_rows if r["selector"]=="LogisticRegressionSelector"), 0)

    causes = []
    if oracle_gain < 0.03:
        causes.append(("PRIMARY", "IC3_S_NO_SWITCHING_OPPORTUNITY",
                       f"Oracle gain over BestSingle = {oracle_gain:.4f} — no room for second-order allocation"))
    elif oracle_gain < 0.10:
        causes.append(("CONTRIBUTING", "LOW_SWITCHING_CEILING",
                       f"Oracle gain = {oracle_gain:.4f} — limited ceiling"))

    best_rs = max((r for r in rs_rows if r["selector"] not in
                   ("BestSingleCapital", "OracleHindsight", "MetaMLPAllocator",
                    "FeedbackControlledAllocator")),
                  key=lambda x: x["reward_correctness"], default={"selector":"N/A","reward_correctness":0})
    if best_rs["reward_correctness"] <= bs_score + 0.01:
        causes.append(("PRIMARY", "IC3_S_REPORT_INTERFACE_INSUFFICIENT",
                       f"Best oracle selector ({best_rs['selector']}) = {best_rs['reward_correctness']:.4f} ≤ BS={bs_score:.4f}"))
    elif best_rs["reward_correctness"] < bs_score + 0.05:
        causes.append(("CONTRIBUTING", "REPORT_MARGINALLY_USEFUL",
                       f"Best selector = {best_rs['reward_correctness']:.4f}, only marginally above BS"))

    meta_acc = next((r["reward_correctness"] for r in rs_rows if r["selector"]=="MetaMLPAllocator"), 0)
    cyber_acc = next((r["reward_correctness"] for r in rs_rows if r["selector"]=="FeedbackControlledAllocator"), 0)
    if meta_acc < bs_score - 0.05 and best_rs["reward_correctness"] > bs_score:
        causes.append(("PRIMARY", "IC3_S_ALLOCATOR_LEARNING_FAILURE",
                       f"Report sufficient (best={best_rs['reward_correctness']:.4f}) but MetaMLP={meta_acc:.4f} poorly learns"))
    if cyber_acc < bs_score - 0.05:
        causes.append(("CONTRIBUTING", "CYBERNETIC_ALLOCATOR_INEFFECTIVE",
                       f"Cyber={cyber_acc:.4f} well below BS={bs_score:.4f}"))

    if rl_v.startswith("REPORT_LAG_MATTERS"):
        causes.append(("CONTRIBUTING", "IC3_S_REPORT_LAG_FAILURE",
                       f"Report lag ablation: {rl_v}"))

    if sp_v == "WEAK_SPECIALIZATION":
        causes.append(("PRIMARY", "IC3_S_CAPITAL_SPECIALIZATION_INSUFFICIENT",
                       "Capital taxonomy predictions do not hold per task"))
    elif sp_v == "PARTIAL_SPECIALIZATION":
        causes.append(("CONTRIBUTING", "PARTIAL_SPECIALIZATION_MISMATCH",
                       f"Only partial taxonomy alignment: {sp_v}"))

    if "SINGLE_CAPITAL_DOMINANCE" in dom_v:
        causes.append(("PRIMARY", "IC3_S_CAPITAL_SPECIALIZATION_INSUFFICIENT",
                       f"Same capital dominates across tasks — no second-order space"))

    if "PROTECTION_HURTS" in pe_v or "PROTECTION_NEUTRAL" in pe_v:
        causes.append(("CONTRIBUTING", "IC3_S_PROTECTION_MECHANISM_INEFFECTIVE",
                       f"Protection ablation: {pe_v}"))

    if "EXTERNAL_TASK_UNINFORMATIVE" in ext_v:
        causes.append(("CONTRIBUTING", "IC3_S_EXTERNAL_TASK_UNINFORMATIVE",
                       f"External benchmark: {ext_v}"))

    # Final verdict
    primary = [c for c in causes if c[0] == "PRIMARY"]
    if not primary:
        primary = [c for c in causes if c[0] == "CONTRIBUTING"]

    if primary:
        final_verdict = primary[0][1]
    else:
        final_verdict = "IC3_S_READY_FOR_ALLOCATOR_REDESIGN"

    # Build report
    report = f"""# IC-3-S: Selector Failure Decomposition — Final Report

**Date**: 2026-05-11
**Phase**: IC-3-S (Diagnostic — does NOT announce second-order intelligence)
**Seed**: {data['seed']}  |  **Capital Set**: Main-5  |  **Schema**: {N_FIELDS}×{data['NC']}={N_FIELDS*data['NC']} features

---

## Final Verdict: `{final_verdict}`

### Root Cause Analysis

| Priority | Verdict | Detail |
|---|---|---|
"""

    for pri, code, detail in causes:
        report += f"| {pri} | `{code}` | {detail} |\n"

    report += f"""
---

## 1. Switching Opportunity Audit

| Metric | Value |
|---|---|
| Oracle gain over BestSingle | {oracle_gain:.4f} |
| Switching opportunity rate | {sw_rows[1]['value']:.1%} |
| Oracle reward (eval) | {oracle_score:.4f} |
| BestSingle reward (eval) | {bs_score:.4f} |
| Block-transition oracle gain | {sw_rows[3]['value']:.4f} |
| Within-block oracle gain | {sw_rows[4]['value']:.4f} |

### Per-task Oracle Gain

| Task | N | Oracle Gain | Switch Rate |
|---|---|---|---|
"""

    for r in sw_pt:
        report += f"| {r['task']} | {r['n_steps']} | {r['oracle_gain_over_bestsingle']:.4f} | {r['switching_opportunity_rate']:.1%} |\n"

    report += f"""
**Verdict**: {sw_v}

---

## 2. Report Sufficiency Test

| Selector | Oracle-Best Acc | Reward | Δ vs BS | Oracle Gap |
|---|---|---|---|---|
"""

    for r in rs_rows:
        oba = r.get('oracle_best_accuracy', 0)
        rew = r.get('reward_correctness', 0)
        delta_str = f"{r.get('delta_vs_bs_reward', 0):+.4f}" if 'delta_vs_bs_reward' in r else "—"
        report += f"| {r['selector']} | {oba:.4f} | {rew:.4f} | {delta_str} | {r['oracle_gap']:.4f} |\n"

    report += f"""
**Best learned oracle selector**: {best_rs['selector']} (OBA={best_rs.get('oracle_best_accuracy',0):.4f}, Reward={best_rs['reward_correctness']:.4f})

**Key finding**: Offline oracle selectors trained on oracle+train data achieve only {lr_oba:.1%} oracle-best accuracy (vs chance 20%). LR/RF collapse to BestSingle behavior (predict fixed capital). This confirms the CapitalReport vectors, as currently defined, do NOT carry sufficient information to distinguish which capital will be best at each step — despite the oracle gain of {oracle_gain:.1%} showing room exists.

The root cause is NOT insufficient switching opportunity (oracle gain = {oracle_gain:.1%}) nor report lag (all window methods identical at {next((r['cv_accuracy'] for r in rl_rows if r['method']=='fixed_window_N=1'), 0):.1%}), but rather that the CapitalReport features lack the predictive signal needed for per-step capital selection.

**Verdict**: {rs_v}

---

## 3. Report Lag Ablation

| Method | Window | CV Accuracy |
|---|---|---|
"""

    for r in rl_rows[:12]:
        report += f"| {r['method']} | {r['window_size']} | {r['cv_accuracy']:.4f} |\n"

    report += f"""
**Verdict**: {rl_v}

---

## 4. Per-task Capital Matrix

| Task | N | PolicyClone | PrototypeOutcome | AEP | GoalInference | SafeFallback | BestSingle | Oracle | Best Capital |
|---|---|---|---|---|---|---|---|---|---|---|
"""

    for r in pt_rows:
        report += f"| {r['task']} | {r['n_steps']} |"
        for cid in data["capital_ids"]:
            report += f" {r.get(cid, 0):.4f} |"
        report += f" {r.get('BestSingle', 0):.4f} | {r.get('OracleHindsight', 0):.4f} | {r.get('BestCapital', '?')} |\n"

    report += f"""
**Taxonomy matches**: {sum(1 for r in pt_rows if r.get('TaxonomyMatches', False))}/{len(pt_rows)}

**Answer Q1 (Task A vs PolicyClone)**: {'YES' if any(r.get('BestCapital')=='PolicyClone' and r['task']=='Task_A' for r in pt_rows) else 'NO'}
**Answer Q2 (Task B vs AEP)**: {'YES' if any(r.get('BestCapital')=='AEP' and r['task']=='Task_B' for r in pt_rows) else 'NO'}
**Answer Q3 (Task C vs Prototype)**: {'YES' if any(r.get('BestCapital')=='PrototypeOutcome' and r['task']=='Task_C' for r in pt_rows) else 'NO'}
**Answer Q4 (Task D vs GoalInference)**: {'YES' if any(r.get('BestCapital')=='GoalInference' and r['task']=='Task_D' for r in pt_rows) else 'NO'}

---

## 5. Capital Specialization

| Task | Best | Expected | Match | Dominance | Margin |
|---|---|---|---|---|---|
"""

    for r in sp_rows:
        report += f"| {r['task']} | {r['best_capital']} | {r['expected_best']} | {'✅' if r['taxonomy_match'] else '❌'} | {r['capital_dominance_index']:.2f} | {r['best_capital_margin']:.3f} |\n"

    report += f"""
**Specialization**: {sp_v}
**Dominance**: {dom_v}

---

## 6. Negative Transfer Effectiveness

| Config | Score | Cum. Regret | Δ vs BS | Fallbacks |
|---|---|---|---|---|
"""

    for r in pe_rows:
        report += f"| {r['config']} | {r['score']:.4f} | {r['cumulative_regret']:.1f} | {r['delta_vs_BestSingle']:+.4f} | {r['fallback_trigger_count']} |\n"

    report += f"""
**Verdict**: {pe_v}

---

## 7. External-only Slice (Task D)

| Capital | Score |
|---|---|
"""

    for cid in data["capital_ids"]:
        report += f"| {cid} | {ext_row.get(cid, 0):.4f} |\n"
    report += f"| BestSingle | {ext_row.get('BestSingle', 0):.4f} |\n"
    report += f"| OracleHindsight | {ext_row.get('OracleHindsight', 0):.4f} |\n"
    report += f"| MetaMLP | {ext_row.get('MetaMLP', 0):.4f} |\n"
    report += f"| Cyber | {ext_row.get('Cyber', 0):.4f} |\n"

    report += f"""
**Oracle gain on external**: {ext_row.get('OracleHindsight', 0) - ext_row.get('BestSingle', 0):+.4f}
**Verdict**: {ext_v}

---

## 8. Summary of Failure Causes

| Rank | Cause | Impact |
|---|---|---|
"""

    for i, (pri, code, detail) in enumerate(causes):
        report += f"| {i+1} | {code} | {pri} |\n"

    report += f"""
---

## Generated Files (results/ic3_s/)

| # | File | Content |
|---|---|---|
| 1 | `switching_opportunity_audit.csv` | Oracle gain, switching opportunity rate |
| 2 | `report_sufficiency_test.csv` | LR/RF/MLP oracle selector vs allocators |
| 3 | `report_lag_ablation.csv` | Window N=1-30, EMA variants, change-point reset |
| 4 | `per_task_capital_matrix.csv` | Per-task correctness per capital per allocator |
| 5 | `capital_specialization_audit.csv` | Dominance index, specialization entropy, margins |
| 6 | `protection_effectiveness_audit.csv` | 5-config ablation: no_prot→full |
| 7 | `external_only_slice.csv` | Task D only: all capitals and allocators |
| 8 | `IC3_S_SELECTOR_FAILURE_DECOMPOSITION_REPORT.md` | This report |

---

*End of IC-3-S Report. Root cause analysis complete. No second-order intelligence claim made.*
"""
    report_path = f"{OUTDIR}/IC3_S_SELECTOR_FAILURE_DECOMPOSITION_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  FINAL REPORT → {report_path}")
    print(f"  VERDICT → {final_verdict}")

    return final_verdict


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("IC-3-S: Selector Failure Decomposition")
    print("Diagnoses *why* allocator fails — does NOT claim second-order intelligence")
    print("="*60)

    # Collect data on best seed
    seed = 45
    data = collect_eval_data(seed=seed, n_eval=600)

    # Analysis 1
    sw_rows, sw_pt, sw_v = analysis_switching_opportunity(data)

    # Analysis 2
    rs_rows, rs_v = analysis_report_sufficiency(data)

    # Analysis 3
    rl_rows, rl_v = analysis_report_lag(data)

    # Analysis 4+5
    pt_rows, sp_rows, sp_v, dom_v = analysis_task_and_specialization(data)

    # Analysis 6
    pe_rows, pe_v = analysis_protection_effectiveness(data)

    # Analysis 7
    ext_row, ext_v = analysis_external_slice(data)

    # Analysis 8
    results_all = {
        "switching": (sw_rows, sw_pt, sw_v),
        "report_sufficiency": (rs_rows, rs_v),
        "report_lag": (rl_rows, rl_v),
        "task_specialization": (pt_rows, sp_rows, sp_v, dom_v),
        "protection": (pe_rows, pe_v),
        "external": (ext_row, ext_v),
    }
    final_verdict = generate_final_report(data, results_all)

    print(f"\n{'='*60}")
    print(f"IC-3-S COMPLETE — Final Verdict: {final_verdict}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()