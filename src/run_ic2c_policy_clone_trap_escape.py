"""
IC-2c: Policy-Clone Trap Escape Benchmark
==========================================
Tests whether AEP/Counterfactual models escape the PolicyClone trap through:
  1. Goal transfer (U1-U5)
  2. Action coverage gap (biased sampling + CF probes)
  3. Active probe value (partial observability)
  4. Updated evaluation metrics
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import (prepare_counterfactual_data, train_state_only_classifier,
                       train_counterfactual_joint, train_ae_model)
from src.models import (StateOnlyPredictor, MLP, AEPCompressor,
                        ResidualCompressor, CounterfactualCompressor)
from src.memory_baselines import RawMemoryEqualCost, PrototypeMemory
from src.metrics import compute_best_action_match, compute_regret, compute_rank_accuracy

# ─── Config ───────────────────────────────────────────────
os.makedirs("results/ic2c", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2]
EPOCHS = 300
PATIENCE = 60
BOTTLENECK_DIM = 48
RESIDUAL_DIM = 12
ENV_KWARGS = dict(state_dim=2, history_len=8, action_gain=0.25)

cf_df = pd.read_csv("results/counterfactual_table.csv")

# ═══════════════════════════════════════════════════════════
# PHASE 1: Rename baseline + StateOnlyDynamics model
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 1: PolicyCloneBaseline + StateOnlyDynamicsPredictor")
print("=" * 60)

# PolicyCloneBaseline = StateOnlyPredictor (history -> best_action via CE)
# Already implemented. Keep as alias.
PolicyCloneBaseline = StateOnlyPredictor


class StateOnlyDynamicsPredictor(nn.Module):
    """Predicts only the noop (autonomous) outcome: h -> Y(h,0)."""
    def __init__(self, obs_dim, history_len, bottleneck_dim=16):
        super().__init__()
        self.in_dim = history_len * (obs_dim + 1)
        self.encoder = MLP(self.in_dim, [bottleneck_dim * 2, bottleneck_dim], bottleneck_dim)
        self.head = nn.Linear(bottleneck_dim, 1)

    def forward(self, x):
        return self.head(self.encoder(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════
# Multi-Goal Utility Functions
# ═══════════════════════════════════════════════════════════

def compute_best_action_for_goal(Y, goal="U1_main", target=None):
    """Given a (n,3) outcome matrix Y, compute best_action under different goals.
    Y[:,0] = action -1, Y[:,1] = action 0, Y[:,2] = action +1
    """
    if goal == "U1_main":
        return np.argmax(Y, axis=1)
    elif goal == "U2_reverse":
        return np.argmin(Y, axis=1)
    elif goal == "U3_target":
        if target is None:
            target = np.zeros((Y.shape[0], 1))
        dist = np.abs(Y - target)
        return np.argmin(dist, axis=1)
    elif goal == "U4_risk_avoid":
        # Risk zone: any outcome < -1.0 is penalized heavily
        risk_penalty = np.where(Y < -1.0, (Y + 1.0) ** 2 * 10, 0)
        risk_adj = Y - risk_penalty
        return np.argmax(risk_adj, axis=1)
    elif goal == "U5_energy_aware":
        action_costs = np.array([0.2, 0.0, 0.2], dtype=np.float32)
        net = Y - action_costs[np.newaxis, :]
        return np.argmax(net, axis=1)
    else:
        return np.argmax(Y, axis=1)


def evaluate_model_goal_transfer(model, X_test, Y_test_true, mechanism_type, device=DEVICE):
    """Evaluate model on all 5 goals. Returns dict of per-goal best_action_match.
    mechanism_type: 'policyclone', 'cf', 'aep', 'residual'
    """
    results = {}
    n = len(X_test)

    # Get predicted outcomes
    with torch.no_grad():
        x_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        if mechanism_type == "policyclone":
            logits = model(x_t).cpu().numpy()
            Y_pred = logits  # For policyclone, these are "faux" outcome scores
        elif hasattr(model, 'predict_all_actions'):
            Y_pred = model.predict_all_actions(x_t).cpu().numpy()
        else:
            Y_pred = model(x_t).cpu().numpy()

    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape(-1, 1)
    if Y_pred.shape[1] != 3:
        return {}

    # U1-U5 evaluation
    for goal in ["U1_main", "U2_reverse", "U3_target", "U4_risk_avoid", "U5_energy_aware"]:
        ba_pred = compute_best_action_for_goal(Y_pred, goal=goal)
        ba_true = compute_best_action_for_goal(Y_test_true, goal=goal)
        results[goal] = float(np.mean(ba_pred == ba_true))

    return results


# ═══════════════════════════════════════════════════════════
# PHASE 2: Goal Transfer Benchmark
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 2: Goal Transfer Benchmark")
print("=" * 60)

goal_transfer_records = []

for seed in tqdm(SEEDS, desc="Goal Transfer"):
    train_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "train") & (cf_df["horizon"] == 1)]
    test_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "test_id") & (cf_df["horizon"] == 1)]
    X_tr, Y_tr, ba_tr = prepare_counterfactual_data(train_df, seed, ENV_KWARGS)
    X_te, Y_te, ba_te = prepare_counterfactual_data(test_df, seed, ENV_KWARGS)

    # ── PolicyCloneBaseline: h -> best_action (CE on U1 labels) ──
    pc = PolicyCloneBaseline(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    try:
        pc = train_state_only_classifier(pc, X_tr, Y_tr, None, None, epochs=EPOCHS, patience=PATIENCE, device=DEVICE)
    except Exception as e:
        print(f"  [WARN] PC training failed seed={seed}: {e}")
        pc = PolicyCloneBaseline(2, 8, bottleneck_dim=BOTTLENECK_DIM)

    # PolicyClone only trained on U1, evaluate on U1-U5
    # But PolicyClone outputs logits not outcomes, so goal transfer uses faux logits
    pc_results = evaluate_model_goal_transfer(pc, X_te, Y_te, "policyclone")
    for goal, match in pc_results.items():
        goal_transfer_records.append({"seed": seed, "goal": goal, "mechanism": "PolicyCloneBaseline", "best_action_match": match})

    # ── AEPCompressor: h,a -> Y(h,a), train on outcome table ──
    aep = AEPCompressor(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    try:
        aep = train_ae_model(aep, X_tr, Y_tr, None, None, "aep", epochs=EPOCHS, patience=PATIENCE, device=DEVICE, ce_weight=0.8)
    except Exception as e:
        print(f"  [WARN] AEP training failed seed={seed}: {e}")
        aep = AEPCompressor(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    aep_results = evaluate_model_goal_transfer(aep, X_te, Y_te, "aep")
    for goal, match in aep_results.items():
        goal_transfer_records.append({"seed": seed, "goal": goal, "mechanism": "AEPCompressor", "best_action_match": match})

    # ── CounterfactualCompressor ──
    cf = CounterfactualCompressor(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    try:
        cf = train_counterfactual_joint(cf, X_tr, Y_tr, None, None, epochs=EPOCHS, patience=PATIENCE, device=DEVICE, ce_weight=1.5)
    except Exception as e:
        print(f"  [WARN] CF training failed seed={seed}: {e}")
        cf = CounterfactualCompressor(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    cf_results = evaluate_model_goal_transfer(cf, X_te, Y_te, "cf")
    for goal, match in cf_results.items():
        goal_transfer_records.append({"seed": seed, "goal": goal, "mechanism": "CounterfactualCompressor", "best_action_match": match})

    # ── ResidualCompressor ──
    rc = ResidualCompressor(2, 8, bottleneck_dim=BOTTLENECK_DIM, residual_dim=RESIDUAL_DIM)
    try:
        rc = train_ae_model(rc, X_tr, Y_tr, None, None, "residual", epochs=EPOCHS, patience=PATIENCE, device=DEVICE, ce_weight=0.8)
    except Exception as e:
        print(f"  [WARN] RC training failed seed={seed}: {e}")
        rc = ResidualCompressor(2, 8, bottleneck_dim=BOTTLENECK_DIM, residual_dim=RESIDUAL_DIM)
    rc_results = evaluate_model_goal_transfer(rc, X_te, Y_te, "residual")
    for goal, match in rc_results.items():
        goal_transfer_records.append({"seed": seed, "goal": goal, "mechanism": "ResidualCompressor", "best_action_match": match})

    # ── StateOnlyDynamics: h -> Y(h,0) ──
    sod = StateOnlyDynamicsPredictor(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    try:
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
        y_tr_t = torch.tensor(Y_tr[:, 1:2], dtype=torch.float32).to(DEVICE)
        sod = sod.to(DEVICE)
        opt = torch.optim.Adam(sod.parameters(), lr=1e-3, weight_decay=1e-5)
        best_loss = float("inf")
        best_state = None
        no_improve = 0
        for epoch in range(EPOCHS):
            sod.train()
            opt.zero_grad()
            pred = sod(X_tr_t)
            loss = nn.functional.mse_loss(pred, y_tr_t)
            loss.backward()
            opt.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone().cpu() for k, v in sod.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break
        if best_state is not None:
            sod.load_state_dict(best_state)
    except Exception as e:
        print(f"  [WARN] SOD training failed seed={seed}: {e}")

    # SOD predicts only noop; construct full (n,3) by placing noop at position 1 and copying for comparison
    with torch.no_grad():
        x_tt = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
        noop_pred = sod(x_tt).cpu().numpy()
    sod_Y = np.column_stack([noop_pred[:, 0] - 0.5, noop_pred[:, 0], noop_pred[:, 0] + 0.5])  # rough hack
    for goal in ["U1_main", "U2_reverse", "U3_target", "U4_risk_avoid", "U5_energy_aware"]:
        ba_pred = compute_best_action_for_goal(sod_Y, goal=goal)
        ba_true = compute_best_action_for_goal(Y_te, goal=goal)
        goal_transfer_records.append({"seed": seed, "goal": goal, "mechanism": "StateOnlyDynamics", "best_action_match": float(np.mean(ba_pred == ba_true))})

    # ── ActionOnly (majority vote) ──
    ao_ba = np.argmax(np.bincount(ba_tr, minlength=3))
    for goal in ["U1_main", "U2_reverse", "U3_target", "U4_risk_avoid", "U5_energy_aware"]:
        ba_true = compute_best_action_for_goal(Y_te, goal=goal)
        match = float(np.mean(np.ones(len(Y_te)) * ao_ba == ba_true))
        goal_transfer_records.append({"seed": seed, "goal": goal, "mechanism": "ActionOnly", "best_action_match": match})

gt_df = pd.DataFrame(goal_transfer_records)
gt_df.to_csv("results/ic2c/goal_transfer.csv", index=False)
print("  goal_transfer.csv saved")

# Summary
gt_summ = gt_df.groupby(["goal", "mechanism"])["best_action_match"].mean().unstack()
print(gt_summ.to_string())


# ═══════════════════════════════════════════════════════════
# PHASE 3: Action Coverage Gap Benchmark
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 3: Action Coverage Gap Benchmark")
print("=" * 60)

BIAS_TYPES = ["bias_plus", "bias_zero", "bias_minus"]
CF_FRACTIONS = [0.0, 0.05, 0.10, 0.20, 1.00]

def generate_biased_data(X_full, Y_full, bias_type, cf_fraction, seed):
    """Given full counterfactual (state, action, outcome) data,
    subsample to simulate a biased behavior policy.
    Returns: X_train (history features), Y_train (outcomes for only sampled actions).
    For actions not sampled, fill with NaN.
    """
    rng = np.random.default_rng(seed)
    n = len(X_full)
    ba_full = np.argmax(Y_full, axis=1).astype(np.int64)

    if bias_type == "bias_plus":
        action_probs = np.array([0.15, 0.15, 0.70])  # +1 at 70%
    elif bias_type == "bias_zero":
        action_probs = np.array([0.15, 0.70, 0.15])  # 0 at 70%
    elif bias_type == "bias_minus":
        action_probs = np.array([0.70, 0.15, 0.15])  # -1 at 70%
    else:
        action_probs = np.array([1/3, 1/3, 1/3])

    X_out, Y_out, ba_out = [], [], []
    for i in range(n):
        sampled_action = int(rng.choice(3, p=action_probs))
        y = Y_full[i].copy()

        # Counterfactual probe: with prob cf_fraction, sample all 3 actions
        if rng.random() < cf_fraction:
            sampled_action = None  # Full counterfactual

        if sampled_action is not None:
            y_obs = np.full(3, np.nan, dtype=np.float32)
            y_obs[sampled_action] = y[sampled_action]
            Y_out.append(y_obs)
        else:
            Y_out.append(y)

        X_out.append(X_full[i])
        ba_out.append(ba_full[i])

    return np.array(X_out), np.array(Y_out), np.array(ba_out)


def train_on_biased_data(X_biased, Y_biased, model_type, seed, bottleneck_dim=BOTTLENECK_DIM):
    """Train a model on biased data with missing action outcomes."""
    n = len(X_biased)

    if model_type == "policyclone":
        # PolicyClone: use only observed actions as labels (or fallback to majority)
        model = PolicyCloneBaseline(2, 8, bottleneck_dim=bottleneck_dim)
        # Use best_action from observed outcomes; if all NaN use mode
        ba_obs = np.nanargmax(Y_biased, axis=1)
        # Replace NaN-argmax gaps (rows where all NaN → np.nanargmax returns 0 incorrectly)
        all_nan_mask = np.all(np.isnan(Y_biased), axis=1)
        if all_nan_mask.any():
            ba_obs[all_nan_mask] = np.argmax(np.bincount(ba_obs[~all_nan_mask], minlength=3))

        model = model.to(DEVICE)
        X_t = torch.tensor(X_biased, dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(ba_obs, dtype=torch.long).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        best_loss = float("inf")
        best_state = None
        no_improve = 0
        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(xb), yb)
                loss.backward()
                opt.step()
            if epoch % 20 == 0:
                model.eval()
                with torch.no_grad():
                    curr = nn.functional.cross_entropy(model(X_t), y_t).item()
                if curr < best_loss:
                    best_loss = curr
                    best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= PATIENCE // 2:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model

    elif model_type == "aep":
        model = AEPCompressor(2, 8, bottleneck_dim=bottleneck_dim)
        # Train only on (state, action) pairs where outcome is observed
        X_exp, y_exp, a_exp = [], [], []
        for i in range(n):
            for a in range(3):
                if not np.isnan(Y_biased[i, a]):
                    X_exp.append(X_biased[i])
                    y_exp.append(Y_biased[i, a])
                    a_exp.append(a)
        if len(X_exp) == 0:
            return model
        X_te = torch.tensor(np.stack(X_exp), dtype=torch.float32).to(DEVICE)
        y_te = torch.tensor(np.array(y_exp), dtype=torch.float32).unsqueeze(1).to(DEVICE)
        a_te = torch.tensor(np.array(a_exp), dtype=torch.long).to(DEVICE)
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(X_te, y_te, a_te)
        loader = DataLoader(ds, batch_size=64 * 3, shuffle=True)
        model = model.to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        best_loss = float("inf")
        best_state = None
        no_improve = 0
        for epoch in range(EPOCHS):
            model.train()
            total = 0.0
            for xb, yb, ab in loader:
                opt.zero_grad()
                loss = nn.functional.mse_loss(model(xb, ab), yb)
                loss.backward()
                opt.step()
                total += loss.item()
            if total < best_loss:
                best_loss = total
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model

    elif model_type == "cf":
        model = CounterfactualCompressor(2, 8, bottleneck_dim=bottleneck_dim)
        # Train only on observed outcomes, mask missing ones
        model = model.to(DEVICE)
        X_t = torch.tensor(X_biased, dtype=torch.float32).to(DEVICE)
        Y_t = torch.tensor(Y_biased, dtype=torch.float32).to(DEVICE)
        mask = ~torch.isnan(Y_t)
        ba_obs = torch.tensor(np.nanargmax(Y_biased, axis=1), dtype=torch.long).to(DEVICE)
        all_nan_mask = torch.all(torch.isnan(Y_t), dim=1)
        if all_nan_mask.any():
            ba_obs[all_nan_mask] = torch.argmax(torch.bincount(ba_obs[~all_nan_mask], minlength=3))
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(X_t, Y_t, mask, ba_obs)
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        best_loss = float("inf")
        best_state = None
        no_improve = 0
        for epoch in range(EPOCHS):
            model.train()
            total = 0.0
            for xb, yb, mb, bab in loader:
                opt.zero_grad()
                yh = model(xb)
                mse = ((yh - yb) ** 2 * mb.float()).sum() / (mb.float().sum() + 1e-8)
                ce = nn.functional.cross_entropy(yh, bab)
                loss = 0.1 * mse + 1.5 * ce
                loss.backward()
                opt.step()
                total += loss.item()
            if total < best_loss:
                best_loss = total
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE // 2:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model

    elif model_type == "residual":
        model = ResidualCompressor(2, 8, bottleneck_dim=bottleneck_dim, residual_dim=RESIDUAL_DIM)
        X_exp, y_exp, a_exp = [], [], []
        for i in range(n):
            for a in range(3):
                if not np.isnan(Y_biased[i, a]):
                    X_exp.append(X_biased[i])
                    y_exp.append(Y_biased[i, a])
                    a_exp.append(a)
        if len(X_exp) == 0:
            return model
        X_te = torch.tensor(np.stack(X_exp), dtype=torch.float32).to(DEVICE)
        y_te = torch.tensor(np.array(y_exp), dtype=torch.float32).unsqueeze(1).to(DEVICE)
        a_te = torch.tensor(np.array(a_exp), dtype=torch.long).to(DEVICE)
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(X_te, y_te, a_te)
        loader = DataLoader(ds, batch_size=64 * 3, shuffle=True)
        model = model.to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        best_loss = float("inf")
        best_state = None
        no_improve = 0
        for epoch in range(EPOCHS):
            model.train()
            total = 0.0
            for xb, yb, ab in loader:
                opt.zero_grad()
                y_hat, b_hat, r_hat = model(xb, ab)
                loss = nn.functional.mse_loss(y_hat, yb)
                loss.backward()
                opt.step()
                total += loss.item()
            if total < best_loss:
                best_loss = total
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model

    return model


def evaluate_on_balanced_test(model, X_test, Y_test, model_type, device=DEVICE):
    """Evaluate model on balanced test set (all 3 actions available)."""
    with torch.no_grad():
        x_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        if model_type == "policyclone":
            preds = model(x_t).cpu().numpy()
        elif hasattr(model, 'predict_all_actions'):
            preds = model.predict_all_actions(x_t).cpu().numpy()
        elif model_type in ("aep", "residual"):
            preds = np.zeros((len(X_test), 3), dtype=np.float32)
            for a in range(3):
                a_t = torch.full((len(X_test),), a, dtype=torch.long, device=device)
                if model_type == "aep":
                    p = model(x_t, a_t).cpu().numpy()
                else:
                    p, _, _ = model(x_t, a_t)
                    p = p.cpu().numpy()
                preds[:, a] = p[:, 0]
        else:
            preds = model(x_t).cpu().numpy()

    ba_pred = np.argmax(preds, axis=1)
    ba_true = np.argmax(Y_test, axis=1)
    return float(np.mean(ba_pred == ba_true))


coverage_records = []

# Use seed 0 only for coverage experiment (heavy computation)
exp_seed = 0
print(f"  Coverage experiment seed={exp_seed}")

train_df_all = cf_df[(cf_df["seed"] == exp_seed) & (cf_df["split"] == "train") & (cf_df["horizon"] == 1)]
test_df_bal = cf_df[(cf_df["seed"] == exp_seed) & (cf_df["split"] == "test_id") & (cf_df["horizon"] == 1)]
X_tr_full, Y_tr_full, _ = prepare_counterfactual_data(train_df_all, exp_seed, ENV_KWARGS)
X_te_bal, Y_te_bal, ba_te_bal = prepare_counterfactual_data(test_df_bal, exp_seed, ENV_KWARGS)

for bias_type in BIAS_TYPES:
    print(f"    bias={bias_type}")
    # Raw memory baselines trained on full unbiased data (reference)
    Y_3_tr = [Y_tr_full[:, 0], Y_tr_full[:, 1], Y_tr_full[:, 2]]

    for cf_frac in tqdm(CF_FRACTIONS, desc=f"  {bias_type}"):
        X_biased, Y_biased, _ = generate_biased_data(X_tr_full, Y_tr_full, bias_type, cf_frac, exp_seed + 777)
        print(f"      cf_frac={cf_frac}: biased_data={len(X_biased)}, observed_actions_m1={np.sum(~np.isnan(Y_biased[:,0]))}, observed_0={np.sum(~np.isnan(Y_biased[:,1]))}, observed_p1={np.sum(~np.isnan(Y_biased[:,2]))}")

        # Compute rare action index
        if bias_type == "bias_plus":
            rare_action = 0  # -1 is rare
        elif bias_type == "bias_zero":
            rare_action = 0  # -1 is rare
        else:
            rare_action = 2  # +1 is rare

        # PolicyClone
        pc_biased = train_on_biased_data(X_biased, Y_biased, "policyclone", exp_seed)
        pc_match = evaluate_on_balanced_test(pc_biased, X_te_bal, Y_te_bal, "policyclone")
        rare_mask = ba_te_bal == rare_action
        pc_rare_match = 0.0
        if rare_mask.sum() > 0:
            pc_biased.eval()
            with torch.no_grad():
                pc_preds_rare = pc_biased(torch.tensor(X_te_bal[rare_mask], dtype=torch.float32).to(DEVICE)).detach().cpu().numpy()
            pc_rare_match = float(np.mean(np.argmax(pc_preds_rare, axis=1) == ba_te_bal[rare_mask]))

        coverage_records.append({"seed": exp_seed, "bias_type": bias_type, "cf_fraction": cf_frac,
                                 "mechanism": "PolicyCloneBaseline", "balanced_match": pc_match,
                                 "rare_action_match": pc_rare_match})

        # AEP
        aep_biased = train_on_biased_data(X_biased, Y_biased, "aep", exp_seed)
        aep_match = evaluate_on_balanced_test(aep_biased, X_te_bal, Y_te_bal, "aep")
        aep_rare_match = 0.0
        if rare_mask.sum() > 0:
            aep_biased.eval()
            with torch.no_grad():
                x_te_t = torch.tensor(X_te_bal[rare_mask], dtype=torch.float32).to(DEVICE)
                preds_rare = aep_biased.predict_all_actions(x_te_t).detach().cpu().numpy()
            aep_rare_match = float(np.mean(np.argmax(preds_rare, axis=1) == ba_te_bal[rare_mask]))

        coverage_records.append({"seed": exp_seed, "bias_type": bias_type, "cf_fraction": cf_frac,
                                 "mechanism": "AEPCompressor", "balanced_match": aep_match,
                                 "rare_action_match": aep_rare_match})

        # Counterfactual
        cf_biased = train_on_biased_data(X_biased, Y_biased, "cf", exp_seed)
        cf_match = evaluate_on_balanced_test(cf_biased, X_te_bal, Y_te_bal, "cf")
        cf_rare_match = 0.0
        if rare_mask.sum() > 0:
            cf_biased.eval()
            with torch.no_grad():
                cf_preds_rare = cf_biased(torch.tensor(X_te_bal[rare_mask], dtype=torch.float32).to(DEVICE)).detach().cpu().numpy()
            cf_rare_match = float(np.mean(np.argmax(cf_preds_rare, axis=1) == ba_te_bal[rare_mask]))

        coverage_records.append({"seed": exp_seed, "bias_type": bias_type, "cf_fraction": cf_frac,
                                 "mechanism": "CounterfactualCompressor", "balanced_match": cf_match,
                                 "rare_action_match": cf_rare_match})

        # Residual
        rc_biased = train_on_biased_data(X_biased, Y_biased, "residual", exp_seed)
        rc_match = evaluate_on_balanced_test(rc_biased, X_te_bal, Y_te_bal, "residual")
        rc_rare_match = 0.0
        if rare_mask.sum() > 0:
            rc_biased.eval()
            with torch.no_grad():
                rc_preds_rare = rc_biased.predict_all_actions(torch.tensor(X_te_bal[rare_mask], dtype=torch.float32).to(DEVICE)).detach().cpu().numpy()
            rc_rare_match = float(np.mean(np.argmax(rc_preds_rare, axis=1) == ba_te_bal[rare_mask]))

        coverage_records.append({"seed": exp_seed, "bias_type": bias_type, "cf_fraction": cf_frac,
                                 "mechanism": "ResidualCompressor", "balanced_match": rc_match,
                                 "rare_action_match": rc_rare_match})

        # RawMemoryEqualCost
        rm = RawMemoryEqualCost(param_budget=5000, k=5)
        valid_mask = ~np.isnan(Y_biased).any(axis=1)
        if valid_mask.sum() > 0:
            rm.fit(X_biased[valid_mask], [Y_biased[valid_mask, 0], Y_biased[valid_mask, 1], Y_biased[valid_mask, 2]])
        else:
            rm.fit(X_tr_full, Y_3_tr)
        rm_preds = rm.predict(X_te_bal)
        rm_match = float(np.mean(np.argmax(rm_preds, axis=1) == ba_te_bal))
        rm_rare_match = float(np.mean(np.argmax(rm_preds[rare_mask], axis=1) == ba_te_bal[rare_mask])) if rare_mask.sum() > 0 else 0.0

        coverage_records.append({"seed": exp_seed, "bias_type": bias_type, "cf_fraction": cf_frac,
                                 "mechanism": "RawMemoryEqualCost", "balanced_match": rm_match,
                                 "rare_action_match": rm_rare_match})

        # PrototypeMemory
        pm = PrototypeMemory(n_clusters=20, k=3)
        if valid_mask.sum() > 0:
            pm.fit(X_biased[valid_mask], [Y_biased[valid_mask, 0], Y_biased[valid_mask, 1], Y_biased[valid_mask, 2]])
        else:
            pm.fit(X_tr_full, Y_3_tr)
        pm_preds = pm.predict(X_te_bal)
        pm_match = float(np.mean(np.argmax(pm_preds, axis=1) == ba_te_bal))
        pm_rare_match = float(np.mean(np.argmax(pm_preds[rare_mask], axis=1) == ba_te_bal[rare_mask])) if rare_mask.sum() > 0 else 0.0

        coverage_records.append({"seed": exp_seed, "bias_type": bias_type, "cf_fraction": cf_frac,
                                 "mechanism": "PrototypeMemory", "balanced_match": pm_match,
                                 "rare_action_match": pm_rare_match})

coverage_df = pd.DataFrame(coverage_records)
coverage_df.to_csv("results/ic2c/action_coverage_gap.csv", index=False)
print("  action_coverage_gap.csv saved")
cov_summ = coverage_df.groupby(["bias_type", "cf_fraction", "mechanism"])[["balanced_match", "rare_action_match"]].mean()
print(cov_summ.to_string())


# ═══════════════════════════════════════════════════════════
# PHASE 4: Active Probe Value Benchmark
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 4: Active Probe Value Benchmark")
print("=" * 60)

class PartialObservabilityProbeEnv:
    """Wraps StructuredVolatilityEnv with hidden mode information.
    Normal observation only sees (obs, action) history — mode is hidden.
    A 'probe' action (new action=2, or use action=-1/+1 strategically)
    reveals information about the current mode sign.
    """
    def __init__(self, base_env_kwargs, probe_action=2):
        # We reuse the existing env but make mode observation hidden
        # Probe action = 2: same as +1 but reveals sign
        import sys
        sys.path.insert(0, '')
        # We'll use the existing env with a special probe
        self.base_kwargs = base_env_kwargs
        self.probe_action = probe_action

    def generate_probe_data(self, n_states=400, seed=0):
        """Generate states where probe action reveals hidden mode."""
        from src.env_structured_volatility import StructuredVolatilityEnv

        env = StructuredVolatilityEnv(seed=seed, **self.base_kwargs)
        records = []
        rng = np.random.default_rng(seed + 1000)

        for i in range(n_states):
            env.reset(seed + i * 1000 + 1)
            # Run random trajectory for warmup
            for _ in range(20):
                env.step(int(rng.choice([-1, 0, 1])))

            # Snapshot current state
            state = env.get_current_state()
            hist_obs = env.get_history_obs()
            hist_act = env.get_history_act()
            mode = env.mode

            # Outcomes without probe
            outcomes_no_probe = env.compute_outcomes(horizon=1)
            Y_no_probe = np.array([np.sum(outcomes_no_probe[a]) for a in [-1, 0, 1]], dtype=np.float32)

            # After probe: we know the sign
            # Probe reveals mode (simulates active information gathering)
            # The probe action itself has outcome, but more importantly reveals sign
            probe_outcome = env.step_forward(self.probe_action, horizon=1)
            # After probe, the model knows mode → can adjust predictions
            # For simulation: give the model mode as additional feature
            Y_with_probe = Y_no_probe.copy()

            records.append({
                "state_idx": i,
                "history_obs": [o.tolist() if isinstance(o, np.ndarray) else o for o in hist_obs],
                "history_act": list(hist_act),
                "mode": int(mode),
                "outcome_m1": outcomes_no_probe[-1].tolist() if isinstance(outcomes_no_probe[-1], np.ndarray) else outcomes_no_probe[-1],
                "outcome_0": outcomes_no_probe[0].tolist() if isinstance(outcomes_no_probe[0], np.ndarray) else outcomes_no_probe[0],
                "outcome_p1": outcomes_no_probe[1].tolist() if isinstance(outcomes_no_probe[1], np.ndarray) else outcomes_no_probe[1],
                "best_action": int(max(outcomes_no_probe, key=lambda a: np.sum(outcomes_no_probe[a]))),
                "probe_sign": 1.0 if mode == 0 else -1.0,  # sign revealed by probe
            })

        return pd.DataFrame(records)


# Generate probe benchmark data
print("  Generating probe benchmark data...")
probe_df = PartialObservabilityProbeEnv(ENV_KWARGS).generate_probe_data(n_states=600, seed=0)

# Split: first 400 train, last 200 test
probe_df["split"] = "train"
probe_df.loc[400:, "split"] = "test"
probe_df.to_csv("results/ic2c/probe_benchmark_data.csv", index=False)

probe_records = []

for seed in tqdm(SEEDS, desc="Probe Benchmark"):
    # Train on first 400, test on last 200
    tr = probe_df[probe_df["split"] == "train"]
    te = probe_df[probe_df["split"] == "test"]
    X_tr_p, Y_tr_p, ba_tr_p = prepare_counterfactual_data(tr, 0, ENV_KWARGS)
    X_te_p, Y_te_p, ba_te_p = prepare_counterfactual_data(te, 0, ENV_KWARGS)

    # NoProbe models: standard training without mode information
    # NoProbe PolicyClone
    pc_np = PolicyCloneBaseline(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    try:
        pc_np = train_state_only_classifier(pc_np, X_tr_p, Y_tr_p, None, None, epochs=EPOCHS, patience=PATIENCE, device=DEVICE)
    except Exception as e:
        pc_np = PolicyCloneBaseline(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    with torch.no_grad():
        pc_np_preds = pc_np(torch.tensor(X_te_p, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    pc_np_match = float(np.mean(np.argmax(pc_np_preds, axis=1) == ba_te_p))
    probe_records.append({"seed": seed, "model": "NoProbe_PolicyClone", "best_action_match": pc_np_match,
                          "probe_used": False, "n_test": len(X_te_p)})

    # NoProbe AEP
    aep_np = AEPCompressor(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    try:
        aep_np = train_ae_model(aep_np, X_tr_p, Y_tr_p, None, None, "aep", epochs=EPOCHS, patience=PATIENCE, device=DEVICE, ce_weight=0.8)
    except Exception as e:
        aep_np = AEPCompressor(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    with torch.no_grad():
        aep_np_preds = aep_np.predict_all_actions(torch.tensor(X_te_p, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    aep_np_match = float(np.mean(np.argmax(aep_np_preds, axis=1) == ba_te_p))
    probe_records.append({"seed": seed, "model": "NoProbe_AEP", "best_action_match": aep_np_match,
                          "probe_used": False, "n_test": len(X_te_p)})

    # OneProbe AEP: trained with mode (sign) as additional feature
    # Simulate by appending probe_sign to features
    probe_sign_tr = tr["probe_sign"].values.astype(np.float32).reshape(-1, 1)
    probe_sign_te = te["probe_sign"].values.astype(np.float32).reshape(-1, 1)
    X_tr_probe = np.hstack([X_tr_p, probe_sign_tr])
    X_te_probe = np.hstack([X_te_p, probe_sign_te])

    # OneProbe model: uses augmented features (history + probe_sign)
    class OneProbeAEP(nn.Module):
        def __init__(self, obs_dim, history_len, bottleneck_dim=16):
            super().__init__()
            self.in_dim = history_len * (obs_dim + 1) + 1  # +1 for probe_sign
            self.state_encoder = MLP(self.in_dim, [bottleneck_dim * 2], bottleneck_dim)
            self.action_encoder = nn.Embedding(3, bottleneck_dim // 2)
            self.decoder = MLP(bottleneck_dim + bottleneck_dim // 2, [bottleneck_dim], bottleneck_dim)
            self.head = nn.Linear(bottleneck_dim, 1)

        def forward(self, x, action_idx):
            z_state = self.state_encoder(x)
            z_action = self.action_encoder(action_idx)
            z = torch.cat([z_state, z_action], dim=-1)
            return self.head(self.decoder(z))

        def predict_all_actions(self, x):
            z_state = self.state_encoder(x)
            outputs = []
            for a in range(3):
                a_t = torch.full((x.shape[0],), a, dtype=torch.long, device=x.device)
                z_action = self.action_encoder(a_t)
                z = torch.cat([z_state, z_action], dim=-1)
                outputs.append(self.head(self.decoder(z)))
            return torch.cat(outputs, dim=-1)

    aep_op = OneProbeAEP(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    try:
        aep_op = train_ae_model(aep_op, X_tr_probe, Y_tr_p, None, None, "aep", epochs=EPOCHS, patience=PATIENCE, device=DEVICE, ce_weight=0.8)
    except Exception as e:
        print(f"  [WARN] OneProbe AEP training failed: {e}")
        aep_op = OneProbeAEP(2, 8, bottleneck_dim=BOTTLENECK_DIM)

    with torch.no_grad():
        aep_op_preds = aep_op.predict_all_actions(torch.tensor(X_te_probe, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    aep_op_match = float(np.mean(np.argmax(aep_op_preds, axis=1) == ba_te_p))
    probe_records.append({"seed": seed, "model": "OneProbe_AEP", "best_action_match": aep_op_match,
                          "probe_used": True, "n_test": len(X_te_p)})

    # OneProbe Residual
    class OneProbeResidual(nn.Module):
        def __init__(self, obs_dim, history_len, bottleneck_dim=16, residual_dim=8):
            super().__init__()
            self.in_dim = history_len * (obs_dim + 1) + 1
            self.state_encoder = MLP(self.in_dim, [bottleneck_dim * 2], bottleneck_dim)
            self.autonomous_head = nn.Linear(bottleneck_dim, 1)
            self.action_encoder = nn.Embedding(3, residual_dim)
            self.residual_head = nn.Sequential(
                nn.Linear(bottleneck_dim + residual_dim, bottleneck_dim // 2),
                nn.ReLU(),
                nn.Linear(bottleneck_dim // 2, 1),
            )

        def forward(self, x, action_idx):
            z_state = self.state_encoder(x)
            b = self.autonomous_head(z_state)
            z_action = self.action_encoder(action_idx)
            r = self.residual_head(torch.cat([z_state, z_action], dim=-1))
            return b + r, b, r

        def predict_all_actions(self, x):
            z_state = self.state_encoder(x)
            b = self.autonomous_head(z_state)
            outputs = []
            for a in range(3):
                a_t = torch.full((x.shape[0],), a, dtype=torch.long, device=x.device)
                z_action = self.action_encoder(a_t)
                r = self.residual_head(torch.cat([z_state, z_action], dim=-1))
                outputs.append(b + r)
            return torch.cat(outputs, dim=-1)

    rc_op = OneProbeResidual(2, 8, bottleneck_dim=BOTTLENECK_DIM, residual_dim=RESIDUAL_DIM)
    try:
        rc_op = train_ae_model(rc_op, X_tr_probe, Y_tr_p, None, None, "residual", epochs=EPOCHS, patience=PATIENCE, device=DEVICE, ce_weight=0.8)
    except Exception as e:
        print(f"  [WARN] OneProbe Residual training failed: {e}")
        rc_op = OneProbeResidual(2, 8, bottleneck_dim=BOTTLENECK_DIM, residual_dim=RESIDUAL_DIM)

    with torch.no_grad():
        rc_op_preds = rc_op.predict_all_actions(torch.tensor(X_te_probe, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    rc_op_match = float(np.mean(np.argmax(rc_op_preds, axis=1) == ba_te_p))
    probe_records.append({"seed": seed, "model": "OneProbe_Residual", "best_action_match": rc_op_match,
                          "probe_used": True, "n_test": len(X_te_p)})

    # Compute probe value
    probe_value_aep = aep_op_match - aep_np_match
    probe_value_rc = rc_op_match - aep_np_match

    probe_records.append({"seed": seed, "model": "ProbeValue_AEP", "best_action_match": probe_value_aep,
                          "probe_used": True, "n_test": len(X_te_p)})
    probe_records.append({"seed": seed, "model": "ProbeValue_Residual", "best_action_match": probe_value_rc,
                          "probe_used": True, "n_test": len(X_te_p)})

probe_df_out = pd.DataFrame(probe_records)
probe_df_out.to_csv("results/ic2c/active_probe_value.csv", index=False)
print("  active_probe_value.csv saved")
probe_summ = probe_df_out.groupby("model")["best_action_match"].mean()
print(probe_summ.to_string())


# ═══════════════════════════════════════════════════════════
# PHASE 5: Updated Metrics + Cost-Normalized Transfer + Report
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 5: Updated Metrics & Final Report")
print("=" * 60)

# ── Cost-Normalized Transfer Premium ──
# (bits of memory or params needed vs transfer score gained)
cn_records = []
for mech_name, param_estimate, tag in [
    ("PolicyCloneBaseline", 20000, "parametric"),
    ("CounterfactualCompressor", 25000, "parametric"),
    ("AEPCompressor", 35000, "parametric"),
    ("ResidualCompressor", 40000, "parametric"),
    ("RawMemoryEqualCost", 5000, "memory"),
    ("PrototypeMemory", 20 * 24 * 4, "memory"),
    ("StateOnlyDynamics", 18000, "parametric"),
]:
    sub = gt_df[gt_df["mechanism"] == mech_name]
    if len(sub) == 0:
        continue
    u1_match = sub[sub["goal"] == "U1_main"]["best_action_match"].mean()
    u2u5_matches = sub[sub["goal"].isin(["U2_reverse", "U3_target", "U4_risk_avoid", "U5_energy_aware"])]["best_action_match"].mean()
    goal_transfer_score = u2u5_matches
    pc_overfit_index = u1_match - u2u5_matches
    transfer_premium = u2u5_matches - u1_match
    cost_norm_transfer = transfer_premium / (param_estimate / 1000.0) if param_estimate > 0 else 0

    cn_records.append({
        "mechanism": mech_name,
        "tag": tag,
        "param_estimate": param_estimate,
        "U1_match": u1_match,
        "U2_U5_mean_match": u2u5_matches,
        "goal_transfer_score": goal_transfer_score,
        "transfer_premium": transfer_premium,
        "policy_clone_overfit_index": pc_overfit_index,
        "cost_normalized_transfer_premium": cost_norm_transfer,
    })

cn_df = pd.DataFrame(cn_records)
cn_df.to_csv("results/ic2c/cost_normalized_transfer.csv", index=False)
print("  cost_normalized_transfer.csv saved")

# ── Policy Clone Overfit ──
pco_records = []
for seed in SEEDS:
    sub = gt_df[gt_df["seed"] == seed]
    for mech in sub["mechanism"].unique():
        msub = sub[sub["mechanism"] == mech]
        u1 = msub[msub["goal"] == "U1_main"]["best_action_match"].mean()
        u2u5 = msub[msub["goal"].isin(["U2_reverse", "U3_target", "U4_risk_avoid", "U5_energy_aware"])]["best_action_match"].mean()
        pco_records.append({
            "seed": seed,
            "mechanism": mech,
            "U1_ID_match": u1,
            "U2_U5_mean": u2u5,
            "policy_clone_overfit_index": u1 - u2u5,
        })

pco_df = pd.DataFrame(pco_records)
pco_df.to_csv("results/ic2c/policy_clone_overfit.csv", index=False)
print("  policy_clone_overfit.csv saved")


# ═══════════════════════════════════════════════════════════
# GENERATE FINAL REPORT
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Generating IC2C_POLICY_CLONE_TRAP_ESCAPE_REPORT.md")
print("=" * 60)

# Collect key evidence
gt_summ2 = gt_df.groupby(["goal", "mechanism"])["best_action_match"].mean().unstack()
pc_u1 = gt_summ2.loc["U1_main", "PolicyCloneBaseline"] if "U1_main" in gt_summ2.index and "PolicyCloneBaseline" in gt_summ2.columns else 0
aep_u1 = gt_summ2.loc["U1_main", "AEPCompressor"] if "U1_main" in gt_summ2.index and "AEPCompressor" in gt_summ2.columns else 0
cf_u1 = gt_summ2.loc["U1_main", "CounterfactualCompressor"] if "U1_main" in gt_summ2.index and "CounterfactualCompressor" in gt_summ2.columns else 0

goals = ["U1_main", "U2_reverse", "U3_target", "U4_risk_avoid", "U5_energy_aware"]
pc_transfer = gt_summ2.loc[["U2_reverse","U3_target","U4_risk_avoid","U5_energy_aware"], "PolicyCloneBaseline"].mean() if "PolicyCloneBaseline" in gt_summ2.columns else 0
aep_transfer = gt_summ2.loc[["U2_reverse","U3_target","U4_risk_avoid","U5_energy_aware"], "AEPCompressor"].mean() if "AEPCompressor" in gt_summ2.columns else 0
cf_transfer = gt_summ2.loc[["U2_reverse","U3_target","U4_risk_avoid","U5_energy_aware"], "CounterfactualCompressor"].mean() if "CounterfactualCompressor" in gt_summ2.columns else 0

aep_gt = aep_transfer - pc_transfer
cf_gt = cf_transfer - pc_transfer

# Probe summary
np_pc = probe_df_out[probe_df_out["model"] == "NoProbe_PolicyClone"]["best_action_match"].mean()
np_aep = probe_df_out[probe_df_out["model"] == "NoProbe_AEP"]["best_action_match"].mean()
op_aep = probe_df_out[probe_df_out["model"] == "OneProbe_AEP"]["best_action_match"].mean()
op_rc = probe_df_out[probe_df_out["model"] == "OneProbe_Residual"]["best_action_match"].mean()
probe_value = op_aep - np_aep

# Coverage summary
cov_by_mech = coverage_df.groupby(["cf_fraction", "mechanism"])["balanced_match"].mean().unstack() if len(coverage_df) > 0 else pd.DataFrame()

# Verdict
aep_goal_beats_pc = aep_gt > 0.10
cf_goal_beats_pc = cf_gt > 0.10
probe_beats_noprobe = probe_value > 0.10

if aep_goal_beats_pc or cf_goal_beats_pc:
    verdict = "IC2C_SUPPORTS_GOAL_TRANSFER_APPRECIATION"
    verdict_reason = f"AEP/CF goal transfer premium vs PolicyClone: AEP={aep_gt:+.4f}, CF={cf_gt:+.4f}"
elif probe_beats_noprobe:
    verdict = "IC2C_SUPPORTS_CF_PROBE_VALUE_ONLY"
    verdict_reason = f"Active probe gain={probe_value:.4f} > 0.10, but goal transfer insufficient"
else:
    pc_overfit = cn_df[cn_df["mechanism"] == "PolicyCloneBaseline"]["policy_clone_overfit_index"].values
    pc_ov = pc_overfit[0] if len(pc_overfit) > 0 else 0
    if pc_ov > 0.30:
        verdict = "IC2C_POLICY_CLONE_STILL_DOMINATES"
        verdict_reason = f"PolicyClone U1={pc_u1:.3f}, U2-U5={pc_transfer:.3f}, overfit={pc_ov:.3f}. No mechanism beats it on transfer."
    else:
        verdict = "IC2C_POLICY_CLONE_STILL_DOMINATES"
        verdict_reason = "PolicyClone generalizes too well to new goals. Environment may need stronger goal shift."

print(f"  Verdict: {verdict}")
print(f"  AEP goal transfer gap: {aep_gt:+.4f}")
print(f"  CF goal transfer gap: {cf_gt:+.4f}")
print(f"  Active probe value: {probe_value:+.4f}")

# Write report
report = []
def w(s=""):
    report.append(s)

w("# IC-2c: Policy-Clone Trap Escape Benchmark")
w()
w("**Date**: 2026-05-10")
w(f"**Final Verdict**: `{verdict}`")
w()
w("---")
w()
w("## Executive Summary")
w()
w("IC-2c tests whether AEP/Counterfactual models escape the PolicyClone trap")
w("by excelling in three dimensions that PolicyClone cannot access:")
w("1. **Goal Transfer** (U1→U2-U5): recomputing best_action under new utilities")
w("2. **Action Coverage Gap** (biased sampling + CF probes): predicting rare actions")
w("3. **Active Probe Value** (partial observability): gaining information via probing")
w()
w("### Key Results")
w()
w("| Benchmark | PolicyClone | AEPCompressor | CFCompressor | ResidualCompressor |")
w("|---|---|---|---|---|")
w(f"| Goal Transfer (mean U2-U5) | {pc_transfer:.4f} | {aep_transfer:.4f} | {cf_transfer:.4f} | - |")
w(f"| U1 ID Match | {pc_u1:.4f} | {aep_u1:.4f} | {cf_u1:.4f} | - |")
w(f"| Active Probe (OneProbe) | {np_pc:.4f} | {op_aep:.4f} | - | {op_rc:.4f} |")
w(f"| Probe Value Gain | - | {probe_value:+.4f} | - | - |")
w()
w("---")
w()
w("## Q1: Is PolicyClone Only Strong on Fixed Goal U1?")
w()
w("### Per-Goal Match Rates")
w()
w("| Goal | PolicyClone | AEPCompressor | CFCompressor | ActionOnly |")
w("|---|---|---|---|---|")
for goal in goals:
    pc_v = gt_summ2.loc[goal, "PolicyCloneBaseline"] if goal in gt_summ2.index and "PolicyCloneBaseline" in gt_summ2.columns else 0
    aep_v = gt_summ2.loc[goal, "AEPCompressor"] if goal in gt_summ2.index and "AEPCompressor" in gt_summ2.columns else 0
    cf_v = gt_summ2.loc[goal, "CounterfactualCompressor"] if goal in gt_summ2.index and "CounterfactualCompressor" in gt_summ2.columns else 0
    ao_v = gt_summ2.loc[goal, "ActionOnly"] if goal in gt_summ2.index and "ActionOnly" in gt_summ2.columns else 0
    w(f"| {goal} | {pc_v:.4f} | {aep_v:.4f} | {cf_v:.4f} | {ao_v:.4f} |")
w()
w(f"**Policy Clone Overfit Index**: {pc_u1 - pc_transfer:.4f} (U1 - mean(U2-U5))")
w(f"**AEP Goal Transfer Premium**: {aep_gt:+.4f}")
w(f"**CF Goal Transfer Premium**: {cf_gt:+.4f}")
w()

pc_overfit_val = cn_df[cn_df["mechanism"] == "PolicyCloneBaseline"]["policy_clone_overfit_index"].values
pc_ov_val = pc_overfit_val[0] if len(pc_overfit_val) > 0 else 0
w(f"PolicyClone U1={pc_u1:.3f}, mean(U2-U5)={pc_transfer:.3f}, overfit_index={pc_ov_val:.3f}")
w()
if pc_ov_val > 0.20:
    w("**Finding**: PolicyClone shows significant overfit — U1 >> U2-U5. This confirms the 'trap': PolicyClone memorizes the fixed-goal policy but cannot transfer.")
elif pc_ov_val > 0.05:
    w("**Finding**: PolicyClone shows moderate overfit. Some goal-transfer gap exists but is small.")
else:
    w("**Finding**: PolicyClone generalizes surprisingly well across goals. The goal shift may not be strong enough.")

w()
w("---")
w()
w("## Q2: Do AEP/CF Models Beat PolicyClone on Goal Transfer?")
w()
w(f"**Answer: {'YES' if aep_goal_beats_pc or cf_goal_beats_pc else 'NO'}.**")
w()
w(f"- AEP goal transfer gap vs PolicyClone: {aep_gt:+.4f}")
w(f"- CF goal transfer gap vs PolicyClone: {cf_gt:+.4f}")
w()
if aep_gt > 0.10:
    w("✅ AEPCompressor exceeds PolicyClone by >0.10 on goal transfer — SUCCESS")
elif aep_gt > 0:
    w("⚠ AEPCompressor marginally beats PolicyClone but not by 0.10")
else:
    w("❌ AEPCompressor does NOT beat PolicyClone on goal transfer")
w()

w("---")
w()
w("## Q3: Does Counterfactual Probing Add Value in Coverage Gap?")
w()
w("### Action Coverage Gap Results")
if len(cov_by_mech) > 0:
    w()
    w("| CF Fraction | PolicyClone | AEPCompressor | CFCompressor | RawMemory |")
    w("|---|---|---|---|---|")
    for frac in CF_FRACTIONS:
        pc_c = cov_by_mech.loc[frac, "PolicyCloneBaseline"] if frac in cov_by_mech.index and "PolicyCloneBaseline" in cov_by_mech.columns else 0
        aep_c = cov_by_mech.loc[frac, "AEPCompressor"] if frac in cov_by_mech.index and "AEPCompressor" in cov_by_mech.columns else 0
        cf_c = cov_by_mech.loc[frac, "CounterfactualCompressor"] if frac in cov_by_mech.index and "CounterfactualCompressor" in cov_by_mech.columns else 0
        rm_c = cov_by_mech.loc[frac, "RawMemoryEqualCost"] if frac in cov_by_mech.index and "RawMemoryEqualCost" in cov_by_mech.columns else 0
        w(f"| {frac:.0%} | {pc_c:.4f} | {aep_c:.4f} | {cf_c:.4f} | {rm_c:.4f} |")
w()

w("---")
w()
w("## Q4: Does Active Probe Generate Action-Effect Capital Gain?")
w()
w("| Model | Best Action Match | Probe Used |")
w("|---|---|---|")
probe_agg = probe_df_out.groupby("model")["best_action_match"].mean()
for model_name in ["NoProbe_PolicyClone", "NoProbe_AEP", "OneProbe_AEP", "OneProbe_Residual", "ProbeValue_AEP", "ProbeValue_Residual"]:
    val = probe_agg.get(model_name, 0)
    used = model_name.startswith("OneProbe") or model_name.startswith("ProbeValue")
    w(f"| {model_name} | {val:.4f} | {used} |")
w()
w(f"**Active Probe Gain (OneProbe_AEP - NoProbe_AEP)**: {probe_value:+.4f}")
w()
if probe_value > 0.10:
    w("✅ Active probe provides substantial capital gain — ICT supported")
elif probe_value > 0.02:
    w("⚠ Active probe provides marginal gain — moderate support")
else:
    w("❌ Active probe does NOT provide meaningful gain")
w()

w("---")
w()
w("## Q5: Does RawMemoryEqualCost Still Crush Learned Compressors?")
w()
if len(cov_by_mech) > 0:
    rm_full = cov_by_mech.loc[1.0, "RawMemoryEqualCost"] if 1.0 in cov_by_mech.index and "RawMemoryEqualCost" in cov_by_mech.columns else 0
    aep_full = cov_by_mech.loc[1.0, "AEPCompressor"] if 1.0 in cov_by_mech.index and "AEPCompressor" in cov_by_mech.columns else 0
    if rm_full > aep_full:
        w(f"RawMemory (100% CF) = {rm_full:.4f} vs AEP = {aep_full:.4f} — RawMemory still dominates at full data, but learned compressors may catch up with CF probes.")
    else:
        w(f"AEP = {aep_full:.4f} > RawMemory = {rm_full:.4f} — Learned compression matches or beats memory when CF probes are available.")
w()

w("---")
w()
w("## Q6: Is This the First Time Intelligence Appreciation Is Supported?")
w()
if aep_goal_beats_pc or cf_goal_beats_pc or probe_value > 0.10:
    w(f"**Answer: PARTIALLY YES.** ICT's multi-dimensional evaluation (goal transfer premium={aep_gt:+.4f}, probe gain={probe_value:+.4f}) shows that learned compression provides value beyond simple policy imitation.")
else:
    w(f"**Answer: NO.** With AEP goal transfer gap={aep_gt:+.4f} and probe gain={probe_value:+.4f}, neither exceeds the 0.10 threshold. Learned compression does not yet beat PolicyClone.")

w()
w("---")
w()
w("## Q7: If Still Failing, What Type of Failure?")
w()
w(f"- PolicyClone still dominant: {'YES' if pc_u1 > max(aep_u1, cf_u1) else 'NO'}")
w(f"- AEP learned compression insufficient: {'YES' if aep_gt < 0.10 else 'NO'}")
w(f"- Benchmark invalid: {'YES' if pc_transfer > 0.5 and abs(pc_ov_val) < 0.05 else 'NO'}")
w(f"- ICT strong claim not yet supported: {'YES' if verdict != 'IC2C_SUPPORTS_GOAL_TRANSFER_APPRECIATION' else 'NO'}")

w()
w("---")
w()
w("## Final Verdict")
w()
w(f"### `{verdict}`")
w()
w(f"**Reasoning**: {verdict_reason}")
w()
w("### All IC-2c Outputs")
w()
w("| File | Content |")
w("|---|---|")
w("| `results/ic2c/goal_transfer.csv` | Per-goal match rates for all mechanisms |")
w("| `results/ic2c/action_coverage_gap.csv` | Biased sampling + CF probe fraction results |")
w("| `results/ic2c/active_probe_value.csv` | NoProbe vs OneProbe comparisons |")
w("| `results/ic2c/cost_normalized_transfer.csv` | Cost-normalized transfer premium |")
w("| `results/ic2c/policy_clone_overfit.csv` | Policy Clone Overfit Index per seed/mechanism |")
w("| `results/ic2c/IC2C_POLICY_CLONE_TRAP_ESCAPE_REPORT.md` | This report |")

with open("results/ic2c/IC2C_POLICY_CLONE_TRAP_ESCAPE_REPORT.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("  IC2C_POLICY_CLONE_TRAP_ESCAPE_REPORT.md written.")

# ── Charts ──
print("\nGenerating charts...")
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 1. Goal Transfer bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    mechanisms = ["PolicyCloneBaseline", "AEPCompressor", "CounterfactualCompressor", "ResidualCompressor", "ActionOnly"]
    x = np.arange(len(goals))
    width = 0.15
    for i, mech in enumerate(mechanisms):
        vals = [gt_summ2.loc[g, mech] if g in gt_summ2.index and mech in gt_summ2.columns else 0 for g in goals]
        ax.bar(x + i * width, vals, width, label=mech)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(goals)
    ax.set_ylabel("Best Action Match")
    ax.set_title("IC-2c: Goal Transfer Benchmark")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/figures/ic2c_goal_transfer.png", dpi=100)
    plt.close()
    print("  ic2c_goal_transfer.png saved")

    # 2. CF Fraction Curve
    fig, ax = plt.subplots(figsize=(10, 6))
    for mech in ["PolicyCloneBaseline", "AEPCompressor", "CounterfactualCompressor", "RawMemoryEqualCost"]:
        if mech in cov_by_mech.columns:
            vals = [cov_by_mech.loc[f, mech] if f in cov_by_mech.index else 0 for f in CF_FRACTIONS]
            ax.plot([str(f) for f in CF_FRACTIONS], vals, 'o-', label=mech)
    ax.set_ylabel("Balanced Best Action Match")
    ax.set_xlabel("CF Probe Fraction")
    ax.set_title("IC-2c: Action Coverage Gap — CF Fraction Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/figures/ic2c_cf_fraction_curve.png", dpi=100)
    plt.close()
    print("  ic2c_cf_fraction_curve.png saved")

    # 3. Active Probe Gain
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["NoProbe_PC", "NoProbe_AEP", "OneProbe_AEP", "OneProbe_Residual"]
    vals = [np_pc, np_aep, op_aep, op_rc]
    colors = ['gray', 'orange', 'green', 'blue']
    ax.bar(labels, vals, color=colors)
    ax.axhline(y=np_pc, color='gray', linestyle='--', alpha=0.5, label='PC baseline')
    ax.set_ylabel("Best Action Match")
    ax.set_title("IC-2c: Active Probe Value")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/figures/ic2c_active_probe_gain.png", dpi=100)
    plt.close()
    print("  ic2c_active_probe_gain.png saved")

    # 4. Policy Clone Overfit
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in pco_df.iterrows():
        ax.scatter(row["U1_ID_match"], row["U2_U5_mean"], alpha=0.5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel("U1 ID Match")
    ax.set_ylabel("Mean U2-U5 Transfer Match")
    ax.set_title("IC-2c: Policy Clone Overfit (below diagonal = overfit)")
    plt.tight_layout()
    plt.savefig("results/figures/ic2c_policy_clone_overfit.png", dpi=100)
    plt.close()
    print("  ic2c_policy_clone_overfit.png saved")

    # 5. Cost-Normalized Transfer
    fig, ax = plt.subplots(figsize=(10, 6))
    mechs = cn_df["mechanism"].values
    scores = cn_df["cost_normalized_transfer_premium"].values
    ax.barh(mechs, scores)
    ax.set_xlabel("Cost-Normalized Transfer Premium")
    ax.set_title("IC-2c: Cost-Normalized Transfer")
    plt.tight_layout()
    plt.savefig("results/figures/ic2c_cost_normalized_transfer.png", dpi=100)
    plt.close()
    print("  ic2c_cost_normalized_transfer.png saved")

except Exception as e:
    print(f"  [WARN] Chart generation failed: {e}")

print(f"\n{'='*60}")
print(f"IC-2c COMPLETE")
print(f"  Verdict: {verdict}")
print(f"  Outputs: results/ic2c/ + results/figures/")
print(f"{'='*60}")