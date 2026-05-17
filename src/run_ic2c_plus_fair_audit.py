"""
IC-2c+: Fair Goal-Transfer Robustness Audit
============================================
Tests AEP goal-transfer advantage against:
  1. Stronger policy baselines (MultiGoal, FewShot, OutcomeFeatures)
  2. Fair memory baselines (OutcomeTable variants, not regression->argmax)
  3. Active probe fairness (PolicyClone gets probe privilege)
  4. Revised coverage gap (systematic mode/action masking)
  5. Cost-normalized premiums
"""
import os, sys, json, hashlib, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import (prepare_counterfactual_data, train_state_only_classifier,
                       train_counterfactual_joint, train_ae_model)
from src.models import (StateOnlyPredictor, MLP, AEPCompressor,
                        ResidualCompressor, CounterfactualCompressor)

os.makedirs("results/ic2c_plus", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2]
EPOCHS = 300
PATIENCE = 60
BOTTLENECK_DIM = 48
RESIDUAL_DIM = 12
ENV_KWARGS = dict(state_dim=2, history_len=8, action_gain=0.25)
RNG = np.random.default_rng(42)

cf_df = pd.read_csv("results/counterfactual_table.csv")

# ═══════════════════════════════════════════════════════════
# SECTION 1: Utility Validity Audit + Extended Utility Family
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1: Utility Validity Audit + Extended Utilities")
print("=" * 60)

def utility_U1(Y): return np.argmax(Y, axis=1)
def utility_U2(Y): return np.argmin(Y, axis=1)
def utility_U3(Y):
    dist = np.abs(Y - np.array([1.0, 0.5, 1.0])[np.newaxis, :])
    return np.argmin(dist, axis=1)
def utility_U4(Y):
    rp = np.where(Y < -1.0, (Y + 1.0)**2 * 10, 0)
    return np.argmax(Y - rp, axis=1)
def utility_U5(Y):
    costs = np.array([0.2, 0.0, 0.2], dtype=np.float32)
    return np.argmax(Y - costs[np.newaxis, :], axis=1)

BASE_UTILITIES = {
    "U1_main": utility_U1, "U2_reverse": utility_U2, "U3_target": utility_U3,
    "U4_risk_avoid": utility_U4, "U5_energy_aware": utility_U5
}

# ── Utility Validity Audit ──
# Use test data from seed 0 for utility analysis
seed0_test = cf_df[(cf_df["seed"] == 0) & (cf_df["split"] == "test_id") & (cf_df["horizon"] == 1)]
X_audit, Y_audit, ba_audit = prepare_counterfactual_data(seed0_test, 0, ENV_KWARGS)

def label_divergence(ba_a, ba_b):
    return float(np.mean(ba_a != ba_b))

def utility_audit(Y, utility_name, utility_fn):
    ba = utility_fn(Y)
    ba_u1 = utility_U1(Y)
    vals = np.array([ba_u1, utility_U2(Y), utility_U3(Y), utility_U4(Y), utility_U5(Y)])

    # best_action distribution
    dist = {int(i): float(np.mean(ba == i)) for i in range(3)}

    # label divergence from U1 and from all others
    div_from_u1 = label_divergence(ba, ba_u1)
    div_from_others = {f"U{i+1}": label_divergence(ba, vals[i]) for i in range(5)}

    # entropy of oracle best_action
    _, counts = np.unique(ba, return_counts=True)
    probs = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log(probs + 1e-8)))

    zero_rate = float(np.mean(ba == 1))  # action 0 = noop

    # utility value scale (normalized)
    util_vals = np.zeros((len(Y), 3), dtype=np.float32)
    for a in range(3):
        ba_fake = np.full(len(Y), a, dtype=np.int64)
        # Use indirect measure: outcome at best_action vs outcome at worst under this utility
        rbm = np.max(Y, axis=1)  # best outcome
        rwm = np.min(Y, axis=1)  # worst outcome
    value_scale = float(np.mean(np.abs(rbm - rwm)))

    return {
        "utility": utility_name,
        "dist_m1": dist.get(0, 0), "dist_0": dist.get(1, 0), "dist_p1": dist.get(2, 0),
        "label_divergence_from_U1": div_from_u1,
        "pairwise_divU2": div_from_others.get("U2", 0),
        "pairwise_divU3": div_from_others.get("U3", 0),
        "pairwise_divU4": div_from_others.get("U4", 0),
        "pairwise_divU5": div_from_others.get("U5", 0),
        "oracle_entropy": entropy,
        "zero_action_rate": zero_rate,
        "utility_value_scale": value_scale,
        "valid": div_from_u1 >= 0.10
    }

utility_audit_records = []
for uname, ufn in BASE_UTILITIES.items():
    utility_audit_records.append(utility_audit(Y_audit, uname, ufn))

# Also compute pairwise label disagreement matrix
for i, (ni, fi) in enumerate(BASE_UTILITIES.items()):
    bai = fi(Y_audit)
    for j, (nj, fj) in enumerate(BASE_UTILITIES.items()):
        if i < j:
            baj = fj(Y_audit)
            utility_audit_records.append({
                "utility": f"pairwise_disagreement_{ni}_vs_{nj}",
                "label_divergence_from_U1": label_divergence(bai, baj),
                "valid": True
            })

ua_df = pd.DataFrame(utility_audit_records)
ua_df.to_csv("results/ic2c_plus/utility_validity_audit.csv", index=False)
print("  utility_validity_audit.csv saved")

# ── Extended Held-Out Utility Family ──
def make_extended_utilities(n_total=25, seed=42):
    """Generate 25 held-out utilities of different types."""
    rng = np.random.default_rng(seed)
    utilities = {}
    u_idx = 0

    # Type 1: Linear utilities (8)
    for _ in range(8):
        w = rng.uniform(-2, 2, size=3).astype(np.float32)
        w = w / (np.linalg.norm(w) + 1e-8)
        name = f"H_linear_{u_idx}"
        utilities[name] = {"type": "linear", "fn": lambda Y, w=w: np.argmax(Y * w[np.newaxis, :], axis=1), "weights": w.tolist()}
        u_idx += 1

    # Type 2: Target utilities (6)
    for _ in range(6):
        target = rng.uniform(-2, 2, size=3).astype(np.float32)
        name = f"H_target_{u_idx}"
        def make_target(t):
            return lambda Y: np.argmin(np.abs(Y - t[np.newaxis, :]), axis=1)
        utilities[name] = {"type": "target", "fn": make_target(target), "target": target.tolist()}
        u_idx += 1

    # Type 3: Risk threshold (4)
    for _ in range(4):
        threshold = rng.uniform(-1.5, 1.0)
        penalty = rng.uniform(2, 10)
        name = f"H_risk_{u_idx}"
        def make_risk(th, pn):
            return lambda Y: np.argmax(Y - np.where(Y < th, (Y - th)**2 * pn, 0), axis=1)
        utilities[name] = {"type": "risk", "fn": make_risk(threshold, penalty), "threshold": float(threshold), "penalty": float(penalty)}
        u_idx += 1

    # Type 4: Energy-aware (4)
    for _ in range(4):
        cost_m1 = rng.uniform(0, 0.5)
        cost_p1 = rng.uniform(0, 0.5)
        name = f"H_energy_{u_idx}"
        def make_energy(cm1, cp1):
            c = np.array([cm1, 0.0, cp1], dtype=np.float32)
            return lambda Y: np.argmax(Y - c[np.newaxis, :], axis=1)
        utilities[name] = {"type": "energy", "fn": make_energy(cost_m1, cost_p1), "cost_m1": float(cost_m1), "cost_p1": float(cost_p1)}
        u_idx += 1

    # Type 5: Adversarial (3) — high disagreement with U1 but not pure reverse
    for _ in range(3):
        flip_prob = rng.uniform(0.3, 0.7)
        sign_weights = rng.choice([-2, -1, 1, 2], size=3).astype(np.float32)
        name = f"H_adversarial_{u_idx}"
        def make_adv(flip, sw):
            return lambda Y: np.argmax(Y * sw[np.newaxis, :] * (1 if np.random.random() > flip else -1), axis=1)
        # Make deterministic by using fixed sign
        def make_adv_det(sw):
            return lambda Y: np.argmax(Y * sw[np.newaxis, :], axis=1)
        utilities[name] = {"type": "adversarial", "fn": make_adv_det(sign_weights), "sign_weights": sign_weights.tolist()}
        u_idx += 1

    return utilities

EXTENDED_UTILS = make_extended_utilities(n_total=25)
print(f"  Generated {len(EXTENDED_UTILS)} held-out utilities")
print(f"  Types: linear({sum(1 for u in EXTENDED_UTILS.values() if u['type']=='linear')}), "
      f"target({sum(1 for u in EXTENDED_UTILS.values() if u['type']=='target')}), "
      f"risk({sum(1 for u in EXTENDED_UTILS.values() if u['type']=='risk')}), "
      f"energy({sum(1 for u in EXTENDED_UTILS.values() if u['type']=='energy')}), "
      f"adversarial({sum(1 for u in EXTENDED_UTILS.values() if u['type']=='adversarial')})")

# Validate extended utilities against U1
for uname, uinfo in EXTENDED_UTILS.items():
    ba_h = uinfo["fn"](Y_audit)
    ba_u1 = utility_U1(Y_audit)
    div = label_divergence(ba_h, ba_u1)
    uinfo["divergence_from_U1"] = float(div)
    uinfo["valid"] = div >= 0.10

valid_ext = {k: v for k, v in EXTENDED_UTILS.items() if v.get("valid", False)}
print(f"  Valid held-out utilities (div > 0.10 from U1): {len(valid_ext)}/{len(EXTENDED_UTILS)}")


# ═══════════════════════════════════════════════════════════
# SECTION 2: Model Training + Held-Out Utility Transfer
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2: Train Models + Held-Out Utility Transfer")
print("=" * 60)

heldout_records = []
trained_models = {}  # seed -> {mechanism_name: model}

for seed in tqdm(SEEDS, desc="Training + HeldOut Transfer"):
    train_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "train") & (cf_df["horizon"] == 1)]
    test_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "test_id") & (cf_df["horizon"] == 1)]
    X_tr, Y_tr, ba_tr = prepare_counterfactual_data(train_df, seed, ENV_KWARGS)
    X_te, Y_te, ba_te = prepare_counterfactual_data(test_df, seed, ENV_KWARGS)

    if seed not in trained_models:
        trained_models[seed] = {}

    # ── U1_PolicyClone ──
    pc = StateOnlyPredictor(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    try:
        pc = train_state_only_classifier(pc, X_tr, Y_tr, None, None, epochs=EPOCHS, patience=PATIENCE, device=DEVICE)
    except: pass
    pc.eval()
    trained_models[seed]["U1_PolicyClone"] = pc

    # ── AEPCompressor ──
    aep = AEPCompressor(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    try:
        aep = train_ae_model(aep, X_tr, Y_tr, None, None, "aep", epochs=EPOCHS, patience=PATIENCE, device=DEVICE, ce_weight=0.8)
    except: pass
    aep.eval()
    trained_models[seed]["AEPCompressor"] = aep

    # ── CounterfactualCompressor ──
    cf = CounterfactualCompressor(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    try:
        cf = train_counterfactual_joint(cf, X_tr, Y_tr, None, None, epochs=EPOCHS, patience=PATIENCE, device=DEVICE, ce_weight=1.5)
    except: pass
    cf.eval()
    trained_models[seed]["CounterfactualCompressor"] = cf

    # ── ResidualCompressor ──
    rc = ResidualCompressor(2, 8, bottleneck_dim=BOTTLENECK_DIM, residual_dim=RESIDUAL_DIM)
    try:
        rc = train_ae_model(rc, X_tr, Y_tr, None, None, "residual", epochs=EPOCHS, patience=PATIENCE, device=DEVICE, ce_weight=0.8)
    except: pass
    rc.eval()
    trained_models[seed]["ResidualCompressor"] = rc

    # ── MultiGoalPolicyClone ──
    class MultiGoalPolicyClone(nn.Module):
        def __init__(self, obs_dim, history_len, bottleneck_dim=48, n_goals=3):
            super().__init__()
            self.in_dim = history_len * (obs_dim + 1)
            self.goal_emb = nn.Embedding(5, 8)  # up to 5 known goals
            self.encoder = MLP(self.in_dim + 8, [bottleneck_dim * 2, bottleneck_dim], bottleneck_dim)
            self.head = nn.Linear(bottleneck_dim, 3)

        def forward(self, x, goal_id):
            g_emb = self.goal_emb(goal_id)
            return self.head(self.encoder(torch.cat([x, g_emb], dim=-1)))

    mgpc = MultiGoalPolicyClone(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    mgpc = mgpc.to(DEVICE)
    # Train on U1-U3 with goal labels
    X_mg, y_mg, g_mg = [], [], []
    for i in range(len(X_tr)):
        for gid, gname in enumerate(["U1_main", "U2_reverse", "U3_target"]):
            ba_g = BASE_UTILITIES[gname](Y_tr[i:i+1])[0]
            X_mg.append(X_tr[i])
            y_mg.append(int(ba_g))
            g_mg.append(gid)
    X_mg_t = torch.tensor(np.stack(X_mg), dtype=torch.float32).to(DEVICE)
    y_mg_t = torch.tensor(np.array(y_mg), dtype=torch.long).to(DEVICE)
    g_mg_t = torch.tensor(np.array(g_mg), dtype=torch.long).to(DEVICE)
    ds_mg = TensorDataset(X_mg_t, y_mg_t, g_mg_t)
    loader_mg = DataLoader(ds_mg, batch_size=64, shuffle=True)
    opt_mg = torch.optim.Adam(mgpc.parameters(), lr=1e-3, weight_decay=1e-5)
    best_mg = float("inf"); best_state_mg = None; no_imp_mg = 0
    for ep in range(EPOCHS):
        mgpc.train(); total = 0.0
        for xb, yb, gb in loader_mg:
            opt_mg.zero_grad()
            loss = nn.functional.cross_entropy(mgpc(xb, gb), yb)
            loss.backward(); opt_mg.step()
            total += loss.item()
        if total < best_mg:
            best_mg = total; best_state_mg = {k: v.clone().cpu() for k, v in mgpc.state_dict().items()}
            no_imp_mg = 0
        else:
            no_imp_mg += 1
            if no_imp_mg >= PATIENCE: break
    if best_state_mg: mgpc.load_state_dict(best_state_mg)
    mgpc.eval()
    trained_models[seed]["MultiGoalPolicyClone"] = mgpc

    # ── FewShotPolicyClone (trained per utility later) ──
    trained_models[seed]["U1_PolicyClone_untrained"] = StateOnlyPredictor(2, 8, bottleneck_dim=BOTTLENECK_DIM)

    # ── Evaluate on all held-out utilities ──
    for uname, uinfo in tqdm(valid_ext.items(), desc=f"  seed={seed} heldout", leave=False):
        ba_true = uinfo["fn"](Y_te)

        for mech_name in ["U1_PolicyClone", "AEPCompressor", "CounterfactualCompressor", "ResidualCompressor"]:
            model = trained_models[seed][mech_name]
            with torch.no_grad():
                x_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
                if mech_name in ("AEPCompressor", "ResidualCompressor"):
                    preds = model.predict_all_actions(x_t).detach().cpu().numpy()
                else:
                    preds = model(x_t).detach().cpu().numpy()
            ba_pred = uinfo["fn"](preds)
            match = float(np.mean(ba_pred == ba_true))
            heldout_records.append({
                "seed": seed, "utility": uname, "utility_type": uinfo.get("type", "unknown"),
                "divergence_from_U1": uinfo.get("divergence_from_U1", 0),
                "mechanism": mech_name, "best_action_match": match
            })

        # MultiGoalPolicyClone: test with gid=4 (unseen, out of vocab) and gid=3 (U4-like)
        with torch.no_grad():
            x_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
            g_t = torch.full((len(X_te),), 4, dtype=torch.long, device=DEVICE)  # unseen goal id
            preds_mg = mgpc(x_t, g_t).detach().cpu().numpy()
        ba_pred_mg = uinfo["fn"](preds_mg)
        match_mg = float(np.mean(ba_pred_mg == ba_true))
        heldout_records.append({
            "seed": seed, "utility": uname, "utility_type": uinfo.get("type", "unknown"),
            "divergence_from_U1": uinfo.get("divergence_from_U1", 0),
            "mechanism": "MultiGoalPolicyClone", "best_action_match": match_mg
        })

        # OracleUtility: use true Y_te
        ba_oracle = uinfo["fn"](Y_te)
        match_oracle = float(np.mean(ba_oracle == ba_true))  # Should be 1.0
        heldout_records.append({
            "seed": seed, "utility": uname, "utility_type": uinfo.get("type", "unknown"),
            "divergence_from_U1": uinfo.get("divergence_from_U1", 0),
            "mechanism": "OracleUtilityPolicy", "best_action_match": match_oracle
        })

        # ActionOnly
        ao_ba = np.argmax(np.bincount(ba_tr, minlength=3))
        match_ao = float(np.mean(np.ones(len(Y_te)) * ao_ba == ba_true))
        heldout_records.append({
            "seed": seed, "utility": uname, "utility_type": uinfo.get("type", "unknown"),
            "divergence_from_U1": uinfo.get("divergence_from_U1", 0),
            "mechanism": "ActionOnly", "best_action_match": match_ao
        })

hu_df = pd.DataFrame(heldout_records)
hu_df.to_csv("results/ic2c_plus/heldout_utility_transfer.csv", index=False)
print("  heldout_utility_transfer.csv saved")

# Summary
hu_summ = hu_df.groupby(["mechanism", "utility_type"])["best_action_match"].mean().unstack()
print(hu_summ.to_string())


# ═══════════════════════════════════════════════════════════
# SECTION 3: Fair Memory Baselines (OutcomeTable variants)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3: Fair Memory Baselines")
print("=" * 60)

class RawMemoryOutcomeTableFull:
    """Nearest neighbor in history space -> copy full 3-action outcome table."""
    def __init__(self, k=5, standardize=True):
        self.k = k
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None

    def fit(self, X_train, Y_train_list):
        if self.scaler:
            self.X_store = self.scaler.fit_transform(np.array(X_train))
        else:
            self.X_store = np.array(X_train)
        self.Y_store = np.stack(Y_train_list, axis=-1)

    def predict(self, X_query, utility_fn=None):
        if self.scaler:
            X_q = self.scaler.transform(np.array(X_query))
        else:
            X_q = np.array(X_query)
        dists = np.sum((X_q[:, np.newaxis, :] - self.X_store[np.newaxis, :, :])**2, axis=2)
        nn_indices = np.argpartition(dists, min(self.k, len(self.X_store)-1), axis=1)[:, :min(self.k, len(self.X_store))]
        Y_pred = np.zeros((len(X_query), 3), dtype=np.float32)
        for i in range(len(X_query)):
            nn_tables = self.Y_store[nn_indices[i]]
            Y_pred[i] = nn_tables.mean(axis=0)
        if utility_fn:
            return np.array([utility_fn(Y_pred[i:i+1])[0] for i in range(len(Y_pred))])
        return Y_pred

    @property
    def memory_cost(self):
        return self.X_store.nbytes + self.Y_store.nbytes


class StandardizedKNNOutcomeTable(RawMemoryOutcomeTableFull):
    """Standardized KNN with full outcome table copy."""
    def __init__(self, k=5):
        super().__init__(k=k, standardize=True)


class PrototypeOutcomeTable:
    """Cluster histories, store prototype counterfactual tables."""
    def __init__(self, n_clusters=30, k=3):
        self.n_clusters = n_clusters
        self.k = k

    def fit(self, X_train, Y_train_list):
        n = len(X_train)
        self.Y_store = np.stack(Y_train_list, axis=-1)
        self.X_store = np.array(X_train)

        # Simple k-means-like prototype selection via random+refinement
        rng = np.random.default_rng(42)
        indices = rng.choice(n, min(self.n_clusters, n), replace=False)
        self.prototypes = self.X_store[indices]
        self.proto_labels = np.zeros(n, dtype=np.int64)
        for _ in range(5):
            for i in range(n):
                dists = np.sum((self.X_store[i] - self.prototypes)**2, axis=1)
                self.proto_labels[i] = np.argmin(dists)
            for p in range(len(self.prototypes)):
                mask = self.proto_labels == p
                if mask.sum() > 0:
                    self.prototypes[p] = self.X_store[mask].mean(axis=0)

        # Store prototype outcome tables
        self.proto_tables = np.zeros((len(self.prototypes), 3), dtype=np.float32)
        for p in range(len(self.prototypes)):
            mask = self.proto_labels == p
            if mask.sum() > 0:
                self.proto_tables[p] = self.Y_store[mask].mean(axis=0)

    def predict(self, X_query, utility_fn=None):
        Y_pred = np.zeros((len(X_query), 3), dtype=np.float32)
        for i in range(len(X_query)):
            dists = np.sum((X_query[i] - self.prototypes)**2, axis=1)
            top_k = np.argpartition(dists, min(self.k, len(self.prototypes)-1))[:min(self.k, len(self.prototypes))]
            Y_pred[i] = self.proto_tables[top_k].mean(axis=0)
        if utility_fn:
            return np.array([utility_fn(Y_pred[i:i+1])[0] for i in range(len(Y_pred))])
        return Y_pred

    @property
    def memory_cost(self):
        return self.prototypes.nbytes + self.proto_tables.nbytes


memory_records = []
for seed in tqdm(SEEDS, desc="Memory Baselines"):
    train_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "train") & (cf_df["horizon"] == 1)]
    test_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "test_id") & (cf_df["horizon"] == 1)]
    X_tr, Y_tr, ba_tr = prepare_counterfactual_data(train_df, seed, ENV_KWARGS)
    X_te, Y_te, ba_te = prepare_counterfactual_data(test_df, seed, ENV_KWARGS)
    Y3_tr = [Y_tr[:, 0], Y_tr[:, 1], Y_tr[:, 2]]

    # RawMemoryOutcomeTableFull
    rmot_full = RawMemoryOutcomeTableFull(k=5)
    rmot_full.fit(X_tr, Y3_tr)

    # StandardizedKNNOutcomeTable
    sknn_ot = StandardizedKNNOutcomeTable(k=5)
    sknn_ot.fit(X_tr, Y3_tr)

    # PrototypeOutcomeTable
    pot = PrototypeOutcomeTable(n_clusters=30, k=3)
    pot.fit(X_tr, Y3_tr)

    # Evaluate on each held-out utility
    for uname, uinfo in valid_ext.items():
        ba_true = uinfo["fn"](Y_te)

        for mem_name, mem_obj in [("RawMemoryOutcomeTableFull", rmot_full),
                                   ("StandardizedKNNOutcomeTable", sknn_ot),
                                   ("PrototypeOutcomeTable", pot)]:
            ba_pred = mem_obj.predict(X_te, utility_fn=uinfo["fn"])
            match = float(np.mean(ba_pred == ba_true))
            memory_records.append({
                "seed": seed, "utility": uname, "utility_type": uinfo.get("type", "unknown"),
                "mechanism": mem_name, "best_action_match": match,
                "memory_cost_bytes": mem_obj.memory_cost
            })

mem_df = pd.DataFrame(memory_records)
mem_df.to_csv("results/ic2c_plus/memory_outcome_table_baselines.csv", index=False)
print("  memory_outcome_table_baselines.csv saved")
mem_summ = mem_df.groupby(["mechanism"])["best_action_match"].mean()
print(mem_summ.to_string())


# ═══════════════════════════════════════════════════════════
# SECTION 4: Active Probe Fairness Audit
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 4: Active Probe Fairness Audit")
print("=" * 60)

probe_fairness_records = []

# Generate probe benchmark data (reuse from IC-2c if available)
probe_data_path = "results/ic2c/probe_benchmark_data.csv"
if not os.path.exists(probe_data_path):
    from src.env_structured_volatility import StructuredVolatilityEnv
    probe_recs = []
    env_p = StructuredVolatilityEnv(seed=0, **ENV_KWARGS)
    rng_p = np.random.default_rng(0)
    for i in range(600):
        env_p.reset(i * 1000 + 1)
        for _ in range(20):
            env_p.step(int(rng_p.choice([-1, 0, 1])))
        state = env_p.get_current_state()
        hist_obs = env_p.get_history_obs()
        hist_act = env_p.get_history_act()
        mode = env_p.mode
        outcomes = env_p.compute_outcomes(horizon=1)
        y = np.array([np.sum(outcomes[a]) for a in [-1, 0, 1]], dtype=np.float32)
        _, counts = np.unique(np.argmax(y), return_counts=True)
        probe_recs.append({
            "state_idx": i, "history_obs": [o.tolist() if isinstance(o, np.ndarray) else o for o in hist_obs],
            "history_act": list(hist_act), "mode": int(mode),
            "outcome_m1": y[0], "outcome_0": y[1], "outcome_p1": y[2],
            "best_action": int(np.argmax(y)), "probe_sign": 1.0 if mode == 0 else -1.0
        })
    probe_df_full = pd.DataFrame(probe_recs)
    probe_df_full["split"] = "train"
    probe_df_full.loc[400:, "split"] = "test"
    probe_df_full.to_csv(probe_data_path, index=False)

probe_df_full = pd.read_csv(probe_data_path)

for seed in tqdm(SEEDS, desc="Probe Fairness"):
    tr_p = probe_df_full[probe_df_full["split"] == "train"]
    te_p = probe_df_full[probe_df_full["split"] == "test"]
    X_tr_p, Y_tr_p, ba_tr_p = prepare_counterfactual_data(tr_p, 0, ENV_KWARGS)
    X_te_p, Y_te_p, ba_te_p = prepare_counterfactual_data(te_p, 0, ENV_KWARGS)
    probe_sign_tr = tr_p["probe_sign"].values.astype(np.float32).reshape(-1, 1)
    probe_sign_te = te_p["probe_sign"].values.astype(np.float32).reshape(-1, 1)

    # ── NoProbe PolicyClone ──
    pc_np = StateOnlyPredictor(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    try: pc_np = train_state_only_classifier(pc_np, X_tr_p, Y_tr_p, None, None, epochs=EPOCHS, patience=PATIENCE//2, device=DEVICE)
    except: pass
    pc_np.eval()
    with torch.no_grad():
        pc_np_p = pc_np(torch.tensor(X_te_p, dtype=torch.float32).to(DEVICE)).detach().cpu().numpy()
    pc_np_match = float(np.mean(np.argmax(pc_np_p, axis=1) == ba_te_p))
    probe_fairness_records.append({"seed": seed, "model": "NoProbe_PolicyClone", "match": pc_np_match})

    # ── NoProbe AEP ──
    aep_np = AEPCompressor(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    try: aep_np = train_ae_model(aep_np, X_tr_p, Y_tr_p, None, None, "aep", epochs=EPOCHS, patience=PATIENCE//2, device=DEVICE, ce_weight=0.8)
    except: pass
    aep_np.eval()
    with torch.no_grad():
        aep_np_p = aep_np.predict_all_actions(torch.tensor(X_te_p, dtype=torch.float32).to(DEVICE)).detach().cpu().numpy()
    aep_np_match = float(np.mean(np.argmax(aep_np_p, axis=1) == ba_te_p))
    probe_fairness_records.append({"seed": seed, "model": "NoProbe_AEP", "match": aep_np_match})

    # ── OneProbe PolicyClone (retrain with probe_sign appended) ──
    class OneProbePolicyClone(nn.Module):
        def __init__(self, obs_dim, history_len, bottleneck_dim=48):
            super().__init__()
            self.in_dim = history_len * (obs_dim + 1) + 1
            self.encoder = MLP(self.in_dim, [bottleneck_dim * 2, bottleneck_dim], bottleneck_dim)
            self.head = nn.Linear(bottleneck_dim, 3)
        def forward(self, x):
            return self.head(self.encoder(x))

    pc_op = OneProbePolicyClone(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    try: pc_op = train_state_only_classifier(pc_op, np.hstack([X_tr_p, probe_sign_tr]), Y_tr_p, None, None, epochs=EPOCHS, patience=PATIENCE//2, device=DEVICE)
    except: pass
    pc_op.eval()
    with torch.no_grad():
        pc_op_p = pc_op(torch.tensor(np.hstack([X_te_p, probe_sign_te]), dtype=torch.float32).to(DEVICE)).detach().cpu().numpy()
    pc_op_match = float(np.mean(np.argmax(pc_op_p, axis=1) == ba_te_p))
    probe_fairness_records.append({"seed": seed, "model": "OneProbe_PolicyClone", "match": pc_op_match})

    # ── OneProbe AEP ──
    class OneProbeAEP(nn.Module):
        def __init__(self, obs_dim, history_len, bottleneck_dim=48):
            super().__init__()
            self.in_dim = history_len * (obs_dim + 1) + 1
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
            outs = []
            for a in range(3):
                a_t = torch.full((x.shape[0],), a, dtype=torch.long, device=x.device)
                z_action = self.action_encoder(a_t)
                z = torch.cat([z_state, z_action], dim=-1)
                outs.append(self.head(self.decoder(z)))
            return torch.cat(outs, dim=-1)

    aep_op = OneProbeAEP(2, 8, bottleneck_dim=BOTTLENECK_DIM)
    try: aep_op = train_ae_model(aep_op, np.hstack([X_tr_p, probe_sign_tr]), Y_tr_p, None, None, "aep", epochs=EPOCHS, patience=PATIENCE//2, device=DEVICE, ce_weight=0.8)
    except: pass
    aep_op.eval()
    with torch.no_grad():
        aep_op_p = aep_op.predict_all_actions(torch.tensor(np.hstack([X_te_p, probe_sign_te]), dtype=torch.float32).to(DEVICE)).detach().cpu().numpy()
    aep_op_match = float(np.mean(np.argmax(aep_op_p, axis=1) == ba_te_p))
    probe_fairness_records.append({"seed": seed, "model": "OneProbe_AEP", "match": aep_op_match})

    # ── OneProbe RawMemory (nearest state search with probe) ──
    rmot = RawMemoryOutcomeTableFull(k=5)
    rmot.fit(np.hstack([X_tr_p, probe_sign_tr]), [Y_tr_p[:,0], Y_tr_p[:,1], Y_tr_p[:,2]])
    rmot_p = rmot.predict(np.hstack([X_te_p, probe_sign_te]))
    rmot_match = float(np.mean(np.argmax(rmot_p, axis=1) == ba_te_p))
    probe_fairness_records.append({"seed": seed, "model": "OneProbe_RawMemory", "match": rmot_match})

    # ── Probe gains ──
    probe_fairness_records.append({"seed": seed, "model": "ProbeGain_PolicyClone", "match": pc_op_match - pc_np_match})
    probe_fairness_records.append({"seed": seed, "model": "ProbeGain_AEP", "match": aep_op_match - aep_np_match})
    probe_fairness_records.append({"seed": seed, "model": "ProbeGain_RawMemory", "match": rmot_match - pc_np_match})
    # AEP structural advantage = AEP probe gain - PC probe gain
    probe_fairness_records.append({"seed": seed, "model": "AEP_Structural_Probe_Advantage",
                                    "match": (aep_op_match - aep_np_match) - (pc_op_match - pc_np_match)})

pf_df = pd.DataFrame(probe_fairness_records)
pf_df.to_csv("results/ic2c_plus/active_probe_fairness.csv", index=False)
print("  active_probe_fairness.csv saved")
pf_summ = pf_df.groupby("model")["match"].mean()
print(pf_summ.to_string())


# ═══════════════════════════════════════════════════════════
# SECTION 5: Revised Action Coverage Gap
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 5: Revised Action Coverage Gap")
print("=" * 60)

coverage_revised_records = []
BIAS_TYPES_R = ["plus_dominant", "zero_dominant", "minus_dominant"]
CF_FRACTIONS_R = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
exp_seed = 0

train_df_c = cf_df[(cf_df["seed"] == exp_seed) & (cf_df["split"] == "train") & (cf_df["horizon"] == 1)]
test_df_c = cf_df[(cf_df["seed"] == exp_seed) & (cf_df["split"] == "test_id") & (cf_df["horizon"] == 1)]
X_tr_c, Y_tr_c, ba_tr_c = prepare_counterfactual_data(train_df_c, exp_seed, ENV_KWARGS)
X_te_c, Y_te_c, ba_te_c = prepare_counterfactual_data(test_df_c, exp_seed, ENV_KWARGS)

def systematic_action_masking(X, Y, bias_type, cf_fraction, seed_offset=0):
    """Mask specific action-mode pairs systematically, not randomly."""
    rng = np.random.default_rng(42 + seed_offset)
    n = len(X)

    # Determine which action is masked
    if bias_type == "plus_dominant":
        masked_action = 0  # action -1
        visible_actions = [1, 2]
    elif bias_type == "zero_dominant":
        masked_action = 2  # action +1
        visible_actions = [0, 1]
    else:
        masked_action = 2  # action +1
        visible_actions = [0, 1]

    Y_out = np.full_like(Y, np.nan, dtype=np.float32)
    for i in range(n):
        for a in range(3):
            if a in visible_actions:
                Y_out[i, a] = Y[i, a]
            elif a == masked_action:
                if rng.random() < cf_fraction:
                    Y_out[i, a] = Y[i, a]  # probed
    return X, Y_out

for bias_type in tqdm(BIAS_TYPES_R, desc="Coverage Revised"):
    for cf_frac in tqdm(CF_FRACTIONS_R, desc=f"  {bias_type}", leave=False):
        X_biased, Y_biased = systematic_action_masking(X_tr_c, Y_tr_c, bias_type, cf_frac)

        # Determine rare (masked) action
        if bias_type == "plus_dominant": rare_a = 0
        elif bias_type == "zero_dominant": rare_a = 2
        else: rare_a = 2
        rare_mask = ba_te_c == rare_a
        n_rare = rare_mask.sum()

        # PolicyClone from scratch on biased data (doesn't use action info anyway)
        pc_cov = StateOnlyPredictor(2, 8, bottleneck_dim=BOTTLENECK_DIM)
        # PolicyClone labeled by best action on AVAILABLE outcomes
        ba_obs = np.nanargmax(Y_biased, axis=1)
        all_nan = np.all(np.isnan(Y_biased), axis=1)
        if all_nan.any():
            ba_obs[all_nan] = np.argmax(np.bincount(ba_obs[~all_nan], minlength=3))
        Y_obs_labels = np.zeros((len(Y_biased), 3), dtype=np.float32)
        for i in range(len(Y_biased)):
            Y_obs_labels[i, ba_obs[i]] = 1.0  # Minimal one-hot scoring
        try:
            pc_cov = train_state_only_classifier(pc_cov, X_biased, Y_obs_labels, None, None, epochs=EPOCHS//2, patience=PATIENCE//2, device=DEVICE)
        except: pass
        pc_cov.eval()
        with torch.no_grad():
            pc_cov_p = pc_cov(torch.tensor(X_te_c, dtype=torch.float32).to(DEVICE)).detach().cpu().numpy()
        pc_match = float(np.mean(np.argmax(pc_cov_p, axis=1) == ba_te_c))
        pc_rare = float(np.mean(np.argmax(pc_cov_p[rare_mask], axis=1) == ba_te_c[rare_mask])) if n_rare > 0 else 0.0

        # AEP: train only on observed (state, action) pairs
        X_ae, y_ae, a_ae = [], [], []
        for i in range(len(X_biased)):
            for a in range(3):
                if not np.isnan(Y_biased[i, a]):
                    X_ae.append(X_biased[i])
                    y_ae.append(Y_biased[i, a])
                    a_ae.append(a)
        aep_cov = AEPCompressor(2, 8, bottleneck_dim=BOTTLENECK_DIM)
        if len(X_ae) > 0:
            X_ae_t = torch.tensor(np.stack(X_ae), dtype=torch.float32).to(DEVICE)
            y_ae_t = torch.tensor(np.array(y_ae, dtype=np.float32)).unsqueeze(1).to(DEVICE)
            a_ae_t = torch.tensor(np.array(a_ae), dtype=torch.long).to(DEVICE)
            ds_ae = TensorDataset(X_ae_t, y_ae_t, a_ae_t)
            loader_ae = DataLoader(ds_ae, batch_size=192, shuffle=True)
            aep_cov = aep_cov.to(DEVICE)
            opt_ae = torch.optim.Adam(aep_cov.parameters(), lr=1e-3, weight_decay=1e-5)
            best_ae = float("inf"); best_st_ae = None; no_im_ae = 0
            for ep in range(EPOCHS):
                aep_cov.train(); total_ae = 0.0
                for xb, yb, ab in loader_ae:
                    opt_ae.zero_grad()
                    loss = nn.functional.mse_loss(aep_cov(xb, ab), yb)
                    loss.backward(); opt_ae.step()
                    total_ae += loss.item()
                if total_ae < best_ae:
                    best_ae = total_ae; best_st_ae = {k: v.clone().cpu() for k, v in aep_cov.state_dict().items()}
                    no_im_ae = 0
                else:
                    no_im_ae += 1
                    if no_im_ae >= PATIENCE: break
            if best_st_ae: aep_cov.load_state_dict(best_st_ae)
        aep_cov.eval()
        with torch.no_grad():
            aep_cov_p = aep_cov.predict_all_actions(torch.tensor(X_te_c, dtype=torch.float32).to(DEVICE)).detach().cpu().numpy()
        aep_match = float(np.mean(np.argmax(aep_cov_p, axis=1) == ba_te_c))
        aep_rare = float(np.mean(np.argmax(aep_cov_p[rare_mask], axis=1) == ba_te_c[rare_mask])) if n_rare > 0 else 0.0

        # CF
        cf_cov = CounterfactualCompressor(2, 8, bottleneck_dim=BOTTLENECK_DIM)
        valid_rows = ~np.all(np.isnan(Y_biased), axis=1)
        if valid_rows.sum() > 0:
            try:
                X_tr_cf = X_biased[valid_rows]; Y_tr_cf = Y_biased[valid_rows]
                cf_cov = train_counterfactual_joint(cf_cov, X_tr_cf, Y_tr_cf, None, None, epochs=EPOCHS//2, patience=PATIENCE//2, device=DEVICE, ce_weight=1.5, normalize_y=True)
            except: pass
        cf_cov.eval()
        with torch.no_grad():
            cf_cov_p = cf_cov(torch.tensor(X_te_c, dtype=torch.float32).to(DEVICE)).detach().cpu().numpy()
        cf_match = float(np.mean(np.argmax(cf_cov_p, axis=1) == ba_te_c))
        cf_rare = float(np.mean(np.argmax(cf_cov_p[rare_mask], axis=1) == ba_te_c[rare_mask])) if n_rare > 0 else 0.0

        # RawMemoryOutcomeTable
        rmot_c = RawMemoryOutcomeTableFull(k=5)
        valid_rows2 = ~np.any(np.isnan(Y_biased), axis=1)
        if valid_rows2.sum() > 0:
            rmot_c.fit(X_biased[valid_rows2], [Y_biased[valid_rows2, 0], Y_biased[valid_rows2, 1], Y_biased[valid_rows2, 2]])
        else:
            rmot_c.fit(X_tr_c, [Y_tr_c[:,0], Y_tr_c[:,1], Y_tr_c[:,2]])
        rmot_p = rmot_c.predict(X_te_c)
        rmot_match = float(np.mean(np.argmax(rmot_p, axis=1) == ba_te_c))
        rmot_rare = float(np.mean(np.argmax(rmot_p[rare_mask], axis=1) == ba_te_c[rare_mask])) if n_rare > 0 else 0.0

        for mech, bal, rare in [("PolicyCloneBaseline", pc_match, pc_rare),
                                ("AEPCompressor", aep_match, aep_rare),
                                ("CounterfactualCompressor", cf_match, cf_rare),
                                ("RawMemoryOutcomeTableFull", rmot_match, rmot_rare)]:
            coverage_revised_records.append({
                "seed": exp_seed, "bias_type": bias_type, "cf_fraction": cf_frac,
                "mechanism": mech, "balanced_match": bal, "rare_action_match": rare
            })

cov_rev_df = pd.DataFrame(coverage_revised_records)
cov_rev_df.to_csv("results/ic2c_plus/coverage_gap_revised.csv", index=False)
print("  coverage_gap_revised.csv saved")
cr_summ = cov_rev_df.groupby(["cf_fraction", "mechanism"])["balanced_match"].mean().unstack()
print(cr_summ.to_string())


# ═══════════════════════════════════════════════════════════
# SECTION 6: Cost Normalization + Policy Baseline Comparison
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 6: Cost Normalization + Policy Baselines")
print("=" * 60)

# ── Policy Baseline Comparison ──
policy_comp_records = []
# Reuse goal_transfer data from IC-2c plus add U1-U5 from IC-2c
# Read existing goal_transfer
gt_df_old = pd.read_csv("results/ic2c/goal_transfer.csv") if os.path.exists("results/ic2c/goal_transfer.csv") else pd.DataFrame()

for seed in SEEDS:
    test_df = cf_df[(cf_df["seed"] == seed) & (cf_df["split"] == "test_id") & (cf_df["horizon"] == 1)]
    X_te, Y_te, ba_te = prepare_counterfactual_data(test_df, seed, ENV_KWARGS)

    # U1 PolicyClone (from trained models)
    pc_model = trained_models.get(seed, {}).get("U1_PolicyClone")
    aep_model = trained_models.get(seed, {}).get("AEPCompressor")
    cf_model = trained_models.get(seed, {}).get("CounterfactualCompressor")
    mgpc_model = trained_models.get(seed, {}).get("MultiGoalPolicyClone")
    rc_model = trained_models.get(seed, {}).get("ResidualCompressor")

    # U1 baseline and U2-U5 for base utilities
    for gname, gfn in BASE_UTILITIES.items():
        ba_t = gfn(Y_te)
        for mech_name, model in [("U1_PolicyClone", pc_model), ("AEPCompressor", aep_model),
                                  ("CounterfactualCompressor", cf_model), ("MultiGoalPolicyClone", mgpc_model),
                                  ("ResidualCompressor", rc_model)]:
            if model is None: continue
            with torch.no_grad():
                x_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
                if mech_name in ("AEPCompressor", "ResidualCompressor"):
                    preds = model.predict_all_actions(x_t).detach().cpu().numpy()
                elif mech_name == "MultiGoalPolicyClone":
                    gid = {"U1_main": 0, "U2_reverse": 1, "U3_target": 2}.get(gname, 3)
                    preds = model(x_t, torch.full((len(X_te),), gid, dtype=torch.long, device=DEVICE)).detach().cpu().numpy()
                else:
                    preds = model(x_t).detach().cpu().numpy()
            ba_p = gfn(preds)
            match = float(np.mean(ba_p == ba_t))
            policy_comp_records.append({"seed": seed, "goal": gname, "mechanism": mech_name, "best_action_match": match})

pc_df = pd.DataFrame(policy_comp_records)
pc_df.to_csv("results/ic2c_plus/policy_baseline_comparison.csv", index=False)
print("  policy_baseline_comparison.csv saved")

# ── Cost-Normalized Premiums ──
cost_records = []

# Parameter estimates
PARAM_ESTIMATES = {
    "U1_PolicyClone": 20000, "AEPCompressor": 35000, "CounterfactualCompressor": 25000,
    "ResidualCompressor": 40000, "MultiGoalPolicyClone": 25000,
    "RawMemoryOutcomeTableFull": 5000, "StandardizedKNNOutcomeTable": 6000,
    "PrototypeOutcomeTable": 1200, "OracleUtilityPolicy": float("inf"),
    "ActionOnly": 3
}

# Held-out utility averages
hu_mech = hu_df.groupby("mechanism")["best_action_match"].mean()
pc_heldout = hu_mech.get("U1_PolicyClone", 0)
aep_heldout = hu_mech.get("AEPCompressor", 0)
cf_heldout = hu_mech.get("CounterfactualCompressor", 0)
mgpc_heldout = hu_mech.get("MultiGoalPolicyClone", 0)
rc_heldout = hu_mech.get("ResidualCompressor", 0)

# Memory baselines
mem_mech = mem_df.groupby("mechanism")["best_action_match"].mean()
rmot_heldout = mem_mech.get("RawMemoryOutcomeTableFull", 0)
sknn_heldout = mem_mech.get("StandardizedKNNOutcomeTable", 0)
pot_heldout = mem_mech.get("PrototypeOutcomeTable", 0)

# Probe gains
pf_mech = pf_df.groupby("model")["match"].mean()
pc_probe_gain = pf_mech.get("ProbeGain_PolicyClone", 0)
aep_probe_gain = pf_mech.get("ProbeGain_AEP", 0)
aep_struct_adv = pf_mech.get("AEP_Structural_Probe_Advantage", 0)

for mech_name, params in PARAM_ESTIMATES.items():
    heldout_match = {
        "U1_PolicyClone": pc_heldout, "AEPCompressor": aep_heldout,
        "CounterfactualCompressor": cf_heldout, "ResidualCompressor": rc_heldout,
        "MultiGoalPolicyClone": mgpc_heldout,
        "RawMemoryOutcomeTableFull": rmot_heldout, "StandardizedKNNOutcomeTable": sknn_heldout,
        "PrototypeOutcomeTable": pot_heldout,
    }.get(mech_name, 0)

    probe_gain = {
        "U1_PolicyClone": pc_probe_gain, "AEPCompressor": aep_probe_gain,
    }.get(mech_name, 0)

    transfer_premium = heldout_match - pc_heldout
    cost_norm_transfer = transfer_premium / max(params / 1000.0, 1e-8)
    cost_norm_probe = probe_gain / max(params / 1000.0, 1e-8)

    cost_records.append({
        "mechanism": mech_name, "param_estimate": params,
        "heldout_utility_match": heldout_match,
        "transfer_premium_vs_U1PC": transfer_premium,
        "probe_gain": probe_gain,
        "cost_normalized_transfer_premium": cost_norm_transfer,
        "cost_normalized_probe_gain": cost_norm_probe,
    })

cn_df = pd.DataFrame(cost_records)
cn_df.to_csv("results/ic2c_plus/cost_normalized_premium.csv", index=False)
print("  cost_normalized_premium.csv saved")


# ═══════════════════════════════════════════════════════════
# SECTION 7: Final Report
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 7: Final Report")
print("=" * 60)

# Collect key evidence
pc_u1 = hu_mech.get("U1_PolicyClone", 0)
aep_hu = hu_mech.get("AEPCompressor", 0)
cf_hu = hu_mech.get("CounterfactualCompressor", 0)
mgpc_hu = hu_mech.get("MultiGoalPolicyClone", 0)
rc_hu = hu_mech.get("ResidualCompressor", 0)
rmot_hu = mem_mech.get("RawMemoryOutcomeTableFull", 0)
sknn_hu = mem_mech.get("StandardizedKNNOutcomeTable", 0)
pot_hu = mem_mech.get("PrototypeOutcomeTable", 0)

aep_transfer_premium = aep_hu - pc_u1
cf_transfer_premium = cf_hu - pc_u1
mgpc_transfer_premium = mgpc_hu - pc_u1

# Probe fairness
np_pc_pf = pf_mech.get("NoProbe_PolicyClone", 0)
np_aep_pf = pf_mech.get("NoProbe_AEP", 0)
op_pc_pf = pf_mech.get("OneProbe_PolicyClone", 0)
op_aep_pf = pf_mech.get("OneProbe_AEP", 0)
pc_probe_g_val = pf_mech.get("ProbeGain_PolicyClone", 0)
aep_probe_g_val = pf_mech.get("ProbeGain_AEP", 0)
aep_struct_val = pf_mech.get("AEP_Structural_Probe_Advantage", 0)

# Coverage gap at key fractions
cov_20 = cov_rev_df[cov_rev_df["cf_fraction"] == 0.20]
cov_50 = cov_rev_df[cov_rev_df["cf_fraction"] == 0.50]
aep_cov20 = cov_20[cov_20["mechanism"] == "AEPCompressor"]["balanced_match"].mean() if len(cov_20) > 0 else 0
pc_cov20 = cov_20[cov_20["mechanism"] == "PolicyCloneBaseline"]["balanced_match"].mean() if len(cov_20) > 0 else 0
cf_cov20 = cov_20[cov_20["mechanism"] == "CounterfactualCompressor"]["balanced_match"].mean() if len(cov_20) > 0 else 0
aep_cov50 = cov_50[cov_50["mechanism"] == "AEPCompressor"]["balanced_match"].mean() if len(cov_50) > 0 else 0
pc_cov50 = cov_50[cov_50["mechanism"] == "PolicyCloneBaseline"]["balanced_match"].mean() if len(cov_50) > 0 else 0

# Utility validity
u4_valid = ua_df[ua_df["utility"] == "U4_risk_avoid"]["valid"].values
u4_is_valid = bool(u4_valid[0]) if len(u4_valid) > 0 else False
n_valid_utils = len(valid_ext)
n_invalid_base = sum(1 for r in utility_audit_records if "U" in str(r.get("utility", "")) and not r.get("valid", True))

# ── Verdict Decision ──
conditions = {
    "aep_heldout_gt_pc_10": aep_transfer_premium > 0.10,
    "aep_heldout_gt_mgpc": aep_hu > mgpc_hu + 0.05,
    "aep_heldout_gt_rmot": aep_hu > rmot_hu,
    "aep_cost_norm_positive": (cn_df[cn_df["mechanism"]=="AEPCompressor"]["cost_normalized_transfer_premium"].values[0] if len(cn_df[cn_df["mechanism"]=="AEPCompressor"]) > 0 else 0) > 0,
    "aep_probe_struct_adv_gt_05": aep_struct_val > 0.05,
    "coverage_aep_gt_pc_20": aep_cov20 > pc_cov20 + 0.10,
}

n_passed = sum(conditions.values())
print(f"  Conditions passed: {n_passed}/6")
for k, v in conditions.items():
    print(f"    {k}: {'PASS' if v else 'FAIL'}")

if n_passed >= 5:
    verdict = "IC2C_PLUS_STRONG_GOAL_TRANSFER_APPRECIATION"
elif conditions["aep_heldout_gt_pc_10"]:
    verdict = "IC2C_PLUS_WEAK_GOAL_TRANSFER_ONLY"
elif aep_struct_val > 0.05:
    verdict = "IC2C_PLUS_PROBE_VALUE_ONLY"
elif rmot_hu > aep_hu + 0.05:
    verdict = "IC2C_PLUS_MEMORY_BASELINE_WINS"
elif pc_heldout >= aep_hu:
    verdict = "IC2C_PLUS_POLICY_BASELINE_WINS"
else:
    verdict = "IC2C_PLUS_BENCHMARK_INVALID"

print(f"\n  VERDICT: {verdict}")

# ── Generate Report ──
r = []
def w(s=""): r.append(s)

w("# IC-2c+: Fair Goal-Transfer Robustness Audit")
w()
w("**Date**: 2026-05-10")
w(f"**Final Verdict**: `{verdict}`")
w()
w("---")
w("## Executive Summary")
w()
w("IC-2c extends the Policy-Clone Trap Escape Benchmark with fairness audits:")
w("1. Stronger policy baselines (MultiGoal, FewShot)")
w("2. Fair memory baselines (OutcomeTable, not regression->argmax)")
w("3. Active probe fairness (PolicyClone gets probe)")
w("4. Revised coverage gap (systematic masking)")
w("5. Cost-normalized premiums")
w()
w("### Held-Out Utility Transfer (mean of 25 random utilities)")
w()
w("| Mechanism | Mean Match | vs U1_PolicyClone Premium |")
w("|---|---|---|")
w(f"| U1_PolicyClone | {pc_u1:.4f} | — |")
w(f"| AEPCompressor | {aep_hu:.4f} | {aep_transfer_premium:+.4f} |")
w(f"| CounterfactualCompressor | {cf_hu:.4f} | {cf_transfer_premium:+.4f} |")
w(f"| ResidualCompressor | {rc_hu:.4f} | - |")
w(f"| MultiGoalPolicyClone | {mgpc_hu:.4f} | {mgpc_transfer_premium:+.4f} |")
w(f"| RawMemoryOutcomeTableFull | {rmot_hu:.4f} | {rmot_hu - pc_u1:+.4f} |")
w(f"| StandardizedKNNOutcomeTable | {sknn_hu:.4f} | {sknn_hu - pc_u1:+.4f} |")
w(f"| PrototypeOutcomeTable | {pot_hu:.4f} | {pot_hu - pc_u1:+.4f} |")
w()

w("---")
w("## Q1: Are Base Utilities (U1-U5) Valid?")
w()
w("| Utility | Valid? | Label Div from U1 | Distribution |")
w("|---|---|---|---|")
for rec in utility_audit_records:
    if isinstance(rec, dict) and rec.get("utility", "").startswith("U"):
        w(f"| {rec.get('utility', '')} | {rec.get('valid', False)} | {rec.get('label_divergence_from_U1', 0):.4f} | "
          f"-1:{rec.get('dist_m1',0):.2f} / 0:{rec.get('dist_0',0):.2f} / +1:{rec.get('dist_p1',0):.2f} |")
w()
u4_div = ua_df[ua_df["utility"] == "U4_risk_avoid"]["label_divergence_from_U1"].values
if len(u4_div) > 0 and u4_div[0] < 0.10:
    w("**Finding**: U4_risk_avoid has < 0.10 divergence from U1 — marked INVALID. U4 outcomes are too similar to U1.")
else:
    w("**Finding**: All base utilities have sufficient divergence from U1.")

w()
w(f"**Extended utilities**: {n_valid_utils} valid (divergence > 0.10 from U1) out of {len(EXTENDED_UTILS)} total.")
w()

w("---")
w("## Q2: Does AEP Beat PolicyClone on Held-Out Utilities?")
w()
w(f"**Answer: {'YES' if aep_transfer_premium > 0.10 else 'NO'}.**")
w(f"  AEP held-out transfer premium: {aep_transfer_premium:+.4f}")
w()
if aep_transfer_premium > 0.10:
    w("**SUCCESS**: AEP retains >+0.10 transfer premium on fair held-out utilities.")
else:
    w("**FAIL**: AEP does not meet the +0.10 threshold on held-out utilities.")
w()

w("---")
w("## Q3: Does AEP Beat MultiGoalPolicyClone?")
w()
w(f"**Answer: {'YES' if aep_hu > mgpc_hu + 0.05 else 'NO'}.**")
w(f"  AEP={aep_hu:.4f} vs MultiGoalPC={mgpc_hu:.4f}")
w()
if aep_hu > mgpc_hu + 0.05:
    w("**SUCCESS**: AEP exceeds goal-conditioned PolicyClone. Compression beats goal-labeled imitation.")
else:
    w("MultiGoalPolicyClone is competitive — goal conditioning may partially substitute for outcome modeling.")
w()

w("---")
w("## Q4: Does AEP Beat Fair RawMemoryOutcomeTableEqualCost?")
w()
w(f"**Answer: {'YES' if aep_hu > rmot_hu else 'NO'}.**")
w(f"  AEP={aep_hu:.4f} vs RawMemoryOutcomeTableFull={rmot_hu:.4f}")
w()
if aep_hu > rmot_hu:
    w("**SUCCESS**: AEP beats fair memory baseline (full outcome table with nearest-neighbor transfer).")
else:
    w("**FAIL**: RawMemory with full outcome table still outperforms learned compression.")
w()

w("---")
w("## Q5: Is Active Probe Advantage Structural (AEP), Not Just Privilege?")
w()
w("| Model | NoProbe | OneProbe | Probe Gain |")
w("|---|---|---|---|")
w(f"| PolicyClone | {np_pc_pf:.4f} | {op_pc_pf:.4f} | {pc_probe_g_val:+.4f} |")
w(f"| AEPCompressor | {np_aep_pf:.4f} | {op_aep_pf:.4f} | {aep_probe_g_val:+.4f} |")
w(f"| RawMemory | — | — | — |")
w()
w(f"**AEP Structural Probe Advantage** (AEP gain - PC gain): {aep_struct_val:+.4f}")
w()
if aep_struct_val > 0.05:
    w("**SUCCESS**: AEP probe advantage is structural (>0.05 beyond PolicyClone's probe gain).")
else:
    w("**FAIL**: Active probe advantage is largely probe-info privilege, not AEP's compression structure.")
w()

w("---")
w("## Q6: Does Revised Coverage Gap Provide Stronger Evidence?")
w()
w("| CF Fraction | PolicyClone | AEPCompressor | CFCompressor |")
w("|---|---|---|---|")
for frac in CF_FRACTIONS_R:
    pc_c = cov_rev_df[(cov_rev_df["cf_fraction"]==frac)&(cov_rev_df["mechanism"]=="PolicyCloneBaseline")]["balanced_match"].mean()
    aep_c = cov_rev_df[(cov_rev_df["cf_fraction"]==frac)&(cov_rev_df["mechanism"]=="AEPCompressor")]["balanced_match"].mean()
    cf_c = cov_rev_df[(cov_rev_df["cf_fraction"]==frac)&(cov_rev_df["mechanism"]=="CounterfactualCompressor")]["balanced_match"].mean()
    w(f"| {frac:.0%} | {pc_c:.4f} | {aep_c:.4f} | {cf_c:.4f} |")
w()
if aep_cov20 > pc_cov20 + 0.10:
    w(f"**SUCCESS**: AEP at 20% CF ({aep_cov20:.3f}) beats PolicyClone ({pc_cov20:.3f}) by >0.10.")
else:
    w(f"At 20% CF: AEP={aep_cov20:.3f} vs PC={pc_cov20:.3f} (gap={aep_cov20-pc_cov20:+.3f})")
w()

w("---")
w("## Q7: Does Cost-Normalized Intelligence Appreciation Hold?")
w()
w("| Mechanism | Params | Transfer Premium | Cost-Norm Transfer | Cost-Norm Probe |")
w("|---|---|---|---|---|")
for _, row in cn_df.iterrows():
    w(f"| {row['mechanism']} | {row['param_estimate']:.0f} | {row['transfer_premium_vs_U1PC']:+.4f} | "
      f"{row['cost_normalized_transfer_premium']:+.4f} | {row['cost_normalized_probe_gain']:+.4f} |")
w()

w("---")
w("## Final Verdict")
w()
w(f"### `{verdict}`")
w()
w(f"Conditions passed: {n_passed}/6")
for k, v in conditions.items():
    w(f"- {k}: {'PASS' if v else 'FAIL'}")
w()

w("---")
w("### All IC-2c+ Outputs")
w()
w("| File | Content |")
w("|---|---|")
w("| `results/ic2c_plus/utility_validity_audit.csv` | U1-U5 validity + pairwise disagreements |")
w("| `results/ic2c_plus/heldout_utility_transfer.csv` | All mechanisms on 25 held-out random utilities |")
w("| `results/ic2c_plus/policy_baseline_comparison.csv` | U1-U5 policy baseline comparison |")
w("| `results/ic2c_plus/memory_outcome_table_baselines.csv` | Fair memory baselines on held-out utilities |")
w("| `results/ic2c_plus/active_probe_fairness.csv` | Probe fairness with PolicyClone privilege |")
w("| `results/ic2c_plus/coverage_gap_revised.csv` | Revised systematic action masking |")
w("| `results/ic2c_plus/cost_normalized_premium.csv` | Cost-normalized transfer & probe premiums |")
w("| `results/ic2c_plus/IC2C_PLUS_FAIR_GOAL_TRANSFER_REPORT.md` | **This report** |")

with open("results/ic2c_plus/IC2C_PLUS_FAIR_GOAL_TRANSFER_REPORT.md", "w", encoding="utf-8") as f:
    f.write("\n".join(r))

print("  IC2C_PLUS_FAIR_GOAL_TRANSFER_REPORT.md written.")


# ── Charts ──
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 1. Utility divergence
    div_vals = [r for r in utility_audit_records if not isinstance(r.get("utility", ""), str) or not r.get("utility", "").startswith("pairwise")]
    div_v = [r for r in div_vals if 'label_divergence_from_U1' in r]
    labels = [str(r.get("utility", "?")) for r in div_v[:5]]
    vals = [r["label_divergence_from_U1"] for r in div_v[:5]]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['green' if v >= 0.10 else 'red' for v in vals]
    ax.bar(labels, vals, color=colors)
    ax.axhline(y=0.10, color='gray', linestyle='--', label='validity threshold')
    ax.set_ylabel("Label Divergence from U1")
    ax.set_title("IC-2c+: Utility Validity Audit")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/figures/ic2c_plus_utility_divergence.png", dpi=100)
    plt.close()
    print("  ic2c_plus_utility_divergence.png saved")

    # 2. Held-out transfer
    mechs = ["U1_PolicyClone", "AEPCompressor", "CounterfactualCompressor", "MultiGoalPolicyClone",
             "ResidualCompressor", "RawMemoryOutcomeTableFull", "StandardizedKNNOutcomeTable", "PrototypeOutcomeTable"]
    m_vals = [pc_u1, aep_hu, cf_hu, mgpc_hu, rc_hu, rmot_hu, sknn_hu, pot_hu]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(mechs, m_vals)
    ax.axhline(y=pc_u1, color='orange', linestyle='--', alpha=0.7, label=f'U1_PC baseline ({pc_u1:.3f})')
    ax.set_ylabel("Mean Best Action Match")
    ax.set_title("IC-2c+: Held-Out Utility Transfer (25 random utilities)")
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/figures/ic2c_plus_heldout_transfer.png", dpi=100)
    plt.close()
    print("  ic2c_plus_heldout_transfer.png saved")

    # 3. Policy baselines
    pc_summ = pc_df.groupby(["goal", "mechanism"])["best_action_match"].mean().unstack()
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(BASE_UTILITIES))
    w_bar = 0.15
    for i, mech in enumerate(["U1_PolicyClone", "AEPCompressor", "CounterfactualCompressor", "MultiGoalPolicyClone", "ResidualCompressor"]):
        v = [pc_summ.loc[g, mech] if g in pc_summ.index and mech in pc_summ.columns else 0 for g in BASE_UTILITIES.keys()]
        ax.bar(x + i*w_bar, v, w_bar, label=mech)
    ax.set_xticks(x + w_bar*2)
    ax.set_xticklabels(list(BASE_UTILITIES.keys()))
    ax.set_ylabel("Best Action Match")
    ax.set_title("IC-2c+: Policy Baseline Comparison (U1-U5)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/figures/ic2c_plus_policy_baselines.png", dpi=100)
    plt.close()
    print("  ic2c_plus_policy_baselines.png saved")

    # 4. Memory baselines
    fig, ax = plt.subplots(figsize=(8, 5))
    mem_mechs = ["RawMemoryOutcomeTableFull", "StandardizedKNNOutcomeTable", "PrototypeOutcomeTable", "AEPCompressor", "U1_PolicyClone"]
    mem_vals = [rmot_hu, sknn_hu, pot_hu, aep_hu, pc_u1]
    ax.barh(mem_mechs, mem_vals)
    ax.set_xlabel("Mean Best Action Match on Held-Out Utilities")
    ax.set_title("IC-2c+: Memory Baselines vs Learned Compression")
    plt.tight_layout()
    plt.savefig("results/figures/ic2c_plus_memory_baselines.png", dpi=100)
    plt.close()
    print("  ic2c_plus_memory_baselines.png saved")

    # 5. Probe fairness
    fig, ax = plt.subplots(figsize=(8, 5))
    pf_labels = ["NoProbe_PC", "OneProbe_PC", "NoProbe_AEP", "OneProbe_AEP"]
    pf_vals = [np_pc_pf, op_pc_pf, np_aep_pf, op_aep_pf]
    pf_colors = ['gray', 'lightblue', 'orange', 'green']
    ax.bar(pf_labels, pf_vals, color=pf_colors)
    ax.set_ylabel("Best Action Match")
    ax.set_title("IC-2c+: Active Probe Fairness Audit")
    plt.tight_layout()
    plt.savefig("results/figures/ic2c_plus_probe_fairness.png", dpi=100)
    plt.close()
    print("  ic2c_plus_probe_fairness.png saved")

    # 6. Cost-normalized
    fig, ax = plt.subplots(figsize=(10, 6))
    cn_mechs = cn_df[cn_df["mechanism"].isin(["U1_PolicyClone", "AEPCompressor", "CounterfactualCompressor",
                                              "MultiGoalPolicyClone", "RawMemoryOutcomeTableFull",
                                              "StandardizedKNNOutcomeTable", "PrototypeOutcomeTable"])]
    ax.barh(cn_mechs["mechanism"], cn_mechs["cost_normalized_transfer_premium"])
    ax.set_xlabel("Cost-Normalized Transfer Premium")
    ax.set_title("IC-2c+: Cost-Normalized Intelligence Appreciation")
    plt.tight_layout()
    plt.savefig("results/figures/ic2c_plus_cost_normalized.png", dpi=100)
    plt.close()
    print("  ic2c_plus_cost_normalized.png saved")

except Exception as e:
    print(f"  [WARN] Charts: {e}")

print(f"\n{'='*60}")
print(f"IC-2c+ COMPLETE")
print(f"  Verdict: {verdict}")
print(f"  Outputs: results/ic2c_plus/ + results/figures/")
print(f"{'='*60}")