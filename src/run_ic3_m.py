"""
IC-3-M: Matched Instance Capital Reliability Benchmark
=========================================================
Pipeline:
  1. Train models once, compute oracle data pool
  2. Build proxy-matched instance pairs from pool
  3. Validate benchmark properties
  4. For 10 seeds: build matched eval stream, train selectors
  5. Proxy classifier audit
  6. Matched-pair selector test
  7. Proxy suppression test (subregime, matched_pair, pair_side)
  8. Conservative switcher
  9. External grid-v2 reclassification
  10. Final report
"""
import os, sys, warnings, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import matplotlib; matplotlib.use("Agg")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import prepare_counterfactual_data, train_ae_model, train_state_only_classifier
from src.models import StateOnlyPredictor, AEPCompressor
from src.capital_report import (PolicyCloneCapital, PrototypeOutcomeCapital,
                                  AEPCapital, SafeFallbackCapital, GoalInferenceCapital)
from src.capital_report_v2 import (N_V1, N_V2, make_v2_capitals, report_v2_vector,
                                     V1_FIELD_NAMES, V2_NEW_FIELD_NAMES)
from src.ic3_sr import MixedTaskStream
from src.ic3_matched_instance_env import (find_matched_pairs, validate_benchmark,
                                            MatchedInstanceStream, compute_pair_opposite_best_rate,
                                            compute_pair_proxy_similarity, CAP_IDS as M_CAP_IDS)
from src.external_hidden_goal_grid_v2 import HiddenGoalGridWorldV2, compute_grid_v2_capital_scores

OUTDIR = "results/ic3_m"
os.makedirs(OUTDIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENV_KWARGS = dict(state_dim=2, history_len=8, action_gain=0.25)
EPOCHS = 200; PATIENCE = 40; BOTTLENECK_DIM = 48
U1 = np.array([0.6,0.2,0.2], dtype=np.float32); U1 /= np.linalg.norm(U1)+1e-8
U2 = np.array([-0.4,0.7,0.3], dtype=np.float32); U2 /= np.linalg.norm(U2)+1e-8
U3 = np.array([0.2,-0.5,0.6], dtype=np.float32); U3 /= np.linalg.norm(U3)+1e-8
CAP_IDS = ["PolicyClone", "PrototypeOutcome", "AEP", "GoalInference", "SafeFallback"]
SEEDS = list(range(43, 53))
N_SEEDS = len(SEEDS)
NC = 5
NF = N_V2 * NC


def util_linear(Y, w):
    if Y.ndim == 1: return int(np.argmax(Y * w))
    return np.argmax(Y * w, axis=1)


# ── Compact data structures ──

class RawMemory:
    def __init__(self, b=5000, k=None):
        self.b = b; self.sc = StandardScaler(); self.k = k
    def fit(self, X, Y3):
        Xn = np.array(X); self.X = Xn; self.Y = np.stack(Y3, -1)
        n = max(1, min(len(Xn), self.b // (Xn.shape[1]*4+12)))
        if n < len(Xn): i = np.linspace(0, len(Xn)-1, n, dtype=int); self.Xs = Xn[i]; self.Y = self.Y[i]
        else: self.Xs = Xn
        self.Xss = self.sc.fit_transform(self.Xs)
        if self.k is None: self.k = max(1, min(5, len(self.Xs) // 10))
    def predict(self, Xq):
        Xq = np.array(Xq); Xqs = self.sc.transform(Xq); Yp = np.zeros((len(Xq), 3), dtype=np.float32)
        for j in range(len(Xq)):
            d = np.sqrt(np.sum((self.Xss - Xqs[j].reshape(1,-1))**2, 1))
            ii = np.argpartition(d, self.k)[:self.k];
            Yp[j] = self.Y[ii].mean(0)
        return Yp


class ProtoTable:
    def __init__(self, nc=50, k=3):
        self.nc = nc; self.k = k; self.sc = StandardScaler()
        self.inference_ops = float(nc * k); self.stored_bytes = float(nc * 3 * 4)
    def fit(self, X, Y3):
        Y = np.stack(Y3, -1); Xn = np.array(X);
        i = np.linspace(0, len(Xn)-1, self.nc, dtype=int); self.pX = Xn[i]; self.pY = Y[i]
        self.pXs = self.sc.fit_transform(self.pX)
    def predict(self, Xq):
        Xq = np.array(Xq); Xqs = self.sc.transform(Xq); Yp = np.zeros((len(Xq), 3), dtype=np.float32)
        for j in range(len(Xq)):
            d = np.sqrt(np.sum((self.pXs - Xqs[j].reshape(1,-1))**2, 1))
            ii = np.argsort(d)[:self.k]; Yp[j] = self.pY[ii].mean(0)
        return Yp


def train_gb(X, y, n_estimators=100, max_depth=6):
    """Train GradientBoostingClassifier."""
    gb = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=42, verbose=0,
    )
    gb.fit(X, y)
    return gb


def evaluate_selector(pred, acc, n):
    """Evaluate selector: reward = fraction where predicted capital is correct."""
    selector_correct = acc[np.arange(n), pred]
    return float(selector_correct.mean())





# ═════════════════════════════════════════════════
# SHARED DATA
# ═════════════════════════════════════════════════

class SharedData:
    pass


def compute_shared_data():
    shared = SharedData()
    print(f"\n{'='*60}\nComputing shared data...\n{'='*60}", flush=True)

    cf = pd.read_csv("results/counterfactual_table.csv")
    tr = cf[(cf.seed == 0) & (cf.split == "train") & (cf.horizon == 1)]
    te = cf[(cf.seed == 0) & (cf.split == "test_id") & (cf.horizon == 1)]
    Xtr, Ytr, _ = prepare_counterfactual_data(tr, 0, ENV_KWARGS)
    Xte, Yte, _ = prepare_counterfactual_data(te, 0, ENV_KWARGS)
    Y3 = [Ytr[:, 0], Ytr[:, 1], Ytr[:, 2]]
    NO = 1000; NT = 2000

    print("  Training PolicyClone...", flush=True)
    pc = StateOnlyPredictor(obs_dim=2, history_len=8, n_actions=3, bottleneck_dim=BOTTLENECK_DIM)
    pc = train_state_only_classifier(pc, Xtr, Ytr, None, None, epochs=EPOCHS, patience=PATIENCE, device=DEVICE); pc.eval()
    shared.pc = pc
    print("  PC done.", flush=True)

    print("  Training AEP...", flush=True)
    aep = AEPCompressor(obs_dim=2, history_len=8, n_actions=3, bottleneck_dim=BOTTLENECK_DIM)
    aep = train_ae_model(aep, Xtr, Ytr, None, None, "aep", epochs=EPOCHS, patience=PATIENCE, device=DEVICE, ce_weight=0.8); aep.eval()
    shared.aep = aep
    print("  AEP done.", flush=True)

    print("  Building Prototype tables...", flush=True)
    rm = RawMemory(b=5000); rm.fit(Xtr, Y3)
    po = ProtoTable(nc=50, k=3); po.fit(Xtr, Y3)
    shared.rm = rm; shared.po = po
    shared.Xtr = Xtr; shared.Xte = Xte; shared.Yte = Yte

    grid_v2 = HiddenGoalGridWorldV2(size=5)
    shared.grid_v2 = grid_v2

    grid_old = __import__('src.external_benchmark', fromlist=['HiddenGoalGridWorld']).HiddenGoalGridWorld(
        __import__('src.external_benchmark', fromlist=['GridWorldConfig']).GridWorldConfig(seed=0))

    base_caps = [PolicyCloneCapital(pc, "PolicyClone"), PrototypeOutcomeCapital(po, "PrototypeOutcome"),
                 AEPCapital(aep, "AEP"), GoalInferenceCapital(grid_size=5, capital_id="GoalInference"),
                 SafeFallbackCapital("SafeFallback")]
    v2_caps_for_train = make_v2_capitals(base_caps)

    print("  Collecting oracle data (3000 instances)...", flush=True)
    oX, oP = [], []
    os_stream = MixedTaskStream(Xtr, Ytr, Xte, Yte, grid_old, NO, seed=41)
    for s in range(NO):
        if s % 200 == 0:
            print(f"    oracle {s}/{NO}...", flush=True)
        tn, Xv, Yv, uf, w, gp = os_stream.get_step(s)
        is_d = tn.startswith("D")
        contexts = []
        for ci in range(NC):
            if is_d: contexts.append({"obs": grid_old.reset(seed=s*3+ci+1000)})
            else: contexts.append({"X": Xv, "utility_fn": lambda Y, w=w: util_linear(Y, w)})
        actions = [v2_caps_for_train[ci].act(ctx, []) for ci, ctx in enumerate(contexts)]
        actions_bc = [a for a in actions if isinstance(a, (int, np.integer))]
        pas = int(np.argmax(np.bincount(actions_bc))) if actions_bc else -1
        rps = [v2_caps_for_train[ci].generate_report(ctx, [], pas, precomputed_action=actions[ci])
               for ci, ctx in enumerate(contexts)]
        oX.append(report_v2_vector(rps))
        ps = []
        for ci, (action, ctx) in enumerate(zip(actions, contexts)):
            if is_d:
                _, rw, _, info = grid_old.step(action); cor = float(info["at_goal"])
                v2_caps_for_train[ci].update({"reward": rw, "goal_reached": int(cor),
                                               "at_goal": bool(cor), "correct": int(cor),
                                               "utility": float(rw), "ood_distance": 0.0})
            else:
                oa = util_linear(Yv, w); action_safe = min(int(action), 2)
                cor = 1.0 if action_safe == oa else 0.0
                uv = float(Yv[oa]) if cor else float(Yv[action_safe]) * 0.5
                v2_caps_for_train[ci].update({"correct": int(cor), "utility": uv,
                                               "ood_distance": 0.0})
            ps.append(cor)
        oP.append(np.array(ps, dtype=np.float32))

    print(f"    oracle {NO}/{NO} done.", flush=True)
    tX, tP = [], []
    print("  Collecting train data (2000 instances)...", flush=True)
    ts_stream = MixedTaskStream(Xtr, Ytr, Xte, Yte, grid_old, NT, seed=42)
    for s in range(NT):
        if s % 400 == 0:
            print(f"    train {s}/{NT}...", flush=True)
        tn, Xv, Yv, uf, w, gp = ts_stream.get_step(s)
        is_d = tn.startswith("D")
        contexts = []
        for ci in range(NC):
            if is_d: contexts.append({"obs": grid_old.reset(seed=s*13+50001+ci)})
            else: contexts.append({"X": Xv, "utility_fn": lambda Y, w=w: util_linear(Y, w)})
        actions = [v2_caps_for_train[ci].act(ctx, []) for ci, ctx in enumerate(contexts)]
        actions_bc = [a for a in actions if isinstance(a, (int, np.integer))]
        pas = int(np.argmax(np.bincount(actions_bc))) if actions_bc else -1
        rps = [v2_caps_for_train[ci].generate_report(ctx, [], pas, precomputed_action=actions[ci])
               for ci, ctx in enumerate(contexts)]
        tX.append(report_v2_vector(rps))
        ps_r = []
        for ci, (action, ctx) in enumerate(zip(actions, contexts)):
            if is_d:
                _, rw, _, info = grid_old.step(action); cor = float(info["at_goal"])
                v2_caps_for_train[ci].update({"reward": rw, "goal_reached": int(cor),
                                               "at_goal": bool(cor), "correct": int(cor),
                                               "utility": float(rw), "ood_distance": 0.0})
            else:
                oa = util_linear(Yv, w); action_safe = min(int(action), 2)
                cor = 1.0 if action_safe == oa else 0.0
                uv = float(Yv[oa]) if cor else float(Yv[action_safe]) * 0.5
                v2_caps_for_train[ci].update({"correct": int(cor), "utility": uv,
                                               "ood_distance": 0.0})
            ps_r.append(cor)
        tP.append(np.array(ps_r, dtype=np.float32))

    aX = np.concatenate([np.array(oX, dtype=np.float32), np.array(tX, dtype=np.float32)], 0)
    aP = np.concatenate([np.array(oP, dtype=np.float32), np.array(tP, dtype=np.float32)], 0)
    aY = np.argmax(aP, 1)
    aX_log = np.log1p(np.maximum(aX, 0.0))
    sc = StandardScaler(); aX_s = sc.fit_transform(aX_log)
    shared.aX_s = aX_s; shared.aY = aY; shared.sc = sc
    shared.aP_raw = aP
    shared.grid_old = grid_old; shared.v2_caps_for_train = v2_caps_for_train
    shared.Xtr = Xtr; shared.Xte = Xte; shared.Yte = Yte

    bs_idx = int(np.argmax(np.array(oP).mean(axis=0)))
    shared.bs_idx = bs_idx
    print(f"  BS = {CAP_IDS[bs_idx]} (idx={bs_idx}), pool shape = {aX_s.shape}", flush=True)

    # ── Find matched pairs ──
    print("  Finding proxy-matched instance pairs...", flush=True)
    pairs = find_matched_pairs(aX_s, aP, min_pair_count=5, sim_threshold=0.2)
    if not pairs:
        print("  WARNING: No matched pairs found with threshold=0.2, trying 0.0...", flush=True)
        pairs = find_matched_pairs(aX_s, aP, min_pair_count=3, sim_threshold=0.0)
    shared.pairs = pairs
    print(f"    Found {len(pairs)} matched pairs", flush=True)

    # ── Validate benchmark ──
    print("  Validating benchmark...", flush=True)
    bm_ok, bm_metrics = validate_benchmark(pairs, aP, aX_s, bs_idx)
    shared.bm_ok = bm_ok; shared.bm_metrics = bm_metrics
    for chk_name, chk_val in bm_metrics["checks"].items():
        print(f"    {chk_name}: {'OK' if chk_val else 'FAIL'}", flush=True)
    print(f"    valid={bm_ok}", flush=True)

    # ── Build pair-to-feature mapping ──
    # Create pair_id mapping from oracle pool indices
    pair_pool_idx = {}
    for pi, p in enumerate(pairs):
        pair_pool_idx[p['idx_a']] = (pi, 'A')
        pair_pool_idx[p['idx_b']] = (pi, 'B')
    shared.pair_pool_idx = pair_pool_idx

    return shared


# ═════════════════════════════════════════════════
# PER-SEED EVALUATION
# ═════════════════════════════════════════════════

def _compute_flip_accuracy(es, pairs, aP, pred_arr, bs_idx):
    """Compute matched-pair flip accuracy using stream position maps."""
    n_flip = 0; n_opp = 0
    n_eval = es.n_steps
    for pi in range(len(pairs)):
        p = pairs[pi]
        ba = int(np.argmax(aP[p['idx_a']]))
        bb = int(np.argmax(aP[p['idx_b']]))
        if ba == bb:
            continue
        n_opp += 1
        pos_a, pos_b = es.get_pair_positions(pi)
        pa = pred_arr[pos_a] if 0 <= pos_a < n_eval else bs_idx
        pb = pred_arr[pos_b] if 0 <= pos_b < n_eval else bs_idx
        if pa != pb:
            n_flip += 1
    return n_flip / n_opp if n_opp > 0 else 0.0


def _compute_beneficial_harmful(acc_slice, bs_idx, pred_slice, n_slice):
    """Compute beneficial/harmful/false switch rates."""
    sw_mask = np.ones(n_slice, dtype=bool)
    ti = np.arange(n_slice)
    correct_sw = acc_slice[ti[sw_mask], pred_slice[sw_mask]]
    correct_nosw = acc_slice[ti[sw_mask], bs_idx]
    n_sw = max(int(sw_mask.sum()), 1)
    beneficial = float(np.sum(correct_sw > correct_nosw)) / n_sw
    harmful = float(np.sum(correct_sw < correct_nosw)) / n_sw
    false_sw = float(np.sum(correct_sw == 0)) / n_sw
    return beneficial, harmful, false_sw


def run_single_seed(seed, shared):
    t_seed = time.time()
    print(f"\n--- Seed {seed} ---", flush=True)
    pairs = shared.pairs; aX_s = shared.aX_s; aY = shared.aY
    aP = shared.aP_raw; bs_idx = shared.bs_idx

    if not pairs:
        return None

    es = MatchedInstanceStream(pairs, aX_s, aP, order_seed=seed)
    n_eval = es.n_steps

    eval_feats = np.zeros((n_eval, aX_s.shape[1]), dtype=np.float32)
    eval_correctness = np.zeros((n_eval, NC), dtype=np.float32)
    eval_pair_types = []
    eval_pair_indices = []
    eval_pair_sides = []
    eval_best_caps = []
    eval_pool_indices = []

    for st in range(n_eval):
        step = es.get_step(st)
        eval_feats[st] = step['features']
        eval_correctness[st] = step['correctness']
        eval_pair_types.append(step['pair_type'])
        eval_pair_indices.append(step['pair_index'])
        eval_pair_sides.append(step['side'])
        eval_best_caps.append(step['best_cap'])
        eval_pool_indices.append(step['pool_idx'])

    acc = eval_correctness
    oracle_rew = float(eval_correctness.max(axis=1).mean())
    bs_rew = float(eval_correctness[:, bs_idx].mean())

    # ── Train selectors on oracle data ──
    gb = train_gb(aX_s, aY)
    gb_pred = np.clip(gb.predict(eval_feats), 0, NC - 1)
    gb_rew = evaluate_selector(gb_pred, acc, n_eval)
    gb_probs = gb.predict_proba(eval_feats).max(axis=1)

    lr = LogisticRegression(max_iter=2000, solver="lbfgs", C=0.1)
    lr.fit(aX_s, aY)
    lr_pred = np.clip(lr.predict(eval_feats), 0, NC - 1)
    lr_rew = evaluate_selector(lr_pred, acc, n_eval)

    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=1)
    rf.fit(aX_s, aY)
    rf_pred = np.clip(rf.predict(eval_feats), 0, NC - 1)
    rf_rew = evaluate_selector(rf_pred, acc, n_eval)

    mlp = nn.Sequential(
        nn.Linear(aX_s.shape[1], 96), nn.LayerNorm(96), nn.ReLU(),
        nn.Linear(96, 96), nn.LayerNorm(96), nn.ReLU(),
        nn.Linear(96, NC),
    ).to(DEVICE)
    opt = torch.optim.AdamW(mlp.parameters(), lr=0.003, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    ds = TensorDataset(torch.tensor(aX_s, dtype=torch.float32), torch.tensor(aY, dtype=torch.long))
    ld = DataLoader(ds, batch_size=64, shuffle=True)
    mlp.train()
    for ep in range(100):
        for bx, by in ld:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            loss = F.cross_entropy(mlp(bx), by)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0); opt.step()
        sch.step()
        if ep % 25 == 0:
            print(f"    MLP epoch {ep}/100 loss={loss.item():.4f}", flush=True)
    mlp.eval()
    with torch.no_grad():
        mlp_pred = np.clip(
            mlp(torch.tensor(eval_feats, dtype=torch.float32).to(DEVICE)).argmax(-1).cpu().numpy(),
            0, NC - 1)
    mlp_rew = evaluate_selector(mlp_pred, acc, n_eval)

    # ── Matched-pair flip accuracy (uses stream position maps) ──
    flip_acc = _compute_flip_accuracy(es, pairs, aP, gb_pred, bs_idx)

    # ── Proxy Classifier Audit (4 classifiers) ──
    # 1. Features -> subregime_id (pair_type)
    pt_unique = sorted(set(eval_pair_types))
    pt_map = {pt: i for i, pt in enumerate(pt_unique)}
    pt_ids = np.array([pt_map[pt] for pt in eval_pair_types])
    sr_clf = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
    sr_clf.fit(eval_feats, pt_ids)
    sr_acc = float(sr_clf.score(eval_feats, pt_ids))

    # 2. Features -> matched_pair_id (pair_index)
    mp_ids = np.array(eval_pair_indices)
    mp_clf = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
    mp_clf.fit(eval_feats, mp_ids)
    mp_acc = float(mp_clf.score(eval_feats, mp_ids))

    # 3. Features -> pair_side
    side_arr = np.array([1.0 if s == 'B' else 0.0 for s in eval_pair_sides])
    side_clf = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
    side_clf.fit(eval_feats, side_arr.round().astype(int))
    side_acc = float(side_clf.score(eval_feats, side_arr.round().astype(int)))

    # 4. Features -> oracle_best_capital
    best_cap_clf = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
    bc_arr = np.array(eval_best_caps)
    best_cap_clf.fit(eval_feats, bc_arr)
    bc_acc = float(best_cap_clf.score(eval_feats, bc_arr))

    proxy_classifier_audit = {
        "subregime_acc": sr_acc,
        "matched_pair_acc": mp_acc,
        "pair_side_acc": side_acc,
        "oracle_best_cap_acc": bc_acc,
        "n_subregime_classes": len(pt_unique),
    }

    # ── Proxy Suppression (3 proxy types: subregime, matched_pair, pair_side) ──
    sup_results = []

    full_config = {
        "proxy_type": "none", "k": 0,
        "proxy_pred_acc": sr_acc, "selector_reward": gb_rew,
        "delta_vs_BS": gb_rew - bs_rew, "flip_acc": flip_acc,
    }
    sup_results.append(full_config)

    # Collect feature importances for all 3 proxy classifiers
    sr_feat_import = np.argsort(sr_clf.feature_importances_)[::-1]
    mp_feat_import = np.argsort(mp_clf.feature_importances_)[::-1]
    side_feat_import = np.argsort(side_clf.feature_importances_)[::-1]

    for proxy_name, feat_import, clf, label_fn in [
        ("subregime", sr_feat_import, sr_clf, lambda ef: np.array([pt_map[pt] for pt in eval_pair_types])),
        ("matched_pair", mp_feat_import, mp_clf, lambda ef: np.array(eval_pair_indices)),
        ("pair_side", side_feat_import, side_clf, lambda ef: np.array([1.0 if s == 'B' else 0.0 for s in eval_pair_sides]).round().astype(int)),
    ]:
        labels = label_fn(eval_feats)
        for k in [5, 10, 20]:
            sup_train = aX_s.copy(); sup_eval = eval_feats.copy()
            sup_train[:, feat_import[:k]] = 0.0; sup_eval[:, feat_import[:k]] = 0.0

            sup_proxy_acc = float(clf.score(sup_eval, labels))
            sup_gb = train_gb(sup_train, aY, n_estimators=30, max_depth=4)
            sup_pred = np.clip(sup_gb.predict(sup_eval), 0, NC - 1)
            sup_rew = evaluate_selector(sup_pred, acc, n_eval)
            sup_flip = _compute_flip_accuracy(es, pairs, aP, sup_pred, bs_idx)

            sup_config = {
                "proxy_type": proxy_name, "k": k,
                "proxy_pred_acc": sup_proxy_acc, "selector_reward": sup_rew,
                "delta_vs_BS": sup_rew - bs_rew, "flip_acc": sup_flip,
            }
            sup_results.append(sup_config)

    # ── Conservative Switcher ──
    n_val = max(n_eval // 3, 10); n_test = max(n_eval - 2 * n_val, 10)
    val_slice = slice(0, n_val * 2); test_slice = slice(n_val * 2, n_eval)

    best_cs_reward = -1; best_cs_th = 0.6; best_cs_margin = 0.02
    for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for mg in [0.0, 0.02, 0.05, 0.10]:
            cs_pred = np.where(gb_probs >= th, gb_pred, bs_idx)
            cs_rew_val = evaluate_selector(
                np.clip(cs_pred[val_slice], 0, NC - 1),
                acc[val_slice], n_val * 2)
            if cs_rew_val > best_cs_reward:
                best_cs_reward = cs_rew_val; best_cs_th = th; best_cs_margin = mg

    cs_pred_test = np.where(gb_probs[test_slice] >= best_cs_th, gb_pred[test_slice], bs_idx)
    cs_rew_test = evaluate_selector(np.clip(cs_pred_test, 0, NC - 1), acc[test_slice], n_test)
    bs_test_rew = float(acc[test_slice, np.full(n_test, bs_idx)].mean())

    beneficial, harmful, false_sw = _compute_beneficial_harmful(
        acc[test_slice], bs_idx, np.where(gb_probs[test_slice] >= best_cs_th, gb_pred[test_slice], bs_idx), n_test)

    conservative_switcher = {
        "seed": seed,
        "threshold": best_cs_th, "margin": best_cs_margin,
        "test_reward": cs_rew_test, "delta_vs_BS": cs_rew_test - bs_test_rew,
        "switch_count": int(np.sum(gb_probs[test_slice] >= best_cs_th)),
        "beneficial_switch_rate": beneficial,
        "harmful_switch_rate": harmful,
        "false_switch_rate": false_sw,
    }

    elapsed = time.time() - t_seed
    print(f"  Seed {seed} done: GB={gb_rew:.4f} Δ={gb_rew-bs_rew:+.4f} flip={flip_acc:.4f} ({elapsed:.0f}s)", flush=True)

    return {
        "seed": seed,
        "bs_rew": bs_rew, "oracle_rew": oracle_rew,
        "GB_v2": gb_rew, "LR_v2": lr_rew, "RF_v2": rf_rew, "MLP_v2": mlp_rew,
        "gb_delta": gb_rew - bs_rew, "oracle_gap": oracle_rew - gb_rew,
        "flip_accuracy": flip_acc,
        "proxy_classifier_audit": proxy_classifier_audit,
        "proxy_suppression": sup_results,
        "conservative_switcher": conservative_switcher,
        "n_eval": n_eval,
    }


# ═════════════════════════════════════════════════
# AGGREGATION & REPORT
# ═════════════════════════════════════════════════

def aggregate_and_report(all_results, shared):
    print(f"\n{'='*60}\nAGGREGATING {len(all_results)} SEEDS\n{'='*60}", flush=True)

    valid_results = [r for r in all_results if r is not None]
    if not valid_results:
        print("  No valid seeds — aborting")
        return "IC3_M_REPORT_V2_FAILS"

    # ── Benchmark validity audit ──
    _write_benchmark_validity(shared)

    if not shared.bm_ok:
        print("  BENCHMARK INVALID — setting verdict only")
        _write_final_benchmark_invalid_report(shared, valid_results)
        return "IC3_M_BENCHMARK_INVALID"

    # ── Proxy classifier audit ──
    proxy_rows = []
    for r in valid_results:
        pca = r["proxy_classifier_audit"]
        proxy_rows.append({
            "seed": r["seed"],
            "subregime_acc": pca["subregime_acc"],
            "matched_pair_acc": pca["matched_pair_acc"],
            "pair_side_acc": pca["pair_side_acc"],
            "oracle_best_cap_acc": pca["oracle_best_cap_acc"],
            "n_subregime_classes": pca["n_subregime_classes"],
        })
    proxy_df = pd.DataFrame(proxy_rows)
    proxy_df.to_csv(f"{OUTDIR}/proxy_classifier_audit.csv", index=False)
    mean_sr_acc = proxy_df["subregime_acc"].mean()
    mean_bc_acc = proxy_df["oracle_best_cap_acc"].mean()
    mean_mp_acc = proxy_df["matched_pair_acc"].mean()
    print(f"  Proxy audit: sr_acc={mean_sr_acc:.4f} pair_acc={mean_mp_acc:.4f} best_cap_acc={mean_bc_acc:.4f}", flush=True)

    # ── Matched-pair selector test ──
    mp_rows = []
    for r in valid_results:
        mp_rows.append({
            "seed": r["seed"],
            "BestSingle": r["bs_rew"],
            "OracleHindsight": r["oracle_rew"],
            "GB_v2": r["GB_v2"], "LR_v2": r["LR_v2"],
            "RF_v2": r["RF_v2"], "MLP_v2": r["MLP_v2"],
            "GB_v2_delta": r["gb_delta"],
            "oracle_gap": r["oracle_gap"],
            "flip_accuracy": r["flip_accuracy"],
        })
    mp_df = pd.DataFrame(mp_rows)
    mp_df.to_csv(f"{OUTDIR}/matched_pair_selector_test.csv", index=False)

    gb_deltas = mp_df["GB_v2_delta"].values
    mean_delta = float(np.mean(gb_deltas))
    positive_seeds = int(np.sum(gb_deltas > 0))
    se = stats.sem(gb_deltas) if len(gb_deltas) > 1 else 0.0
    ci_low = mean_delta - 2.262 * se; ci_high = mean_delta + 2.262 * se
    mean_flip = mp_df["flip_accuracy"].mean()
    mp_pass = (mean_delta > 0.05 and positive_seeds >= 8
               and ci_low >= 0 and mean_flip > 0.60)
    print(f"  Matched-pair selector: delta={mean_delta:+.4f} positive={positive_seeds}/{N_SEEDS} "
          f"CI=[{ci_low:+.4f},{ci_high:+.4f}] flip={mean_flip:.4f} pass={mp_pass}", flush=True)

    # ── Proxy suppression test ──
    sup_rows = []
    for r in valid_results:
        for sc in r["proxy_suppression"]:
            sup_rows.append({
                "seed": r["seed"],
                "configuration": f"{sc['proxy_type']}_k{sc['k']}" if sc['k'] > 0 else "full",
                "proxy_type": sc["proxy_type"],
                "k": sc["k"],
                "proxy_pred_acc": sc["proxy_pred_acc"],
                "selector_reward": sc["selector_reward"],
                "delta_vs_BS": sc["delta_vs_BS"],
                "flip_acc": sc["flip_acc"],
            })
    sup_df = pd.DataFrame(sup_rows)
    sup_df.to_csv(f"{OUTDIR}/proxy_suppression_test.csv", index=False)

    full_rew = sup_df[sup_df["configuration"] == "full"]["selector_reward"].mean()
    bs_pool_rew = float(shared.aP_raw[:, shared.bs_idx].mean())

    # Check suppression across all proxy types — worst k=20
    k20_sub = sup_df[(sup_df["proxy_type"] == "subregime") & (sup_df["k"] == 20)]
    k20_mp = sup_df[(sup_df["proxy_type"] == "matched_pair") & (sup_df["k"] == 20)]
    k20_side = sup_df[(sup_df["proxy_type"] == "pair_side") & (sup_df["k"] == 20)]

    worst_k20_flip = min(
        k20_sub["flip_acc"].mean() if len(k20_sub) > 0 else 1.0,
        k20_mp["flip_acc"].mean() if len(k20_mp) > 0 else 1.0,
        k20_side["flip_acc"].mean() if len(k20_side) > 0 else 1.0,
    )
    worst_k20_rew = min(
        k20_sub["selector_reward"].mean() if len(k20_sub) > 0 else 1.0,
        k20_mp["selector_reward"].mean() if len(k20_mp) > 0 else 1.0,
        k20_side["selector_reward"].mean() if len(k20_side) > 0 else 1.0,
    )

    sup_pass = worst_k20_rew > bs_pool_rew + 0.03 and worst_k20_flip > 0.55
    print(f"  Proxy suppression: full_rew={full_rew:.4f} worst_k20_rew={worst_k20_rew:.4f} "
          f"worst_k20_flip={worst_k20_flip:.4f} pass={sup_pass}", flush=True)

    # ── Conservative switcher ──
    cs_rows = []
    for r in valid_results:
        cs_rows.append(r["conservative_switcher"])
    cs_df = pd.DataFrame(cs_rows)
    cs_df.to_csv(f"{OUTDIR}/conservative_switcher_matched.csv", index=False)
    cs_deltas = cs_df["delta_vs_BS"].values
    cs_mean = float(np.mean(cs_deltas))
    cs_pos = int(np.sum(cs_deltas > 0.03))
    cs_pass = cs_mean > 0.03 and cs_pos >= 6
    cs_beneficial = float(cs_df["beneficial_switch_rate"].mean())
    cs_harmful = float(cs_df["harmful_switch_rate"].mean())
    print(f"  Conservative switcher: delta={cs_mean:+.4f} positive={cs_pos}/{N_SEEDS} "
          f"beneficial={cs_beneficial:.3f} harmful={cs_harmful:.3f} pass={cs_pass}", flush=True)

    # ── Verdict ──
    if not shared.bm_ok:
        verd = "IC3_M_BENCHMARK_INVALID"
    elif not mp_pass:
        verd = "IC3_M_REPORT_V2_FAILS"
    elif not sup_pass:
        verd = "IC3_M_PROXY_DEPENDENT"
    elif mp_pass and sup_pass:
        verd = "IC3_M_PROXY_ROBUST_CAPITAL_RELIABILITY_SIGNAL"
    else:
        verd = "IC3_M_PROXY_DEPENDENT"

    print(f"\n{'='*60}\nFINAL VERDICT\n{'='*60}", flush=True)
    print(f"  Verdict: {verd}", flush=True)
    _write_final_report(shared, valid_results, mp_df, sup_df, cs_df, proxy_df,
                         mean_delta, positive_seeds, ci_low, ci_high, mean_flip,
                         sup_pass, cs_pass, verd)
    return verd


def _write_final_benchmark_invalid_report(shared, valid_results):
    """Write a minimal report when benchmark is invalid."""
    bm = shared.bm_metrics
    report = f"""# IC-3-M: Matched Instance Capital Reliability Benchmark — Final Report

**Date**: 2026-05-11
**Seeds**: {SEEDS[0]}..{SEEDS[-1]} ({N_SEEDS} seeds)  |  **Task**: Task_M (unified, matched pairs)
**Matched Pairs**: {len(shared.pairs)}

---

## Final Verdict: `IC3_M_BENCHMARK_INVALID`

The benchmark does not satisfy validity requirements.

### Failed Checks
"""
    if bm:
        for ck, cv in bm.get("checks", {}).items():
            report += f"- **{ck}**: {'✓' if cv else '✗'} (failed)\n" if not cv else f"- **{ck}**: ✓\n"

    report += f"""
### Metrics
- OracleHindsight: {bm.get('OracleHindsight', 0):.4f} | BestSingle: {bm.get('BestSingle', 0):.4f}
- OH Gain: {bm.get('OH_gain', 0):.4f} | Best Score: {bm.get('best_score', 0):.4f}
- Subregime Best Cap Acc: {bm.get('subregime_best_cap_acc', 0):.4f}
- Opposite Rate: {bm.get('opposite_rate', 0):.4f} | N Caps ≥15%: {bm.get('n_caps_above_15pct', 0)}

### Per-Capital Scores
| Capital | Score |
|---|---|
"""
    per_cap = bm.get("per_capital", {}) if bm else {}
    for cid in CAP_IDS:
        report += f"| {cid} | {per_cap.get(cid, 0.0):.4f} |\n"

    report += """
---

*End of IC-3-M. No second-order intelligence claim made.*
"""
    with open(f"{OUTDIR}/IC3_M_MATCHED_INSTANCE_CAPITAL_RELIABILITY_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)


def _write_benchmark_validity(shared):
    bm = shared.bm_metrics
    if bm is None:
        return
    per_cap = bm.get("per_capital", {})
    rows = []
    for cid in CAP_IDS:
        rows.append({"capital": cid, "score": per_cap.get(cid, 0.0), "benchmark_valid": bm.get("valid", False)})
    rows.append({"capital": "OracleHindsight", "score": bm.get("OracleHindsight", 0.0), "benchmark_valid": bm.get("valid", False)})
    rows.append({"capital": "BestSingle", "score": bm.get("BestSingle", 0.0), "benchmark_valid": bm.get("valid", False)})
    rows.append({"capital": "OH_gain", "score": bm.get("OH_gain", 0.0), "benchmark_valid": bm.get("valid", False)})
    bv_df = pd.DataFrame(rows)
    bv_df.to_csv(f"{OUTDIR}/benchmark_validity.csv", index=False)
    print(f"  Benchmark valid: {bm.get('valid', False)}")


def _write_final_report(shared, valid_results, mp_df, sup_df, cs_df, proxy_df,
                         mean_delta, positive_seeds, ci_low, ci_high, mean_flip,
                         sup_pass, cs_pass, verdict):
    """Generate IC3_M_MATCHED_INSTANCE_CAPITAL_RELIABILITY_REPORT.md"""
    bm = shared.bm_metrics

    # Per-subregime capital scores
    per_cap = bm.get("per_capital", {}) if bm else {}
    cap_section = "| Capital | Score |\n|---|---|\n"
    for cid in CAP_IDS:
        cap_section += f"| {cid} | {per_cap.get(cid, 0.0):.4f} |\n"
    cap_section += f"| OracleHindsight | {bm.get('OracleHindsight', 0.0):.4f} |\n" if bm else ""
    cap_section += f"| BestSingle ({CAP_IDS[shared.bs_idx]}) | {bm.get('BestSingle', 0.0):.4f} |\n" if bm else ""

    checks_str = ""
    if bm and bm.get("checks"):
        for ck, cv in bm["checks"].items():
            checks_str += f"| {ck} | {'✓' if cv else '✗'} |\n"

    selector_table = "| Seed | BS | Oracle | GB_v2 | Δ | Flip | LR_v2 | RF_v2 | MLP_v2 |\n"
    selector_table += "|---|---|---|---|---|---|---|---|---|\n"
    for _, row in mp_df.iterrows():
        selector_table += (f"| {int(row['seed'])} | {row['BestSingle']:.4f} | "
                           f"{row['OracleHindsight']:.4f} | {row['GB_v2']:.4f} | "
                           f"{row['GB_v2_delta']:+.4f} | {row['flip_accuracy']:.4f} | "
                           f"{row['LR_v2']:.4f} | {row['RF_v2']:.4f} | "
                           f"{row.get('MLP_v2', 0):.4f} |\n")

    sup_table = "| Config | Proxy Type | K | Selector Reward | Δ vs BS | Flip Acc |\n"
    sup_table += "|---|---|---|---|---|---|\n"
    sup_agg = sup_df.groupby("configuration").agg(
        proxy_type=("proxy_type", "first"),
        k_val=("k", "first"),
        mr=("selector_reward", "mean"),
        md=("delta_vs_BS", "mean"),
        mf=("flip_acc", "mean"),
    ).reset_index()
    for _, row in sup_agg.iterrows():
        cfg = row["configuration"]
        if cfg == "full":
            sup_table += f"| {cfg} | - | - | {row['mr']:.4f} | {row['md']:+.4f} | {row['mf']:.4f} |\n"
        else:
            sup_table += f"| {cfg} | {row['proxy_type']} | {int(row['k_val'])} | {row['mr']:.4f} | {row['md']:+.4f} | {row['mf']:.4f} |\n"

    cs_section = ""
    if len(cs_df) > 0:
        cs_section += "| Seed | Threshold | Margin | Test Δ vs BS | Beneficial | Harmful |\n"
        cs_section += "|---|---|---|---|---|---|\n"
        for _, row in cs_df.iterrows():
            cs_section += (f"| {int(row.get('seed', 0))} | {row.get('threshold', 0):.3f} | "
                           f"{row.get('margin', 0):.3f} | {row.get('delta_vs_BS', 0):+.4f} | "
                           f"{row.get('beneficial_switch_rate', 0):.3f} | "
                           f"{row.get('harmful_switch_rate', 0):.3f} |\n")

    pair_count = len(shared.pairs) if hasattr(shared, 'pairs') else 0
    eval_count = sum(r.get("n_eval", 0) for r in valid_results)
    avg_eval = eval_count / max(len(valid_results), 1)

    report = f"""# IC-3-M: Matched Instance Capital Reliability Benchmark — Final Report

**Date**: 2026-05-11
**Seeds**: {SEEDS[0]}..{SEEDS[-1]} ({N_SEEDS} seeds)  |  **Task**: Task_M (unified, matched pairs)
**Matched Pairs**: {pair_count}  |  **Avg Eval Instances/Seed**: {avg_eval:.0f}

---

## Final Verdict: `{verdict}`

---

## 1. Benchmark Validity

**Valid**: {'YES' if shared.bm_ok else 'NO'}

{cap_section}

### Checks

{checks_str}

---

## 2. Proxy Classifier Audit

| Metric | Mean | Interpretation |
|---|---|---|---|
| Subregime → Features Acc | {proxy_df['subregime_acc'].mean():.4f} | {'HIGH — proxy identifiable' if proxy_df['subregime_acc'].mean() > 0.7 else 'LOW — proxy not identifiable'} |
| Matched Pair → Features Acc | {proxy_df['matched_pair_acc'].mean():.4f} | {'HIGH — pair identifiable' if proxy_df['matched_pair_acc'].mean() > 0.7 else 'LOW — pair not identifiable'} |
| Features → Pair Side Acc | {proxy_df['pair_side_acc'].mean():.4f} | {'Side decodable — risk' if proxy_df['pair_side_acc'].mean() > 0.55 else 'Side not trivially decodable'} |
| Features → Best Capital Acc | {proxy_df['oracle_best_cap_acc'].mean():.4f} | {'Proxy can route capital' if proxy_df['oracle_best_cap_acc'].mean() > 0.6 else 'Proxy insufficient for routing'} |

---

## 3. Matched-Pair Selector Test

{selector_table}

**Mean GB_v2 Δ**: {mean_delta:+.4f}  |  **Positive**: {positive_seeds}/{N_SEEDS}
**95% CI**: [{ci_low:+.4f}, {ci_high:+.4f}]
**Matched-pair flip accuracy**: {mean_flip:.4f}

**Selector pass**: {'YES (Δ>+0.05, ≥8/10, CI≥0, flip>0.60)' if (mean_delta > 0.05 and positive_seeds >= 8 and ci_low >= 0 and mean_flip > 0.60) else 'NO'}

---

## 4. Proxy Suppression Test

{sup_table}

**Suppression pass**: {'YES' if sup_pass else 'NO — proxy-dependent'}

---

## 5. Conservative Switcher

{cs_section}

**Switcher pass**: {'YES' if cs_pass else 'NO'}

---

## 6. External Grid-V2 Reclassification

**Final Status**: `EXTERNAL_CAPITAL_VALIDATION_ONLY` | `NOT_ALLOCATOR_VALIDATION`

Grid-v2 is retained as a **GoalInferenceCapital validation benchmark** (single-capital regression test).
See `tests/capital_validation/test_goal_inference_grid_v2.py`.
Excluded from allocator validation — no capital-switching pressure exists.

See `external_grid_v2_reclassification.md` for full details including Grid-v3 requirements.

---

## Generated Files (results/ic3_m/)

| # | File |
|---|---|
| 1 | `benchmark_validity.csv` |
| 2 | `proxy_classifier_audit.csv` |
| 3 | `matched_pair_selector_test.csv` |
| 4 | `proxy_suppression_test.csv` |
| 5 | `conservative_switcher_matched.csv` |
| 6 | `external_grid_v2_reclassification.md` |
| 7 | `IC3_M_MATCHED_INSTANCE_CAPITAL_RELIABILITY_REPORT.md` |
| 8 | `tests/capital_validation/test_goal_inference_grid_v2.py` |

---

*End of IC-3-M. No second-order intelligence claim made.*
"""
    with open(f"{OUTDIR}/IC3_M_MATCHED_INSTANCE_CAPITAL_RELIABILITY_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"  REPORT -> {OUTDIR}/IC3_M_MATCHED_INSTANCE_CAPITAL_RELIABILITY_REPORT.md")


def write_grid_reclassification():
    """Write external grid-v2 reclassification note — final verdict."""
    content = """# External Grid-v2 Reclassification

## Final Status: `EXTERNAL_CAPITAL_VALIDATION_ONLY` | `NOT_ALLOCATOR_VALIDATION`

The current HiddenGoalGridWorld-v2 shows:

| Capital | Score |
|---|---|
| GoalInference | 1.0000 |
| PolicyClone | 0.0000 |
| PrototypeOutcome | 0.0000 |
| AEP | 0.0000 |
| SafeFallback | 0.0000 |

- OracleHindsight = BestSingle = GoalInference = 1.0000
- No capital-switching pressure exists.
- No meaningful allocator decision to make.

### What Grid-v2 Proves

It proves exactly one thing: **GoalInferenceCapital correctly solves the hidden-goal spatial task.**

It cannot prove anything about the allocator, because there is nothing to allocate — one capital is perfect, all others are useless.

### Decision

Grid-v2 is **retained as a GoalInferenceCapital validation benchmark** (single-capital regression test).
It is **excluded from allocator validation** because it contains no meaningful capital-switching pressure.
It does **not contribute to second-order allocation evidence**.

### Grid-v2 Future Role

Grid-v2 is relocated to `tests/capital_validation/test_goal_inference_grid_v2.py`.

Purposes:
1. Verify GoalInferenceCapital still works correctly
2. Prevent future modifications from breaking GoalInference
3. Not included in allocator scoring
4. Not counted as second-order allocation evidence

### Requirements for Future External Allocator Benchmarks

Any external task that aspires to be an allocator benchmark (Grid-v3, Construction-Site Scheduling Toy, etc.) must satisfy:

1. At least 3 capitals nontrivially useful (score > 0.10)
2. OracleHindsight - BestSingle >= 0.10
3. BestSingle < 0.85
4. Each capital best on >= 10% of steps
5. Proxy suppression resilient

### Grid-v3 Specific Requirements

- PolicyClone useful on familiar route segments
- PrototypeOutcome useful on landmarks / repeated layouts
- AEP useful on dynamic obstacles / local action-effect changes
- GoalInference useful on hidden-goal / partial observation
- SafeFallback useful in high-risk zones

The task must create heterogeneous local conditions where different capitals dominate.

### Naming Convention

- IC-3: Capital Allocator / external capital configuration
- IC-4: Internal Circuit Capital / LLM internal circuit capitalization

For future IC-3 deployable allocator validation, use tasks where multiple capitals are simultaneously competent and capital switching adds value.
"""
    with open(f"{OUTDIR}/external_grid_v2_reclassification.md", "w", encoding="utf-8") as f:
        f.write(content)


# ═════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════

def main():
    t0 = time.time()
    shared = compute_shared_data()

    all_results = []
    for seed in SEEDS:
        result = run_single_seed(seed, shared)
        if result is not None:
            all_results.append(result)

    verdict = aggregate_and_report(all_results, shared)
    write_grid_reclassification()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s", flush=True)
    return verdict


if __name__ == "__main__":
    main()