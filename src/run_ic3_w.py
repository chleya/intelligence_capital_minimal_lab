"""
IC-3-W: Within-Task Capital Switching Benchmark
=================================================
Pipeline:
  1. Train models once, compute exclusive capital-correct indices
  2. Validate subregime taxonomy (W1/W2/W3/W4)
  3. Build WithinTaskMixedRegimeStream (all task_id = Task_W)
  4. For 10 seeds: collect eval data, train selectors
  5. Within-task selector test
  6. Proxy suppression within-task
  7. Conservative switcher
  8. External grid diagnostic
  9. Final report
"""
import os, sys, warnings, time
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
from src.ic3_within_task_env import (WithinTaskMixedRegimeStream,
                                       build_per_capital_exclusive_indices,
                                       validate_subregime_taxonomy)
from src.external_hidden_goal_grid_v2 import HiddenGoalGridWorldV2, compute_grid_v2_capital_scores

OUTDIR = "results/ic3_w"
os.makedirs(OUTDIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENV_KWARGS = dict(state_dim=2, history_len=8, action_gain=0.25)
EPOCHS = 200; PATIENCE = 40; BOTTLENECK_DIM = 48
U1 = np.array([0.6,0.2,0.2], dtype=np.float32); U1 /= np.linalg.norm(U1)+1e-8
U2 = np.array([-0.4,0.7,0.3], dtype=np.float32); U2 /= np.linalg.norm(U2)+1e-8
U3 = np.array([0.2,-0.5,0.6], dtype=np.float32); U3 /= np.linalg.norm(U3)+1e-8
CAP_IDS = ["PolicyClone","PrototypeOutcome","AEP","GoalInference","SafeFallback"]
SEEDS = list(range(43, 53))
N_SEEDS = len(SEEDS)
NC = 5
NF = N_V2 * NC


def util_linear(Y, w):
    if Y.ndim==1: return int(np.argmax(Y*w))
    return np.argmax(Y*w, axis=1)


# ── Compact data structures ──

class RawMemory:
    def __init__(self,b=5000,k=None):
        self.b=b; self.sc=StandardScaler(); self.k=k
    def fit(self,X,Y3):
        Xn=np.array(X); self.X=Xn; self.Y=np.stack(Y3,-1)
        n=max(1,min(len(Xn),self.b//(Xn.shape[1]*4+12)))
        if n<len(Xn): i=np.linspace(0,len(Xn)-1,n,dtype=int); self.Xs=Xn[i]; self.Y=self.Y[i]
        else: self.Xs=Xn
        self.Xss=self.sc.fit_transform(self.Xs)
        if self.k is None: self.k=max(1,min(5,len(self.Xs)//10))
    def predict(self,Xq):
        Xq=np.array(Xq); Xqs=self.sc.transform(Xq); Yp=np.zeros((len(Xq),3),dtype=np.float32)
        for j in range(len(Xq)):
            d=np.sum((Xqs[j]-self.Xss)**2,1); k=min(self.k,len(self.Xs))
            ni=np.argpartition(d,k-1)[:k] if k>0 else [0]; Yp[j]=self.Y[ni].mean(axis=0)
        return Yp

class ProtoTable:
    def __init__(self,nc=50,k=3): self.nc=nc; self.k=k; self.inference_ops=float(nc*k); self.stored_bytes=float(nc*3*4)
    def fit(self,X,Y3):
        Xn=np.array(X); Yt=np.stack(Y3,-1); n=len(Xn); nc=min(self.nc,n)
        rng=np.random.default_rng(42); i=rng.choice(n,nc,replace=False); self.P=Xn[i]; self.L=np.zeros(n,dtype=np.int64)
        for _ in range(10):
            for j in range(n): self.L[j]=np.argmin(np.sum((Xn[j]-self.P)**2,1))
            for p in range(nc):
                m=self.L==p
                if m.sum()>0: self.P[p]=Xn[m].mean(axis=0)
        self.PT=np.zeros((nc,3),dtype=np.float32)
        for p in range(nc):
            m=self.L==p
            if m.sum()>0: self.PT[p]=Yt[m].mean(axis=0)
    def predict(self,Xq):
        Xq=np.array(Xq); Yp=np.zeros((len(Xq),3),dtype=np.float32)
        for j in range(len(Xq)):
            d=np.sum((Xq[j]-self.P)**2,1); k=min(self.k,len(self.P))
            tk=np.argpartition(d,k-1)[:k]; Yp[j]=self.PT[tk].mean(axis=0)
        return Yp


def train_gb(X_train, y_train, n_estimators=100, max_depth=6):
    gb = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    gb.fit(X_train, y_train)
    return gb

def evaluate_selector(pred_idx, acc, n_eval):
    sp = np.clip(pred_idx, 0, NC-1)
    return float(acc[np.arange(n_eval), sp].mean())


# ═════════════════════════════════════════════════
# SHARED DATA
# ═════════════════════════════════════════════════

class SharedData:
    pass

def compute_shared_data():
    shared = SharedData()
    print(f"\n{'='*60}\nComputing shared data...\n{'='*60}")

    cf = pd.read_csv("results/counterfactual_table.csv")
    tr = cf[(cf.seed==0)&(cf.split=="train")&(cf.horizon==1)]
    te = cf[(cf.seed==0)&(cf.split=="test_id")&(cf.horizon==1)]
    Xtr, Ytr, _ = prepare_counterfactual_data(tr, 0, ENV_KWARGS)
    Xte, Yte, _ = prepare_counterfactual_data(te, 0, ENV_KWARGS)
    Y3 = [Ytr[:,0], Ytr[:,1], Ytr[:,2]]
    NO = 1000; NT = 2000

    print("  Training PolicyClone...")
    pc = StateOnlyPredictor(obs_dim=2, history_len=8, n_actions=3, bottleneck_dim=BOTTLENECK_DIM)
    pc = train_state_only_classifier(pc, Xtr, Ytr, None, None, epochs=EPOCHS, patience=PATIENCE, device=DEVICE); pc.eval()
    shared.pc = pc

    print("  Training AEP...")
    aep = AEPCompressor(obs_dim=2, history_len=8, n_actions=3, bottleneck_dim=BOTTLENECK_DIM)
    aep = train_ae_model(aep, Xtr, Ytr, None, None, "aep", epochs=EPOCHS, patience=PATIENCE, device=DEVICE, ce_weight=0.8); aep.eval()
    shared.aep = aep

    print("  Building Prototype tables...")
    rm = RawMemory(b=5000); rm.fit(Xtr, Y3)
    po = ProtoTable(nc=50, k=3); po.fit(Xtr, Y3)
    shared.rm = rm; shared.po = po
    shared.Xtr = Xtr; shared.Xte = Xte; shared.Yte = Yte

    # Grid-v2 environment
    grid_v2 = HiddenGoalGridWorldV2(size=5)
    shared.grid_v2 = grid_v2

    # Exclusive capital-correct indices
    print("  Computing exclusive capital-correct indices...")
    pc_only, aep_only, po_only = build_per_capital_exclusive_indices(
        Xte, Yte, pc, aep, po, device=DEVICE, U1=U1, util_fn=util_linear)
    print(f"    PC-only={len(pc_only)} AEP-only={len(aep_only)} PO-only={len(po_only)}")
    shared.pc_only = pc_only; shared.aep_only = aep_only; shared.po_only = po_only

    # Validate subregime taxonomy
    print("  Validating subregime taxonomy...")
    tax_scores, tax_match, tax_margin_ok, tax_valid = validate_subregime_taxonomy(
        pc_only, aep_only, po_only, Xte, Yte, pc, aep, po, grid_v2, device=DEVICE, U1=U1, util_fn=util_linear)
    shared.tax_scores = tax_scores; shared.tax_match = tax_match
    shared.tax_margin_ok = tax_margin_ok; shared.tax_valid = tax_valid

    for k, v in tax_scores.items():
        print(f"    {k}: best={v['best']} expected={v['expected']} margin={v['margin']:.4f} match={v['match']}")
    print(f"    match={tax_match}/4 margin_ok={tax_margin_ok}/4 valid={tax_valid}")

    # Oracle + train data (standard stream)
    print("  Collecting oracle + train data...")
    grid_old = __import__('src.external_benchmark', fromlist=['HiddenGoalGridWorld']).HiddenGoalGridWorld(
        __import__('src.external_benchmark', fromlist=['GridWorldConfig']).GridWorldConfig(seed=0))

    base_caps = [PolicyCloneCapital(pc,"PolicyClone"), PrototypeOutcomeCapital(po,"PrototypeOutcome"),
                 AEPCapital(aep,"AEP"), GoalInferenceCapital(grid_size=5,capital_id="GoalInference"),
                 SafeFallbackCapital("SafeFallback")]
    v2_caps_for_train = make_v2_capitals(base_caps)

    oX, oP = [], []
    os_stream = MixedTaskStream(Xtr, Ytr, Xte, Yte, grid_old, NO, seed=41)
    for s in range(NO):
        tn, Xv, Yv, uf, w, gp = os_stream.get_step(s)
        is_d = tn.startswith("D")
        contexts = []
        for ci in range(NC):
            if is_d: contexts.append({"obs": grid_old.reset(seed=s*3+ci+1000)})
            else: contexts.append({"X": Xv, "utility_fn": lambda Y, w=w: util_linear(Y, w)})
        actions = [v2_caps_for_train[ci].act(ctx, []) for ci, ctx in enumerate(contexts)]
        actions_bc = [a for a in actions if isinstance(a, (int, np.integer))]
        pas = int(np.argmax(np.bincount(actions_bc))) if actions_bc else -1
        rps = [v2_caps_for_train[ci].generate_report(ctx, [], pas, precomputed_action=actions[ci]) for ci, ctx in enumerate(contexts)]
        oX.append(report_v2_vector(rps))
        ps = []
        for ci, (action, ctx) in enumerate(zip(actions, contexts)):
            if is_d:
                _, rw, _, info = grid_old.step(action); cor = float(info["at_goal"])
                v2_caps_for_train[ci].update({"reward":rw,"goal_reached":int(cor),"at_goal":bool(cor),"correct":int(cor),"utility":float(rw),"ood_distance":0.0})
            else:
                oa = util_linear(Yv, w); action_safe = min(int(action), 2); cor = 1.0 if action_safe==oa else 0.0
                uv = float(Yv[oa]) if cor else float(Yv[action_safe])*0.5
                nn_d = np.sqrt(np.mean((Xv-Xtr)**2,1))
                v2_caps_for_train[ci].update({"correct":int(cor),"utility":uv,"ood_distance":float(np.min(nn_d))})
            ps.append(cor)
        oP.append(np.array(ps, dtype=np.float32))

    tX, tP = [], []
    ts_stream = MixedTaskStream(Xtr, Ytr, Xte, Yte, grid_old, NT, seed=42)
    for s in range(NT):
        tn, Xv, Yv, uf, w, gp = ts_stream.get_step(s)
        is_d = tn.startswith("D")
        contexts = []
        for ci in range(NC):
            if is_d: contexts.append({"obs": grid_old.reset(seed=s*13+50001+ci)})
            else: contexts.append({"X": Xv, "utility_fn": lambda Y, w=w: util_linear(Y, w)})
        actions = [v2_caps_for_train[ci].act(ctx, []) for ci, ctx in enumerate(contexts)]
        actions_bc = [a for a in actions if isinstance(a, (int, np.integer))]
        pas = int(np.argmax(np.bincount(actions_bc))) if actions_bc else -1
        rps = [v2_caps_for_train[ci].generate_report(ctx, [], pas, precomputed_action=actions[ci]) for ci, ctx in enumerate(contexts)]
        tX.append(report_v2_vector(rps))
        ps_r = []
        for ci, (action, ctx) in enumerate(zip(actions, contexts)):
            if is_d:
                _, rw, _, info = grid_old.step(action); cor = float(info["at_goal"])
                v2_caps_for_train[ci].update({"reward":rw,"goal_reached":int(cor),"at_goal":bool(cor),"correct":int(cor),"utility":float(rw),"ood_distance":0.0})
            else:
                oa = util_linear(Yv, w); action_safe = min(int(action), 2); cor = 1.0 if action_safe==oa else 0.0
                uv = float(Yv[oa]) if cor else float(Yv[action_safe])*0.5
                nn_d = np.sqrt(np.mean((Xv-Xtr)**2,1))
                v2_caps_for_train[ci].update({"correct":int(cor),"utility":uv,"ood_distance":float(np.min(nn_d))})
            ps_r.append(cor)
        tP.append(np.array(ps_r, dtype=np.float32))

    aX = np.concatenate([np.array(oX, dtype=np.float32), np.array(tX, dtype=np.float32)], 0)
    aP = np.concatenate([np.array(oP, dtype=np.float32), np.array(tP, dtype=np.float32)], 0)
    aY = np.argmax(aP, 1)
    aX_log = np.log1p(np.maximum(aX, 0.0))
    sc = StandardScaler(); aX_s = sc.fit_transform(aX_log)
    shared.aX_s = aX_s; shared.aY = aY; shared.sc = sc
    shared.aP_raw = aP
    bs_idx = int(np.argmax(np.array(oP).mean(axis=0)))
    shared.bs_idx = bs_idx

    print(f"  BS = {CAP_IDS[bs_idx]} (idx={bs_idx}), data shape = {aX_s.shape}")
    return shared


# ═════════════════════════════════════════════════
# PER-SEED EVALUATION
# ═════════════════════════════════════════════════

def run_single_seed(seed, shared):
    print(f"\n--- Seed {seed} ---")
    Xte = shared.Xte; Yte = shared.Yte; grid_v2 = shared.grid_v2
    pc = shared.pc; aep = shared.aep; po = shared.po
    pc_only = shared.pc_only; aep_only = shared.aep_only; po_only = shared.po_only
    aX_s = shared.aX_s; aY = shared.aY; sc = shared.sc; bs_idx = shared.bs_idx
    n_per = 150; n_eval = n_per * 4

    # Build eval stream
    es = WithinTaskMixedRegimeStream(pc_only, aep_only, po_only, Xte, Yte, grid_v2,
                                     n_per_subregime=n_per, block_size=12, seed=seed)

    eval_v2 = make_v2_capitals([PolicyCloneCapital(pc,"PolicyClone"), PrototypeOutcomeCapital(po,"PrototypeOutcome"),
                                 AEPCapital(aep,"AEP"), GoalInferenceCapital(grid_size=5,capital_id="GoalInference"),
                                 SafeFallbackCapital("SafeFallback")])

    acc = np.zeros((n_eval, NC), dtype=np.float32)
    rpt_vecs = np.zeros((n_eval, NF), dtype=np.float32)
    sr_labels = []

    for st in range(n_eval):
        tn, Xv, Yv, sr_id = es.get_step(st)
        sr_labels.append(sr_id)

        if sr_id == "W4":
            rps = []
            actions_first = []
            for ci in range(NC):
                obs = grid_v2.reset(seed=st * 7 + seed * 1000 + st + ci)
                ctx0 = {"obs": obs, "_reset": True}
                a0 = eval_v2[ci].act(ctx0, [])
                actions_first.append(a0)
                reached = False
                for _step in range(grid_v2.max_steps):
                    obs, _, done, info = grid_v2.step(a0)
                    if info["at_goal"]:
                        reached = True
                        break
                    if done:
                        break
                    ctx = {"obs": obs}
                    a0 = eval_v2[ci].act(ctx, [])
                acc[st, ci] = float(reached)
                rps.append(eval_v2[ci].generate_report(ctx0, [], -1, precomputed_action=actions_first[ci]))
            actions_bc = [a for a in actions_first if isinstance(a, (int, np.integer))]
            pas_pt = int(np.argmax(np.bincount(actions_bc))) if actions_bc else -1
        else:
            contexts = [{"X": Xv, "utility_fn": lambda Y, w=U1: util_linear(Y, w)} for _ in range(NC)]
            actions = [eval_v2[ci].act(ctx, []) for ci, ctx in enumerate(contexts)]
            actions_bc = [a for a in actions if isinstance(a, (int, np.integer))]
            pas_pt = int(np.argmax(np.bincount(actions_bc))) if actions_bc else -1
            rps = [eval_v2[ci].generate_report(ctx, [], pas_pt, precomputed_action=actions[ci]) for ci, ctx in enumerate(contexts)]
            oa = util_linear(Yv, U1)
            for ci, a in enumerate(actions):
                acc[st, ci] = 1.0 if min(int(a), 2) == oa else 0.0

        rpt_vecs[st] = report_v2_vector(rps)

    eval_log = np.log1p(np.maximum(rpt_vecs, 0.0))
    eval_s = sc.transform(eval_log)

    oracle_rew = acc.max(axis=1)
    bs_rew = float(acc[np.arange(n_eval), np.full(n_eval, bs_idx)].mean())

    # ── Train selectors ──
    # GB_v2
    gb = train_gb(aX_s, aY)
    gb_pred = np.clip(gb.predict(eval_s), 0, NC-1)
    gb_rew = evaluate_selector(gb_pred, acc, n_eval)
    gb_probs = gb.predict_proba(eval_s).max(axis=1)

    # LR
    lr = LogisticRegression(max_iter=2000, solver="lbfgs", C=0.1)
    lr.fit(aX_s, aY)
    lr_pred = np.clip(lr.predict(eval_s), 0, NC-1)
    lr_rew = evaluate_selector(lr_pred, acc, n_eval)

    # RF
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=1)
    rf.fit(aX_s, aY)
    rf_pred = np.clip(rf.predict(eval_s), 0, NC-1)
    rf_rew = evaluate_selector(rf_pred, acc, n_eval)

    # MLP
    mlp = nn.Sequential(nn.Linear(aX_s.shape[1], 96), nn.LayerNorm(96), nn.ReLU(),
                        nn.Linear(96, 96), nn.LayerNorm(96), nn.ReLU(),
                        nn.Linear(96, NC)).to(DEVICE)
    opt = torch.optim.AdamW(mlp.parameters(), lr=0.003, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    ds = TensorDataset(torch.tensor(aX_s, dtype=torch.float32), torch.tensor(aY, dtype=torch.long))
    ld = DataLoader(ds, batch_size=64, shuffle=True)
    mlp.train()
    for _ in range(300):
        for bx, by in ld:
            bx, by = bx.to(DEVICE), by.to(DEVICE); loss = F.cross_entropy(mlp(bx), by)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0); opt.step()
        sch.step()
    mlp.eval()
    with torch.no_grad():
        mlp_pred = np.clip(mlp(torch.tensor(eval_s, dtype=torch.float32).to(DEVICE)).argmax(-1).cpu().numpy(), 0, NC-1)
    mlp_rew = evaluate_selector(mlp_pred, acc, n_eval)

    # ── Proxy Suppression (subregime) ──
    sr_map = {"W1": 0, "W2": 1, "W3": 2, "W4": 3}
    sr_ids = np.array([sr_map.get(sr, 0) for sr in sr_labels])

    sr_clf = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
    sr_clf.fit(eval_s, sr_ids)
    sr_acc_full = float(sr_clf.score(eval_s, sr_ids))
    sr_features = np.argsort(sr_clf.feature_importances_)[::-1]

    proxy_suppression = {"full": {"sr_pred_acc": sr_acc_full, "selector_reward": gb_rew,
                                   "delta_vs_BS": gb_rew - bs_rew}}
    for k in [5, 10, 20]:
        sup_train = aX_s.copy(); sup_eval = eval_s.copy()
        sup_train[:, sr_features[:k]] = 0.0; sup_eval[:, sr_features[:k]] = 0.0
        sr_acc_supp = float(np.mean(np.argmax(sr_clf.predict_proba(sup_eval), 1) == sr_ids))
        sup_gb = train_gb(sup_train, aY, n_estimators=50, max_depth=4)
        sup_pred = np.clip(sup_gb.predict(sup_eval), 0, NC-1)
        sup_rew = evaluate_selector(sup_pred, acc, n_eval)
        proxy_suppression[f"suppress_k{k}"] = {"sr_pred_acc": sr_acc_supp, "selector_reward": sup_rew,
                                                 "delta_vs_BS": sup_rew - bs_rew}

    # ── Per-subregime analysis ──
    per_sr = {}
    for sr_name in ["W1", "W2", "W3", "W4"]:
        mask = np.array([s == sr_name for s in sr_labels])
        if mask.sum() < 5: continue
        gb_sr = evaluate_selector(gb_pred[mask], acc[mask], int(mask.sum()))
        bs_sr = float(acc[mask, bs_idx].mean())
        oh_sr = float(oracle_rew[mask].mean())
        per_sr[sr_name] = {"GB_v2": gb_sr, "BestSingle": bs_sr,
                           "OracleHindsight": oh_sr, "delta": gb_sr - bs_sr}

    # ── Conservative Switcher ──
    n_val = n_eval // 3; n_test = n_eval - 2*n_val
    val_slice = slice(0, n_val*2); test_slice = slice(n_val*2, n_eval)

    best_cs_reward = -1; best_cs_th = 0.6; best_cs_margin = 0.02
    for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for mg in [0.0, 0.02, 0.05, 0.10]:
            cs_pred = np.where(gb_probs >= th, gb_pred, bs_idx)
            cs_rew_val = evaluate_selector(np.clip(cs_pred[val_slice], 0, NC-1), acc[val_slice], n_val*2)
            if cs_rew_val > best_cs_reward:
                best_cs_reward = cs_rew_val; best_cs_th = th; best_cs_margin = mg

    cs_pred_test = np.where(gb_probs[test_slice] >= best_cs_th, gb_pred[test_slice], bs_idx)
    cs_rew_test = evaluate_selector(np.clip(cs_pred_test, 0, NC-1), acc[test_slice], n_test)
    bs_test_rew = float(acc[test_slice, np.full(n_test, bs_idx)].mean())

    sw_mask = gb_probs[test_slice] >= best_cs_th
    n_sw = int(sw_mask.sum())
    if n_sw > 0:
        test_acc = acc[test_slice]
        ti = np.arange(n_test)
        correct_sw = test_acc[ti[sw_mask], gb_pred[test_slice][sw_mask]]
        correct_nosw = test_acc[ti[sw_mask], bs_idx]
        beneficial = float(np.sum(correct_sw > correct_nosw)) / n_sw
        harmful = float(np.sum(correct_sw < correct_nosw)) / n_sw
        false_sw = float(np.sum(correct_sw == 0)) / n_sw
    else:
        beneficial = 0; harmful = 0; false_sw = 0

    conservative_switcher = {
        "threshold": best_cs_th, "margin": best_cs_margin,
        "test_reward": cs_rew_test, "delta_vs_BS": cs_rew_test - bs_test_rew,
        "switch_count": n_sw, "beneficial_switch_rate": beneficial,
        "harmful_switch_rate": harmful, "false_switch_rate": false_sw,
    }

    return {
        "seed": seed, "bs_rew": bs_rew, "oracle_rew": float(oracle_rew.mean()),
        "GB_v2": gb_rew, "LR_v2": lr_rew, "RF_v2": rf_rew, "MLP_v2": mlp_rew,
        "gb_delta": gb_rew - bs_rew, "oracle_gap": float(oracle_rew.mean()) - gb_rew,
        "proxy_suppression": proxy_suppression,
        "per_subregime": per_sr,
        "conservative_switcher": conservative_switcher,
    }


# ═════════════════════════════════════════════════
# AGGREGATION & REPORT
# ═════════════════════════════════════════════════

def aggregate_and_report(all_results, shared):
    print(f"\n{'='*60}\nAGGREGATING {len(all_results)} SEEDS\n{'='*60}")

    # ── Per-subregime capital matrix ──
    tax_df_rows = []
    for k, v in shared.tax_scores.items():
        row = {"subregime": k, "expected_best": v["expected"], "actual_best": v["best"],
               "margin": v["margin"], "match": v["match"]}
        for cid in CAP_IDS:
            row[cid] = v["scores"].get(cid, 0.0)
        tax_df_rows.append(row)
    tax_df = pd.DataFrame(tax_df_rows)
    tax_df.to_csv(f"{OUTDIR}/per_subregime_capital_matrix.csv", index=False)
    print(f"  Taxonomy: match={shared.tax_match}/4 margin_ok={shared.tax_margin_ok}/4 valid={shared.tax_valid}")

    if not shared.tax_valid:
        print("  BENCHMARK INVALID — aborting")
        return "IC3_W_CAPITAL_TAXONOMY_FAILS"

    # ── Within-task selector test ──
    wt_rows = []
    for r in all_results:
        wt_rows.append({
            "seed": r["seed"], "BestSingle": r["bs_rew"], "OracleHindsight": r["oracle_rew"],
            "GB_v2": r["GB_v2"], "LR_v2": r["LR_v2"], "RF_v2": r["RF_v2"], "MLP_v2": r["MLP_v2"],
            "GB_v2_delta": r["gb_delta"], "oracle_gap": r["oracle_gap"],
            "cumulative_regret": r["oracle_rew"] * 600 - r["GB_v2"] * 600,
        })
    wt_df = pd.DataFrame(wt_rows)
    wt_df.to_csv(f"{OUTDIR}/within_task_selector_test.csv", index=False)

    gb_deltas = wt_df["GB_v2_delta"].values
    mean_delta = float(np.mean(gb_deltas))
    positive_seeds = int(np.sum(gb_deltas > 0))
    se = stats.sem(gb_deltas) if len(gb_deltas) > 1 else 0
    ci_low = mean_delta - 2.262 * se; ci_high = mean_delta + 2.262 * se
    wt_pass = mean_delta > 0.05 and positive_seeds >= 8 and ci_low >= 0
    print(f"  Within-task selector: delta={mean_delta:+.4f} positive={positive_seeds}/{N_SEEDS} CI=[{ci_low:+.4f},{ci_high:+.4f}] pass={wt_pass}")

    # Per-subregime aggregate
    per_sr_agg = defaultdict(list)
    for r in all_results:
        for sr_name, val in r["per_subregime"].items():
            per_sr_agg[sr_name].append(val)
    per_sr_rows = []
    for sr_name in ["W1","W2","W3","W4"]:
        if sr_name not in per_sr_agg: continue
        vals = per_sr_agg[sr_name]
        per_sr_rows.append({
            "subregime": sr_name, "mean_GB_v2": float(np.mean([v["GB_v2"] for v in vals])),
            "mean_BestSingle": float(np.mean([v["BestSingle"] for v in vals])),
            "mean_delta": float(np.mean([v["delta"] for v in vals])),
            "positive_seeds": int(np.sum([v["delta"] > 0 for v in vals])),
        })

    # ── Proxy suppression ──
    sup_by_cfg = defaultdict(list)
    for r in all_results:
        for cfg, vals in r["proxy_suppression"].items():
            sup_by_cfg[cfg].append(vals)
    sup_rows = []
    for cfg in sorted(sup_by_cfg):
        vals = sup_by_cfg[cfg]
        sup_rows.append({
            "configuration": cfg,
            "mean_sr_pred_acc": float(np.mean([v["sr_pred_acc"] for v in vals])),
            "mean_selector_reward": float(np.mean([v["selector_reward"] for v in vals])),
            "mean_delta_vs_BS": float(np.mean([v["delta_vs_BS"] for v in vals])),
        })
    sup_df = pd.DataFrame(sup_rows)
    sup_df.to_csv(f"{OUTDIR}/within_task_proxy_suppression.csv", index=False)

    full_sr_acc = sup_rows[0]["mean_sr_pred_acc"] if sup_rows else 0
    k20_acc = next((r["mean_sr_pred_acc"] for r in sup_rows if r["configuration"]=="suppress_k20"), 0)
    k20_rew = next((r["mean_selector_reward"] for r in sup_rows if r["configuration"]=="suppress_k20"), 0)
    cls_full_rew = next((r["mean_selector_reward"] for r in sup_rows if r["configuration"]=="full"), 0)
    sr_proxy_dependent = k20_rew < cls_full_rew - 0.05 or k20_rew < wt_df["BestSingle"].mean() + 0.03
    print(f"  Proxy suppression: full_sr_acc={full_sr_acc:.3f} k20_acc={k20_acc:.3f} sr_dependent={sr_proxy_dependent}")

    # ── Conservative switcher ──
    cs_best = [r["conservative_switcher"] for r in all_results]
    cs_deltas = np.array([c["delta_vs_BS"] for c in cs_best])
    cs_mean = float(np.mean(cs_deltas))
    cs_pos = int(np.sum(cs_deltas > 0.03))
    cs_beneficial = np.mean([c["beneficial_switch_rate"] for c in cs_best])
    cs_harmful = np.mean([c["harmful_switch_rate"] for c in cs_best])
    cs_rows = []
    for r in all_results:
        cs_rows.append(r["conservative_switcher"])
    cs_df = pd.DataFrame(cs_rows)
    cs_df.to_csv(f"{OUTDIR}/conservative_switcher.csv", index=False)
    cs_pass = cs_mean > 0.03 and cs_pos >= 6 and cs_beneficial > cs_harmful
    print(f"  Conservative switcher: delta={cs_mean:+.4f} positive={cs_pos}/{N_SEEDS} pass={cs_pass}")

    # ── External grid diagnostic ──
    grid_scores = compute_grid_v2_capital_scores(shared.grid_v2,
        make_v2_capitals([PolicyCloneCapital(shared.pc,"PolicyClone"), PrototypeOutcomeCapital(shared.po,"PrototypeOutcome"),
                          AEPCapital(shared.aep,"AEP"), GoalInferenceCapital(grid_size=5,capital_id="GoalInference"),
                          SafeFallbackCapital("SafeFallback")]),
        n_trials=300, seed_offset=777)
    grid_mean = grid_scores.mean(axis=0)
    grid_dict = {cid: float(grid_mean[ci]) for ci, cid in enumerate(CAP_IDS)}
    grid_max = float(np.max(grid_mean))
    grid_oh = float(grid_scores.max(axis=1).mean())
    second_best = float(np.sort(grid_mean)[-2]) if len(grid_mean) >= 2 else 0.0
    grid_ok = grid_max >= 0.20 and (grid_oh - grid_max >= 0.05 or grid_max - second_best >= 0.10)
    grid_rows = [{"capital": cid, "score": grid_dict[cid], "max_capital": grid_max < 0.10} for cid in CAP_IDS]
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(f"{OUTDIR}/external_grid_v2_diagnostic.csv", index=False)
    print(f"  Grid-v2: max={grid_max:.4f} oh={grid_oh:.4f} ok={grid_ok}")

    # ═══════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════
    print(f"\n{'='*60}\nFINAL VERDICT\n{'='*60}")

    if not shared.tax_valid:
        verdict = "IC3_W_CAPITAL_TAXONOMY_FAILS"
    elif not wt_pass:
        verdict = "IC3_W_REPORT_V2_FAILS_WITHIN_TASK"
    elif not grid_ok:
        verdict = "IC3_W_EXTERNAL_GRID_STILL_UNINFORMATIVE"
    elif sr_proxy_dependent:
        verdict = "IC3_W_SUBREGIME_PROXY_DEPENDENT"
    elif wt_pass and not sr_proxy_dependent:
        verdict = "IC3_W_PROXY_ROBUST_WITHIN_TASK_SIGNAL"
    else:
        verdict = "IC3_W_SUBREGIME_PROXY_DEPENDENT"

    if not sr_proxy_dependent and wt_pass and cs_pass and grid_ok:
        verdict = "IC3_W_READY_FOR_DEPLOYABLE_ALLOCATOR"

    # ── Report ──
    report = f"""# IC-3-W: Within-Task Capital Switching Benchmark — Final Report

**Date**: 2026-05-11
**Seeds**: {SEEDS[0]}..{SEEDS[-1]} ({N_SEEDS} seeds)  |  **Task**: Task_W (unified, hidden sub-regimes)

---

## Final Verdict: `{verdict}`

---

## 1. Subregime Capital Taxonomy

| Subregime | Expected | Actual Best | Margin | Match |
|---|---|---|---|---|
"""
    for tr in tax_df_rows:
        report += f"| {tr['subregime']} | {tr['expected_best']} | **{tr['actual_best']}** | {tr['margin']:.4f} | {'YES' if tr['match'] else 'NO'} |\n"

    per_cap_str = "| Capital |"
    for tr in tax_df_rows:
        per_cap_str += f" {tr['subregime']} |"
    per_cap_str += "\n|---|---|---|---|---|\n"
    for cid in CAP_IDS:
        per_cap_str += f"| {cid} |"
        for tr in tax_df_rows:
            per_cap_str += f" {tr[cid]:.4f} |"
        per_cap_str += "\n"

    tax_str = f'YES — match {shared.tax_match}/4, margin_ok {shared.tax_margin_ok}/4' if shared.tax_valid else 'NO — benchmark invalid'

    report += f"""
{per_cap_str}
**Taxonomy valid**: {tax_str}

---

## 2. Within-Task Selector Test

| Seed | BS | Oracle | GB_v2 | Δ | LR_v2 | RF_v2 | MLP_v2 |
|---|---|---|---|---|---|---|---|
"""
    for _, row in wt_df.iterrows():
        report += f"| {int(row['seed'])} | {row['BestSingle']:.4f} | {row['OracleHindsight']:.4f} | {row['GB_v2']:.4f} | {row['GB_v2_delta']:+.4f} | {row['LR_v2']:.4f} | {row['RF_v2']:.4f} | {row['MLP_v2']:.4f} |\n"

    report += f"""
**Mean GB_v2 Δ**: {mean_delta:+.4f}  |  **Positive**: {positive_seeds}/{N_SEEDS}  |  **95% CI**: [{ci_low:+.4f}, {ci_high:+.4f}]
**Within-task pass**: {'YES' if wt_pass else 'NO'} (need Δ>+0.05, ≥8/10, CI≥0)

---

## 3. Per-Subregime Selector Performance

| Subregime | GB_v2 | BestSingle | Δ | Positive |
|---|---|---|---|---|
"""
    for sr in per_sr_rows:
        report += f"| {sr['subregime']} | {sr['mean_GB_v2']:.4f} | {sr['mean_BestSingle']:.4f} | {sr['mean_delta']:+.4f} | {sr['positive_seeds']}/{N_SEEDS} |\n"

    report += f"""
---

## 4. Proxy Suppression (Subregime)

| Config | SR Pred Acc | Selector Reward | Δ vs BS |
|---|---|---|---|
"""
    for sr in sup_rows:
        report += f"| {sr['configuration']} | {sr['mean_sr_pred_acc']:.4f} | {sr['mean_selector_reward']:.4f} | {sr['mean_delta_vs_BS']:+.4f} |\n"

    report += f"""
**Proxy dependence**: {'SUBREGIME_PROXY_DEPENDENT — removing sr-predictive features crashes selector' if sr_proxy_dependent else 'NOT proxy-dependent — signal survives sr feature removal'}

---

## 5. Conservative Switcher

| Metric | Value |
|---|---|
| Test Δ vs BS | {cs_mean:+.4f} |
| Positive seeds (>3%) | {cs_pos}/{N_SEEDS} |
| Beneficial switch rate | {cs_beneficial:.3f} |
| Harmful switch rate | {cs_harmful:.3f} |
| Robust | {'YES' if cs_pass else 'NO'} |

---

## 6. External Grid-v2 Diagnostic

| Capital | Score |
|---|---|
"""
    for cid in CAP_IDS:
        report += f"| {cid} | {grid_dict[cid]:.4f} |\n"

    report += f"""
**Grid-v2**: max={grid_max:.4f} oracle={grid_oh:.4f} status={'OK' if grid_ok else 'STILL UNINFORMATIVE'}

---

## Generated Files (results/ic3_w/)

| # | File |
|---|---|
| 1 | `per_subregime_capital_matrix.csv` |
| 2 | `within_task_selector_test.csv` |
| 3 | `per_subregime_capital_matrix.csv` |
| 4 | `within_task_proxy_suppression.csv` |
| 5 | `conservative_switcher.csv` |
| 6 | `external_grid_v2_diagnostic.csv` |
| 7 | `IC3_W_WITHIN_TASK_CAPITAL_SWITCHING_REPORT.md` |

---

*End of IC-3-W. No second-order intelligence claim made.*
"""

    rpath = f"{OUTDIR}/IC3_W_WITHIN_TASK_CAPITAL_SWITCHING_REPORT.md"
    with open(rpath, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"  REPORT -> {rpath}")
    print(f"  VERDICT -> {verdict}")
    return verdict


# ═════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════

def main():
    t0 = time.time()
    shared = compute_shared_data()

    all_results = []
    for seed in SEEDS:
        try:
            result = run_single_seed(seed, shared)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR on seed {seed}: {e}")
            import traceback; traceback.print_exc()

    if len(all_results) == 0:
        print("FATAL: no seeds completed successfully")
        return

    verdict = aggregate_and_report(all_results, shared)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    return verdict


if __name__ == "__main__":
    main()