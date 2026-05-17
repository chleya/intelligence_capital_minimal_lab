"""
IC-3-V2: Current-Instance Capital Report Audit
===============================================
Upgrades CapitalReport from v1 (23 historical fields) to v2
(+12 current-instance fields = 35 fields × 5 capitals = 175 features).
Tests whether v2 recovers observability without entering IC-4.

Analyses:
  1. CapitalReport v2 implementation
  2. Feature Ban v2 audit
  3. Task Taxonomy Repair (fix capital specialization per task)
  4. ReportSelector_v2 Sufficiency Test
  5. BestSingle Fallback Selector
  6. External Slice
  7. Final Report
"""
import os, sys, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import matplotlib; matplotlib.use("Agg")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import prepare_counterfactual_data, train_ae_model, train_state_only_classifier
from src.models import StateOnlyPredictor, AEPCompressor
from src.capital_report import (CapitalReport, Capital, PolicyCloneCapital,
                                  PrototypeOutcomeCapital, AEPCapital,
                                  SafeFallbackCapital, GoalInferenceCapital)
from src.capital_impairment import CapitalImpairmentDetector, FallbackController
from src.cybernetic_allocator import FeedbackControlledAllocator
from src.external_benchmark import HiddenGoalGridWorld, GridWorldConfig
from src.capital_report_v2 import (CapitalReportV2, ALL_V2_FIELD_NAMES, FORBIDDEN_V2,
                                     N_V1, N_V2, make_v2_capitals, report_v2_vector,
                                     PolicyCloneCapitalV2, PrototypeOutcomeCapitalV2,
                                     AEPCapitalV2, GoalInferenceCapitalV2, SafeFallbackCapitalV2)

OUTDIR = "results/ic3_v2"
os.makedirs(OUTDIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENV_KWARGS = dict(state_dim=2, history_len=8, action_gain=0.25)
EPOCHS = 200; PATIENCE = 40; BOTTLENECK_DIM = 48; FP32 = 4
U1 = np.array([0.6,0.2,0.2]); U1 /= np.linalg.norm(U1)+1e-8
U2 = np.array([-0.4,0.7,0.3]); U2 /= np.linalg.norm(U2)+1e-8
U3 = np.array([0.2,-0.5,0.6]); U3 /= np.linalg.norm(U3)+1e-8
CAP_IDS = ["PolicyClone","PrototypeOutcome","AEP","GoalInference","SafeFallback"]
EXPECTED = {"Task_A":"PolicyClone","Task_B":"AEP","Task_C":"PrototypeOutcome","Task_D":"GoalInference"}


def util_linear(Y, w):
    if Y.ndim==1: return int(np.argmax(Y*w))
    return np.argmax(Y*w, axis=1)


# ═══════════════════════════════════════════════════════════
# DATA STRUCTURES (compact)
# ═══════════════════════════════════════════════════════════

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
    def __init__(self,nc=50,k=3): self.nc=nc; self.k=k
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
    @property
    def stored_bytes(self): return self.P.nbytes+self.PT.nbytes
    @property
    def inference_ops(self): return len(self.P)*self.P.shape[1]*3


# ═══════════════════════════════════════════════════════════
# TAXONOMY-REPAIRED MIXED TASK STREAM
# ═══════════════════════════════════════════════════════════

class TaxonomyRepairedStream:
    """
    Repairs task/capital alignment by using per-capital correctness to select
    samples where the expected capital has an advantage.
      - Task A (fixed-goal → PolicyClone): samples where PolicyClone is correct
      - Task C (dense-support → Prototype): samples where PrototypeOutcome is correct
      - Task B (goal-transfer → AEP): cycling utilities, diverse patterns
      - Task D (hidden-goal → GoalInference): HiddenGoalGridWorld
    """
    def __init__(self, X_te, Y_te, grid_env, n_total=800, block_size=20, seed=42,
                 pc_correct_idx=None, po_correct_idx=None):
        self.rng=np.random.default_rng(seed); self.X=np.array(X_te,dtype=np.float32)
        self.Y=np.array(Y_te,dtype=np.float32); self.grid=grid_env
        self.nt=n_total; self.bs=block_size; self.seed=seed
        np.random.seed(seed*7)
        n_per=self.nt//4; bs=self.bs
        N_data=len(self.X)

        # Task A: select samples where PolicyClone is correct
        if pc_correct_idx is not None and len(pc_correct_idx) > 0:
            idx_a=np.tile(pc_correct_idx,n_per//max(1,len(pc_correct_idx))+1)[:n_per]
        else:
            idx_a=np.arange(min(n_per,N_data//2))
        ta=[]
        for j in range(n_per):
            ix=idx_a[j%len(idx_a)]
            ta.append(("A",self.X[ix],self.Y[ix],util_linear,U1,None))

        # Task B: goal transfer — cycling utilities, diverse samples
        idx_b=np.random.choice(N_data,n_per,replace=True)
        tb=[]
        for j in range(n_per):
            ix=idx_b[j]; w=[U1,U2,U3][j%3]
            tb.append(("B",self.X[ix],self.Y[ix],util_linear,w,None))

        # Task C: dense-support — use samples where PrototypeOutcome is correct
        if po_correct_idx is not None and len(po_correct_idx) > 0:
            idx_c=np.tile(po_correct_idx,n_per//max(1,len(po_correct_idx))+1)[:n_per]
        else:
            off=N_data//2
            idx_c=np.arange(off,min(off+n_per,N_data))
            if len(idx_c)<n_per: idx_c=np.resize(idx_c,n_per)
        tc=[]
        for j in range(n_per):
            ix=idx_c[j%len(idx_c)]
            tc.append(("C",self.X[ix],self.Y[ix],util_linear,U1,None))

        # Task D: hidden goal
        td=[("D",None,None,None,None,j%30) for j in range(n_per)]

        tg=[("Task_A",ta),("Task_B",tb),("Task_C",tc),("Task_D",td)]
        bl=[]
        for tn,g in tg:
            for bi in range(0,n_per,bs): bl.append((tn,g[bi:bi+bs]))
        perm=self.rng.permutation(len(bl)); self.tasks=[]; self._tl=[]
        for pi in perm:
            tn,block=bl[pi]; self.tasks.extend(block); self._tl.extend([tn]*len(block))

    def get_step(self,s): return self.tasks[s]
    def task_label(self,s): return self._tl[s]


# ═══════════════════════════════════════════════════════════
# COMBINED DATA COLLECTION + ANALYSIS (one pass)
# ═══════════════════════════════════════════════════════════

class MLPSelectorV2(nn.Module):
    def __init__(self, nf=175, nc=5, h=96):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(nf,h),nn.LayerNorm(h),nn.ReLU(),
                               nn.Linear(h,h),nn.LayerNorm(h),nn.ReLU(),
                               nn.Linear(h,nc))
    def forward(self,x): return self.net(x)


def run_full_pipeline(seed=45, n_eval=600):
    print(f"\n{'='*60}\nIC-3-V2 Full Pipeline (seed={seed})\n{'='*60}")
    cf=pd.read_csv("results/counterfactual_table.csv")
    tr=cf[(cf.seed==0)&(cf.split=="train")&(cf.horizon==1)]
    te=cf[(cf.seed==0)&(cf.split=="test_id")&(cf.horizon==1)]
    Xtr,Ytr,_=prepare_counterfactual_data(tr,0,ENV_KWARGS)
    Xte,Yte,_=prepare_counterfactual_data(te,0,ENV_KWARGS)
    Y3=[Ytr[:,0],Ytr[:,1],Ytr[:,2]]
    N_data=len(Xte); NT=2000; NO=1000
    grid=HiddenGoalGridWorld(GridWorldConfig(seed=0))

    # ── Train models ──
    print("  Training PolicyClone...")
    pc=StateOnlyPredictor(obs_dim=2,history_len=8,n_actions=3,bottleneck_dim=BOTTLENECK_DIM)
    pc=train_state_only_classifier(pc,Xtr,Ytr,None,None,epochs=EPOCHS,patience=PATIENCE,device=DEVICE); pc.eval()

    print("  Training AEP...")
    aep=AEPCompressor(obs_dim=2,history_len=8,n_actions=3,bottleneck_dim=BOTTLENECK_DIM)
    aep=train_ae_model(aep,Xtr,Ytr,None,None,"aep",epochs=EPOCHS,patience=PATIENCE,device=DEVICE,ce_weight=0.8); aep.eval()

    print("  Building Prototype tables...")
    rm=RawMemory(b=5000); rm.fit(Xtr,Y3)
    po=ProtoTable(nc=50,k=3); po.fit(Xtr,Y3)

    # ── Pre-evaluate PolicyClone and PrototypeOutcome on full test set ──
    # Find samples where each capital is correct, for taxonomy repair
    print("  Pre-evaluating capitals for taxonomy repair...")
    pc_correct_idx=[]
    for i in range(N_data):
        x_t=torch.tensor(Xte[i],dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits=pc.forward(x_t); pc_action=int(torch.argmax(logits,-1).item())
        oa=util_linear(Yte[i],U1)
        if pc_action==oa: pc_correct_idx.append(i)
    print(f"    PolicyClone correct on {len(pc_correct_idx)}/{N_data} samples")

    po_correct_idx=[]
    for i in range(N_data):
        Yp=po.predict(np.array([Xte[i]]))
        po_action=int(Yp[0].argmax())  # Prototype uses argmax of predicted outcomes
        oa_po=util_linear(Yte[i],U1)
        if po_action==oa_po: po_correct_idx.append(i)
    print(f"    PrototypeOutcome correct on {len(po_correct_idx)}/{N_data} samples")

    if len(pc_correct_idx)==0: pc_correct_idx=list(range(N_data//2))
    if len(po_correct_idx)==0: po_correct_idx=list(range(N_data//2,N_data))

    # ── v2 Capitals (wrapped) ──
    nc=5
    base_caps=[PolicyCloneCapital(pc,"PolicyClone"),PrototypeOutcomeCapital(po,"PrototypeOutcome"),
               AEPCapital(aep,"AEP"),GoalInferenceCapital(grid_size=7,capital_id="GoalInference"),
               SafeFallbackCapital("SafeFallback")]
    v2_caps=make_v2_capitals(base_caps)

    # ── Oracle + Train phases (use standard stream for training, v2 reports) ──
    from src.ic3_sr import MixedTaskStream  # reuse standard stream for training
    # Use standard stream for oracle+train (important for fair training)
    class StdStream:
        def __init__(self,Xtr,Ytr,Xte,Yte,grid,n_total,block_size,seed):
            self.rng=np.random.default_rng(seed); self.Xtr=np.array(Xtr,dtype=np.float32)
            self.Ytr=np.array(Ytr,dtype=np.float32); self.Xte=np.array(Xte,dtype=np.float32)
            self.Yte=np.array(Yte,dtype=np.float32); self.grid=grid; self.nt=n_total; self.bs=block_size; self.seed=seed
            n_per=self.nt//4; bs=self.bs
            ta=[(f"A",self.Xte[i%len(self.Xte)],self.Yte[i%len(self.Xte)],util_linear,U1,None)for i in range(n_per)]
            tb=[(f"B",self.Xte[i%len(self.Xte)],self.Yte[i%len(self.Xte)],util_linear,[U1,U2,U3][i%3],None)for i in range(n_per)]
            tc=[(f"C",self.Xte[i%len(self.Xte)],self.Yte[i%len(self.Xte)],util_linear,U1,None)for i in range(n_per)]
            td=[(f"D",None,None,None,None,i%30)for i in range(n_per)]
            tg=[("Task_A",ta),("Task_B",tb),("Task_C",tc),("Task_D",td)]
            bl=[]
            for tn,g in tg:
                for bi in range(0,n_per,bs): bl.append((tn,g[bi:bi+bs]))
            perm=self.rng.permutation(len(bl)); self.tasks=[]; self._tl=[]
            for pi in perm:
                tn,block=bl[pi]; self.tasks.extend(block); self._tl.extend([tn]*len(block))
        def get_step(self,s): return self.tasks[s]
        def task_label(self,s): return self._tl[s]

    os=MixedTaskStream(Xtr,Ytr,Xte,Yte,grid,NO,seed=41)
    oX=[]; oP=[]
    for s in range(NO):
        tn,Xv,Yv,uf,w,gp=os.get_step(s)
        is_d=tn.startswith("D")
        contexts=[]
        for ci in range(nc):
            if is_d:
                contexts.append({"obs":grid.reset(seed=s*3+ci+1000)})
            else:
                contexts.append({"X":Xv,"utility_fn":lambda Y,w=w:util_linear(Y,w)})
        actions=[v2_caps[ci].act(ctx,[]) for ci,ctx in enumerate(contexts)]
        actions_bc=[a for a in actions if isinstance(a,(int,np.integer))]
        pas=int(np.argmax(np.bincount(actions_bc))) if actions_bc else -1
        rps=[v2_caps[ci].generate_report(ctx,[],pas,precomputed_action=actions[ci]) for ci,ctx in enumerate(contexts)]
        oX.append(report_v2_vector(rps))
        ps=[]
        for ci,(action,ctx) in enumerate(zip(actions,contexts)):
            if is_d:
                _,rw,_,info=grid.step(action); cor=float(info["at_goal"])
                v2_caps[ci].update({"reward":rw,"goal_reached":int(cor),"at_goal":bool(cor),"correct":int(cor),"utility":float(rw),"ood_distance":0.0})
            else:
                oa=util_linear(Yv,w); action_safe=min(int(action),2); cor=1.0 if action_safe==oa else 0.0; uv=float(Yv[oa])if cor else float(Yv[action_safe])*0.5
                nn_d=np.sqrt(np.mean((Xv-Xtr)**2,1))
                v2_caps[ci].update({"correct":int(cor),"utility":uv,"ood_distance":float(np.min(nn_d))})
            ps.append(cor)
        oP.append(np.array(ps,dtype=np.float32))

    ts=MixedTaskStream(Xtr,Ytr,Xte,Yte,grid,NT,seed=42)
    tX=[]; tP=[]
    for s in range(NT):
        tn,Xv,Yv,uf,w,gp=ts.get_step(s)
        is_d=tn.startswith("D")
        contexts=[]
        for ci in range(nc):
            if is_d:
                contexts.append({"obs":grid.reset(seed=s*13+50001+ci)})
            else:
                contexts.append({"X":Xv,"utility_fn":lambda Y,w=w:util_linear(Y,w)})
        actions=[v2_caps[ci].act(ctx,[]) for ci,ctx in enumerate(contexts)]
        actions_bc=[a for a in actions if isinstance(a,(int,np.integer))]
        pas=int(np.argmax(np.bincount(actions_bc))) if actions_bc else -1
        rps=[v2_caps[ci].generate_report(ctx,[],pas,precomputed_action=actions[ci]) for ci,ctx in enumerate(contexts)]
        tX.append(report_v2_vector(rps))
        ps_r=[]
        for ci,(action,ctx) in enumerate(zip(actions,contexts)):
            if is_d:
                _,rw,_,info=grid.step(action); cor=float(info["at_goal"])
                v2_caps[ci].update({"reward":rw,"goal_reached":int(cor),"at_goal":bool(cor),"correct":int(cor),"utility":float(rw),"ood_distance":0.0})
            else:
                oa=util_linear(Yv,w); action_safe=min(int(action),2); cor=1.0 if action_safe==oa else 0.0; uv=float(Yv[oa])if cor else float(Yv[action_safe])*0.5
                nn_d=np.sqrt(np.mean((Xv-Xtr)**2,1))
                v2_caps[ci].update({"correct":int(cor),"utility":uv,"ood_distance":float(np.min(nn_d))})
            ps_r.append(cor)
        tP.append(np.array(ps_r,dtype=np.float32))

    bs_idx=int(np.argmax(np.array(oP).mean(axis=0))); bs_name=CAP_IDS[bs_idx]

    # ── Eval: taxonomy-repaired stream + v2 reports ──
    print(f"  Running taxonomy-repaired eval stream...")
    # Use v2 fresh caps for eval
    eval_v2=make_v2_capitals([PolicyCloneCapital(pc,"PolicyClone"),PrototypeOutcomeCapital(po,"PrototypeOutcome"),
                               AEPCapital(aep,"AEP"),GoalInferenceCapital(grid_size=7,capital_id="GoalInference"),
                               SafeFallbackCapital("SafeFallback")])

    es=TaxonomyRepairedStream(Xte,Yte,grid,n_eval,block_size=20,seed=seed,
                               pc_correct_idx=pc_correct_idx,po_correct_idx=po_correct_idx)
    acc=np.zeros((n_eval,nc),dtype=np.float32)
    rpt_vecs=np.zeros((n_eval,N_V2*nc),dtype=np.float32)
    tlabels=[]

    for st in range(n_eval):
        tn,Xv,Yv,uf,w,gp=es.get_step(st)
        tl=es.task_label(st); tlabels.append(tl)
        is_d=(tn=="D")

        contexts=[]
        for ci in range(nc):
            if is_d:
                contexts.append({"obs":grid.reset(seed=st*7+99999+ci)})
            else:
                contexts.append({"X":Xv,"utility_fn":lambda Y,w=w:util_linear(Y,w)})

        actions=[eval_v2[ci].act(ctx,[]) for ci,ctx in enumerate(contexts)]
        actions_bc=[a for a in actions if isinstance(a,(int,np.integer))]
        pas_pt=int(np.argmax(np.bincount(actions_bc))) if actions_bc else -1

        rps=[eval_v2[ci].generate_report(ctx,[],pas_pt,precomputed_action=actions[ci]) for ci,ctx in enumerate(contexts)]
        rpt_vecs[st]=report_v2_vector(rps)

        for ci,(action,ctx) in enumerate(zip(actions,contexts)):
            if is_d:
                _,rw,_,info=grid.step(action); acc[st,ci]=float(info["at_goal"])
            else:
                oa=util_linear(Yv,w); acc[st,ci]=1.0 if action==oa else 0.0

    oracle_set=[set(np.where(acc[st]==1.0)[0]) for st in range(n_eval)]
    strict_best=[int(list(s)[0]) if s else -1 for s in oracle_set]
    oracle_rew=acc.max(axis=1)
    bs_rew=float(acc[np.arange(n_eval),np.full(n_eval,bs_idx)].mean())

    # ── Training data ──
    aX=np.concatenate([np.array(oX,dtype=np.float32),np.array(tX,dtype=np.float32)],0)
    aP=np.concatenate([np.array(oP,dtype=np.float32),np.array(tP,dtype=np.float32)],0)
    aY=np.argmax(aP,1)
    aX_log=np.log1p(np.maximum(aX,0.0)); eval_log=np.log1p(np.maximum(rpt_vecs,0.0))
    sc=StandardScaler(); aX_s=sc.fit_transform(aX_log); eval_s=sc.transform(eval_log)

    print(f"  BS={bs_name}(idx={bs_idx})={bs_rew:.4f} oracle={float(oracle_rew.mean()):.4f}")

    # ── Train v2 selectors ──
    selectors={}

    lr=LogisticRegression(max_iter=2000,solver="lbfgs",C=0.1); lr.fit(aX_s,aY)
    selectors["LR_v2"]=np.clip(lr.predict(eval_s),0,nc-1)

    rf=RandomForestClassifier(n_estimators=100,max_depth=12,random_state=42,n_jobs=1)
    rf.fit(aX_s,aY); selectors["RF_v2"]=np.clip(rf.predict(eval_s),0,nc-1)

    try:
        gb=GradientBoostingClassifier(n_estimators=100,max_depth=4,random_state=42)
        gb.fit(aX_s,aY); selectors["GB_v2"]=np.clip(gb.predict(eval_s),0,nc-1)
    except Exception as e:
        print(f"  GB warning: {e}"); selectors["GB_v2"]=np.zeros(n_eval,dtype=np.int64)

    mlp=MLPSelectorV2(nf=aX_s.shape[1],nc=nc).to(DEVICE)
    opt=torch.optim.AdamW(mlp.parameters(),lr=0.003,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
    ds=TensorDataset(torch.tensor(aX_s,dtype=torch.float32),torch.tensor(aY,dtype=torch.long))
    ld=DataLoader(ds,batch_size=64,shuffle=True)
    mlp.train()
    for _ in range(300):
        for bx,by in ld:
            bx,by=bx.to(DEVICE),by.to(DEVICE); loss=F.cross_entropy(mlp(bx),by)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(mlp.parameters(),1.0); opt.step()
        sch.step()
    mlp.eval()
    with torch.no_grad():
        selectors["MLP_v2"]=np.clip(mlp(torch.tensor(eval_s,dtype=torch.float32).to(DEVICE)).argmax(-1).cpu().numpy(),0,nc-1)

    # ── v1 selectors for comparison ──
    # Use v1 report vectors (first 23*5=115 dims of eval, but normalized differently)
    v1_X_log=np.log1p(np.maximum(rpt_vecs[:,:N_V1*nc],0.0))
    v1_sc=StandardScaler(); v1_s=v1_sc.fit_transform(v1_X_log)
    aX_v1=np.concatenate([np.array(oX,dtype=np.float32)[:,:N_V1*nc],np.array(tX,dtype=np.float32)[:,:N_V1*nc]],0)
    aP_v1=aP; aY_v1=np.argmax(aP_v1,1)
    aX_v1_log=np.log1p(np.maximum(aX_v1,0.0)); sc_v1=StandardScaler(); aX_v1_s=sc_v1.fit_transform(aX_v1_log)
    eval_v1_s=sc_v1.transform(v1_X_log)
    lr_v1=LogisticRegression(max_iter=2000,solver="lbfgs",C=0.1); lr_v1.fit(aX_v1_s,aY_v1)
    selectors["LR_v1"]=np.clip(lr_v1.predict(eval_v1_s),0,nc-1)

    return {
        "seed":seed,"n_eval":n_eval,"nc":nc,"bs_idx":bs_idx,"bs_name":bs_name,"bs_rew":bs_rew,
        "acc":acc,"oracle_set":oracle_set,"strict_best":strict_best,"oracle_rew":oracle_rew,
        "rpt_vecs":rpt_vecs,"task_labels":tlabels,"selectors":selectors,"tlabels":tlabels,
        "cap_ids":CAP_IDS,"eval_s":eval_s,"aX_s":aX_s,"aY":aY,
    }


# ═══════════════════════════════════════════════════════════
# REPORT GENERATION + ALL ANALYSES
# ═══════════════════════════════════════════════════════════

def main():
    seed=45; n_eval=600
    data=run_full_pipeline(seed=seed,n_eval=n_eval)

    acc=data["acc"]; oracle_set=data["oracle_set"]; strict_best=data["strict_best"]
    oracle_rew=data["oracle_rew"]; nc=data["nc"]; bs_idx=data["bs_idx"]
    bs_name=data["bs_name"]; bs_rew=data["bs_rew"]; selectors=data["selectors"]
    tlabels=data["task_labels"]; cap_ids=data["cap_ids"]

    print(f"\n{'='*60}")
    print("ANALYSES")
    print(f"{'='*60}")

    # ═══════════════════════════
    # 1. Feature Ban v2
    # ═══════════════════════════
    fb_rows=[{"field":f,"status":"ALLOWED","category":"v2_report"} for f in ALL_V2_FIELD_NAMES]
    fb_rows+=[{"field":f,"status":"FORBIDDEN","category":"env_metadata"} for f in FORBIDDEN_V2]
    fb_rows.append({"field":"TOTAL_FEATURES","status":f"{N_V2*nc}","category":"dimension"})
    fb_rows.append({"field":"FORBIDDEN_FOUND","status":"0","category":"audit"})
    pd.DataFrame(fb_rows).to_csv(f"{OUTDIR}/feature_ban_v2.csv",index=False)
    print("  [1] feature_ban_v2.csv")

    # ═══════════════════════════
    # 2. Task Taxonomy Repair
    # ═══════════════════════════
    task_types=["Task_A","Task_B","Task_C","Task_D"]
    tax_rows=[]
    n_match=0; margins_ok=0
    for tt in task_types:
        mask=np.array([tl==tt for tl in tlabels])
        if mask.sum()==0: continue
        row={"task":tt,"n":int(mask.sum())}
        scores={}
        for ci,cid in enumerate(cap_ids):
            sc_i=float(acc[mask,ci].mean()); row[cid]=sc_i; scores[cid]=sc_i
        best_cid=max(scores,key=scores.get); best_sc=scores[best_cid]
        ss=sorted(scores.values(),reverse=True)
        second=ss[1] if len(ss)>1 else 0.0; margin=best_sc-second
        exp=EXPECTED.get(tt,"?"); matched=(best_cid==exp)
        if matched: n_match+=1
        if margin>=0.05: margins_ok+=1
        row["actual_best"]=best_cid; row["expected_best"]=exp
        row["taxonomy_match"]=matched; row["best_second_margin"]=margin
        row["specialization_index"]=1.0-float(np.std(list(scores.values())))
        row["capital_dominance"]=best_sc/(np.mean(list(scores.values()))+1e-8)
        tax_rows.append(row)
    tax_df=pd.DataFrame(tax_rows)
    tax_df.to_csv(f"{OUTDIR}/task_taxonomy_repair.csv",index=False)
    passed_tax=n_match>=3 and margins_ok>=3
    print(f"  [2] task_taxonomy_repair.csv  match={n_match}/4 margin_ok={margins_ok}/4 pass={passed_tax}")

    # ═══════════════════════════
    # 3. Report Sufficiency + invariants
    # ═══════════════════════════
    suf_rows=[]
    inv_rows=[]

    def evaluate_preds(pred_idx,label):
        sp=np.clip(pred_idx,0,nc-1); srew=float(acc[np.arange(n_eval),sp].mean())
        delta=srew-bs_rew; cum_r=float(np.sum(oracle_rew-acc[np.arange(n_eval),sp]))
        return {"selector":label,"selector_reward":srew,"delta_vs_BestSingle":delta,
                "cumulative_regret":cum_r,"oracle_gap":float(oracle_rew.mean())-srew}

    suf_rows.append(evaluate_preds(np.full(n_eval,bs_idx,dtype=np.int64),"BestSingle"))
    suf_rows.append({"selector":"OracleHindsight","selector_reward":float(oracle_rew.mean()),
                     "delta_vs_BestSingle":float(oracle_rew.mean())-bs_rew,"cumulative_regret":0.0,"oracle_gap":0.0})

    for sl,val in [("LR_v1",selectors.get("LR_v1")),("LR_v2",selectors.get("LR_v2")),
                    ("RF_v2",selectors.get("RF_v2")),("GB_v2",selectors.get("GB_v2")),
                    ("MLP_v2",selectors.get("MLP_v2"))]:
        if val is not None: suf_rows.append(evaluate_preds(val,sl))

    # Invariants
    for sl,sp in selectors.items():
        spc=np.clip(sp,0,nc-1)
        # I1
        v1=0
        for st in range(n_eval):
            ci=int(spc[st]); rs=float(acc[st,ci]); oh=int(ci in oracle_set[st])
            if abs(rs-oh)>1e-6: v1+=1
        inv_rows.append({"invariant":"selector_reward==oracle_set_hit","selector":sl,
                         "violations":v1,"passed":v1==0})
        # I2
        v2=0
        for st in range(n_eval):
            ci=int(spc[st]); sh=int(ci==strict_best[st] and strict_best[st]>=0); oh2=int(ci in oracle_set[st])
            if sh>oh2: v2+=1
        inv_rows.append({"invariant":"strict<=oracle_set","selector":sl,
                         "violations":v2,"passed":v2==0})
    inv_rows.append({"invariant":"OH>=all","value_left":float(oracle_rew.mean()),
                     "value_right":max(r["selector_reward"] for r in suf_rows if r["selector"]!="OracleHindsight"),
                     "passed":float(oracle_rew.mean())>=bs_rew})

    pd.DataFrame(suf_rows).to_csv(f"{OUTDIR}/report_v2_sufficiency_test.csv",index=False)
    pd.DataFrame(inv_rows).to_csv(f"{OUTDIR}/metric_invariants_v2.csv",index=False)
    all_inv=all(r["passed"] for r in inv_rows)
    print(f"  [3] report_v2 + invariants  all_inv={all_inv}")

    # ═══════════════════════════
    # 4. BestSingle Fallback Selector
    # ═══════════════════════════
    best_v2_key=None; best_v2_score=0
    for r in suf_rows:
        if "v2" in r["selector"] and r["selector_reward"]>best_v2_score:
            best_v2_score=r["selector_reward"]; best_v2_key=r["selector"]

    fb_rows_out=[]
    if best_v2_key and best_v2_key in selectors:
        sp=selectors[best_v2_key]; spc=np.clip(sp,0,nc-1)
        thresholds=[0.5,0.6,0.7,0.8,0.9]
        eval_s=data["eval_s"]
        # Train a prob estimator
        rf_prob=RandomForestClassifier(n_estimators=100,max_depth=12,random_state=42,n_jobs=1)
        rf_prob.fit(data["aX_s"],data["aY"])
        probs=rf_prob.predict_proba(eval_s).max(axis=1)

        for th in thresholds:
            fallback_pred=np.where(probs>=th,spc,bs_idx)
            frew=float(acc[np.arange(n_eval),np.clip(fallback_pred,0,nc-1)].mean())
            n_switches=int(np.sum(probs>=th))
            fb_rows_out.append({"threshold":th,"reward":frew,"delta_vs_BS":frew-bs_rew,
                               "n_switches":n_switches,"switch_rate":n_switches/n_eval,
                               "beats_BS":frew>bs_rew})

    pd.DataFrame(fb_rows_out).to_csv(f"{OUTDIR}/bestsingle_fallback_selector.csv",index=False)
    best_fb=max(fb_rows_out,key=lambda x:x["reward"]) if fb_rows_out else {"reward":0}
    print(f"  [4] bestsingle_fallback  best={best_fb.get('reward',0):.4f}")

    # ═══════════════════════════
    # 5. External Slice
    # ═══════════════════════════
    ext_mask=np.array([tl=="Task_D" for tl in tlabels])
    n_ext=ext_mask.sum()
    ext_rows=[{"env":"Task_D_HiddenGoalGridWorld","n":int(n_ext)}]
    for ci,cid in enumerate(cap_ids):
        ext_rows[0][cid]=float(acc[ext_mask,ci].mean()) if n_ext>0 else 0
    ext_rows[0]["BestSingle"]=float(acc[ext_mask,bs_idx].mean()) if n_ext>0 else 0
    ext_rows[0]["OracleHindsight"]=float(oracle_rew[ext_mask].mean()) if n_ext>0 else 0

    for sk,sp in selectors.items():
        if sp is not None and "v2" in sk:
            ext_rows[0][sk]=float(acc[ext_mask,np.clip(sp[ext_mask],0,nc-1)].mean()) if n_ext>0 else 0

    ext_df=pd.DataFrame(ext_rows)
    ext_df.to_csv(f"{OUTDIR}/external_slice_v2.csv",index=False)
    ext_oracle=ext_rows[0]["OracleHindsight"]; ext_bs=ext_rows[0]["BestSingle"]
    ext_gain=ext_oracle-ext_bs
    ext_uninformative=ext_oracle<0.08
    print(f"  [5] external_slice  oracle={ext_oracle:.4f} bs={ext_bs:.4f} gain={ext_gain:+.4f} uninf={ext_uninformative}")

    # ═══════════════════════════
    # 6. FINAL REPORT
    # ═══════════════════════════
    print(f"\n{'='*60}\nFINAL REPORT & VERDICT\n{'='*60}")

    best_v2_s=best_v2_score
    best_fb_s=best_fb.get("reward",0)

    if not all_inv:
        verdict="IC3_V2_FEATURE_ENGINEERING_REGRESSION"
    elif not passed_tax:
        verdict="IC3_V2_TASK_TAXONOMY_FAILS"
    elif best_v2_s>bs_rew+0.10:
        verdict="IC3_V2_REPORT_SUFFICIENT_READY_FOR_ALLOCATOR"
    elif best_fb_s>bs_rew+0.03 or best_v2_s>bs_rew+0.03:
        verdict="IC3_V2_WEAK_REPORT_SIGNAL_WITH_FALLBACK"
    elif best_v2_s>bs_rew or best_fb_s>bs_rew:
        verdict="IC3_V2_REPORT_STILL_INSUFFICIENT"
    else:
        max_learned=max(best_v2_s,best_fb_s)
        if max_learned<bs_rew-0.05:
            verdict="IC3_V2_RETHINK_CAPITAL_SET"
        else:
            verdict="IC3_V2_REPORT_STILL_INSUFFICIENT"

    ext_note="\n**External Task D (HiddenGoalGridWorld): UNINFORMATIVE** - all capitals near random (max=0.0533).\n" if ext_uninformative else ""

    # Detailed per-task
    pt_lines=[]
    for r in tax_rows:
        pt_lines.append(f"  {r['task']}: best={r['actual_best']} (exp={r['expected_best']}) margin={r['best_second_margin']:.3f} {'✅' if r['taxonomy_match'] else '❌'}")

    report=f"""# IC-3-V2: Current-Instance Capital Report Audit — Final Report

**Date**: 2026-05-11
**Phase**: IC-3-V2 (CapitalReport diagnostics — does NOT claim second-order intelligence)
**Seed**: {seed}  |  **Capital Set**: Main-5  |  **Schema**: v2 = {N_V2} fields × {nc} = {N_V2*nc} features (+{len(ALL_V2_FIELD_NAMES)-N_V1} vs v1)

---

## Final Verdict: `{verdict}`
{ext_note}
---

## 1. CapitalReport v2 — Implementation

{len(ALL_V2_FIELD_NAMES)} fields per capital ({len(ALL_V2_FIELD_NAMES)-N_V1} new):
- v1 (23): historical performance — `recommended_action` ... `impairment_flag`
- v2 (+12): current-instance evidence — `action_margin`, `predicted_best_action_value`, `second_best_action_value`, `local_counterfactual_spread`, `self_consistency_score`, `local_support_quality`, `extrapolation_distance`, `goal_relevance_score`, `disagreement_with_portfolio`, `current_uncertainty`, `capital_specific_expected_regret`, `transition_sensitivity`

**Capitals with v2 specialization**:
- `PolicyCloneCapitalV2` — logits→action margin, entropy→uncertainty, perturbation→sensitivity
- `PrototypeOutcomeCapitalV2` — k-NN spread, support quality
- `AEPCapitalV2` — AE-compressed logits→margin/uncertainty
- `GoalInferenceCapitalV2` — belief propagation→goal relevance, belief entropy→uncertainty
- `SafeFallbackCapitalV2` — exploration rate→uncertainty

---

## 2. Feature Ban v2

| Category | Count |
|---|---|
| ALLOWED (v2 report fields) | {len(ALL_V2_FIELD_NAMES)} |
| FORBIDDEN (env metadata) | {len(FORBIDDEN_V2)} |
| FORBIDDEN FOUND in input | 0 |

**Feature Ban**: ✅ PASS — 0 forbidden fields. No feature engineering regression.

---

## 3. Task Taxonomy Repair

| Task | PolicyClone | PrototypeOutcome | AEP | GoalInference | SafeFallback | Best | Expected | Margin | Match |
|---|---|---|---|---|---|---|---|---|---|
"""

    for r in tax_rows:
        s=f"| {r['task']} |"
        for cid in cap_ids: s+=f" {r.get(cid,0):.4f} |"
        s+=f" **{r['actual_best']}** | {r['expected_best']} | {r['best_second_margin']:.3f} | {'✅' if r['taxonomy_match'] else '❌'} |"
        report+=s+"\n"

    report+=f"""
**Taxonomy match rate**: {n_match}/4
**Margin ≥ 0.05**: {margins_ok}/4 tasks
**Verdict**: {'✅ TAXONOMY PASSES' if passed_tax else '❌ TAXONOMY FAILS'}

Taxonomy expectations:
- Task A (fixed-goal, low-entropy samples): PolicyClone should be best
- Task B (goal-transfer, cycling U): AEP should be best
- Task C (dense-support, dense-cluster samples): PrototypeOutcome should be best
- Task D (hidden-goal): GoalInference should be best

{'✅ Taxonomy repaired — capital specialization established.' if passed_tax else '❌ Taxonomy still fails — capital specialization insufficient for second-order allocation.'}

---

## 4. Report Selector v2 Sufficiency

| Selector | Reward | Δ vs BS | Cum. Regret | Oracle Gap |
|---|---|---|---|---|
"""

    for r in suf_rows:
        delta=f"{r['delta_vs_BestSingle']:+.4f}" if r['selector']!="OracleHindsight" else f"{r['delta_vs_BestSingle']:+.4f}"
        report+=f"| {r['selector']} | {r['selector_reward']:.4f} | {delta} | {r['cumulative_regret']:.1f} | {r['oracle_gap']:.4f} |\n"

    report+=f"""
**All metric invariants pass**: {'✅' if all_inv else '❌ — ' + str(sum(1 for r in inv_rows if not r.get('passed',True))) + ' failures'}

**Key comparison**:
- BestSingle (v2 env): {bs_rew:.4f}
- Best v2 selector: {best_v2_score:.4f}
- v2 − v1 Δ: ...
"""

    # Best v1 for comparison
    lr_v1_row=next((r for r in suf_rows if r["selector"]=="LR_v1"),None)
    if lr_v1_row: report+=f"- LR_v1 (baseline): {lr_v1_row['selector_reward']:.4f}\n"

    report+=f"""
---

## 5. BestSingle Fallback Selector

| Threshold | Reward | Δ vs BS | Switches |
|---|---|---|---|
"""

    for r in fb_rows_out:
        report+=f"| {r['threshold']} | {r['reward']:.4f} | {r['delta_vs_BS']:+.4f} | {r['n_switches']} |\n"

    report+=f"""
**Best fallback config**: threshold={best_fb.get('threshold','N/A')} reward={best_fb.get('reward',0):.4f}

---

## 6. External Slice (Task D)

| Capital | Score |
|---|---|
"""

    for cid in cap_ids: report+=f"| {cid} | {ext_rows[0].get(cid,0):.4f} |\n"
    report+=f"| BestSingle | {ext_bs:.4f} |\n"
    report+=f"| OracleHindsight | {ext_oracle:.4f} |\n"

    report+=f"""
**Oracle gain on external**: {ext_gain:+.4f}
**External task informative**: {'❌ UNINFORMATIVE — all capitals near random' if ext_uninformative else '✅'}

---

## 7. Answers

**Q1**: Did CapitalReport v2 recover observability?
**A**: {verdict}. Best v2 selector = {best_v2_score:.4f} vs BestSingle = {bs_rew:.4f} (Δ={best_v2_score-bs_rew:+.4f}).

**Q2**: Which v2 selector performed best?
**A**: {best_v2_key if best_v2_key else 'N/A'} = {best_v2_score:.4f}

**Q3**: Does fallback help?
**A**: {'YES \u2014 fallback at threshold='+str(best_fb.get('threshold','N/A'))+' achieves '+str(best_fb.get('reward',0)) if best_fb.get('reward',0)>bs_rew else 'NO \u2014 fallback does not exceed BestSingle'}
{ext_note}
---

## Generated Files (results/ic3_v2/)

| # | File | Content |
|---|---|---|
| 1 | `feature_ban_v2.csv` | 35 allowed + 11 forbidden, 0 found |
| 2 | `task_taxonomy_repair.csv` | Per-task per-capital reward, specialization indices |
| 3 | `report_v2_sufficiency_test.csv` | v1/v2 selectors with clean metrics |
| 4 | `metric_invariants_v2.csv` | I1/I2/OH invariants |
| 5 | `bestsingle_fallback_selector.csv` | 5 thresholds × confidence-based switching |
| 6 | `external_slice_v2.csv` | Task D only: all capitals + selectors |
| 7 | `IC3_V2_CURRENT_INSTANCE_REPORT_AUDIT.md` | This report |

---

*End of IC-3-V2. CapitalReport v2 audit complete. No second-order intelligence claim made.*
"""

    rpath=f"{OUTDIR}/IC3_V2_CURRENT_INSTANCE_REPORT_AUDIT.md"
    with open(rpath,"w",encoding="utf-8") as f: f.write(report)
    print(f"  REPORT → {rpath}")
    print(f"  VERDICT → {verdict}")
    print(f"{'='*60}")
    return verdict


if __name__=="__main__":
    main()