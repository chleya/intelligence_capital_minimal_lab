"""
IC-3-W: WithinTaskMixedRegimeEnv
==================================
Constructs a unified Task_W where all samples have the SAME task_id,
but internally contain hidden sub-regimes with different best capitals:

  W1: PolicyClone best   (synthetic samples where PC correct, others wrong)
  W2: AEPCapital best    (synthetic samples where AEP correct, others wrong)
  W3: PrototypeOutcome best (synthetic samples where PO correct, others wrong)
  W4: GoalInference best (HiddenGoalGridWorld-v2)

Selector sees only CapitalReport features, never task_id or subregime_id.
Subregime_id is for offline audit ONLY.
"""
import numpy as np


class WithinTaskMixedRegimeStream:
    """
    Single-task evaluation stream with hidden sub-regimes.

    Parameters
    ----------
    pc_correct_only : list of int
        Sample indices where PolicyClone is correct AND AEP+PO are wrong
    aep_correct_only : list of int
        Sample indices where AEP is correct AND PC+PO are wrong
    po_correct_only : list of int
        Sample indices where PrototypeOutcome is correct AND PC+AEP are wrong
    Xte, Yte : ndarray
        Full test set observations and labels
    grid_env : HiddenGoalGridWorldV2
        Improved grid environment for W4
    n_per_subregime : int
        Number of evaluation steps per sub-regime (default 150)
    block_size : int
        Block size for interleaving (default 12)
    seed : int
        Random seed for shuffling
    """
    def __init__(self, pc_correct_only, aep_correct_only, po_correct_only,
                 Xte, Yte, grid_env, n_per_subregime=150, block_size=12, seed=42):
        self.X = np.array(Xte, dtype=np.float32)
        self.Y = np.array(Yte, dtype=np.float32)
        self.grid = grid_env
        self.np = n_per_subregime
        self.bs = block_size
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed * 13)

        ncpp = len(pc_correct_only) if pc_correct_only else 0
        naep = len(aep_correct_only) if aep_correct_only else 0
        npo = len(po_correct_only) if po_correct_only else 0

        # W1: PolicyClone best — tile PC-only samples
        if ncpp > 0:
            tiled_pc = np.tile(pc_correct_only, n_per_subregime // max(1, ncpp) + 1)[:n_per_subregime]
            tw1 = [("W", self.X[ix % ncpp], self.Y[ix % ncpp], "W1") for ix in tiled_pc]
        else:
            tw1 = []

        # W2: AEP best — tile AEP-only samples
        if naep > 0:
            tiled_aep = np.tile(aep_correct_only, n_per_subregime // max(1, naep) + 1)[:n_per_subregime]
            tw2 = [("W", self.X[ix % naep], self.Y[ix % naep], "W2") for ix in tiled_aep]
        else:
            tw2 = []

        # W3: PrototypeOutcome best — tile PO-only samples
        if npo > 0:
            tiled_po = np.tile(po_correct_only, n_per_subregime // max(1, npo) + 1)[:n_per_subregime]
            tw3 = [("W", self.X[ix % npo], self.Y[ix % npo], "W3") for ix in tiled_po]
        else:
            tw3 = []

        # W4: GoalInference / HiddenGoalGridWorld-v2
        tw4 = [("W", None, None, "W4") for _ in range(n_per_subregime)]

        # Combine all sub-regimes
        all_tasks_raw = tw1 + tw2 + tw3 + tw4
        n_total = len(all_tasks_raw)

        # Split into blocks and shuffle blocks
        blocks = []
        for bi in range(0, n_total, block_size):
            blocks.append(all_tasks_raw[bi:bi + block_size])

        perm = self.rng.permutation(len(blocks))
        self.tasks = []
        self._subregime_ids = []
        for pi in perm:
            for step in blocks[pi]:
                self.tasks.append(step)
                self._subregime_ids.append(step[3])  # hidden subregime id

        self.nt = len(self.tasks)

    def get_step(self, s):
        """Returns (task_label, X, Y, subregime_id). X/Y can be None for grid tasks."""
        if s >= self.nt:
            return ("W", None, None, "W0")
        tn, Xv, Yv, sr_id = self.tasks[s]
        return (tn, Xv, Yv, sr_id)

    def task_label(self, s):
        """Always returns 'Task_W' for all steps."""
        return "Task_W"

    def subregime_id(self, s):
        """Returns hidden subregime id (W1/W2/W3/W4). For audit ONLY."""
        if s >= self.nt:
            return "W0"
        return self._subregime_ids[s]


def build_per_capital_exclusive_indices(Xte, Yte, pc, aep, po, device="cpu",
                                          U1=None, util_fn=None):
    """
    Pre-compute per-sample correctness for all 5 capitals and find
    samples where EXACTLY ONE capital is correct (exclusive correct).

    Returns:
        pc_only: indices where PolicyClone is correct, AEP and PO are wrong
        aep_only: indices where AEP is correct, PC and PO are wrong
        po_only: indices where PrototypeOutcome is correct, PC and AEP are wrong
    """
    if U1 is None:
        U1 = np.array([0.6, 0.2, 0.2], dtype=np.float32)
        U1 /= np.linalg.norm(U1) + 1e-8
    if util_fn is None:
        def util_fn(Y, w): return int(np.argmax(Y * w)) if Y.ndim == 1 else np.argmax(Y * w, axis=1)

    import torch
    N = len(Xte)
    pc_correct = np.zeros(N, dtype=bool)
    aep_correct = np.zeros(N, dtype=bool)
    po_correct = np.zeros(N, dtype=bool)

    pc_mod = pc
    pc_mod.eval()

    for i in range(N):
        x_t = torch.tensor(Xte[i], dtype=torch.float32).unsqueeze(0).to(device)
        oa = util_fn(Yte[i], U1)

        # PolicyClone
        with torch.no_grad():
            logits = pc_mod.forward(x_t)
            pc_action = int(torch.argmax(logits, -1).item())
        pc_correct[i] = (pc_action == oa)

        # AEP
        with torch.no_grad():
            aep_values = aep.predict_all_actions(x_t)
            aep_action = int(torch.argmax(aep_values, -1).item())
        aep_correct[i] = (aep_action == oa)

        # PrototypeOutcome
        Yp = po.predict(np.array([Xte[i]]))
        po_action = int(Yp[0].argmax())
        po_correct[i] = (po_action == oa)

    # Exclusive correctness
    pc_only = [i for i in range(N) if pc_correct[i] and not aep_correct[i] and not po_correct[i]]
    aep_only = [i for i in range(N) if aep_correct[i] and not pc_correct[i] and not po_correct[i]]
    po_only = [i for i in range(N) if po_correct[i] and not pc_correct[i] and not aep_correct[i]]

    return pc_only, aep_only, po_only


def validate_subregime_taxonomy(pc_only, aep_only, po_only, Xte, Yte, pc, aep, po, grid_env,
                                  device="cpu", U1=None, util_fn=None):
    """
    Validate that each sub-regime has a clearly different best capital.
    Computes per-capital reward on each sub-regime.
    """
    if U1 is None:
        U1 = np.array([0.6, 0.2, 0.2], dtype=np.float32)
        U1 /= np.linalg.norm(U1) + 1e-8
    if util_fn is None:
        def util_fn(Y, w): return int(np.argmax(Y * w)) if Y.ndim == 1 else np.argmax(Y * w, axis=1)

    import torch
    NC = 5
    CAP_IDS = ["PolicyClone", "PrototypeOutcome", "AEP", "GoalInference", "SafeFallback"]

    def eval_on_indices(indices, name):
        if len(indices) == 0:
            return {cid: 0.0 for cid in CAP_IDS}, None, 0.0
        n = len(indices)
        acc = np.zeros((n, NC), dtype=np.float32)
        for j, idx in enumerate(indices):
            Xv = Xte[idx]; Yv = Yte[idx]
            x_t = torch.tensor(Xv, dtype=torch.float32).unsqueeze(0).to(device)
            oa = util_fn(Yv, U1)

            with torch.no_grad():
                pc_logits = pc.forward(x_t)
                acc[j, 0] = 1.0 if int(torch.argmax(pc_logits, -1).item()) == oa else 0.0
                aep_values = aep.predict_all_actions(x_t)
                acc[j, 2] = 1.0 if int(torch.argmax(aep_values, -1).item()) == oa else 0.0

            Yp = po.predict(np.array([Xv]))
            acc[j, 1] = 1.0 if int(Yp[0].argmax()) == oa else 0.0

        scores = {cid: float(acc[:, ci].mean()) for ci, cid in enumerate(CAP_IDS)}
        best_ci = int(np.argmax([scores[cid] for cid in CAP_IDS]))
        best_cap = CAP_IDS[best_ci]
        best_score = float(acc[:, best_ci].mean())
        sorted_scores = sorted([scores[cid] for cid in CAP_IDS], reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 0.0
        return scores, best_cap, margin

    w1_s, w1_best, w1_margin = eval_on_indices(pc_only, "W1_PolicyClone")
    w2_s, w2_best, w2_margin = eval_on_indices(aep_only, "W2_AEP")
    w3_s, w3_best, w3_margin = eval_on_indices(po_only, "W3_PrototypeOutcome")

    # W4: grid — evaluate with GoalInference
    from src.capital_report import GoalInferenceCapital, PolicyCloneCapital, PrototypeOutcomeCapital, AEPCapital, SafeFallbackCapital
    from src.capital_report_v2 import make_v2_capitals

    grid_caps_v2 = make_v2_capitals([
        PolicyCloneCapital(pc, "PolicyClone"),
        PrototypeOutcomeCapital(po, "PrototypeOutcome"),
        AEPCapital(aep, "AEP"),
        GoalInferenceCapital(grid_size=5, capital_id="GoalInference"),
        SafeFallbackCapital("SafeFallback"),
    ])
    n_w4 = 100
    w4_acc = np.zeros((n_w4, NC), dtype=np.float32)
    for s in range(n_w4):
        for ci in range(NC):
            obs = grid_env.reset(seed=s * 7 + 500 + ci * 13)
            a = grid_caps_v2[ci].act({"obs": obs, "_reset": True}, [])
            reached = False
            for _step in range(grid_env.max_steps):
                obs, _, done, info = grid_env.step(a)
                if info["at_goal"]:
                    reached = True
                    break
                if done:
                    break
                a = grid_caps_v2[ci].act({"obs": obs}, [])
            w4_acc[s, ci] = float(reached)
    w4_s = {cid: float(w4_acc[:, ci].mean()) for ci, cid in enumerate(CAP_IDS)}
    w4_best_ci = int(np.argmax([w4_s[cid] for cid in CAP_IDS]))
    w4_best = CAP_IDS[w4_best_ci]
    w4_margin = sorted([w4_s[cid] for cid in CAP_IDS], reverse=True)
    w4_margin = w4_margin[0] - w4_margin[1] if len(w4_margin) >= 2 else 0.0

    all_scores = {
        "W1_PolicyClone": {"scores": w1_s, "best": w1_best, "expected": "PolicyClone",
                           "margin": w1_margin, "match": w1_best == "PolicyClone"},
        "W2_AEP": {"scores": w2_s, "best": w2_best, "expected": "AEP",
                   "margin": w2_margin, "match": w2_best == "AEP"},
        "W3_PrototypeOutcome": {"scores": w3_s, "best": w3_best, "expected": "PrototypeOutcome",
                                "margin": w3_margin, "match": w3_best == "PrototypeOutcome"},
        "W4_GoalInference": {"scores": w4_s, "best": w4_best, "expected": "GoalInference",
                             "margin": w4_margin, "match": w4_best == "GoalInference"},
    }

    match_count = sum(1 for v in all_scores.values() if v["match"])
    margin_ok = sum(1 for v in all_scores.values() if v["margin"] >= 0.05)
    oh_bs_gap = max(max(v["scores"].values()) for v in all_scores.values()) - min(
        max(v["scores"].values()) for v in all_scores.values())

    valid = match_count >= 3 and margin_ok >= 3

    return all_scores, match_count, margin_ok, valid