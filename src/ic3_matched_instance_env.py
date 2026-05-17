"""
IC-3-M: Matched Instance Capital Reliability Environment
=========================================================
Proxy-matched instance pairs — same task_id, paired instances that look
similar to proxy classifiers but require different best capitals.

Key components:
  - find_matched_pairs(): Build matched pairs from oracle data pool
  - MatchedInstanceStream: Eval stream yielding matched pairs
  - validate_benchmark(): Benchmark validity audit
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

CAP_IDS = ["PolicyClone", "PrototypeOutcome", "AEP", "GoalInference", "SafeFallback"]
NC = 5

# Pair types: (cap_a, cap_b, region_label)
PAIR_SPECS = [
    (0, 2, "R1_PC_vs_AEP"),
    (2, 1, "R2_AEP_vs_PO"),
    (1, 0, "R3_PO_vs_PC"),
    (3, 2, "R4_GI_vs_AEP"),
    (3, 1, "R4_GI_vs_PO"),
]


def find_matched_pairs(aX_s, aP_raw, min_pair_count=8, sim_threshold=0.3):
    """Find proxy-matched instance pairs from oracle data pool.

    Args:
        aX_s: (N, D) scaled CapitalReport features
        aP_raw: (N, 5) per-capital correctness
        min_pair_count: minimum pairs per region type
        sim_threshold: minimum cosine similarity for matching

    Returns:
        List of dicts: {idx_a, idx_b, pair_type, similarity, cap_a, cap_b}
    """
    pairs = []
    for cap_a, cap_b, ptype in PAIR_SPECS:
        side_a_mask = (aP_raw[:, cap_a] > 0.5) & (aP_raw[:, cap_b] < 0.5) & (aP_raw[:, cap_a] > 0.5)
        side_b_mask = (aP_raw[:, cap_a] < 0.5) & (aP_raw[:, cap_b] > 0.5) & (aP_raw[:, cap_b] > 0.5)
        idx_a = np.where(side_a_mask)[0]
        idx_b = np.where(side_b_mask)[0]
        if len(idx_a) < 3 or len(idx_b) < 3:
            continue

        feats_a = aX_s[idx_a].astype(np.float64)
        feats_b = aX_s[idx_b].astype(np.float64)

        # Normalize for cosine similarity
        na = np.linalg.norm(feats_a, axis=1, keepdims=True) + 1e-8
        nb = np.linalg.norm(feats_b, axis=1, keepdims=True) + 1e-8
        sim = (feats_b / nb) @ (feats_a / na).T

        used_a = set()
        region_pairs = []
        for bi_local in range(len(idx_b)):
            sorted_a = np.argsort(-sim[bi_local])
            for ai_local in sorted_a:
                if ai_local in used_a:
                    continue
                sim_val = float(sim[bi_local, ai_local])
                if sim_val < sim_threshold:
                    break
                used_a.add(int(ai_local))
                region_pairs.append({
                    'idx_a': int(idx_a[ai_local]),
                    'idx_b': int(idx_b[bi_local]),
                    'pair_type': ptype,
                    'cap_a': cap_a,
                    'cap_b': cap_b,
                    'similarity': sim_val,
                })
                break

        if len(region_pairs) >= min_pair_count:
            pairs.extend(region_pairs)

    return pairs


def compute_pair_proxy_similarity(pairs, aX_s):
    """Compute proxy feature similarity within each matched pair."""
    sims = []
    for p in pairs:
        fa = aX_s[p['idx_a']].astype(np.float64)
        fb = aX_s[p['idx_b']].astype(np.float64)
        na, nb = np.linalg.norm(fa)+1e-8, np.linalg.norm(fb)+1e-8
        sims.append(float((fa / na) @ (fb / nb)))
    return float(np.mean(sims))


def compute_pair_opposite_best_rate(pairs, aP_raw):
    """Fraction of pairs where sides have opposite best capitals."""
    n_opp = 0
    for p in pairs:
        ba = int(np.argmax(aP_raw[p['idx_a']]))
        bb = int(np.argmax(aP_raw[p['idx_b']]))
        if ba != bb:
            n_opp += 1
    return n_opp / len(pairs) if pairs else 0.0


def validate_benchmark(pairs, aP_raw, aX_s, bs_idx):
    """Validate benchmark properties against IC-3-M requirements.

    Returns (ok, dict) where dict has all metrics.
    """
    n = aP_raw.shape[0]
    all_pool_indices = list({p['idx_a'] for p in pairs} | {p['idx_b'] for p in pairs})
    if not all_pool_indices:
        return False, {"error": "no matched pairs"}

    pool_mask = np.zeros(n, dtype=bool)
    for i in all_pool_indices:
        pool_mask[i] = True

    pool_correct = aP_raw[pool_mask]
    oh = pool_correct.max(axis=1).mean()
    bs = pool_correct[:, bs_idx].mean()
    oh_gain = float(oh - bs)

    per_cap_mean = pool_correct.mean(axis=0)
    best_score = float(np.max(per_cap_mean))
    second_best = float(np.sort(per_cap_mean)[-2])
    n_caps_above_15pct = int(np.sum(per_cap_mean >= 0.15))

    # Subregime -> best capital accuracy
    # Here "subregime" = pair_type; "best capital" = per-instance argmax
    type_map = {}
    pool_best_cap = np.argmax(pool_correct, axis=1)
    pool_idx_list = list(pool_mask.nonzero()[0])
    for pi_local, pi in enumerate(pool_idx_list):
        for p in pairs:
            if p['idx_a'] == pi:
                type_map.setdefault(p['pair_type'], []).append([pool_best_cap[pi_local], pi_local])
                break
            if p['idx_b'] == pi:
                type_map.setdefault(p['pair_type'], []).append([pool_best_cap[pi_local], pi_local])
                break

    sr_best_cap_acc = 0.5
    if type_map:
        accs = []
        for pt, entries in type_map.items():
            if len(entries) < 3:
                continue
            caps = [e[0] for e in entries]
            mode_cap = max(set(caps), key=caps.count)
            accs.append(sum(1 for c in caps if c == mode_cap) / len(caps))
        sr_best_cap_acc = float(np.mean(accs)) if accs else 0.5

    opposite_rate = compute_pair_opposite_best_rate(pairs, aP_raw)
    proxy_sim = compute_pair_proxy_similarity(pairs, aX_s)

    checks = {
        "oh_gain": oh_gain >= 0.10,
        "best_below_85": best_score < 0.85,
        "caps_above_15pct": n_caps_above_15pct >= 3,
        "sr_best_cap_acc_low": sr_best_cap_acc <= 0.60,
        "opposite_rate_high": opposite_rate >= 0.60,
    }
    all_ok = all(checks.values())

    metrics = {
        "OracleHindsight": float(oh),
        "BestSingle": float(bs),
        "OH_gain": oh_gain,
        "best_score": best_score,
        "second_best": second_best,
        "n_caps_above_15pct": n_caps_above_15pct,
        "subregime_best_cap_acc": sr_best_cap_acc,
        "opposite_rate": opposite_rate,
        "proxy_sim_mean": proxy_sim,
        "n_pairs": len(pairs),
        "n_pool": len(all_pool_indices),
        "per_capital": {CAP_IDS[i]: float(per_cap_mean[i]) for i in range(NC)},
        "checks": checks,
        "valid": all_ok,
    }
    return all_ok, metrics


class MatchedInstanceStream:
    """Eval stream yielding matched-pair instances from oracle data pool.

    Yields per step: features, correctness_vec, pair_type, side, pair_id, best_cap.
    For use in selector evaluation — selector uses features, correctness is ground truth.
    Also builds a lookup map from pool_idx -> stream_position for accurate flip computation.
    """

    def __init__(self, pairs, aX_s, aP_raw, order_seed=42):
        self.pairs = list(pairs)
        self.aX_s = aX_s
        self.aP_raw = aP_raw
        self.rng = np.random.RandomState(order_seed)

        self.steps = []
        for pi, p in enumerate(self.pairs):
            self.steps.append({
                'features': aX_s[p['idx_a']],
                'correctness': aP_raw[p['idx_a']],
                'pair_type': p['pair_type'],
                'side': 'A',
                'pair_index': pi,
                'pool_idx': p['idx_a'],
                'best_cap': int(np.argmax(aP_raw[p['idx_a']])),
            })
            self.steps.append({
                'features': aX_s[p['idx_b']],
                'correctness': aP_raw[p['idx_b']],
                'pair_type': p['pair_type'],
                'side': 'B',
                'pair_index': pi,
                'pool_idx': p['idx_b'],
                'best_cap': int(np.argmax(aP_raw[p['idx_b']])),
            })

        self.rng.shuffle(self.steps)
        self.n_steps = len(self.steps)

        # Build pool_idx -> step_position map
        self._pool_to_pos = {}
        for st in range(self.n_steps):
            self._pool_to_pos[self.steps[st]['pool_idx']] = st

        # Build pair_index -> (pos_A, pos_B) map
        self._pair_to_positions = {}
        for st in range(self.n_steps):
            pi = self.steps[st]['pair_index']
            side = self.steps[st]['side']
            if pi not in self._pair_to_positions:
                self._pair_to_positions[pi] = {}
            self._pair_to_positions[pi][side] = st

    def get_step(self, idx):
        return self.steps[idx % self.n_steps]

    def get_pos_for_pool_idx(self, pool_idx):
        """Get stream position for a given pool index, or -1."""
        return self._pool_to_pos.get(pool_idx, -1)

    def get_pair_positions(self, pair_index):
        """Get (pos_A, pos_B) stream positions for a given pair_index."""
        pos = self._pair_to_positions.get(pair_index, {})
        return pos.get('A', -1), pos.get('B', -1)


def build_matched_eval_stream(pairs, aX_s, aP_raw, seed=42):
    """Build a MatchedInstanceStream for eval."""
    return MatchedInstanceStream(pairs, aX_s, aP_raw, order_seed=seed)