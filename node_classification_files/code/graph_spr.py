"""
Graph-SPR: Spectral Path Regression on graph-diffused angular coordinates.

Model:
    y_hat_v = c0 + sum_q A_q * cos(m_q^T * tilde_theta_v)

Two path selection modes:
    'greedy' : original forward greedy (fast, no graph guidance)
    'beam'   : guided beam search with:
                 - Graph-Laplacian smoothness scoring
                 - Homophily-adaptive neighbourhood consistency
                 - Structural seeding (forces structural paths into model)
                 - Crossover between beams (recombines feature + structural paths)

Best diffusion: PPR (Personalised PageRank) with alpha=0.15
"""

import numpy as np
from itertools import combinations
import scipy.sparse as sp


# ===========================================================================
# SECTION 1: Core linear algebra utilities
# ===========================================================================

def build_design_matrix(tilde_Theta, paths):
    """
    Build design matrix Phi.
    Phi[i,0] = 1 (intercept)
    Phi[i,q+1] = cos(m_q^T tilde_theta_i)
    """
    N    = tilde_Theta.shape[0]
    cols = [np.ones((N, 1))]
    for m in paths:
        phase = tilde_Theta @ m
        cols.append(np.cos(phase).reshape(N, 1))
    return np.hstack(cols)


def ridge_solve(Phi, y, lam):
    """
    Closed-form ridge: beta = (Phi^T Phi + lam*I)^{-1} Phi^T y
    """
    A    = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
    b    = Phi.T @ y
    return np.linalg.solve(A, b)


def predict(Phi, beta):
    return Phi @ beta


# ===========================================================================
# SECTION 2: Candidate generation
# ===========================================================================

def generate_candidates(D, max_support=1, max_freq=2):
    """
    Generate sparse integer frequency vectors m in Z^D.
    Deduplicated by primitive ray (m and -m give same cos).
    """
    candidates = []
    seen       = set()
    freqs      = [f for f in range(-max_freq, max_freq + 1) if f != 0]

    for k in range(1, max_support + 1):
        for support in combinations(range(D), k):
            freq_combos = _product_list(freqs, k)
            for fc in freq_combos:
                m = np.zeros(D, dtype=np.int32)
                for idx, s in enumerate(support):
                    m[s] = fc[idx]
                first_nonzero = m[m != 0][0]
                if first_nonzero < 0:
                    m = -m
                key = tuple(m.tolist())
                if key not in seen:
                    seen.add(key)
                    candidates.append(m.copy())
    return candidates


def _product_list(items, repeat):
    if repeat == 1:
        return [(x,) for x in items]
    sub = _product_list(items, repeat - 1)
    return [(x,) + s for x in items for s in sub]


# ===========================================================================
# SECTION 3: Graph-guided scoring utilities
# ===========================================================================

def compute_graph_laplacian(edge_index, num_nodes):
    """
    Build unnormalised sparse graph Laplacian L = D - A.
    Precomputed once before the search loop.
    """
    row  = edge_index[0].numpy()
    col  = edge_index[1].numpy()
    N    = num_nodes
    data = np.ones(len(row), dtype=np.float32)
    A    = sp.csr_matrix((data, (row, col)), shape=(N, N))
    A    = (A + A.T) / 2                              # symmetrise
    deg  = np.array(A.sum(axis=1)).flatten()
    D    = sp.diags(deg)
    L    = (D - A).tocsr()
    # Store edge arrays for NC computation
    A_sym = (A > 0).astype(np.float32)
    rows, cols = A_sym.nonzero()
    return L, rows.astype(np.int32), cols.astype(np.int32)


def compute_homophily(y, train_mask, edge_src, edge_dst):
    """
    Estimate graph homophily ratio h from training edges.
    h = fraction of edges where both endpoints have same label.
    Only uses edges where both endpoints are in training set.

    Returns h in [0, 1].
    0.5 = mixed (no preference for smooth or rough features).
    """
    both = train_mask[edge_src] & train_mask[edge_dst]
    src_t = edge_src[both]
    dst_t = edge_dst[both]
    if len(src_t) == 0:
        return 0.5
    same = (y[src_t] == y[dst_t])
    return float(same.mean())


def smoothness_score(phi_m, L):
    """
    Rayleigh quotient: phi_m^T L phi_m / phi_m^T phi_m
    Measures how much phi_m varies across edges.
    Small = smooth (good for homophilic).
    Large = rough  (good for heterophilic after sign flip).

    Normalised to [0,1] by dividing by theoretical max (2 for
    normalised Laplacian; we divide by L's largest diagonal for stability).
    """
    denom = float(phi_m @ phi_m)
    if denom < 1e-10:
        return 0.0
    Lphi  = L @ phi_m                          # sparse-dense product O(E)
    raw   = float(phi_m @ Lphi) / denom
    # Normalise by max degree so score is roughly in [0,1]
    max_diag = L.diagonal().max()
    if max_diag > 0:
        raw /= max_diag
    return float(np.clip(raw, 0, 1))


def neighbourhood_consistency(phi_m, edge_src, edge_dst):
    """
    NC(m) = mean |phi_m(u) - phi_m(v)| over all edges.
    Normalised to [0,1] by dividing by 2 (max possible for cos features).
    Low NC = feature agrees across edges = homophily-compatible.
    High NC = feature disagrees across edges = heterophily-compatible.
    """
    if len(edge_src) == 0:
        return 0.5
    diff = np.abs(phi_m[edge_src].astype(np.float64)
                  - phi_m[edge_dst].astype(np.float64))
    return float(diff.mean()) / 2.0   # normalise to [0,1]


def guided_score(val_acc, smooth, nc, h, gamma, lam_nc):
    """
    Combined guided score for one candidate path.

        score = val_acc
              - gamma * (2h-1) * smooth    ← smoothness term
              + lam_nc * [h*(1-nc) + (1-h)*nc]  ← NC reward

    Smoothness term:
        h > 0.5 (homophilic): (2h-1) > 0 → penalise roughness
        h < 0.5 (heterophilic): (2h-1) < 0 → reward roughness
        h = 0.5 (mixed): term vanishes

    NC reward:
        h > 0.5: reward low NC (smooth features)
        h < 0.5: reward high NC (discriminative features)
    """
    smooth_term = gamma * (2 * h - 1) * smooth
    nc_reward   = lam_nc * (h * (1 - nc) + (1 - h) * nc)
    return val_acc - smooth_term + nc_reward


# ===========================================================================
# SECTION 4: Evaluation helpers
# ===========================================================================

def _eval_score(y_true, y_pred, is_classification):
    """Balanced accuracy for classification, R2 for regression."""
    if is_classification:
        pos_mask = y_true == 1
        neg_mask = y_true == 0
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return 0.0
        threshold   = np.median(y_pred)
        pred_binary = (y_pred > threshold).astype(int)
        tpr = np.mean(pred_binary[pos_mask] == 1)
        tnr = np.mean(pred_binary[neg_mask] == 0)
        return (tpr + tnr) / 2.0
    else:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)


def _val_accuracy(Phi_val, beta, y_val, is_classification):
    y_hat = predict(Phi_val, beta)
    return _eval_score(y_val, y_hat, is_classification)


# ===========================================================================
# SECTION 5: Original greedy path selection (kept for comparison)
# ===========================================================================

def greedy_path_selection(tilde_Theta, y, train_mask, val_mask,
                           max_paths=50, lambda_grid=None,
                           max_support=1, max_freq=2,
                           patience=5, verbose=True):
    """
    Original forward greedy selection — no graph guidance.
    Kept for ablation comparison with beam search.
    """
    if lambda_grid is None:
        lambda_grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    is_cls   = np.issubdtype(y.dtype, np.integer) or len(np.unique(y)) <= 20
    y_train  = y[train_mask].astype(np.float64)
    y_val    = y[val_mask].astype(np.float64)
    T_train  = tilde_Theta[train_mask]
    T_val    = tilde_Theta[val_mask]

    selected   = []
    best_lam   = lambda_grid[0]
    val_scores = []
    best_val   = -np.inf
    no_improve = 0

    D          = tilde_Theta.shape[1]
    candidates = generate_candidates(D, max_support, max_freq)
    if verbose:
        print(f"  [Greedy] {len(candidates)} candidates")

    Phi_tr = np.ones((train_mask.sum(), 1))
    Phi_vl = np.ones((val_mask.sum(),   1))

    for step in range(max_paths):
        best_score = -np.inf
        best_m     = None
        best_lam_s = lambda_grid[0]

        for m in candidates:
            if any(np.array_equal(m, p) for p in selected):
                continue
            col_tr = np.cos(T_train @ m).reshape(-1, 1)
            col_vl = np.cos(T_val   @ m).reshape(-1, 1)
            Phi_a  = np.hstack([Phi_tr, col_tr])
            Phi_av = np.hstack([Phi_vl, col_vl])
            lams   = lambda_grid if step == 0 else [best_lam]
            for lam in lams:
                beta  = ridge_solve(Phi_a, y_train, lam)
                score = _val_accuracy(Phi_av, beta, y_val, is_cls)
                if score > best_score:
                    best_score = score; best_m = m.copy(); best_lam_s = lam

        if best_m is None:
            break
        selected.append(best_m)
        if step == 0:
            best_lam = best_lam_s
        col_tr  = np.cos(T_train @ best_m).reshape(-1, 1)
        col_vl  = np.cos(T_val   @ best_m).reshape(-1, 1)
        Phi_tr  = np.hstack([Phi_tr, col_tr])
        Phi_vl  = np.hstack([Phi_vl, col_vl])
        val_scores.append(best_score)
        if verbose:
            nonzero = np.where(best_m != 0)[0].tolist()
            print(f"  [Greedy] step {step+1:3d} | val={best_score:.4f} "
                  f"| dims={nonzero} | lam={best_lam:.0e}")
        if best_score > best_val + 1e-4:
            best_val = best_score; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  [Greedy] early stop at step {step+1}")
                break

    return selected, best_lam, val_scores


# ===========================================================================
# SECTION 6: Guided beam search path selection (NEW)
# ===========================================================================

def guided_beam_path_selection(
    tilde_Theta,         # [N, D]  all nodes — needed for smoothness/NC
    y,                   # [N]     all labels
    train_mask,          # [N] bool
    val_mask,            # [N] bool
    edge_index,          # torch.LongTensor [2, E]
    struct_start_idx,    # int or None — where structural block starts
    beam_width=3,
    max_paths=50,
    lambda_grid=None,
    max_support=1,
    max_freq=2,
    patience=5,
    gamma=0.05,          # smoothness penalty weight
    lam_nc=0.05,         # NC reward weight
    crossover_interval=5,
    verbose=True,
):
    """
    Guided beam search path selection.

    Improvements over greedy:
        1. Graph-Laplacian smoothness score with homophily sign-flip
        2. Neighbourhood consistency reward (homophily-adaptive)
        3. Structural seeding — forces structural paths into initial beams
        4. Beam search — B parallel solutions instead of one
        5. Crossover — recombines feature paths from one beam with
           structural paths from another

    Args:
        tilde_Theta:      [N, D] diffused angular coordinates ALL nodes
        y:                [N] integer class labels
        train_mask:       [N] bool
        val_mask:         [N] bool
        edge_index:       torch LongTensor [2, E]
        struct_start_idx: int — first dimension of structural block.
                          None means no structural block (pure feature model).
        beam_width:       B, number of parallel beams
        gamma:            smoothness penalty weight (small positive)
        lam_nc:           NC reward weight (small positive)
        crossover_interval: apply crossover every this many steps
        ...

    Returns:
        best_paths:  list of np.ndarray [D]
        best_lambda: float
        val_scores:  list of float
    """
    if lambda_grid is None:
        lambda_grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    N   = tilde_Theta.shape[0]
    D   = tilde_Theta.shape[1]
    is_cls = np.issubdtype(y.dtype, np.integer) or len(np.unique(y)) <= 20

    y_train  = y[train_mask].astype(np.float64)
    y_val    = y[val_mask].astype(np.float64)
    T_train  = tilde_Theta[train_mask]    # [N_train, D]
    T_val    = tilde_Theta[val_mask]      # [N_val,   D]

    # ------------------------------------------------------------------
    # PREPROCESSING: graph-guided scoring infrastructure
    # ------------------------------------------------------------------
    print(f"  [Beam] Preprocessing graph-guided scoring...")

    # Graph Laplacian and edge arrays
    L, edge_src, edge_dst = compute_graph_laplacian(edge_index, N)

    # Homophily ratio — estimated from training nodes only
    h = compute_homophily(y, train_mask, edge_src, edge_dst)
    if verbose:
        print(f"  [Beam] Homophily ratio h={h:.3f} "
              f"({'homophilic' if h > 0.5 else 'heterophilic'})")

    # Generate all candidates once
    candidates = generate_candidates(D, max_support, max_freq)
    if verbose:
        print(f"  [Beam] {len(candidates)} candidates, "
              f"beam_width={beam_width}")

    # Lambda selection (done once on first step, then fixed)
    best_lam = lambda_grid[0]

    # ------------------------------------------------------------------
    # HELPER: evaluate one candidate added to an existing beam
    # ------------------------------------------------------------------
    def eval_candidate(m, Phi_tr_cur, Phi_vl_cur, lams):
        """
        Given current beam design matrices and a candidate m,
        returns (val_acc, beta, phi_m_all, best_lam_found).
        phi_m_all is needed for smoothness/NC computation.
        """
        phi_m_all = np.cos(tilde_Theta @ m).astype(np.float64)  # [N]
        col_tr    = phi_m_all[train_mask].reshape(-1, 1)
        col_vl    = phi_m_all[val_mask].reshape(-1, 1)
        Phi_a     = np.hstack([Phi_tr_cur, col_tr])
        Phi_av    = np.hstack([Phi_vl_cur, col_vl])

        best_acc  = -np.inf
        best_beta = None
        best_l    = lams[0]
        for lam in lams:
            beta = ridge_solve(Phi_a, y_train, lam)
            acc  = _val_accuracy(Phi_av, beta, y_val, is_cls)
            if acc > best_acc:
                best_acc = acc; best_beta = beta; best_l = lam

        return best_acc, best_beta, phi_m_all, best_l

    # ------------------------------------------------------------------
    # PHASE 1: Structural seeding
    # ------------------------------------------------------------------
    # Force-evaluate structural candidates and seed beams with them.
    # This guarantees structural information enters at least some beams.
    # Without this, all beams start with feature paths and structural
    # dimensions are never explored.

    print(f"  [Beam] Phase 1: Structural seeding...")

    Phi_init_tr = np.ones((train_mask.sum(), 1))  # intercept only
    Phi_init_vl = np.ones((val_mask.sum(),   1))

    beam_seeds = []   # list of (path, val_acc, phi_m_all)

    if struct_start_idx is not None:
        # Evaluate all structural candidates (dims >= struct_start_idx)
        struct_candidates = [
            m for m in candidates
            if len(m) > 0 and np.any(m[struct_start_idx:] != 0)
            and np.all(m[:struct_start_idx] == 0)
        ]
        if verbose:
            print(f"  [Beam] {len(struct_candidates)} structural candidates "
                  f"(dims {struct_start_idx}+)")

        for m in struct_candidates:
            acc, beta, phi_all, _ = eval_candidate(
                m, Phi_init_tr, Phi_init_vl, lambda_grid)
            smooth = smoothness_score(phi_all, L)
            nc     = neighbourhood_consistency(phi_all, edge_src, edge_dst)
            score  = guided_score(acc, smooth, nc, h, gamma, lam_nc)
            beam_seeds.append((m.copy(), score, acc))

        # Sort by guided score
        beam_seeds.sort(key=lambda x: x[1], reverse=True)
        beam_seeds = beam_seeds[:beam_width]

        if verbose:
            print(f"  [Beam] Structural seeds: "
                  f"{[np.where(s[0]!=0)[0].tolist() for s in beam_seeds]}")

    # If fewer structural seeds than beam_width, fill from feature candidates
    feature_seeds_needed = beam_width - len(beam_seeds)
    if feature_seeds_needed > 0:
        feat_candidates = [
            m for m in candidates
            if struct_start_idx is None
            or np.all(m[struct_start_idx:] == 0)
        ]
        feat_scores = []
        for m in feat_candidates[:min(50, len(feat_candidates))]:
            # Evaluate top 50 feature candidates quickly
            acc, _, phi_all, _ = eval_candidate(
                m, Phi_init_tr, Phi_init_vl, lambda_grid)
            smooth = smoothness_score(phi_all, L)
            nc     = neighbourhood_consistency(phi_all, edge_src, edge_dst)
            score  = guided_score(acc, smooth, nc, h, gamma, lam_nc)
            feat_scores.append((m.copy(), score, acc))
        feat_scores.sort(key=lambda x: x[1], reverse=True)
        beam_seeds += feat_scores[:feature_seeds_needed]

    # ------------------------------------------------------------------
    # Initialise B beams
    # Each beam is a dict containing:
    #   'paths'    : list of selected paths
    #   'Phi_tr'   : current design matrix on train nodes
    #   'Phi_vl'   : current design matrix on val nodes
    #   'val_acc'  : current validation accuracy
    # ------------------------------------------------------------------

    # Select best_lam from seed evaluation
    best_lam = lambda_grid[0]
    if beam_seeds:
        # Re-evaluate top seed with full lambda grid to pick lam
        seed_m = beam_seeds[0][0]
        _, _, _, found_lam = eval_candidate(
            seed_m, Phi_init_tr, Phi_init_vl, lambda_grid)
        best_lam = found_lam

    beams = []
    for seed_m, seed_score, seed_acc in beam_seeds:
        phi_all   = np.cos(tilde_Theta @ seed_m).astype(np.float64)
        col_tr    = phi_all[train_mask].reshape(-1, 1)
        col_vl    = phi_all[val_mask].reshape(-1, 1)
        beta      = ridge_solve(
            np.hstack([Phi_init_tr, col_tr]), y_train, best_lam)
        beams.append({
            'paths':   [seed_m],
            'Phi_tr':  np.hstack([Phi_init_tr, col_tr]),
            'Phi_vl':  np.hstack([Phi_init_vl, col_vl]),
            'val_acc': seed_acc,
        })

    if not beams:
        # Fallback: single empty beam
        beams = [{'paths': [], 'Phi_tr': Phi_init_tr,
                  'Phi_vl': Phi_init_vl, 'val_acc': 0.0}]

    # ------------------------------------------------------------------
    # PHASE 2: Beam search loop
    # ------------------------------------------------------------------
    print(f"  [Beam] Phase 2: Beam search (B={len(beams)})...")

    all_val_scores = []
    best_global    = -np.inf
    no_improve     = 0

    for step in range(max_paths):

        # ---- Expand each beam by one candidate ----
        # For each beam, find best candidate to add using guided score.
        # Result: B expanded beams (one new path added to each).

        expanded = []   # (beam_idx, candidate_m, guided_score, val_acc,
                        #  new_Phi_tr, new_Phi_vl)

        lams_this_step = lambda_grid if step == 0 else [best_lam]

        for b_idx, beam in enumerate(beams):
            best_score_b = -np.inf
            best_m_b     = None
            best_acc_b   = 0.0
            best_Phi_tr  = None
            best_Phi_vl  = None

            selected_keys = {tuple(p.tolist()) for p in beam['paths']}

            for m in candidates:
                if tuple(m.tolist()) in selected_keys:
                    continue

                acc, beta, phi_all, found_lam = eval_candidate(
                    m, beam['Phi_tr'], beam['Phi_vl'], lams_this_step)

                smooth = smoothness_score(phi_all, L)
                nc     = neighbourhood_consistency(
                    phi_all, edge_src, edge_dst)
                score  = guided_score(acc, smooth, nc, h, gamma, lam_nc)

                if score > best_score_b:
                    best_score_b = score
                    best_m_b     = m.copy()
                    best_acc_b   = acc
                    if step == 0:
                        best_lam = found_lam
                    # Build new Phi for this candidate
                    phi_tr = phi_all[train_mask].reshape(-1, 1)
                    phi_vl = phi_all[val_mask].reshape(-1, 1)
                    best_Phi_tr = np.hstack([beam['Phi_tr'], phi_tr])
                    best_Phi_vl = np.hstack([beam['Phi_vl'], phi_vl])

            if best_m_b is not None:
                expanded.append({
                    'paths':   beam['paths'] + [best_m_b],
                    'Phi_tr':  best_Phi_tr,
                    'Phi_vl':  best_Phi_vl,
                    'val_acc': best_acc_b,
                    'guided':  best_score_b,
                })

        if not expanded:
            break

        # Keep top B beams by guided score
        expanded.sort(key=lambda x: x['guided'], reverse=True)
        beams = expanded[:beam_width]

        # Remove 'guided' key (not needed in beam state)
        for b in beams:
            b.pop('guided', None)

        best_step_acc = max(b['val_acc'] for b in beams)
        all_val_scores.append(best_step_acc)

        if verbose:
            best_b = max(beams, key=lambda b: b['val_acc'])
            last_path = best_b['paths'][-1]
            nonzero   = np.where(last_path != 0)[0].tolist()
            ptype     = _path_type(last_path, struct_start_idx)
            print(f"  [Beam] step {step+1:3d} | "
                  f"best_val={best_step_acc:.4f} | "
                  f"dims={nonzero} | type={ptype} | "
                  f"beams={len(beams)}")

        # ---- Crossover every crossover_interval steps ----
        if (len(beams) >= 2
                and struct_start_idx is not None
                and (step + 1) % crossover_interval == 0):

            beams = _crossover(beams, struct_start_idx,
                               tilde_Theta, train_mask, val_mask,
                               y_train, y_val, best_lam,
                               is_cls, L, edge_src, edge_dst,
                               h, gamma, lam_nc, verbose)

        # ---- Early stopping ----
        if best_step_acc > best_global + 1e-4:
            best_global = best_step_acc
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  [Beam] early stop at step {step+1}")
                break

    # Return the paths from the best beam
    best_beam  = max(beams, key=lambda b: b['val_acc'])
    best_paths = best_beam['paths']

    if verbose:
        print(f"  [Beam] Selected {len(best_paths)} paths | "
              f"final_val={best_beam['val_acc']:.4f}")
        _print_path_summary(best_paths, struct_start_idx)

    return best_paths, best_lam, all_val_scores


# ===========================================================================
# SECTION 7: Crossover operator
# ===========================================================================

def _crossover(beams, struct_start_idx,
               tilde_Theta, train_mask, val_mask,
               y_train, y_val, lam,
               is_cls, L, edge_src, edge_dst,
               h, gamma, lam_nc, verbose):
    """
    Crossover between top 2 beams.

    Strategy:
        Take feature-block paths from Beam 1 (best beam — knows good words)
        Take structural-block paths from Beam 2 (second beam — knows good structure)
        Combine into child beam
        If child beats worst beam, replace worst beam with child

    Intuition:
        Beam 1 explored mostly feature space and found best feature paths.
        Beam 2 explored more structural space and found better structural paths.
        The child inherits the best of both — optimal word features AND
        optimal community structure features.
        Neither parent beam could discover this combination alone because
        each was following its own greedy trajectory.
    """
    if len(beams) < 2:
        return beams

    # Sort by current validation accuracy
    beams_sorted = sorted(beams, key=lambda b: b['val_acc'], reverse=True)
    beam1 = beams_sorted[0]   # best — likely has good feature paths
    beam2 = beams_sorted[1]   # second — may have good structural paths

    # Split paths into feature-only and structural
    def is_structural(m):
        return struct_start_idx is not None and np.any(m[struct_start_idx:] != 0)

    feat_paths   = [m for m in beam1['paths'] if not is_structural(m)]
    struct_paths = [m for m in beam2['paths'] if is_structural(m)]

    if not struct_paths:
        # Beam 2 has no structural paths — no useful crossover
        return beams

    # Build child path set
    child_paths = feat_paths + struct_paths

    # Deduplicate child paths
    seen_keys   = set()
    unique_child = []
    for m in child_paths:
        k = tuple(m.tolist())
        if k not in seen_keys:
            seen_keys.add(k)
            unique_child.append(m)

    if not unique_child:
        return beams

    # Evaluate child beam — build its design matrix and compute val_acc
    N_train    = train_mask.sum()
    N_val      = val_mask.sum()
    Phi_tr_c   = np.ones((N_train, 1))
    Phi_vl_c   = np.ones((N_val,   1))
    for m in unique_child:
        phi_all = np.cos(tilde_Theta @ m).astype(np.float64)
        Phi_tr_c = np.hstack([Phi_tr_c, phi_all[train_mask].reshape(-1, 1)])
        Phi_vl_c = np.hstack([Phi_vl_c, phi_all[val_mask].reshape(-1, 1)])

    beta_c   = ridge_solve(Phi_tr_c, y_train, lam)
    acc_c    = _val_accuracy(Phi_vl_c, beta_c, y_val, is_cls)

    child_beam = {
        'paths':   unique_child,
        'Phi_tr':  Phi_tr_c,
        'Phi_vl':  Phi_vl_c,
        'val_acc': acc_c,
    }

    # Replace worst beam if child is better
    worst_idx = min(range(len(beams)), key=lambda i: beams[i]['val_acc'])
    if acc_c > beams[worst_idx]['val_acc']:
        beams[worst_idx] = child_beam
        if verbose:
            feat_dims   = [np.where(m != 0)[0].tolist() for m in feat_paths]
            struct_dims = [np.where(m != 0)[0].tolist() for m in struct_paths]
            print(f"  [Beam] Crossover: child acc={acc_c:.4f} "
                  f"(feat from B1={feat_dims[:2]}, "
                  f"struct from B2={struct_dims[:2]})")

    return beams


# ===========================================================================
# SECTION 8: Path type classification and summary
# ===========================================================================

def _path_type(m, struct_start_idx):
    """Classify a path as feature-only, structural-only, or cross."""
    if struct_start_idx is None:
        return "feature"
    nonzero = np.where(m != 0)[0]
    feat_dims   = nonzero[nonzero < struct_start_idx]
    struct_dims = nonzero[nonzero >= struct_start_idx]
    if len(feat_dims) > 0 and len(struct_dims) > 0:
        return "CROSS"         # feature × structural interaction
    elif len(struct_dims) > 0:
        return "structural"
    else:
        return "feature"


def _print_path_summary(paths, struct_start_idx):
    """Print a summary of selected paths showing type distribution."""
    types = [_path_type(m, struct_start_idx) for m in paths]
    n_feat   = types.count("feature")
    n_struct = types.count("structural")
    n_cross  = types.count("CROSS")
    print(f"  [Beam] Path types: {n_feat} feature, "
          f"{n_struct} structural, {n_cross} CROSS")
    if n_cross > 0:
        print(f"  [Beam] *** Cross paths found! "
              f"Feature-topology interactions discovered ***")


# ===========================================================================
# SECTION 9: GraphSPR model class
# ===========================================================================

class GraphSPR:
    """
    Graph Spectral Path Regression.

    Two selection modes:
        'greedy': original forward greedy (fast, no graph guidance)
        'beam':   guided beam search with smoothness/NC/crossover

    Usage:
        model = GraphSPR(mode='beam', beam_width=3, gamma=0.05)
        model.fit(tilde_Theta, y, train_mask, val_mask,
                  edge_index=data.edge_index,
                  struct_start_idx=meta['struct_start_idx'])
        acc = model.evaluate(tilde_Theta, y, test_mask)
    """

    def __init__(self, K=4, r=0.5, max_paths=50,
                 max_support=1, max_freq=2,
                 lambda_grid=None, patience=5,
                 mode='beam',
                 beam_width=3,
                 gamma=0.05,
                 lam_nc=0.05,
                 crossover_interval=5,
                 verbose=True):
        self.K                  = K
        self.r                  = r
        self.max_paths          = max_paths
        self.max_support        = max_support
        self.max_freq           = max_freq
        self.lambda_grid        = lambda_grid
        self.patience           = patience
        self.mode               = mode
        self.beam_width         = beam_width
        self.gamma              = gamma
        self.lam_nc             = lam_nc
        self.crossover_interval = crossover_interval
        self.verbose            = verbose

        self.selected_paths = None
        self.beta           = None
        self.best_lambda    = None
        self.val_scores     = None

    def fit(self, tilde_Theta, y, train_mask, val_mask,
            edge_index=None, struct_start_idx=None):
        """
        Run path selection then final ridge solve.

        Args:
            tilde_Theta:      [N, D] all nodes
            y:                [N] labels
            train_mask:       [N] bool
            val_mask:         [N] bool
            edge_index:       torch LongTensor [2, E] — required for beam mode
            struct_start_idx: int or None — start of structural block
        """
        if self.mode == 'beam':
            if edge_index is None:
                raise ValueError(
                    "edge_index required for beam mode. "
                    "Pass data.edge_index to model.fit()")
            self.selected_paths, self.best_lambda, self.val_scores = \
                guided_beam_path_selection(
                    tilde_Theta, y, train_mask, val_mask,
                    edge_index        = edge_index,
                    struct_start_idx  = struct_start_idx,
                    beam_width        = self.beam_width,
                    max_paths         = self.max_paths,
                    lambda_grid       = self.lambda_grid,
                    max_support       = self.max_support,
                    max_freq          = self.max_freq,
                    patience          = self.patience,
                    gamma             = self.gamma,
                    lam_nc            = self.lam_nc,
                    crossover_interval= self.crossover_interval,
                    verbose           = self.verbose,
                )
        else:  # greedy
            self.selected_paths, self.best_lambda, self.val_scores = \
                greedy_path_selection(
                    tilde_Theta, y, train_mask, val_mask,
                    max_paths   = self.max_paths,
                    lambda_grid = self.lambda_grid,
                    max_support = self.max_support,
                    max_freq    = self.max_freq,
                    patience    = self.patience,
                    verbose     = self.verbose,
                )

        # Final ridge solve on full training set with selected paths
        Phi_train       = build_design_matrix(
            tilde_Theta[train_mask], self.selected_paths)
        y_train         = y[train_mask].astype(np.float64)
        self.beta       = ridge_solve(Phi_train, y_train, self.best_lambda)
        return self

    def predict_raw(self, tilde_Theta, mask=None):
        T   = tilde_Theta if mask is None else tilde_Theta[mask]
        Phi = build_design_matrix(T, self.selected_paths)
        return Phi @ self.beta

    def predict(self, tilde_Theta, mask=None):
        raw       = self.predict_raw(tilde_Theta, mask)
        threshold = np.median(raw)
        return (raw > threshold).astype(int)

    def evaluate(self, tilde_Theta, y, mask):
        preds = self.predict(tilde_Theta, mask)
        return np.mean(preds == y[mask])

    def path_summary(self, struct_start_idx=None, feature_names=None):
        print(f"\n{'='*55}")
        print(f"Selected {len(self.selected_paths)} paths | "
              f"lambda={self.best_lambda:.0e} | mode={self.mode}")
        print(f"{'='*55}")
        for i, m in enumerate(self.selected_paths):
            nonzero = np.where(m != 0)[0]
            ptype   = _path_type(m, struct_start_idx)
            if feature_names is not None:
                terms = [f"{m[j]}*θ_{feature_names[j]}" for j in nonzero]
            else:
                terms = [f"{m[j]}*θ_{j}" for j in nonzero]
            print(f"  Path {i+1:2d}: cos({' + '.join(terms)}) "
                  f"[{ptype}]  coeff={self.beta[i+1]:.4f}")
        print(f"{'='*55}\n")
        if struct_start_idx is not None:
            _print_path_summary(self.selected_paths, struct_start_idx)
