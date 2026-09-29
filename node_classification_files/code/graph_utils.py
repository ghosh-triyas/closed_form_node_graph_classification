"""
Graph utilities: diffusion operator, angular transformation,
structural descriptors (LapPE), and augmented node representation.

Core idea:
    tilde_Theta = (I + P_hat) @ Theta_aug
where:
    Theta_aug  = arccos(scale([X | S]))
    X          = raw node features
    S          = structural descriptors (squared Laplacian eigenvectors)
    P_hat      = sum_k r^k * P^k  (diagonal zeroed)
    
The structural block S gives SPR explicit access to community membership.
Cross paths m with support spanning both blocks capture feature-topology
interactions — the primary novel contribution over vanilla SPR.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


# ---------------------------------------------------------------------------
# 1. Transition matrix  P = D^{-1} A  (sparse)
# ---------------------------------------------------------------------------

def compute_transition_matrix(edge_index, num_nodes):
    row  = edge_index[0].numpy()
    col  = edge_index[1].numpy()
    data = np.ones(len(row), dtype=np.float32)
    A      = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    degree = np.array(A.sum(axis=1)).flatten()
    degree[degree == 0] = 1.0
    D_inv  = sp.diags(1.0 / degree)
    P      = (D_inv @ A).tocsr()
    return P


# ---------------------------------------------------------------------------
# 2. Diffusion operator  P_hat = sum_{k=1}^{K} r^k * P^k  (sparse)
# ---------------------------------------------------------------------------

def compute_diffusion_operator(P, K=4, r=0.5):
    N       = P.shape[0]
    P_hat   = sp.csr_matrix((N, N), dtype=np.float32)
    P_power = sp.eye(N, format='csr', dtype=np.float32)
    for k in range(1, K + 1):
        P_power = (P_power @ P).tocsr()
        P_hat   = P_hat + (r ** k) * P_power
    P_hat    = P_hat.tolil()
    P_hat.setdiag(0)
    P_hat    = P_hat.tocsr()
    P_hat.eliminate_zeros()
    return P_hat


# ---------------------------------------------------------------------------
# 3. Laplacian Positional Encoding (LapPE) — squared Fiedler components
#
# For each node v, computes:
#     s_v = [phi_1(v)^2, phi_2(v)^2, ..., phi_K(v)^2]
#
# Squaring removes sign ambiguity (phi_k and -phi_k give same result).
# Values are in [0, 1] — map to [-1, 1] via 2*s - 1 for arccos transform.
#
# Uses scipy eigsh (Lanczos/ARPACK) for top-K eigenvectors of sparse L.
# Cost: O(K * N * nnz) — fast for sparse graphs.
# ---------------------------------------------------------------------------

def compute_laplacian_pe(edge_index, num_nodes, K=8):
    """
    Compute squared Laplacian eigenvector components per node.

    Args:
        edge_index: torch.LongTensor [2, E]
        num_nodes:  int
        K:          int, number of eigenvectors (excluding trivial phi_0)

    Returns:
        S: np.ndarray [N, K], values in [-1, 1] ready for arccos
    """
    row  = edge_index[0].numpy()
    col  = edge_index[1].numpy()
    data = np.ones(len(row), dtype=np.float64)

    # Build symmetric adjacency (ensure undirected)
    A = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = (A + A.T) / 2  # symmetrise

    # Normalised graph Laplacian  L = I - D^{-1/2} A D^{-1/2}
    degree  = np.array(A.sum(axis=1)).flatten()
    degree[degree == 0] = 1.0
    D_invsqrt = sp.diags(1.0 / np.sqrt(degree))
    L = sp.eye(num_nodes) - D_invsqrt @ A @ D_invsqrt
    L = L.tocsr()

    # Request K+1 smallest eigenvectors — skip phi_0 (constant, no info)
    # which='SM' = smallest magnitude eigenvalues
    print(f"  Computing top {K} Laplacian eigenvectors (Lanczos)...")
    try:
        # sigma=1e-6 shifts spectrum for better numerical stability
        eigenvalues, eigenvectors = spla.eigsh(
            L, k=K+1, which='SM', tol=1e-6, maxiter=num_nodes * 10
        )
        # Sort by eigenvalue (ascending)
        idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        # Skip phi_0 (eigenvalue ~0, constant vector)
        eigenvectors = eigenvectors[:, 1:K+1]   # [N, K]
    except Exception as e:
        print(f"  Warning: eigsh failed ({e}), using random features")
        eigenvectors = np.random.randn(num_nodes, K)

    # Square to remove sign ambiguity: phi_k(v)^2 in [0, 1]
    S_raw = eigenvectors ** 2   # [N, K]

    # Map [0, 1] -> [-1, 1] for arccos transform
    S = 2 * S_raw - 1           # [N, K], values in [-1, 1]

    print(f"  LapPE shape: {S.shape}, range: [{S.min():.3f}, {S.max():.3f}]")
    return S.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Feature scaling to [-1, 1] using robust tanh (per column, train only)
# ---------------------------------------------------------------------------

def robust_tanh_scale(X, train_mask, eps=1e-8):
    X_train  = X[train_mask]
    center   = np.median(X_train, axis=0)
    scale    = np.median(np.abs(X_train - center), axis=0) + eps
    X_scaled = np.tanh((X - center) / scale)
    return X_scaled, center, scale


# ---------------------------------------------------------------------------
# 5. Angular transformation  Theta = arccos(X_scaled)
# ---------------------------------------------------------------------------

def angular_transform(X_scaled):
    X_clipped = np.clip(X_scaled, -1 + 1e-7, 1 - 1e-7)
    return np.arccos(X_clipped).astype(np.float32)


# ---------------------------------------------------------------------------
# 6. Graph-diffused angular coordinates  (sparse-dense multiply)
# ---------------------------------------------------------------------------

def compute_diffused_angular(Theta, P_hat):
    diffused    = P_hat @ Theta
    tilde_Theta = Theta + diffused
    return tilde_Theta.astype(np.float32)


# ---------------------------------------------------------------------------
# 7. Feature prescreening — SEPARATE budgets for feature and structural blocks
#
# Feature block:    keep top top_T_feat by max-OvR correlation
# Structural block: keep ALL K eigenvectors (small, always informative)
#
# Separate prescreening guarantees structural dimensions are present
# so SPR can discover cross paths (feature x community interactions).
# ---------------------------------------------------------------------------

def prescreen_features_separate(tilde_Theta_feat, tilde_Theta_struct,
                                  y, train_mask, top_T_feat=80):
    """
    Prescreen feature block and keep full structural block.

    Args:
        tilde_Theta_feat:   np.ndarray [N, D],  diffused feature angular coords
        tilde_Theta_struct: np.ndarray [N, K],  diffused structural angular coords
        y:                  np.ndarray [N],      integer labels
        train_mask:         np.ndarray bool [N]
        top_T_feat:         int, features to keep from feature block

    Returns:
        tilde_Theta_combined: np.ndarray [N, top_T_feat + K]
        feat_indices:         np.ndarray [top_T_feat], kept feature indices
        struct_start_idx:     int, where structural block starts in combined
    """
    T_train = tilde_Theta_feat[train_mask]
    y_train = y[train_mask]
    num_classes = len(np.unique(y_train))
    D = T_train.shape[1]

    # Max absolute OvR correlation across all classes
    scores = np.zeros(D)
    for c in range(num_classes):
        y_binary = (y_train == c).astype(np.float32)
        corrs    = np.abs(np.corrcoef(T_train.T, y_binary)[-1, :-1])
        corrs    = np.nan_to_num(corrs, nan=0.0)
        scores   = np.maximum(scores, corrs)

    top_T_feat = min(top_T_feat, D)
    feat_indices = np.argsort(scores)[::-1][:top_T_feat]
    feat_selected = tilde_Theta_feat[:, feat_indices]

    # Concatenate: [selected features | ALL structural dims]
    tilde_Theta_combined = np.hstack([feat_selected, tilde_Theta_struct])
    struct_start_idx     = top_T_feat

    K = tilde_Theta_struct.shape[1]
    print(f"  Prescreening: kept top {top_T_feat} features + {K} structural dims")
    print(f"  Combined shape: {tilde_Theta_combined.shape}")
    print(f"  Feature block: dims 0-{top_T_feat-1}")
    print(f"  Structural block: dims {top_T_feat}-{top_T_feat+K-1}")

    return tilde_Theta_combined, feat_indices, struct_start_idx


# ---------------------------------------------------------------------------
# 8. Full pipeline: raw features + LapPE + graph -> augmented tilde_Theta
# ---------------------------------------------------------------------------

def build_diffused_angular_matrix(X, edge_index, num_nodes, train_mask,
                                   K=4, r=0.5, top_T=None, y=None,
                                   lap_K=8, use_lap_pe=True):
    """
    Full memory-efficient preprocessing pipeline with optional LapPE.

    Steps:
        1. Sparse transition matrix P
        2. Sparse diffusion operator P_hat
        3. Compute LapPE (squared Fiedler components)  [if use_lap_pe]
        4. Scale raw features + structural descriptors to [-1, 1]
        5. Angular transform -> Theta_feat, Theta_struct
        6. Diffuse both: tilde_Theta = (I + P_hat) @ Theta
        7. Prescreen feature block separately, keep full structural block
        8. Concatenate -> final augmented angular matrix

    Args:
        X:          np.ndarray [N, D]
        edge_index: torch.LongTensor [2, E]
        num_nodes:  int
        train_mask: np.ndarray bool [N]
        K:          int, diffusion hops
        r:          float, decay rate
        top_T:      int, features to keep from feature block
        y:          np.ndarray [N], labels
        lap_K:      int, number of Laplacian eigenvectors
        use_lap_pe: bool, whether to add LapPE structural block

    Returns:
        tilde_Theta: np.ndarray [N, top_T + lap_K] or [N, D + lap_K]
        meta:        dict
    """
    print(f"  Step 1/6: Computing sparse transition matrix...")
    P = compute_transition_matrix(edge_index, num_nodes)
    print(f"    P: sparse {P.shape}, nnz={P.nnz}")

    print(f"  Step 2/6: Computing sparse diffusion operator (K={K}, r={r})...")
    P_hat = compute_diffusion_operator(P, K=K, r=r)
    print(f"    P_hat: sparse {P_hat.shape}, nnz={P_hat.nnz}")

    # --- Feature block ---
    print(f"  Step 3/6: Scaling raw features...")
    X_scaled, center, scale = robust_tanh_scale(X, train_mask)

    print(f"  Step 4/6: Angular transform on features...")
    Theta_feat = angular_transform(X_scaled)
    print(f"    Theta_feat shape: {Theta_feat.shape}")

    print(f"  Step 5/6: Diffusing feature angular coordinates...")
    tilde_Theta_feat = compute_diffused_angular(Theta_feat, P_hat)

    # --- Structural block (LapPE) ---
    tilde_Theta_struct = None
    if use_lap_pe:
        print(f"  Step 5b/6: Computing LapPE (lap_K={lap_K})...")
        S = compute_laplacian_pe(edge_index, num_nodes, K=lap_K)
        # S is already in [-1, 1] — apply arccos directly
        Theta_struct = angular_transform(S)
        print(f"    Theta_struct shape: {Theta_struct.shape}")
        # Diffuse structural coordinates too
        tilde_Theta_struct = compute_diffused_angular(Theta_struct, P_hat)
        print(f"    tilde_Theta_struct shape: {tilde_Theta_struct.shape}")

    # --- Prescreening and concatenation ---
    print(f"  Step 6/6: Prescreening and concatenation...")
    feat_indices     = None
    struct_start_idx = None

    if use_lap_pe and tilde_Theta_struct is not None:
        # Separate prescreening
        top_T_feat = top_T if top_T is not None else tilde_Theta_feat.shape[1]
        assert y is not None, "y must be provided for prescreening"
        tilde_Theta, feat_indices, struct_start_idx = \
            prescreen_features_separate(
                tilde_Theta_feat, tilde_Theta_struct,
                y, train_mask, top_T_feat=top_T_feat
            )
    else:
        # No structural block — prescreen feature block only
        tilde_Theta = tilde_Theta_feat
        if top_T is not None and top_T < tilde_Theta.shape[1]:
            assert y is not None
            from graph_utils import prescreen_features_separate
            # Use simple OvR prescreening on feature block alone
            T_train = tilde_Theta[train_mask]
            y_train = y[train_mask]
            num_classes = len(np.unique(y_train))
            D = T_train.shape[1]
            scores = np.zeros(D)
            for c in range(num_classes):
                y_binary = (y_train == c).astype(np.float32)
                corrs    = np.abs(np.corrcoef(T_train.T, y_binary)[-1, :-1])
                scores   = np.maximum(scores, np.nan_to_num(corrs, nan=0.0))
            feat_indices = np.argsort(scores)[::-1][:top_T]
            tilde_Theta  = tilde_Theta[:, feat_indices]
            print(f"  Prescreening: kept top {top_T} features")

    print(f"  Final tilde_Theta shape: {tilde_Theta.shape}")

    meta = {
        'P_hat':            P_hat,
        'center':           center,
        'scale':            scale,
        'feat_indices':     feat_indices,
        'struct_start_idx': struct_start_idx,
        'lap_K':            lap_K,
    }
    return tilde_Theta, meta


# ---------------------------------------------------------------------------
# 9. RWPE — Random Walk Positional Encoding
#
# For each node v, computes diagonal entries of transition matrix powers:
#     s_v = [(P^2)_vv, (P^4)_vv, (P^6)_vv, (P^8)_vv, ...]
#
# (P^k)_vv = probability of returning to v after exactly k steps.
# k=2: encodes degree structure
# k=3: encodes triangle participation
# k=4: encodes 4-cycles and 2-hop loop structure
# higher k: increasingly global loop structure
#
# Values are in [0, 1] — map to [-1, 1] via 2*s - 1 for arccos.
# No eigendecomposition needed — just sparse matrix powers.
# ---------------------------------------------------------------------------

def compute_rwpe(P, K_rw=8):
    """
    Compute Random Walk Positional Encoding per node.

    Args:
        P:    scipy.sparse.csr_matrix [N, N], row-stochastic transition matrix
        K_rw: int, number of return probability steps to compute
              Uses steps 1, 2, ..., K_rw

    Returns:
        S: np.ndarray [N, K_rw], values in [-1, 1] ready for arccos
    """
    N = P.shape[0]
    diags = []

    P_power = sp.eye(N, format='csr', dtype=np.float32)  # P^0 = I

    for k in range(1, K_rw + 1):
        P_power = (P_power @ P).tocsr()   # P^k
        # Extract diagonal — return probability after k steps
        diag_k  = np.array(P_power.diagonal()).flatten()
        diags.append(diag_k)

    S_raw = np.stack(diags, axis=1)   # [N, K_rw], values in [0, 1]

    # Map [0, 1] -> [-1, 1] for arccos transform
    S = 2 * S_raw - 1                  # [N, K_rw]

    print(f"  RWPE shape: {S.shape}, range: [{S.min():.3f}, {S.max():.3f}]")
    return S.astype(np.float32)


def build_diffused_angular_matrix_rwpe(X, edge_index, num_nodes, train_mask,
                                        K=4, r=0.5, top_T=None, y=None,
                                        rw_K=8):
    """
    Full pipeline with RWPE structural block (no LapPE).

    RWPE encodes local structural role via return probabilities.
    Complements LapPE which encodes global community membership.

    Args:
        X:          np.ndarray [N, D]
        edge_index: torch.LongTensor [2, E]
        num_nodes:  int
        train_mask: np.ndarray bool [N]
        K:          int, diffusion hops
        r:          float, decay rate
        top_T:      int, features to keep from feature block
        y:          np.ndarray [N], labels
        rw_K:       int, number of RWPE steps

    Returns:
        tilde_Theta: np.ndarray [N, top_T + rw_K]
        meta:        dict
    """
    print(f"  Step 1/6: Computing sparse transition matrix...")
    P = compute_transition_matrix(edge_index, num_nodes)
    print(f"    P: sparse {P.shape}, nnz={P.nnz}")

    print(f"  Step 2/6: Computing sparse diffusion operator (K={K}, r={r})...")
    P_hat = compute_diffusion_operator(P, K=K, r=r)
    print(f"    P_hat: sparse {P_hat.shape}, nnz={P_hat.nnz}")

    # --- Feature block ---
    print(f"  Step 3/6: Scaling raw features...")
    X_scaled, center, scale = robust_tanh_scale(X, train_mask)

    print(f"  Step 4/6: Angular transform on features...")
    Theta_feat = angular_transform(X_scaled)

    print(f"  Step 5/6: Diffusing feature angular coordinates...")
    tilde_Theta_feat = compute_diffused_angular(Theta_feat, P_hat)

    # --- RWPE structural block ---
    print(f"  Step 5b/6: Computing RWPE (rw_K={rw_K})...")
    S_rw = compute_rwpe(P, K_rw=rw_K)
    Theta_struct = angular_transform(S_rw)
    tilde_Theta_struct = compute_diffused_angular(Theta_struct, P_hat)
    print(f"    tilde_Theta_struct (RWPE) shape: {tilde_Theta_struct.shape}")

    # --- Prescreening and concatenation ---
    print(f"  Step 6/6: Prescreening and concatenation...")
    top_T_feat = top_T if top_T is not None else tilde_Theta_feat.shape[1]
    assert y is not None, "y must be provided for prescreening"

    tilde_Theta, feat_indices, struct_start_idx = \
        prescreen_features_separate(
            tilde_Theta_feat, tilde_Theta_struct,
            y, train_mask, top_T_feat=top_T_feat
        )

    print(f"  Final tilde_Theta shape: {tilde_Theta.shape}")

    meta = {
        'P_hat':            P_hat,
        'center':           center,
        'scale':            scale,
        'feat_indices':     feat_indices,
        'struct_start_idx': struct_start_idx,
        'rw_K':             rw_K,
    }
    return tilde_Theta, meta


# ---------------------------------------------------------------------------
# 10. LapPE + RWPE combined pipeline
#
# Concatenates both structural blocks:
#     [LapPE (lap_K dims) | RWPE (rw_K dims)]
#
# LapPE: global community membership via squared Fiedler components
# RWPE:  local structural role via return probabilities
#
# Together they cover both global and local structural information.
# Final augmented vector: [top_T features | lap_K LapPE | rw_K RWPE]
# ---------------------------------------------------------------------------

def build_diffused_angular_matrix_combined(X, edge_index, num_nodes, train_mask,
                                            K=4, r=0.5, top_T=None, y=None,
                                            lap_K=8, rw_K=4):
    """
    Full pipeline with LapPE + RWPE combined structural block.

    Structural block = [squared Fiedler components | return probabilities]
    Total structural dims = lap_K + rw_K

    Args:
        X:          np.ndarray [N, D]
        edge_index: torch.LongTensor [2, E]
        num_nodes:  int
        train_mask: np.ndarray bool [N]
        K:          int, diffusion hops
        r:          float, decay rate
        top_T:      int, features to keep from feature block
        y:          np.ndarray [N], labels
        lap_K:      int, number of Laplacian eigenvectors
        rw_K:       int, number of RWPE steps

    Returns:
        tilde_Theta: np.ndarray [N, top_T + lap_K + rw_K]
        meta:        dict
    """
    print(f"  Step 1/6: Computing sparse transition matrix...")
    P = compute_transition_matrix(edge_index, num_nodes)
    print(f"    P: sparse {P.shape}, nnz={P.nnz}")

    print(f"  Step 2/6: Computing sparse diffusion operator (K={K}, r={r})...")
    P_hat = compute_diffusion_operator(P, K=K, r=r)
    print(f"    P_hat: sparse {P_hat.shape}, nnz={P_hat.nnz}")

    # --- Feature block ---
    print(f"  Step 3/6: Scaling raw features...")
    X_scaled, center, scale = robust_tanh_scale(X, train_mask)

    print(f"  Step 4/6: Angular transform on features...")
    Theta_feat = angular_transform(X_scaled)

    print(f"  Step 5/6: Diffusing feature angular coordinates...")
    tilde_Theta_feat = compute_diffused_angular(Theta_feat, P_hat)

    # --- LapPE structural block ---
    print(f"  Step 5b/6: Computing LapPE (lap_K={lap_K})...")
    S_lap = compute_laplacian_pe(edge_index, num_nodes, K=lap_K)
    Theta_lap = angular_transform(S_lap)
    tilde_Theta_lap = compute_diffused_angular(Theta_lap, P_hat)
    print(f"    tilde_Theta_lap shape: {tilde_Theta_lap.shape}")

    # --- RWPE structural block ---
    print(f"  Step 5c/6: Computing RWPE (rw_K={rw_K})...")
    S_rw = compute_rwpe(P, K_rw=rw_K)
    Theta_rw = angular_transform(S_rw)
    tilde_Theta_rw = compute_diffused_angular(Theta_rw, P_hat)
    print(f"    tilde_Theta_rw shape: {tilde_Theta_rw.shape}")

    # --- Concatenate structural blocks ---
    tilde_Theta_struct = np.hstack([tilde_Theta_lap, tilde_Theta_rw])
    print(f"    Combined structural block shape: {tilde_Theta_struct.shape}")
    print(f"    LapPE dims: 0-{lap_K-1}, RWPE dims: {lap_K}-{lap_K+rw_K-1}")

    # --- Prescreening and concatenation ---
    print(f"  Step 6/6: Prescreening and concatenation...")
    top_T_feat = top_T if top_T is not None else tilde_Theta_feat.shape[1]
    assert y is not None, "y must be provided for prescreening"

    tilde_Theta, feat_indices, struct_start_idx = \
        prescreen_features_separate(
            tilde_Theta_feat, tilde_Theta_struct,
            y, train_mask, top_T_feat=top_T_feat
        )

    print(f"  Final tilde_Theta shape: {tilde_Theta.shape}")
    print(f"  Feature block: dims 0-{top_T_feat-1}")
    print(f"  LapPE block:   dims {top_T_feat}-{top_T_feat+lap_K-1}")
    print(f"  RWPE block:    dims {top_T_feat+lap_K}-{top_T_feat+lap_K+rw_K-1}")

    meta = {
        'P_hat':            P_hat,
        'center':           center,
        'scale':            scale,
        'feat_indices':     feat_indices,
        'struct_start_idx': struct_start_idx,
        'lap_K':            lap_K,
        'rw_K':             rw_K,
    }
    return tilde_Theta, meta


# ---------------------------------------------------------------------------
# 11. HKS — Heat Kernel Signature
#
# HKS_v(t) = sum_k exp(-lambda_k * t) * phi_k(v)^2
#
# At small t: heat has not diffused far — captures local structure (degree,
#             local density). Similar to RWPE at short hops.
# At large t: heat has diffused globally — captures global position
#             (community membership). Similar to LapPE.
#
# T time scales give T structural dimensions per node.
# Reuses eigenvectors already computed for LapPE — no extra eigendecomp.
#
# Values are always positive (sum of non-negative terms).
# Map to [-1, 1] via column-wise normalisation then 2*s-1.
# ---------------------------------------------------------------------------

def compute_hks(eigenvalues, eigenvectors, T=6):
    """
    Compute Heat Kernel Signature at T time scales per node.

    Time scales chosen logarithmically between t_min and t_max
    to capture both local and global structure.

    Args:
        eigenvalues:  np.ndarray [K], Laplacian eigenvalues (sorted ascending)
        eigenvectors: np.ndarray [N, K], corresponding eigenvectors
        T:            int, number of time scales

    Returns:
        S: np.ndarray [N, T], values in [-1, 1] ready for arccos
    """
    # Skip lambda_0 ~ 0 (trivial eigenvalue)
    # Use lambda_1 onwards
    lam  = eigenvalues[1:]       # [K-1]
    phi  = eigenvectors[:, 1:]   # [N, K-1]

    # Logarithmic time scale from 4/lambda_max to 4/lambda_min+1
    lam_min = lam[lam > 1e-6].min() if (lam > 1e-6).any() else 1e-2
    lam_max = lam.max()
    t_min   = 4.0 / lam_max
    t_max   = 4.0 / lam_min
    times   = np.exp(np.linspace(np.log(t_min), np.log(t_max), T))

    print(f"  HKS time scales: {times.round(4).tolist()}")

    N      = phi.shape[0]
    hks    = np.zeros((N, T), dtype=np.float64)

    phi_sq = phi ** 2   # [N, K-1]

    for i, t in enumerate(times):
        weights    = np.exp(-lam * t)   # [K-1]
        hks[:, i]  = phi_sq @ weights   # [N]

    # Normalise each time scale column to [0, 1]
    col_min = hks.min(axis=0, keepdims=True)
    col_max = hks.max(axis=0, keepdims=True)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0
    hks_norm = (hks - col_min) / col_range   # [N, T], in [0, 1]

    # Map [0, 1] -> [-1, 1] for arccos
    S = 2 * hks_norm - 1

    print(f"  HKS shape: {S.shape}, range: [{S.min():.3f}, {S.max():.3f}]")
    return S.astype(np.float32)


def compute_laplacian_eig(edge_index, num_nodes, K=8):
    """
    Compute top-K Laplacian eigenvectors and eigenvalues.
    Returns both so HKS can reuse them without recomputing.

    Args:
        edge_index: torch.LongTensor [2, E]
        num_nodes:  int
        K:          int, number of eigenvectors (excluding trivial)

    Returns:
        eigenvalues:  np.ndarray [K+1], includes lambda_0
        eigenvectors: np.ndarray [N, K+1], includes phi_0
    """
    row  = edge_index[0].numpy()
    col  = edge_index[1].numpy()
    data = np.ones(len(row), dtype=np.float64)

    A = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = (A + A.T) / 2

    # Unnormalised Laplacian L = D - A
    degree = np.array(A.sum(axis=1)).flatten()
    degree[degree == 0] = 1.0
    D = sp.diags(degree)
    L = (D - A).tocsr()

    print(f"  Computing top {K+1} Laplacian eigenvectors...")
    try:
        eigenvalues, eigenvectors = spla.eigsh(
            L, k=K+1, which='SM', tol=1e-6, maxiter=num_nodes * 10
        )
        idx          = np.argsort(eigenvalues)
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    except Exception as e:
        print(f"  Warning: eigsh failed ({e}), using random features")
        eigenvalues  = np.zeros(K+1)
        eigenvectors = np.random.randn(num_nodes, K+1)

    return eigenvalues, eigenvectors


def build_diffused_angular_matrix_hks(X, edge_index, num_nodes, train_mask,
                                       K=4, r=0.5, top_T=None, y=None,
                                       lap_K=8, hks_T=6):
    """
    Full pipeline with LapPE + HKS combined structural block.

    LapPE and HKS share the same eigendecomposition — computed once.
    Total structural dims = lap_K + hks_T

    Args:
        X:          np.ndarray [N, D]
        edge_index: torch.LongTensor [2, E]
        num_nodes:  int
        train_mask: np.ndarray bool [N]
        K:          int, diffusion hops
        r:          float, decay rate
        top_T:      int, features to keep
        y:          np.ndarray [N], labels
        lap_K:      int, number of LapPE eigenvectors
        hks_T:      int, number of HKS time scales

    Returns:
        tilde_Theta: np.ndarray [N, top_T + lap_K + hks_T]
        meta:        dict
    """
    print(f"  Step 1/6: Computing sparse transition matrix...")
    P = compute_transition_matrix(edge_index, num_nodes)
    print(f"    P: sparse {P.shape}, nnz={P.nnz}")

    print(f"  Step 2/6: Computing sparse diffusion operator (K={K}, r={r})...")
    P_hat = compute_diffusion_operator(P, K=K, r=r)

    # --- Feature block ---
    print(f"  Step 3/6: Scaling raw features...")
    X_scaled, center, scale = robust_tanh_scale(X, train_mask)

    print(f"  Step 4/6: Angular transform on features...")
    Theta_feat = angular_transform(X_scaled)

    print(f"  Step 5/6: Diffusing feature angular coordinates...")
    tilde_Theta_feat = compute_diffused_angular(Theta_feat, P_hat)

    # --- Shared eigendecomposition for LapPE + HKS ---
    print(f"  Step 5b/6: Computing shared Laplacian eigendecomposition...")
    eigenvalues, eigenvectors = compute_laplacian_eig(
        edge_index, num_nodes, K=lap_K
    )

    # --- LapPE block ---
    print(f"  Computing LapPE from eigenvectors...")
    phi_sq  = eigenvectors[:, 1:lap_K+1] ** 2   # skip phi_0
    S_lap   = 2 * phi_sq - 1                      # [N, lap_K]
    S_lap   = np.clip(S_lap, -1+1e-7, 1-1e-7).astype(np.float32)
    print(f"  LapPE shape: {S_lap.shape}, range: [{S_lap.min():.3f}, {S_lap.max():.3f}]")
    Theta_lap        = angular_transform(S_lap)
    tilde_Theta_lap  = compute_diffused_angular(Theta_lap, P_hat)

    # --- HKS block (reuses same eigenvectors) ---
    print(f"  Computing HKS (T={hks_T} time scales)...")
    S_hks   = compute_hks(eigenvalues, eigenvectors, T=hks_T)
    Theta_hks        = angular_transform(S_hks)
    tilde_Theta_hks  = compute_diffused_angular(Theta_hks, P_hat)

    # --- Concatenate structural blocks ---
    tilde_Theta_struct = np.hstack([tilde_Theta_lap, tilde_Theta_hks])
    print(f"  Combined structural (LapPE+HKS): {tilde_Theta_struct.shape}")
    print(f"  LapPE dims: 0-{lap_K-1}  HKS dims: {lap_K}-{lap_K+hks_T-1}")

    # --- Prescreening and concatenation ---
    print(f"  Step 6/6: Prescreening and concatenation...")
    top_T_feat = top_T if top_T is not None else tilde_Theta_feat.shape[1]
    assert y is not None

    tilde_Theta, feat_indices, struct_start_idx = \
        prescreen_features_separate(
            tilde_Theta_feat, tilde_Theta_struct,
            y, train_mask, top_T_feat=top_T_feat
        )

    print(f"  Final tilde_Theta shape: {tilde_Theta.shape}")

    meta = {
        'P_hat':            P_hat,
        'center':           center,
        'scale':            scale,
        'feat_indices':     feat_indices,
        'struct_start_idx': struct_start_idx,
        'lap_K':            lap_K,
        'hks_T':            hks_T,
    }
    return tilde_Theta, meta


# ---------------------------------------------------------------------------
# 12. HKS + RWPE combined pipeline
#
# Replaces LapPE with HKS in the combined structural block.
# HKS is isometry invariant and multiscale — may generalise better
# than raw squared eigenvectors (LapPE) on some graphs.
#
# Final augmented vector: [top_T features | hks_T HKS | rw_K RWPE]
# ---------------------------------------------------------------------------

def build_diffused_angular_matrix_hks_rwpe(X, edge_index, num_nodes, train_mask,
                                            K=4, r=0.5, top_T=None, y=None,
                                            lap_K=8, hks_T=6, rw_K=12):
    """
    Full pipeline with HKS + RWPE combined structural block.

    Args:
        X:          np.ndarray [N, D]
        edge_index: torch.LongTensor [2, E]
        num_nodes:  int
        train_mask: np.ndarray bool [N]
        K:          int, diffusion hops
        r:          float, decay rate
        top_T:      int, features to keep
        y:          np.ndarray [N], labels
        lap_K:      int, eigenvectors for HKS computation
        hks_T:      int, HKS time scales
        rw_K:       int, RWPE steps

    Returns:
        tilde_Theta: np.ndarray [N, top_T + hks_T + rw_K]
        meta:        dict
    """
    print(f"  Step 1/6: Computing sparse transition matrix...")
    P = compute_transition_matrix(edge_index, num_nodes)
    print(f"    P: sparse {P.shape}, nnz={P.nnz}")

    print(f"  Step 2/6: Computing sparse diffusion operator (K={K}, r={r})...")
    P_hat = compute_diffusion_operator(P, K=K, r=r)

    # --- Feature block ---
    print(f"  Step 3/6: Scaling raw features...")
    X_scaled, center, scale = robust_tanh_scale(X, train_mask)

    print(f"  Step 4/6: Angular transform on features...")
    Theta_feat = angular_transform(X_scaled)

    print(f"  Step 5/6: Diffusing feature angular coordinates...")
    tilde_Theta_feat = compute_diffused_angular(Theta_feat, P_hat)

    # --- Shared eigendecomposition for HKS ---
    print(f"  Step 5b/6: Computing Laplacian eigendecomposition (lap_K={lap_K})...")
    eigenvalues, eigenvectors = compute_laplacian_eig(
        edge_index, num_nodes, K=lap_K
    )

    # --- HKS block ---
    print(f"  Computing HKS (T={hks_T} time scales)...")
    S_hks = compute_hks(eigenvalues, eigenvectors, T=hks_T)
    Theta_hks       = angular_transform(S_hks)
    tilde_Theta_hks = compute_diffused_angular(Theta_hks, P_hat)
    print(f"    tilde_Theta_hks shape: {tilde_Theta_hks.shape}")

    # --- RWPE block ---
    print(f"  Computing RWPE (rw_K={rw_K})...")
    S_rw = compute_rwpe(P, K_rw=rw_K)
    Theta_rw       = angular_transform(S_rw)
    tilde_Theta_rw = compute_diffused_angular(Theta_rw, P_hat)
    print(f"    tilde_Theta_rw shape: {tilde_Theta_rw.shape}")

    # --- Concatenate structural blocks ---
    tilde_Theta_struct = np.hstack([tilde_Theta_hks, tilde_Theta_rw])
    print(f"  Combined structural (HKS+RWPE): {tilde_Theta_struct.shape}")
    print(f"  HKS dims: 0-{hks_T-1}  RWPE dims: {hks_T}-{hks_T+rw_K-1}")

    # --- Prescreening and concatenation ---
    print(f"  Step 6/6: Prescreening and concatenation...")
    top_T_feat = top_T if top_T is not None else tilde_Theta_feat.shape[1]
    assert y is not None

    tilde_Theta, feat_indices, struct_start_idx = \
        prescreen_features_separate(
            tilde_Theta_feat, tilde_Theta_struct,
            y, train_mask, top_T_feat=top_T_feat
        )

    print(f"  Final tilde_Theta shape: {tilde_Theta.shape}")
    print(f"  Feature block: dims 0-{top_T_feat-1}")
    print(f"  HKS block:     dims {top_T_feat}-{top_T_feat+hks_T-1}")
    print(f"  RWPE block:    dims {top_T_feat+hks_T}-{top_T_feat+hks_T+rw_K-1}")

    meta = {
        'P_hat':            P_hat,
        'center':           center,
        'scale':            scale,
        'feat_indices':     feat_indices,
        'struct_start_idx': struct_start_idx,
        'hks_T':            hks_T,
        'rw_K':             rw_K,
    }
    return tilde_Theta, meta


# ---------------------------------------------------------------------------
# 13. Degree + Clustering Coefficient structural features
#
# Two simple per-node scalars:
#   degree(v)  = number of neighbours, normalised by max degree
#   cc(v)      = fraction of v's neighbours that are connected to each other
#                = 2 * triangles(v) / (d_v * (d_v - 1))
#
# Fast: O(M) to compute. No eigendecomposition.
# Often surprisingly informative — degree distinguishes hubs from leaves,
# clustering coefficient distinguishes clique nodes from bridge nodes.
# ---------------------------------------------------------------------------

def compute_degree_clustering(edge_index, num_nodes):
    """
    Compute normalised degree and clustering coefficient per node.

    Args:
        edge_index: torch.LongTensor [2, E]
        num_nodes:  int

    Returns:
        S: np.ndarray [N, 2], values in [-1, 1] ready for arccos
           col 0: normalised degree mapped to [-1, 1]
           col 1: clustering coefficient mapped to [-1, 1]
    """
    row  = edge_index[0].numpy()
    col  = edge_index[1].numpy()
    data = np.ones(len(row), dtype=np.float32)

    A      = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A      = ((A + A.T) > 0).astype(np.float32)  # symmetrise, binary
    degree = np.array(A.sum(axis=1)).flatten()    # [N]

    # --- Normalised degree ---
    max_deg = degree.max()
    max_deg = max_deg if max_deg > 0 else 1.0
    deg_norm = degree / max_deg                    # [N], in [0, 1]

    # --- Clustering coefficient ---
    # cc(v) = (number of edges among neighbours of v) / (d_v * (d_v-1) / 2)
    # Efficiently: A^2[v,v] = number of paths of length 2 from v to v
    #            = number of triangles * 2 (each triangle counted twice)
    # cc(v) = A^2[v,v] / (d_v * (d_v - 1))

    A2_diag = np.array((A @ A).diagonal()).flatten()   # triangles * 2
    denom   = degree * (degree - 1)
    cc      = np.zeros(num_nodes, dtype=np.float32)
    mask    = denom > 0
    cc[mask] = A2_diag[mask] / denom[mask]             # [N], in [0, 1]

    # Stack and map [0,1] -> [-1,1]
    S_raw = np.stack([deg_norm, cc], axis=1)            # [N, 2]
    S     = 2 * S_raw - 1                               # [N, 2], in [-1, 1]

    print(f"  Degree+CC shape: {S.shape}")
    print(f"  Degree range:  [{S[:,0].min():.3f}, {S[:,0].max():.3f}]")
    print(f"  CC range:      [{S[:,1].min():.3f}, {S[:,1].max():.3f}]")
    return S.astype(np.float32)


def build_diffused_angular_matrix_degcc(X, edge_index, num_nodes, train_mask,
                                         K=4, r=0.5, top_T=None, y=None):
    """
    Full pipeline with degree + clustering coefficient structural block.

    Args:
        X, edge_index, num_nodes, train_mask, K, r, top_T, y: as before

    Returns:
        tilde_Theta: np.ndarray [N, top_T + 2]
        meta:        dict
    """
    print(f"  Step 1/5: Computing sparse transition matrix...")
    P = compute_transition_matrix(edge_index, num_nodes)

    print(f"  Step 2/5: Computing sparse diffusion operator...")
    P_hat = compute_diffusion_operator(P, K=K, r=r)

    print(f"  Step 3/5: Scaling + angular transform on features...")
    X_scaled, center, scale = robust_tanh_scale(X, train_mask)
    Theta_feat       = angular_transform(X_scaled)
    tilde_Theta_feat = compute_diffused_angular(Theta_feat, P_hat)

    print(f"  Step 4/5: Computing degree + clustering coefficient...")
    S_dc             = compute_degree_clustering(edge_index, num_nodes)
    Theta_struct     = angular_transform(S_dc)
    tilde_Theta_struct = compute_diffused_angular(Theta_struct, P_hat)

    print(f"  Step 5/5: Prescreening and concatenation...")
    top_T_feat = top_T if top_T is not None else tilde_Theta_feat.shape[1]
    assert y is not None
    tilde_Theta, feat_indices, struct_start_idx = \
        prescreen_features_separate(
            tilde_Theta_feat, tilde_Theta_struct,
            y, train_mask, top_T_feat=top_T_feat
        )
    print(f"  Final tilde_Theta shape: {tilde_Theta.shape}")

    meta = {'P_hat': P_hat, 'center': center, 'scale': scale,
            'feat_indices': feat_indices, 'struct_start_idx': struct_start_idx}
    return tilde_Theta, meta


# ---------------------------------------------------------------------------
# 14. Personalised PageRank diffusion operator
#
# Instead of P_hat = sum_k r^k P^k (polynomial diffusion),
# use Personalised PageRank:
#     PPR = alpha * (I - (1-alpha) * P)^{-1}
#
# This is what APPNP uses. Key difference from polynomial diffusion:
# - Infinite-order approximation (not truncated at K hops)
# - Teleportation parameter alpha controls locality
# - alpha=0.15: 85% propagation, 15% teleport back to source
# - Better at preserving node identity while aggregating global context
#
# Computed via power iteration: PPR ≈ sum_{k=0}^{inf} (1-alpha)^k * alpha * P^k
# Truncated at convergence (tol=1e-5) or max_iter steps.
# ---------------------------------------------------------------------------

def compute_ppr_diffusion(P, alpha=0.15, max_iter=30, tol=1e-5):
    """
    Compute Personalised PageRank diffusion operator via power iteration.

    PPR = alpha * sum_{k=0}^{inf} (1-alpha)^k * P^k
        = alpha * (I - (1-alpha)*P)^{-1}

    Approximated by truncating the infinite sum when change < tol.
    Result is sparse — sparsified by zeroing entries < 1e-4.

    Args:
        P:        scipy.sparse.csr_matrix [N, N], row-stochastic
        alpha:    float, teleport probability (0.15 recommended)
        max_iter: int, maximum power iterations
        tol:      float, convergence tolerance

    Returns:
        PPR_hat: scipy.sparse.csr_matrix [N, N], diagonal zeroed
    """
    N       = P.shape[0]
    PPR     = sp.csr_matrix((N, N), dtype=np.float32)
    P_power = sp.eye(N, format='csr', dtype=np.float32)   # P^0 = I
    coeff   = alpha

    for k in range(max_iter):
        contribution = coeff * P_power
        PPR          = PPR + contribution

        # Check convergence — max absolute change
        max_change = abs(contribution).max()
        if max_change < tol:
            print(f"  PPR converged at iteration {k+1} (max_change={max_change:.2e})")
            break

        P_power = (P_power @ P).tocsr()
        coeff   = coeff * (1 - alpha)

    # Zero diagonal — node does not mix with itself
    PPR = PPR.tolil()
    PPR.setdiag(0)
    PPR = PPR.tocsr()

    # Sparsify — zero very small entries to keep memory manageable
    PPR.data[PPR.data < 1e-4] = 0
    PPR.eliminate_zeros()

    print(f"  PPR_hat: sparse {PPR.shape}, nnz={PPR.nnz}")
    return PPR


def build_diffused_angular_matrix_ppr(X, edge_index, num_nodes, train_mask,
                                       alpha=0.15, top_T=None, y=None,
                                       lap_K=8, use_lap_pe=True):
    """
    Full pipeline using Personalised PageRank diffusion instead of
    polynomial diffusion. Everything else identical to standard pipeline.

    Args:
        X, edge_index, num_nodes, train_mask, top_T, y: as before
        alpha:      float, PPR teleport probability
        lap_K:      int, LapPE eigenvectors
        use_lap_pe: bool, whether to add LapPE structural block

    Returns:
        tilde_Theta: np.ndarray [N, top_T + lap_K or top_T]
        meta:        dict
    """
    print(f"  Step 1/5: Computing sparse transition matrix...")
    P = compute_transition_matrix(edge_index, num_nodes)
    print(f"    P: sparse {P.shape}, nnz={P.nnz}")

    print(f"  Step 2/5: Computing PPR diffusion (alpha={alpha})...")
    P_hat = compute_ppr_diffusion(P, alpha=alpha)

    print(f"  Step 3/5: Scaling + angular transform on features...")
    X_scaled, center, scale = robust_tanh_scale(X, train_mask)
    Theta_feat       = angular_transform(X_scaled)
    tilde_Theta_feat = compute_diffused_angular(Theta_feat, P_hat)

    tilde_Theta_struct = None
    if use_lap_pe:
        print(f"  Step 4/5: Computing LapPE (lap_K={lap_K})...")
        S_lap            = compute_laplacian_pe(edge_index, num_nodes, K=lap_K)
        Theta_struct     = angular_transform(S_lap)
        tilde_Theta_struct = compute_diffused_angular(Theta_struct, P_hat)

    print(f"  Step 5/5: Prescreening and concatenation...")
    top_T_feat = top_T if top_T is not None else tilde_Theta_feat.shape[1]
    assert y is not None

    if use_lap_pe and tilde_Theta_struct is not None:
        tilde_Theta, feat_indices, struct_start_idx = \
            prescreen_features_separate(
                tilde_Theta_feat, tilde_Theta_struct,
                y, train_mask, top_T_feat=top_T_feat
            )
    else:
        T_train = tilde_Theta_feat[train_mask]
        y_train = y[train_mask]
        num_classes = len(np.unique(y_train))
        D = T_train.shape[1]
        scores = np.zeros(D)
        for c in range(num_classes):
            y_binary = (y_train == c).astype(np.float32)
            corrs    = np.abs(np.corrcoef(T_train.T, y_binary)[-1, :-1])
            scores   = np.maximum(scores, np.nan_to_num(corrs, nan=0.0))
        feat_indices = np.argsort(scores)[::-1][:top_T_feat]
        tilde_Theta  = tilde_Theta_feat[:, feat_indices]
        struct_start_idx = None
        print(f"  Prescreening: kept top {top_T_feat} features")

    print(f"  Final tilde_Theta shape: {tilde_Theta.shape}")

    meta = {'P_hat': P_hat, 'center': center, 'scale': scale,
            'feat_indices': feat_indices,
            'struct_start_idx': struct_start_idx,
            'alpha': alpha}
    return tilde_Theta, meta


# ---------------------------------------------------------------------------
# 15. PPR + RWPE combined pipeline
#
# Uses Personalised PageRank diffusion (better than polynomial on most graphs)
# with RWPE structural block (best for co-purchase graphs like Amazon Photo).
# ---------------------------------------------------------------------------

def build_diffused_angular_matrix_ppr_rwpe(X, edge_index, num_nodes, train_mask,
                                            alpha=0.15, top_T=None, y=None,
                                            rw_K=12):
    """
    PPR diffusion + RWPE structural block.

    Args:
        X, edge_index, num_nodes, train_mask, top_T, y: as before
        alpha: float, PPR teleport probability
        rw_K:  int, RWPE steps

    Returns:
        tilde_Theta: np.ndarray [N, top_T + rw_K]
        meta:        dict
    """
    print(f"  Step 1/5: Computing sparse transition matrix...")
    P = compute_transition_matrix(edge_index, num_nodes)

    print(f"  Step 2/5: Computing PPR diffusion (alpha={alpha})...")
    P_hat = compute_ppr_diffusion(P, alpha=alpha)

    print(f"  Step 3/5: Scaling + angular transform on features...")
    X_scaled, center, scale = robust_tanh_scale(X, train_mask)
    Theta_feat       = angular_transform(X_scaled)
    tilde_Theta_feat = compute_diffused_angular(Theta_feat, P_hat)

    print(f"  Step 4/5: Computing RWPE (rw_K={rw_K})...")
    S_rw             = compute_rwpe(P, K_rw=rw_K)
    Theta_struct     = angular_transform(S_rw)
    tilde_Theta_struct = compute_diffused_angular(Theta_struct, P_hat)

    print(f"  Step 5/5: Prescreening and concatenation...")
    top_T_feat = top_T if top_T is not None else tilde_Theta_feat.shape[1]
    assert y is not None
    tilde_Theta, feat_indices, struct_start_idx = \
        prescreen_features_separate(
            tilde_Theta_feat, tilde_Theta_struct,
            y, train_mask, top_T_feat=top_T_feat
        )
    print(f"  Final tilde_Theta shape: {tilde_Theta.shape}")

    meta = {'P_hat': P_hat, 'center': center, 'scale': scale,
            'feat_indices': feat_indices,
            'struct_start_idx': struct_start_idx,
            'alpha': alpha, 'rw_K': rw_K}
    return tilde_Theta, meta


# ---------------------------------------------------------------------------
# 16. PPR + LapPE + RWPE — full kitchen sink
# ---------------------------------------------------------------------------

def build_diffused_angular_matrix_ppr_combined(X, edge_index, num_nodes,
                                                train_mask, alpha=0.15,
                                                top_T=None, y=None,
                                                lap_K=8, rw_K=12):
    """
    PPR diffusion + LapPE + RWPE structural block.
    Maximum structural information.

    Returns:
        tilde_Theta: np.ndarray [N, top_T + lap_K + rw_K]
    """
    print(f"  Step 1/5: Computing sparse transition matrix...")
    P = compute_transition_matrix(edge_index, num_nodes)

    print(f"  Step 2/5: Computing PPR diffusion (alpha={alpha})...")
    P_hat = compute_ppr_diffusion(P, alpha=alpha)

    print(f"  Step 3/5: Scaling + angular transform on features...")
    X_scaled, center, scale = robust_tanh_scale(X, train_mask)
    Theta_feat       = angular_transform(X_scaled)
    tilde_Theta_feat = compute_diffused_angular(Theta_feat, P_hat)

    print(f"  Step 4a/5: Computing LapPE (lap_K={lap_K})...")
    S_lap              = compute_laplacian_pe(edge_index, num_nodes, K=lap_K)
    Theta_lap          = angular_transform(S_lap)
    tilde_Theta_lap    = compute_diffused_angular(Theta_lap, P_hat)

    print(f"  Step 4b/5: Computing RWPE (rw_K={rw_K})...")
    S_rw               = compute_rwpe(P, K_rw=rw_K)
    Theta_rw           = angular_transform(S_rw)
    tilde_Theta_rw     = compute_diffused_angular(Theta_rw, P_hat)

    tilde_Theta_struct = np.hstack([tilde_Theta_lap, tilde_Theta_rw])
    print(f"  Combined structural: {tilde_Theta_struct.shape}")

    print(f"  Step 5/5: Prescreening and concatenation...")
    top_T_feat = top_T if top_T is not None else tilde_Theta_feat.shape[1]
    assert y is not None
    tilde_Theta, feat_indices, struct_start_idx = \
        prescreen_features_separate(
            tilde_Theta_feat, tilde_Theta_struct,
            y, train_mask, top_T_feat=top_T_feat
        )
    print(f"  Final tilde_Theta shape: {tilde_Theta.shape}")

    meta = {'P_hat': P_hat, 'center': center, 'scale': scale,
            'feat_indices': feat_indices,
            'struct_start_idx': struct_start_idx,
            'alpha': alpha, 'lap_K': lap_K, 'rw_K': rw_K}
    return tilde_Theta, meta


# ---------------------------------------------------------------------------
# 17. Multi-Layer Angular Diffusion
#
# Applies the angular transform iteratively, alternating with diffusion:
#     Theta^(0) = arccos(tanh_scale(X))           -- layer 0
#     Theta^(1) = arccos(tanh_scale(P_hat @ Theta^(0)))  -- layer 1
#     Theta^(2) = arccos(tanh_scale(P_hat @ Theta^(1)))  -- layer 2
#
# Each layer applies diffusion then the nonlinear arccos transform.
# This creates cross-hop nonlinear interactions analogous to deep GNNs.
#
# Final representation: concatenate all layers (Jumping Knowledge):
#     tilde_Theta = [Theta^(0) | Theta^(1) | Theta^(2)]
#
# Why this is more expressive than one-layer:
#     Theta^(2) contains terms like arccos(sum_j P_hat_ij * arccos(...))
#     which are nonlinear functions of 2-hop neighbourhood -- impossible
#     to express with one diffusion step.
#
# Connection to GCN:
#     GCN layer: H^(l+1) = ReLU(A_hat H^(l) W^(l))
#     MLAD layer: Theta^(l+1) = arccos(tanh_scale(P_hat Theta^(l)))
#     Both: diffuse then apply nonlinearity. MLAD uses arccos instead of
#     ReLU and tanh_scale instead of learned W -- closed-form, no backprop.
# ---------------------------------------------------------------------------

def multi_layer_angular_diffusion(X, P_hat, train_mask, num_layers=2,
                                    residual_alpha=0.5):
    """
    Apply iterative angular diffusion with residual connections.

    Each layer: Theta^(l+1) = arccos(tanh_scale(
                    alpha * P_hat @ Theta^(l) + (1-alpha) * Theta^(0)))

    The residual term (1-alpha)*Theta^(0) preserves original node features
    at every layer, preventing over-smoothing from repeated PPR diffusion.
    With alpha=0.5, each layer blends diffused and original equally.

    Args:
        X:              np.ndarray [N, D], raw node features
        P_hat:          scipy.sparse [N, N], diffusion operator
        train_mask:     np.ndarray bool [N]
        num_layers:     int, number of diffusion layers (default 2)
        residual_alpha: float, weight on diffused component (default 0.5)

    Returns:
        representations: list of np.ndarray [N, D], one per layer
    """
    # Layer 0: raw angular coordinates
    X_scaled, center, scale = robust_tanh_scale(X, train_mask)
    Theta_0 = angular_transform(X_scaled)   # anchor for residual
    representations = [Theta_0]
    Theta = Theta_0

    for l in range(num_layers):
        # Diffuse current angular coordinates
        diffused = (P_hat @ Theta).astype(np.float32)   # [N, D]

        # Residual blend: mix diffused with original
        blended = residual_alpha * diffused + (1 - residual_alpha) * Theta_0

        # Re-fit tanh scale and apply arccos
        blended_scaled, _, _ = robust_tanh_scale(blended, train_mask)
        Theta = angular_transform(blended_scaled)        # [N, D]
        representations.append(Theta)

        print(f"  Layer {l+1}: shape={Theta.shape}, "
              f"range=[{Theta.min():.3f}, {Theta.max():.3f}]")

    return representations


def build_diffused_angular_matrix_multilayer(X, edge_index, num_nodes,
                                              train_mask,
                                              num_layers=2,
                                              use_ppr=True,
                                              alpha=0.15,
                                              K=4, r=0.5,
                                              top_T=None, y=None,
                                              lap_K=8, use_lap_pe=True):
    """
    Full pipeline with multi-layer angular diffusion.

    Uses Jumping Knowledge (JK) concatenation of all layer representations.
    Prescreens each layer's features separately then concatenates.

    Args:
        X:          np.ndarray [N, D]
        edge_index: torch.LongTensor [2, E]
        num_nodes:  int
        train_mask: np.ndarray bool [N]
        num_layers: int, number of diffusion layers (1, 2, or 3)
        use_ppr:    bool, use PPR diffusion (True) or polynomial (False)
        alpha:      float, PPR teleport probability
        K, r:       polynomial diffusion params (if use_ppr=False)
        top_T:      int, features to keep per layer from feature block
        y:          np.ndarray [N], labels
        lap_K:      int, LapPE eigenvectors
        use_lap_pe: bool, whether to add LapPE structural block

    Returns:
        tilde_Theta: np.ndarray [N, top_T*(num_layers+1) + lap_K]
        meta:        dict
    """
    print(f"\n  Multi-Layer Angular Diffusion "
          f"(num_layers={num_layers}, use_ppr={use_ppr})")

    print(f"  Step 1: Computing transition matrix...")
    P = compute_transition_matrix(edge_index, num_nodes)

    print(f"  Step 2: Computing diffusion operator...")
    if use_ppr:
        P_hat = compute_ppr_diffusion(P, alpha=alpha)
    else:
        P_hat = compute_diffusion_operator(P, K=K, r=r)

    # --- Multi-layer angular diffusion ---
    print(f"  Step 3: Multi-layer angular diffusion...")
    representations = multi_layer_angular_diffusion(
        X, P_hat, train_mask, num_layers=num_layers)
    # representations[l] has shape [N, D] for each layer l

    # --- Prescreen each layer separately and concatenate ---
    print(f"  Step 4: Prescreening each layer...")

    if top_T is None:
        top_T = min(80, X.shape[1])

    # Reduce top_T per layer to maintain fixed total budget.
    # With num_layers+1 layers, use top_T//(num_layers+1) per layer
    # so total dims ~ top_T (same as single layer).
    # This prevents overfitting from too many dimensions.
    num_total_layers = len(representations)  # num_layers + 1
    top_T_per_layer  = max(20, top_T // num_total_layers)
    print(f"  top_T per layer: {top_T_per_layer} "
          f"(total budget {top_T}, {num_total_layers} layers)")

    assert y is not None, "y required for prescreening"
    y_train  = y[train_mask]
    num_cls  = len(np.unique(y_train))

    def prescreen_layer(Theta_l, top_k):
        """Keep top_k features from layer Theta_l by OvR correlation."""
        T_train = Theta_l[train_mask]
        D       = T_train.shape[1]
        scores  = np.zeros(D)
        for c in range(num_cls):
            yb   = (y_train == c).astype(np.float32)
            corr = np.abs(np.corrcoef(T_train.T, yb)[-1, :-1])
            scores = np.maximum(scores, np.nan_to_num(corr, nan=0.0))
        top_k = min(top_k, D)
        idx   = np.argsort(scores)[::-1][:top_k]
        return Theta_l[:, idx], idx

    # Prescreen each layer
    layer_selected = []
    for l, Theta_l in enumerate(representations):
        Theta_sel, idx_sel = prescreen_layer(Theta_l, top_T_per_layer)
        layer_selected.append(Theta_sel)
        print(f"    Layer {l}: kept top {top_T_per_layer} features "
              f"from {Theta_l.shape[1]} dims")

    # Concatenate all layers (Jumping Knowledge)
    tilde_Theta_feat = np.hstack(layer_selected)
    print(f"  JK concatenated shape: {tilde_Theta_feat.shape}")

    # --- Optional LapPE structural block ---
    if use_lap_pe:
        print(f"  Step 5: Computing LapPE (lap_K={lap_K})...")
        S_lap   = compute_laplacian_pe(edge_index, num_nodes, K=lap_K)
        # Diffuse structural coords through the same operator
        Theta_s = angular_transform(S_lap)
        tilde_Theta_struct = compute_diffused_angular(Theta_s, P_hat)

        # Concatenate feature JK + structural
        tilde_Theta = np.hstack([tilde_Theta_feat, tilde_Theta_struct])
        struct_start_idx = tilde_Theta_feat.shape[1]
        print(f"  Final shape (with LapPE): {tilde_Theta.shape}")
        print(f"  Feature JK block: dims 0-{struct_start_idx-1}")
        print(f"  Structural block: dims {struct_start_idx}-"
              f"{struct_start_idx+lap_K-1}")
    else:
        tilde_Theta      = tilde_Theta_feat
        struct_start_idx = None
        print(f"  Final shape (no LapPE): {tilde_Theta.shape}")

    meta = {
        'P_hat':            P_hat,
        'struct_start_idx': struct_start_idx,
        'num_layers':       num_layers,
        'use_ppr':          use_ppr,
    }
    return tilde_Theta, meta


# ---------------------------------------------------------------------------
# 18. Kernel matrix builder for Kernel-SPR
#
# Computes the arc-cosine kernel over PPR-diffused angular coordinates
# using ALL features — no prescreening, no information loss.
# ---------------------------------------------------------------------------

def build_kernel_matrix_ppr(X, edge_index, num_nodes, train_mask,
                              alpha_ppr=0.15, kernel_degree=1,
                              use_lappe=True, lap_K=8):
    """
    Build arc-cosine kernel matrix for Kernel-SPR.

    Pipeline:
        1. PPR diffusion operator
        2. tanh_scale(X) on ALL D features (no prescreening)
        3. arccos transform -> tilde_Theta [N, D]
        4. Optionally append LapPE [N, D+lap_K]
        5. Arc-cosine kernel K[N, N]

    Args:
        X:            np.ndarray [N, D], raw features
        edge_index:   torch.LongTensor [2, E]
        num_nodes:    int
        train_mask:   np.ndarray bool [N]
        alpha_ppr:    float, PPR teleport probability
        kernel_degree: int, arc-cosine kernel degree (0, 1, or 2)
        use_lappe:    bool, whether to append LapPE features to kernel input
        lap_K:        int, number of LapPE eigenvectors

    Returns:
        K:           np.ndarray [N, N], arc-cosine kernel matrix
        tilde_Theta: np.ndarray [N, D or D+lap_K], feature matrix
                     (stored for interpretability analysis)
    """
    from kernel_spr import arc_cosine_kernel

    print(f"\n  Kernel-SPR pipeline "
          f"(alpha_ppr={alpha_ppr}, degree={kernel_degree}, "
          f"use_lappe={use_lappe})")
    print(f"  Input: {num_nodes} nodes, {X.shape[1]} features (ALL — no prescreening)")

    # Step 1: PPR diffusion
    print(f"  Step 1: PPR diffusion (alpha={alpha_ppr})...")
    P     = compute_transition_matrix(edge_index, num_nodes)
    P_hat = compute_ppr_diffusion(P, alpha=alpha_ppr)

    # Step 2-3: Scale + angular transform on ALL features
    print(f"  Step 2: Scaling + angular transform (D={X.shape[1]})...")
    X_scaled, _, _ = robust_tanh_scale(X, train_mask)
    Theta          = angular_transform(X_scaled)

    # Step 4: PPR diffusion of angular coordinates
    print(f"  Step 3: Diffusing angular coordinates...")
    tilde_Theta = compute_diffused_angular(Theta, P_hat)  # [N, D]

    # Step 5: Optionally append LapPE
    if use_lappe:
        print(f"  Step 4: Adding LapPE (lap_K={lap_K})...")
        S_lap            = compute_laplacian_pe(edge_index, num_nodes, K=lap_K)
        Theta_struct     = angular_transform(S_lap)
        tilde_struct     = compute_diffused_angular(Theta_struct, P_hat)
        tilde_Theta      = np.hstack([tilde_Theta, tilde_struct])
        print(f"  tilde_Theta with LapPE: {tilde_Theta.shape}")

    # Step 6: Compute arc-cosine kernel
    print(f"  Step 5: Arc-cosine kernel...")
    K = arc_cosine_kernel(tilde_Theta, degree=kernel_degree)

    return K, tilde_Theta


# ---------------------------------------------------------------------------
# 19. Label Propagation Features
#
# Runs label propagation (Zhou et al. 2004) using only the graph structure
# and training labels — no node features. The resulting [N, C] soft label
# predictions encode community membership purely from graph topology.
#
# These are concatenated to tilde_Theta before the arc-cosine kernel,
# giving the model explicit label information beyond PPR-diffused features.
#
# Why this closes the Cora gap:
#   - PPR-diffused features tell the model "this node uses word X a lot"
#   - LP features tell the model "this node is probably class Y based on
#     its graph neighbourhood" — direct community information
#   - Together they are complementary: feature similarity + graph community
#
# LP formula (iterative):
#   F^(t+1) = (1-alpha) * A_tilde @ F^(t) + alpha * Y
#   where Y[v] = one-hot(y[v]) for training nodes, 0 for test nodes
#   A_tilde = D^{-1/2} A D^{-1/2} (symmetric normalised adjacency)
#   alpha = restart probability (keeps predictions close to training labels)
# ---------------------------------------------------------------------------

def label_propagation(edge_index, num_nodes, y, train_mask,
                      alpha=0.1, num_steps=50):
    """
    Label propagation using symmetric normalised adjacency.

    Args:
        edge_index: torch.LongTensor [2, E]
        num_nodes:  int
        y:          np.ndarray [N], integer class labels
        train_mask: np.ndarray bool [N]
        alpha:      float, clamping factor (default 0.1)
                    small alpha = more propagation from graph
                    large alpha = stay closer to training labels
        num_steps:  int, propagation steps (default 50)

    Returns:
        F: np.ndarray [N, C], soft label predictions in [0, 1]
           Row-normalised so each row sums to 1.
    """
    C = int(y.max()) + 1
    N = num_nodes

    # Build symmetric normalised adjacency (with self-loops)
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    data = np.ones(len(row), dtype=np.float32)
    A = sp.csr_matrix((data, (row, col)), shape=(N, N))
    A = A + A.T
    A.data = np.clip(A.data, 0, 1)
    A = A + sp.eye(N, format='csr', dtype=np.float32)
    deg = np.asarray(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)
    D_inv_sqrt = sp.diags(d_inv_sqrt, format='csr')
    A_tilde = D_inv_sqrt @ A @ D_inv_sqrt   # [N, N] sparse

    # One-hot label matrix — training nodes only
    Y = np.zeros((N, C), dtype=np.float32)
    Y[train_mask] = np.eye(C)[y[train_mask]]

    # Iterative propagation: F <- (1-alpha)*A_tilde@F + alpha*Y
    F = Y.copy()
    for _ in range(num_steps):
        F = (1.0 - alpha) * (A_tilde @ F) + alpha * Y

    # Row-normalise to get soft probabilities
    row_sums = F.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    F = F / row_sums   # [N, C], rows sum to 1

    print(f"  LP features: shape={F.shape}, "
          f"train_acc={float(np.mean(F[train_mask].argmax(1) == y[train_mask])):.4f}")
    return F.astype(np.float32)


def build_kernel_matrix_ppr_with_lp(X, edge_index, num_nodes, train_mask, y,
                                      alpha_ppr=0.15, kernel_degree=1,
                                      use_lappe=True, lap_K=8,
                                      lp_alpha=0.1, lp_steps=50,
                                      lp_scale=1.0):
    """
    Kernel-SPR with Label Propagation features appended.

    Pipeline:
        1. PPR-diffused angular coordinates of ALL node features [N, D]
        2. Optional LapPE structural dims [N, D+lap_K]
        3. Label propagation soft predictions [N, C]
        4. Angular transform LP predictions -> [N, C] in [0, pi]
        5. Concatenate: [N, D+lap_K+C]
        6. Arc-cosine kernel [N, N]

    The LP features give the model direct community membership information
    from graph structure, complementing the PPR-diffused word features.

    Args:
        X:          np.ndarray [N, D], raw node features
        edge_index: torch.LongTensor [2, E]
        num_nodes:  int
        train_mask: np.ndarray bool [N]
        y:          np.ndarray [N], integer labels
        alpha_ppr:  float, PPR teleport probability
        kernel_degree: int, arc-cosine degree
        use_lappe:  bool, whether to add LapPE
        lap_K:      int, LapPE eigenvectors
        lp_alpha:   float, LP clamping factor
        lp_steps:   int, LP propagation steps
        lp_scale:   float, scale LP features before concatenation

    Returns:
        K:           np.ndarray [N, N], arc-cosine kernel matrix
        tilde_Theta: np.ndarray [N, D+lap_K+C], full feature matrix
    """
    from kernel_spr import arc_cosine_kernel
    import scipy.sparse as sp

    print(f"\n  Kernel-SPR + LP features pipeline")
    print(f"  Input: {num_nodes} nodes, {X.shape[1]} features + LP")

    # Step 1: PPR diffusion
    print(f"  Step 1: PPR diffusion (alpha={alpha_ppr})...")
    P     = compute_transition_matrix(edge_index, num_nodes)
    P_hat = compute_ppr_diffusion(P, alpha=alpha_ppr)

    # Step 2: Angular transform on ALL features
    print(f"  Step 2: Scaling + angular transform...")
    X_scaled, _, _ = robust_tanh_scale(X, train_mask)
    Theta           = angular_transform(X_scaled)
    tilde_Theta     = compute_diffused_angular(Theta, P_hat)  # [N, D]

    # Step 3: Optional LapPE
    if use_lappe:
        print(f"  Step 3: LapPE (lap_K={lap_K})...")
        S_lap        = compute_laplacian_pe(edge_index, num_nodes, K=lap_K)
        Theta_struct = angular_transform(S_lap)
        tilde_struct = compute_diffused_angular(Theta_struct, P_hat)
        tilde_Theta  = np.hstack([tilde_Theta, tilde_struct])

    # Step 4: Label propagation features
    print(f"  Step 4: Label propagation features "
          f"(alpha={lp_alpha}, steps={lp_steps})...")
    lp_feats = label_propagation(
        edge_index, num_nodes, y, train_mask,
        alpha=lp_alpha, num_steps=lp_steps
    )                                           # [N, C], soft probabilities

    # Angular transform LP features (map [0,1] -> arccos -> [0, pi])
    # Scale by lp_scale to control LP influence vs feature influence
    lp_scaled  = lp_feats * 2.0 - 1.0         # map [0,1] -> [-1,1]
    lp_scaled  = np.clip(lp_scaled, -1+1e-7, 1-1e-7)
    lp_angular = np.arccos(lp_scaled).astype(np.float32)  # [N, C]
    lp_angular = lp_angular * lp_scale

    # Concatenate
    tilde_Theta = np.hstack([tilde_Theta, lp_angular])
    print(f"  Full feature matrix: {tilde_Theta.shape} "
          f"(feat+struct+LP={tilde_Theta.shape[1]})")

    # Step 5: Arc-cosine kernel
    print(f"  Step 5: Arc-cosine kernel...")
    K = arc_cosine_kernel(tilde_Theta, degree=kernel_degree)

    return K, tilde_Theta


# ---------------------------------------------------------------------------
# 20. Combined Kernel: K_feat + lambda * K_LP
#
# Instead of concatenating LP features into tilde_Theta (where they get
# diluted by 1441 other dims), compute a separate kernel on LP predictions
# and add it to the feature kernel with a tunable weight lambda.
#
# K_total = K_feat + lambda * K_LP
#
# K_LP on 7 dims has no concentration of measure — it genuinely separates
# nodes in different communities. K_feat captures feature similarity.
# Together they provide complementary information.
#
# Sum of positive semi-definite kernels is PSD — valid kernel ridge solve.
# ---------------------------------------------------------------------------

def build_combined_kernel(X, edge_index, num_nodes, train_mask, y,
                           alpha_ppr=0.15, use_lappe=True, lap_K=8,
                           lp_alpha=0.1, lp_steps=50,
                           lp_lambda=1.0):
    """
    Build K_total = K_feat + lp_lambda * K_LP.

    K_feat: arc-cosine kernel on PPR-diffused features + LapPE
    K_LP:   arc-cosine kernel on LP soft predictions alone

    Args:
        lp_lambda: float, weight on LP kernel relative to feature kernel

    Returns:
        K_total: np.ndarray [N, N]
        K_feat:  np.ndarray [N, N]  (feature kernel alone)
        K_lp:    np.ndarray [N, N]  (LP kernel alone)
    """
    from kernel_spr import arc_cosine_kernel
    import scipy.sparse as sp

    print(f"  Building combined kernel (lp_lambda={lp_lambda:.2f})...")

    # Build K_feat (PPR + LapPE, no LP)
    P     = compute_transition_matrix(edge_index, num_nodes)
    P_hat = compute_ppr_diffusion(P, alpha=alpha_ppr)
    X_sc, _, _ = robust_tanh_scale(X, train_mask)
    Theta       = angular_transform(X_sc)
    tT          = compute_diffused_angular(Theta, P_hat)

    if use_lappe:
        S_lap    = compute_laplacian_pe(edge_index, num_nodes, K=lap_K)
        Ts       = angular_transform(S_lap)
        tTs      = compute_diffused_angular(Ts, P_hat)
        tT       = np.hstack([tT, tTs])

    K_feat = arc_cosine_kernel(tT, degree=1)

    # Build K_LP (LP features only, no feature dims)
    lp_feats   = label_propagation(edge_index, num_nodes, y, train_mask,
                                    alpha=lp_alpha, num_steps=lp_steps)
    lp_scaled  = lp_feats * 2.0 - 1.0
    lp_scaled  = np.clip(lp_scaled, -1+1e-7, 1-1e-7)
    lp_angular = np.arccos(lp_scaled).astype(np.float32)

    K_lp = arc_cosine_kernel(lp_angular, degree=1)
    print(f"  K_feat range: [{K_feat.min():.3f}, {K_feat.max():.3f}]")
    print(f"  K_LP   range: [{K_lp.min():.3f},  {K_lp.max():.3f}]")

    K_total = K_feat + lp_lambda * K_lp
    return K_total, K_feat, K_lp
