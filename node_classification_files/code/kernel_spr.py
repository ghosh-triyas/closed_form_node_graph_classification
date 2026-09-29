"""
Kernel Spectral Path Regression (Kernel-SPR)

Replaces explicit Chebyshev path selection with an arc-cosine kernel
over PPR-diffused angular coordinates.

Key idea:
    Current model:  y = Phi @ beta,  Phi[i,q] = cos(m_q^T tilde_theta_i)
                    Selects Q paths explicitly via greedy/beam search

    Kernel model:   y = K @ alpha,   K[i,j] = k(tilde_theta_i, tilde_theta_j)
                    Implicitly uses ALL Chebyshev harmonics simultaneously

The arc-cosine kernel is:
    k(x,y) = (1/pi) * (sin(theta) + (pi-theta)*cos(theta))
    theta = arccos(x_norm · y_norm)

This is the dot product in the RKHS of a single-layer ReLU network at
infinite width (Neal 1996, Cho & Saul 2009). It corresponds to integrating
over ALL possible frequency vectors m with a natural decay for higher
frequencies — equivalent to infinite paths, all selected simultaneously.

Why this beats explicit path selection:
    - Uses ALL 1433 features (no prescreening, no information loss)
    - No greedy/beam search needed — avoids myopic selection
    - Kernel regularisation naturally controls model complexity
    - Still closed-form: alpha = (K_train + lambda*I)^{-1} y_train

Training cost: O(N^2 * D) for kernel + O(N_train^3) for solve
    Cora: 2708^2 * 1433 ~= 10B ops (seconds with numpy BLAS)
    Solve: 140^3 = 2.7M ops (microseconds)
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Arc-cosine kernel
# ---------------------------------------------------------------------------

def arc_cosine_kernel(Theta, degree=1):
    """
    Arc-cosine kernel of given degree over angular coordinate matrix.

    Degree 0: k(x,y) = 1 - theta/pi
    Degree 1: k(x,y) = (sin(theta) + (pi-theta)*cos(theta)) / pi
    Degree 2: k(x,y) = (3*sin(theta)*cos(theta) + (pi-theta)*(1+2*cos^2)) / (3*pi)

    Args:
        Theta:  np.ndarray [N, D], PPR-diffused angular coordinates
        degree: int, 0, 1, or 2 (default 1)

    Returns:
        K: np.ndarray [N, N], symmetric positive semi-definite kernel matrix
    """
    N = Theta.shape[0]
    print(f"  Computing arc-cosine kernel (N={N}, D={Theta.shape[1]})...")

    # Row-normalise for angle computation
    norms  = np.linalg.norm(Theta, axis=1, keepdims=True)
    norms  = np.maximum(norms, 1e-8)
    T_norm = Theta / norms                              # [N, D], unit rows

    # Gram matrix — batch matrix multiply, fast with BLAS
    G = (T_norm @ T_norm.T).astype(np.float64)         # [N, N], in [-1,1]
    G = np.clip(G, -1.0 + 1e-7, 1.0 - 1e-7)

    theta = np.arccos(G)                                # [N, N], in [0, pi]

    if degree == 0:
        K = 1.0 - theta / np.pi
    elif degree == 1:
        K = (np.sin(theta) + (np.pi - theta) * np.cos(theta)) / np.pi
    elif degree == 2:
        cos_t = np.cos(theta)
        K = (3 * np.sin(theta) * cos_t
             + (np.pi - theta) * (1 + 2 * cos_t ** 2)) / (3 * np.pi)
    else:
        raise ValueError(f"degree must be 0, 1, or 2, got {degree}")

    # Symmetrise for numerical stability
    K = 0.5 * (K + K.T)

    print(f"  K: shape={K.shape}, "
          f"range=[{K.min():.3f}, {K.max():.3f}], "
          f"diag_mean={np.diag(K).mean():.3f}")
    return K.astype(np.float64)


# ---------------------------------------------------------------------------
# 2. Kernel Ridge Regression (One-vs-Rest multiclass)
# ---------------------------------------------------------------------------

def rbf_kernel(Theta, sigma=None):
    """
    RBF (Gaussian) kernel over angular coordinates.

    K(u,v) = exp(-||u - v||^2 / (2 * sigma^2))

    Unlike arc-cosine, RBF uses actual Euclidean distances —
    no L2 normalisation step means no concentration of measure.
    Features retain magnitude information.

    sigma is selected as median pairwise distance if not provided.
    Sweep sigma on validation for best results.

    Args:
        Theta: np.ndarray [N, D]
        sigma: float or None (auto = median pairwise distance)

    Returns:
        K: np.ndarray [N, N]
    """
    N = Theta.shape[0]
    print(f"  Computing RBF kernel (N={N}, D={Theta.shape[1]})...")

    # Efficient squared distance via ||u-v||^2 = ||u||^2 + ||v||^2 - 2*u.v
    sq_norms = np.sum(Theta ** 2, axis=1)                     # [N]
    G        = Theta @ Theta.T                                 # [N, N]
    D2       = sq_norms[:, None] + sq_norms[None, :] - 2 * G  # [N, N]
    D2       = np.maximum(D2, 0.0)                             # numerical safety

    if sigma is None:
        # Median heuristic on a sample of distances
        sample = min(500, N)
        idx    = np.random.choice(N, sample, replace=False)
        d_sample = D2[np.ix_(idx, idx)]
        sigma  = float(np.sqrt(np.median(d_sample[d_sample > 0])))
        sigma  = max(sigma, 1e-8)
        print(f"  Auto sigma (median heuristic): {sigma:.4f}")

    K = np.exp(-D2 / (2.0 * sigma ** 2))
    print(f"  K: shape={K.shape}, "
          f"range=[{K.min():.3f}, {K.max():.3f}], "
          f"diag_mean={np.diag(K).mean():.3f}")
    return K.astype(np.float64), sigma


class KernelGraphSPR:
    """
    Kernel Spectral Path Regression — arc-cosine kernel + kernel ridge OvR.

    Unlike GraphSPR which selects Q explicit paths, KernelGraphSPR uses
    the entire angular feature space implicitly via the kernel.

    Usage:
        model = KernelGraphSPR()
        K = build_kernel_matrix_ppr(X, edge_index, ...)
        model.fit(K, y, train_mask, val_mask)
        acc = model.evaluate(K, y, test_mask)
    """

    def __init__(self, lambda_grid=None, verbose=True):
        self.lambda_grid = lambda_grid or [
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0
        ]
        self.verbose     = verbose
        self.alpha       = None   # dual weights [N_train, C]
        self.train_mask  = None
        self.best_lambda = None
        self.num_classes = None

    def fit(self, K, y, train_mask, val_mask, degree=None, gamma=0.5):
        """
        Fit kernel ridge OvR classifier.

        Selects lambda on validation set, then re-solves on full train set.

        Args:
            K:          np.ndarray [N, N], precomputed kernel matrix
            y:          np.ndarray [N], integer class labels 0..C-1
            train_mask: np.ndarray bool [N]
            val_mask:   np.ndarray bool [N]
            degree:     np.ndarray [N] or None, node degrees for adaptive reg
                        If provided, Lambda_uu = lambda * degree_u^{-gamma}
                        High-degree nodes get less regularisation (more trust).
            gamma:      float, degree exponent (default 0.5)
        """
        self.train_mask  = train_mask
        self.num_classes = len(np.unique(y[train_mask]))
        C       = self.num_classes
        N_train = train_mask.sum()

        K_tr   = K[np.ix_(train_mask, train_mask)]   # [N_tr, N_tr]
        K_val  = K[np.ix_(val_mask,   train_mask)]   # [N_val, N_tr]
        y_tr   = y[train_mask]
        y_val  = y[val_mask]

        # Build degree-based diagonal regulariser if degrees provided
        if degree is not None:
            deg_tr  = degree[train_mask].astype(np.float64)
            deg_tr  = np.maximum(deg_tr, 1.0)
            # Normalise: reg_diag sums to N_train (same total as scalar lambda)
            reg_raw = deg_tr ** (-gamma)
            reg_diag = reg_raw * (N_train / reg_raw.sum())
        else:
            reg_diag = np.ones(N_train)

        # --- Lambda selection on validation ---
        best_val = -np.inf
        best_lam = self.lambda_grid[0]

        for lam in self.lambda_grid:
            A    = K_tr + lam * np.diag(reg_diag)
            S    = np.zeros((val_mask.sum(), C))
            for c in range(C):
                yc    = (y_tr == c).astype(np.float64)
                alpha = np.linalg.solve(A, yc)
                S[:, c] = K_val @ alpha
            acc = np.mean(S.argmax(axis=1) == y_val)
            if acc > best_val:
                best_val = acc
                best_lam = lam

        if self.verbose:
            reg_type = "adaptive" if degree is not None else "scalar"
            print(f"  [KernelSPR] lambda={best_lam:.0e}  "
                  f"val_acc={best_val:.4f}  reg={reg_type}")

        self.best_lambda = best_lam

        # --- Final solve with best lambda ---
        A = K_tr + best_lam * np.diag(reg_diag)
        self.alpha = np.zeros((N_train, C))
        for c in range(C):
            yc = (y_tr == c).astype(np.float64)
            self.alpha[:, c] = np.linalg.solve(A, yc)

        return self

    def predict_scores(self, K, mask=None):
        """Raw scores [N_masked, C]."""
        if mask is not None:
            K_q = K[np.ix_(mask, self.train_mask)]
        else:
            K_q = K[:, self.train_mask]
        return K_q @ self.alpha                        # [N_masked, C]

    def predict(self, K, mask=None):
        return self.predict_scores(K, mask).argmax(axis=1)

    def evaluate(self, K, y, mask):
        preds = self.predict(K, mask)
        return float(np.mean(preds == y[mask]))
