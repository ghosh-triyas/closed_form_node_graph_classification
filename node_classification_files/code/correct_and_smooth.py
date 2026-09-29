"""
Correct and Smooth (C&S) — Huang et al., ICLR 2021
https://arxiv.org/abs/2010.13993

Post-processing for any base node classifier (including Kernel-SPR and
Graph-SPR) using two label propagation passes:

  Step 1 — CORRECT:
    Compute residual error on training nodes.
    Propagate error to all nodes via label spreading.
    Add scaled error to base predictions.

  Step 2 — SMOOTH:
    Propagate corrected predictions via label spreading.
    Blend with known training labels.

Both steps use the same label spreading operator:
    S = (1 - alpha) * (D^{-1/2} A D^{-1/2}) + alpha * I
    propagation: F <- (1-alpha) * A_tilde @ F + alpha * F_0

This is exactly label spreading (Zhou et al. 2004) applied iteratively.

Key insight (from paper): errors on connected nodes are positively
correlated (homophily). Spreading training errors corrects test errors.
Labels on connected nodes are similar — smoothing improves boundary nodes.

Usage:
    cs = CorrectAndSmooth(num_correct=50, num_smooth=50,
                          alpha_correct=0.5, alpha_smooth=0.5)
    final_preds = cs.fit_transform(
        base_scores,   # [N, C] raw scores from Kernel-SPR
        y,             # [N]   integer labels (-1 for unlabeled)
        edge_index,    # [2, E] graph edges
        num_nodes,
        train_mask,
    )
"""

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Normalised adjacency for label spreading
# ---------------------------------------------------------------------------

def build_normalised_adj(edge_index, num_nodes):
    """
    Build symmetric normalised adjacency A_tilde = D^{-1/2} A D^{-1/2}.
    Adds self-loops before normalisation (standard for label spreading).

    Returns scipy.sparse [N, N] float32.
    """
    # Convert edge_index to scipy sparse
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    data = np.ones(len(row), dtype=np.float32)
    A = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    # Symmetrise (in case of directed edges)
    A = A + A.T
    A.data = np.clip(A.data, 0, 1)   # binary after symmetrise

    # Add self-loops
    A = A + sp.eye(num_nodes, format='csr', dtype=np.float32)

    # D^{-1/2} A D^{-1/2} normalisation
    deg = np.asarray(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)
    D_inv_sqrt = sp.diags(d_inv_sqrt, format='csr')
    A_tilde = D_inv_sqrt @ A @ D_inv_sqrt

    return A_tilde


# ---------------------------------------------------------------------------
# Label spreading propagation
# ---------------------------------------------------------------------------

def label_spreading(F, A_tilde, alpha, num_steps):
    """
    Iterative label spreading:
        F^(t+1) = (1-alpha) * A_tilde @ F^(t) + alpha * F^(0)

    Args:
        F:         np.ndarray [N, C], initial feature matrix
        A_tilde:   scipy.sparse [N, N], normalised adjacency
        alpha:     float, restart probability (higher = stay closer to F^0)
        num_steps: int, number of propagation steps

    Returns:
        F_prop: np.ndarray [N, C], propagated features
    """
    F0 = F.copy()
    F_prop = F.copy()

    for _ in range(num_steps):
        F_prop = (1.0 - alpha) * (A_tilde @ F_prop) + alpha * F0

    return F_prop


# ---------------------------------------------------------------------------
# Correct and Smooth
# ---------------------------------------------------------------------------

class CorrectAndSmooth:
    """
    C&S post-processor for node classification.

    Parameters:
        num_correct:   int, label spreading steps for Correct (default 50)
        num_smooth:    int, label spreading steps for Smooth (default 50)
        alpha_correct: float, restart prob in Correct step (default 0.5)
        alpha_smooth:  float, restart prob in Smooth step (default 0.5)
        scale:         float, error scaling in Correct step (default 1.0)
                       The paper uses autoscale; we use a fixed scale for
                       simplicity and sweep it on val set.
        autoscale:     bool, auto-scale errors (default True, paper default)
    """

    def __init__(self, num_correct=50, num_smooth=50,
                 alpha_correct=0.5, alpha_smooth=0.5,
                 scale=1.0, autoscale=True):
        self.num_correct   = num_correct
        self.num_smooth    = num_smooth
        self.alpha_correct = alpha_correct
        self.alpha_smooth  = alpha_smooth
        self.scale         = scale
        self.autoscale     = autoscale

    def fit_transform(self, base_scores, y, edge_index,
                      num_nodes, train_mask, val_mask=None):
        """
        Apply Correct and Smooth to base scores.

        Args:
            base_scores: np.ndarray [N, C], raw scores from base model
                         (does NOT need to be softmaxed)
            y:           np.ndarray [N], integer class labels
            edge_index:  torch.LongTensor [2, E]
            num_nodes:   int
            train_mask:  np.ndarray bool [N]
            val_mask:    np.ndarray bool [N] or None

        Returns:
            final_preds: np.ndarray [N, C], post-processed scores (softmaxed)
        """
        C = base_scores.shape[1]

        # --- Softmax base scores ---
        # Shift for numerical stability
        S = base_scores - base_scores.max(axis=1, keepdims=True)
        S = np.exp(S)
        S = S / S.sum(axis=1, keepdims=True)            # [N, C], in (0,1)

        # --- Build normalised adjacency ---
        print("  [C&S] Building normalised adjacency...")
        A_tilde = build_normalised_adj(edge_index, num_nodes)

        # --- One-hot labels for training nodes ---
        Y = np.zeros((num_nodes, C), dtype=np.float32)
        Y[train_mask] = np.eye(C)[y[train_mask]]        # [N_train, C] one-hot

        # ---------------------------------------------------------------
        # Step 1: CORRECT
        # ---------------------------------------------------------------
        print(f"  [C&S] Correct step (alpha={self.alpha_correct}, "
              f"steps={self.num_correct})...")

        # Residual error on training nodes
        error = np.zeros((num_nodes, C), dtype=np.float32)
        error[train_mask] = Y[train_mask] - S[train_mask]  # [N_train, C]

        if self.autoscale:
            # Auto-scale: make error magnitudes comparable to prediction
            # magnitudes. Scale so that the mean absolute error on training
            # nodes equals the mean absolute prediction on training nodes.
            scale = float(
                np.abs(S[train_mask]).sum() /
                np.abs(error[train_mask]).sum()
            )
            scale = min(scale, 1000.0)   # cap to avoid blow-up
        else:
            scale = self.scale

        error_scaled = error * scale

        # Propagate error across graph
        error_prop = label_spreading(
            error_scaled, A_tilde,
            self.alpha_correct, self.num_correct
        )                                                # [N, C]

        # Add propagated error to base predictions
        S_corrected = S + error_prop                    # [N, C]

        # ---------------------------------------------------------------
        # Step 2: SMOOTH
        # ---------------------------------------------------------------
        print(f"  [C&S] Smooth step (alpha={self.alpha_smooth}, "
              f"steps={self.num_smooth})...")

        # Normalise corrected predictions to [0,1]
        # (clamp negatives first to avoid softmax issues)
        S_corrected = np.clip(S_corrected, 0, None)
        row_sums = S_corrected.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        S_norm = S_corrected / row_sums                 # [N, C], in [0,1]

        # Replace training node predictions with ground truth
        # (they are already known — lock them in before smoothing)
        S_with_labels = S_norm.copy()
        S_with_labels[train_mask] = Y[train_mask]

        # Smooth across graph
        S_smooth = label_spreading(
            S_with_labels, A_tilde,
            self.alpha_smooth, self.num_smooth
        )                                               # [N, C]

        return S_smooth

    def select_hyperparams(self, base_scores, y, edge_index,
                           num_nodes, train_mask, val_mask,
                           alpha_grid=None, steps_grid=None):
        """
        Sweep C&S hyperparameters on validation set.

        Tries all combinations of (alpha_correct, alpha_smooth, steps)
        and picks the best by val accuracy.

        Returns best (alpha_correct, alpha_smooth, num_steps, val_acc).
        """
        if alpha_grid is None:
            alpha_grid = [0.2, 0.5, 0.8]
        if steps_grid is None:
            steps_grid = [10, 50]

        y_val = y[val_mask]
        best_val = -1
        best_params = (0.5, 0.5, 50)

        for alpha_c in alpha_grid:
            for alpha_s in alpha_grid:
                for steps in steps_grid:
                    cs_tmp = CorrectAndSmooth(
                        num_correct=steps, num_smooth=steps,
                        alpha_correct=alpha_c, alpha_smooth=alpha_s,
                        autoscale=True
                    )
                    S = cs_tmp.fit_transform(
                        base_scores, y, edge_index,
                        num_nodes, train_mask, val_mask
                    )
                    preds = S.argmax(axis=1)
                    val_acc = float(np.mean(preds[val_mask] == y_val))

                    if val_acc > best_val:
                        best_val = val_acc
                        best_params = (alpha_c, alpha_s, steps)

        print(f"  [C&S] Best params: alpha_c={best_params[0]}, "
              f"alpha_s={best_params[1]}, steps={best_params[2]}, "
              f"val_acc={best_val:.4f}")
        return best_params + (best_val,)
