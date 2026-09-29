"""
Closed-form classifiers for Persistent Hodge Spectrum vectors.

Pipeline:
    X ∈ R^{N x D}  →  angular transform  →  arc-cosine kernel K ∈ R^{N x N}
                                           →  kernel ridge regression (closed-form)

Training is a single matrix solve:
    α = (K_train + λI)^{-1} y_train
    no backpropagation, no iterative optimisation.
"""

import numpy as np
from typing import Optional, Tuple


# ────────────────────────────────────────────────────────────────────
# Feature preprocessing
# ────────────────────────────────────────────────────────────────────

def robust_scale(X: np.ndarray,
                  train_mask: Optional[np.ndarray] = None
                  ) -> np.ndarray:
    """
    Robust standardisation using training set statistics.
    Clips outliers at 3 standard deviations.
    """
    if train_mask is None:
        train_mask = np.ones(X.shape[0], dtype=bool)

    mu  = X[train_mask].mean(axis=0)
    std = X[train_mask].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    X_scaled = (X - mu) / std
    X_scaled = np.clip(X_scaled, -3.0, 3.0)
    return X_scaled


def angular_transform(X: np.ndarray) -> np.ndarray:
    """
    Map scaled features to angular coordinates via arccos(tanh(x)).

    Input:  X ∈ R^{N x D}  (any real values)
    Output: Θ ∈ [0, π]^{N x D}
    """
    X_clipped = np.clip(X, -10.0, 10.0)
    return np.arccos(np.tanh(X_clipped)).astype(np.float32)


# ────────────────────────────────────────────────────────────────────
# Kernels
# ────────────────────────────────────────────────────────────────────

def arc_cosine_kernel(Theta: np.ndarray,
                       degree: int = 1) -> np.ndarray:
    """
    Arc-cosine kernel of given degree.

    For degree=1:
        k(u, v) = (1/π) * (sin θ + (π - θ) cos θ)
        where θ = arccos(clip(u·v / (||u|| ||v||), -1+ε, 1-ε))

    Corresponds to an infinite neural network with ReLU activations.
    """
    # Normalise rows to unit sphere
    norms = np.linalg.norm(Theta, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    Theta_n = Theta / norms

    # Cosine similarity
    cos_theta = np.clip(Theta_n @ Theta_n.T, -1.0 + 1e-7, 1.0 - 1e-7)
    theta     = np.arccos(cos_theta)

    if degree == 0:
        K = 1.0 - theta / np.pi
    elif degree == 1:
        K = (1.0 / np.pi) * (np.sin(theta) + (np.pi - theta) * cos_theta)
    else:
        raise ValueError(f"Only degree 0 or 1 supported, got {degree}")

    return K.astype(np.float64)


def rbf_kernel(X: np.ndarray,
                sigma: Optional[float] = None) -> np.ndarray:
    """
    RBF (Gaussian) kernel: K(u,v) = exp(-||u-v||^2 / (2σ^2))

    σ defaults to the median pairwise distance (median heuristic).
    """
    sq_norms = np.sum(X ** 2, axis=1)
    D2 = np.maximum(sq_norms[:, None] + sq_norms[None, :] - 2 * X @ X.T,
                    0.0)
    if sigma is None:
        sample = D2[np.triu_indices_from(D2, k=1)]
        sigma  = float(np.sqrt(np.median(sample[sample > 0])))
        sigma  = max(sigma, 1e-8)
    K = np.exp(-D2 / (2.0 * sigma ** 2))
    return K.astype(np.float64)


# ────────────────────────────────────────────────────────────────────
# Kernel ridge regression (closed-form)
# ────────────────────────────────────────────────────────────────────

class KernelRidgeClassifier:
    """
    One-vs-rest kernel ridge regression classifier.

    Training: α = (K_train + λI)^{-1} Y_train   [single matrix solve]
    Predict:  f(x) = K(x, X_train) @ α           [matrix-vector product]

    Completely closed-form. No iterative optimisation.
    """

    def __init__(self,
                  lambda_grid: Optional[list] = None,
                  kernel: str = 'arccos'):
        self.lambda_grid = lambda_grid or \
            [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
        self.kernel      = kernel
        self.alpha_      = None     # dual weights [N_train, C]
        self.best_lambda_ = None
        self.K_train_     = None
        self.y_train_     = None
        self.num_classes_ = None

    def _build_kernel(self, X: np.ndarray) -> np.ndarray:
        if self.kernel == 'arccos':
            Theta = angular_transform(X)
            return arc_cosine_kernel(Theta, degree=1)
        elif self.kernel == 'rbf':
            return rbf_kernel(X)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _build_kernel_cross(self, X_test: np.ndarray,
                              X_train: np.ndarray) -> np.ndarray:
        """Build cross-kernel K(X_test, X_train)."""
        if self.kernel == 'arccos':
            T_test  = angular_transform(X_test)
            T_train = angular_transform(X_train)

            norms_te = np.linalg.norm(T_test,  axis=1, keepdims=True)
            norms_tr = np.linalg.norm(T_train, axis=1, keepdims=True)
            norms_te = np.where(norms_te < 1e-10, 1.0, norms_te)
            norms_tr = np.where(norms_tr < 1e-10, 1.0, norms_tr)

            Tn_te = T_test  / norms_te
            Tn_tr = T_train / norms_tr

            cos_theta = np.clip(Tn_te @ Tn_tr.T,
                                -1.0 + 1e-7, 1.0 - 1e-7)
            theta = np.arccos(cos_theta)
            K_cross = (1.0 / np.pi) * (np.sin(theta) +
                                         (np.pi - theta) * cos_theta)
            return K_cross.astype(np.float64)

        elif self.kernel == 'rbf':
            # Use same sigma as training kernel
            raise NotImplementedError("RBF cross-kernel: store sigma at fit")
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray) -> 'KernelRidgeClassifier':
        """
        Fit by selecting λ on validation set. Single matrix solve.
        """
        self.X_train_ = X_train
        self.y_train_ = y_train
        self.num_classes_ = len(np.unique(y_train))
        C = self.num_classes_
        N = len(y_train)

        # Scale features using training set
        self.scaler_mu_  = X_train.mean(axis=0)
        self.scaler_std_ = X_train.std(axis=0)
        self.scaler_std_ = np.where(self.scaler_std_ < 1e-8, 1.0,
                                     self.scaler_std_)
        X_tr_sc = (X_train - self.scaler_mu_) / self.scaler_std_
        X_va_sc = (X_val   - self.scaler_mu_) / self.scaler_std_

        # Build kernels
        K_tr    = self._build_kernel(X_tr_sc)
        K_tr_va = self._build_kernel_cross(X_va_sc, X_tr_sc)

        # One-hot encode labels
        Y_tr = np.zeros((N, C), dtype=np.float64)
        for i, yi in enumerate(y_train): Y_tr[i, yi] = 1.0

        best_val, best_lam = -1.0, self.lambda_grid[0]
        best_alpha = None

        for lam in self.lambda_grid:
            A = K_tr + lam * np.eye(N)
            alpha = np.linalg.solve(A, Y_tr)       # [N, C]
            scores_val = K_tr_va @ alpha            # [N_val, C]
            preds_val  = scores_val.argmax(axis=1)
            acc = float(np.mean(preds_val == y_val))
            if acc > best_val:
                best_val, best_lam, best_alpha = acc, lam, alpha.copy()

        self.best_lambda_ = best_lam
        self.alpha_       = best_alpha
        self.X_train_sc_  = X_tr_sc

        print(f"  KRR: λ={best_lam:.0e}  val_acc={best_val:.4f}")
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels for X_test."""
        X_te_sc = (X_test - self.scaler_mu_) / self.scaler_std_
        K_cross = self._build_kernel_cross(X_te_sc, self.X_train_sc_)
        scores  = K_cross @ self.alpha_
        return scores.argmax(axis=1)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Return accuracy on (X_test, y_test)."""
        return float(np.mean(self.predict(X_test) == y_test))
