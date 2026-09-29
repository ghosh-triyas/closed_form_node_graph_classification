"""
Main experiment runner for Graph-SPR.

Diffusion modes (--mode):
    baseline    : no structural encoding, polynomial diffusion
    lappe       : LapPE structural block, polynomial diffusion
    rwpe        : RWPE structural block, polynomial diffusion
    combined    : LapPE+RWPE, polynomial diffusion
    hks         : HKS+LapPE, polynomial diffusion
    hks_rwpe    : HKS+RWPE, polynomial diffusion
    ppr         : PPR diffusion, no structural block  ← BEST on citation graphs
    ppr_lappe   : PPR diffusion + LapPE               ← BEST on Actor
    ppr_rwpe    : PPR diffusion + RWPE
    ppr_combined: PPR diffusion + LapPE + RWPE
    degcc       : degree + clustering coefficient

Path selection (--selection):
    greedy : original forward greedy (fast)
    beam   : guided beam search with smoothness/NC/crossover (new, better)

Usage:
    # Best setup for citation graphs
    python run_experiments.py --dataset cora --mode ppr --selection beam

    # Best setup for Actor
    python run_experiments.py --dataset actor --mode ppr_lappe --selection beam

    # Best setup for Amazon Photo
    python run_experiments.py --dataset amazon_photo --mode combined --selection beam

    # Quick baseline comparison
    python run_experiments.py --dataset cora --mode ppr --selection greedy
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datasets import load_dataset
from graph_utils import (build_diffused_angular_matrix,
                                build_diffused_angular_matrix_multilayer,
                                build_diffused_angular_matrix_rwpe,
                                build_diffused_angular_matrix_combined,
                                build_diffused_angular_matrix_hks,
                                build_diffused_angular_matrix_hks_rwpe,
                                build_diffused_angular_matrix_degcc,
                                build_diffused_angular_matrix_ppr,
                                build_diffused_angular_matrix_ppr_rwpe,
                                build_diffused_angular_matrix_ppr_combined)
from multiclass import MulticlassGraphSPR
from kernel_spr import KernelGraphSPR
from correct_and_smooth import CorrectAndSmooth
from graph_utils import (build_kernel_matrix_ppr,
                                build_kernel_matrix_ppr_with_lp,
                                build_combined_kernel)
from baselines import run_gcn, run_sgc, run_mlp, run_gat, run_gin, run_graphsage


# ---------------------------------------------------------------------------
# Dataset-specific defaults
# Best diffusion mode and structural encoding per dataset from experiments:
#   Cora:         ppr        (PPR alone, no structural block)
#   CiteSeer:     ppr        (PPR alone)
#   Amazon Photo: combined   (polynomial + LapPE + RWPE rw_K=12)
#   Actor:        ppr_lappe  (PPR + LapPE)
# ---------------------------------------------------------------------------
DATASET_DEFAULTS = {
    'cora':         {'top_T': 80, 'max_support': 1, 'max_paths': 50,
                     'lap_K': 8, 'rw_K': 8,  'best_mode': 'ppr'},
    'citeseer':     {'top_T': 80, 'max_support': 1, 'max_paths': 50,
                     'lap_K': 8, 'rw_K': 8,  'best_mode': 'ppr'},
    'amazon_photo': {'top_T': 80, 'max_support': 1, 'max_paths': 30,
                     'lap_K': 8, 'rw_K': 12, 'best_mode': 'combined'},
    'actor':        {'top_T': 60, 'max_support': 1, 'max_paths': 20,
                     'lap_K': 8, 'rw_K': 8,  'best_mode': 'ppr_lappe'},
}


def build_tilde_theta(mode, X, data, train_mask, top_T, y, lap_K, rw_K):
    """
    Build graph-diffused angular coordinate matrix for given mode.
    Returns (tilde_Theta, meta).
    """
    num_nodes = data.num_nodes
    ei        = data.edge_index

    if mode == 'rwpe':
        return build_diffused_angular_matrix_rwpe(
            X, ei, num_nodes, train_mask,
            K=4, r=0.5, top_T=top_T, y=y, rw_K=rw_K)

    elif mode == 'lappe':
        return build_diffused_angular_matrix(
            X, ei, num_nodes, train_mask,
            K=4, r=0.5, top_T=top_T, y=y,
            lap_K=lap_K, use_lap_pe=True)

    elif mode == 'combined':
        return build_diffused_angular_matrix_combined(
            X, ei, num_nodes, train_mask,
            K=4, r=0.5, top_T=top_T, y=y,
            lap_K=lap_K, rw_K=rw_K)

    elif mode == 'hks':
        return build_diffused_angular_matrix_hks(
            X, ei, num_nodes, train_mask,
            K=4, r=0.5, top_T=top_T, y=y,
            lap_K=lap_K, hks_T=6)

    elif mode == 'hks_rwpe':
        return build_diffused_angular_matrix_hks_rwpe(
            X, ei, num_nodes, train_mask,
            K=4, r=0.5, top_T=top_T, y=y,
            lap_K=lap_K, hks_T=6, rw_K=rw_K)

    elif mode == 'degcc':
        return build_diffused_angular_matrix_degcc(
            X, ei, num_nodes, train_mask,
            K=4, r=0.5, top_T=top_T, y=y)

    elif mode == 'ppr':
        return build_diffused_angular_matrix_ppr(
            X, ei, num_nodes, train_mask,
            alpha=0.15, top_T=top_T, y=y,
            lap_K=lap_K, use_lap_pe=False)

    elif mode == 'ppr_lappe':
        return build_diffused_angular_matrix_ppr(
            X, ei, num_nodes, train_mask,
            alpha=0.15, top_T=top_T, y=y,
            lap_K=lap_K, use_lap_pe=True)

    elif mode == 'ppr_rwpe':
        return build_diffused_angular_matrix_ppr_rwpe(
            X, ei, num_nodes, train_mask,
            alpha=0.15, top_T=top_T, y=y, rw_K=rw_K)

    elif mode == 'ppr_combined':
        return build_diffused_angular_matrix_ppr_combined(
            X, ei, num_nodes, train_mask,
            alpha=0.15, top_T=top_T, y=y,
            lap_K=lap_K, rw_K=rw_K)

    elif mode in ('multilayer', 'multilayer2', 'multilayer3'):
        num_layers = 3 if mode == 'multilayer3' else (2 if mode == 'multilayer2' else 2)
        return build_diffused_angular_matrix_multilayer(
            X, ei, num_nodes, train_mask,
            num_layers=num_layers, use_ppr=True, alpha=0.15,
            top_T=top_T, y=y, lap_K=lap_K, use_lap_pe=True)

    else:  # baseline
        return build_diffused_angular_matrix(
            X, ei, num_nodes, train_mask,
            K=4, r=0.5, top_T=top_T, y=y,
            lap_K=lap_K, use_lap_pe=False)


def run_graph_spr(dataset_name, mode='ppr', selection='beam',
                  K=4, r=0.5, max_paths=50,
                  max_support=1, max_freq=2, patience=5,
                  top_T=80, lap_K=8, rw_K=8,
                  beam_width=3, gamma=0.05, lam_nc=0.05,
                  crossover_interval=5,
                  verbose=True, seed=42):
    """
    Full pipeline for one dataset.

    Args:
        mode:       diffusion + structural encoding mode
        selection:  'beam' (guided) or 'greedy' (original)
        beam_width: B, number of parallel beams (only for selection='beam')
        gamma:      smoothness penalty weight
        lam_nc:     NC reward weight
    """
    # 1. Load data
    data, X, y, train_mask, val_mask, test_mask, num_classes = \
        load_dataset(dataset_name, split_seed=seed)

    # 2. Build diffused angular coordinates
    print(f"\nBuilding tilde_Theta (mode={mode})...")
    tilde_Theta, meta = build_tilde_theta(
        mode, X, data, train_mask, top_T, y, lap_K, rw_K)
    print(f"tilde_Theta shape: {tilde_Theta.shape}")

    struct_start_idx = meta.get('struct_start_idx', None)
    if struct_start_idx is not None:
        print(f"Structural block starts at dim {struct_start_idx}")

    # 3. Train Graph-SPR
    model = MulticlassGraphSPR(
        num_classes       = num_classes,
        K                 = K,
        r                 = r,
        max_paths         = max_paths,
        max_support       = max_support,
        max_freq          = max_freq,
        patience          = patience,
        mode              = selection,
        beam_width        = beam_width,
        gamma             = gamma,
        lam_nc            = lam_nc,
        crossover_interval= crossover_interval,
        verbose           = verbose,
    )
    model.fit(
        tilde_Theta, y, train_mask, val_mask,
        edge_index       = data.edge_index,
        struct_start_idx = struct_start_idx,
    )

    # 4. Evaluate
    train_acc = model.evaluate(tilde_Theta, y, train_mask)
    val_acc   = model.evaluate(tilde_Theta, y, val_mask)
    test_acc  = model.evaluate(tilde_Theta, y, test_mask)

    print(f"\nGraph-SPR ({mode}, {selection}) on {dataset_name}:")
    print(f"  Train: {train_acc:.4f}  Val: {val_acc:.4f}  "
          f"Test: {test_acc:.4f}")

    paths_per_class = [len(m.selected_paths) for m in model.models]
    print(f"  Paths per class: {paths_per_class}")

    # Show path type breakdown if structural block present
    if struct_start_idx is not None and selection == 'beam':
        from graph_spr import _path_type
        for c, m in enumerate(model.models):
            types = [_path_type(p, struct_start_idx)
                     for p in m.selected_paths]
            n_cross = types.count('CROSS')
            if n_cross > 0:
                print(f"  Class {c}: {n_cross} cross paths found ✓")

    return test_acc, model, meta


def run_cs_mlp(dataset_name, hidden=256, num_layers=3,
              dropout=0.5, epochs=300, lr=0.01, weight_decay=5e-4,
              sweep_params=True, seed=42):
    """
    MLP (no graph) + Correct & Smooth post-processing.
    Matches exactly the setup in Huang et al. 2021 (C&S paper)
    which achieves 87.61% on Cora.

    The key insight from the paper: use a base predictor that
    IGNORES graph structure, then let C&S use the graph for
    correction and smoothing. This gives C&S more residual
    signal to exploit than starting from a graph-aware model.
    """
    import torch
    import torch.nn.functional as F

    data, X, y, train_mask, val_mask, test_mask, num_classes = \
        load_dataset(dataset_name, split_seed=seed)

    N, D = X.shape
    C = num_classes

    # --- Row-normalize features (critical for MLP performance) ---
    # The C&S paper uses T.NormalizeFeatures() which row-normalises.
    # Without this, MLP on bag-of-words gets ~55%; with it ~72%.
    X_norm = X.copy().astype(np.float32)
    row_sums = X_norm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    X_norm = X_norm / row_sums

    # --- Build MLP (no BatchNorm — unstable with 140 training nodes) ---
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_dim = D
            for _ in range(num_layers - 1):
                layers += [torch.nn.Linear(in_dim, hidden),
                           torch.nn.ReLU(),
                           torch.nn.Dropout(dropout)]
                in_dim = hidden
            layers.append(torch.nn.Linear(in_dim, C))
            self.net = torch.nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    torch.manual_seed(seed)
    model = MLP()
    # weight_decay=0 as in C&S paper (L2 reg hurts with normalised features)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    X_t  = torch.tensor(X_norm, dtype=torch.float32)
    y_t  = torch.tensor(y, dtype=torch.long)
    tr   = torch.tensor(train_mask)
    va   = torch.tensor(val_mask)
    te   = torch.tensor(test_mask)

    best_val, best_acc, patience_ctr = 0, 0, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out  = model(X_t)
        loss = F.cross_entropy(out[tr], y_t[tr])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out_e = model(X_t)
            va_acc = (out_e[va].argmax(1) == y_t[va]).float().mean().item()
            te_acc = (out_e[te].argmax(1) == y_t[te]).float().mean().item()
        if va_acc > best_val:
            best_val, best_acc = va_acc, te_acc
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= 50:
                break

    model.eval()
    with torch.no_grad():
        scores = model(X_t).numpy()   # [N, C] raw logits

    print(f"  MLP base: val={best_val:.4f} test={best_acc:.4f}")

    # --- C&S post-processing ---
    print("\nApplying Correct & Smooth...")
    cs = CorrectAndSmooth(autoscale=True)
    if sweep_params:
        alpha_c, alpha_s, steps, bv = cs.select_hyperparams(
            scores, y, data.edge_index, data.num_nodes,
            train_mask, val_mask,
            alpha_grid=[0.2, 0.5, 0.8],
            steps_grid=[10, 50],
        )
        cs = CorrectAndSmooth(num_correct=steps, num_smooth=steps,
                              alpha_correct=alpha_c, alpha_smooth=alpha_s,
                              autoscale=True)

    final = cs.fit_transform(
        scores, y, data.edge_index, data.num_nodes, train_mask, val_mask
    )
    preds    = final.argmax(axis=1)
    test_acc = float(np.mean(preds[test_mask] == y[test_mask]))
    val_acc2 = float(np.mean(preds[val_mask]  == y[val_mask]))
    print(f"\nMLP + C&S on {dataset_name}:")
    print(f"  Val: {val_acc2:.4f}  Test: {test_acc:.4f}")
    return test_acc


def run_cs_kernel(dataset_name, alpha_ppr=0.15, use_lappe=True,
                 alpha_correct=0.5, alpha_smooth=0.5,
                 num_steps=50, sweep_params=True,
                 seed=42, verbose=True):
    """
    Kernel-SPR + Correct & Smooth post-processing.

    Pipeline:
        1. Build arc-cosine kernel over PPR-diffused angular coords
        2. Fit kernel ridge regression (Kernel-SPR)
        3. Get raw scores [N, C] from kernel model
        4. Apply C&S post-processing (Correct + Smooth)

    Returns test accuracy.
    """
    data, X, y, train_mask, val_mask, test_mask, num_classes = \
        load_dataset(dataset_name, split_seed=seed)

    # Step 1-2: Kernel-SPR
    print(f"\nBuilding kernel matrix...")
    K, tilde_Theta = build_kernel_matrix_ppr(
        X, data.edge_index, data.num_nodes, train_mask,
        alpha_ppr=alpha_ppr, use_lappe=use_lappe,
    )

    model = KernelGraphSPR(verbose=verbose)
    model.fit(K, y, train_mask, val_mask)

    # Step 3: Get raw scores for ALL nodes
    all_mask = np.ones(data.num_nodes, dtype=bool)
    scores = model.predict_scores(K, mask=all_mask)   # [N, C]

    # Step 4: C&S post-processing
    print("\nApplying Correct & Smooth...")
    cs = CorrectAndSmooth(autoscale=True)

    if sweep_params:
        print("  Sweeping C&S hyperparameters on val set...")
        alpha_c, alpha_s, steps, best_val = cs.select_hyperparams(
            scores, y, data.edge_index, data.num_nodes,
            train_mask, val_mask,
            alpha_grid=[0.2, 0.5, 0.8],
            steps_grid=[10, 50, 100],
        )
        cs = CorrectAndSmooth(
            num_correct=steps, num_smooth=steps,
            alpha_correct=alpha_c, alpha_smooth=alpha_s,
            autoscale=True
        )

    final_scores = cs.fit_transform(
        scores, y, data.edge_index, data.num_nodes,
        train_mask, val_mask
    )

    # Evaluate
    preds = final_scores.argmax(axis=1)
    train_acc = float(np.mean(preds[train_mask] == y[train_mask]))
    val_acc   = float(np.mean(preds[val_mask]   == y[val_mask]))
    test_acc  = float(np.mean(preds[test_mask]  == y[test_mask]))

    print(f"\nKernel-SPR + C&S on {dataset_name}:")
    print(f"  Train: {train_acc:.4f}  Val: {val_acc:.4f}  Test: {test_acc:.4f}")

    return test_acc


def run_rbf_kernel(dataset_name, alpha_ppr=0.15, use_lappe=True,
                   use_cs=False, seed=42, verbose=True):
    """
    Kernel-SPR with RBF (Gaussian) kernel instead of arc-cosine.

    RBF uses Euclidean distances — no L2 normalisation — so it avoids
    the concentration of measure problem that limits arc-cosine on
    high-dimensional PPR features.

    Sweeps sigma (bandwidth) on validation:
    - Small sigma: only very close nodes are similar (sharp boundaries)
    - Large sigma: all nodes similar (underfits)
    - Auto-sigma (median heuristic) is the default starting point

    Also sweeps alpha_ppr to find optimal diffusion scale.
    """
    from kernel_spr import rbf_kernel

    data, X, y, train_mask, val_mask, test_mask, num_classes = \
        load_dataset(dataset_name, split_seed=seed)

    best_val   = -1
    best_acc   = 0
    best_cfg   = None

    # Sweep PPR alpha + sigma together
    alpha_grid = [0.10, 0.15, 0.20]
    sigma_grid = [0.5, 1.0, 2.0, 5.0, None]  # None = auto median heuristic

    print(f"  Sweeping (alpha, sigma) grid...")
    for alpha in alpha_grid:
        # Build PPR features
        K_tmp, tT = build_kernel_matrix_ppr(
            X, data.edge_index, data.num_nodes, train_mask,
            alpha_ppr=alpha, use_lappe=use_lappe,
        )
        # Now replace the arc-cosine kernel with RBF
        for sig in sigma_grid:
            K_rbf, actual_sigma = rbf_kernel(tT, sigma=sig)
            m = KernelGraphSPR(verbose=False)
            m.fit(K_rbf, y, train_mask, val_mask)
            va = m.evaluate(K_rbf, y, val_mask)
            te = m.evaluate(K_rbf, y, test_mask)
            print(f"  alpha={alpha} sigma={sig} -> "
                  f"val={va:.4f} test={te:.4f}")
            if va > best_val:
                best_val  = va
                best_acc  = te
                best_cfg  = (alpha, actual_sigma)
                best_K    = K_rbf
                best_model= m

    print(f"  Best: alpha={best_cfg[0]} sigma={best_cfg[1]:.3f} "
          f"val={best_val:.4f}")

    if use_cs:
        scores = best_model.predict_scores(best_K,
                     mask=np.ones(data.num_nodes, dtype=bool))
        cs = CorrectAndSmooth(autoscale=True)
        ac, as_, st, bv = cs.select_hyperparams(
            scores, y, data.edge_index, data.num_nodes,
            train_mask, val_mask,
            alpha_grid=[0.2, 0.5, 0.8], steps_grid=[10, 50],
        )
        cs = CorrectAndSmooth(num_correct=st, num_smooth=st,
                              alpha_correct=ac, alpha_smooth=as_,
                              autoscale=True)
        final    = cs.fit_transform(
            scores, y, data.edge_index, data.num_nodes, train_mask, val_mask)
        preds    = final.argmax(axis=1)
        best_acc = float(np.mean(preds[test_mask] == y[test_mask]))
        print(f"\nRBF-SPR + C&S on {dataset_name}: Test={best_acc:.4f}")
    else:
        print(f"\nRBF-SPR on {dataset_name}: Test={best_acc:.4f}")

    return best_acc


def run_multilayer_kernel(dataset_name, num_layers=3,
                          alpha_ppr=0.15, use_lappe=True,
                          seed=42, verbose=True):
    """
    Multi-layer Kernel-SPR + C&S (generalised to N layers).

    Each layer:
      - Takes previous layer output + original features
      - Builds new kernel
      - Solves kernel ridge
      - Applies C&S

    Terminates early if val accuracy stops improving.
    Only used on datasets where two-layer helps (Amazon Photo).
    """
    from kernel_spr import arc_cosine_kernel

    data, X, y, train_mask, val_mask, test_mask, num_classes = \
        load_dataset(dataset_name, split_seed=seed)
    N = data.num_nodes

    # Build base feature matrix once
    K_base, tilde_Theta_base = build_kernel_matrix_ppr(
        X, data.edge_index, N, train_mask,
        alpha_ppr=alpha_ppr, use_lappe=use_lappe,
    )
    D_base = tilde_Theta_base.shape[1]
    C      = num_classes

    tilde_Theta_curr = tilde_Theta_base.copy()
    best_val_prev    = -1
    best_test        = 0
    layer_out        = None

    for layer in range(1, num_layers + 1):
        print(f"\n=== LAYER {layer} ===")

        # Build kernel on current features
        K = arc_cosine_kernel(tilde_Theta_curr, degree=1)
        print(f"  K range: [{K.min():.3f}, {K.max():.3f}]  "
              f"shape: {tilde_Theta_curr.shape}")

        # Solve
        model = KernelGraphSPR(verbose=verbose)
        model.fit(K, y, train_mask, val_mask)
        va_raw = model.evaluate(K, y, val_mask)

        # C&S
        scores = model.predict_scores(K, mask=np.ones(N, dtype=bool))
        cs = CorrectAndSmooth(autoscale=True)
        ac, as_, st, bv = cs.select_hyperparams(
            scores, y, data.edge_index, N, train_mask, val_mask,
            alpha_grid=[0.2, 0.5, 0.8], steps_grid=[10, 50],
        )
        cs = CorrectAndSmooth(num_correct=st, num_smooth=st,
                              alpha_correct=ac, alpha_smooth=as_,
                              autoscale=True)
        layer_out = cs.fit_transform(
            scores, y, data.edge_index, N, train_mask, val_mask)

        preds   = layer_out.argmax(axis=1)
        val_acc = float(np.mean(preds[val_mask]  == y[val_mask]))
        te_acc  = float(np.mean(preds[test_mask] == y[test_mask]))
        print(f"  Layer {layer}: val={val_acc:.4f}  test={te_acc:.4f}")

        # Early stopping — stop if val no longer improves
        if val_acc <= best_val_prev:
            print(f"  Val did not improve ({val_acc:.4f} <= {best_val_prev:.4f}), stopping.")
            break

        best_val_prev = val_acc
        best_test     = te_acc

        # Augment features for next layer
        if layer < num_layers:
            l_scaled  = np.clip(layer_out * 2.0 - 1.0, -1+1e-7, 1-1e-7)
            l_angular = np.arccos(l_scaled).astype(np.float32)
            l_scale   = float(np.sqrt(D_base / C))
            tilde_Theta_curr = np.hstack([
                tilde_Theta_base, l_angular * l_scale
            ])
            print(f"  Next layer features: {tilde_Theta_curr.shape} "
                  f"(base {D_base} + layer{layer} {C} × {l_scale:.1f})")

    print(f"\nMulti-layer Kernel-SPR ({num_layers} layers) on {dataset_name}:")
    print(f"  Best test: {best_test:.4f}")
    return best_test


def run_two_layer_kernel(dataset_name, alpha_ppr=0.15, use_lappe=True,
                         seed=42, verbose=True):
    """
    Two-layer Kernel-SPR + C&S.

    Layer 1:  Original features → PPR → Kernel → Ridge → C&S → Scores¹
    Layer 2:  [Original features | Scores¹] → PPR → Kernel → Ridge → C&S → Final

    Layer 2 sees both:
      - Word-level similarity (PPR-diffused features)
      - Community-level predictions (output of Layer 1 + C&S)

    Analogous to 2-layer GCN but fully closed-form.
    Scores¹ from C&S encode label propagation information that
    original features do not contain — giving Layer 2 strictly
    more information than Layer 1.

    Still completely closed-form:
      alpha_1 = (K1_train + lambda*I)^{-1} y_train   [Layer 1 solve]
      alpha_2 = (K2_train + lambda*I)^{-1} y_train   [Layer 2 solve]
    No gradients anywhere.
    """
    data, X, y, train_mask, val_mask, test_mask, num_classes = \
        load_dataset(dataset_name, split_seed=seed)
    N = data.num_nodes

    # ---------------------------------------------------------------
    # LAYER 1: Standard Kernel-SPR + C&S
    # ---------------------------------------------------------------
    print("\n=== LAYER 1 ===")
    K1, tilde_Theta1 = build_kernel_matrix_ppr(
        X, data.edge_index, N, train_mask,
        alpha_ppr=alpha_ppr, use_lappe=use_lappe,
    )
    model1 = KernelGraphSPR(verbose=verbose)
    model1.fit(K1, y, train_mask, val_mask)

    # Get raw scores from Layer 1
    scores1 = model1.predict_scores(K1, mask=np.ones(N, dtype=bool))

    # Apply C&S to Layer 1 scores
    cs1 = CorrectAndSmooth(autoscale=True)
    ac, as_, st, bv = cs1.select_hyperparams(
        scores1, y, data.edge_index, N, train_mask, val_mask,
        alpha_grid=[0.2, 0.5, 0.8], steps_grid=[10, 50],
    )
    cs1 = CorrectAndSmooth(num_correct=st, num_smooth=st,
                           alpha_correct=ac, alpha_smooth=as_,
                           autoscale=True)
    layer1_out = cs1.fit_transform(
        scores1, y, data.edge_index, N, train_mask, val_mask
    )  # [N, C] — soft predictions after Layer 1 + C&S

    # Layer 1 test accuracy
    l1_test = float(np.mean(
        layer1_out.argmax(1)[test_mask] == y[test_mask]))
    print(f"  Layer 1 test: {l1_test:.4f}")

    # ---------------------------------------------------------------
    # LAYER 2: Augment features with Layer 1 output, rebuild kernel
    # ---------------------------------------------------------------
    print("\n=== LAYER 2 ===")

    # Angular transform Layer 1 output (soft probabilities in [0,1])
    # Map [0,1] -> [-1,1] -> arccos -> [0, pi]
    l1_scaled  = np.clip(layer1_out * 2.0 - 1.0, -1+1e-7, 1-1e-7)
    l1_angular = np.arccos(l1_scaled).astype(np.float32)  # [N, C]

    # Build new augmented feature matrix: original PPR features + Layer 1
    # tilde_Theta1 is [N, D+8], l1_angular is [N, C]
    # Scale l1_angular to match tilde_Theta1 magnitude
    D1  = tilde_Theta1.shape[1]
    C   = num_classes
    # Balance: Layer 1 features contribute equally to original features
    l1_scale   = float(np.sqrt(D1 / C))
    tilde_Theta2 = np.hstack([tilde_Theta1,
                               l1_angular * l1_scale])   # [N, D1+C]

    print(f"  Layer 2 feature matrix: {tilde_Theta2.shape} "
          f"(original {D1} + layer1 {C} × scale {l1_scale:.1f})")

    # Compute new kernel on augmented features
    from kernel_spr import arc_cosine_kernel
    K2 = arc_cosine_kernel(tilde_Theta2, degree=1)
    print(f"  K2 range: [{K2.min():.3f}, {K2.max():.3f}]")

    # Solve Layer 2
    model2 = KernelGraphSPR(verbose=verbose)
    model2.fit(K2, y, train_mask, val_mask)

    # Apply C&S to Layer 2
    scores2 = model2.predict_scores(K2, mask=np.ones(N, dtype=bool))
    cs2 = CorrectAndSmooth(autoscale=True)
    ac2, as2, st2, bv2 = cs2.select_hyperparams(
        scores2, y, data.edge_index, N, train_mask, val_mask,
        alpha_grid=[0.2, 0.5, 0.8], steps_grid=[10, 50],
    )
    cs2 = CorrectAndSmooth(num_correct=st2, num_smooth=st2,
                           alpha_correct=ac2, alpha_smooth=as2,
                           autoscale=True)
    final = cs2.fit_transform(
        scores2, y, data.edge_index, N, train_mask, val_mask
    )

    preds    = final.argmax(axis=1)
    test_acc = float(np.mean(preds[test_mask] == y[test_mask]))
    val_acc  = float(np.mean(preds[val_mask]  == y[val_mask]))

    print(f"\nTwo-layer Kernel-SPR on {dataset_name}:")
    print(f"  Layer 1 + C&S: {l1_test:.4f}")
    print(f"  Layer 2 + C&S: {test_acc:.4f}  (val={val_acc:.4f})")
    print(f"  Improvement:   {test_acc - l1_test:+.4f}")
    return test_acc


def run_alpha_sweep(dataset_name, use_lappe=True,
                    use_cs=False, seed=42, verbose=True):
    """
    Sweep PPR alpha values to find optimal neighbourhood scale.

    Different datasets need different alpha:
    - Small alpha (0.05): global diffusion, ~20 hops
    - Medium alpha (0.15): standard, ~7 hops  
    - Large alpha (0.50): local, ~2 hops
    - Very large alpha (0.90): almost no diffusion, ~1 hop

    Selects best alpha on validation set.
    """
    data, X, y, train_mask, val_mask, test_mask, num_classes = \
        load_dataset(dataset_name, split_seed=seed)

    alpha_grid = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90]

    best_val   = -1
    best_alpha = 0.15
    best_K     = None
    best_model = None

    print(f"  Sweeping PPR alpha: {alpha_grid}")

    for alpha in alpha_grid:
        K, _ = build_kernel_matrix_ppr(
            X, data.edge_index, data.num_nodes, train_mask,
            alpha_ppr=alpha, use_lappe=use_lappe,
        )
        m = KernelGraphSPR(verbose=False)
        m.fit(K, y, train_mask, val_mask)
        va = m.evaluate(K, y, val_mask)
        te = m.evaluate(K, y, test_mask)
        print(f"  alpha={alpha:.2f} -> val={va:.4f}  test={te:.4f}")
        if va > best_val:
            best_val   = va
            best_alpha = alpha
            best_K     = K
            best_model = m

    print(f"  Best alpha={best_alpha} val={best_val:.4f}")

    if use_cs:
        scores = best_model.predict_scores(best_K,
                     mask=np.ones(data.num_nodes, dtype=bool))
        cs = CorrectAndSmooth(autoscale=True)
        ac, as_, st, bv = cs.select_hyperparams(
            scores, y, data.edge_index, data.num_nodes,
            train_mask, val_mask,
            alpha_grid=[0.2, 0.5, 0.8], steps_grid=[10, 50],
        )
        cs = CorrectAndSmooth(num_correct=st, num_smooth=st,
                              alpha_correct=ac, alpha_smooth=as_,
                              autoscale=True)
        final    = cs.fit_transform(
            scores, y, data.edge_index, data.num_nodes, train_mask, val_mask)
        preds    = final.argmax(axis=1)
        test_acc = float(np.mean(preds[test_mask] == y[test_mask]))
        print(f"\nAlpha-sweep kernel + C&S on {dataset_name}: Test={test_acc:.4f}")
    else:
        test_acc  = best_model.evaluate(best_K, y, test_mask)
        print(f"\nAlpha-sweep kernel on {dataset_name}: Test={test_acc:.4f}")

    return test_acc


def run_adaptive_kernel(dataset_name, alpha_ppr=0.15, use_lappe=True,
                        use_cs=False, gamma=0.5, seed=42, verbose=True):
    """
    Kernel-SPR with adaptive degree-weighted regularisation.

    Lambda_uu = lambda * degree_u^{-gamma}
    High-degree nodes (well-connected, reliable neighbourhood) get
    less regularisation. Low-degree nodes (sparse, noisy) get more.

    Sweeps gamma in [0, 0.25, 0.5, 0.75, 1.0] on val set.
    gamma=0 -> scalar lambda (equivalent to standard kernel_ppr_lappe)
    """
    import torch
    from torch_geometric.utils import degree as pyg_degree

    data, X, y, train_mask, val_mask, test_mask, num_classes = \
        load_dataset(dataset_name, split_seed=seed)

    print(f"\nBuilding kernel matrix...")
    K, _ = build_kernel_matrix_ppr(
        X, data.edge_index, data.num_nodes, train_mask,
        alpha_ppr=alpha_ppr, use_lappe=use_lappe,
    )

    # Node degrees
    deg = pyg_degree(data.edge_index[0],
                     num_nodes=data.num_nodes).numpy()  # [N]

    gamma_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_val   = -1
    best_gamma = 0.0
    best_model = None

    print(f"  Sweeping gamma: {gamma_grid}")
    for g in gamma_grid:
        model_tmp = KernelGraphSPR(verbose=False)
        model_tmp.fit(K, y, train_mask, val_mask,
                      degree=(deg if g > 0 else None), gamma=g)
        va = model_tmp.evaluate(K, y, val_mask)
        print(f"  gamma={g:.2f} -> val={va:.4f}")
        if va > best_val:
            best_val   = va
            best_gamma = g
            best_model = model_tmp

    print(f"  Best gamma={best_gamma} val={best_val:.4f}")

    if use_cs:
        scores = best_model.predict_scores(K,
                     mask=np.ones(data.num_nodes, dtype=bool))
        cs = CorrectAndSmooth(autoscale=True)
        ac, as_, st, bv = cs.select_hyperparams(
            scores, y, data.edge_index, data.num_nodes,
            train_mask, val_mask,
            alpha_grid=[0.2, 0.5, 0.8], steps_grid=[10, 50],
        )
        cs = CorrectAndSmooth(num_correct=st, num_smooth=st,
                              alpha_correct=ac, alpha_smooth=as_,
                              autoscale=True)
        final    = cs.fit_transform(
            scores, y, data.edge_index, data.num_nodes, train_mask, val_mask)
        preds    = final.argmax(axis=1)
        test_acc = float(np.mean(preds[test_mask] == y[test_mask]))
        print(f"\nAdaptive kernel + C&S on {dataset_name}: Test={test_acc:.4f}")
    else:
        test_acc  = best_model.evaluate(K, y, test_mask)
        train_acc = best_model.evaluate(K, y, train_mask)
        print(f"\nAdaptive kernel on {dataset_name}:")
        print(f"  Train: {train_acc:.4f}  Val: {best_val:.4f}  Test: {test_acc:.4f}")

    return test_acc


def run_combined_kernel(dataset_name, alpha_ppr=0.15, use_lappe=True,
                        use_cs=False, seed=42, verbose=True):
    """
    Kernel-SPR with combined kernel: K_feat + lambda * K_LP.

    Sweeps lp_lambda in [0, 0.1, 0.5, 1, 2, 5, 10] on val set.
    lambda=0 -> pure feature kernel (equivalent to kernel_ppr_lappe)
    lambda=large -> LP dominates
    """
    data, X, y, train_mask, val_mask, test_mask, num_classes = \
        load_dataset(dataset_name, split_seed=seed)

    lambda_grid = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    best_val   = -1
    best_lam   = 0.0
    best_K     = None
    best_model = None

    print(f"  Sweeping lp_lambda: {lambda_grid}")

    for lam in lambda_grid:
        K_total, K_feat, K_lp = build_combined_kernel(
            X, data.edge_index, data.num_nodes, train_mask, y,
            alpha_ppr=alpha_ppr, use_lappe=use_lappe,
            lp_lambda=lam,
        )
        model_tmp = KernelGraphSPR(verbose=False)
        model_tmp.fit(K_total, y, train_mask, val_mask)
        va = model_tmp.evaluate(K_total, y, val_mask)
        print(f"  lambda={lam:.1f} -> val={va:.4f}")
        if va > best_val:
            best_val   = va
            best_lam   = lam
            best_K     = K_total
            best_model = model_tmp

    print(f"  Best lambda={best_lam} val={best_val:.4f}")

    if use_cs:
        scores = best_model.predict_scores(best_K,
                     mask=np.ones(data.num_nodes, dtype=bool))
        cs = CorrectAndSmooth(autoscale=True)
        ac, as_, st, bv = cs.select_hyperparams(
            scores, y, data.edge_index, data.num_nodes,
            train_mask, val_mask,
            alpha_grid=[0.2, 0.5, 0.8], steps_grid=[10, 50],
        )
        cs = CorrectAndSmooth(num_correct=st, num_smooth=st,
                              alpha_correct=ac, alpha_smooth=as_,
                              autoscale=True)
        final    = cs.fit_transform(
            scores, y, data.edge_index, data.num_nodes, train_mask, val_mask)
        preds    = final.argmax(axis=1)
        test_acc = float(np.mean(preds[test_mask] == y[test_mask]))
        print(f"\nCombined kernel + C&S on {dataset_name}: Test={test_acc:.4f}")
    else:
        test_acc  = best_model.evaluate(best_K, y, test_mask)
        train_acc = best_model.evaluate(best_K, y, train_mask)
        print(f"\nCombined kernel on {dataset_name}:")
        print(f"  Train: {train_acc:.4f}  Val: {best_val:.4f}  Test: {test_acc:.4f}")

    return test_acc


def run_lp_kernel_sweep(dataset_name, alpha_ppr=0.15, use_lappe=True,
                        use_cs=False, seed=42, verbose=True):
    """
    Kernel-SPR with LP features — sweeps lp_scale on val set.

    The key problem with LP features in high-D kernels is dimensionality
    dilution. With D=1448 total dims, 7 LP dims contribute only 0.5%.
    The correct scale is sqrt(D_feat/D_LP) to balance contributions.

    We sweep lp_scale in [1, 5, 10, 15, 20, 30] and pick the best.
    """
    data, X, y, train_mask, val_mask, test_mask, num_classes = \
        load_dataset(dataset_name, split_seed=seed)

    D_feat = X.shape[1] + (8 if use_lappe else 0)
    D_lp   = num_classes
    # Balanced scale: LP contributes equally to all feature dims
    scale_balanced = float(np.sqrt(D_feat / D_lp))
    # Sweep around the balanced point
    scale_grid = [1.0, scale_balanced/2, scale_balanced,
                  scale_balanced*2, scale_balanced*4]
    scale_grid = [round(s, 2) for s in scale_grid]

    print(f"  LP scale sweep: {scale_grid}")
    print(f"  (balanced scale = sqrt({D_feat}/{D_lp}) = {scale_balanced:.1f})")

    best_val  = -1
    best_scale = 1.0
    best_K     = None

    for lp_s in scale_grid:
        K, _ = build_kernel_matrix_ppr_with_lp(
            X, data.edge_index, data.num_nodes, train_mask, y,
            alpha_ppr=alpha_ppr, use_lappe=use_lappe,
            lp_alpha=0.1, lp_steps=50, lp_scale=lp_s,
        )
        model_tmp = KernelGraphSPR(verbose=False)
        model_tmp.fit(K, y, train_mask, val_mask)
        va = model_tmp.evaluate(K, y, val_mask)
        print(f"  lp_scale={lp_s:.1f} -> val={va:.4f}")
        if va > best_val:
            best_val   = va
            best_scale = lp_s
            best_K     = K
            best_model = model_tmp

    print(f"  Best lp_scale={best_scale} val={best_val:.4f}")

    if use_cs:
        scores = best_model.predict_scores(best_K,
                     mask=np.ones(data.num_nodes, dtype=bool))
        cs = CorrectAndSmooth(autoscale=True)
        ac, as_, st, bv = cs.select_hyperparams(
            scores, y, data.edge_index, data.num_nodes,
            train_mask, val_mask,
            alpha_grid=[0.2, 0.5, 0.8], steps_grid=[10, 50],
        )
        cs = CorrectAndSmooth(num_correct=st, num_smooth=st,
                              alpha_correct=ac, alpha_smooth=as_,
                              autoscale=True)
        final = cs.fit_transform(
            scores, y, data.edge_index, data.num_nodes, train_mask, val_mask)
        preds    = final.argmax(axis=1)
        test_acc = float(np.mean(preds[test_mask] == y[test_mask]))
        print(f"\nKernel-SPR+LP+C&S on {dataset_name}: Test={test_acc:.4f}")
    else:
        test_acc  = best_model.evaluate(best_K, y, test_mask)
        train_acc = best_model.evaluate(best_K, y, train_mask)
        val_acc2  = best_model.evaluate(best_K, y, val_mask)
        print(f"\nKernel-SPR+LP on {dataset_name}:")
        print(f"  Train: {train_acc:.4f}  Val: {val_acc2:.4f}  Test: {test_acc:.4f}")

    return test_acc


def run_lp_kernel(dataset_name, alpha_ppr=0.15, use_lappe=True,
                 lp_alpha=0.1, lp_steps=50, lp_scale=1.0,
                 use_cs=False, seed=42, verbose=True):
    """
    Kernel-SPR with Label Propagation features + optional C&S.

    Adds LP soft predictions as additional kernel dimensions.
    LP runs purely on graph structure (no node features),
    giving the model direct community membership information.

    lp_alpha: LP clamping factor — smaller = more graph propagation
    lp_scale: weight of LP features relative to node features
    use_cs:   whether to apply C&S post-processing after kernel
    """
    data, X, y, train_mask, val_mask, test_mask, num_classes = \
        load_dataset(dataset_name, split_seed=seed)

    print(f"\nBuilding LP-augmented kernel...")
    K, tilde_Theta = build_kernel_matrix_ppr_with_lp(
        X, data.edge_index, data.num_nodes, train_mask, y,
        alpha_ppr  = alpha_ppr,
        use_lappe  = use_lappe,
        lp_alpha   = lp_alpha,
        lp_steps   = lp_steps,
        lp_scale   = lp_scale,
    )

    model = KernelGraphSPR(verbose=verbose)
    model.fit(K, y, train_mask, val_mask)

    if use_cs:
        all_mask = np.ones(data.num_nodes, dtype=bool)
        scores   = model.predict_scores(K, mask=all_mask)
        cs       = CorrectAndSmooth(autoscale=True)
        alpha_c, alpha_s, steps, bv = cs.select_hyperparams(
            scores, y, data.edge_index, data.num_nodes,
            train_mask, val_mask,
            alpha_grid=[0.2, 0.5, 0.8], steps_grid=[10, 50],
        )
        cs = CorrectAndSmooth(num_correct=steps, num_smooth=steps,
                              alpha_correct=alpha_c, alpha_smooth=alpha_s,
                              autoscale=True)
        final  = cs.fit_transform(
            scores, y, data.edge_index, data.num_nodes, train_mask, val_mask)
        preds    = final.argmax(axis=1)
        train_acc = float(np.mean(preds[train_mask] == y[train_mask]))
        val_acc   = float(np.mean(preds[val_mask]   == y[val_mask]))
        test_acc  = float(np.mean(preds[test_mask]  == y[test_mask]))
        print(f"\nKernel-SPR+LP+C&S on {dataset_name}:")
    else:
        train_acc = model.evaluate(K, y, train_mask)
        val_acc   = model.evaluate(K, y, val_mask)
        test_acc  = model.evaluate(K, y, test_mask)
        print(f"\nKernel-SPR+LP on {dataset_name}:")

    print(f"  Train: {train_acc:.4f}  Val: {val_acc:.4f}  Test: {test_acc:.4f}")
    return test_acc


def run_kernel_spr(dataset_name, alpha_ppr=0.15, use_lappe=True,
                   kernel_degree=1, seed=42, verbose=True):
    """
    Kernel-SPR pipeline.

    Uses arc-cosine kernel over PPR-diffused angular coordinates
    of ALL features — no prescreening, no path selection.

    Returns test accuracy.
    """
    data, X, y, train_mask, val_mask, test_mask, num_classes =         load_dataset(dataset_name, split_seed=seed)

    print(f"Building kernel matrix (all {X.shape[1]} features)...")
    K, tilde_Theta = build_kernel_matrix_ppr(
        X, data.edge_index, data.num_nodes, train_mask,
        alpha_ppr     = alpha_ppr,
        kernel_degree = kernel_degree,
        use_lappe     = use_lappe,
    )

    model = KernelGraphSPR(verbose=verbose)
    model.fit(K, y, train_mask, val_mask)

    train_acc = model.evaluate(K, y, train_mask)
    val_acc   = model.evaluate(K, y, val_mask)
    test_acc  = model.evaluate(K, y, test_mask)

    print(f"Kernel-SPR on {dataset_name}:")
    print(f"  Train: {train_acc:.4f}  Val: {val_acc:.4f}  "
          f"Test: {test_acc:.4f}")
    print(f"  Lambda: {model.best_lambda:.0e}")

    return test_acc


def run_all_datasets(args):
    datasets = ['cora', 'citeseer', 'amazon_photo', 'actor']
    if args.dataset != 'all':
        datasets = [args.dataset]

    results = {}

    for ds in datasets:
        print(f"\n{'#'*60}")
        print(f"# {ds.upper()}  mode={args.mode}  "
              f"selection={args.selection}")
        print(f"{'#'*60}")

        defaults    = DATASET_DEFAULTS.get(ds, {})
        top_T       = args.top_T       or defaults.get('top_T', 80)
        max_support = args.max_support or defaults.get('max_support', 1)
        max_paths   = args.max_paths   or defaults.get('max_paths', 50)
        lap_K       = args.lap_K       or defaults.get('lap_K', 8)
        rw_K        = args.rw_K        or defaults.get('rw_K', 8)

        # Use dataset-specific best mode if --mode auto
        mode = args.mode
        if mode == 'auto':
            mode = defaults.get('best_mode', 'ppr')
            print(f"  Auto-selected mode: {mode}")

        try:
            # Kernel-SPR mode — completely separate pipeline
            if mode in ('kernel_ppr', 'kernel_ppr_lappe', 'cs_kernel', 'cs_kernel_lappe', 'cs_mlp', 'lp_kernel', 'lp_kernel_cs', 'combined_kernel', 'combined_kernel_cs', 'adaptive_kernel', 'adaptive_kernel_cs', 'alpha_sweep', 'alpha_sweep_cs', 'two_layer_kernel', 'multilayer_kernel', 'rbf_kernel', 'rbf_kernel_cs'):
                if mode in ('rbf_kernel', 'rbf_kernel_cs'):
                    gspr_acc = run_rbf_kernel(
                        ds,
                        alpha_ppr = 0.15,
                        use_lappe = True,
                        use_cs    = (mode == 'rbf_kernel_cs'),
                        seed      = args.seed,
                        verbose   = args.verbose,
                    )
                elif mode == 'multilayer_kernel':
                    gspr_acc = run_multilayer_kernel(
                        ds,
                        num_layers = 5,
                        alpha_ppr  = 0.15,
                        use_lappe  = True,
                        seed       = args.seed,
                        verbose    = args.verbose,
                    )
                elif mode == 'two_layer_kernel':
                    gspr_acc = run_two_layer_kernel(
                        ds,
                        alpha_ppr = 0.15,
                        use_lappe = True,
                        seed      = args.seed,
                        verbose   = args.verbose,
                    )
                elif mode in ('alpha_sweep', 'alpha_sweep_cs'):
                    gspr_acc = run_alpha_sweep(
                        ds,
                        use_lappe = True,
                        use_cs    = (mode == 'alpha_sweep_cs'),
                        seed      = args.seed,
                        verbose   = args.verbose,
                    )
                elif mode in ('adaptive_kernel', 'adaptive_kernel_cs'):
                    gspr_acc = run_adaptive_kernel(
                        ds,
                        alpha_ppr = 0.15,
                        use_lappe = True,
                        use_cs    = (mode == 'adaptive_kernel_cs'),
                        seed      = args.seed,
                        verbose   = args.verbose,
                    )
                elif mode in ('combined_kernel', 'combined_kernel_cs'):
                    gspr_acc = run_combined_kernel(
                        ds,
                        alpha_ppr = 0.15,
                        use_lappe = True,
                        use_cs    = (mode == 'combined_kernel_cs'),
                        seed      = args.seed,
                        verbose   = args.verbose,
                    )
                elif mode in ('lp_kernel', 'lp_kernel_cs'):
                    # lp_scale balances LP vs feature contribution:
                    # sqrt(D_feat / D_LP) makes LP contribute equally
                    # to all feature dimensions combined.
                    # D_feat=1441 (cora), D_LP=7 -> scale=sqrt(1441/7)=14.4
                    # We sweep lp_scale on val set automatically.
                    gspr_acc = run_lp_kernel_sweep(
                        ds,
                        alpha_ppr = 0.15,
                        use_lappe = True,
                        use_cs    = (mode == 'lp_kernel_cs'),
                        seed      = args.seed,
                        verbose   = args.verbose,
                    )
                elif mode == 'cs_mlp':
                    gspr_acc = run_cs_mlp(
                        ds, seed=args.seed
                    )
                elif mode in ('cs_kernel', 'cs_kernel_lappe'):
                    gspr_acc = run_cs_kernel(
                        ds,
                        alpha_ppr  = 0.15,
                        use_lappe  = (mode == 'cs_kernel_lappe'),
                        sweep_params = True,
                        seed       = args.seed,
                        verbose    = args.verbose,
                    )
                else:
                    gspr_acc = run_kernel_spr(
                        ds,
                        alpha_ppr    = 0.15,
                        use_lappe    = (mode == 'kernel_ppr_lappe'),
                        kernel_degree= 1,
                        seed         = args.seed,
                        verbose      = args.verbose,
                    )
                data, X, y, train_mask, val_mask, test_mask, num_classes =                     load_dataset(ds, split_seed=args.seed)
                print("Running baselines...")
                gcn_acc  = run_gcn(data, num_classes, seed=args.seed)
                sgc_acc  = run_sgc(data, num_classes, seed=args.seed)
                gat_acc  = run_gat(data, num_classes, seed=args.seed)
                gin_acc  = run_gin(data, num_classes, seed=args.seed)
                sage_acc = run_graphsage(data, num_classes, seed=args.seed)
                mlp_acc  = run_mlp(X, y, train_mask, val_mask, test_mask,
                                   num_classes, seed=args.seed)
                label = f"Graph-SPR({mode},kernel)"
                results[ds] = {
                    label:   gspr_acc,
                    'GCN':   gcn_acc,
                    'SGC':   sgc_acc,
                    'GAT':   gat_acc,
                    'GIN':   gin_acc,
                    'SAGE':  sage_acc,
                    'MLP':   mlp_acc,
                }
                print_results_table(results, mode, 'kernel')
                continue

            gspr_acc, model, meta = run_graph_spr(
                ds,
                mode              = mode,
                selection         = args.selection,
                max_paths         = max_paths,
                max_support       = max_support,
                max_freq          = args.max_freq,
                patience          = args.patience,
                top_T             = top_T,
                lap_K             = lap_K,
                rw_K              = rw_K,
                beam_width        = args.beam_width,
                gamma             = args.gamma,
                lam_nc            = args.lam_nc,
                crossover_interval= args.crossover_interval,
                verbose           = args.verbose,
                seed              = args.seed,
            )

            # Baselines
            data, X, y, train_mask, val_mask, test_mask, num_classes = \
                load_dataset(ds, split_seed=args.seed)
            print("\nRunning baselines...")
            gcn_acc  = run_gcn(data, num_classes, seed=args.seed)
            sgc_acc  = run_sgc(data, num_classes, seed=args.seed)
            gat_acc  = run_gat(data, num_classes, seed=args.seed)
            gin_acc  = run_gin(data, num_classes, seed=args.seed)
            sage_acc = run_graphsage(data, num_classes, seed=args.seed)
            mlp_acc  = run_mlp(X, y, train_mask, val_mask, test_mask,
                               num_classes, seed=args.seed)

            label = f"Graph-SPR({mode},{args.selection})"
            results[ds] = {
                label:   gspr_acc,
                'GCN':   gcn_acc,
                'SGC':   sgc_acc,
                'GAT':   gat_acc,
                'GIN':   gin_acc,
                'SAGE':  sage_acc,
                'MLP':   mlp_acc,
            }

        except Exception as e:
            print(f"Error on {ds}: {e}")
            import traceback; traceback.print_exc()
            results[ds] = {}

    if args.mode not in ('kernel_ppr', 'kernel_ppr_lappe', 'cs_kernel', 'cs_kernel_lappe', 'cs_mlp', 'lp_kernel', 'lp_kernel_cs', 'combined_kernel', 'combined_kernel_cs', 'adaptive_kernel', 'adaptive_kernel_cs', 'alpha_sweep', 'alpha_sweep_cs', 'two_layer_kernel', 'multilayer_kernel', 'rbf_kernel', 'rbf_kernel_cs'):
        print_results_table(results, args.mode, args.selection)
    return results


def print_results_table(results, mode, selection):
    label   = f"Graph-SPR({mode},{selection})"
    methods = [label, 'GCN', 'SGC', 'GAT', 'GIN', 'SAGE', 'MLP']
    w       = 75
    print(f"\n{'='*w}")
    print(f"{'Dataset':<22}", end='')
    for m in methods:
        print(f"{m:>13}", end='')
    print()
    print('-' * w)
    for ds, res in results.items():
        print(f"{ds:<22}", end='')
        for m in methods:
            val = res.get(m, None)
            if val is not None:
                print(f"{val*100:>12.2f}%", end='')
            else:
                print(f"{'N/A':>13}", end='')
        print()
    print('=' * w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph-SPR Experiments')
    parser.add_argument('--dataset',    type=str, default='all')
    parser.add_argument('--mode',       type=str, default='ppr',
                        help='Diffusion+encoding mode. Use "auto" for '
                             'dataset-specific best.')
    parser.add_argument('--selection',  type=str, default='beam',
                        choices=['greedy', 'beam'],
                        help='Path selection: greedy (original) or '
                             'beam (guided, new)')

    # Diffusion hyperparameters
    parser.add_argument('--K',           type=int,   default=4)
    parser.add_argument('--r',           type=float, default=0.5)
    parser.add_argument('--top_T',       type=int,   default=None)
    parser.add_argument('--lap_K',       type=int,   default=None)
    parser.add_argument('--rw_K',        type=int,   default=None)
    parser.add_argument('--hks_T',       type=int,   default=6)

    # Path selection hyperparameters
    parser.add_argument('--max_paths',   type=int,   default=None)
    parser.add_argument('--max_support', type=int,   default=None)
    parser.add_argument('--max_freq',    type=int,   default=2)
    parser.add_argument('--patience',    type=int,   default=5)

    # Beam search hyperparameters
    parser.add_argument('--beam_width',  type=int,   default=3,
                        help='Number of parallel beams (default 3)')
    parser.add_argument('--gamma',       type=float, default=0.05,
                        help='Smoothness penalty weight (default 0.05)')
    parser.add_argument('--lam_nc',      type=float, default=0.05,
                        help='NC reward weight (default 0.05)')
    parser.add_argument('--crossover_interval', type=int, default=5,
                        help='Apply crossover every N steps (default 5)')

    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--verbose',     action='store_true', default=True)

    args = parser.parse_args()
    run_all_datasets(args)
