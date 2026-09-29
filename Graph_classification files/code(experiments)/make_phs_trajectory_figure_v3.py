"""
make_phs_trajectory_figure_v3.py

FIX from v2: the right panel (first non-zero L1 eigenvalue) was
silently wrong for the mutagenic example, because K1=5 was smaller
than that graph's beta_1=7 -- meaning ALL 5 requested eigenvalues
were zero (part of the 7-dimensional kernel), and the script never
saw far enough into the spectrum to find a genuine non-zero value.
This is a TRUNCATION ARTIFACT, not a real finding: dim(ker L1) = 7
was computed correctly (from the FULL eigendecomposition, before
truncation -- see hodge.py's compute_spectrum), but the truncated
K1=5-sized eigenvalue array literally cannot contain anything beyond
the 5th smallest, so it can never reach eigenvalue #6 or #7, let
alone the first genuinely non-zero one after them.

FIX: request K1 safely larger than the maximum beta_1 either example
reaches (7 here), so the returned array actually extends past the
zero-eigenvalue block into real spectral information. We use K1=12
for this figure specifically (comfortable margin), independent of
whatever K1 is used elsewhere in the paper's main classification
pipeline -- this is purely for correct VISUALIZATION, not a change
to the classification method itself.
"""
import sys
sys.path.insert(0, '/Users/X/Desktop/TopoSPR')
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from ricci import compute_ollivier_ricci
from hodge import hodge_spectra
from phs import build_filtration_steps, get_subgraph_at_threshold

sys.path.insert(0, '.')
from run_all_graph_datasets_with_significance import load_tu_raw


def compute_phs_trajectory(G, T=8, K0=5, K1=12, alpha_ricci=0.5):
    """K1 default raised from 5 to 12 -- see module docstring. Also
    now returns beta_1 alongside the eigenvalue at each step, so we
    can VERIFY at call time that K1 was large enough (beta_1 < K1
    at every step), rather than silently trusting it."""
    curvature = compute_ollivier_ricci(G, alpha=alpha_ricci)
    if not curvature:
        return None, None, None, None
    thresholds, sorted_edges = build_filtration_steps(curvature, T=T)
    beta1_traj, eig_traj, k1_sufficient = [], [], []
    for tau in thresholds:
        G_t = get_subgraph_at_threshold(G, sorted_edges, tau)
        spec0, spec1, b0, b1 = hodge_spectra(G_t, K0=K0, K1=K1)
        beta1_traj.append(b1)
        nonzero = spec1[spec1 > 1e-8]
        eig_traj.append(nonzero[0] if len(nonzero) > 0 else np.nan)
        # Sanity flag: did we actually have enough K1 slack to see
        # PAST the zero block at this step?
        k1_sufficient.append(b1 < K1)
    return thresholds, beta1_traj, eig_traj, k1_sufficient


def final_beta1(G):
    if G.number_of_edges() == 0:
        return 0
    _, _, b0, b1 = hodge_spectra(G, K0=1, K1=1)
    return b1


if __name__ == '__main__':
    print("Loading MUTAG...")
    graphs, labels = load_tu_raw('MUTAG')

    # Reuse the same pair-selection logic as v2 (search for the best
    # available contrast rather than a naive first-match).
    b1_index = {0: [], 1: []}
    for i, (G, y) in enumerate(zip(graphs, labels)):
        b1 = final_beta1(G)
        b1_index[int(y)].append((i, b1, G.number_of_nodes()))

    def filter_size(cands, lo=12, hi=28):
        f = [c for c in cands if lo <= c[2] <= hi]
        return f if f else cands

    nonmut_candidates = filter_size(sorted(b1_index[0], key=lambda x: x[1]))
    mut_candidates = filter_size(sorted(b1_index[1], key=lambda x: -x[1]))
    nonmut_idx, nonmut_b1, nonmut_n = nonmut_candidates[0]
    mut_idx, mut_b1, mut_n = mut_candidates[0]

    print(f"Non-mutagenic example: graph {nonmut_idx} "
          f"({nonmut_n} nodes, final beta_1={nonmut_b1})")
    print(f"Mutagenic example: graph {mut_idx} "
          f"({mut_n} nodes, final beta_1={mut_b1})")

    # K1 must comfortably exceed the LARGER of the two final beta_1
    # values -- set explicitly and verified below, not just assumed.
    K1_FIGURE = max(mut_b1, nonmut_b1) + 5
    print(f"\nUsing K1={K1_FIGURE} for this figure (max beta_1={max(mut_b1,nonmut_b1)}, "
          f"+5 safety margin)")

    T = 8
    print("\nComputing mutagenic trajectory...")
    thr_m, beta1_m, eig_m, ok_m = compute_phs_trajectory(
        graphs[mut_idx], T=T, K1=K1_FIGURE)
    print("Computing non-mutagenic trajectory...")
    thr_n, beta1_n, eig_n, ok_n = compute_phs_trajectory(
        graphs[nonmut_idx], T=T, K1=K1_FIGURE)

    print(f"\nMutagenic beta_1:     {beta1_m}")
    print(f"Mutagenic eigenvalue: {[round(x,4) if not np.isnan(x) else 'NaN' for x in eig_m]}")
    print(f"K1 sufficient at every step (mutagenic)? {all(ok_m)}")
    print(f"\nNon-mutagenic beta_1:     {beta1_n}")
    print(f"Non-mutagenic eigenvalue: {[round(x,4) if not np.isnan(x) else 'NaN' for x in eig_n]}")
    print(f"K1 sufficient at every step (non-mutagenic)? {all(ok_n)}")

    if not (all(ok_m) and all(ok_n)):
        print("\n  WARNING: K1 was still insufficient at some step even "
              "with the safety margin -- increase K1_FIGURE further "
              "before trusting the right panel.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    steps = list(range(1, T+1))

    axes[0].plot(steps, beta1_m, 'o-', color='#d62728',
                 label=f'Mutagenic ({mut_b1} rings)', linewidth=2, markersize=7)
    axes[0].plot(steps, beta1_n, 's-', color='#1f77b4',
                 label=f'Non-mutagenic ({nonmut_b1} rings)', linewidth=2, markersize=7)
    axes[0].set_xlabel('Filtration step $t$')
    axes[0].set_ylabel(r'$\beta_1^{(t)}$ (independent cycles)')
    axes[0].set_title('Betti number trajectory')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    max_b1 = max(max(beta1_m), max(beta1_n))
    axes[0].set_yticks(range(0, max_b1 + 2))

    axes[1].plot(steps, eig_m, 'o-', color='#d62728',
                 label=f'Mutagenic ({mut_b1} rings)', linewidth=2, markersize=7)
    axes[1].plot(steps, eig_n, 's-', color='#1f77b4',
                 label=f'Non-mutagenic ({nonmut_b1} rings)', linewidth=2, markersize=7)
    axes[1].set_xlabel('Filtration step $t$')
    axes[1].set_ylabel(f'Eigenvalue #{max(mut_b1,nonmut_b1)+1} of $L_1$ '
                        f'(first past the zero block)')
    axes[1].set_title('Hodge spectrum trajectory')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = 'paper_outputs/figures/phs_trajectory_figure_v3.pdf'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f"\nSaved corrected figure to {out_path} (and .png)")
