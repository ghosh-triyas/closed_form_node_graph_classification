"""
make_ring_scatter_supplement.py

Dataset-level version of the single-pair Figure 1 -- computes final
beta_1 (ring count) and the first non-zero L1 eigenvalue past the
kernel, for EVERY MUTAG graph, colored by mutagenic/non-mutagenic
class. Directly answers the reviewer note that Figure 1 (n=1) is
anecdotal.

Reuses the K1-truncation fix from make_phs_trajectory_figure_v3.py:
K1 must exceed each graph's own beta_1, or the eigenvalue reads as a
false zero (the exact bug we found and fixed for the single-pair
figure). Here we set K1 per-graph, comfortably above its own beta_1.
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

from hodge import hodge_spectra

sys.path.insert(0, '.')
from run_all_graph_datasets_with_significance import load_tu_raw


def final_beta1_and_eigenvalue(G, K1_margin=5):
    """Compute the FINAL-step (whole graph) beta_1 and first non-zero
    L1 eigenvalue, with K1 set safely above this specific graph's own
    beta_1 -- avoiding the truncation bug found earlier."""
    if G.number_of_edges() == 0:
        return 0, np.nan
    # First pass: cheap nullity-only check to learn this graph's beta_1
    _, _, _, b1_check = hodge_spectra(G, K0=1, K1=1)
    K1 = b1_check + K1_margin
    spec0, spec1, b0, b1 = hodge_spectra(G, K0=1, K1=K1)
    nonzero = spec1[spec1 > 1e-8]
    eig = nonzero[0] if len(nonzero) > 0 else np.nan
    return b1, eig


if __name__ == '__main__':
    print("Loading MUTAG...")
    graphs, labels = load_tu_raw('MUTAG')
    print(f"  {len(graphs)} graphs")

    print("Computing final beta_1 and first non-zero L1 eigenvalue "
          "for every graph...")
    beta1_vals, eig_vals, classes = [], [], []
    n_insufficient = 0
    for i, (G, y) in enumerate(zip(graphs, labels)):
        if i % 20 == 0:
            print(f"    {i}/{len(graphs)}", end='\r')
        b1, eig = final_beta1_and_eigenvalue(G)
        beta1_vals.append(b1)
        eig_vals.append(eig)
        classes.append(int(y))
        if np.isnan(eig) and b1 > 0:
            n_insufficient += 1
    print(f"\n  Done. Graphs where K1 margin was still insufficient: "
          f"{n_insufficient} (should be 0 or near-0)")

    beta1_vals = np.array(beta1_vals)
    eig_vals = np.array(eig_vals)
    classes = np.array(classes)

    # Drop any graph where eigenvalue is undefined (beta_1=0, e.g. a
    # tree-like non-mutagenic graph with no ring at all -- no
    # "eigenvalue past the kernel" exists there, correctly excluded
    # rather than plotted as a false zero).
    valid = ~np.isnan(eig_vals)
    print(f"  Graphs with at least one ring (plotted): {valid.sum()} / {len(graphs)}")
    print(f"  Graphs with zero rings (excluded, beta_1=0): {(~valid).sum()}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for cls, label, color, marker in [(1, 'Mutagenic', '#d62728', 'o'),
                                        (0, 'Non-mutagenic', '#1f77b4', 's')]:
        mask = valid & (classes == cls)
        ax.scatter(beta1_vals[mask], eig_vals[mask], c=color, marker=marker,
                   label=f'{label} (n={mask.sum()})', alpha=0.6, s=40,
                   edgecolors='none')

    ax.set_xlabel(r'Ring count $\beta_1$ (final graph)')
    ax.set_ylabel(r'First non-zero eigenvalue of $L_1$')
    ax.set_title('Ring count vs. spectral tightness across all of MUTAG')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = '/Users/X/Desktop/TopoSPR/paper_outputs/figures/ring_scatter_mutag.pdf'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f"\nSaved to {out_path} (and .png)")

    # Quick summary stats, useful for the supplement's caption
    print(f"\nMean eigenvalue, mutagenic (ring>0 only):     "
          f"{eig_vals[valid & (classes==1)].mean():.4f}")
    print(f"Mean eigenvalue, non-mutagenic (ring>0 only): "
          f"{eig_vals[valid & (classes==0)].mean():.4f}")
    print("(Compare to the beta_1 means already in Appendix C.2: "
          "3.55 mutagenic vs 1.74 non-mutagenic)")
