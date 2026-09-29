"""
make_tsne_phs.py

t-SNE projection of PHS feature vectors for MUTAG, colored by class --
COMET-style visualization of whether the representation itself
separates classes before any classifier is applied. Saved directly
into paper_outputs/, never /tmp.
"""
import sys
sys.path.insert(0, '/Users/X/Desktop/TopoSPR')
import numpy as np
import networkx as nx
import warnings, os
warnings.filterwarnings('ignore')

from hodge import hodge_spectra
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from run_all_graph_datasets_with_significance import load_tu_raw, compute_phs_features

FIGURE_DIR = '/Users/X/Desktop/TopoSPR/paper_outputs/figures'
os.makedirs(FIGURE_DIR, exist_ok=True)

if __name__ == '__main__':
    print("Loading MUTAG...")
    graphs, labels = load_tu_raw('MUTAG')
    print(f"  {len(graphs)} graphs")

    print("Computing PHS features (headline config: ricci)...")
    N = len(graphs)
    x0 = compute_phs_features(graphs[0], T=8, K0=5, K1=5, mode='ricci')
    X = np.zeros((N, len(x0)), dtype=np.float32)
    for i, G in enumerate(graphs):
        if i % 20 == 0:
            print(f"    {i}/{N}", end='\r')
        X[i] = compute_phs_features(G, T=8, K0=5, K1=5, mode='ricci')
    print(f"\n  Feature matrix: {X.shape}")

    print("Running t-SNE (perplexity=30)...")
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    tsne = TSNE(n_components=2, perplexity=min(30, N//4), random_state=42, init='pca')
    X2d = tsne.fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for cls, name, color in [(1, 'Mutagenic', '#d62728'), (0, 'Non-mutagenic', '#1f77b4')]:
        mask = labels == cls
        ax.scatter(X2d[mask, 0], X2d[mask, 1], c=color, label=f'{name} (n={mask.sum()})',
                   alpha=0.7, s=35, edgecolors='none')
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.set_title('t-SNE of PHS feature space (MUTAG)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(FIGURE_DIR, 'tsne_phs_mutag.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved to {out_path}")
