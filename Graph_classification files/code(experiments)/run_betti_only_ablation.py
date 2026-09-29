"""
run_betti_only_ablation.py

Blocker #7 from the review: zero out the eigenvalue blocks
(lambda^(t), mu^(t)) of phi(G), keep only the Betti number blocks
(beta_0^(t), beta_1^(t)), and rerun classification. This is the
direct empirical test of the paper's central theoretical claim --
Theorem 2 says PHS is STRICTLY more expressive than the Betti-number
sequence alone; this ablation checks whether that extra information
actually pays off empirically, or whether Betti numbers alone already
capture most of the signal.

Uses each dataset's actual headline filtration combination (matching
Table 4's "Full pipeline (reported)" row), but strips eigenvalues
from every filtration in that combination, keeping only [beta_0^(t),
beta_1^(t)] per step per filtration.

SELF-CONTAINED -- only needs hodge.py, classifier.py, and the raw TU
data files already on disk. Results saved directly into
paper_outputs/, never /tmp.
"""
import sys
sys.path.insert(0, '/Users/X/Desktop/TopoSPR')
import numpy as np
import networkx as nx
import warnings, os, csv
warnings.filterwarnings('ignore')

from hodge import hodge_spectra
from classifier import KernelRidgeClassifier
from sklearn.model_selection import StratifiedKFold

# ── SAVE LOCATION: project folder, never /tmp ───────────────────────
OUTPUT_DIR = '/Users/X/Desktop/TopoSPR/paper_outputs/data'
FIGURE_DIR = '/Users/X/Desktop/TopoSPR/paper_outputs/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


def load_tu_raw(name, base='/tmp/tu_datasets'):
    """Data files themselves stay in /tmp (re-downloaded each session,
    as established) -- only RESULTS are saved persistently below."""
    d = f'{base}/{name}'
    with open(f'{d}/{name}_graph_indicator.txt') as f:
        graph_ind = [int(float(x.strip())) for x in f]
    with open(f'{d}/{name}_A.txt') as f:
        edges_all = [tuple(int(float(x)) for x in l.strip().split(','))
                     for l in f]
    with open(f'{d}/{name}_graph_labels.txt') as f:
        graph_labels = [int(float(x.strip())) for x in f]
    node_attrs = None
    try:
        with open(f'{d}/{name}_node_attributes.txt') as f:
            node_attrs = []
            for line in f:
                vals = [float(v) for v in line.strip().split(',')]
                node_attrs.append(np.array(vals, dtype=np.float32))
    except: pass
    node_labels = None
    try:
        with open(f'{d}/{name}_node_labels.txt') as f:
            node_labels = [int(float(x.strip())) for x in f]
    except: pass
    n_graphs = max(graph_ind)
    node_to_graph = {i+1: graph_ind[i] for i in range(len(graph_ind))}
    graphs, labels = [], []
    for g_id in range(1, n_graphs + 1):
        G = nx.Graph()
        nodes_g = [i+1 for i, gi in enumerate(graph_ind) if gi == g_id]
        for n in nodes_g:
            if node_attrs is not None:
                G.add_node(n, feat=node_attrs[n-1])
            elif node_labels is not None:
                G.add_node(n, feat=np.array([node_labels[n-1]], dtype=np.float32))
            else:
                G.add_node(n)
        for u, v in edges_all:
            if node_to_graph.get(u) == g_id and u != v:
                G.add_edge(u, v)
        graphs.append(G)
        labels.append(graph_labels[g_id - 1])
    unique = sorted(set(labels))
    lmap = {l: i for i, l in enumerate(unique)}
    labels = np.array([lmap[l] for l in labels])
    return graphs, labels


# ── Ricci curvature (needed for the 'ricci' filtration mode) ────────
import ot
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

def lazy_dist(G, node, alpha=0.5):
    nbrs = list(G.neighbors(node))
    deg = len(nbrs)
    if deg == 0:
        return {node: 1.0}
    mu = {node: alpha}
    w = (1.0 - alpha) / deg
    for v in nbrs:
        mu[v] = mu.get(v, 0.0) + w
    return mu

def w1_distance(mu_u, mu_v, D, node_list):
    support = sorted(set(mu_u) | set(mu_v))
    ni = {n: node_list.index(n) for n in support}
    a = np.array([mu_u.get(n, 0.0) for n in support])
    b = np.array([mu_v.get(n, 0.0) for n in support])
    a, b = a/a.sum(), b/b.sum()
    M = np.array([[D[ni[s], ni[t]] for t in support] for s in support])
    return ot.emd2(a, b, M)

def ricci_curvature(G, alpha=0.5):
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    row, col, data = [], [], []
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        row += [i, j]; col += [j, i]; data += [1.0, 1.0]
    A = csr_matrix((data, (row, col)), shape=(n, n))
    D = shortest_path(A, method='D', directed=False)
    curv = {}
    for u, v in G.edges():
        mu_u, mu_v = lazy_dist(G, u, alpha), lazy_dist(G, v, alpha)
        d_uv = D[idx[u], idx[v]]
        w1 = w1_distance(mu_u, mu_v, D, nodes)
        curv[(u, v)] = 1.0 - w1 / d_uv
    return curv


def _filtration_edge_values(G, mode, ricci_cache=None):
    """Returns sorted (u,v,value) triples for the requested filtration
    mode. 'ricci_cache' avoids recomputing curvature repeatedly when a
    combination includes it more than once."""
    edges = list(G.edges())
    if not edges:
        return []
    if mode == 'ricci':
        curv = ricci_cache if ricci_cache is not None else ricci_curvature(G)
        vals = [(u, v, curv.get((u, v), curv.get((v, u)))) for u, v in edges]
    elif mode == 'degree':
        deg = dict(G.degree())
        vals = [(u, v, (deg[u]+deg[v])/2.0) for u, v in edges]
    elif mode == 'label':
        nodes = list(G.nodes())
        has_feat = G.number_of_nodes() > 0 and G.nodes[nodes[0]].get('feat') is not None
        if not has_feat:
            return []
        f_vals = {n: (float(np.argmax(G.nodes[n]['feat'])) if len(G.nodes[n]['feat'])>1
                       else float(G.nodes[n]['feat'][0])) for n in nodes}
        vals = [(u, v, (f_vals[u]+f_vals[v])/2.0) for u, v in edges]
    else:
        raise ValueError(mode)
    return sorted(vals, key=lambda x: x[2])


def compute_betti_only_single_filtration(G, mode, T=10, ricci_cache=None):
    """Betti-only feature block for ONE filtration: [beta_0^(t),
    beta_1^(t)] for t=1..T, i.e. the eigenvalue blocks lambda^(t),
    mu^(t) are never computed at all here (not just zeroed after the
    fact -- genuinely never spent compute on them, though this
    doesn't change the RESULT, only saves time)."""
    edge_vals = _filtration_edge_values(G, mode, ricci_cache=ricci_cache)
    if not edge_vals:
        return np.zeros(T * 2)
    vals = [v for (_, _, v) in edge_vals]
    vmin, vmax = vals[0], vals[-1]
    thresholds = (np.linspace(vmin, vmax, T) if abs(vmax-vmin) > 1e-10
                  else np.linspace(vmin-0.1, vmax+0.1, T))
    feats = []
    for tau in thresholds:
        G_t = nx.Graph()
        G_t.add_nodes_from(G.nodes())
        for u, v, val in edge_vals:
            if val <= tau:
                G_t.add_edge(u, v)
        # K0=K1=1 minimizes wasted eigenvalue computation; we only
        # keep b0,b1 regardless, but hodge_spectra needs a K>=1.
        _, _, b0, b1 = hodge_spectra(G_t, K0=1, K1=1)
        feats.append([float(b0), float(b1)])
    return np.array(feats).flatten()


def compute_betti_only_features(G, modes, T=10):
    """modes: list of filtration mode strings matching the dataset's
    headline combination (e.g. ['ricci'] for MUTAG,
    ['ricci','degree'] for PTC-MR, etc.)"""
    ricci_cache = ricci_curvature(G) if 'ricci' in modes else None
    parts = [compute_betti_only_single_filtration(G, m, T=T, ricci_cache=ricci_cache)
             for m in modes]
    return np.concatenate(parts)


def run_crossval(X, labels):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accs = []
    for tr_idx, te_idx in skf.split(X, labels):
        n_val = max(1, len(tr_idx)//10)
        clf = KernelRidgeClassifier(kernel='arccos')
        clf.fit(X[tr_idx][n_val:], labels[tr_idx][n_val:],
                X[tr_idx][:n_val], labels[tr_idx][:n_val])
        accs.append(clf.score(X[te_idx], labels[te_idx]))
    return np.mean(accs)*100, np.std(accs)*100


# ── Each dataset's ACTUAL headline filtration combination, matching
#    Table 4's "Full pipeline (reported)" row exactly ────────────────
HEADLINE_CONFIGS = {
    'MUTAG':    ['ricci'],
    'PTC_MR':   ['ricci', 'degree'],
    'DHFR':     ['ricci', 'degree', 'label'],
    'NCI1':     ['ricci', 'degree', 'label'],
    'PROTEINS': ['degree', 'label'],   # Degree + SSE, matches C.4's fix
}

# Full-PHS headline numbers already established, for direct comparison
FULL_PHS_HEADLINE = {
    'MUTAG': 91.15, 'PTC_MR': 64.31, 'DHFR': 75.95,
    'NCI1': 74.79, 'PROTEINS': 71.34,
}

if __name__ == '__main__':
    results = []
    for name, modes in HEADLINE_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {name}  (Betti-only, modes={modes})")
        print(f"{'='*60}")
        graphs, labels = load_tu_raw(name)
        print(f"  {len(graphs)} graphs")

        N = len(graphs)
        x0 = compute_betti_only_features(graphs[0], modes)
        X = np.zeros((N, len(x0)), dtype=np.float32)
        for i, G in enumerate(graphs):
            if i % 200 == 0:
                print(f"    {i}/{N}", end='\r')
            X[i] = compute_betti_only_features(G, modes)
        print(f"\n  Betti-only feature matrix: {X.shape} "
              f"(vs. full PHS's much larger dimension)")

        mean, std = run_crossval(X, labels)
        full_phs = FULL_PHS_HEADLINE[name]
        gap = full_phs - mean
        print(f"  Betti-only: {mean:.2f}% +/- {std:.2f}%")
        print(f"  Full PHS:   {full_phs:.2f}%")
        print(f"  Gap (eigenvalues' contribution): {gap:+.2f} points")
        results.append((name, mean, std, full_phs, gap))

    # ── Save results directly into the project folder ───────────────
    out_csv = os.path.join(OUTPUT_DIR, 'betti_only_ablation_results.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'betti_only_mean', 'betti_only_std',
                          'full_phs_mean', 'gap'])
        writer.writerows(results)
    print(f"\n{'='*60}")
    print(f"Saved to {out_csv}")
    print(f"{'='*60}")
    print("\nSummary (positive gap = eigenvalues help; near-zero or")
    print("negative = Betti numbers alone already capture most signal):")
    for name, mean, std, full_phs, gap in results:
        print(f"  {name:10s}: Betti-only {mean:6.2f}%  Full PHS {full_phs:6.2f}%  Gap {gap:+6.2f}")
