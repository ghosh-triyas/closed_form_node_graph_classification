"""
run_all_graph_datasets_with_significance.py

Runs BOTH PHS (your method, best config per dataset) AND our own WL
subtree kernel (see run_wl_kernel_baseline.py from earlier) on all
five graph classification datasets, under the IDENTICAL 10-fold split
(random_state=42), saving per-fold scores for both -- enabling a
genuine paired Wilcoxon significance test against WL specifically.

We use OUR OWN WL run rather than the literature-reported WL numbers
because only a self-run baseline gives us the per-fold scores a valid
significance test requires. GCN/GIN/GraphSAGE remain literature-only
comparisons (mean +/- std) in the main results table -- these CANNOT
be significance-tested without re-running them ourselves under
identical folds, which is a much larger undertaking (see
run_gnn_baselines.py) than is realistic to redo for every dataset
here. State this scope limitation explicitly in the paper.
"""
import sys
sys.path.insert(0, '/Users/X/Desktop/TopoSPR')
import numpy as np
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

from phs import compute_phs
from hodge import hodge_spectra
from classifier import KernelRidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from collections import Counter

sys.path.insert(0, '.')
from save_perfold_results import save_fold_scores


def load_tu_raw(name, base='/tmp/tu_datasets'):
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


def compute_degree_phs(G, T=10, K0=5, K1=5):
    edges = list(G.edges())
    if not edges:
        return np.zeros(T * (K0 + K1 + 2))
    deg = dict(G.degree())
    edge_vals = sorted([(u, v, (deg[u]+deg[v])/2.0)
                        for u, v in edges], key=lambda x: x[2])
    vals = [x[2] for x in edge_vals]
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
        s0, s1, b0, b1 = hodge_spectra(G_t, K0=K0, K1=K1)
        feats.append(np.concatenate([s0, s1, [float(b0), float(b1)]]))
    return np.stack(feats).flatten().astype(np.float32)


def compute_feat_phs(G, T=10, K0=5, K1=5, mode='label'):
    edges = list(G.edges())
    if not edges:
        return np.zeros(T * (K0 + K1 + 2))
    nodes = list(G.nodes())
    has_feat = G.number_of_nodes() > 0 and G.nodes[nodes[0]].get('feat') is not None
    if not has_feat:
        return np.zeros(T * (K0 + K1 + 2))
    if mode == 'label':
        f_vals = {n: float(np.argmax(G.nodes[n]['feat'])) if len(G.nodes[n]['feat'])>1
                   else float(G.nodes[n]['feat'][0]) for n in nodes}
    else:
        f_vals = {n: float(np.linalg.norm(G.nodes[n]['feat'])) for n in nodes}
    edge_vals = sorted([(u, v, (f_vals[u]+f_vals[v])/2.0)
                        for u, v in edges], key=lambda x: x[2])
    vals = [x[2] for x in edge_vals]
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
        s0, s1, b0, b1 = hodge_spectra(G_t, K0=K0, K1=K1)
        feats.append(np.concatenate([s0, s1, [float(b0), float(b1)]]))
    return np.stack(feats).flatten().astype(np.float32)


def compute_phs_features(G, T, K0, K1, mode):
    parts = [compute_phs(G, T=T, K0=K0, K1=K1)]
    if 'degree' in mode:
        parts.append(compute_degree_phs(G, T=T, K0=K0, K1=K1))
    if 'label' in mode:
        parts.append(compute_feat_phs(G, T=T, K0=K0, K1=K1, mode='label'))
    if 'feat' in mode:
        parts.append(compute_feat_phs(G, T=T, K0=K0, K1=K1, mode='feat'))
    return np.concatenate(parts)


# ── WL kernel (self-run baseline, gives real per-fold data) ─────────
def wl_subtree_labels(G, num_iterations):
    current = {v: G.degree(v) for v in G.nodes()}
    if G.number_of_nodes() > 0:
        sample_node = list(G.nodes())[0]
        if G.nodes[sample_node].get('feat') is not None:
            current = {v: (int(np.argmax(G.nodes[v]['feat']))
                       if len(G.nodes[v]['feat'])>1
                       else int(round(float(G.nodes[v]['feat'][0]))))
                       for v in G.nodes()}
    all_iter_labels = [dict(current)]
    for _ in range(num_iterations):
        new_labels = {}
        for v in G.nodes():
            nbr_multiset = tuple(sorted(current[u] for u in G.neighbors(v)))
            new_labels[v] = (current[v], nbr_multiset)
        current = new_labels
        all_iter_labels.append(dict(current))
    return all_iter_labels


def compute_wl_kernel_matrix(graphs, num_iterations=4, verbose=True):
    N = len(graphs)
    all_labels = [wl_subtree_labels(G, num_iterations) for G in graphs]
    K = np.zeros((N, N), dtype=np.float64)
    for h in range(num_iterations + 1):
        label_to_id = {}
        graph_counts = []
        for g_labels in all_labels:
            counts = Counter()
            for v, lbl in g_labels[h].items():
                if lbl not in label_to_id:
                    label_to_id[lbl] = len(label_to_id)
                counts[label_to_id[lbl]] += 1
            graph_counts.append(counts)
        n_labels_h = len(label_to_id)
        M = np.zeros((N, n_labels_h), dtype=np.float64)
        for i, counts in enumerate(graph_counts):
            for lbl_id, c in counts.items():
                M[i, lbl_id] = c
        K += M @ M.T
    return K


def run_phs_crossval(X, labels, dataset_name):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accs = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, labels)):
        n_val = max(1, len(tr_idx)//10)
        clf = KernelRidgeClassifier(kernel='arccos')
        clf.fit(X[tr_idx][n_val:], labels[tr_idx][n_val:],
                X[tr_idx][:n_val], labels[tr_idx][:n_val])
        acc = clf.score(X[te_idx], labels[te_idx])
        accs.append(acc)
        print(f"    Fold {fold+1:2d}: {acc*100:.2f}%")
    save_fold_scores(dataset_name, 'PHS', accs)
    print(f"  PHS: {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%")
    return accs


def run_wl_crossval(K, labels, dataset_name, C_grid=None):
    C_grid = C_grid or [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accs = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(K, labels)):
        n_val = max(1, len(tr_idx)//10)
        val_idx, tr_idx2 = tr_idx[:n_val], tr_idx[n_val:]
        best_val_acc, best_C = -1, C_grid[0]
        for C in C_grid:
            K_tr = K[np.ix_(tr_idx2, tr_idx2)]
            K_val = K[np.ix_(val_idx, tr_idx2)]
            clf = SVC(kernel='precomputed', C=C)
            clf.fit(K_tr, labels[tr_idx2])
            val_acc = clf.score(K_val, labels[val_idx])
            if val_acc > best_val_acc:
                best_val_acc, best_C = val_acc, C
        K_tr_full = K[np.ix_(tr_idx, tr_idx)]
        K_te = K[np.ix_(te_idx, tr_idx)]
        clf = SVC(kernel='precomputed', C=best_C)
        clf.fit(K_tr_full, labels[tr_idx])
        acc = clf.score(K_te, labels[te_idx])
        accs.append(acc)
        print(f"    Fold {fold+1:2d}: {acc*100:.2f}%")
    save_fold_scores(dataset_name, 'WL_selfrun', accs)
    print(f"  WL (self-run): {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%")
    return accs


CONFIGS = {
    'MUTAG':    {'T': 8,  'K0': 5, 'K1': 5, 'mode': 'ricci'},
    'PTC_MR':   {'T': 10, 'K0': 5, 'K1': 5, 'mode': 'ricci+degree'},
    'DHFR':     {'T': 10, 'K0': 5, 'K1': 5, 'mode': 'ricci+degree+feat'},
    'NCI1':     {'T': 10, 'K0': 5, 'K1': 5, 'mode': 'ricci+degree+label'},
    'PROTEINS': {'T': 12, 'K0': 5, 'K1': 5, 'mode': 'ricci+degree'},
}

if __name__ == '__main__':
    for name, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")
        graphs, labels = load_tu_raw(name)
        print(f"  {len(graphs)} graphs")

        print("  Computing PHS features...")
        N = len(graphs)
        x0 = compute_phs_features(graphs[0], T=cfg['T'], K0=cfg['K0'],
                                    K1=cfg['K1'], mode=cfg['mode'])
        X = np.zeros((N, len(x0)), dtype=np.float32)
        for i, G in enumerate(graphs):
            if i % 200 == 0:
                print(f"    {i}/{N}", end='\r')
            X[i] = compute_phs_features(G, T=cfg['T'], K0=cfg['K0'],
                                          K1=cfg['K1'], mode=cfg['mode'])
        print(f"\n  PHS feature matrix: {X.shape}")

        print("  Running PHS cross-validation...")
        run_phs_crossval(X, labels, name)

        print("  Computing WL kernel matrix (self-run baseline)...")
        K = compute_wl_kernel_matrix(graphs, num_iterations=4)
        print("  Running WL cross-validation...")
        run_wl_crossval(K, labels, name)

    print(f"\n{'='*60}")
    print("All done. Per-fold results saved to /tmp/perfold_results/")
    print("Run run_significance_tests.py next to compute Wilcoxon p-values.")
    print(f"{'='*60}")
