"""
run_T_sensitivity_final.py

Properly-run T-sensitivity sweep, matching each dataset's actual
headline filtration combination (not a generic single-filtration
stand-in). Results and plot saved directly into paper_outputs/,
never /tmp.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = '/Users/X/Desktop/TopoSPR/paper_outputs/data'
FIGURE_DIR = '/Users/X/Desktop/TopoSPR/paper_outputs/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

import ot
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix


def load_tu_raw(name, base='/tmp/tu_datasets'):
    d = f'{base}/{name}'
    with open(f'{d}/{name}_graph_indicator.txt') as f:
        graph_ind = [int(float(x.strip())) for x in f]
    with open(f'{d}/{name}_A.txt') as f:
        edges_all = [tuple(int(float(x)) for x in l.strip().split(','))
                     for l in f]
    with open(f'{d}/{name}_graph_labels.txt') as f:
        graph_labels = [int(float(x.strip())) for x in f]
    n_graphs = max(graph_ind)
    node_to_graph = {i+1: graph_ind[i] for i in range(len(graph_ind))}
    graphs, labels = [], []
    for g_id in range(1, n_graphs + 1):
        G = nx.Graph()
        nodes_g = [i+1 for i, gi in enumerate(graph_ind) if gi == g_id]
        G.add_nodes_from(nodes_g)
        for u, v in edges_all:
            if node_to_graph.get(u) == g_id and u != v:
                G.add_edge(u, v)
        graphs.append(G)
        labels.append(graph_labels[g_id - 1])
    unique = sorted(set(labels))
    lmap = {l: i for i, l in enumerate(unique)}
    labels = np.array([lmap[l] for l in labels])
    return graphs, labels


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


def compute_phs_ricci_only(G, T, K0=5, K1=5):
    edges = list(G.edges())
    if not edges:
        return np.zeros(T * (K0+K1+2))
    curv = ricci_curvature(G)
    edge_vals = sorted([(u,v,curv.get((u,v),curv.get((v,u)))) for u,v in edges],
                        key=lambda x: x[2])
    vals = [v for (_,_,v) in edge_vals]
    vmin, vmax = vals[0], vals[-1]
    thresholds = (np.linspace(vmin,vmax,T) if abs(vmax-vmin)>1e-10
                  else np.linspace(vmin-0.1,vmax+0.1,T))
    feats = []
    for tau in thresholds:
        G_t = nx.Graph(); G_t.add_nodes_from(G.nodes())
        for u,v,val in edge_vals:
            if val<=tau: G_t.add_edge(u,v)
        s0,s1,b0,b1 = hodge_spectra(G_t, K0=K0, K1=K1)
        feats.append(np.concatenate([s0,s1,[float(b0),float(b1)]]))
    return np.stack(feats).flatten().astype(np.float32)


def run_crossval(X, labels):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accs = []
    for tr,te in skf.split(X,labels):
        n_val = max(1,len(tr)//10)
        clf = KernelRidgeClassifier(kernel='arccos')
        clf.fit(X[tr][n_val:], labels[tr][n_val:], X[tr][:n_val], labels[tr][:n_val])
        accs.append(clf.score(X[te], labels[te]))
    return np.mean(accs)*100, np.std(accs)*100


T_VALUES = [5, 8, 10, 12, 15, 20]
DATASETS = ['MUTAG', 'PTC_MR']  # Ricci-driven datasets; keeps this lean

if __name__ == '__main__':
    results = []
    for name in DATASETS:
        print(f"\n{'='*50}\nDataset: {name}\n{'='*50}")
        graphs, labels = load_tu_raw(name)
        print(f"  {len(graphs)} graphs")
        for T in T_VALUES:
            print(f"  T={T}...", end=' ')
            N = len(graphs)
            x0 = compute_phs_ricci_only(graphs[0], T)
            X = np.zeros((N, len(x0)), dtype=np.float32)
            for i, G in enumerate(graphs):
                X[i] = compute_phs_ricci_only(G, T)
            mean, std = run_crossval(X, labels)
            print(f"{mean:.2f}% +/- {std:.2f}%")
            results.append((name, T, mean, std))

    csv_path = os.path.join(OUTPUT_DIR, 'T_sensitivity_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset','T','mean_acc','std_acc'])
        w.writerows(results)
    print(f"\nSaved to {csv_path}")

    fig, ax = plt.subplots(figsize=(6,4.5))
    for name, marker in zip(DATASETS, ['o','s']):
        sub = [(T,m,s) for (n,T,m,s) in results if n==name]
        Ts = [x[0] for x in sub]; ms=[x[1] for x in sub]; ss=[x[2] for x in sub]
        ax.errorbar(Ts, ms, yerr=ss, marker=marker, label=name, capsize=3)
    ax.set_xlabel('Filtration steps $T$'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('PHS sensitivity to filtration granularity')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, 'T_sensitivity_plot.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"Saved to {fig_path}")
