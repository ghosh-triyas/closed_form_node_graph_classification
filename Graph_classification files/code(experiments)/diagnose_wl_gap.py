"""
diagnose_wl_gap.py

Isolates why fix_wl_normalization.py's "OLD (unnormalized)" branch
got 73.54% on PROTEINS -- much closer to published (75.0%) -- while
the ORIGINAL run_all_graph_datasets_with_significance.py's WL
computation got 64.82% for the exact same "unnormalized" setup.
Something differs between the two loaders/label-handling, not
normalization. This script checks the most likely cause directly:
whether PROTEINS' actual node labels (the 3-way SSE indicator) are
being correctly used to initialize WL, or silently falling back to
degree-only initial labels.
"""
import sys
sys.path.insert(0, '/Users/X/Desktop/TopoSPR')
import numpy as np
import networkx as nx

d = '/tmp/tu_datasets/PROTEINS'
name = 'PROTEINS'

print("="*60)
print("STEP 1: Does PROTEINS_node_labels.txt exist and load correctly?")
print("="*60)
try:
    with open(f'{d}/{name}_node_labels.txt') as f:
        node_labels = [int(float(x.strip())) for x in f]
    print(f"  File found: {len(node_labels)} node labels loaded")
    print(f"  Distinct label values: {sorted(set(node_labels))}")
    print(f"  (Expect ~3 distinct values for PROTEINS' SSE indicator)")
except FileNotFoundError:
    print("  FILE NOT FOUND -- this alone would explain everything: "
          "WL would silently fall back to degree-only initial labels, "
          "losing all the real node-identity signal.")
    node_labels = None

print("\n" + "="*60)
print("STEP 2: Load graphs BOTH ways, compare initial WL labels")
print("="*60)

with open(f'{d}/{name}_graph_indicator.txt') as f:
    graph_ind = [int(float(x.strip())) for x in f]
with open(f'{d}/{name}_A.txt') as f:
    edges_all = [tuple(int(float(x)) for x in l.strip().split(','))
                 for l in f]
with open(f'{d}/{name}_graph_labels.txt') as f:
    graph_labels_raw = [int(float(x.strip())) for x in f]

node_to_graph = {i+1: graph_ind[i] for i in range(len(graph_ind))}
n_graphs = max(graph_ind)

def build_graph(g_id, with_labels):
    G = nx.Graph()
    nodes_g = [i+1 for i, gi in enumerate(graph_ind) if gi == g_id]
    for n in nodes_g:
        if with_labels and node_labels is not None:
            G.add_node(n, feat=np.array([node_labels[n-1]], dtype=np.float32))
        else:
            G.add_node(n)
    for u, v in edges_all:
        if node_to_graph.get(u) == g_id and u != v:
            G.add_edge(u, v)
    return G

G_with = build_graph(1, with_labels=True)
G_without = build_graph(1, with_labels=False)

def initial_wl_labels(G):
    current = {v: G.degree(v) for v in G.nodes()}
    if G.number_of_nodes() > 0:
        sample_node = list(G.nodes())[0]
        if G.nodes[sample_node].get('feat') is not None:
            current = {v: int(round(float(G.nodes[v]['feat'][0])))
                       for v in G.nodes()}
    return current

labels_with = initial_wl_labels(G_with)
labels_without = initial_wl_labels(G_without)

print(f"  Graph 1, WITH node-label loading:")
print(f"    Initial labels (first 5): {dict(list(labels_with.items())[:5])}")
print(f"    Distinct values: {sorted(set(labels_with.values()))}")

print(f"\n  Graph 1, WITHOUT node-label loading (degree fallback):")
print(f"    Initial labels (first 5): {dict(list(labels_without.items())[:5])}")
print(f"    Distinct values: {sorted(set(labels_without.values()))}")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)
if labels_with != labels_without:
    print("  CONFIRMED: node-label loading changes WL's initial labels.")
    print("  This is almost certainly why the two scripts gave different")
    print("  results -- check whether the ORIGINAL")
    print("  run_all_graph_datasets_with_significance.py script's loader")
    print("  correctly reaches the 'with_labels=True' branch for PROTEINS,")
    print("  or silently falls through to degree-only labels due to a")
    print("  bug in its node_attrs/node_labels priority-check logic.")
else:
    print("  Node labels don't change initial WL labels here -- the real")
    print("  cause is something else (SVM C grid, h-iteration count, or")
    print("  fold-split randomness). Report this output for further help.")
