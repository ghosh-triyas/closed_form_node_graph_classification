"""
Ollivier-Ricci Curvature for graphs.

For edge (u, v):
    κ(u, v) = 1 - W_1(μ_u, μ_v)

where μ_u is the α-lazy random walk distribution at node u:
    μ_u(w) = α       if w == u
    μ_u(w) = (1-α)/d(u)  if w ~ u  (neighbour)
    μ_u(w) = 0       otherwise

W_1 is the 1-Wasserstein (Earth Mover's) distance using shortest-path
graph distances as the ground metric.

Reference: Ollivier, Y. (2009). Ricci curvature of Markov chains on
           metric spaces. J. Functional Analysis.
"""

import numpy as np
import networkx as nx
import ot                          # Python Optimal Transport (POT)
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix


def _lazy_distribution(G: nx.Graph, node, alpha: float = 0.5) -> dict:
    """
    α-lazy random walk distribution at `node`.

    μ_u(w) = α       if w == node
             (1-α)/deg  if w is a neighbour of node
             0          otherwise

    If node is isolated (degree 0), μ_u = {node: 1.0}.
    """
    nbrs = list(G.neighbors(node))
    deg  = len(nbrs)
    if deg == 0:
        return {node: 1.0}
    mu = {node: alpha}
    w  = (1.0 - alpha) / deg
    for v in nbrs:
        mu[v] = mu.get(v, 0.0) + w
    return mu


def _w1_distance(mu_u: dict, mu_v: dict,
                 dist_matrix: np.ndarray,
                 node_list: list) -> float:
    """
    1-Wasserstein distance between two discrete distributions
    using graph shortest-path distances as ground metric.

    Uses the POT library for efficient linear-programming OT.
    """
    # Collect all support nodes
    support = sorted(set(mu_u) | set(mu_v))
    n       = len(support)
    idx     = {node: i for i, node in enumerate(support)}
    ni      = {node: node_list.index(node) for node in support}

    # Build weight vectors — normalise to sum to 1 (required by POT)
    a = np.array([mu_u.get(node, 0.0) for node in support], dtype=np.float64)
    b = np.array([mu_v.get(node, 0.0) for node in support], dtype=np.float64)
    a = a / (a.sum() + 1e-12)
    b = b / (b.sum() + 1e-12)

    # Build cost matrix (submatrix of dist_matrix for support nodes)
    M = np.array([[dist_matrix[ni[s], ni[t]] for t in support]
                  for s in support], dtype=np.float64)

    # Solve OT — emd2 returns the scalar W1 cost
    w1 = ot.emd2(a, b, M)
    return w1


def compute_ollivier_ricci(G: nx.Graph,
                            alpha: float = 0.5,
                            verbose: bool = False) -> dict:
    """
    Compute Ollivier-Ricci curvature for every edge in G.

    Returns:
        curvature: dict {(u, v): κ(u, v)} for all edges
                   (both (u,v) and (v,u) are stored for convenience)

    Args:
        G:       undirected NetworkX graph
        alpha:   laziness parameter (default 0.5)
        verbose: print progress
    """
    nodes     = list(G.nodes())
    n         = len(nodes)
    node_idx  = {node: i for i, node in enumerate(nodes)}

    # Build adjacency matrix for shortest-path computation
    row, col, data = [], [], []
    for u, v in G.edges():
        i, j = node_idx[u], node_idx[v]
        row += [i, j]; col += [i, j]
        # Use w.get('weight', 1.0) if weighted, else 1.0
        w = G[u][v].get('weight', 1.0)
        data += [w, w]
    A_sp = csr_matrix((data, (row, col)), shape=(n, n))

    # All-pairs shortest paths
    D = shortest_path(A_sp, method='D', directed=False)

    curvature = {}
    edges = list(G.edges())
    for idx_e, (u, v) in enumerate(edges):
        if verbose and idx_e % 50 == 0:
            print(f"  Ricci: edge {idx_e}/{len(edges)}", end='\r')

        mu_u = _lazy_distribution(G, u, alpha)
        mu_v = _lazy_distribution(G, v, alpha)

        d_uv = D[node_idx[u], node_idx[v]]
        if d_uv < 1e-10:
            kappa = 1.0
        else:
            w1   = _w1_distance(mu_u, mu_v, D, nodes)
            kappa = 1.0 - w1 / d_uv

        curvature[(u, v)] = kappa
        curvature[(v, u)] = kappa

    if verbose:
        print()
    return curvature


def curvature_to_edge_attr(G: nx.Graph,
                            curvature: dict) -> nx.Graph:
    """
    Attach curvature as edge attribute 'ricci' to graph G.
    Returns G with edge attributes set.
    """
    G2 = G.copy()
    for (u, v), kappa in curvature.items():
        if G2.has_edge(u, v):
            G2[u][v]['ricci'] = kappa
    return G2
