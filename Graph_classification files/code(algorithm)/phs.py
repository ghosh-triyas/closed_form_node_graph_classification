"""
Persistent Hodge Spectrum (PHS).

Pipeline for one graph G:
    1. Compute Ollivier-Ricci curvature on all edges
    2. Sort edges by curvature (sublevel filtration:
       most negative first → most positive last)
    3. At T evenly-spaced filtration steps, build the subgraph G_t
       consisting of all edges with curvature ≤ threshold_t
    4. At each step t, compute:
         spec_L0^(t)  : K0 smallest eigenvalues of node Laplacian
         spec_L1^(t)  : K1 smallest eigenvalues of edge Hodge Laplacian
                        (optionally chemistry-weighted — see Option B below)
         β_0^(t), β_1^(t): Betti numbers
    5. Stack into matrix PHS ∈ R^{T × (K0 + K1 + 2)}
    6. Flatten → fixed-length vector φ(G) ∈ R^{T*(K0+K1+2)}

The Persistent Hodge Spectrum is strictly more expressive than
standard persistent homology:
    - Standard PH tracks only β_0, β_1 (nullities)
    - PHS tracks the FULL spectrum (nullities + non-zero eigenvalues)
    - Two graphs with identical persistence diagrams can have
      different PHS (different non-zero eigenvalues)
    → Expressivity hierarchy:
      Betti numbers ⊊ Persistence diagrams ⊊ PHS

────────────────────────────────────────────────────────────────────
NEW (Option B): Chemistry-weighted PHS
────────────────────────────────────────────────────────────────────
The Ricci-curvature FILTRATION ORDER is unaffected by this option —
edges still enter sorted purely by curvature, exactly as before.
What changes is what L_1's eigenvalues MEASURE at each step: with
feature_weighting != 'none', a Carbon-only ring and a chemically
different (e.g. Oxygen-substituted) ring of the IDENTICAL shape now
produce DIFFERENT non-zero L_1 eigenvalues, even though they have
identical curvature and hence identical filtration order/Betti
numbers. This directly targets datasets (e.g. NCI1) where the label
depends on WHICH atoms form a ring, not just ring shape/count alone.

Node weights are computed ONCE per graph (they don't depend on which
edges are present at a given filtration step — only on node features,
which never change across steps) and reused at every one of the T
filtration steps.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Optional, Dict
from ricci import compute_ollivier_ricci
from hodge import (hodge_spectra, compute_node_weights,
                    compute_label_frequencies,
                    compute_label_cooccurrence_embedding,
                    compute_edge_weights_from_embedding)


# ────────────────────────────────────────────────────────────────────
# Filtration
# ────────────────────────────────────────────────────────────────────

def build_filtration_steps(curvature: dict,
                            T: int = 10
                            ) -> Tuple[np.ndarray, List[Tuple]]:
    """
    Build T evenly-spaced filtration thresholds from the Ricci curvatures.

    Sublevel filtration: at threshold τ, include all edges with κ ≤ τ.

    Returns:
        thresholds: np.ndarray [T], evenly spaced from min to max curvature
        sorted_edges: list of (u, v, κ) sorted by κ ascending
    """
    # Get unique directed edges (take each undirected edge once)
    seen  = set()
    items = []
    for (u, v), kappa in curvature.items():
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            items.append((u, v, kappa))

    if not items:
        return np.zeros(T), []

    items.sort(key=lambda x: x[2])          # sort by curvature ascending
    kappas = [k for _, _, k in items]

    kmin, kmax = kappas[0], kappas[-1]
    if abs(kmax - kmin) < 1e-10:
        # All edges have same curvature — single step
        thresholds = np.linspace(kmin - 0.01, kmax + 0.01, T)
    else:
        thresholds = np.linspace(kmin, kmax, T)

    return thresholds, items


def get_subgraph_at_threshold(G: nx.Graph,
                               sorted_edges: List[Tuple],
                               threshold: float) -> nx.Graph:
    """
    Build the subgraph containing all nodes and edges with κ ≤ threshold.

    All original nodes are kept (even isolated ones) to maintain
    consistent node identity across filtration steps.
    """
    G_t = nx.Graph()
    G_t.add_nodes_from(G.nodes(data=True))   # keep all nodes (+ their
                                              # 'feat' attribute, needed
                                              # for Option B weighting)

    for (u, v, kappa) in sorted_edges:
        if kappa <= threshold:
            G_t.add_edge(u, v, ricci=kappa)

    return G_t


# ────────────────────────────────────────────────────────────────────
# Persistent Hodge Spectrum
# ────────────────────────────────────────────────────────────────────

def compute_phs(G: nx.Graph,
                T: int = 10,
                K0: int = 5,
                K1: int = 5,
                alpha_ricci: float = 0.5,
                include_betti: bool = True,
                feature_weighting: str = 'none',
                label_to_element: Optional[Dict[int, str]] = None,
                label_counts: Optional[Dict] = None,
                label_total: Optional[int] = None,
                cooccurrence_embedding: Optional[np.ndarray] = None,
                verbose: bool = False
                ) -> np.ndarray:
    """
    Compute the Persistent Hodge Spectrum for graph G.

    Args:
        feature_weighting: ... (see previous modes) ...
                            'cooccurrence'      -> MOST EXPRESSIVE option.
                                                    Weight each EDGE by
                                                    the distance between
                                                    its two endpoints'
                                                    label co-occurrence
                                                    embeddings (see
                                                    compute_label_
                                                    cooccurrence_embedding).
                                                    Distinguishes DIFFERENT
                                                    rare atom types from
                                                    EACH OTHER, unlike
                                                    'rarity' (which only
                                                    separates common from
                                                    uncommon). Requires
                                                    `cooccurrence_embedding`,
                                                    computed ONCE over the
                                                    full dataset.
        cooccurrence_embedding: np.ndarray [n_labels, num_dims], output
                            of compute_label_cooccurrence_embedding,
                            required only when feature_weighting==
                            'cooccurrence'.
        (other args unchanged from previous version)

    Returns:
        phs_vector: np.ndarray [T * (K0 + K1 + 2*include_betti)]
    """
    if G.number_of_nodes() == 0:
        dim = T * (K0 + K1 + 2 * int(include_betti))
        return np.zeros(dim)

    # Step 0 (Option B): compute node/edge weights ONCE — they don't
    # change across filtration steps.
    node_weights = None
    edge_weights = None
    if feature_weighting == 'cooccurrence':
        if cooccurrence_embedding is None:
            raise ValueError(
                "feature_weighting='cooccurrence' requires "
                "cooccurrence_embedding (call "
                "compute_label_cooccurrence_embedding(all_graphs) once "
                "over your full dataset first).")
        edge_weights = compute_edge_weights_from_embedding(
            G, cooccurrence_embedding)
    elif feature_weighting != 'none':
        node_weights = compute_node_weights(
            G, mode=feature_weighting,
            label_to_element=label_to_element,
            label_counts=label_counts,
            label_total=label_total)

    # Step 1: Ollivier-Ricci curvature
    if verbose:
        print(f"  Computing Ricci curvature ({G.number_of_edges()} edges)...")
    curvature = compute_ollivier_ricci(G, alpha=alpha_ricci)

    if not curvature:
        # No edges — return zeros
        dim = T * (K0 + K1 + 2 * int(include_betti))
        return np.zeros(dim)

    # Step 2: Build filtration steps
    thresholds, sorted_edges = build_filtration_steps(curvature, T=T)

    # Step 3: Compute spectra at each filtration step
    step_features = []
    for t_idx, tau in enumerate(thresholds):
        G_t  = get_subgraph_at_threshold(G, sorted_edges, tau)
        spec0, spec1, b0, b1 = hodge_spectra(
            G_t, K0=K0, K1=K1, node_weights=node_weights,
            edge_weights=edge_weights)

        if include_betti:
            feat = np.concatenate([spec0, spec1,
                                    [float(b0), float(b1)]])
        else:
            feat = np.concatenate([spec0, spec1])

        step_features.append(feat)

    phs_matrix = np.stack(step_features, axis=0)   # [T, feat_dim]
    phs_vector = phs_matrix.flatten()               # [T * feat_dim]

    return phs_vector.astype(np.float32)


def compute_phs_batch(graphs: List[nx.Graph],
                       T: int = 10,
                       K0: int = 5,
                       K1: int = 5,
                       alpha_ricci: float = 0.5,
                       include_betti: bool = True,
                       feature_weighting: str = 'none',
                       label_to_element: Optional[Dict[int, str]] = None,
                       cooccurrence_num_dims: int = 4,
                       verbose: bool = True
                       ) -> np.ndarray:
    """
    Compute PHS for a list of graphs.

    If feature_weighting=='rarity', label frequencies are computed
    ONCE over the entire `graphs` list before the per-graph loop.

    If feature_weighting=='cooccurrence', the label co-occurrence
    spectral embedding is computed ONCE over the entire `graphs` list
    before the per-graph loop (this is the most expressive option —
    see compute_label_cooccurrence_embedding's docstring).

    Both precomputation steps use only node features and graph
    structure — never graph-level class labels — so there is no
    train/test leakage, exactly analogous to how Ricci curvature
    itself is computed over all graphs together.

    Returns:
        X: np.ndarray [N, D] where D = T*(K0+K1+2)
    """
    N   = len(graphs)
    dim = T * (K0 + K1 + 2 * int(include_betti))
    X   = np.zeros((N, dim), dtype=np.float32)

    label_counts, label_total = None, None
    cooc_embedding = None

    if feature_weighting == 'rarity':
        label_counts, label_total = compute_label_frequencies(graphs)
        if verbose:
            print(f"  Rarity weighting: {len(label_counts)} distinct "
                  f"labels found across {label_total} total node "
                  f"observations.")

    elif feature_weighting == 'cooccurrence':
        cooc_embedding = compute_label_cooccurrence_embedding(
            graphs, num_dims=cooccurrence_num_dims)
        if verbose:
            print(f"  Co-occurrence embedding: {cooc_embedding.shape[0]} "
                  f"labels embedded into {cooc_embedding.shape[1]} dims.")

    for i, G in enumerate(graphs):
        if verbose and (i % 10 == 0 or i == N - 1):
            print(f"  PHS: graph {i+1}/{N}", end='\r')
        X[i] = compute_phs(G, T=T, K0=K0, K1=K1,
                            alpha_ricci=alpha_ricci,
                            include_betti=include_betti,
                            feature_weighting=feature_weighting,
                            label_to_element=label_to_element,
                            label_counts=label_counts,
                            label_total=label_total,
                            cooccurrence_embedding=cooc_embedding)

    if verbose:
        print(f"\n  PHS complete. Feature matrix: {X.shape}")
    return X


# ────────────────────────────────────────────────────────────────────
# Persistence diagram extraction (for baseline comparison)
# ────────────────────────────────────────────────────────────────────

def compute_persistence_diagram(G: nx.Graph,
                                  curvature: dict,
                                  T: int = 10
                                  ) -> Tuple[List, List]:
    """
    Extract H_0 and H_1 persistence pairs from Ricci filtration.

    H_0 pair (b, d): a connected component is born at b and dies at d
                     when it merges with an older component.
    H_1 pair (b, d): a cycle is born at b (when its completing edge
                     is added) and dies at d (when it gets filled).

    Returns:
        dgm0: list of (birth, death) pairs for H_0
        dgm1: list of (birth, death) pairs for H_1
    """
    thresholds, sorted_edges = build_filtration_steps(curvature, T=100)

    # Track components via Union-Find
    parent = {node: node for node in G.nodes()}
    rank   = {node: 0    for node in G.nodes()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False, None, None
        if rank[rx] < rank[ry]: rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]: rank[rx] += 1
        return True, rx, ry

    # Birth time for each component = -inf (all nodes present from start)
    birth_time = {node: -np.inf for node in G.nodes()}

    dgm0 = []
    dgm1 = []

    for (u, v, kappa) in sorted_edges:
        merged, rx, ry = union(u, v)
        if merged:
            # H_0 death: younger component dies, older survives
            dgm0.append((kappa, kappa))   # birth=-inf simplified
        else:
            # H_1 birth: cycle created (completing edge)
            # H_1 death: approximated by next threshold or inf
            dgm1.append((kappa, np.inf))

    return dgm0, dgm1
