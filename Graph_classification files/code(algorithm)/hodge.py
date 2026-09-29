"""
Hodge Laplacian construction and spectral computation.

The Hodge 1-Laplacian L_1 acts on edge signals f: E -> R.
It decomposes as:

    L_1 = L_1^down + L_1^up
        = B_1^T B_1  +  B_2 B_2^T

where:
    B_1 ∈ R^{|V| x |E|}  — node-edge incidence matrix (boundary operator ∂_1)
    B_2 ∈ R^{|E| x |T|}  — edge-triangle incidence matrix (boundary operator ∂_2)
    T   = set of triangles (2-simplices / filled 3-cliques)

Key property:
    ker(L_1) = cycle space  →  dim(ker L_1) = β_1 (first Betti number)
    Non-zero eigenvalues encode geometric complexity of cycles.

If no triangles exist: L_1 = B_1^T B_1 (upper Laplacian only).
If triangles exist:    L_1 = B_1^T B_1 + B_2 B_2^T.

References:
    Eckmann, B. (1944). Harmonische Funktionen und Randwertaufgaben.
    Lim, L.H. (2020). Hodge Laplacians on graphs. SIAM Review.
    Schaub et al. (2020). Random walks on simplicial complexes.

────────────────────────────────────────────────────────────────────
NEW (Option B): Chemistry-weighted Hodge 1-Laplacian
────────────────────────────────────────────────────────────────────
Standard (combinatorial) L_1 treats every node/edge/triangle as
geometrically/chemically identical — B_1, B_2 only ever contain
{+1, -1, 0}. Two rings with the SAME shape but DIFFERENT atom types
(e.g. an all-Carbon hexagon vs. an Oxygen-substituted hexagon) are
therefore INVISIBLE to L_1's eigenvalues, even though chemically
very different.

This module adds an OPTIONAL node-weighting step:
    B_1_weighted[u, e] = mu(u) * B_1[u, e]      (scale by NODE weight)
    B_2_weighted[:, T] = nu(T) * B_2[:, T]       (scale by TRIANGLE weight,
                                                   nu(T) = geometric mean of
                                                   the 3 node weights)
    L_1_weighted = B_1_weighted^T B_1_weighted + B_2_weighted B_2_weighted^T

mu(u) can be:
    - Ollivier/chemical "electronegativity" of the atom at node u
      (requires a verified label->element mapping for the dataset)
    - L2 norm of the node's raw feature vector (dataset-agnostic,
      works for ANY dataset with continuous OR categorical features,
      requires no domain knowledge)

L_0 is intentionally left UNWEIGHTED in this module — only L_1's
"cycle quality" measurement is made chemistry-aware. Connectivity
(beta_0) is unaffected.

When node_weights=None (the default), every function in this module
reproduces EXACTLY the original unweighted behaviour — this is a
strictly backward-compatible, opt-in extension.
"""

import numpy as np
import networkx as nx
from scipy import linalg
from typing import List, Tuple, Optional, Dict


# ────────────────────────────────────────────────────────────────────
# Chemistry constants (Option B)
# ────────────────────────────────────────────────────────────────────

# Pauling electronegativity scale — standard, well-established chemistry
# constants. Confident for all elements listed.
ELECTRONEGATIVITY: Dict[str, float] = {
    'H': 2.20, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
    'Br': 2.96, 'I': 2.66,
}

# MUTAG's commonly-cited 7-label convention (atom-type index -> element),
# repeated across several benchmark papers analysing this dataset.
#
# ⚠ VERIFY before using for published results — this is the conventional
#   ordering reported in the literature, not independently re-derived
#   from MUTAG's raw files in this session. Cross-check against the
#   dataset's original documentation if exact correctness is critical.
MUTAG_LABEL_TO_ELEMENT: Dict[int, str] = {
    0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br',
}

# NCI1 has 37 atom-type labels. No verified label->element mapping is
# provided here — fabricating one risks silently wrong chemistry.
# Use mode='feat_norm' for NCI1 (dataset-agnostic, no mapping required)
# until you have confirmed the exact mapping from the dataset's source.
NCI1_LABEL_TO_ELEMENT: Dict[int, str] = {}   # intentionally empty — fill in
                                              # only after verification


def compute_label_frequencies(graphs: List) -> Tuple[Dict, int]:
    """
    Count occurrences of each raw node label across an ENTIRE dataset
    (all graphs, both train and test — this uses only node features/
    structure, never graph-level CLASS labels, so there is no leakage,
    exactly analogous to how PHS itself processes all graphs together).

    Args:
        graphs: list of NetworkX graphs

    Returns:
        counts: dict {label: occurrence count across all graphs}
        total:  int, total number of (node, label) observations
    """
    from collections import Counter
    counts = Counter()
    total  = 0
    for G in graphs:
        for v in G.nodes():
            feat = G.nodes[v].get('feat')
            if feat is None:
                lbl = None
            else:
                arr = np.asarray(feat).flatten()
                lbl = int(np.argmax(arr)) if arr.size > 1 \
                      else int(round(float(arr[0])))
            counts[lbl] += 1
            total += 1
    return dict(counts), total


def compute_label_cooccurrence_embedding(graphs: List,
                                          num_dims: int = 4
                                          ) -> np.ndarray:
    """
    Build a small META-GRAPH over LABEL TYPES (not the original atoms):
    label A connects to label B with weight = how often an atom of
    type A is directly bonded to an atom of type B, counted across the
    ENTIRE dataset. Then spectrally embed THIS meta-graph (same
    Laplacian-eigenvector machinery as everywhere else in this
    codebase) to get a genuinely multi-dimensional, dataset-intrinsic
    representation per label.

    Why this fixes rarity's limitation: rarity collapses every atom to
    ONE number (its frequency rank), so two equally-rare-but-chemically-
    unrelated atoms (e.g. Sulfur and Bromine, if equally uncommon) get
    nearly identical weight. This co-occurrence embedding instead asks
    "which OTHER atom types does this one tend to bond with" — a
    genuinely relational, multi-dimensional signal, derived ENTIRELY
    from the dataset's own bonding structure. No external chemistry
    table, no risk of an unverified label->element mapping.

    Args:
        graphs:   list of NetworkX graphs (node features = one-hot or
                  scalar categorical labels, as set by datasets.py)
        num_dims: embedding dimensionality per label (small — this is
                  an eigendecomposition of a tiny n_labels x n_labels
                  matrix, not the original graphs, so this is fast
                  regardless of dataset size)

    Returns:
        embedding: np.ndarray [n_labels, num_dims]
                   embedding[label_id] -> that label's spectral vector
    """
    # Determine number of distinct labels present
    max_label = -1
    for G in graphs:
        for v in G.nodes():
            feat = G.nodes[v].get('feat')
            if feat is None:
                continue
            arr = np.asarray(feat).flatten()
            lbl = int(np.argmax(arr)) if arr.size > 1 \
                  else int(round(float(arr[0])))
            max_label = max(max_label, lbl)
    n_labels = max_label + 1
    if n_labels <= 0:
        return np.zeros((1, num_dims))

    # Build the label-level co-occurrence (bonding) matrix
    W = np.zeros((n_labels, n_labels), dtype=np.float64)
    for G in graphs:
        node_label = {}
        for v in G.nodes():
            feat = G.nodes[v].get('feat')
            if feat is None:
                continue
            arr = np.asarray(feat).flatten()
            node_label[v] = int(np.argmax(arr)) if arr.size > 1 \
                             else int(round(float(arr[0])))
        for u, v in G.edges():
            if u in node_label and v in node_label:
                lu, lv = node_label[u], node_label[v]
                W[lu, lv] += 1.0
                W[lv, lu] += 1.0

    # Normalised Laplacian of this small label-meta-graph
    deg = W.sum(axis=1)
    deg_safe = np.where(deg < 1e-10, 1.0, deg)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg_safe))
    L_label = np.eye(n_labels) - D_inv_sqrt @ W @ D_inv_sqrt

    eigvals, eigvecs = np.linalg.eigh(L_label)
    # Skip the trivial constant eigenvector (eigenvalue ~0, index 0);
    # take the next `num_dims` — exactly the same "Fiedler-and-beyond"
    # logic used for LapPE elsewhere in this project.
    k = min(num_dims, n_labels - 1)
    embedding = eigvecs[:, 1:1 + k]
    if k < num_dims:   # pad if there aren't enough distinct labels
        embedding = np.pad(embedding, ((0, 0), (0, num_dims - k)))

    return embedding


def compute_edge_weights_from_embedding(G: nx.Graph,
                                          embedding: np.ndarray
                                          ) -> Dict:
    """
    Compute a per-EDGE chemical-context weight using the label
    co-occurrence embedding: w(u,v) = ||embed(label(u)) - embed(label(v))||.

    This directly captures "how chemically/contextually DIFFERENT are
    the two endpoints of this specific bond" — e.g. a C-C edge (same
    label, distance 0) gets a small weight; a C-O edge (different,
    contextually distinct labels) gets a larger weight — WITHOUT ever
    reducing either atom to a single scalar in isolation first.

    Returns:
        edge_weights: dict {(u,v): float} for every edge in G
                      (both orderings stored, for lookup convenience)
    """
    edge_weights = {}
    node_label = {}
    for v in G.nodes():
        feat = G.nodes[v].get('feat')
        if feat is None:
            node_label[v] = 0
            continue
        arr = np.asarray(feat).flatten()
        node_label[v] = int(np.argmax(arr)) if arr.size > 1 \
                         else int(round(float(arr[0])))

    n_labels = embedding.shape[0]
    for u, v in G.edges():
        lu = min(node_label[u], n_labels - 1)
        lv = min(node_label[v], n_labels - 1)
        dist = float(np.linalg.norm(embedding[lu] - embedding[lv]))
        edge_weights[(u, v)] = dist + 1.0  # +1 keeps same-type edges
                                             # at weight 1, not 0
        edge_weights[(v, u)] = edge_weights[(u, v)]

    return edge_weights


def compute_node_weights(G: nx.Graph,
                          mode: str = 'rarity',
                          label_to_element: Optional[Dict[int, str]] = None,
                          label_counts: Optional[Dict] = None,
                          label_total: Optional[int] = None,
                          default_weight: float = 2.55  # Carbon's value
                          ) -> Optional[Dict]:
    """
    Compute a per-node scalar weight for the chemistry-weighted Hodge
    Laplacian.

    Args:
        G:    NetworkX graph. Each node may have a 'feat' attribute
              (as set by datasets.py / pyg_to_networkx).
        mode: 'none'              -> returns None (no weighting; caller
                                       should skip weighting entirely)
              'rarity'            -> RECOMMENDED DEFAULT. Weight =
                                       inverse-frequency (IDF-style) of
                                       the node's label, computed
                                       DIRECTLY from the dataset itself
                                       via `label_counts`/`label_total`
                                       (see compute_label_frequencies).
                                       No external chemistry table, no
                                       label->element mapping needed at
                                       all — common labels (e.g. Carbon
                                       in organic molecules) get weight
                                       near 1; rare labels (typically
                                       the chemically distinctive
                                       heteroatoms) get progressively
                                       larger weight. Fully dataset-
                                       intrinsic and leakage-free (uses
                                       only node features, never graph
                                       class labels).
              'feat_norm'         -> weight = L2 norm of node's raw
                                       feature vector.
                                       ⚠ ONLY meaningful for CONTINUOUS
                                       multi-dimensional features (e.g.
                                       DHFR's 56D chemistry attributes).
                                       USELESS for one-hot encoded
                                       categorical features.
              'label_index'       -> weight = raw integer label index.
                                       Arbitrary ordering, kept only
                                       for comparison/ablation purposes
                                       — 'rarity' is a strictly better
                                       dataset-intrinsic default.
              'electronegativity' -> weight = electronegativity of the
                                       atom, looked up via
                                       `label_to_element`. REQUIRES a
                                       *verified* mapping for the specific
                                       dataset — use only when you have
                                       confirmed this mapping externally
                                       (see hodge.py's caveat on
                                       MUTAG_LABEL_TO_ELEMENT).
        label_to_element: dict, required only when mode==
                          'electronegativity'.
        label_counts, label_total: outputs of compute_label_frequencies,
                          required only when mode=='rarity'.
        default_weight:   fallback weight for nodes with missing/unknown
                          features.

    Returns:
        node_weights: dict {node: float}, or None if mode=='none'.
    """
    if mode == 'none':
        return None

    node_weights = {}

    if mode == 'rarity':
        if label_counts is None or label_total is None:
            raise ValueError(
                "mode='rarity' requires label_counts and label_total "
                "(call compute_label_frequencies(all_graphs) once over "
                "your full dataset first, then pass the results in).")
        for n in G.nodes():
            feat = G.nodes[n].get('feat')
            if feat is None:
                node_weights[n] = default_weight
                continue
            arr = np.asarray(feat).flatten()
            label = int(np.argmax(arr)) if arr.size > 1 \
                    else int(round(float(arr[0])))
            count = label_counts.get(label, 1)
            node_weights[n] = float(np.log(label_total / count) + 1.0)

    elif mode == 'feat_norm':
        # Auto-detect one-hot encoding and warn — this mode is
        # mathematically guaranteed useless in that case.
        sample_feats = [G.nodes[n].get('feat') for n in G.nodes()
                        if G.nodes[n].get('feat') is not None]
        if sample_feats:
            arr0 = np.asarray(sample_feats[0]).flatten()
            looks_one_hot = (arr0.size > 1 and
                             np.isclose(np.sum(arr0), 1.0) and
                             np.all((arr0 == 0) | (arr0 == 1)))
            if looks_one_hot:
                print("  WARNING: feature_weighting='feat_norm' but "
                      "features look one-hot encoded — L2 norm will be "
                      "1.0 for EVERY node, giving NO discriminative "
                      "signal. Use mode='rarity' instead for "
                      "categorical/one-hot features.")
        for n in G.nodes():
            feat = G.nodes[n].get('feat')
            if feat is None:
                node_weights[n] = default_weight
            else:
                node_weights[n] = float(np.linalg.norm(feat)) + 1e-6

    elif mode == 'label_index':
        for n in G.nodes():
            feat = G.nodes[n].get('feat')
            if feat is None:
                node_weights[n] = default_weight
                continue
            arr = np.asarray(feat).flatten()
            if arr.size > 1:
                label = int(np.argmax(arr))   # one-hot -> class index
            else:
                label = float(arr[0])          # scalar categorical label
            node_weights[n] = float(label) + 1.0

    elif mode == 'electronegativity':
        if not label_to_element:
            raise ValueError(
                "mode='electronegativity' requires a non-empty "
                "label_to_element mapping. Use mode='rarity' "
                "instead if you don't have a verified mapping for "
                "this dataset.")
        for n in G.nodes():
            feat = G.nodes[n].get('feat')
            label = None
            if feat is not None:
                arr = np.asarray(feat).flatten()
                if arr.size > 1:
                    label = int(np.argmax(arr))
                else:
                    label = int(round(float(arr[0])))
            element = label_to_element.get(label, None)
            node_weights[n] = ELECTRONEGATIVITY.get(element, default_weight)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return node_weights


# ────────────────────────────────────────────────────────────────────
# Boundary operators
# ────────────────────────────────────────────────────────────────────

def build_B1(G: nx.Graph) -> Tuple[np.ndarray, list, list]:
    """
    Build oriented node-edge incidence matrix B_1.

    For each oriented edge e = (u, v) with u < v:
        B_1[u, e] = +1,  B_1[v, e] = -1

    Returns:
        B1:    np.ndarray [|V|, |E|]
        nodes: list of nodes (row ordering)
        edges: list of edges (col ordering), each edge (u,v) with u<v
    """
    nodes = sorted(G.nodes())
    edges = sorted([(min(u, v), max(u, v)) for u, v in G.edges()])
    edges = list(dict.fromkeys(edges))   # deduplicate

    n_nodes = len(nodes)
    n_edges = len(edges)

    node_idx = {node: i for i, node in enumerate(nodes)}
    B1 = np.zeros((n_nodes, n_edges), dtype=np.float64)

    for j, (u, v) in enumerate(edges):
        B1[node_idx[u], j] = +1.0
        B1[node_idx[v], j] = -1.0

    return B1, nodes, edges


def build_B2(G: nx.Graph,
             edges: list) -> Tuple[np.ndarray, list]:
    """
    Build oriented edge-triangle incidence matrix B_2.

    A triangle is a set of three nodes {a, b, c} that form a clique.
    Standard orientation: triangle (a, b, c) with a < b < c has
    oriented boundary = +(a,b) - (a,c) + (b,c).

    Returns:
        B2:        np.ndarray [|E|, |T|]
        triangles: list of (a,b,c) tuples, a<b<c — exposed so the
                   weighting step (Option B) can look up each
                   triangle's constituent nodes.
    """
    edge_idx = {e: i for i, e in enumerate(edges)}
    triangles = []
    nodes = sorted(G.nodes())

    # Find all triangles (cliques of size 3)
    for i, a in enumerate(nodes):
        for b in G.neighbors(a):
            if b > a:
                for c in G.neighbors(b):
                    if c > b and G.has_edge(a, c):
                        triangles.append((a, b, c))   # a < b < c

    n_edges    = len(edges)
    n_triangles = len(triangles)

    if n_triangles == 0:
        return np.zeros((n_edges, 0), dtype=np.float64), []

    B2 = np.zeros((n_edges, n_triangles), dtype=np.float64)

    for k, (a, b, c) in enumerate(triangles):
        # Boundary: +(a,b) - (a,c) + (b,c)
        if (a, b) in edge_idx: B2[edge_idx[(a, b)], k] = +1.0
        if (a, c) in edge_idx: B2[edge_idx[(a, c)], k] = -1.0
        if (b, c) in edge_idx: B2[edge_idx[(b, c)], k] = +1.0

    return B2, triangles


def _apply_node_weighting(B1: np.ndarray,
                            B2: np.ndarray,
                            nodes: list,
                            triangles: list,
                            node_weights: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale B1's rows by node weights, and B2's columns by each
    triangle's geometric-mean node weight (Option B, node-level).

    Geometric mean (rather than raw product) keeps the triangle weight
    on the same numerical scale as the individual node weights —
    interpretable as "average electronegativity of this ring fragment" —
    rather than compounding multiplicatively as more atoms are involved.
    """
    node_idx = {n: i for i, n in enumerate(nodes)}

    B1_w = B1.copy()
    for n in nodes:
        w = node_weights.get(n, 1.0)
        B1_w[node_idx[n], :] *= w

    B2_w = B2.copy()
    for k, (a, b, c) in enumerate(triangles):
        wa = node_weights.get(a, 1.0)
        wb = node_weights.get(b, 1.0)
        wc = node_weights.get(c, 1.0)
        nu = (wa * wb * wc) ** (1.0 / 3.0)   # geometric mean
        B2_w[:, k] *= nu

    return B1_w, B2_w


def _apply_edge_weighting(B1: np.ndarray,
                            B2: np.ndarray,
                            edges: list,
                            triangles: list,
                            edge_weights: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale B1's COLUMNS (one per edge) by that edge's own weight, and
    B2's columns (one per triangle) by the geometric mean of its three
    constituent edges' weights (Option B, EDGE-level — distinct from
    node-level weighting above).

    This is the more expressive option for co-occurrence-embedding-
    based weights: it directly encodes "how different are this
    SPECIFIC bond's two endpoints", rather than reducing each atom
    to an isolated scalar before any comparison happens.
    """
    edge_idx = {e: i for i, e in enumerate(edges)}

    B1_w = B1.copy()
    for e in edges:
        w = edge_weights.get(e, 1.0)
        B1_w[:, edge_idx[e]] *= w

    B2_w = B2.copy()
    for k, (a, b, c) in enumerate(triangles):
        e_ab = (min(a,b), max(a,b))
        e_ac = (min(a,c), max(a,c))
        e_bc = (min(b,c), max(b,c))
        w_ab = edge_weights.get(e_ab, 1.0)
        w_ac = edge_weights.get(e_ac, 1.0)
        w_bc = edge_weights.get(e_bc, 1.0)
        nu = (w_ab * w_ac * w_bc) ** (1.0 / 3.0)
        B2_w[:, k] *= nu

    return B1_w, B2_w


def build_hodge_laplacians(G: nx.Graph,
                            node_weights: Optional[Dict] = None,
                            edge_weights: Optional[Dict] = None
                            ) -> Tuple[np.ndarray, np.ndarray,
                                       np.ndarray, list, list]:
    """
    Build both Hodge Laplacians for graph G.

    L_0 = D - A          (standard graph/node Laplacian — ALWAYS
                           unweighted, regardless of weighting options)
    L_1 = B1^T B1 + B2 B2^T   (Hodge 1-Laplacian on edges — chemistry-
                                weighted via node_weights OR edge_weights
                                if provided. Provide at most ONE of the
                                two; edge_weights takes precedence if
                                both are given.)

    Args:
        G:            NetworkX graph
        node_weights: dict {node: float} — atom-level scalar weighting
                      (e.g. rarity, electronegativity). None for the
                      original unweighted L_1.
        edge_weights: dict {(u,v): float} — bond-level weighting (e.g.
                      label co-occurrence embedding distance). More
                      expressive than node_weights: directly compares
                      the two endpoints of each specific bond, rather
                      than reducing each atom to an isolated scalar
                      first.

    Returns:
        L0:    np.ndarray [|V|, |V|]   (always unweighted)
        L1:    np.ndarray [|E|, |E|]   (weighted iff node_weights or
                                        edge_weights given)
        B1:    np.ndarray [|V|, |E|]   (unweighted, raw incidence)
        nodes: list of nodes
        edges: list of edges
    """
    if G.number_of_nodes() == 0:
        return (np.zeros((0, 0)), np.zeros((0, 0)),
                np.zeros((0, 0)), [], [])

    # L_0: standard graph Laplacian — unaffected by any weighting option
    nodes_sorted = sorted(G.nodes())
    L0 = nx.laplacian_matrix(G, nodelist=nodes_sorted).toarray().astype(np.float64)

    if G.number_of_edges() == 0:
        return L0, np.zeros((0, 0)), np.zeros((0, 0)), nodes_sorted, []

    # B_1: node-edge incidence
    B1, nodes, edges = build_B1(G)

    # B_2: edge-triangle incidence (now also returns the triangle list)
    B2, triangles = build_B2(G, edges)

    if edge_weights is not None:
        B1_eff, B2_eff = _apply_edge_weighting(
            B1, B2, edges, triangles, edge_weights)
    elif node_weights is not None:
        B1_eff, B2_eff = _apply_node_weighting(
            B1, B2, nodes, triangles, node_weights)
    else:
        B1_eff, B2_eff = B1, B2   # original, unweighted behaviour

    # L_1 = B1^T B1 + B2 B2^T  (chemistry-weighted iff weighting given)
    L1_down = B1_eff.T @ B1_eff                        # [|E|, |E|]
    L1_up   = B2_eff @ B2_eff.T if B2_eff.shape[1] > 0 else \
              np.zeros((len(edges), len(edges)))       # [|E|, |E|]
    L1 = L1_down + L1_up

    return L0, L1, B1, nodes, edges


# ────────────────────────────────────────────────────────────────────
# Spectral computation
# ────────────────────────────────────────────────────────────────────

def compute_spectrum(M: np.ndarray,
                     K: int = 5,
                     tol: float = 1e-8) -> Tuple[np.ndarray, int]:
    """
    Compute the K smallest non-negative eigenvalues of symmetric
    matrix M (assumed PSD, i.e., a Laplacian).

    Returns:
        eigenvalues: np.ndarray [K], sorted ascending, padded with zeros
                     if M has fewer than K eigenvalues
        nullity:     int, number of eigenvalues < tol (= Betti number)

    Note: chemistry-weighted L_1 is still provably PSD — it is a sum
    of Gram matrices (X^T X form) just like the unweighted version,
    since weighting only rescales B1's rows / B2's columns before the
    same B^T B / BB^T construction. The nullity (Betti number count)
    is UNCHANGED by weighting — only the non-zero eigenvalues shift.
    This is mathematically guaranteed, not just empirically observed.
    """
    n = M.shape[0]
    if n == 0:
        return np.zeros(K), 0

    # Full eigendecomposition for small matrices
    # For large matrices (|E| > 500), use partial decomposition
    if n <= 500:
        eigs = linalg.eigvalsh(M)    # sorted ascending, full decomp
    else:
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix as sp_csr
        k_req = min(K + 2, n - 2)
        try:
            # Add small random perturbation to avoid zero starting vector
            v0 = np.random.RandomState(42).rand(n)
            eigs_p = eigsh(sp_csr(M), k=k_req, which='SM',
                           return_eigenvectors=False, tol=1e-5, v0=v0,
                           maxiter=n * 10)
            eigs = np.sort(eigs_p)
        except Exception:
            # Fallback: full eigendecomposition (slower but safe)
            eigs = linalg.eigvalsh(M)

    eigs = np.maximum(eigs, 0.0)    # numerical safety: clip negatives to 0
    nullity = int(np.sum(eigs < tol))

    # Return top-K eigenvalues (including zeros)
    if len(eigs) >= K:
        result = eigs[:K]
    else:
        result = np.pad(eigs, (0, K - len(eigs)), constant_values=0.0)

    return result, nullity


def hodge_spectra(G: nx.Graph,
                  K0: int = 5,
                  K1: int = 5,
                  node_weights: Optional[Dict] = None,
                  edge_weights: Optional[Dict] = None
                  ) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Compute the spectra of L_0 and L_1 for graph G.

    Args:
        G:            NetworkX graph
        K0, K1:       number of eigenvalues to return for L_0, L_1
        node_weights: dict {node: float} for atom-level weighted L_1.
        edge_weights: dict {(u,v): float} for bond-level weighted L_1
                      (more expressive — see build_hodge_laplacians).
                      Provide at most one of node_weights/edge_weights.
        Both None (default) reproduce the original unweighted behaviour.

    Returns:
        spec_L0:  np.ndarray [K0], smallest K0 eigenvalues of L_0
        spec_L1:  np.ndarray [K1], smallest K1 eigenvalues of L_1
                  (weighted iff node_weights or edge_weights given)
        beta_0:   int, 0th Betti number (connected components)
        beta_1:   int, 1st Betti number (independent cycles —
                  UNCHANGED by weighting, see compute_spectrum's note)
    """
    L0, L1, B1, nodes, edges = build_hodge_laplacians(
        G, node_weights=node_weights, edge_weights=edge_weights)

    spec_L0, beta_0 = compute_spectrum(L0, K=K0)
    if L1.shape[0] > 0:
        spec_L1, beta_1 = compute_spectrum(L1, K=K1)
    else:
        spec_L1, beta_1 = np.zeros(K1), 0

    return spec_L0, spec_L1, beta_0, beta_1
