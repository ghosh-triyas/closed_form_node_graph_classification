"""
Dataset loading for graph classification benchmarks.

Supports TU datasets: MUTAG, PTC_MR, PROTEINS, DHFR, REDDIT-B, IMDB-B

Converts torch_geometric Data objects to NetworkX graphs for PHS computation.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Optional
import os


def pyg_to_networkx(data) -> nx.Graph:
    """
    Convert a torch_geometric Data object to a NetworkX undirected graph.

    Node features (if present) are stored as node attribute 'feat'.
    Edge index is used to build the graph structure.
    """
    G = nx.Graph()

    n = data.num_nodes
    G.add_nodes_from(range(n))

    # Add node features if available
    if data.x is not None:
        x_np = data.x.numpy()
        for i in range(n):
            G.nodes[i]['feat'] = x_np[i]

    # Add edges
    if data.edge_index is not None:
        edge_index = data.edge_index.numpy()
        for j in range(edge_index.shape[1]):
            u = int(edge_index[0, j])
            v = int(edge_index[1, j])
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)

    return G


def load_tu_dataset(name: str,
                     root: str = '/tmp/tu_datasets'
                     ) -> Tuple[List[nx.Graph], np.ndarray]:
    """
    Load a TU benchmark dataset.

    Args:
        name: dataset name ('MUTAG', 'PTC_MR', 'PROTEINS',
                            'DHFR', 'REDDIT-BINARY', 'IMDB-BINARY')
        root: directory to cache downloaded datasets

    Returns:
        graphs: list of NetworkX graphs
        labels: np.ndarray [N], integer class labels
    """
    try:
        from torch_geometric.datasets import TUDataset
    except ImportError:
        raise ImportError("torch_geometric required. "
                          "pip install torch_geometric")

    print(f"Loading {name}...")
    dataset = TUDataset(root=root, name=name, use_node_attr=True)

    graphs = []
    labels = []

    for data in dataset:
        G = pyg_to_networkx(data)
        graphs.append(G)
        labels.append(int(data.y.item()))

    labels = np.array(labels, dtype=np.int64)

    # Map labels to 0-indexed integers
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])

    print(f"  {name}: {len(graphs)} graphs, "
          f"{len(unique_labels)} classes, "
          f"avg nodes={np.mean([G.number_of_nodes() for G in graphs]):.1f}, "
          f"avg edges={np.mean([G.number_of_edges() for G in graphs]):.1f}")

    return graphs, labels


def train_test_split(graphs: List[nx.Graph],
                      labels: np.ndarray,
                      fold: int,
                      n_folds: int = 10,
                      seed: int = 42
                      ) -> Tuple[List, np.ndarray, List, np.ndarray]:
    """
    Stratified k-fold split for graph classification.

    Returns train and test graphs+labels for a given fold.
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                           random_state=seed)
    splits = list(skf.split(graphs, labels))
    train_idx, test_idx = splits[fold]

    G_train = [graphs[i] for i in train_idx]
    G_test  = [graphs[i] for i in test_idx]
    y_train = labels[train_idx]
    y_test  = labels[test_idx]

    return G_train, y_train, G_test, y_test


def dataset_statistics(graphs: List[nx.Graph],
                         labels: np.ndarray,
                         name: str) -> None:
    """Print dataset statistics."""
    n_graphs    = len(graphs)
    n_classes   = len(np.unique(labels))
    avg_nodes   = np.mean([G.number_of_nodes() for G in graphs])
    avg_edges   = np.mean([G.number_of_edges() for G in graphs])
    has_feats   = any(G.nodes[list(G.nodes())[0]].get('feat') is not None
                      for G in graphs if G.number_of_nodes() > 0)

    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"  Graphs:       {n_graphs}")
    print(f"  Classes:      {n_classes}  {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"  Avg nodes:    {avg_nodes:.1f}")
    print(f"  Avg edges:    {avg_edges:.1f}")
    print(f"  Node features: {'yes' if has_feats else 'no (structure only)'}")
    print(f"{'='*50}")
