"""
Dataset loading using PyTorch Geometric.
Supports: Cora, CiteSeer, AmazonPhoto, Actor
"""

import numpy as np
import torch
from torch_geometric.datasets import Planetoid, Amazon, Actor
from torch_geometric.transforms import NormalizeFeatures
import os


DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def load_dataset(name, split_seed=42):
    """
    Load a node classification dataset.

    Args:
        name: str, one of ['cora', 'citeseer', 'amazon_photo', 'actor']
        split_seed: int

    Returns:
        data:       PyG Data object
        X:          np.ndarray [N, D], raw features
        y:          np.ndarray [N], integer labels
        train_mask: np.ndarray bool [N]
        val_mask:   np.ndarray bool [N]
        test_mask:  np.ndarray bool [N]
        num_classes: int
    """
    name_lower = name.lower()

    if name_lower == 'cora':
        dataset = Planetoid(root=DATA_ROOT, name='Cora',
                            transform=NormalizeFeatures())
        data = dataset[0]

    elif name_lower == 'citeseer':
        dataset = Planetoid(root=DATA_ROOT, name='CiteSeer',
                            transform=NormalizeFeatures())
        data = dataset[0]

    elif name_lower == 'amazon_photo':
        dataset = Amazon(root=DATA_ROOT, name='Photo',
                         transform=NormalizeFeatures())
        data = dataset[0]
        # Amazon does not have default splits — create them
        data = _random_split(data, num_train_per_class=20,
                             num_val=500, seed=split_seed)

    elif name_lower == 'actor':
        dataset = Actor(root=os.path.join(DATA_ROOT, 'Actor'),
                        transform=NormalizeFeatures())
        data = dataset[0]
        # Actor has 10 splits — use split 0
        data.train_mask = data.train_mask[:, 0]
        data.val_mask   = data.val_mask[:, 0]
        data.test_mask  = data.test_mask[:, 0]

    else:
        raise ValueError(f"Unknown dataset: {name}. "
                         f"Choose from: cora, citeseer, amazon_photo, actor")

    X          = data.x.numpy().astype(np.float64)
    y          = data.y.numpy().astype(np.int64)
    train_mask = data.train_mask.numpy().astype(bool)
    val_mask   = data.val_mask.numpy().astype(bool)
    test_mask  = data.test_mask.numpy().astype(bool)
    num_classes = int(y.max()) + 1

    print(f"\nDataset: {name}")
    print(f"  Nodes: {X.shape[0]}, Features: {X.shape[1]}, Classes: {num_classes}")
    print(f"  Train: {train_mask.sum()}, Val: {val_mask.sum()}, "
          f"Test: {test_mask.sum()}")

    return data, X, y, train_mask, val_mask, test_mask, num_classes


def _random_split(data, num_train_per_class=20, num_val=500, seed=42):
    """Create random train/val/test split for datasets without default splits."""
    rng = np.random.RandomState(seed)
    y   = data.y.numpy()
    N   = len(y)

    train_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        chosen = rng.choice(idx, size=min(num_train_per_class, len(idx)),
                            replace=False)
        train_idx.extend(chosen.tolist())

    remaining = list(set(range(N)) - set(train_idx))
    rng.shuffle(remaining)
    val_idx  = remaining[:num_val]
    test_idx = remaining[num_val:]

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    return data
