"""
Baseline models for comparison.
    - GCN  (Kipf & Welling 2017)
    - SGC  (Wu et al. 2019) — closest prior work to Graph-SPR preprocessing
    - MLP  (no graph structure, ablation)

All baselines use PyTorch Geometric and are trained with Adam.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv
import torch.nn as nn


# ---------------------------------------------------------------------------
# GCN
# ---------------------------------------------------------------------------

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def run_gcn(data, num_classes, hidden=64, epochs=200, lr=0.01,
            weight_decay=5e-4, dropout=0.5, seed=42):
    torch.manual_seed(seed)
    device = torch.device('cpu')

    model = GCN(data.num_features, hidden, num_classes, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y          = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask   = data.val_mask.to(device)
    test_mask  = data.test_mask.to(device)

    best_val_acc  = 0
    best_test_acc = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out  = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(x, edge_index)
            pred     = out_eval.argmax(dim=1)
            val_acc  = (pred[val_mask] == y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = test_acc

    print(f"  GCN  test acc: {best_test_acc:.4f}")
    return best_test_acc


# ---------------------------------------------------------------------------
# SGC
# ---------------------------------------------------------------------------

class SGC(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K=2):
        super().__init__()
        self.conv = SGConv(in_channels, out_channels, K=K, cached=True)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


def run_sgc(data, num_classes, K=2, epochs=200, lr=0.2,
            weight_decay=5e-5, seed=42):
    torch.manual_seed(seed)
    device = torch.device('cpu')

    model = SGC(data.num_features, num_classes, K=K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y          = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask   = data.val_mask.to(device)
    test_mask  = data.test_mask.to(device)

    best_val_acc  = 0
    best_test_acc = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out  = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred     = out.argmax(dim=1)
            val_acc  = (pred[val_mask] == y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = test_acc

    print(f"  SGC  test acc: {best_test_acc:.4f}")
    return best_test_acc


# ---------------------------------------------------------------------------
# MLP (no graph — ablation)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


def run_mlp(X, y, train_mask, val_mask, test_mask, num_classes,
            hidden=64, epochs=200, lr=0.01, weight_decay=5e-4,
            dropout=0.5, seed=42):
    torch.manual_seed(seed)

    x_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    model     = MLP(X.shape[1], hidden, num_classes, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    best_val_acc  = 0
    best_test_acc = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out  = model(x_tensor)
        loss = F.cross_entropy(out[train_mask], y_tensor[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred     = out.argmax(dim=1)
            val_acc  = (pred[val_mask] == y_tensor[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == y_tensor[test_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = test_acc

    print(f"  MLP  test acc: {best_test_acc:.4f}")
    return best_test_acc


# ---------------------------------------------------------------------------
# GAT  (Veličković et al. 2018)
# ---------------------------------------------------------------------------

from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=8, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels,
                             heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels,
                             heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def run_gat(data, num_classes, hidden=8, heads=8, epochs=1000,
            lr=0.005, weight_decay=5e-4, dropout=0.6, patience=100, seed=42):
    """
    GAT with early stopping (patience=100) and correct eval-mode inference.
    Matches original Velickovic et al. 2018 setup:
      - hidden=8, heads=8, dropout=0.6, lr=0.005, weight_decay=5e-4
      - Early stopping on validation loss (not accuracy)
    """
    torch.manual_seed(seed)
    device = torch.device('cpu')

    model = GAT(data.num_features, hidden, num_classes,
                heads=heads, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y          = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask   = data.val_mask.to(device)
    test_mask  = data.test_mask.to(device)

    best_val_loss = float('inf')
    best_test_acc = 0
    patience_ctr  = 0

    for epoch in range(epochs):
        # --- Train step ---
        model.train()
        optimizer.zero_grad()
        out  = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # --- Eval step (fresh forward pass, no dropout) ---
        model.eval()
        with torch.no_grad():
            out_eval = model(x, edge_index)
            val_loss = F.cross_entropy(out_eval[val_mask], y[val_mask]).item()
            pred     = out_eval.argmax(dim=1)
            test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

        # Early stopping on val loss (original GAT paper criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_acc = test_acc
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    print(f"  GAT  test acc: {best_test_acc:.4f}  (stopped at epoch {epoch+1})")
    return best_test_acc


# ---------------------------------------------------------------------------
# GIN  (Xu et al. 2019)
# ---------------------------------------------------------------------------

from torch_geometric.nn import GINConv

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs   = torch.nn.ModuleList()

        # Input layer
        mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.convs.append(GINConv(mlp, train_eps=True))

        # Hidden layers
        for _ in range(num_layers - 2):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=True))

        # Output projection
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


def run_gin(data, num_classes, hidden=64, num_layers=2, epochs=200,
            lr=0.01, weight_decay=5e-4, dropout=0.5, seed=42):
    torch.manual_seed(seed)
    device = torch.device('cpu')

    model = GIN(data.num_features, hidden, num_classes,
                num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y          = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask   = data.val_mask.to(device)
    test_mask  = data.test_mask.to(device)

    best_val_acc  = 0
    best_test_acc = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out  = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(x, edge_index)
            pred     = out_eval.argmax(dim=1)
            val_acc  = (pred[val_mask] == y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = test_acc

    print(f"  GIN  test acc: {best_test_acc:.4f}")
    return best_test_acc


# ---------------------------------------------------------------------------
# GraphSAGE  (Hamilton et al. 2017)
# ---------------------------------------------------------------------------

from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs   = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def run_graphsage(data, num_classes, hidden=256, num_layers=2, epochs=500,
                  lr=0.01, weight_decay=5e-4, dropout=0.5, patience=50, seed=42):
    """
    GraphSAGE with hidden=256 (standard for citation benchmarks),
    correct eval-mode inference, and early stopping.
    """
    torch.manual_seed(seed)
    device = torch.device('cpu')

    model = GraphSAGE(data.num_features, hidden, num_classes,
                      num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y          = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask   = data.val_mask.to(device)
    test_mask  = data.test_mask.to(device)

    best_val_acc  = 0
    best_test_acc = 0
    patience_ctr  = 0

    for epoch in range(epochs):
        # --- Train step ---
        model.train()
        optimizer.zero_grad()
        out  = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # --- Eval step (fresh forward pass, no dropout) ---
        model.eval()
        with torch.no_grad():
            out_eval = model(x, edge_index)
            pred     = out_eval.argmax(dim=1)
            val_acc  = (pred[val_mask] == y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = test_acc
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    print(f"  SAGE test acc: {best_test_acc:.4f}  (stopped at epoch {epoch+1})")
    return best_test_acc
