import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
os.environ['TORCH_GEOMETRIC_NO_DOWNLOAD'] = '1'

from torch_geometric.datasets import Planetoid, Amazon, Actor
from torch_geometric.transforms import NormalizeFeatures

root = './data'

# These will now find the raw files locally and just process them
# without downloading
print("Processing Cora...")
try:
    d = Planetoid(root=root, name='Cora', transform=NormalizeFeatures())
    print(f"  Cora OK — {d[0].num_nodes} nodes")
except Exception as e:
    print(f"  Cora error: {e}")

print("Processing CiteSeer...")
try:
    d = Planetoid(root=root, name='CiteSeer', transform=NormalizeFeatures())
    print(f"  CiteSeer OK — {d[0].num_nodes} nodes")
except Exception as e:
    print(f"  CiteSeer error: {e}")

print("Processing Amazon Photo...")
try:
    d = Amazon(root=root, name='Photo', transform=NormalizeFeatures())
    print(f"  Amazon Photo OK — {d[0].num_nodes} nodes")
except Exception as e:
    print(f"  Amazon Photo error: {e}")

print("Actor already downloaded — skipping.")
print("\nDone.")