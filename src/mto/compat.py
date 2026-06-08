"""Compatibility shims for DetaNet on PyTorch 2.7 + torch_geometric without pyg-lib>=0.6.

Must be imported BEFORE any DetaNet import.
"""

import torch
import torch_geometric.nn as tgnn
import torch_cluster

# e3nn 0.4.4 compatibility with PyTorch 2.7 (torch.load weights_only=True default)
torch.serialization.add_safe_globals([slice])

# torch_geometric 2.8 radius_graph requires pyg-lib>=0.6, but torch_cluster has it
tgnn.radius_graph = torch_cluster.radius_graph
