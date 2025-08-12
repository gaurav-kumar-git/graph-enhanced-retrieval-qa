# In src/models/gnn_experimental.py
# This is a NEW file for our Phase 4 experiments.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GNNExperimental(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, aggr='mean'):
        """
        A flexible GNN model specifically for running experiments in Phase 4.
        This class is separate from the original GNNRanker to keep experiments isolated.
        """
        super().__init__()
        
        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")

        self.convs = nn.ModuleList()
        
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels, aggr=aggr))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x