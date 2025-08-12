import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv 

class GNNRanker(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        GNN model for re-ranking passages.

        Args:
            in_channels: Dimensionality of input node features (e.g., 1024 for bge-m3).
            hidden_channels: Dimensionality of the hidden GNN layer.
            out_channels: Dimensionality of the output node features. Should be same as in_channels.
        """
        super().__init__()
        # We'll use two GraphSAGE layers. It's a robust choice for aggregating neighbor info.
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Defines the forward pass of the GNN.

        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, in_channels].
            edge_index (Tensor): Graph connectivity in COO format with shape [2, num_edges].

        Returns:
            Tensor: The updated node feature matrix of shape [num_nodes, out_channels].
        """
        # First GNN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x) # Apply a non-linear activation function
        x = F.dropout(x, p=0.5, training=self.training) # Dropout for regularization

        # Second GNN layer
        x = self.conv2(x, edge_index)

        return x