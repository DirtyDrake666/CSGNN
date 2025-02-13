import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import Sequential as GeometricSequential
from torch_geometric.nn import Linear as GeometricLinear
from torch_geometric.typing import OptPairTensor
from typing import Callable


class GINLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_norm: Callable):
        super(GINLayer, self).__init__()
        # Define the MLP for the GINConv, which consists of a Linear layer followed by ReLU
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.batch_norm = batch_norm

    def forward(self, x: OptPairTensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Apply GINConv with the MLP
        conv = GINConv(self.mlp)
        return conv(x, edge_index)


class CSGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, cutoff, n_classes, dropout, device, residuals=True,
                 l2_norm=False, predict=True):
        super(CSGNN, self).__init__()
        self.cutoff = cutoff
        self.device = device
        self.residuals = residuals
        self.dropout = dropout
        self.l2_norm = l2_norm
        self.predict = predict

        # Feature Encoder that projects initial node representation to d-dim space
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # GIN layers: Use GINLayer with the specified number of layers
        self.convs = nn.ModuleList(
            [GINLayer(hidden_dim, hidden_dim, batch_norm=nn.Identity()) for _ in range(cutoff - 1)]
        )

        self.hidden_dim = hidden_dim
        
        # If predicting, define the final MLP layers
        if self.predict:
            self.linear1 = nn.Linear(hidden_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, n_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for c in self.feature_encoder.children():
            if hasattr(c, "reset_parameters"):
                c.reset_parameters()
        for conv in self.convs:
            for layer in conv.mlp.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        if self.predict:
            self.linear1.reset_parameters()
            self.linear2.reset_parameters()

    def forward(self, data):
        x, batch = data.x, data.batch
        edge_index = getattr(data, 'nb_list')

        # Project initial node features into hidden space
        h = self.feature_encoder(x)

        # Loop through the layers and apply GINConv
        for i in range(self.cutoff - 1):
            h = self.convs[i](h, edge_index)

            # Optionally use position embeddings if provided
            if hasattr(data, f'pos_{i}'):
                pos = getattr(data, f'pos_{i}')
                h = self.convs[i](h, pos)
            if hasattr(data, f'pos2_{i}'):
                pos2 = getattr(data, f'pos2_{i}')
                h = self.convs[i](h, pos2)

        # Pooling operation to aggregate node features
        h = global_add_pool(h, batch)

        # If predicting, apply final MLP layers
        if self.predict:
            h = F.relu(self.linear1(h))
            h = F.dropout(h, training=self.training, p=self.dropout)
            h = self.linear2(h)

        return h

