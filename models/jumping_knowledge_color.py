import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utilities.transformer_helpers import PositionalEncoding


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        deg = torch.sum(adj, dim=1)
        out /= deg.view(-1, 1)
        out = self.W(out)
        return out


class JKColor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.layers.append(GraphConvolution(hidden_dim, hidden_dim))

        self.adj_normalized = None

        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=1)
        self.positional_encoding = PositionalEncoding(d_model=hidden_dim)

    def forward(self, x, edge_index):
        if self.adj_normalized is None:
            self._make_adj(edge_index)

        layer_ouputs = [x]
        for i, layer in enumerate(self.layers):
            x = layer(x, self.adj_normalized)
            # Apply activation and dropout for all layers except the last one
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            layer_ouputs.append(x)

        x = torch.cat(layer_ouputs, dim=1)
        x = x.transpose(0, 1).unsqueeze(0)
        x = self.transformer_encoder(x)
        x = x.squeeze(0).transpose(0, 1)

        return x

    def _make_adj(self, edge_index):
        adj = to_dense_adj(edge_index).squeeze(0).to(edge_index.device)
        self.adj_normalized = adj
