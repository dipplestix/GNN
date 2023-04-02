import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import threading


class GraphSAGEConvolution(nn.Module):
    def __init__(self, in_features, out_features, aggregator='mean'):
        super(GraphSAGEConvolution, self).__init__()
        assert aggregator in ['mean', 'pool'], "Aggregator must be one of ['mean', 'pool']"

        self.aggregator = aggregator
        if aggregator == 'mean':
            self.linear = nn.Linear(in_features, out_features)

        if self.aggregator == 'pool':
            self.linear = nn.Linear(in_features * 2, out_features)
            self.pool_linear = nn.Linear(in_features, in_features)

        self.adj = None
        self.neighbors = None

    def forward(self, x, edge_index):
        if self.adj is None and self.neighbors is None:
            if self.aggregator == 'mean':
                self._make_adj(edge_index)
            if self.aggregator == 'pool':
                self._make_neighbors(edge_index)
        # Aggregate neighbors
        if self.aggregator == 'mean':
            x_neighbors = torch.matmul(self.adj_normalized, x)
            x_concat = x_neighbors

        elif self.aggregator == 'pool':
            pool_x = F.relu(self.pool_linear(x))

            # # Define a worker function for computing the maximum values for each node
            # def worker(node_idx):
            #     rows = self.neighbors[node_idx]
            #     max_vals, _ = torch.max(pool_x[rows], dim=0)
            #     x_neighbors[node_idx] = max_vals
            #
            # # Initialize a tensor to store the maximum values
            # x_neighbors = torch.zeros(x.shape).to(edge_index.device)
            #
            # # Create a list of threads
            # threads = [threading.Thread(target=worker, args=(i,)) for i in range(self.adj.shape[0])]
            #
            # # Start the threads
            # for thread in threads:
            #     thread.start()
            #
            # # Wait for the threads to finish
            # for thread in threads:
            #     thread.join()

            x_neighbors = torch.zeros(x.shape).to(edge_index.device)
            for i in range(self.adj.shape[0]):
                rows = self.neighbors[i]
                max_vals, _ = torch.max(pool_x[rows], dim=0)
                x_neighbors[i] = max_vals

            # Concatenate the node features and the aggregated neighbor features
            x_concat = torch.cat([x, x_neighbors], dim=1)

        # Apply the linear transformation
        out = self.linear(x_concat)

        return out

    def _make_adj(self, edge_index):
        if self.aggregator == 'mean':
            adj = to_dense_adj(edge_index).squeeze(0)
            D = torch.diag(torch.sum(adj, dim=0) ** (-0.5))
            self.adj_normalized = torch.matmul(torch.matmul(D, adj), D)

    def _make_neighbors(self, edge_index):
        self.neighbors = {}
        for i in range(self.adj.shape[0]):
            self.neighbors[i] = edge_index[1][edge_index[0] == i]


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, aggregator='mean'):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphSAGEConvolution(input_dim, hidden_dim, aggregator))

        for _ in range(num_layers - 2):
            self.layers.append(GraphSAGEConvolution(hidden_dim, hidden_dim, aggregator))

        self.layers.append(GraphSAGEConvolution(hidden_dim, output_dim, aggregator))

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            # Apply activation and dropout for all layers except the last one
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x
