import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # Define the linear transformation layer
        self.linear = nn.Linear(in_features, out_features)
        # Define the attention coefficients layer
        self.attention = nn.Linear(2 * out_features, 1)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.adj = None

    def forward(self, h, edge_index):
        if self.adj is None:
            self._make_adj(edge_index)
        # Apply the linear transformation
        Wh = self.linear(h)
        a_input = self._prepare_attentional_mechanism_input(Wh)

        # Compute the attention coefficients
        e = self.leakyrelu(self.attention(a_input).squeeze(2))

        # Create the attention mask using adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(self.adj > 0, e, zero_vec)

        # Apply softmax to the attention coefficients
        attention = F.softmax(attention, dim=1)
        # Apply dropout to the attention coefficients
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Compute the aggregated node features using attention coefficients
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def _make_adj(self, edge_index):
        self.adj = to_dense_adj(edge_index).squeeze(0)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nheads, dropout=0.5, alpha=0.2):
        super(GAT, self).__init__()
        self.dropout = dropout

        # Create multiple attention layers
        self.attentions = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
        # Create the output attention layer
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # Apply dropout to input features
        x = F.dropout(x, self.dropout, training=self.training)
        # Apply multiple attention layers and concatenate their outputs
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # Apply dropout to the concatenated features
        x = F.dropout(x, self.dropout, training=self.training)
        # Apply the output attention layer
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class MultiLayerGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, n_layers):
        super(MultiLayerGAT, self).__init__()
        self.dropout = dropout

        self.attentions_list = nn.ModuleList()
        # Create attention layers for the first layer
        self.attentions_list.append(nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]))

        # Create attention layers for the intermediate layers
        for _ in range(n_layers - 2):
            self.attentions_list.append(nn.ModuleList([GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]))

        # Create the output attention layer
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # Apply dropout to input features
        x = F.dropout(x, self.dropout, training=self.training)
        for attentions in self.attentions_list:
            # Apply multiple attention layers and concatenate their outputs
            x = torch.cat([att(x, adj) for att in attentions], dim=1)
            # Apply dropout to the concatenated features
            x = F.dropout(x, self.dropout, training=self.training)

        # Apply the output attention layer
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
