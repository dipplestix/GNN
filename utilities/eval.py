from torch_geometric.utils import to_dense_adj
import torch


def count_collisions(coloring, edge_index):
    col = 0
    for i in range(edge_index.shape[1]):
        n1, n2 = edge_index[0][i], edge_index[1][i]
        if coloring[n1] == coloring[n2]:
            col += 1
    return col/2


def get_bad_edges(coloring, edge_index):
    bad_edges = {}
    for i in range(edge_index.shape[1]):
        n1, n2 = edge_index[0][i], edge_index[1][i]
        if coloring[n1] == coloring[n2]:
            bad_edges[(n1.item(), n2.item())] = 'red'
    return bad_edges


class PottsLoss:
    def __init__(self, edge_list):
        self.adj = to_dense_adj(edge_list)

    def __call__(self, col_probs):
        loss = torch.sum(torch.mm(col_probs, col_probs.T) * self.adj) / 2
        return loss

# def dot_product_loss(col_probs, edge_list):
#     adj = to_dense_adj(edge_list).squeeze(0)
#     loss = torch.sum(torch.mm(col_probs, col_probs.T) * adj)/2
#     return loss/2


class WangLoss:
    def __init__(self, edge_list, lam=0.2):
        self.adj = to_dense_adj(edge_list)
        self.lam = lam

    def __call__(self, col_probs):
        log_probs = torch.log(col_probs)
        loss = torch.sum(torch.mm(col_probs, col_probs.T) * self.adj) / 2
        loss += -self.lam*torch.trace(torch.mm(col_probs, log_probs.T))
        return loss

# def self_info_loss(col_probs):
#     log_probs = torch.log(col_probs)
#     loss = torch.trace(torch.mm(col_probs, log_probs.T))
#     return loss
#
#
# def wang_loss_function(col_probs, edge_list, lam):
#     # Loss function from Wang's 2023 paper: A Graph Neural Network with Negative Message Passing for Graph Coloring
#     loss = dot_product_loss(col_probs, edge_list) + lam*self_info_loss(col_probs)
#     return loss
