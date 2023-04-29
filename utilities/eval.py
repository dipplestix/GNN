from torch_geometric.utils import to_dense_adj
import torch


def count_collisions(coloring, edge_index):
    col = 0
    for i in range(edge_index.shape[1]):
        n1, n2 = edge_index[0][i], edge_index[1][i]
        if coloring[n1] == coloring[n2]:
            col += 1
    return col/2


def dot_product_loss(col_probs, edge_list):
    adj = to_dense_adj(edge_list)
    loss = torch.sum(torch.mm(col_probs, col_probs.T) * adj)/2
    return loss
