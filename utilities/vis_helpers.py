import networkx as nx
from torch_geometric.utils import to_dense_adj


def plot_coloring(edge_list, coloring=None, with_labels=False):
    graph = nx.from_numpy_array(to_dense_adj(edge_list).squeeze(dim=0).numpy())
    pos = nx.kamada_kawai_layout(graph)
    nx.draw(graph, pos, with_labels=with_labels, node_color=coloring)
