import networkx as nx
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt


def plot_coloring(edge_list, coloring=None, with_labels=False, title=None, bad_edges=None):
    graph = nx.from_numpy_array(to_dense_adj(edge_list).squeeze(dim=0).numpy())
    pos = nx.kamada_kawai_layout(graph)
    # nx.draw(graph, pos, with_labels=with_labels, node_color=coloring)
    nx.draw_networkx_nodes(graph, pos, node_color=coloring)
    nx.draw_networkx_edges(graph, pos, edge_color='grey')
    if bad_edges is not None:
        for edge, color in bad_edges.items():
            nx.draw_networkx_edges(graph, pos, edgelist=[edge], edge_color=color)
    if title:
        plt.title(title)

    plt.show()
