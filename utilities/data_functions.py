import torch


# We used chatGPT to generate this by giving it an example .col file and the desired output
def read_col_file(filename):
    edges = []
    node_colors = {}
    with open(filename) as f:
        for line in f:
            parts = line.split()
            if parts[0] == 'e':
                n1, n2 = int(parts[1]), int(parts[2])
                edges.append((n1, n2))
                edges.append((n2, n1))
            elif parts[0] == 'f':
                n = int(parts[1])
                colors = [int(c) for c in parts[2:]]
                node_colors[n] = colors
    return edges, node_colors


def build_edge_list(edges):
    edge_index = torch.zeros((2, len(edges)), dtype=torch.long)
    for i, (n1, n2) in enumerate(edges):
        edge_index[0][i] = n1-1
        edge_index[1][i] = n2-1
    return edge_index


def build_color_list(n, node_colors):
    color_list = []
    for i in range(n):
        if i+1 in node_colors:
            color_list.append(torch.tensor(node_colors[i+1]))
        else:
            color_list.append(None)
    return color_list


def col_to_edge_list(filename):
    edges, node_colors = read_col_file(filename)
    n = max(max(e) for e in edges)
    edge_index = build_edge_list(edges)
    color_list = build_color_list(n, node_colors)
    return edge_index, color_list
