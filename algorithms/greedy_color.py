import random
import torch


def greedy_color(edge_list, shuffle=True):
    # Implemented based on section 3.1 of "Guide to Graph Colouring" - R.M.R. Lewis
    n = torch.max(edge_list) + 1
    colors = [None]*n
    color_map = {0: set()}
    nodes = list(range(n))
    if shuffle:
        random.shuffle(nodes)

    neighbors = [set() for _ in range(n)]
    for u, v in edge_list.t():
        neighbors[u.item()].add(v.item())

    latest_color = 0
    for node in nodes:
        for color in color_map:
            if not (neighbors[node] & color_map[color]):
                color_map[color].add(node)
                colors[node] = color
                break
        else:
            latest_color += 1
            color_map[latest_color] = {node}
            colors[node] = latest_color
    return colors



