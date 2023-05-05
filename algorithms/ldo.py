import torch
from collections import OrderedDict
import random


def ldo(edge_list):
    # Implemented based on section 3.3 of "Guide to Graph Colouring" - R.M.R. Lewis

    # Set up things
    n = torch.max(edge_list) + 1
    colors = [None]*n
    color_map = OrderedDict()
    color_map[0] = set()
    nodes = list(range(n))
    latest_color = 0

    # Create an adjacency list
    neighbors = [set() for _ in range(n)]
    for u, v in edge_list.t():
        neighbors[u.item()].add(v.item())
        neighbors[v.item()].add(u.item())

    # Find the degrees of each node
    degrees = {node: len(neighbors[node]) for node in nodes}

    while nodes:
        # Find the set of nodes with the highest degree and pick one
        max_degree = max(degrees.values())
        max_nodes = {node for node in nodes if degrees[node] == max_degree}
        node = random.choice(list(max_nodes))

        # Assign a color just like in greedy
        for color in color_map.keys():
            if not (neighbors[node] & color_map[color]):
                color_map[color].add(node)
                colors[node] = color
                break
        else:
            latest_color += 1
            color_map[latest_color] = {node}
            colors[node] = latest_color

        nodes.remove(node)
        degrees.pop(node)
    return colors
