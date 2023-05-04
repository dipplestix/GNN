import torch
from collections import OrderedDict
import random


def dsatur(edge_list):
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

    # Find the degrees and saturation of each node
    degrees = {node: len(neighbors[node]) for node in nodes}
    saturation = {node: 0 for node in nodes}

    while nodes:
        # Find the node with the highest saturation
        max_saturation = -1
        max_degrees = -1
        node_choices = []
        for n in nodes:
            if saturation[n] > max_saturation:
                node_choices = [n]
                max_saturation = saturation[n]
                max_degrees = degrees[n]
            # Tiebreak by the degree of the node in the uncolored subgraph
            if saturation[n] == max_saturation:
                if degrees[n] > max_degrees:
                    node_choices = [n]
                    max_degrees = degrees[n]
                if degrees[n] == max_degrees:
                    node_choices.append(n)
        node = random.choice(node_choices)

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

        # Update saturation and degree in the uncolored subgraph
        for neighbor in neighbors[node]:
            degrees[neighbor] -= 1
            if colors[neighbor] is None:
                saturation[neighbor] += 1

        nodes.remove(node)
    return colors
