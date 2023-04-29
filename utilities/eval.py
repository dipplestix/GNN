def count_collisions(coloring, edge_index):
    col = 0
    for i in range(edge_index.shape[1]):
        n1, n2 = edge_index[0][i], edge_index[1][i]
        if coloring[n1] == coloring[n2]:
            col += 1
    return col/2
