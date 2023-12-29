import functools

import numpy as np


def cell_diam(mesh_cell, mesh):
    cell_diameter = 0.0
    points = mesh.points[mesh_cell.node_tags]
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            pi, pj = points[i], points[j]
            distance = np.linalg.norm(pi - pj)
            cell_diameter = np.max([distance, cell_diameter])
    return cell_diameter


def min_mesh_size(mesh, dim=None):
    if dim is None:
        dim = mesh.dimension
    cells_with_dim = [cell for cell in mesh.cells if cell.dimension == dim]
    min_mesh_size_v = np.min(
        list(map(functools.partial(cell_diam, mesh=mesh), cells_with_dim))
    )
    return min_mesh_size_v
