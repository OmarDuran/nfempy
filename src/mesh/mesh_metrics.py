import functools

import numpy as np


def cell_centroid(mesh_cell, mesh):
    points = mesh.points[mesh_cell.node_tags]
    xc = np.mean(points, axis=0)
    return xc


def cell_diam(mesh_cell, mesh):
    points = mesh.points[mesh_cell.node_tags]
    dxs = [
        p_t - p_o
        for i, p_o in enumerate(points)
        for j, p_t in enumerate(points)
        if i != j
    ]
    cell_diameter = np.max([np.linalg.norm(dx) for dx in dxs])
    return cell_diameter


def mesh_size(mesh, dim=None):
    if dim is None:
        dim = mesh.dimension
    cells_with_dim = [cell for cell in mesh.cells if cell.dimension == dim]
    cell_sizes = list(map(functools.partial(cell_diam, mesh=mesh), cells_with_dim))
    min_mesh_size_v = np.min(cell_sizes)
    mean_mesh_size_v = np.mean(cell_sizes)
    max_mesh_size_v = np.max(cell_sizes)
    return min_mesh_size_v, mean_mesh_size_v, max_mesh_size_v
