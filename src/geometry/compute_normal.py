import numpy as np

from mesh.mesh import Mesh
from mesh.mesh_cell import MeshCell


def normal(mesh: Mesh, cell_c0: MeshCell, cell_c1: MeshCell):
    xc_c1 = np.mean(mesh.points[cell_c1.node_tags], axis=0)
    xc_c0 = np.mean(mesh.points[cell_c0.node_tags], axis=0)
    outward = (xc_c1 - xc_c0) / np.linalg.norm((xc_c1 - xc_c0))
    if cell_c1.dimension == 0:
        normal = outward / np.linalg.norm(outward)
    elif cell_c1.dimension == 1:
        v = mesh.points[cell_c1.node_tags[1]] - mesh.points[cell_c1.node_tags[0]]
        v = v / np.linalg.norm(v)
        w = np.cross(v, outward) / np.linalg.norm(np.cross(v, outward))
        normal = np.cross(v, w)
    elif cell_c1.dimension == 2:
        v = mesh.points[cell_c1.node_tags[0]] - xc_c1
        w = mesh.points[cell_c1.node_tags[1]] - xc_c1
        normal = np.cross(v, w) / np.linalg.norm(np.cross(v, w))
    if normal @ outward < 0.0:
        normal *= -1.0
    return normal
