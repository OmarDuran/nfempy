import pytest
import numpy as np
from geometry.geometry_builder import GeometryBuilder
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh

fracture_tags = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]]


def generate_geometry_2d():
    box_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    g_builder = GeometryBuilder(dimension=2)
    g_builder.build_box_2D(box_points)
    return g_builder


def fracture_2d_set():
    fracture_1 = np.array([[0.5, 0.25, 0], [0.5, 0.75, 0]])
    fracture_2 = np.array([[0.25, 0.5, 0], [0.75, 0.5, 0]])
    fracture_3 = np.array([[0.2, 0.35, 0], [0.85, 0.35, 0]])
    fracture_4 = np.array([[0.15, 0.15, 0], [0.85, 0.85, 0]])
    fracture_5 = np.array([[0.15, 0.85, 0], [0.85, 0.15, 0]])
    fractures = [fracture_1, fracture_2, fracture_3, fracture_4, fracture_5]
    return fractures


def generate_fracture_network(fractures):
    fracture_network = fn.FractureNetwork(dimension=2, physical_tag_shift=10)
    fracture_network.intersect_1D_fractures(fractures)
    fracture_network.build_grahp(all_fixed_d_cells_q=True)
    return fracture_network


def generate_conformal_mesh(fracture_tags):
    mesher = ConformalMesher(dimension=2)
    mesher.set_geometry_builder(generate_geometry_2d())
    fractures = []
    for tag in fracture_tags:
        fractures.append(fracture_2d_set()[tag])
    mesher.set_fracture_network(generate_fracture_network(fractures))
    mesher.set_points()
    mesher.generate(0.1)
    mesher.write_mesh("gmesh.msh")
    return mesher


def generate_mesh(fracture_tags):
    conformal_mesh = generate_conformal_mesh(fracture_tags)
    gmesh = Mesh(dimension=2, file_name="gmesh.msh")
    gmesh.set_conformal_mesher(conformal_mesh)
    gmesh.build_conformal_mesh()
    gmesh.cut_conformity_on_fractures_mds_ec()
    return gmesh


@pytest.mark.parametrize("fracture_tags", fracture_tags)
def test_internal_bc_mesh_circulation(fracture_tags):
    gmesh = generate_mesh(fracture_tags)
    check_q = gmesh.circulate_internal_bc()
    assert check_q[0]
