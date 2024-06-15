import functools
from functools import partial

import numpy as np
import pytest
from geometry.geometry_builder import GeometryBuilder
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh
from spaces.product_space import ProductSpace
from postprocess.l2_error_post_processor import l2_error
from postprocess.projectors import l2_projector

k_orders = [1, 2, 3, 4, 5]

s_functions = [
    lambda x, y, z: np.array([x + y]),
    lambda x, y, z: np.array([x * (1.0 - x) + y * (1.0 - y)]),
    lambda x, y, z: np.array([(x**2) * (1.0 - x) + (y**2) * (1.0 - y)]),
    lambda x, y, z: np.array([(x**3) * (1.0 - x) + (y**3) * (1.0 - y)]),
    lambda x, y, z: np.array([(x**4) * (1.0 - x) + (y**4) * (1.0 - y)]),
]

v_functions = [
    lambda x, y, z: np.array([[y, -x, -z]]),
    lambda x, y, z: np.array(
        [[(1 - y) * y**1, -(1 - x) * x**1, -(1 - z) * z**1]]
    ),
    lambda x, y, z: np.array(
        [[(1 - y) * y**2, -(1 - x) * x**2, -(1 - z) * z**2]]
    ),
    lambda x, y, z: np.array(
        [[(1 - y) * y**3, -(1 - x) * x**3, -(1 - z) * z**3]]
    ),
    lambda x, y, z: np.array(
        [[(1 - y) * y**4, -(1 - x) * x**4, -(1 - z) * z**4]]
    ),
]


def generate_geometry_1d():
    box_points = np.array([[0, 0, 0], [1, 0, 0]])
    g_builder = GeometryBuilder(dimension=1)
    g_builder.build_box_1D(box_points)
    return g_builder


def generate_geometry_2d():
    box_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    g_builder = GeometryBuilder(dimension=2)
    g_builder.build_box_2D(box_points)
    return g_builder


def generate_geometry_3d():
    box_points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )
    g_builder = GeometryBuilder(dimension=3)
    g_builder.build_box(box_points)
    return g_builder


def generate_mesh(h_cell, dim):
    g_builder = None
    if dim == 1:
        g_builder = generate_geometry_1d()
    elif dim == 2:
        g_builder = generate_geometry_2d()
    elif dim == 3:
        g_builder = generate_geometry_3d()

    conformal_mesher = ConformalMesher(dimension=dim)
    conformal_mesher.set_geometry_builder(g_builder)
    conformal_mesher.set_points()
    conformal_mesher.generate(h_cell)
    conformal_mesher.write_mesh("gmesh.msh")

    gmesh = Mesh(dimension=dim, file_name="gmesh.msh")
    gmesh.set_conformal_mesher(conformal_mesher)
    gmesh.build_conformal_mesh()
    # gmesh.write_vtk()
    return gmesh


@pytest.mark.parametrize("k_order", k_orders)
def test_scalar_h1_projector(k_order):
    h_cell = 1.0
    # scalar functions
    s_fun = s_functions[k_order - 1]

    # FESpace: data
    n_components = 1
    family = "Lagrange"
    exact_functions = {
        "s": s_fun,
    }

    for discontinuous in [True, False]:
        for dim in [1, 2, 3]:
            gmesh = generate_mesh(h_cell, dim)

            discrete_spaces_data = {
                "s": (dim, n_components, family, k_order, gmesh),
            }

            v_disc_Q = discontinuous
            discrete_spaces_disc = {
                "s": v_disc_Q,
            }

            discrete_spaces_bc_physical_tags = {
                "s": [],
            }

            space = ProductSpace(discrete_spaces_data)
            space.make_subspaces_discontinuous(discrete_spaces_disc)
            space.build_structures(discrete_spaces_bc_physical_tags)

            alpha = l2_projector(space, exact_functions)
            error_val = l2_error(dim, space, exact_functions, alpha)
            l2_error_q = np.any(np.isclose(np.array(error_val), 0.0, atol=1.0e-13))
            assert l2_error_q


@pytest.mark.parametrize("k_order", k_orders)
def abc_test_vector_hdiv_projector(k_order):
    h_cell = 1.0
    # vector functions
    v_fun = v_functions[k_order - 1]

    # FESpace: data
    n_components = 1
    exact_functions = {
        "v": v_fun,
    }

    for discontinuous in [True, False]:
        for family in ["RT", "BDM"]:
            if family in ["RT"]:
                k_order = k_order + 1
            for dim in [2, 3]:
                gmesh = generate_mesh(h_cell, dim)

                discrete_spaces_data = {
                    "v": (dim, n_components, family, k_order, gmesh),
                }

                v_disc_Q = discontinuous
                discrete_spaces_disc = {
                    "v": v_disc_Q,
                }

                discrete_spaces_bc_physical_tags = {
                    "v": [],
                }

                space = ProductSpace(discrete_spaces_data)
                space.make_subspaces_discontinuous(discrete_spaces_disc)
                space.build_structures(discrete_spaces_bc_physical_tags)

                alpha = l2_projector(space, exact_functions)
                error_val = l2_error(dim, space, exact_functions, alpha)
                l2_error_q = np.any(np.isclose(np.array(error_val), 0.0, atol=1.0e-13))
                assert l2_error_q


@pytest.mark.parametrize("k_order", k_orders)
def abc_test_vector_hcurl_projector(k_order):
    h_cell = 1.0
    # vector functions
    v_fun = v_functions[k_order - 1]

    # FESpace: data
    n_components = 1
    exact_functions = {
        "v": v_fun,
    }

    for discontinuous in [False]:
        for family in ["N1E", "N2E"]:
            if family in ["N1E"]:
                k_order = k_order + 1
            for dim in [2, 3]:
                gmesh = generate_mesh(h_cell, dim)

                discrete_spaces_data = {
                    "v": (dim, n_components, family, k_order, gmesh),
                }

                v_disc_Q = discontinuous
                discrete_spaces_disc = {
                    "v": v_disc_Q,
                }

                discrete_spaces_bc_physical_tags = {
                    "v": [],
                }

                space = ProductSpace(discrete_spaces_data)
                space.make_subspaces_discontinuous(discrete_spaces_disc)
                space.build_structures(discrete_spaces_bc_physical_tags)

                alpha = l2_projector(space, exact_functions)
                error_val = l2_error(dim, space, exact_functions, alpha)
                l2_error_q = np.any(np.isclose(np.array(error_val), 0.0, atol=1.0e-13))
                assert l2_error_q
