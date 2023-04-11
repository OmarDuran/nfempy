import functools
from functools import partial

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse import coo_matrix

from basis.finite_element import FiniteElement
from geometry.geometry_builder import GeometryBuilder
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh
from spaces.dof_map import DoFMap
from topology.mesh_topology import MeshTopology


k_orders = [1, 2, 3, 4, 5]

s_functions = [
    lambda x, y, z: x + y,
    lambda x, y, z: x * (1.0 - x) + y * (1.0 - y),
    lambda x, y, z: (x**2) * (1.0 - x) + (y**2) * (1.0 - y),
    lambda x, y, z: (x**3) * (1.0 - x) + (y**3) * (1.0 - y),
    lambda x, y, z: (x**4) * (1.0 - x) + (y**4) * (1.0 - y),
]

v_functions = [
    lambda x, y, z: np.array([y, -x, -z]),
    lambda x, y, z: np.array([(1 - y) * y**1, -(1 - x) * x**1, -(1 - z) * z**1]),
    lambda x, y, z: np.array([(1 - y) * y**2, -(1 - x) * x**2, -(1 - z) * z**2]),
    lambda x, y, z: np.array([(1 - y) * y**3, -(1 - x) * x**3, -(1 - z) * z**3]),
    lambda x, y, z: np.array([(1 - y) * y**4, -(1 - x) * x**4, -(1 - z) * z**4]),
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
    gmesh.build_conformal_mesh_II()
    # gmesh.write_vtk()
    return gmesh


@pytest.mark.parametrize("k_order", k_orders)
def test_h1_projector(k_order):

    h_cell = 1.0
    # scalar functions
    fun = s_functions[k_order - 1]

    # FESpace: data
    family = "Lagrange"

    for discontinuous in [True, False]:
        for dim in [1, 2, 3]:
            element_type = FiniteElement.type_by_dimension(dim)
            basis_family = FiniteElement.basis_family(family)
            basis_variant = FiniteElement.basis_variant()

            gmesh = generate_mesh(h_cell, dim)

            # Entities by codimension
            # https://defelement.com/ciarlet.html
            mesh_topology = MeshTopology(gmesh)
            cell_ids = mesh_topology.entities_by_codimension(0)

            elements = list(
                map(
                    partial(
                        FiniteElement,
                        mesh=gmesh,
                        k_order=k_order,
                        family=family,
                        discontinuous=discontinuous,
                    ),
                    cell_ids,
                )
            )

            # Assembler
            # Triplets data
            c_size = 0
            n_dof_g = 0
            cell_map = {}
            for element in elements:
                cell = element.cell
                n_dof = 0
                for n_entity_dofs in element.basis_generator.num_entity_dofs:
                    n_dof = n_dof + sum(n_entity_dofs)
                cell_map.__setitem__(cell.id, c_size)
                c_size = c_size + n_dof * n_dof

            row = np.zeros((c_size), dtype=np.int64)
            col = np.zeros((c_size), dtype=np.int64)
            data = np.zeros((c_size), dtype=np.float64)

            # DoF map for a variable supported on the element type
            dof_map = DoFMap(
                mesh_topology,
                basis_family,
                element_type,
                k_order,
                basis_variant,
                discontinuous=discontinuous,
            )
            dof_map.build_entity_maps()
            n_dof_g = dof_map.dof_number()
            rg = np.zeros(n_dof_g)

            def scatter_el_data(element, fun, dof_map, cell_map, row, col, data):

                cell = element.cell
                points, weights = element.quadrature
                phi_tab = element.phi
                (x, jac, det_jac, inv_jac) = element.mapping

                n_dof = element.phi.shape[2]
                js = (n_dof, n_dof)
                rs = n_dof
                j_el = np.zeros(js)
                r_el = np.zeros(rs)

                # linear_base
                for i, omega in enumerate(weights):
                    f_val = fun(x[i, 0], x[i, 1], x[i, 2])
                    r_el = r_el + det_jac[i] * omega * f_val * phi_tab[i, :, 0]
                    j_el = j_el + det_jac[i] * omega * np.outer(
                        phi_tab[i, :, 0], phi_tab[i, :, 0]
                    )

                # scattering dof
                dest = dof_map.destination_indices(cell.id)
                dest = dest[element.dof_ordering]

                c_sequ = cell_map[cell.id]

                # contribute rhs
                rg[dest] += r_el

                # contribute lhs
                block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
                row[block_sequ] += np.repeat(dest, len(dest))
                col[block_sequ] += np.tile(dest, len(dest))
                data[block_sequ] += j_el.ravel()

            [
                scatter_el_data(element, fun, dof_map, cell_map, row, col, data)
                for element in elements
            ]

            jg = coo_matrix((data, (row, col)), shape=(n_dof_g, n_dof_g)).tocsr()
            alpha = sp.linalg.spsolve(jg, rg)

            # Computing L2 error
            def compute_l2_error(element, dof_map):
                l2_error = 0.0
                cell = element.cell
                # scattering dof
                dest = dof_map.destination_indices(cell.id)
                dest = dest[element.dof_ordering]
                alpha_l = alpha[dest]

                (x, jac, det_jac, inv_jac) = element.mapping
                points, weights = element.quadrature
                phi_tab = element.phi
                for i, pt in enumerate(points):
                    p_e = fun(x[i, 0], x[i, 1], x[i, 2])
                    p_h = np.dot(alpha_l, phi_tab[i, :, 0])
                    l2_error += det_jac[i] * weights[i] * (p_h - p_e) * (p_h - p_e)

                return l2_error

            error_vec = [compute_l2_error(element, dof_map) for element in elements]
            l2_error = functools.reduce(lambda x, y: x + y, error_vec)
            l2_error_q = np.isclose(np.sqrt(l2_error), 0.0, atol=1.0e-14)
            assert l2_error_q


@pytest.mark.parametrize("k_order", k_orders)
def test_hdiv_hcurl_projector(k_order):

    h_cell = 1.0
    # scalar functions
    fun = v_functions[k_order - 1]

    discontinuous = True
    # FESpace: data
    for family in ["RT", "BDM"]:
        if family in ["RT", "N1E", "N2E"]:
            k_order = k_order + 1
        for dim in [2, 3]:

            element_type = FiniteElement.type_by_dimension(dim)
            basis_family = FiniteElement.basis_family(family)
            basis_variant = FiniteElement.basis_variant()

            gmesh = generate_mesh(h_cell, dim)

            # Entities by codimension
            # https://defelement.com/ciarlet.html
            mesh_topology = MeshTopology(gmesh)
            cell_ids = mesh_topology.entities_by_codimension(0)

            elements = list(
                map(
                    partial(
                        FiniteElement,
                        mesh=gmesh,
                        k_order=k_order,
                        family=family,
                        discontinuous=discontinuous,
                    ),
                    cell_ids,
                )
            )

            # Assembler
            # Triplets data
            c_size = 0
            n_dof_g = 0
            cell_map = {}
            for element in elements:
                cell = element.cell
                n_dof = 0
                for n_entity_dofs in element.basis_generator.num_entity_dofs:
                    n_dof = n_dof + sum(n_entity_dofs)
                cell_map.__setitem__(cell.id, c_size)
                c_size = c_size + n_dof * n_dof

            row = np.zeros((c_size), dtype=np.int64)
            col = np.zeros((c_size), dtype=np.int64)
            data = np.zeros((c_size), dtype=np.float64)

            # DoF map for a variable supported on the element type
            dof_map = DoFMap(
                mesh_topology,
                basis_family,
                element_type,
                k_order,
                basis_variant,
                discontinuous=discontinuous,
            )
            dof_map.build_entity_maps()
            n_dof_g = dof_map.dof_number()
            rg = np.zeros(n_dof_g)

            def scatter_el_data(element, fun, dof_map, cell_map, row, col, data):

                cell = element.cell
                points, weights = element.quadrature
                phi_tab = element.phi
                (x, jac, det_jac, inv_jac) = element.mapping

                n_dof = element.phi.shape[1]
                js = (n_dof, n_dof)
                rs = n_dof
                j_el = np.zeros(js)
                r_el = np.zeros(rs)

                # linear_base
                for i, omega in enumerate(weights):
                    f_val = fun(x[i, 0], x[i, 1], x[i, 2])
                    r_el = r_el + det_jac[i] * omega * phi_tab[i, :, :] @ f_val
                    for d in range(3):
                        j_el = j_el + det_jac[i] * omega * np.outer(
                            phi_tab[i, :, d], phi_tab[i, :, d]
                        )

                # scattering dof
                dest = dof_map.destination_indices(cell.id)
                dest = dest[element.dof_ordering]

                c_sequ = cell_map[cell.id]

                # contribute rhs
                rg[dest] += r_el

                # contribute lhs
                block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
                row[block_sequ] += np.repeat(dest, len(dest))
                col[block_sequ] += np.tile(dest, len(dest))
                data[block_sequ] += j_el.ravel()

            [
                scatter_el_data(element, fun, dof_map, cell_map, row, col, data)
                for element in elements
            ]

            jg = coo_matrix((data, (row, col)), shape=(n_dof_g, n_dof_g)).tocsr()
            alpha = sp.linalg.spsolve(jg, rg)

            # Computing L2 error
            def compute_l2_error(element, dof_map):
                l2_error = 0.0
                cell = element.cell
                # scattering dof
                dest = dof_map.destination_indices(cell.id)
                dest = dest[element.dof_ordering]
                alpha_l = alpha[dest]

                (x, jac, det_jac, inv_jac) = element.mapping
                points, weights = element.quadrature
                phi_tab = element.phi
                for i, pt in enumerate(points):
                    u_e = fun(x[i, 0], x[i, 1], x[i, 2])
                    u_h = np.dot(alpha_l, phi_tab[i, :, :])
                    l2_error += (
                        det_jac[i] * weights[i] * np.dot((u_h - u_e), (u_h - u_e))
                    )

                return l2_error

            error_vec = [compute_l2_error(element, dof_map) for element in elements]
            l2_error = functools.reduce(lambda x, y: x + y, error_vec)
            l2_error_q = np.isclose(np.sqrt(l2_error), 0.0, atol=1.0e-14)
            assert l2_error_q
