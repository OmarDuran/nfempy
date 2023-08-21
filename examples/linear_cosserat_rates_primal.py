import copy
import csv
import functools
import marshal
import sys
import time

# from itertools import permutations
from functools import partial, reduce

import basix
import matplotlib.colors as mcolors

# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plot
import meshio
import networkx as nx
import numpy as np
import pypardiso as sp_solver
import scipy.sparse as sp
from numpy import linalg as la
from scipy.sparse import coo_matrix
from shapely.geometry import LineString

import geometry.fracture_network as fn
from basis.element_data import ElementData
from basis.finite_element import FiniteElement
from geometry.domain import Domain
from geometry.domain_market import (
    build_box_1D,
    build_box_2D,
    build_box_2D_with_lines,
    build_box_3D,
    build_box_3D_with_planes,
    build_disjoint_lines,
    read_fractures_file,
)
from geometry.edge import Edge
from geometry.geometry_builder import GeometryBuilder
from geometry.geometry_cell import GeometryCell
from geometry.mapping import evaluate_linear_shapes, evaluate_mapping, store_mapping
from geometry.shape_manipulation import ShapeManipulation
from geometry.vertex import Vertex
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh
from spaces.discrete_field import DiscreteField
from spaces.dof_map import DoFMap
from topology.mesh_topology import MeshTopology


def matrix_plot(J, sparse_q=True):

    if sparse_q:
        plot.matshow(J.todense())
    else:
        plot.matshow(J)
    plot.colorbar(orientation="vertical")
    plot.set_cmap("seismic")
    plot.show()


def h1_elasticity(k_order, gmesh, write_vtk_q=False):

    dim = gmesh.dimension
    # Material data

    m_lambda = 1.0
    m_mu = 1.0

    # FESpace: data
    n_components = 2
    if dim == 3:
        n_components = 3

    discontinuous = True
    family = "Lagrange"

    u_field = DiscreteField(dim, n_components, family, k_order, gmesh)
    # u_field.build_structures([2, 3])
    if dim == 2:
        u_field.build_structures([2, 3, 4, 5])
    elif dim == 3:
        u_field.build_structures([2, 3, 4, 5, 6, 7])

    st = time.time()
    # Assembler
    # Triplets data
    c_size = 0
    n_dof_g = 0
    cell_map = {}
    for element in u_field.elements:
        cell = element.data.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs) * n_components
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    for element in u_field.bc_elements:
        cell = element.data.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs) * n_components
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    row = np.zeros((c_size), dtype=np.int64)
    col = np.zeros((c_size), dtype=np.int64)
    data = np.zeros((c_size), dtype=np.float64)

    n_dof_g = u_field.dof_map.dof_number()
    rg = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    et = time.time()
    elapsed_time = et - st
    print("Triplets creation time:", elapsed_time, "seconds")

    st = time.time()

    # facturated solution for
    f_exact = lambda x, y, z: np.array(
        [np.sin(np.pi * x) * y * (1 - y), np.sin(np.pi * y) * x * (1 - x)]
    )
    f_rhs_x = lambda x, y, z: -(
        np.pi * (-1 + 2 * x) * (m_lambda + m_mu) * np.cos(np.pi * y)
    ) + (-2 * m_mu + (np.pi**2) * (-1 + y) * y * (m_lambda + 2 * m_mu)) * np.sin(
        np.pi * x
    )
    f_rhs_y = lambda x, y, z: -(
        np.pi * (-1 + 2 * y) * (m_lambda + m_mu) * np.cos(np.pi * x)
    ) + (-2 * m_mu + (np.pi**2) * (-1 + x) * x * (m_lambda + 2 * m_mu)) * np.sin(
        np.pi * y
    )
    f_rhs = lambda x, y, z: np.array([-f_rhs_x(x, y, z), -f_rhs_y(x, y, z)])
    if dim == 3:
        f_exact = lambda x, y, z: np.array(
            [
                np.sin(np.pi * x) * y * (1 - y) * z * (1 - z),
                np.sin(np.pi * y) * x * (1 - x) * z * (1 - z),
                np.sin(np.pi * z) * x * (1 - x) * y * (1 - y),
            ]
        )
        f_rhs_x = (
            lambda x, y, z: -2
            * (np.pi**2)
            * (-1 + y)
            * y
            * (-1 + z)
            * z
            * m_mu
            * np.sin(np.pi * x)
            + (-1 + z)
            * z
            * m_mu
            * (np.pi * (-1 + 2 * x) * np.cos(np.pi * y) + 2 * np.sin(np.pi * x))
            + (-1 + y)
            * y
            * m_mu
            * (np.pi * (-1 + 2 * x) * np.cos(np.pi * z) + 2 * np.sin(np.pi * x))
            + np.pi
            * m_lambda
            * (
                (-1 + 2 * x) * (-1 + z) * z * np.cos(np.pi * y)
                + (-1 + y)
                * y
                * (
                    (-1 + 2 * x) * np.cos(np.pi * z)
                    - np.pi * (-1 + z) * z * np.sin(np.pi * x)
                )
            )
        )
        f_rhs_y = (
            lambda x, y, z: -2
            * (np.pi**2)
            * (-1 + x)
            * x
            * (-1 + z)
            * z
            * m_mu
            * np.sin(np.pi * y)
            + (-1 + z)
            * z
            * m_mu
            * (np.pi * (-1 + 2 * y) * np.cos(np.pi * x) + 2 * np.sin(np.pi * y))
            + (-1 + x)
            * x
            * m_mu
            * (np.pi * (-1 + 2 * y) * np.cos(np.pi * z) + 2 * np.sin(np.pi * y))
            + np.pi
            * m_lambda
            * (
                (-1 + 2 * y) * (-1 + z) * z * np.cos(np.pi * x)
                + (-1 + x)
                * x
                * (
                    (-1 + 2 * y) * np.cos(np.pi * z)
                    - np.pi * (-1 + z) * z * np.sin(np.pi * y)
                )
            )
        )
        f_rhs_z = (
            lambda x, y, z: -2
            * (np.pi**2)
            * (-1 + x)
            * x
            * (-1 + y)
            * y
            * m_mu
            * np.sin(np.pi * z)
            + (-1 + y)
            * y
            * m_mu
            * (np.pi * (-1 + 2 * z) * np.cos(np.pi * x) + 2 * np.sin(np.pi * z))
            + (-1 + x)
            * x
            * m_mu
            * (np.pi * (-1 + 2 * z) * np.cos(np.pi * y) + 2 * np.sin(np.pi * z))
            + np.pi
            * m_lambda
            * (
                (-1 + y) * y * (-1 + 2 * z) * np.cos(np.pi * x)
                + (-1 + x)
                * x
                * (
                    (-1 + 2 * z) * np.cos(np.pi * y)
                    - np.pi * (-1 + y) * y * np.sin(np.pi * z)
                )
            )
        )
        f_rhs = lambda x, y, z: np.array(
            [-f_rhs_x(x, y, z), -f_rhs_y(x, y, z), -f_rhs_z(x, y, z)]
        )

    def scatter_form_data(
        element, m_lambda, m_mu, f_rhs, u_field, cell_map, row, col, data
    ):

        n_components = u_field.n_comp
        el_data: ElementData = element.data

        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # destination indexes
        dest = u_field.dof_map.destination_indices(cell.id)

        n_phi = phi_tab.shape[2]
        n_dof = n_phi * n_components
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # Partial local vectorization
        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
        phi_s_star = det_jac * weights * phi_tab[0, :, :, 0].T

        for c in range(n_components):
            b = c
            e = (c + 1) * n_phi * n_components + 1
            r_el[b:e:n_components] += phi_s_star @ f_val_star[c]

        # vectorized blocks
        phi_star_dirs = [[1, 2], [0, 2], [0, 1]]
        for i, omega in enumerate(weights):
            grad_phi = inv_jac[i].T @ phi_tab[1 : phi_tab.shape[0] + 1, i, :, 0]

            for i_d in range(n_components):
                for j_d in range(n_components):
                    phi_outer = np.outer(grad_phi[j_d], grad_phi[i_d])
                    stress_grad = m_mu * phi_outer
                    if i_d == j_d:
                        phi_outer_star = np.zeros((n_phi, n_phi))
                        for d in phi_star_dirs[i_d]:
                            phi_outer_star += np.outer(grad_phi[d], grad_phi[d])
                        stress_grad += (
                            m_lambda + m_mu
                        ) * phi_outer + m_mu * phi_outer_star
                    else:
                        stress_grad += m_lambda * np.outer(grad_phi[i_d], grad_phi[j_d])
                    j_el[
                        i_d : n_dof + 1 : n_components, j_d : n_dof + 1 : n_components
                    ] += (det_jac[i] * omega * stress_grad)

        # scattering data
        c_sequ = cell_map[cell.id]

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

    [
        scatter_form_data(
            element, m_lambda, m_mu, f_rhs, u_field, cell_map, row, col, data
        )
        for element in u_field.elements
    ]

    def scatter_bc_form_data(element, u_field, cell_map, row, col, data):

        n_components = u_field.n_comp
        el_data: ElementData = element.data

        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # find high-dimension neigh
        entity_map = u_field.dof_map.mesh_topology.entity_map_by_dimension(
            cell.dimension
        )
        neigh_list = list(entity_map.predecessors(cell.id))
        neigh_check_q = len(neigh_list) > 0
        assert neigh_check_q

        neigh_cell_id = neigh_list[0]
        neigh_cell_index = u_field.id_to_element[neigh_cell_id]
        neigh_cell = u_field.elements[neigh_cell_index].data.cell

        # destination indexes
        dest_neigh = u_field.dof_map.destination_indices(neigh_cell_id)
        dest = u_field.dof_map.bc_destination_indices(neigh_cell_id, cell.id)

        n_phi = phi_tab.shape[2]
        n_dof = n_phi * n_components
        js = (n_dof, n_dof)
        j_el = np.zeros(js)

        # local blocks
        beta = 1.0e12
        jac_block = np.zeros((n_phi, n_phi))
        for i, omega in enumerate(weights):
            phi = phi_tab[0, i, :, 0]
            jac_block += beta * det_jac[i] * omega * np.outer(phi, phi)

        for c in range(n_components):
            b = c
            e = (c + 1) * n_phi * n_components + 1
            j_el[b:e:n_components, b:e:n_components] += jac_block

        # scattering data
        c_sequ = cell_map[cell.id]

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

    [
        scatter_bc_form_data(element, u_field, cell_map, row, col, data)
        for element in u_field.bc_elements
    ]

    jg = coo_matrix((data, (row, col)), shape=(n_dof_g, n_dof_g)).tocsr()
    et = time.time()
    elapsed_time = et - st
    print("Assembly time:", elapsed_time, "seconds")

    # solving ls
    st = time.time()
    # alpha = sp.linalg.spsolve(jg, rg)
    alpha = sp_solver.spsolve(jg, rg)
    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    # Computing L2 error
    def compute_l2_error(element, u_field):
        l2_error = 0.0
        n_components = u_field.n_comp
        el_data = element.data
        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # scattering dof
        dest = u_field.dof_map.destination_indices(cell.id)
        alpha_l = alpha[dest]

        # vectorization
        n_phi = phi_tab.shape[2]
        p_e_s = f_exact(x[:, 0], x[:, 1], x[:, 2])
        alpha_star = np.array(np.split(alpha_l, n_phi))
        p_h_s = (phi_tab[0, :, :, 0] @ alpha_star).T
        diff_p = p_e_s - p_h_s
        l2_error = np.sum(det_jac * weights * diff_p * diff_p)
        return l2_error

    st = time.time()
    error_vec = [compute_l2_error(element, u_field) for element in u_field.elements]
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, error_vec))
    print("L2-error: ", l2_error)

    if write_vtk_q:
        # post-process solution
        st = time.time()
        cellid_to_element = dict(zip(u_field.element_ids, u_field.elements))
        # writing solution on mesh points
        vertices = u_field.mesh_topology.entities_by_dimension(0)
        fh_data = np.zeros((len(gmesh.points), n_components))
        fe_data = np.zeros((len(gmesh.points), n_components))
        cell_vertex_map = u_field.mesh_topology.entity_map_by_dimension(0)
        for id in vertices:
            if not cell_vertex_map.has_node(id):
                continue

            pr_ids = list(cell_vertex_map.predecessors(id))
            cell = gmesh.cells[pr_ids[0]]
            if cell.dimension != u_field.dimension:
                continue

            element = cellid_to_element[pr_ids[0]]

            # scattering dof
            dest = u_field.dof_map.destination_indices(cell.id)
            alpha_l = alpha[dest]

            par_points = basix.geometry(u_field.element_type)

            target_node_id = gmesh.cells[id].node_tags[0]
            par_point_id = np.array(
                [
                    i
                    for i, node_id in enumerate(cell.node_tags)
                    if node_id == target_node_id
                ]
            )

            points = gmesh.points[target_node_id]
            if u_field.dimension != 0:
                points = par_points[par_point_id]

            # evaluate mapping
            phi_shapes = evaluate_linear_shapes(points, element.data)
            (x, jac, det_jac, inv_jac) = evaluate_mapping(
                cell.dimension, phi_shapes, gmesh.points[cell.node_tags]
            )
            phi_tab = element.evaluate_basis(points)
            n_phi = phi_tab.shape[2]
            f_e = f_exact(x[:, 0], x[:, 1], x[:, 2])
            alpha_star = np.array(np.split(alpha_l, n_phi))
            f_h = (phi_tab[0, :, :, 0] @ alpha_star).T
            fh_data[target_node_id] = f_h.ravel()
            fe_data[target_node_id] = f_e.ravel()

        mesh_points = gmesh.points
        con_d = np.array([element.data.cell.node_tags for element in u_field.elements])
        meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
        cells_dict = {meshio_cell_types[u_field.dimension]: con_d}
        p_data_dict = {"u_h": fh_data, "u_exact": fe_data}

        mesh = meshio.Mesh(
            points=mesh_points,
            cells=cells_dict,
            # Optionally provide extra data on points, cells, etc.
            point_data=p_data_dict,
        )
        mesh.write("rates_h1_elasticity.vtk")
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return l2_error


def h1_cosserat_elasticity(k_order, gmesh, write_vtk_q=False):

    dim = gmesh.dimension
    # Material data

    m_lambda = 1.0
    m_mu = 1.0
    m_kappa = 1.0
    m_gamma = 1.0

    # FESpace: data
    n_components = 3
    if dim == 3:
        n_components = 6

    family = "Lagrange"

    u_field = DiscreteField(dim, n_components, family, k_order, gmesh)
    # u_field.build_structures([2, 3])
    if dim == 2:
        u_field.build_structures([2, 3, 4, 5])
    elif dim == 3:
        u_field.build_structures([2, 3, 4, 5, 6, 7])

    st = time.time()
    # Assembler
    # Triplets data
    c_size = 0
    n_dof_g = 0
    cell_map = {}
    for element in u_field.elements:
        cell = element.data.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs) * n_components
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    for element in u_field.bc_elements:
        cell = element.data.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs) * n_components
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    row = np.zeros((c_size), dtype=np.int64)
    col = np.zeros((c_size), dtype=np.int64)
    data = np.zeros((c_size), dtype=np.float64)

    n_dof_g = u_field.dof_map.dof_number()
    rg = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    et = time.time()
    elapsed_time = et - st
    print("Triplets creation time:", elapsed_time, "seconds")

    st = time.time()

    # smooth solution
    f_exact = lambda x, y, z: np.array(
        [
            np.sin(np.pi * x) * y * (1 - y),
            np.sin(np.pi * y) * x * (1 - x),
            np.sin(np.pi * x) * np.sin(np.pi * y),
        ]
    )
    f_rhs_x = lambda x, y, z: -(
        (2 * m_kappa + 2 * m_mu - np.pi**2 * (-1 + y) * y * (m_lambda + 2 * m_mu))
        * np.sin(np.pi * x)
    ) + np.pi * np.cos(np.pi * y) * (
        (-1 + 2 * x) * (m_kappa - m_lambda - m_mu) + 2 * m_kappa * np.sin(np.pi * x)
    )
    f_rhs_y = lambda x, y, z: -(
        (2 * m_kappa + 2 * m_mu - np.pi**2 * (-1 + x) * x * (m_lambda + 2 * m_mu))
        * np.sin(np.pi * y)
    ) + np.pi * np.cos(np.pi * x) * (
        (-1 + 2 * y) * (m_kappa - m_lambda - m_mu) - 2 * m_kappa * np.sin(np.pi * y)
    )
    f_rhs_t = lambda x, y, z: -2 * (
        (-1 + 2 * x) * m_kappa * np.sin(np.pi * y)
        + np.sin(np.pi * x)
        * (
            m_kappa
            - 2 * y * m_kappa
            + ((np.pi**2) * m_gamma + 2 * m_kappa) * np.sin(np.pi * y)
        )
    )
    f_rhs = lambda x, y, z: np.array(
        [-f_rhs_x(x, y, z), -f_rhs_y(x, y, z), -f_rhs_t(x, y, z)]
    )
    if dim == 3:
        f_exact = lambda x, y, z: np.array(
            [
                (1 - y) * y * (1 - z) * z * np.sin(np.pi * x),
                (1 - x) * x * (1 - z) * z * np.sin(np.pi * y),
                (1 - x) * x * (1 - y) * y * np.sin(np.pi * z),
                (1 - x) * x * np.sin(np.pi * y) * np.sin(np.pi * z),
                (1 - y) * y * np.sin(np.pi * x) * np.sin(np.pi * z),
                (1 - z) * z * np.sin(np.pi * x) * np.sin(np.pi * y),
            ]
        )
        f_rhs_x = (
            lambda x, y, z: -2
            * (np.pi**2)
            * (1 - y)
            * y
            * (1 - z)
            * z
            * m_mu
            * np.sin(np.pi * x)
            + m_mu
            * (
                np.pi * (1 - x) * (1 - y) * y * np.cos(np.pi * z)
                - np.pi * x * (1 - y) * y * np.cos(np.pi * z)
                - 2 * (1 - y) * y * np.sin(np.pi * x)
            )
            + m_mu
            * (
                np.pi * (1 - x) * (1 - z) * z * np.cos(np.pi * y)
                - np.pi * x * (1 - z) * z * np.cos(np.pi * y)
                - 2 * (1 - z) * z * np.sin(np.pi * x)
            )
            + m_lambda
            * (
                np.pi * (1 - x) * (1 - z) * z * np.cos(np.pi * y)
                - np.pi * x * (1 - z) * z * np.cos(np.pi * y)
                + np.pi * (1 - x) * (1 - y) * y * np.cos(np.pi * z)
                - np.pi * x * (1 - y) * y * np.cos(np.pi * z)
                - (np.pi**2) * (1 - y) * y * (1 - z) * z * np.sin(np.pi * x)
            )
            + m_kappa
            * (
                -(np.pi * (1 - x) * (1 - z) * z * np.cos(np.pi * y))
                + np.pi * x * (1 - z) * z * np.cos(np.pi * y)
                - 2 * (1 - z) * z * np.sin(np.pi * x)
                + 2 * np.pi * z * np.cos(np.pi * y) * np.sin(np.pi * x)
                - 2 * np.pi * (z**2) * np.cos(np.pi * y) * np.sin(np.pi * x)
            )
            + m_kappa
            * (
                -(np.pi * (1 - x) * (1 - y) * y * np.cos(np.pi * z))
                + np.pi * x * (1 - y) * y * np.cos(np.pi * z)
                - 2 * (1 - y) * y * np.sin(np.pi * x)
                - 2 * np.pi * y * np.cos(np.pi * z) * np.sin(np.pi * x)
                + 2 * np.pi * (y**2) * np.cos(np.pi * z) * np.sin(np.pi * x)
            )
        )
        f_rhs_y = (
            lambda x, y, z: -2
            * (np.pi**2)
            * (1 - x)
            * x
            * (1 - z)
            * z
            * m_mu
            * np.sin(np.pi * y)
            + m_mu
            * (
                np.pi * (1 - x) * x * (1 - y) * np.cos(np.pi * z)
                - np.pi * (1 - x) * x * y * np.cos(np.pi * z)
                - 2 * (1 - x) * x * np.sin(np.pi * y)
            )
            + m_mu
            * (
                np.pi * (1 - y) * (1 - z) * z * np.cos(np.pi * x)
                - np.pi * y * (1 - z) * z * np.cos(np.pi * x)
                - 2 * (1 - z) * z * np.sin(np.pi * y)
            )
            + m_lambda
            * (
                np.pi * (1 - y) * (1 - z) * z * np.cos(np.pi * x)
                - np.pi * y * (1 - z) * z * np.cos(np.pi * x)
                + np.pi * (1 - x) * x * (1 - y) * np.cos(np.pi * z)
                - np.pi * (1 - x) * x * y * np.cos(np.pi * z)
                - (np.pi**2) * (1 - x) * x * (1 - z) * z * np.sin(np.pi * y)
            )
            + m_kappa
            * (
                -(np.pi * (1 - y) * (1 - z) * z * np.cos(np.pi * x))
                + np.pi * y * (1 - z) * z * np.cos(np.pi * x)
                - 2 * (1 - z) * z * np.sin(np.pi * y)
                - 2 * np.pi * z * np.cos(np.pi * x) * np.sin(np.pi * y)
                + 2 * np.pi * (z**2) * np.cos(np.pi * x) * np.sin(np.pi * y)
            )
            + m_kappa
            * (
                -(np.pi * (1 - x) * x * (1 - y) * np.cos(np.pi * z))
                + np.pi * (1 - x) * x * y * np.cos(np.pi * z)
                - 2 * (1 - x) * x * np.sin(np.pi * y)
                + 2 * np.pi * x * np.cos(np.pi * z) * np.sin(np.pi * y)
                - 2 * np.pi * (x**2) * np.cos(np.pi * z) * np.sin(np.pi * y)
            )
        )
        f_rhs_z = (
            lambda x, y, z: -2
            * (np.pi**2)
            * (1 - x)
            * x
            * (1 - y)
            * y
            * m_mu
            * np.sin(np.pi * z)
            + m_mu
            * (
                np.pi * (1 - x) * x * (1 - z) * np.cos(np.pi * y)
                - np.pi * (1 - x) * x * z * np.cos(np.pi * y)
                - 2 * (1 - x) * x * np.sin(np.pi * z)
            )
            + m_mu
            * (
                np.pi * (1 - y) * y * (1 - z) * np.cos(np.pi * x)
                - np.pi * (1 - y) * y * z * np.cos(np.pi * x)
                - 2 * (1 - y) * y * np.sin(np.pi * z)
            )
            + m_lambda
            * (
                np.pi * (1 - y) * y * (1 - z) * np.cos(np.pi * x)
                - np.pi * (1 - y) * y * z * np.cos(np.pi * x)
                + np.pi * (1 - x) * x * (1 - z) * np.cos(np.pi * y)
                - np.pi * (1 - x) * x * z * np.cos(np.pi * y)
                - (np.pi**2) * (1 - x) * x * (1 - y) * y * np.sin(np.pi * z)
            )
            + m_kappa
            * (
                -(np.pi * (1 - y) * y * (1 - z) * np.cos(np.pi * x))
                + np.pi * (1 - y) * y * z * np.cos(np.pi * x)
                - 2 * (1 - y) * y * np.sin(np.pi * z)
                + 2 * np.pi * y * np.cos(np.pi * x) * np.sin(np.pi * z)
                - 2 * np.pi * (y**2) * np.cos(np.pi * x) * np.sin(np.pi * z)
            )
            + m_kappa
            * (
                -(np.pi * (1 - x) * x * (1 - z) * np.cos(np.pi * y))
                + np.pi * (1 - x) * x * z * np.cos(np.pi * y)
                - 2 * (1 - x) * x * np.sin(np.pi * z)
                - 2 * np.pi * x * np.cos(np.pi * y) * np.sin(np.pi * z)
                + 2 * np.pi * (x**2) * np.cos(np.pi * y) * np.sin(np.pi * z)
            )
        )
        f_rhs_t_x = (
            lambda x, y, z: -2 * x * m_kappa * np.sin(np.pi * y)
            + 2 * (x**2) * m_kappa * np.sin(np.pi * y)
            + 4 * x * z * m_kappa * np.sin(np.pi * y)
            - 4 * (x**2) * z * m_kappa * np.sin(np.pi * y)
            + 2 * x * m_kappa * np.sin(np.pi * z)
            - 2 * (x**2) * m_kappa * np.sin(np.pi * z)
            - 4 * x * y * m_kappa * np.sin(np.pi * z)
            + 4 * (x**2) * y * m_kappa * np.sin(np.pi * z)
            - 2 * m_gamma * np.sin(np.pi * y) * np.sin(np.pi * z)
            - 2
            * (np.pi**2)
            * (1 - x)
            * x
            * m_gamma
            * np.sin(np.pi * y)
            * np.sin(np.pi * z)
            - 4 * x * m_kappa * np.sin(np.pi * y) * np.sin(np.pi * z)
            + 4 * (x**2) * m_kappa * np.sin(np.pi * y) * np.sin(np.pi * z)
        )
        f_rhs_t_y = (
            lambda x, y, z: 2 * y * m_kappa * np.sin(np.pi * x)
            - 2 * (y**2) * m_kappa * np.sin(np.pi * x)
            - 4 * y * z * m_kappa * np.sin(np.pi * x)
            + 4 * (y**2) * z * m_kappa * np.sin(np.pi * x)
            - 2 * y * m_kappa * np.sin(np.pi * z)
            + 4 * x * y * m_kappa * np.sin(np.pi * z)
            + 2 * (y**2) * m_kappa * np.sin(np.pi * z)
            - 4 * x * (y**2) * m_kappa * np.sin(np.pi * z)
            - 2 * m_gamma * np.sin(np.pi * x) * np.sin(np.pi * z)
            - 2
            * (np.pi**2)
            * (1 - y)
            * y
            * m_gamma
            * np.sin(np.pi * x)
            * np.sin(np.pi * z)
            - 4 * y * m_kappa * np.sin(np.pi * x) * np.sin(np.pi * z)
            + 4 * (y**2) * m_kappa * np.sin(np.pi * x) * np.sin(np.pi * z)
        )
        f_rhs_t_z = (
            lambda x, y, z: -2 * z * m_kappa * np.sin(np.pi * x)
            + 4 * y * z * m_kappa * np.sin(np.pi * x)
            + 2 * (z**2) * m_kappa * np.sin(np.pi * x)
            - 4 * y * (z**2) * m_kappa * np.sin(np.pi * x)
            + 2 * z * m_kappa * np.sin(np.pi * y)
            - 4 * x * z * m_kappa * np.sin(np.pi * y)
            - 2 * (z**2) * m_kappa * np.sin(np.pi * y)
            + 4 * x * (z**2) * m_kappa * np.sin(np.pi * y)
            - 2 * m_gamma * np.sin(np.pi * x) * np.sin(np.pi * y)
            - 2
            * (np.pi**2)
            * (1 - z)
            * z
            * m_gamma
            * np.sin(np.pi * x)
            * np.sin(np.pi * y)
            - 4 * z * m_kappa * np.sin(np.pi * x) * np.sin(np.pi * y)
            + 4 * (z**2) * m_kappa * np.sin(np.pi * x) * np.sin(np.pi * y)
        )
        f_rhs = lambda x, y, z: np.array(
            [
                -f_rhs_x(x, y, z),
                -f_rhs_y(x, y, z),
                -f_rhs_z(x, y, z),
                -f_rhs_t_x(x, y, z),
                -f_rhs_t_y(x, y, z),
                -f_rhs_t_z(x, y, z),
            ]
        )

    def scatter_form_data(
        element,
        m_lambda,
        m_mu,
        m_kappa,
        m_gamma,
        f_rhs,
        u_field,
        cell_map,
        row,
        col,
        data,
    ):

        n_components = u_field.n_comp
        el_data: ElementData = element.data

        n_comp_u = 2
        n_comp_t = 1
        if u_field.dimension == 3:
            n_comp_u = 3
            n_comp_t = 3

        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # destination indexes
        dest = u_field.dof_map.destination_indices(cell.id)

        n_phi = phi_tab.shape[2]
        n_dof = n_phi * n_components
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # Partial local vectorization
        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
        phi_s_star = det_jac * weights * phi_tab[0, :, :, 0].T

        # rotation part
        rotation_block = np.zeros((n_phi, n_phi))
        assymetric_block = np.zeros((n_phi * n_comp_t, n_phi * n_comp_u))
        axial_pairs_idx = [[1, 0]]
        axial_dest_pairs_idx = [[0, 1]]
        if u_field.dimension == 3:
            axial_pairs_idx = [[2, 1], [0, 2], [1, 0]]
            axial_dest_pairs_idx = [[1, 2], [2, 0], [0, 1]]

        phi_star_dirs = [[1, 2], [0, 2], [0, 1]]
        for i, omega in enumerate(weights):
            grad_phi = inv_jac[i].T @ phi_tab[1 : phi_tab.shape[0] + 1, i, :, 0]

            phi = phi_tab[0, i, :, 0]
            for i_c, pair in enumerate(axial_pairs_idx):
                adest = axial_dest_pairs_idx[i_c]
                sigma_rotation_0 = np.outer(phi, grad_phi[pair[0], :])
                sigma_rotation_1 = np.outer(phi, grad_phi[pair[1], :])
                assymetric_block[
                    i_c * n_phi : n_phi + i_c * n_phi, adest[0] : n_dof + 1 : n_comp_u
                ] += (det_jac[i] * omega * 2.0 * m_kappa * sigma_rotation_0)
                assymetric_block[
                    i_c * n_phi : n_phi + i_c * n_phi, adest[1] : n_dof + 1 : n_comp_u
                ] -= (det_jac[i] * omega * 2.0 * m_kappa * sigma_rotation_1)
            rotation_block += det_jac[i] * omega * 4.0 * m_kappa * np.outer(phi, phi)
            for d in range(3):
                rotation_block += (
                    det_jac[i] * omega * m_gamma * np.outer(grad_phi[d], grad_phi[d])
                )

            for i_d in range(n_comp_u):
                for j_d in range(n_comp_u):
                    phi_outer = np.outer(grad_phi[i_d], grad_phi[j_d])
                    stress_grad = m_mu * phi_outer
                    if i_d == j_d:
                        phi_outer_star = np.zeros((n_phi, n_phi))
                        for d in phi_star_dirs[i_d]:
                            phi_outer_star += np.outer(grad_phi[d], grad_phi[d])
                        stress_grad += (m_lambda + m_mu) * phi_outer + (
                            m_mu + m_kappa
                        ) * phi_outer_star
                    else:
                        stress_grad -= m_kappa * phi_outer
                        stress_grad += m_lambda * np.outer(grad_phi[j_d], grad_phi[i_d])
                    j_el[
                        i_d : n_dof + 1 : n_components, j_d : n_dof + 1 : n_components
                    ] += (det_jac[i] * omega * stress_grad)

        for c in range(n_comp_t):
            b = c + n_comp_u
            j_el[
                b : n_dof + 1 : n_components, b : n_dof + 1 : n_components
            ] += rotation_block

        for i_c, adest in enumerate(axial_dest_pairs_idx):
            j_el[
                n_comp_u + i_c : n_dof + 1 : n_components,
                adest[0] : n_dof + 1 : n_components,
            ] += assymetric_block[
                i_c * n_phi : n_phi + i_c * n_phi, adest[0] : n_dof + 1 : n_comp_u
            ]
            j_el[
                adest[0] : n_dof + 1 : n_components,
                n_comp_u + i_c : n_dof + 1 : n_components,
            ] += assymetric_block[
                i_c * n_phi : n_phi + i_c * n_phi, adest[0] : n_dof + 1 : n_comp_u
            ].T
            j_el[
                n_comp_u + i_c : n_dof + 1 : n_components,
                adest[1] : n_dof + 1 : n_components,
            ] += assymetric_block[
                i_c * n_phi : n_phi + i_c * n_phi, adest[1] : n_dof + 1 : n_comp_u
            ]
            j_el[
                adest[1] : n_dof + 1 : n_components,
                n_comp_u + i_c : n_dof + 1 : n_components,
            ] += assymetric_block[
                i_c * n_phi : n_phi + i_c * n_phi, adest[1] : n_dof + 1 : n_comp_u
            ].T

        for c in range(n_components):
            b = c
            e = (c + 1) * n_phi * n_components + 1
            r_el[b:e:n_components] += phi_s_star @ f_val_star[c]

        # scattering data
        c_sequ = cell_map[cell.id]

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

    [
        scatter_form_data(
            element,
            m_lambda,
            m_mu,
            m_kappa,
            m_gamma,
            f_rhs,
            u_field,
            cell_map,
            row,
            col,
            data,
        )
        for element in u_field.elements
    ]

    def scatter_bc_form_data(element, u_field, cell_map, row, col, data):

        n_components = u_field.n_comp
        el_data: ElementData = element.data

        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # find high-dimension neigh
        entity_map = u_field.dof_map.mesh_topology.entity_map_by_dimension(
            cell.dimension
        )
        neigh_list = list(entity_map.predecessors(cell.id))
        neigh_check_q = len(neigh_list) > 0
        assert neigh_check_q

        neigh_cell_id = neigh_list[0]
        neigh_cell_index = u_field.id_to_element[neigh_cell_id]
        neigh_cell = u_field.elements[neigh_cell_index].data.cell

        # destination indexes
        dest_neigh = u_field.dof_map.destination_indices(neigh_cell_id)
        dest = u_field.dof_map.bc_destination_indices(neigh_cell_id, cell.id)

        n_phi = phi_tab.shape[2]
        n_dof = n_phi * n_components
        js = (n_dof, n_dof)
        j_el = np.zeros(js)

        # local blocks
        beta = 1.0e12
        jac_block = np.zeros((n_phi, n_phi))
        for i, omega in enumerate(weights):
            phi = phi_tab[0, i, :, 0]
            jac_block += beta * det_jac[i] * omega * np.outer(phi, phi)

        for c in range(n_components):
            b = c
            e = (c + 1) * n_phi * n_components + 1
            j_el[b:e:n_components, b:e:n_components] += jac_block

        # scattering data
        c_sequ = cell_map[cell.id]

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

    [
        scatter_bc_form_data(element, u_field, cell_map, row, col, data)
        for element in u_field.bc_elements
    ]

    jg = coo_matrix((data, (row, col)), shape=(n_dof_g, n_dof_g)).tocsr()
    et = time.time()
    elapsed_time = et - st
    print("Assembly time:", elapsed_time, "seconds")

    # solving ls
    st = time.time()
    # alpha = sp.linalg.spsolve(jg, rg)
    alpha = sp_solver.spsolve(jg, rg)
    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    # Computing L2 error
    def compute_l2_error(element, u_field):
        l2_error = 0.0
        n_components = u_field.n_comp
        el_data = element.data
        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # scattering dof
        dest = u_field.dof_map.destination_indices(cell.id)
        alpha_l = alpha[dest]

        # vectorization
        n_phi = phi_tab.shape[2]
        p_e_s = f_exact(x[:, 0], x[:, 1], x[:, 2])
        alpha_star = np.array(np.split(alpha_l, n_phi))
        p_h_s = (phi_tab[0, :, :, 0] @ alpha_star).T
        l2_error = np.sum(det_jac * weights * (p_e_s - p_h_s) * (p_e_s - p_h_s))
        return l2_error

    st = time.time()
    error_vec = [compute_l2_error(element, u_field) for element in u_field.elements]
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, error_vec))
    print("L2-error: ", l2_error)

    if write_vtk_q:
        # post-process solution
        st = time.time()
        cellid_to_element = dict(zip(u_field.element_ids, u_field.elements))
        # writing solution on mesh points
        vertices = u_field.mesh_topology.entities_by_dimension(0)
        fh_data = np.zeros((len(gmesh.points), n_components))
        fe_data = np.zeros((len(gmesh.points), n_components))
        cell_vertex_map = u_field.mesh_topology.entity_map_by_dimension(0)
        for id in vertices:
            if not cell_vertex_map.has_node(id):
                continue

            pr_ids = list(cell_vertex_map.predecessors(id))
            cell = gmesh.cells[pr_ids[0]]
            if cell.dimension != u_field.dimension:
                continue

            element = cellid_to_element[pr_ids[0]]

            # scattering dof
            dest = u_field.dof_map.destination_indices(cell.id)
            alpha_l = alpha[dest]

            par_points = basix.geometry(u_field.element_type)

            target_node_id = gmesh.cells[id].node_tags[0]
            par_point_id = np.array(
                [
                    i
                    for i, node_id in enumerate(cell.node_tags)
                    if node_id == target_node_id
                ]
            )

            points = gmesh.points[target_node_id]
            if u_field.dimension != 0:
                points = par_points[par_point_id]

            # evaluate mapping
            phi_shapes = evaluate_linear_shapes(points, element.data)
            (x, jac, det_jac, inv_jac) = evaluate_mapping(
                cell.dimension, phi_shapes, gmesh.points[cell.node_tags]
            )
            phi_tab = element.evaluate_basis(points)
            n_phi = phi_tab.shape[2]
            f_e = f_exact(x[:, 0], x[:, 1], x[:, 2])
            alpha_star = np.array(np.split(alpha_l, n_phi))
            f_h = (phi_tab[0, :, :, 0] @ alpha_star).T
            fh_data[target_node_id] = f_h.ravel()
            fe_data[target_node_id] = f_e.ravel()

        mesh_points = gmesh.points
        con_d = np.array([element.data.cell.node_tags for element in u_field.elements])
        meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
        cells_dict = {meshio_cell_types[u_field.dimension]: con_d}
        p_data_dict = {"u_h": fh_data, "u_exact": fe_data}

        mesh = meshio.Mesh(
            points=mesh_points,
            cells=cells_dict,
            # Optionally provide extra data on points, cells, etc.
            point_data=p_data_dict,
        )
        mesh.write("rates_h1_cosserat_elasticity.vtk")
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return l2_error


def create_domain(dimension):

    if dimension == 1:
        box_points = np.array([[0, 0, 0], [1, 0, 0]])
        domain = build_box_1D(box_points)
        return domain
    elif dimension == 2:
        box_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        domain = build_box_2D(box_points)
        return domain
    else:
        box_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
        domain = build_box_3D(box_points)
        return domain


def create_conformal_mesher(domain: Domain, h, ref_l = 0):
    mesher = ConformalMesher(dimension=domain.dimension)
    mesher.domain = domain
    mesher.generate_from_domain(h, ref_l)
    mesher.write_mesh("gmesh.msh")
    return mesher


def create_mesh(dimension, mesher: ConformalMesher, write_vtk_q=False):
    gmesh = Mesh(dimension=dimension, file_name="gmesh.msh")
    gmesh.set_conformal_mesher(mesher)
    gmesh.build_conformal_mesh()
    if write_vtk_q:
        gmesh.write_vtk()
    return gmesh

def main():

    k_order = 3
    h = 1.0
    n_ref = 4
    dimension = 3
    ref_l = 0

    domain = create_domain(dimension)
    error_data = np.empty((0, 2), float)
    for l in range(n_ref):
        h_val = h * (2**-l)
        mesher = create_conformal_mesher(domain, h, l)
        gmesh = create_mesh(dimension, mesher, False)
        # error_val = h1_elasticity(k_order, gmesh, False)
        error_val = h1_cosserat_elasticity(k_order, gmesh, False)
        error_data = np.append(error_data, np.array([[h_val, error_val]]), axis=0)

    rates_data = np.empty((0, 1), float)
    for i in range(error_data.shape[0] - 1):
        chunk_b = np.log(error_data[i])
        chunk_e = np.log(error_data[i + 1])
        h_step = chunk_e[0] - chunk_b[0]
        partial = (chunk_e - chunk_b) / h_step
        rates_data = np.append(rates_data, np.array([list(partial[1:2])]), axis=0)

    print("error data: ", error_data)
    print("error rates data: ", rates_data)

    np.set_printoptions(precision=4)
    print("rounded error data: ", error_data)
    print("rounded error rates data: ", rates_data)

    return


if __name__ == "__main__":
    main()
