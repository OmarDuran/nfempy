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
import auto_diff as ad
from auto_diff.vecvalder import VecValDer

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
from spaces.discrete_space import DiscreteSpace
from spaces.dof_map import DoFMap
from topology.mesh_topology import MeshTopology
from numba import njit, types


def matrix_plot(J, sparse_q=True):

    if sparse_q:
        plot.matshow(J.todense())
    else:
        plot.matshow(J)
    plot.colorbar(orientation="vertical")
    plot.set_cmap("seismic")
    plot.show()


def h1_laplace(k_order, gmesh, write_vtk_q=False):

    dim = gmesh.dimension
    # Material data

    m_kappa = 1.0

    # FESpace: data
    n_components = 1

    discontinuous = True
    u_family = "Lagrange"

    # potential space
    u_space = DiscreteSpace(dim, n_components, u_family, k_order, gmesh)

    if dim == 2:
        u_space.build_structures([2, 3, 4, 5])
    elif dim == 3:
        u_space.build_structures([2, 3, 4, 5, 6, 7])

    st = time.time()
    # Assembler
    # Triplets data
    c_size = 0
    n_dof_g = 0
    cell_map = {}
    for element in u_space.elements:
        cell = element.data.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs) * n_components
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    for element in u_space.bc_elements:
        cell = element.data.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs) * n_components
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    row = np.zeros((c_size), dtype=np.int64)
    col = np.zeros((c_size), dtype=np.int64)
    data = np.zeros((c_size), dtype=np.float64)

    n_dof_g = u_space.dof_map.dof_number()
    rg = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    et = time.time()
    elapsed_time = et - st
    print("Triplets creation time:", elapsed_time, "seconds")

    st = time.time()

    # exact solution
    f_exact = lambda x, y, z: np.array([(1.0 - x) * x * (1.0 - y) * y])
    f_rhs = lambda x, y, z: np.array([2 * (1 - x) * x + 2 * (1 - y) * y])

    if dim == 3:
        f_exact = lambda x, y, z: np.array(
            [(1.0 - x) * x * (1.0 - y) * y * (1.0 - z) * z]
        )

        f_rhs = lambda x, y, z: np.array(
            [
                2 * (1 - x) * x * (1 - y) * y
                + 2 * (1 - x) * x * (1 - z) * z
                + 2 * (1 - y) * y * (1 - z) * z
            ]
        )

    def scatter_form_data(
        element, m_lambda, m_mu, f_rhs, u_space, cell_map, row, col, data
    ):

        n_components = u_space.n_comp
        el_data: ElementData = element.data

        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # destination indexes
        dest = u_space.dof_map.destination_indices(cell.id)

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

    def scatter_form_data_ad(
        element, m_kappa, f_rhs, u_space, cell_map, row, col, data
    ):

        n_components = u_space.n_comp
        el_data: ElementData = element.data

        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # destination indexes
        dest = u_space.dof_map.destination_indices(cell.id)

        n_phi = phi_tab.shape[2]
        n_dof = n_phi * n_components
        alpha = np.zeros(n_dof)

        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # Partial local vectorization
        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
        phi_s_star = det_jac * weights * phi_tab[0, :, :, 0].T
        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        with ad.AutoDiff(alpha) as alpha:

            el_form = np.zeros(n_dof)
            for c in range(n_components):
                b = c
                e = (c + 1) * n_phi * n_components + 1
                el_form[b:e:n_components] += phi_s_star @ f_val_star[c]

            for i, omega in enumerate(weights):
                if dim == 2:
                    inv_jac_m = np.vstack((inv_jac[i] @ e1, inv_jac[i] @ e2))
                else:
                    inv_jac_m = np.vstack(
                        (inv_jac[i] @ e1, inv_jac[i] @ e2, inv_jac[i] @ e3)
                    )
                grad_phi = (inv_jac_m @ phi_tab[1 : phi_tab.shape[0] + 1, i, :, 0]).T
                grad_uh = alpha @ grad_phi
                grad_uh *= m_kappa
                energy_h = (grad_phi @ grad_uh.T).reshape((n_dof,))
                el_form += det_jac[i] * omega * energy_h

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

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
        scatter_form_data_ad(element, m_kappa, f_rhs, u_space, cell_map, row, col, data)
        for element in u_space.elements
    ]

    def scatter_bc_form_data(element, u_space, cell_map, row, col, data):

        n_components = u_space.n_comp
        el_data: ElementData = element.data

        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # find high-dimension neigh
        entity_map = u_space.dof_map.mesh_topology.entity_map_by_dimension(
            cell.dimension
        )
        neigh_list = list(entity_map.predecessors(cell.id))
        neigh_check_q = len(neigh_list) > 0
        assert neigh_check_q

        neigh_cell_id = neigh_list[0]
        neigh_cell_index = u_space.id_to_element[neigh_cell_id]
        neigh_cell = u_space.elements[neigh_cell_index].data.cell

        # destination indexes
        dest_neigh = u_space.dof_map.destination_indices(neigh_cell_id)
        dest = u_space.dof_map.bc_destination_indices(neigh_cell_id, cell.id)

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
        scatter_bc_form_data(element, u_space, cell_map, row, col, data)
        for element in u_space.bc_elements
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
    def compute_l2_error(element, u_space):
        l2_error = 0.0
        n_components = u_space.n_comp
        el_data = element.data
        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # scattering dof
        dest = u_space.dof_map.destination_indices(cell.id)
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
    error_vec = [compute_l2_error(element, u_space) for element in u_space.elements]
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, error_vec))
    print("L2-error: ", l2_error)

    if write_vtk_q:
        # post-process solution
        st = time.time()
        cellid_to_element = dict(zip(u_space.element_ids, u_space.elements))
        # writing solution on mesh points
        vertices = u_space.mesh_topology.entities_by_dimension(0)
        fh_data = np.zeros((len(gmesh.points), n_components))
        fe_data = np.zeros((len(gmesh.points), n_components))
        cell_vertex_map = u_space.mesh_topology.entity_map_by_dimension(0)
        for id in vertices:
            if not cell_vertex_map.has_node(id):
                continue

            pr_ids = list(cell_vertex_map.predecessors(id))
            cell = gmesh.cells[pr_ids[0]]
            if cell.dimension != u_space.dimension:
                continue

            element = cellid_to_element[pr_ids[0]]

            # scattering dof
            dest = u_space.dof_map.destination_indices(cell.id)
            alpha_l = alpha[dest]

            par_points = basix.geometry(u_space.element_type)

            target_node_id = gmesh.cells[id].node_tags[0]
            par_point_id = np.array(
                [
                    i
                    for i, node_id in enumerate(cell.node_tags)
                    if node_id == target_node_id
                ]
            )

            points = gmesh.points[target_node_id]
            if u_space.dimension != 0:
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
        con_d = np.array([element.data.cell.node_tags for element in u_space.elements])
        meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
        cells_dict = {meshio_cell_types[u_space.dimension]: con_d}
        p_data_dict = {"u_h": fh_data, "u_exact": fe_data}

        mesh = meshio.Mesh(
            points=mesh_points,
            cells=cells_dict,
            # Optionally provide extra data on points, cells, etc.
            point_data=p_data_dict,
        )
        mesh.write("rates_h1_laplace.vtk")
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return l2_error


def hdiv_laplace(k_order, gmesh, write_vtk_q=False):

    dim = gmesh.dimension
    # Material data

    m_kappa = 1.0

    # FESpace: data
    q_components = 1
    u_components = 1

    q_family = "BDM"
    u_family = "Lagrange"

    # flux space
    q_space = DiscreteSpace(
        dim, q_components, q_family, k_order, gmesh, integration_oder=2 * k_order + 1
    )
    if dim == 2:
        q_space.build_structures([2, 3, 4, 5])
    elif dim == 3:
        q_space.build_structures([2, 3, 4, 5, 6, 7])

    # potential space
    u_space = DiscreteSpace(
        dim,
        u_components,
        u_family,
        k_order - 1,
        gmesh,
        integration_oder=2 * k_order + 1,
    )
    u_space.make_discontinuous()
    u_space.build_structures()

    st = time.time()
    # Assembler
    # Triplets data
    c_size = 0
    n_dof_g = 0
    cell_map = {}

    q_n_els = len(q_space.elements)
    u_n_els = len(u_space.elements)
    assert q_n_els == u_n_els

    components = (q_components, u_components)
    spaces = (q_space, u_space)

    for i in range(q_n_els):
        q_element = q_space.elements[i]
        u_element = u_space.elements[i]
        cell = q_element.data.cell
        elements = (q_element, u_element)

        n_dof = 0
        for j, element in enumerate(elements):
            for n_entity_dofs in element.basis_generator.num_entity_dofs:
                n_dof = n_dof + sum(n_entity_dofs) * components[j]

        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    for element in q_space.bc_elements:
        cell = element.data.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs) * q_components
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    row = np.zeros((c_size), dtype=np.int64)
    col = np.zeros((c_size), dtype=np.int64)
    data = np.zeros((c_size), dtype=np.float64)

    q_n_dof_g = q_space.dof_map.dof_number()
    u_n_dof_g = u_space.dof_map.dof_number()
    n_dof_g = q_n_dof_g + u_n_dof_g
    rg = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    et = time.time()
    elapsed_time = et - st
    print("Triplets creation time:", elapsed_time, "seconds")

    st = time.time()

    # exact solution
    f_exact = lambda x, y, z: np.array([(1.0 - x) * x * (1.0 - y) * y])
    q_exact = lambda x, y, z: np.array(
        [
            -((1 - x) * (1 - y) * y) + x * (1 - y) * y,
            -((1 - x) * x * (1 - y)) + (1 - x) * x * y,
        ]
    )
    f_rhs = lambda x, y, z: np.array([2.0 * (1.0 - x) * x + 2.0 * (1.0 - y) * y])

    if dim == 3:
        f_exact = lambda x, y, z: np.array(
            [(1.0 - x) * x * (1.0 - y) * y * (1.0 - z) * z]
        )

        f_rhs = lambda x, y, z: np.array(
            [
                2.0 * (1.0 - x) * x * (1.0 - y) * y
                + 2.0 * (1.0 - x) * x * (1.0 - z) * z
                + 2.0 * (1.0 - y) * y * (1.0 - z) * z
            ]
        )

    def scatter_form_data_ad(i, m_kappa, f_rhs, spaces, cell_map, row, col, data):

        dim = spaces[0].dimension
        q_components = spaces[0].n_comp
        u_components = spaces[1].n_comp

        q_data: ElementData = spaces[0].elements[i].data
        u_data: ElementData = spaces[1].elements[i].data

        cell = q_data.cell

        points = q_data.quadrature.points
        weights = q_data.quadrature.weights
        x = q_data.mapping.x
        det_jac = q_data.mapping.det_jac
        inv_jac = q_data.mapping.inv_jac

        # basis
        q_phi_tab = q_data.basis.phi
        u_phi_tab = u_data.basis.phi

        # destination indexes
        dest_q = q_space.dof_map.destination_indices(cell.id)
        dest_u = u_space.dof_map.destination_indices(cell.id) + q_n_dof_g
        dest = np.concatenate([dest_q, dest_u])
        n_q_phi = q_phi_tab.shape[2]
        n_u_phi = u_phi_tab.shape[2]

        n_q_dof = n_q_phi * q_components
        n_u_dof = n_u_phi * u_components

        n_dof = n_q_dof + n_u_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)
        alpha = np.zeros(n_dof)

        # Partial local vectorization
        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
        phi_s_star = det_jac * weights * u_phi_tab[0, :, :, 0].T

        q_val_star = q_exact(x[:, 0], x[:, 1], x[:, 2])
        u_val_star = f_exact(x[:, 0], x[:, 1], x[:, 2])

        # projectors
        q_j_el = np.zeros((n_q_dof, n_q_dof))
        q_r_el = np.zeros(n_q_dof)

        u_j_el = np.zeros((n_u_dof, n_u_dof))
        u_r_el = np.zeros(n_u_dof)

        # # linear_base
        # for i, omega in enumerate(weights):
        #     q_val = q_exact(x[i, 0], x[i, 1], x[i, 2])
        #     q_r_el = q_r_el + det_jac[i] * omega * q_phi_tab[0, i, :, 0:dim] @ q_val
        #     for d in range(3):
        #         q_j_el = q_j_el + det_jac[i] * omega * np.outer(
        #             q_phi_tab[0, i, :, d], q_phi_tab[0, i, :, d]
        #         )
        # alpha_q = np.linalg.solve(q_j_el, q_r_el)
        #
        # aka = 0
        # for i, omega in enumerate(weights):
        #     u_val = f_exact(x[i, 0], x[i, 1], x[i, 2])
        #     u_r_el = u_r_el + det_jac[i] * omega * u_phi_tab[0, i, :, :] @ u_val
        #     for d in range(1):
        #         u_j_el = u_j_el + det_jac[i] * omega * np.outer(
        #             u_phi_tab[0, i, :, d], u_phi_tab[0, i, :, d]
        #         )
        #
        # alpha_u = np.linalg.solve(u_j_el, u_r_el)

        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        with ad.AutoDiff(alpha) as alpha:

            el_form = np.zeros(n_dof)
            for c in range(u_components):
                el_form[n_q_dof:n_dof:1] += -1.0 * phi_s_star @ f_val_star[c]

            for i, omega in enumerate(weights):
                if dim == 2:
                    inv_jac_m = np.vstack((inv_jac[i] @ e1, inv_jac[i] @ e2))
                else:
                    inv_jac_m = np.vstack(
                        (inv_jac[i] @ e1, inv_jac[i] @ e2, inv_jac[i] @ e3)
                    )

                qh = alpha[:, 0:n_q_dof:1] @ q_phi_tab[0, i, :, 0:dim]
                qh *= 1.0 / m_kappa

                uh = alpha[:, n_q_dof:n_dof:1] @ u_phi_tab[0, i, :, 0:dim]

                grad_qh = q_phi_tab[1 : q_phi_tab.shape[0] + 1, i, :, 0:dim]
                div_vh = np.array(
                    [[np.trace(grad_qh[:, j, :]) / det_jac[i] for j in range(n_q_dof)]]
                )
                div_qh = alpha[:, 0:n_q_dof:1] @ div_vh.T

                equ_1_integrand = (qh @ q_phi_tab[0, i, :, 0:dim].T) + (uh @ div_vh)
                equ_2_integrand = div_qh @ u_phi_tab[0, i, :, 0:dim].T
                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, 0:n_q_dof:1] = equ_1_integrand
                multiphysic_integrand[:, n_q_dof:n_dof:1] = equ_2_integrand

                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

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
        scatter_form_data_ad(i, m_kappa, f_rhs, spaces, cell_map, row, col, data)
        for i in range(q_n_els)
    ]

    # def scatter_bc_form_data(element, u_space, cell_map, row, col, data):
    #
    #     n_components = u_space.n_comp
    #     el_data: ElementData = element.data
    #
    #     cell = el_data.cell
    #     points = el_data.quadrature.points
    #     weights = el_data.quadrature.weights
    #     phi_tab = el_data.basis.phi
    #
    #     x = el_data.mapping.x
    #     det_jac = el_data.mapping.det_jac
    #     inv_jac = el_data.mapping.inv_jac
    #
    #     # find high-dimension neigh
    #     entity_map = u_space.dof_map.mesh_topology.entity_map_by_dimension(
    #         cell.dimension
    #     )
    #     neigh_list = list(entity_map.predecessors(cell.id))
    #     neigh_check_q = len(neigh_list) > 0
    #     assert neigh_check_q
    #
    #     neigh_cell_id = neigh_list[0]
    #     neigh_cell_index = u_space.id_to_element[neigh_cell_id]
    #     neigh_cell = u_space.elements[neigh_cell_index].data.cell
    #
    #     # destination indexes
    #     dest_neigh = u_space.dof_map.destination_indices(neigh_cell_id)
    #     dest = u_space.dof_map.bc_destination_indices(neigh_cell_id, cell.id)
    #
    #     n_phi = phi_tab.shape[2]
    #     n_dof = n_phi * n_components
    #     js = (n_dof, n_dof)
    #     j_el = np.zeros(js)
    #
    #     # local blocks
    #     beta = 1.0e12
    #     jac_block = np.zeros((n_phi, n_phi))
    #     for i, omega in enumerate(weights):
    #         phi = phi_tab[0, i, :, 0]
    #         jac_block += beta * det_jac[i] * omega * np.outer(phi, phi)
    #
    #     for c in range(n_components):
    #         b = c
    #         e = (c + 1) * n_phi * n_components + 1
    #         j_el[b:e:n_components, b:e:n_components] += jac_block
    #
    #     # scattering data
    #     c_sequ = cell_map[cell.id]
    #
    #     # contribute lhs
    #     block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
    #     row[block_sequ] += np.repeat(dest, len(dest))
    #     col[block_sequ] += np.tile(dest, len(dest))
    #     data[block_sequ] += j_el.ravel()

    # [
    #     scatter_bc_form_data(element, u_space, cell_map, row, col, data)
    #     for element in u_space.bc_elements
    # ]

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
    def compute_l2_error(element, u_space):
        l2_error = 0.0
        n_components = u_space.n_comp
        el_data = element.data
        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # scattering dof
        dest = u_space.dof_map.destination_indices(cell.id) + q_n_dof_g
        alpha_l = alpha[dest]

        # vectorization
        n_phi = phi_tab.shape[2]
        p_e_s = f_exact(x[:, 0], x[:, 1], x[:, 2])
        alpha_star = np.array(np.split(alpha_l, n_phi))
        p_h_s = (phi_tab[0, :, :, 0] @ alpha_star).T
        l2_error = np.sum(det_jac * weights * (p_e_s - p_h_s) * (p_e_s - p_h_s))
        return l2_error

    st = time.time()
    error_vec = [compute_l2_error(element, u_space) for element in u_space.elements]
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, error_vec))
    print("L2-error: ", l2_error)

    if write_vtk_q:
        # post-process solution
        st = time.time()
        cellid_to_element = dict(zip(u_space.element_ids, u_space.elements))
        # writing solution on mesh points
        vertices = u_space.mesh_topology.entities_by_dimension(0)
        fh_data = np.zeros((len(gmesh.points), u_components))
        fe_data = np.zeros((len(gmesh.points), u_components))
        cell_vertex_map = u_space.mesh_topology.entity_map_by_dimension(0)
        for id in vertices:
            if not cell_vertex_map.has_node(id):
                continue

            pr_ids = list(cell_vertex_map.predecessors(id))
            cell = gmesh.cells[pr_ids[0]]
            if cell.dimension != u_space.dimension:
                continue

            element = cellid_to_element[pr_ids[0]]

            # scattering dof
            dest = u_space.dof_map.destination_indices(cell.id) + q_n_dof_g
            alpha_l = alpha[dest]

            par_points = basix.geometry(u_space.element_type)

            target_node_id = gmesh.cells[id].node_tags[0]
            par_point_id = np.array(
                [
                    i
                    for i, node_id in enumerate(cell.node_tags)
                    if node_id == target_node_id
                ]
            )

            points = gmesh.points[target_node_id]
            if u_space.dimension != 0:
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
        con_d = np.array([element.data.cell.node_tags for element in u_space.elements])
        meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
        cells_dict = {meshio_cell_types[u_space.dimension]: con_d}
        p_data_dict = {"u_h": fh_data, "u_exact": fe_data}

        mesh = meshio.Mesh(
            points=mesh_points,
            cells=cells_dict,
            # Optionally provide extra data on points, cells, etc.
            point_data=p_data_dict,
        )
        mesh.write("rates_hdiv_laplace.vtk")
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


def create_conformal_mesher(domain: Domain, h, ref_l=0):
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

    k_order = 2
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
        # error_val = h1_laplace(k_order, gmesh, True)
        error_val = hdiv_laplace(k_order, gmesh, True)
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
