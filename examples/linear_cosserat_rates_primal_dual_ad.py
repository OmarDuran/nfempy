import copy
import csv
import functools
import marshal
import sys
import time
from petsc4py import PETSc
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

# from numba import njit, types
import strong_solution_cosserat_elasticity as lce

import psutil
num_cpus = psutil.cpu_count(logical=False)
# import ray
# ray.init(num_cpus=num_cpus)

# from pydiso.mkl_solver import (
#     MKLPardisoSolver as Solver,
#     get_mkl_max_threads,
#     get_mkl_pardiso_max_threads,
#     get_mkl_version,
#     set_mkl_threads,
#     set_mkl_pardiso_threads,
# )


# from matplotlib import pyplot as plt

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

    u_space = DiscreteSpace(dim, n_components, family, k_order, gmesh)
    # u_space.build_structures([2, 3])
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
    u_exact = lce.generalized_displacement(m_lambda, m_mu, m_kappa, m_gamma, dim)
    s_exact = lce.stress(m_lambda, m_mu, m_kappa, m_gamma, dim)
    m_exact = lce.couple_stress(m_lambda, m_mu, m_kappa, m_gamma, dim)
    f_rhs = lce.rhs(m_lambda, m_mu, m_kappa, m_gamma, dim)

    def scatter_form_data(
        element,
        m_lambda,
        m_mu,
        m_kappa,
        m_gamma,
        f_rhs,
        u_space,
        cell_map,
        row,
        col,
        data,
    ):
        n_components = u_space.n_comp
        el_data: ElementData = element.data

        n_comp_u = 2
        n_comp_t = 1
        if u_space.dimension == 3:
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

        # rotation part
        rotation_block = np.zeros((n_phi, n_phi))
        assymetric_block = np.zeros((n_phi * n_comp_t, n_phi * n_comp_u))
        axial_pairs_idx = [[1, 0]]
        axial_dest_pairs_idx = [[0, 1]]
        if u_space.dimension == 3:
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
                ] -= (det_jac[i] * omega * 2.0 * m_kappa * sigma_rotation_0)
                assymetric_block[
                    i_c * n_phi : n_phi + i_c * n_phi, adest[1] : n_dof + 1 : n_comp_u
                ] += (det_jac[i] * omega * 2.0 * m_kappa * sigma_rotation_1)
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
            u_space,
            cell_map,
            row,
            col,
            data,
        )
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
    # alpha = sp.linalg.spsolve(jg, -rg)
    alpha = sp_solver.spsolve(jg, -rg)
    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    # Computing displacement L2 error
    def compute_u_l2_error(element, u_space, dim):
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
        p_e_s = u_exact(x[:, 0], x[:, 1], x[:, 2])[0:dim, :]
        alpha_star = np.array(np.split(alpha_l, n_phi))
        p_h_s = (phi_tab[0, :, :, 0] @ alpha_star[:, 0:dim]).T
        l2_error = np.sum(det_jac * weights * (p_e_s - p_h_s) * (p_e_s - p_h_s))
        return l2_error

    # Computing rotation L2 error
    def compute_t_l2_error(element, u_space, dim):
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

        # for each integration point
        n_phi = phi_tab.shape[2]
        alpha_star = np.array(np.split(alpha_l, n_phi))
        t_e_s = u_exact(x[:, 0], x[:, 1], x[:, 2])[dim:n_components, :]
        t_h_s = (phi_tab[0, :, :, 0] @ alpha_star[:, dim:n_components]).T
        for i, omega in enumerate(weights):
            if dim == 2:
                t_e = np.array([[0.0, -t_e_s[0, i]], [t_e_s[0, i], 0.0]])
                t_h = np.array([[0.0, -t_h_s[0, i]], [t_h_s[0, i], 0.0]])
            else:
                t_e = np.array(
                    [
                        [0.0, -t_e_s[2, i], +t_e_s[1, i]],
                        [+t_e_s[2, i], 0.0, -t_e_s[0, i]],
                        [-t_e_s[1, i], +t_e_s[0, i], 0.0],
                    ]
                )
                t_h = np.array(
                    [
                        [0.0, -t_h_s[2, i], +t_h_s[1, i]],
                        [+t_h_s[2, i], 0.0, -t_h_s[0, i]],
                        [-t_h_s[1, i], +t_h_s[0, i], 0.0],
                    ]
                )
            diff_s = t_e - t_h
            l2_error += det_jac[i] * weights[i] * np.trace(diff_s.T @ diff_s)
        return l2_error

    # Computing stress L2 error
    def compute_s_l2_error(element, u_space, m_mu, m_lambda, m_kappa, dim):
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

        # for each integration point
        n_phi = phi_tab.shape[2]
        alpha_star = np.array(np.split(alpha_l, n_phi))
        t_h_s = (phi_tab[0, :, :, 0] @ alpha_star[:, dim:n_components]).T
        for i, omega in enumerate(weights):
            s_e = s_exact(x[i, 0], x[i, 1], x[i, 2])
            grad_phi = inv_jac[i].T @ phi_tab[1 : phi_tab.shape[0] + 1, i, :, 0]
            grad_uh = grad_phi[0:dim] @ alpha_star[:, 0:dim]
            if dim == 2:
                t_h = np.array([[0.0, -t_h_s[0, i]], [t_h_s[0, i], 0.0]])
            else:
                t_h = np.array(
                    [
                        [0.0, -t_h_s[2, i], +t_h_s[1, i]],
                        [+t_h_s[2, i], 0.0, -t_h_s[0, i]],
                        [-t_h_s[1, i], +t_h_s[0, i], 0.0],
                    ]
                )
            eps_h = grad_uh.T + t_h
            symm_eps = 0.5 * (eps_h + eps_h.T)
            skew_eps = 0.5 * (eps_h - eps_h.T)
            s_h = (
                2.0 * m_mu * symm_eps
                + 2.0 * m_kappa * skew_eps
                + m_lambda * symm_eps.trace() * np.identity(dim)
            )
            diff_s = s_e - s_h
            l2_error += det_jac[i] * weights[i] * np.trace(diff_s.T @ diff_s)
        return l2_error

    # Computing couple stress L2 error
    def compute_m_l2_error(element, u_space, m_gamma, dim):
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

        # for each integration point
        n_phi = phi_tab.shape[2]
        alpha_star = np.array(np.split(alpha_l, n_phi))
        for i, omega in enumerate(weights):
            m_e = m_exact(x[i, 0], x[i, 1], x[i, 2])
            grad_phi = inv_jac[i].T @ phi_tab[1 : phi_tab.shape[0] + 1, i, :, 0]
            grad_th = grad_phi[0:dim] @ alpha_star[:, dim:n_components]
            if dim == 2:
                m_h = m_gamma * grad_th
            else:
                m_h = m_gamma * grad_th.T

            diff_m = m_e - m_h
            if dim == 2:
                l2_error += det_jac[i] * weights[i] * (diff_m.T @ diff_m)[0, 0]
            else:
                l2_error += det_jac[i] * weights[i] * np.trace(diff_m.T @ diff_m)
        return l2_error

    st = time.time()
    u_error_vec = [
        compute_u_l2_error(element, u_space, dim) for element in u_space.elements
    ]
    t_error_vec = [
        compute_t_l2_error(element, u_space, dim) for element in u_space.elements
    ]
    s_error_vec = [
        compute_s_l2_error(element, u_space, m_mu, m_lambda, m_kappa, dim)
        for element in u_space.elements
    ]
    m_error_vec = [
        compute_m_l2_error(element, u_space, m_gamma, dim)
        for element in u_space.elements
    ]
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    u_l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, u_error_vec))
    t_l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, t_error_vec))
    s_l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, s_error_vec))
    m_l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, m_error_vec))
    print("L2-error displacement: ", u_l2_error)
    print("L2-error rotation: ", u_l2_error)
    print("L2-error stress: ", s_l2_error)
    print("L2-error couple stress: ", m_l2_error)

    if write_vtk_q:
        # post-process solution
        st = time.time()

        # writing solution on mesh points
        uh_data = np.zeros((len(gmesh.points), dim))
        ue_data = np.zeros((len(gmesh.points), dim))
        th_data = np.zeros((len(gmesh.points), n_components - dim))
        te_data = np.zeros((len(gmesh.points), n_components - dim))
        sh_data = np.zeros((len(gmesh.points), dim * dim))
        se_data = np.zeros((len(gmesh.points), dim * dim))
        if dim == 2:
            mh_data = np.zeros((len(gmesh.points), dim))
            me_data = np.zeros((len(gmesh.points), dim))
        else:
            mh_data = np.zeros((len(gmesh.points), dim * dim))
            me_data = np.zeros((len(gmesh.points), dim * dim))

        # generalized displacements
        cellid_to_element = dict(zip(u_space.element_ids, u_space.elements))
        vertices = u_space.mesh_topology.entities_by_dimension(0)
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
            alpha_star = np.array(np.split(alpha_l, n_phi))

            # Generalized displacement
            u_e = u_exact(x[:, 0], x[:, 1], x[:, 2])[0:dim, :]
            t_e = u_exact(x[:, 0], x[:, 1], x[:, 2])[dim:n_components, :]
            u_h = (phi_tab[0, :, :, 0] @ alpha_star[:, 0:dim]).T
            t_h = (phi_tab[0, :, :, 0] @ alpha_star[:, dim:n_components]).T

            # stress and couple stress
            i = 0
            s_e = s_exact(x[i, 0], x[i, 1], x[i, 2])
            grad_phi = inv_jac[i].T @ phi_tab[1 : phi_tab.shape[0] + 1, i, :, 0]
            grad_uh = grad_phi[0:dim] @ alpha_star[:, 0:dim]
            if dim == 2:
                th = np.array([[0.0, -t_h[0, i]], [t_h[0, i], 0.0]])
            else:
                th = np.array(
                    [
                        [0.0, -t_h[2, i], +t_h[1, i]],
                        [+t_h[2, i], 0.0, -t_h[0, i]],
                        [-t_h[1, i], +t_h[0, i], 0.0],
                    ]
                )
            eps_h = grad_uh.T + th
            symm_eps = 0.5 * (eps_h + eps_h.T)
            skew_eps = 0.5 * (eps_h - eps_h.T)
            s_h = (
                2.0 * m_mu * symm_eps
                + 2.0 * m_kappa * skew_eps
                + m_lambda * symm_eps.trace() * np.identity(dim)
            )

            m_e = m_exact(x[i, 0], x[i, 1], x[i, 2])
            grad_phi = inv_jac[i].T @ phi_tab[1 : phi_tab.shape[0] + 1, i, :, 0]
            grad_th = grad_phi[0:dim] @ alpha_star[:, dim:n_components]
            if dim == 2:
                m_h = m_gamma * grad_th
            else:
                m_h = m_gamma * grad_th.T

            uh_data[target_node_id] = u_h.ravel()
            ue_data[target_node_id] = u_e.ravel()
            th_data[target_node_id] = t_h.ravel()
            te_data[target_node_id] = t_e.ravel()

            sh_data[target_node_id] = s_h.ravel()
            se_data[target_node_id] = s_e.ravel()
            mh_data[target_node_id] = m_h.ravel()
            me_data[target_node_id] = m_e.ravel()

        mesh_points = gmesh.points
        con_d = np.array([element.data.cell.node_tags for element in u_space.elements])
        meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
        cells_dict = {meshio_cell_types[u_space.dimension]: con_d}
        p_data_dict = {
            "u_h": uh_data,
            "u_exact": ue_data,
            "t_h": th_data,
            "t_exact": te_data,
            "s_h": sh_data,
            "s_exact": se_data,
            "m_h": mh_data,
            "m_exact": me_data,
        }

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

    return np.array([u_l2_error, t_l2_error, s_l2_error, m_l2_error])


def hdiv_cosserat_elasticity(k_order, gmesh, write_vtk_q=False):
    parallel_assembly_q = False

    dim = gmesh.dimension
    # Material data

    m_lambda = 1.0
    m_mu = 1.0
    m_kappa = 1.0
    m_gamma = 1.0

    # FESpace: data
    s_components = 2
    m_components = 1
    u_components = 2
    t_components = 1
    if dim == 3:
        s_components = 3
        m_components = 3
        u_components = 3
        t_components = 3

    s_family = "BDM"
    m_family = "RT"
    u_family = "Lagrange"
    t_family = "Lagrange"

    # stress space
    s_space = DiscreteSpace(
        dim, s_components, s_family, k_order, gmesh, integration_oder=2 * k_order + 1
    )
    if dim == 2:
        s_space.build_structures([2, 3, 4, 5])
    elif dim == 3:
        s_space.build_structures([2, 3, 4, 5, 6, 7])

    # couple stress space
    m_space = DiscreteSpace(
        dim, m_components, m_family, k_order, gmesh, integration_oder=2 * k_order + 1
    )
    if dim == 2:
        m_space.build_structures([2, 3, 4, 5])
    elif dim == 3:
        m_space.build_structures([2, 3, 4, 5, 6, 7])

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

    # rotation space
    t_space = DiscreteSpace(
        dim,
        t_components,
        t_family,
        k_order - 1,
        gmesh,
        integration_oder=2 * k_order + 1,
    )
    t_space.make_discontinuous()
    t_space.build_structures()

    st = time.time()
    # Assembler
    # Triplets data
    c_size = 0
    n_dof_g = 0
    cell_map = {}

    s_n_els = len(s_space.elements)
    m_n_els = len(m_space.elements)
    u_n_els = len(u_space.elements)
    t_n_els = len(t_space.elements)
    assert s_n_els == m_n_els == u_n_els == t_n_els

    components = (s_components, m_components, u_components, t_components)
    spaces = (s_space, m_space, u_space, t_space)

    # for i in range(s_n_els):
    #     s_element = s_space.elements[i]
    #     m_element = m_space.elements[i]
    #     u_element = u_space.elements[i]
    #     t_element = t_space.elements[i]
    #     cell = s_element.data.cell
    #     elements = (s_element, m_element, u_element, t_element)
    #
    #     n_dof = 0
    #     for j, element in enumerate(elements):
    #         for n_entity_dofs in element.basis_generator.num_entity_dofs:
    #             n_dof = n_dof + sum(n_entity_dofs) * components[j]
    #
    #     cell_map.__setitem__(cell.id, c_size)
    #     c_size = c_size + n_dof * n_dof
    #
    # for element in s_space.bc_elements:
    #     cell = element.data.cell
    #     n_dof = 0
    #     for n_entity_dofs in element.basis_generator.num_entity_dofs:
    #         n_dof = n_dof + sum(n_entity_dofs) * s_components
    #     cell_map.__setitem__(cell.id, c_size)
    #     c_size = c_size + n_dof * n_dof

    # row = np.zeros((c_size), dtype=np.int32)
    # col = np.zeros((c_size), dtype=np.int32)
    # data = np.zeros((c_size), dtype=np.float64)

    s_n_dof_g = s_space.dof_map.dof_number()
    m_n_dof_g = m_space.dof_map.dof_number()
    u_n_dof_g = u_space.dof_map.dof_number()
    t_n_dof_g = t_space.dof_map.dof_number()
    n_dof_g = s_n_dof_g + m_n_dof_g + u_n_dof_g + t_n_dof_g
    rg = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    et = time.time()
    elapsed_time = et - st
    print("Triplets creation time:", elapsed_time, "seconds")

    st = time.time()

    A = PETSc.Mat()
    A.createAIJ([n_dof_g, n_dof_g])

    # exact solution
    u_exact = lce.generalized_displacement(m_lambda, m_mu, m_kappa, m_gamma, dim)
    s_exact = lce.stress(m_lambda, m_mu, m_kappa, m_gamma, dim)
    m_exact = lce.couple_stress(m_lambda, m_mu, m_kappa, m_gamma, dim)
    f_rhs = lce.rhs(m_lambda, m_mu, m_kappa, m_gamma, dim)

    def scatter_form_data_ad(i, args):
        (
            A,
            dim,
            components,
            element_data,
            destinations,
            m_lambda,
            m_mu,
            m_kappa,
            m_gamma,
            f_rhs
        ) = args

        s_components, m_components, u_components, t_components = components
        s_data: ElementData = element_data[i][0]
        m_data: ElementData = element_data[i][1]
        u_data: ElementData = element_data[i][2]
        t_data: ElementData = element_data[i][3]

        cell = s_data.cell

        points = s_data.quadrature.points
        weights = s_data.quadrature.weights
        x = s_data.mapping.x
        det_jac = s_data.mapping.det_jac
        inv_jac = s_data.mapping.inv_jac

        # basis
        s_phi_tab = s_data.basis.phi
        m_phi_tab = m_data.basis.phi
        u_phi_tab = u_data.basis.phi
        t_phi_tab = t_data.basis.phi

        # destination indexes
        dest_s = destinations[0][i]
        dest_m = destinations[1][i]
        dest_u = destinations[2][i]
        dest_t = destinations[3][i]

        dest = np.concatenate([dest_s, dest_m, dest_u, dest_t])
        n_s_phi = s_phi_tab.shape[2]
        n_m_phi = m_phi_tab.shape[2]
        n_u_phi = u_phi_tab.shape[2]
        n_t_phi = t_phi_tab.shape[2]

        n_s_dof = n_s_phi * s_components
        n_m_dof = n_m_phi * m_components
        n_u_dof = n_u_phi * u_components
        n_t_dof = n_t_phi * t_components

        n_dof = n_s_dof + n_m_dof + n_u_dof + n_t_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)
        alpha = np.zeros(n_dof)

        # Partial local vectorization
        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
        phi_s_star = det_jac * weights * u_phi_tab[0, :, :, 0].T

        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        Imat = np.identity(dim)
        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)
            for c in range(u_components):
                b = c + n_s_dof + n_m_dof
                e = b + n_u_dof
                el_form[b:e:u_components] += -1.0 * phi_s_star @ f_val_star[c]
            for c in range(t_components):
                b = c + n_s_dof + n_m_dof + n_u_dof
                e = b + n_t_dof
                el_form[b:e:t_components] += (
                    -1.0 * phi_s_star @ f_val_star[c + u_components]
                )

            for i, omega in enumerate(weights):
                if dim == 2:
                    c = 0
                    a_sx = alpha[:, c : n_s_dof + c : s_components]
                    a_ux = alpha[
                        :,
                        n_s_dof
                        + n_m_dof
                        + c : n_s_dof
                        + n_m_dof
                        + n_u_dof
                        + c : u_components,
                    ]

                    a_m = alpha[:, n_s_dof + c : n_s_dof + n_m_dof + c : m_components]
                    a_t = alpha[
                        :,
                        n_s_dof
                        + n_m_dof
                        + n_u_dof
                        + c : n_s_dof
                        + n_m_dof
                        + n_u_dof
                        + n_t_dof
                        + c : t_components,
                    ]

                    c = 1
                    a_sy = alpha[:, c : n_s_dof + c : s_components]
                    a_uy = alpha[
                        :,
                        n_s_dof
                        + n_m_dof
                        + c : n_s_dof
                        + n_m_dof
                        + n_u_dof
                        + c : u_components,
                    ]

                    sx_h = a_sx @ s_phi_tab[0, i, :, 0:dim]
                    sy_h = a_sy @ s_phi_tab[0, i, :, 0:dim]
                    ux_h = a_ux @ u_phi_tab[0, i, :, 0:dim]
                    uy_h = a_uy @ u_phi_tab[0, i, :, 0:dim]

                    mh = a_m @ m_phi_tab[0, i, :, 0:dim]
                    th = a_t @ t_phi_tab[0, i, :, 0:dim]

                    uh = VecValDer(
                        np.hstack((ux_h.val, uy_h.val)), np.hstack((ux_h.der, uy_h.der))
                    )

                    sh = VecValDer(
                        np.vstack((sx_h.val, sy_h.val)), np.vstack((sx_h.der, sy_h.der))
                    )

                    # Stress decomposition
                    Symm_sh = 0.5 * (sh + sh.T)
                    Skew_sh = 0.5 * (sh - sh.T)

                    tr_s_h = VecValDer(sh.val.trace(), sh.der.trace())
                    A_sh = (1.0 / 2.0 * m_mu) * (
                        Symm_sh - (m_lambda / (2.0 * m_mu + dim * m_lambda)) * tr_s_h * Imat
                    ) + (1.0 / 2.0 * m_kappa) * Skew_sh


                    A_mh = (1.0 / m_gamma) * mh

                    grad_s_phi = s_phi_tab[1 : s_phi_tab.shape[0] + 1, i, :, 0:dim]
                    div_tau = np.array(
                        [
                            [
                                np.trace(grad_s_phi[:, j, :]) / det_jac[i]
                                for j in range(n_s_phi)
                            ]
                        ]
                    )

                    grad_m_phi = m_phi_tab[1 : s_phi_tab.shape[0] + 1, i, :, 0:dim]
                    div_v = np.array(
                        [
                            [
                                np.trace(grad_m_phi[:, j, :]) / det_jac[i]
                                for j in range(n_m_phi)
                            ]
                        ]
                    )

                    div_sh_x = a_sx @ div_tau.T
                    div_sh_y = a_sy @ div_tau.T
                    div_sh = VecValDer(
                        np.hstack((div_sh_x.val, div_sh_y.val)),
                        np.hstack((div_sh_x.der, div_sh_y.der)),
                    )

                    div_mh = a_m @ div_v.T

                    Gamma_outer = th * np.array([[0.0, -1.0], [1.0, 0.0]])
                    S_cross = np.array([[Skew_sh[1, 0] - Skew_sh[0, 1]]])

                else:
                    c = 0
                    a_sx = alpha[:, c : n_s_dof + c : s_components]
                    a_ux = alpha[
                        :,
                        n_s_dof
                        + n_m_dof
                        + c : n_s_dof
                        + n_m_dof
                        + n_u_dof
                        + c : u_components,
                    ]
                    a_mx = alpha[:, n_s_dof + c : n_s_dof + n_m_dof + c : m_components]
                    a_tx = alpha[
                        :,
                        n_s_dof
                        + n_m_dof
                        + n_u_dof
                        + c : n_s_dof
                        + n_m_dof
                        + n_u_dof
                        + n_t_dof
                        + c : t_components,
                    ]

                    c = 1
                    a_sy = alpha[:, c : n_s_dof + c : s_components]
                    a_uy = alpha[
                        :,
                        n_s_dof
                        + n_m_dof
                        + c : n_s_dof
                        + n_m_dof
                        + n_u_dof
                        + c : u_components,
                    ]
                    a_my = alpha[:, n_s_dof + c : n_s_dof + n_m_dof + c : m_components]
                    a_ty = alpha[
                        :,
                        n_s_dof
                        + n_m_dof
                        + n_u_dof
                        + c : n_s_dof
                        + n_m_dof
                        + n_u_dof
                        + n_t_dof
                        + c : t_components,
                    ]

                    c = 2
                    a_sz = alpha[:, c : n_s_dof + c : s_components]
                    a_uz = alpha[
                        :,
                        n_s_dof
                        + n_m_dof
                        + c : n_s_dof
                        + n_m_dof
                        + n_u_dof
                        + c : u_components,
                    ]
                    a_mz = alpha[:, n_s_dof + c : n_s_dof + n_m_dof + c : m_components]
                    a_tz = alpha[
                        :,
                        n_s_dof
                        + n_m_dof
                        + n_u_dof
                        + c : n_s_dof
                        + n_m_dof
                        + n_u_dof
                        + n_t_dof
                        + c : t_components,
                    ]

                    sx_h = a_sx @ s_phi_tab[0, i, :, 0:dim]
                    sy_h = a_sy @ s_phi_tab[0, i, :, 0:dim]
                    sz_h = a_sz @ s_phi_tab[0, i, :, 0:dim]

                    mx_h = a_mx @ m_phi_tab[0, i, :, 0:dim]
                    my_h = a_my @ m_phi_tab[0, i, :, 0:dim]
                    mz_h = a_mz @ m_phi_tab[0, i, :, 0:dim]

                    ux_h = a_ux @ u_phi_tab[0, i, :, 0:dim]
                    uy_h = a_uy @ u_phi_tab[0, i, :, 0:dim]
                    uz_h = a_uz @ u_phi_tab[0, i, :, 0:dim]

                    tx_h = a_tx @ t_phi_tab[0, i, :, 0:dim]
                    ty_h = a_ty @ t_phi_tab[0, i, :, 0:dim]
                    tz_h = a_tz @ t_phi_tab[0, i, :, 0:dim]

                    uh = VecValDer(
                        np.hstack((ux_h.val, uy_h.val, uz_h.val)),
                        np.hstack((ux_h.der, uy_h.der, uz_h.der)),
                    )

                    th = VecValDer(
                        np.hstack((tx_h.val, ty_h.val, tz_h.val)),
                        np.hstack((tx_h.der, ty_h.der, tz_h.der)),
                    )

                    sh = VecValDer(
                        np.vstack((sx_h.val, sy_h.val, sz_h.val)),
                        np.vstack((sx_h.der, sy_h.der, sz_h.der)),
                    )

                    mh = VecValDer(
                        np.vstack((mx_h.val, my_h.val, mz_h.val)),
                        np.vstack((mx_h.der, my_h.der, mz_h.der)),
                    )

                    # Stress decomposition
                    Symm_sh = 0.5 * (sh + sh.T)
                    Skew_sh = 0.5 * (sh - sh.T)

                    tr_s_h = VecValDer(sh.val.trace(), sh.der.trace())
                    A_sh = (1.0 / 2.0 * m_mu) * (
                        Symm_sh - (m_lambda / (2.0 * m_mu + dim * m_lambda)) * tr_s_h * Imat
                    ) + (1.0 / 2.0 * m_kappa) * Skew_sh

                    A_mh = (1.0 / m_gamma) * mh

                    grad_s_phi = s_phi_tab[1 : s_phi_tab.shape[0] + 1, i, :, 0:dim]
                    div_tau = np.array(
                        [
                            [
                                np.trace(grad_s_phi[:, j, :]) / det_jac[i]
                                for j in range(n_s_phi)
                            ]
                        ]
                    )

                    div_sh_x = a_sx @ div_tau.T
                    div_sh_y = a_sy @ div_tau.T
                    div_sh_z = a_sz @ div_tau.T

                    div_sh = VecValDer(
                        np.hstack((div_sh_x.val, div_sh_y.val, div_sh_z.val)),
                        np.hstack((div_sh_x.der, div_sh_y.der, div_sh_z.der)),
                    )

                    grad_m_phi = m_phi_tab[1 : m_phi_tab.shape[0] + 1, i, :, 0:dim]
                    div_v = np.array(
                        [
                            [
                                np.trace(grad_m_phi[:, j, :]) / det_jac[i]
                                for j in range(n_m_phi)
                            ]
                        ]
                    )

                    div_mh_x = a_mx @ div_v.T
                    div_mh_y = a_my @ div_v.T
                    div_mh_z = a_mz @ div_v.T

                    div_mh = VecValDer(
                        np.hstack((div_mh_x.val, div_mh_y.val, div_mh_z.val)),
                        np.hstack((div_mh_x.der, div_mh_y.der, div_mh_z.der)),
                    )

                    Gamma_outer = np.array(
                        [
                            [0.0 * th[0, 0], -th[0, 2], +th[0, 1]],
                            [+th[0, 2], 0.0 * th[0, 0], -th[0, 0]],
                            [-th[0, 1], +th[0, 0], 0.0 * th[0, 0]],
                        ]
                    )

                    S_cross = np.array(
                        [
                            [
                                Skew_sh[2, 1] - Skew_sh[1, 2],
                                Skew_sh[0, 2] - Skew_sh[2, 0],
                                Skew_sh[1, 0] - Skew_sh[0, 1],
                            ]
                        ]
                    )

                equ_1_integrand = (
                    (s_phi_tab[0, i, :, 0:dim] @ A_sh.T)
                    + (div_tau.T @ uh)
                    + (s_phi_tab[0, i, :, 0:dim] @ Gamma_outer)
                )
                equ_2_integrand = (m_phi_tab[0, i, :, 0:dim] @ A_mh.T) + (div_v.T @ th)
                equ_3_integrand = u_phi_tab[0, i, :, 0:dim] @ div_sh
                equ_4_integrand = (t_phi_tab[0, i, :, 0:dim] @ div_mh) - (
                    t_phi_tab[0, i, :, 0:dim] @ S_cross
                )

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, 0:n_s_dof:1] = (equ_1_integrand).reshape(
                    (n_s_dof,)
                )
                multiphysic_integrand[:, n_s_dof : n_s_dof + n_m_dof : 1] = (
                    equ_2_integrand
                ).reshape((n_m_dof,))
                multiphysic_integrand[
                    :, n_s_dof + n_m_dof : n_s_dof + n_m_dof + n_u_dof : 1
                ] = (equ_3_integrand).reshape((n_u_dof,))
                multiphysic_integrand[
                    :,
                    n_s_dof
                    + n_m_dof
                    + n_u_dof : n_s_dof
                    + n_m_dof
                    + n_u_dof
                    + n_t_dof : 1,
                ] = (equ_4_integrand).reshape((n_t_dof,))

                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        # return (r_el, j_el)
        # scattering data
        # c_sequ = cell_map[cell.id]

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        # block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
        # row[block_sequ] += np.repeat(dest, len(dest))
        # col[block_sequ] += np.tile(dest, len(dest))
        # data[block_sequ] += j_el.ravel()

        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        for k in range(nnz):
            A.setValue(row = row[k], col = col[k], value = data[k], addv = True )


    # collect destination indexes
    dest_s = [
        s_space.dof_map.destination_indices(spaces[0].elements[i].data.cell.id)
        for i in range(s_n_els)
    ]
    dest_m = [
        m_space.dof_map.destination_indices(spaces[1].elements[i].data.cell.id)
        + s_n_dof_g
        for i in range(m_n_els)
    ]
    dest_u = [
        u_space.dof_map.destination_indices(spaces[2].elements[i].data.cell.id)
        + s_n_dof_g
        + m_n_dof_g
        for i in range(u_n_els)
    ]
    dest_t = [
        t_space.dof_map.destination_indices(spaces[3].elements[i].data.cell.id)
        + s_n_dof_g
        + m_n_dof_g
        + u_n_dof_g
        for i in range(t_n_els)
    ]

    destinations = (dest_s, dest_m, dest_u, dest_t)
    # collect data
    element_data = [
        (
            spaces[0].elements[i].data,
            spaces[1].elements[i].data,
            spaces[2].elements[i].data,
            spaces[3].elements[i].data,
        )
        for i in range(s_n_els)
    ]
    args = (
        A,
        dim,
        components,
        element_data,
        destinations,
        m_lambda,
        m_mu,
        m_kappa,
        m_gamma,
        f_rhs
    )

    indexes = np.array([i for i in range(s_n_els)])
    collection = np.array_split(indexes, num_cpus)

    if parallel_assembly_q:

        @ray.remote
        def scatter_form_data_on_cells(indexes, args):
            return [scatter_form_data_ad(i, args) for i in indexes]

        streaming_actors = [
            scatter_form_data_on_cells.remote(index_set, args)
            for index_set in collection
        ]
        results = ray.get(streaming_actors)
    else:

        def scatter_form_data_on_cells(indexes, args):
            return [scatter_form_data_ad(i, args) for i in indexes]

        results = [
            scatter_form_data_on_cells(index_set, args) for index_set in collection
        ]

    # array_data = [pair for chunk in results for pair in chunk]
    # args = (array_data, element_data, destinations, cell_map, row, col, data)

    # def scatter_residual_jacobian(i, args):
    #     array_data, element_data, destinations, cell_map, row, col, data = args
    #     s_data: ElementData = element_data[i][0]
    #     c_sequ = cell_map[s_data.cell.id]
    #
    #     # destination indexes
    #     dest_s = destinations[0][i]
    #     dest_m = destinations[1][i]
    #     dest_u = destinations[2][i]
    #     dest_t = destinations[3][i]
    #
    #     dest = np.concatenate([dest_s, dest_m, dest_u, dest_t])
    #     r_el, j_el = array_data[i]
    #     # contribute rhs
    #     rg[dest] += r_el
    #
    #     # contribute lhs
    #     block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
    #     row[block_sequ] += np.repeat(dest, len(dest))
    #     col[block_sequ] += np.tile(dest, len(dest))
    #     data[block_sequ] += j_el.ravel()
    #
    # [scatter_residual_jacobian(i, args) for i in indexes]

    # jg = coo_matrix((data, (row, col)), shape=(n_dof_g, n_dof_g)).tocsr()
    # array_data.clear()
    et = time.time()
    elapsed_time = et - st
    print("Assembly time:", elapsed_time, "seconds")

    # solving ls
    st = time.time()
    # alpha = sp.linalg.spsolve(jg, -rg)
    # alpha = sp_solver.spsolve(jg, -rg)

    # jg_iLU = sp.linalg.spilu(jg.tocsc(),drop_tol=1e-2)
    # P = sp.linalg.LinearOperator(jg.shape, jg_iLU.solve)
    # alpha, check = sp.linalg.gmres(jg, -rg, M=P,tol=1e-10)
    # print("successful exit:", check)

    # A = PETSc.Mat()
    # A.createAIJ([n_dof_g, n_dof_g])

    # nnz = data.shape[0]
    # for k in range(nnz):
    #     A.setValue(row = row[k], col = col[k], value = data[k], addv = True )
    A.assemble()

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    b = A.createVecLeft()
    b.array[:] = -rg
    x = A.createVecRight()

    ksp = PETSc.KSP().create()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType('fgmres')
    ksp.setConvergenceHistory()
    ksp.getPC().setType('ilu')
    ksp.solve(b, x)
    alpha = x.array

    # residuals = ksp.getConvergenceHistory()
    # plt.semilogy(residuals)

    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    # Computing displacement L2 error
    def compute_u_l2_error(i, spaces, dim):
        l2_error = 0.0

        u_data: ElementData = spaces[2].elements[i].data
        t_data: ElementData = spaces[3].elements[i].data

        u_components = u_space.n_comp
        t_components = t_space.n_comp

        cell = u_data.cell
        points = u_data.quadrature.points
        weights = u_data.quadrature.weights
        u_phi_tab = u_data.basis.phi
        t_phi_tab = t_data.basis.phi

        x = u_data.mapping.x
        det_jac = u_data.mapping.det_jac
        inv_jac = u_data.mapping.inv_jac

        # scattering dof
        dest_u = u_space.dof_map.destination_indices(cell.id) + s_n_dof_g + m_n_dof_g
        u_alpha_l = alpha[dest_u]

        # vectorization
        u_n_phi = u_phi_tab.shape[2]
        t_n_phi = t_phi_tab.shape[2]
        u_e_s = u_exact(x[:, 0], x[:, 1], x[:, 2])[0:dim, :]
        u_alpha_star = np.array(np.split(u_alpha_l, u_n_phi))
        u_h_s = (u_phi_tab[0, :, :, 0] @ u_alpha_star).T
        diff_p = u_e_s - u_h_s
        l2_error = np.sum(det_jac * weights * diff_p * diff_p)
        return l2_error

    # Computing rotation L2 error
    def compute_t_l2_error(i, spaces, dim):
        l2_error = 0.0

        u_data: ElementData = spaces[2].elements[i].data
        t_data: ElementData = spaces[3].elements[i].data

        u_components = u_space.n_comp
        t_components = t_space.n_comp

        cell = u_data.cell
        points = u_data.quadrature.points
        weights = u_data.quadrature.weights
        u_phi_tab = u_data.basis.phi
        t_phi_tab = t_data.basis.phi

        x = u_data.mapping.x
        det_jac = u_data.mapping.det_jac
        inv_jac = u_data.mapping.inv_jac

        # scattering dof
        dest_t = (
            t_space.dof_map.destination_indices(cell.id)
            + s_n_dof_g
            + m_n_dof_g
            + u_n_dof_g
        )
        t_alpha_l = alpha[dest_t]

        # vectorization
        t_n_phi = t_phi_tab.shape[2]
        t_e_s = u_exact(x[:, 0], x[:, 1], x[:, 2])[dim : u_components + t_components, :]
        t_alpha_star = np.array(np.split(t_alpha_l, t_n_phi))
        t_h_s = (t_phi_tab[0, :, :, 0] @ t_alpha_star).T
        diff_p = t_e_s - t_h_s
        l2_error = np.sum(det_jac * weights * diff_p * diff_p)
        return l2_error

    # Computing stress L2 error
    def compute_s_l2_error(i, spaces, dim):
        l2_error = 0.0
        n_components = s_space.n_comp
        el_data: ElementData = spaces[0].elements[i].data
        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # scattering dof
        dest = s_space.dof_map.destination_indices(cell.id)
        alpha_l = alpha[dest]

        # vectorization
        n_phi = phi_tab.shape[2]

        alpha_star = np.array(np.split(alpha_l, n_phi))
        for i, omega in enumerate(weights):
            s_e = s_exact(x[i, 0], x[i, 1], x[i, 2])
            s_h = np.vstack(
                tuple(
                    [phi_tab[0, i, :, 0:dim].T @ alpha_star[:, d] for d in range(dim)]
                )
            )
            diff_s = s_e - s_h
            l2_error += det_jac[i] * weights[i] * np.trace(diff_s.T @ diff_s)
        return l2_error

    # Computing couple stress L2 error
    def compute_m_l2_error(i, spaces, dim):
        l2_error = 0.0
        n_components = m_space.n_comp
        el_data: ElementData = spaces[1].elements[i].data
        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # scattering dof
        dest = m_space.dof_map.destination_indices(cell.id) + s_n_dof_g
        alpha_l = alpha[dest]

        # vectorization
        n_phi = phi_tab.shape[2]

        alpha_star = np.array(np.split(alpha_l, n_phi))
        for i, omega in enumerate(weights):
            m_e = m_exact(x[i, 0], x[i, 1], x[i, 2])
            m_h = np.vstack(
                tuple(
                    [
                        phi_tab[0, i, :, 0:dim].T @ alpha_star[:, d]
                        for d in range(n_components)
                    ]
                )
            )
            if dim == 2:
                m_h = m_h.T
            diff_m = m_e - m_h
            if dim == 2:
                l2_error += det_jac[i] * weights[i] * (diff_m.T @ diff_m)[0, 0]
            else:
                l2_error += det_jac[i] * weights[i] * np.trace(diff_m.T @ diff_m)
        return l2_error

    st = time.time()
    u_error_vec = [compute_u_l2_error(i, spaces, dim) for i in range(u_n_els)]
    t_error_vec = [compute_t_l2_error(i, spaces, dim) for i in range(u_n_els)]
    s_error_vec = [compute_s_l2_error(i, spaces, dim) for i in range(s_n_els)]
    m_error_vec = [compute_m_l2_error(i, spaces, dim) for i in range(m_n_els)]
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    u_l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, u_error_vec))
    t_l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, t_error_vec))
    s_l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, s_error_vec))
    m_l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, m_error_vec))
    print("L2-error displacement: ", u_l2_error)
    print("L2-error rotation: ", t_l2_error)
    print("L2-error stress: ", s_l2_error)
    print("L2-error couple stress: ", m_l2_error)

    if write_vtk_q:
        # post-process solution
        st = time.time()

        # writing solution on mesh points
        uh_data = np.zeros((len(gmesh.points), u_components))
        ue_data = np.zeros((len(gmesh.points), u_components))
        th_data = np.zeros((len(gmesh.points), t_components))
        te_data = np.zeros((len(gmesh.points), t_components))
        sh_data = np.zeros((len(gmesh.points), dim * dim))
        se_data = np.zeros((len(gmesh.points), dim * dim))
        if dim == 2:
            mh_data = np.zeros((len(gmesh.points), dim))
            me_data = np.zeros((len(gmesh.points), dim))
        else:
            mh_data = np.zeros((len(gmesh.points), dim * dim))
            me_data = np.zeros((len(gmesh.points), dim * dim))

        # displacement
        vertices = u_space.mesh_topology.entities_by_dimension(0)
        cellid_to_element = dict(zip(u_space.element_ids, u_space.elements))
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
            dest = u_space.dof_map.destination_indices(cell.id) + s_n_dof_g + m_n_dof_g
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
            alpha_star = np.array(np.split(alpha_l, n_phi))

            u_e = u_exact(x[:, 0], x[:, 1], x[:, 2])[0:dim, :]
            u_h = (phi_tab[0, :, :, 0] @ alpha_star).T

            uh_data[target_node_id] = u_h.ravel()
            ue_data[target_node_id] = u_e.ravel()

        # rotation
        vertices = t_space.mesh_topology.entities_by_dimension(0)
        cellid_to_element = dict(zip(t_space.element_ids, t_space.elements))
        cell_vertex_map = t_space.mesh_topology.entity_map_by_dimension(0)
        for id in vertices:
            if not cell_vertex_map.has_node(id):
                continue

            pr_ids = list(cell_vertex_map.predecessors(id))
            cell = gmesh.cells[pr_ids[0]]
            if cell.dimension != t_space.dimension:
                continue

            element = cellid_to_element[pr_ids[0]]

            # scattering dof
            dest = (
                t_space.dof_map.destination_indices(cell.id)
                + s_n_dof_g
                + m_n_dof_g
                + u_n_dof_g
            )
            alpha_l = alpha[dest]

            par_points = basix.geometry(t_space.element_type)

            target_node_id = gmesh.cells[id].node_tags[0]
            par_point_id = np.array(
                [
                    i
                    for i, node_id in enumerate(cell.node_tags)
                    if node_id == target_node_id
                ]
            )

            points = gmesh.points[target_node_id]
            if t_space.dimension != 0:
                points = par_points[par_point_id]

            # evaluate mapping
            phi_shapes = evaluate_linear_shapes(points, element.data)
            (x, jac, det_jac, inv_jac) = evaluate_mapping(
                cell.dimension, phi_shapes, gmesh.points[cell.node_tags]
            )
            phi_tab = element.evaluate_basis(points)
            n_phi = phi_tab.shape[2]
            t_e = u_exact(x[:, 0], x[:, 1], x[:, 2])[
                dim : u_components + t_components, :
            ]
            alpha_star = np.array(np.split(alpha_l, n_phi))
            t_h = (phi_tab[0, :, :, 0] @ alpha_star).T
            th_data[target_node_id] = t_h.ravel()
            te_data[target_node_id] = t_e.ravel()

        # stress
        vertices = s_space.mesh_topology.entities_by_dimension(0)
        cellid_to_element = dict(zip(s_space.element_ids, s_space.elements))
        cell_vertex_map = s_space.mesh_topology.entity_map_by_dimension(0)
        for id in vertices:
            if not cell_vertex_map.has_node(id):
                continue

            pr_ids = list(cell_vertex_map.predecessors(id))
            cell = gmesh.cells[pr_ids[0]]
            if cell.dimension != s_space.dimension:
                continue

            element = cellid_to_element[pr_ids[0]]

            # scattering dof
            dest = s_space.dof_map.destination_indices(cell.id)
            alpha_l = alpha[dest]

            par_points = basix.geometry(s_space.element_type)

            target_node_id = gmesh.cells[id].node_tags[0]
            par_point_id = np.array(
                [
                    i
                    for i, node_id in enumerate(cell.node_tags)
                    if node_id == target_node_id
                ]
            )

            points = gmesh.points[target_node_id]
            if s_space.dimension != 0:
                points = par_points[par_point_id]

            # evaluate mapping
            phi_shapes = evaluate_linear_shapes(points, element.data)
            (x, jac, det_jac, inv_jac) = evaluate_mapping(
                cell.dimension, phi_shapes, gmesh.points[cell.node_tags]
            )
            phi_tab = element.evaluate_basis(points)
            n_phi = phi_tab.shape[2]
            s_e = s_exact(x[:, 0], x[:, 1], x[:, 2])
            alpha_star = np.array(np.split(alpha_l, n_phi))
            s_h = np.vstack(
                tuple(
                    [phi_tab[0, 0, :, 0:dim].T @ alpha_star[:, d] for d in range(dim)]
                )
            )
            sh_data[target_node_id] = s_h.ravel()
            se_data[target_node_id] = s_e.ravel()

        # couple stress
        vertices = m_space.mesh_topology.entities_by_dimension(0)
        cellid_to_element = dict(zip(m_space.element_ids, m_space.elements))
        cell_vertex_map = m_space.mesh_topology.entity_map_by_dimension(0)
        for id in vertices:
            if not cell_vertex_map.has_node(id):
                continue

            pr_ids = list(cell_vertex_map.predecessors(id))
            cell = gmesh.cells[pr_ids[0]]
            if cell.dimension != m_space.dimension:
                continue

            element = cellid_to_element[pr_ids[0]]

            # scattering dof
            dest = m_space.dof_map.destination_indices(cell.id) + s_n_dof_g
            alpha_l = alpha[dest]

            par_points = basix.geometry(m_space.element_type)

            target_node_id = gmesh.cells[id].node_tags[0]
            par_point_id = np.array(
                [
                    i
                    for i, node_id in enumerate(cell.node_tags)
                    if node_id == target_node_id
                ]
            )

            points = gmesh.points[target_node_id]
            if s_space.dimension != 0:
                points = par_points[par_point_id]

            # evaluate mapping
            phi_shapes = evaluate_linear_shapes(points, element.data)
            (x, jac, det_jac, inv_jac) = evaluate_mapping(
                cell.dimension, phi_shapes, gmesh.points[cell.node_tags]
            )
            phi_tab = element.evaluate_basis(points)
            n_phi = phi_tab.shape[2]
            m_e = m_exact(x[0, 0], x[0, 1], x[0, 2])
            alpha_star = np.array(np.split(alpha_l, n_phi))
            m_h = np.vstack(
                tuple(
                    [
                        phi_tab[0, 0, :, 0:dim].T @ alpha_star[:, d]
                        for d in range(m_components)
                    ]
                )
            )
            if dim == 2:
                m_h = m_h.T
            mh_data[target_node_id] = m_h.ravel()
            me_data[target_node_id] = m_e.ravel()

        mesh_points = gmesh.points
        con_d = np.array([element.data.cell.node_tags for element in u_space.elements])
        meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
        cells_dict = {meshio_cell_types[u_space.dimension]: con_d}
        p_data_dict = {
            "u_h": uh_data,
            "u_exact": ue_data,
            "t_h": th_data,
            "t_exact": te_data,
            "s_h": sh_data,
            "s_exact": se_data,
            "m_h": mh_data,
            "m_exact": me_data,
        }

        mesh = meshio.Mesh(
            points=mesh_points,
            cells=cells_dict,
            # Optionally provide extra data on points, cells, etc.
            point_data=p_data_dict,
        )
        mesh.write("rates_hdiv_cosserat_elasticity.vtk")
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return np.array([u_l2_error, t_l2_error, s_l2_error, m_l2_error])


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


def perform_convergence_test(configuration: dict):
    # retrieve parameters from dictionary
    k_order = configuration.get("k_order")
    n_ref = configuration.get("n_refinements")
    dimension = configuration.get("dimension")
    mixed_form_q = configuration.get("dual_problem_Q", False)
    write_geometry_vtk = configuration.get("write_geometry_Q", False)
    write_vtk = configuration.get("write_vtk_Q", False)
    report_full_precision_data = configuration.get(
        "report_full_precision_data_Q", False
    )

    # The initial element size
    h = 1.0

    # Create a unit squared or a unit cube
    domain = create_domain(dimension)

    n_data = 5
    error_data = np.empty((0, n_data), float)
    for l in range(n_ref):
        h_val = h * (2**-l)
        mesher = create_conformal_mesher(domain, h, l)
        gmesh = create_mesh(dimension, mesher, write_geometry_vtk)
        if mixed_form_q:
            error_vals = hdiv_cosserat_elasticity(k_order, gmesh, write_vtk)
        else:
            error_vals = h1_cosserat_elasticity(k_order, gmesh, write_vtk)
        chunk = np.concatenate([[h_val], error_vals])
        error_data = np.append(error_data, np.array([chunk]), axis=0)

    rates_data = np.empty((0, n_data - 1), float)
    for i in range(error_data.shape[0] - 1):
        chunk_b = np.log(error_data[i])
        chunk_e = np.log(error_data[i + 1])
        h_step = chunk_e[0] - chunk_b[0]
        partial = (chunk_e - chunk_b) / h_step
        rates_data = np.append(rates_data, np.array([list(partial[1:n_data])]), axis=0)

    # minimal report
    if report_full_precision_data:
        print("error data: ", error_data)
        print("error rates data: ", rates_data)

    np.set_printoptions(precision=3)
    if mixed_form_q:
        print("Dual problem")
    else:
        print("Primal problem")
    print("Polynomial order: ", k_order)
    print("Dimension: ", dimension)
    print("rounded error data: ", error_data)
    print("rounded error rates data: ", rates_data)
    print(" ")
    if mixed_form_q:
        if report_full_precision_data:
            np.savetxt(
                "dual_problem_k"
                + str(k_order)
                + "_"
                + str(dimension)
                + "d_l2_error_data.txt",
                error_data,
                delimiter=",",
                header="element size, displacement, rotation, stress, couple stress",
            )
            np.savetxt(
                "dual_problem_k"
                + str(k_order)
                + "_"
                + str(dimension)
                + "d_l2_expected_order_convergence.txt",
                rates_data,
                delimiter=",",
                header="displacement, rotation, stress, couple stress",
            )
        np.savetxt(
            "dual_problem_k"
            + str(k_order)
            + "_"
            + str(dimension)
            + "d_l2_error_data_rounded.txt",
            error_data,
            fmt="%1.3e",
            delimiter=",",
            header="element size, displacement, rotation, stress, couple stress",
        )
        np.savetxt(
            "dual_problem_k"
            + str(k_order)
            + "_"
            + str(dimension)
            + "d_l2_expected_order_convergence_rounded.txt",
            rates_data,
            fmt="%1.3f",
            delimiter=",",
            header="displacement, rotation, stress, couple stress",
        )

    else:
        if report_full_precision_data:
            np.savetxt(
                "primal_problem_k"
                + str(k_order)
                + "_"
                + str(dimension)
                + "d_l2_error_data.txt",
                error_data,
                delimiter=",",
                header="element size, displacement, rotation, stress, couple stress",
            )
            np.savetxt(
                "primal_problem_k"
                + str(k_order)
                + "_"
                + str(dimension)
                + "d_l2_expected_order_convergence.txt",
                rates_data,
                delimiter=",",
                header="displacement, rotation, stress, couple stress",
            )
        np.savetxt(
            "primal_problem_k"
            + str(k_order)
            + "_"
            + str(dimension)
            + "d_l2_error_data_rounded.txt",
            error_data,
            fmt="%1.3e",
            delimiter=",",
            header="element size, displacement, rotation, stress, couple stress",
        )
        np.savetxt(
            "primal_problem_k"
            + str(k_order)
            + "_"
            + str(dimension)
            + "d_l2_expected_order_convergence_rounded.txt",
            rates_data,
            fmt="%1.3f",
            delimiter=",",
            header="displacement, rotation, stress, couple stress",
        )

    return


def main():
    write_vtk_files_Q = False
    report_full_precision_data_Q = False

    primal_configuration = {
        "n_refinements": 3,
        "write_geometry_Q": write_vtk_files_Q,
        "write_vtk_Q": write_vtk_files_Q,
        "report_full_precision_data_Q": report_full_precision_data_Q,
    }

    # # primal problem
    # for k in [1, 2, 3]:
    #     for d in [2]:
    #         primal_configuration.__setitem__("k_order", k)
    #         primal_configuration.__setitem__("dimension", d)
    #         perform_convergence_test(primal_configuration)

    dual_configuration = {
        "n_refinements": 3,
        "dual_problem_Q": True,
        "write_geometry_Q": write_vtk_files_Q,
        "write_vtk_Q": write_vtk_files_Q,
        "report_full_precision_data_Q": report_full_precision_data_Q,
    }

    # dual problem
    for k in [1]:
        for d in [3]:
            dual_configuration.__setitem__("k_order", k)
            dual_configuration.__setitem__("dimension", d)
            perform_convergence_test(dual_configuration)


if __name__ == "__main__":
    main()
