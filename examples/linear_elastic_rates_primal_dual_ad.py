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
import auto_diff as ad
from auto_diff.vecvalder import VecValDer

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
import strong_solution_elasticity as le


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
    u_exact = le.displacement(m_lambda, m_mu, dim)
    s_exact = le.stress(m_lambda, m_mu, dim)
    f_rhs = le.rhs(m_lambda, m_mu, dim)

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
            r_el[b:e:n_components] += -1.0 * phi_s_star @ f_val_star[c]

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
            element, m_lambda, m_mu, f_rhs, u_space, cell_map, row, col, data
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
    # alpha = sp.linalg.spsolve(jg, rg)
    alpha = sp_solver.spsolve(jg, rg)
    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    # Computing displacement L2 error
    def compute_u_l2_error(element, u_space):
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
        p_e_s = u_exact(x[:, 0], x[:, 1], x[:, 2])
        alpha_star = np.array(np.split(alpha_l, n_phi))
        p_h_s = (phi_tab[0, :, :, 0] @ alpha_star).T
        diff_p = p_e_s - p_h_s
        l2_error = np.sum(det_jac * weights * diff_p * diff_p)
        return l2_error

    # Computing stress L2 error
    def compute_s_l2_error(element, u_space, m_mu, m_lambda, dim):
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
            s_e = s_exact(x[i, 0], x[i, 1], x[i, 2])
            grad_phi = inv_jac[i].T @ phi_tab[1 : phi_tab.shape[0] + 1, i, :, 0]
            grad_uh = grad_phi[0:dim] @ alpha_star
            eps_h = 0.5 * (grad_uh + grad_uh.T)
            s_h = 2.0 * m_mu * eps_h + m_lambda * eps_h.trace() * np.identity(dim)
            diff_s = s_e - s_h
            l2_error += det_jac[i] * weights[i] * np.trace(diff_s.T @ diff_s)
        return l2_error

    st = time.time()
    u_error_vec = [compute_u_l2_error(element, u_space) for element in u_space.elements]
    s_error_vec = [
        compute_s_l2_error(element, u_space, m_mu, m_lambda, dim)
        for element in u_space.elements
    ]
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    u_l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, u_error_vec))
    s_l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, s_error_vec))
    print("L2-error displacement: ", u_l2_error)
    print("L2-error stress: ", s_l2_error)

    if write_vtk_q:
        # post-process solution
        st = time.time()
        cellid_to_element = dict(zip(u_space.element_ids, u_space.elements))
        # writing solution on mesh points
        vertices = u_space.mesh_topology.entities_by_dimension(0)
        uh_data = np.zeros((len(gmesh.points), n_components))
        ue_data = np.zeros((len(gmesh.points), n_components))
        sh_data = np.zeros((len(gmesh.points), dim*dim))
        se_data = np.zeros((len(gmesh.points), dim*dim))
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
            u_e = u_exact(x[:, 0], x[:, 1], x[:, 2])
            s_e = s_exact(x[:, 0], x[:, 1], x[:, 2])
            alpha_star = np.array(np.split(alpha_l, n_phi))
            u_h = (phi_tab[0, :, :, 0] @ alpha_star).T

            grad_phi = inv_jac[0].T @ phi_tab[1 : phi_tab.shape[0] + 1, 0, :, 0]
            grad_uh = grad_phi[0:dim] @ alpha_star
            eps_h = 0.5 * (grad_uh + grad_uh.T)
            s_h = 2.0 * m_mu * eps_h + m_lambda * eps_h.trace() * np.identity(dim)

            uh_data[target_node_id] = u_h.ravel()
            ue_data[target_node_id] = u_e.ravel()
            sh_data[target_node_id] = s_h.ravel()
            se_data[target_node_id] = s_e.ravel()

        mesh_points = gmesh.points
        con_d = np.array([element.data.cell.node_tags for element in u_space.elements])
        meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
        cells_dict = {meshio_cell_types[u_space.dimension]: con_d}
        p_data_dict = {"u_h": uh_data, "u_exact": ue_data, "s_h": sh_data,"s_exact": se_data}

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

    return np.array([u_l2_error, s_l2_error])


def hdiv_elasticity(k_order, gmesh, write_vtk_q=False):

    dim = gmesh.dimension
    # Material data

    m_lambda = 1.0
    m_mu = 1.0

    # FESpace: data
    s_components = 2
    u_components = 2
    t_components = 1
    if dim == 3:
        s_components = 3
        u_components = 3
        t_components = 3

    s_family = "BDM"
    u_family = "Lagrange"
    t_family = "Lagrange"

    # flux space
    s_space = DiscreteSpace(
        dim, s_components, s_family, k_order, gmesh, integration_oder=2 * k_order + 1
    )
    if dim == 2:
        s_space.build_structures([2, 3, 4, 5])
    elif dim == 3:
        s_space.build_structures([2, 3, 4, 5, 6, 7])

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

    # rotations space
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
    u_n_els = len(u_space.elements)
    t_n_els = len(t_space.elements)
    assert s_n_els == u_n_els == t_n_els

    components = (s_components, u_components, t_components)
    spaces = (s_space, u_space, t_space)

    for i in range(s_n_els):
        s_element = s_space.elements[i]
        u_element = u_space.elements[i]
        t_element = t_space.elements[i]
        cell = s_element.data.cell
        elements = (s_element, u_element, t_element)

        n_dof = 0
        for j, element in enumerate(elements):
            for n_entity_dofs in element.basis_generator.num_entity_dofs:
                n_dof = n_dof + sum(n_entity_dofs) * components[j]

        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    for element in s_space.bc_elements:
        cell = element.data.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs) * s_components
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    row = np.zeros((c_size), dtype=np.int64)
    col = np.zeros((c_size), dtype=np.int64)
    data = np.zeros((c_size), dtype=np.float64)

    s_n_dof_g = s_space.dof_map.dof_number()
    u_n_dof_g = u_space.dof_map.dof_number()
    t_n_dof_g = t_space.dof_map.dof_number()
    n_dof_g = s_n_dof_g + u_n_dof_g + t_n_dof_g
    rg = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    et = time.time()
    elapsed_time = et - st
    print("Triplets creation time:", elapsed_time, "seconds")

    st = time.time()

    # exact solution
    u_exact = le.displacement(m_lambda, m_mu, dim)
    t_exact = le.rotations(m_lambda, m_mu, dim)
    s_exact = le.stress(m_lambda, m_mu, dim)
    f_rhs = le.rhs(m_lambda, m_mu, dim)

    # print("displacement: ", u_exact(0.25,0.5,0.75))
    # print("rotation: ", t_exact(0.25, 0.5, 0.75))
    # print("stress: ", s_exact(0.25, 0.5, 0.75))
    # print("rhs: ", f_rhs(0.25, 0.5, 0.75))
    # aka = 0

    def scatter_form_data_ad(
        i, m_lambda, m_mu, f_rhs, spaces, cell_map, row, col, data
    ):

        dim = spaces[0].dimension
        s_components = spaces[0].n_comp
        u_components = spaces[1].n_comp
        t_components = spaces[2].n_comp

        s_data: ElementData = spaces[0].elements[i].data
        u_data: ElementData = spaces[1].elements[i].data
        t_data: ElementData = spaces[2].elements[i].data

        cell = s_data.cell

        points = s_data.quadrature.points
        weights = s_data.quadrature.weights
        x = s_data.mapping.x
        det_jac = s_data.mapping.det_jac
        inv_jac = s_data.mapping.inv_jac

        # basis
        s_phi_tab = s_data.basis.phi
        u_phi_tab = u_data.basis.phi
        t_phi_tab = t_data.basis.phi

        # destination indexes
        dest_s = s_space.dof_map.destination_indices(cell.id)
        dest_u = u_space.dof_map.destination_indices(cell.id) + s_n_dof_g
        dest_t = t_space.dof_map.destination_indices(cell.id) + s_n_dof_g + u_n_dof_g
        dest = np.concatenate([dest_s, dest_u, dest_t])
        n_s_phi = s_phi_tab.shape[2]
        n_u_phi = u_phi_tab.shape[2]
        n_t_phi = t_phi_tab.shape[2]

        n_s_dof = n_s_phi * s_components
        n_u_dof = n_u_phi * u_components
        n_t_dof = n_t_phi * t_components

        n_dof = n_s_dof + n_u_dof + n_t_dof
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
                b = c + n_s_dof
                e = b + n_u_dof
                el_form[b:e:u_components] += -1.0 * phi_s_star @ f_val_star[c]

            for i, omega in enumerate(weights):

                if dim == 2:
                    c = 0
                    a_sx = alpha[:, c : n_s_dof + c : s_components]
                    a_ux = alpha[:, n_s_dof + c : n_s_dof + n_u_dof + c : u_components]
                    c = 1
                    a_sy = alpha[:, c : n_s_dof + c : s_components]
                    a_uy = alpha[:, n_s_dof + c : n_s_dof + n_u_dof + c : u_components]
                    a_gh = alpha[
                        :,
                        n_s_dof + n_u_dof : n_s_dof + n_u_dof + n_t_dof : t_components,
                    ]

                    s_x = a_sx @ s_phi_tab[0, i, :, 0:dim]
                    ux_h = a_ux @ u_phi_tab[0, i, :, 0:dim]
                    s_y = a_sy @ s_phi_tab[0, i, :, 0:dim]
                    uy_h = a_uy @ u_phi_tab[0, i, :, 0:dim]
                    gamma_h = a_gh @ t_phi_tab[0, i, :, 0:dim]

                    u_h = VecValDer(
                        np.hstack((ux_h.val, uy_h.val)), np.hstack((ux_h.der, uy_h.der))
                    )
                    s_h = VecValDer(
                        np.vstack((s_x.val, s_y.val)), np.vstack((s_x.der, s_y.der))
                    )

                    # Stress decomposition
                    Skew_sh = 0.5 * (s_h - s_h.T)

                    tr_s_h = VecValDer(s_h.val.trace(), s_h.der.trace())
                    A_sh = (1.0 / 2.0 * m_mu) * (
                        s_h
                        - (m_lambda / (2.0 * m_mu + dim * m_lambda)) * tr_s_h * Imat
                    )

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

                    div_sh = VecValDer(
                        np.hstack((div_sh_x.val, div_sh_y.val)),
                        np.hstack((div_sh_x.der, div_sh_y.der)),
                    )

                    Gamma_outer = -gamma_h * np.array([[0.0, -1.0], [1.0, 0.0]])
                    S_cross = np.array([[Skew_sh[1, 0] - Skew_sh[0, 1]]])

                else:

                    c = 0
                    a_sx = alpha[:, c : n_s_dof + c : s_components]
                    a_ux = alpha[:, n_s_dof + c : n_s_dof + n_u_dof + c : u_components]
                    a_gx = alpha[
                        :,
                        n_s_dof
                        + n_u_dof
                        + c : n_s_dof
                        + n_u_dof
                        + n_t_dof
                        + c : u_components,
                    ]

                    c = 1
                    a_sy = alpha[:, c : n_s_dof + c : s_components]
                    a_uy = alpha[:, n_s_dof + c : n_s_dof + n_u_dof + c : u_components]
                    a_gy = alpha[
                        :,
                        n_s_dof
                        + n_u_dof
                        + c : n_s_dof
                        + n_u_dof
                        + n_t_dof
                        + c : u_components,
                    ]

                    c = 2
                    a_sz = alpha[:, c : n_s_dof + c : s_components]
                    a_uz = alpha[:, n_s_dof + c : n_s_dof + n_u_dof + c : u_components]
                    a_gz = alpha[
                        :,
                        n_s_dof
                        + n_u_dof
                        + c : n_s_dof
                        + n_u_dof
                        + n_t_dof
                        + c : u_components,
                    ]

                    s_x = a_sx @ s_phi_tab[0, i, :, 0:dim]
                    ux_h = a_ux @ u_phi_tab[0, i, :, 0:dim]
                    gx_h = a_gx @ t_phi_tab[0, i, :, 0:dim]

                    s_y = a_sy @ s_phi_tab[0, i, :, 0:dim]
                    uy_h = a_uy @ u_phi_tab[0, i, :, 0:dim]
                    gy_h = a_gy @ t_phi_tab[0, i, :, 0:dim]

                    s_z = a_sz @ s_phi_tab[0, i, :, 0:dim]
                    uz_h = a_uz @ u_phi_tab[0, i, :, 0:dim]
                    gz_h = a_gz @ t_phi_tab[0, i, :, 0:dim]

                    u_h = VecValDer(
                        np.hstack((ux_h.val, uy_h.val, uz_h.val)),
                        np.hstack((ux_h.der, uy_h.der, uz_h.der)),
                    )

                    g_h = VecValDer(
                        np.hstack((gx_h.val, gy_h.val, gz_h.val)),
                        np.hstack((gx_h.der, gy_h.der, gz_h.der)),
                    )

                    s_h = VecValDer(
                        np.vstack((s_x.val, s_y.val, s_z.val)),
                        np.vstack((s_x.der, s_y.der, s_z.der)),
                    )

                    # Stress decomposition
                    Skew_sh = 0.5 * (s_h - s_h.T)

                    tr_s_h = VecValDer(s_h.val.trace(), s_h.der.trace())
                    A_sh = (1.0 / 2.0 * m_mu) * (
                        s_h
                        - (m_lambda / (2.0 * m_mu + dim * m_lambda)) * tr_s_h * Imat
                    )

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

                    Gamma_outer = -np.array(
                        [
                            [0.0 * g_h[0, 0], -g_h[0, 2], +g_h[0, 1]],
                            [+g_h[0, 2], 0.0 * g_h[0, 0], -g_h[0, 0]],
                            [-g_h[0, 1], +g_h[0, 0], 0.0 * g_h[0, 0]],
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
                    + (div_tau.T @ u_h)
                    + (s_phi_tab[0, i, :, 0:dim] @ Gamma_outer)
                )
                equ_2_integrand = u_phi_tab[0, i, :, 0:dim] @ div_sh
                equ_3_integrand = t_phi_tab[0, i, :, 0:dim] @ S_cross

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, 0:n_s_dof:1] = (equ_1_integrand).reshape(
                    (n_s_dof,)
                )
                multiphysic_integrand[:, n_s_dof : n_s_dof + n_u_dof : 1] = (
                    equ_2_integrand
                ).reshape((n_u_dof,))
                multiphysic_integrand[
                    :, n_s_dof + n_u_dof : n_s_dof + n_u_dof + n_t_dof : 1
                ] = (equ_3_integrand).reshape((n_t_dof,))

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
        scatter_form_data_ad(i, m_lambda, m_mu, f_rhs, spaces, cell_map, row, col, data)
        for i in range(s_n_els)
    ]

    jg = coo_matrix((data, (row, col)), shape=(n_dof_g, n_dof_g)).tocsr()
    et = time.time()
    elapsed_time = et - st
    print("Assembly time:", elapsed_time, "seconds")

    # solving ls
    st = time.time()
    alpha = sp.linalg.spsolve(jg, -rg)
    # alpha = sp_solver.spsolve(jg, -rg)
    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    # Computing displacement L2 error
    def compute_u_l2_error(element, u_space):
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
        dest = u_space.dof_map.destination_indices(cell.id) + s_n_dof_g
        alpha_l = alpha[dest]

        # vectorization
        n_phi = phi_tab.shape[2]
        p_e_s = u_exact(x[:, 0], x[:, 1], x[:, 2])
        alpha_star = np.array(np.split(alpha_l, n_phi))
        p_h_s = (phi_tab[0, :, :, 0] @ alpha_star).T
        diff_p = p_e_s - p_h_s
        l2_error = np.sum(det_jac * weights * diff_p * diff_p)
        return l2_error

    # Computing rotation L2 error
    def compute_t_l2_error(element, t_space):
        l2_error = 0.0
        n_components = t_space.n_comp
        el_data = element.data
        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # scattering dof
        dest = t_space.dof_map.destination_indices(cell.id) + s_n_dof_g + u_n_dof_g
        alpha_l = alpha[dest]

        # vectorization
        n_phi = phi_tab.shape[2]
        alpha_star = np.array(np.split(alpha_l, n_phi))
        t_h_s = (phi_tab[0, :, :, 0] @ alpha_star).T
        for i, omega in enumerate(weights):
            t_e = t_exact(x[i, 0], x[i, 1], x[i, 2])
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

            diff_t = t_e - t_h
            l2_error += det_jac[i] * weights[i] * np.trace(diff_t.T @ diff_t)
        return l2_error

    # Computing stress L2 error
    def compute_s_l2_error(element, s_space, dim):
        l2_error = 0.0
        n_components = s_space.n_comp
        el_data = element.data
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

        c = 0
        alpha_x = alpha_l[c : n_phi * s_components + c : s_components]
        c = 1
        alpha_y = alpha_l[c : n_phi * s_components + c : s_components]
        c = 2
        alpha_z = alpha_l[c : n_phi * s_components + c : s_components]
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

    st = time.time()
    s_error_vec = [
        compute_s_l2_error(element, s_space, dim) for element in s_space.elements
    ]
    u_error_vec = [compute_u_l2_error(element, u_space) for element in u_space.elements]
    t_error_vec = [compute_t_l2_error(element, t_space) for element in t_space.elements]
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    s_l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, s_error_vec))
    u_l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, u_error_vec))
    t_l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, t_error_vec))
    print("L2-error stress: ", s_l2_error)
    print("L2-error displacement: ", u_l2_error)
    print("L2-error rotation: ", t_l2_error)

    if write_vtk_q:
        # post-process solution
        st = time.time()

        # writing solution on mesh points
        vertices = u_space.mesh_topology.entities_by_dimension(0)
        uh_data = np.zeros((len(gmesh.points), u_components))
        ue_data = np.zeros((len(gmesh.points), u_components))
        sh_data = np.zeros((len(gmesh.points), dim * dim))
        se_data = np.zeros((len(gmesh.points), dim*dim))
        th_data = np.zeros((len(gmesh.points), dim*dim))
        te_data = np.zeros((len(gmesh.points), dim*dim))

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
            dest = u_space.dof_map.destination_indices(cell.id) + s_n_dof_g
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
            u_e = u_exact(x[:, 0], x[:, 1], x[:, 2])
            alpha_star = np.array(np.split(alpha_l, n_phi))
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
            dest = t_space.dof_map.destination_indices(cell.id) + s_n_dof_g + u_n_dof_g
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
            t_e = t_exact(x[:, 0], x[:, 1], x[:, 2])
            alpha_star = np.array(np.split(alpha_l, n_phi))
            t_h_s = (phi_tab[0, :, :, 0] @ alpha_star).T
            if dim == 2:
                t_h = np.array([[0.0, -t_h_s[0, 0]], [t_h_s[0, 0], 0.0]])
            else:
                t_h = np.array(
                    [
                        [0.0, -t_h_s[2, 0], +t_h_s[1, 0]],
                        [+t_h_s[2, 0], 0.0, -t_h_s[0, 0]],
                        [-t_h_s[1, 0], +t_h_s[0, 0], 0.0],
                    ]
                )
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
            s_h = np.vstack(tuple([phi_tab[0, 0, :, 0:dim].T @ alpha_star[:, d] for d in range(dim)]))
            sh_data[target_node_id] = s_h.ravel()
            se_data[target_node_id] = s_e.ravel()

        mesh_points = gmesh.points
        con_d = np.array([element.data.cell.node_tags for element in u_space.elements])
        meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
        cells_dict = {meshio_cell_types[u_space.dimension]: con_d}
        p_data_dict = {"u_h": uh_data, "u_exact": ue_data, "t_h": th_data, "t_exact": te_data, "s_h": sh_data,"s_exact": se_data}

        mesh = meshio.Mesh(
            points=mesh_points,
            cells=cells_dict,
            # Optionally provide extra data on points, cells, etc.
            point_data=p_data_dict,
        )
        mesh.write("rates_hdiv_elasticity.vtk")
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return np.array([s_l2_error, u_l2_error, t_l2_error])


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
    n_ref = 3
    dimension = 3
    ref_l = 0
    mixed_form_q = True

    domain = create_domain(dimension)
    n_data = -1
    if mixed_form_q:
        n_data = 4
    else:
        n_data = 3

    error_data = np.empty((0, n_data), float)
    for l in range(n_ref):
        h_val = h * (2**-l)
        mesher = create_conformal_mesher(domain, h, l)
        gmesh = create_mesh(dimension, mesher, False)
        if mixed_form_q:
            error_vals = hdiv_elasticity(k_order, gmesh, True)
        else:
            error_vals = h1_elasticity(k_order, gmesh, True)
        chunk = np.concatenate([[h_val], error_vals])
        error_data = np.append(error_data, np.array([chunk]), axis=0)

    rates_data = np.empty((0, n_data - 1), float)
    for i in range(error_data.shape[0] - 1):
        chunk_b = np.log(error_data[i])
        chunk_e = np.log(error_data[i + 1])
        h_step = chunk_e[0] - chunk_b[0]
        partial = (chunk_e - chunk_b) / h_step
        rates_data = np.append(rates_data, np.array([list(partial[1:n_data])]), axis=0)

    print("error data: ", error_data)
    print("error rates data: ", rates_data)

    np.set_printoptions(precision=4)
    print("rounded error data: ", error_data)
    print("rounded error rates data: ", rates_data)

    return


if __name__ == "__main__":
    main()
