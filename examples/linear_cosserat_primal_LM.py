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
from spaces.discrete_space import DiscreteSpace
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


def lm_h1_elasticity(k_order, gmesh, write_vtk_q=False):

    #
    fixed_point_q = True
    skin_stiffness_q = True
    dim = gmesh.dimension
    domain: Domain = gmesh.conformal_mesher.domain
    domain_tags = [1, 2, 3, 4, 5]
    frac_physical_tags = [
        shape.physical_tag
        for shape in domain.shapes[dim - 1]
        if (shape.physical_tag is not None) and (shape.physical_tag not in domain_tags)
    ]
    skin_p_physical_tags = [1000 * p_tag + 1 for p_tag in frac_physical_tags]
    skin_m_physical_tags = [1000 * p_tag - 1 for p_tag in frac_physical_tags]

    # Material data
    m_lambda = 8.0e9
    m_mu = 8.0e9
    A_n = 0.0
    A_t = 0.0
    Kv = 1.33333e10
    Gv = m_mu
    s_n = -35.0e6

    hs = 1.0e-10
    m_s_lambda = hs * 8.0e9
    m_s_mu = hs * 8.0e9

    # FESpace: data
    n_components = 2
    if dim == 3:
        n_components = 3

    discontinuous = True
    family = "Lagrange"

    u_space = DiscreteSpace(dim, n_components, family, k_order, gmesh)
    um_space = DiscreteSpace(dim - 1, n_components, family, k_order, gmesh)
    up_space = DiscreteSpace(dim - 1, n_components, family, k_order, gmesh)
    l_space = DiscreteSpace(
        dim - 1,
        n_components,
        family,
        k_order - 2,
        gmesh,
        integration_oder=2 * k_order + 1,
    )

    l_space.make_discontinuous()
    # u_space.build_structures([2, 3])
    if dim == 2:
        u_space.build_structures([2, 4])
    elif dim == 3:
        u_space.build_structures([7])
    up_space.build_structures_on_physical_tags(skin_p_physical_tags)
    um_space.build_structures_on_physical_tags(skin_m_physical_tags)
    l_space.build_structures_on_physical_tags(frac_physical_tags)

    # make entitie maps equal
    up_space.dof_map.vertex_map = u_space.dof_map.vertex_map
    um_space.dof_map.vertex_map = u_space.dof_map.vertex_map
    # l_space.dof_map.vertex_map = u_space.dof_map.vertex_map

    up_space.dof_map.edge_map = u_space.dof_map.edge_map
    um_space.dof_map.edge_map = u_space.dof_map.edge_map
    # l_space.dof_map.edge_map = u_space.dof_map.edge_map

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

    for element in up_space.elements:
        cell = element.data.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs) * n_components
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + 3 * n_dof * n_dof

    for element in um_space.elements:
        cell = element.data.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs) * n_components
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + 3 * n_dof * n_dof

    for element in l_space.elements:
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

    n_dof_g = u_space.dof_map.dof_number() + l_space.dof_map.dof_number()
    rg = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    et = time.time()
    elapsed_time = et - st
    print("Triplets creation time:", elapsed_time, "seconds")

    st = time.time()

    f_rhs = lambda x, y, z: np.array([0.0 * (1 - y), 0.0 * (1 - x)])
    s_load = lambda x, y, z: np.array([0.0 * x, s_n + 0.0 * y])
    if dim == 3:
        f_rhs = lambda x, y, z: np.array([(1 - y), -(1 - x), 0.0 * z])

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

    [
        scatter_form_data(
            element, m_lambda, m_mu, f_rhs, u_space, cell_map, row, col, data
        )
        for element in u_space.elements
    ]

    def scatter_K_skin_form_data(
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
        block_sequ = (
            np.array(range(0, len(dest) * len(dest))) + c_sequ + 2 * n_dof * n_dof
        )
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

    if skin_stiffness_q:

        [
            scatter_K_skin_form_data(
                element, m_s_lambda, m_s_mu, f_rhs, up_space, cell_map, row, col, data
            )
            for element in up_space.elements
        ]

        [
            scatter_K_skin_form_data(
                element, m_s_lambda, m_s_mu, f_rhs, um_space, cell_map, row, col, data
            )
            for element in um_space.elements
        ]

    def scatter_skin_form_data(
        element, up_space, l_space, cell_map, row, col, data, sign
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

        # print("cell.sub_cells_ids: ", cell.sub_cells_ids)
        lm_cell_id = cell.sub_cells_ids[1][1]
        lm_cell_index = l_space.id_to_element[lm_cell_id]
        lm_el_data = l_space.elements[lm_cell_index].data

        lm_points = lm_el_data.quadrature.points
        lm_weights = lm_el_data.quadrature.weights
        lm_phi_tab = lm_el_data.basis.phi
        lm_n = lm_el_data.cell.normal[np.array([0, 1])]
        lm_t = np.array([lm_n[1], -lm_n[0]])

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
        dest_lm = l_space.dof_map.destination_indices(lm_cell_id) + u_space.n_dof

        n_phi = phi_tab.shape[2]
        n_phi_lm = lm_phi_tab.shape[2]
        n_dof = n_phi * n_components
        n_dof_lm = n_phi_lm * n_components
        js = (n_dof, n_dof_lm)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # local blocks
        jac_block = np.zeros((n_phi, n_phi_lm))
        for i, omega in enumerate(weights):
            phi_u = phi_tab[0, i, :, 0]
            phi_l = lm_phi_tab[0, i, :, 0]
            jac_block += sign * det_jac[i] * omega * np.outer(phi_u, phi_l)

        for c in range(n_components):
            b = c
            e = (c + 1) * n_phi * n_components + 1
            j_el[b:e:n_components, b:e:n_components] += jac_block

        # scattering data
        c_sequ = cell_map[cell.id]

        block_size = len(dest) * len(dest_lm)

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest_lm))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest_lm))
        col[block_sequ] += np.tile(dest_lm, len(dest))
        data[block_sequ] += j_el.ravel()

        # transpose contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest_lm))) + c_sequ + block_size
        row[block_sequ] += np.repeat(dest_lm, len(dest))
        col[block_sequ] += np.tile(dest, len(dest_lm))
        data[block_sequ] += j_el.T.ravel()

    [
        scatter_skin_form_data(element, u_space, l_space, cell_map, row, col, data, -1)
        for element in up_space.elements
    ]

    [
        scatter_skin_form_data(element, u_space, l_space, cell_map, row, col, data, +1)
        for element in um_space.elements
    ]

    def scatter_lambda_form_data(A_n, A_t, element, l_space, cell_map, row, col, data):

        n_components = l_space.n_comp
        el_data: ElementData = element.data

        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # destination indexes
        dest = l_space.dof_map.destination_indices(cell.id) + u_space.n_dof

        n_phi = phi_tab.shape[2]
        n_dof = n_phi * n_components
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # local blocks
        jac_block = np.zeros((n_phi, n_phi))
        for i, omega in enumerate(weights):
            phi = phi_tab[0, i, :, 0]
            jac_block += det_jac[i] * omega * np.outer(phi, phi)

        A_data = [A_n, A_t]
        for c in range(n_components):
            b = c
            e = (c + 1) * n_phi * n_components + 1
            j_el[b:e:n_components, b:e:n_components] += A_data[c] * jac_block

        # scattering data
        c_sequ = cell_map[cell.id]

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

    [
        scatter_lambda_form_data(A_n, A_t, element, l_space, cell_map, row, col, data)
        for element in l_space.elements
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
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        scale = 1.0
        if cell.material_id == 2:
            scale = 0.0

        # Partial local vectorization
        f_val_star = scale * s_load(x[:, 0], x[:, 1], x[:, 2])
        phi_s_star = det_jac * weights * phi_tab[0, :, :, 0].T
        for c in range(n_components):
            b = c
            e = (c + 1) * n_phi * n_components + 1
            r_el[b:e:n_components] += phi_s_star @ f_val_star[c]

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

        # contribute rhs
        rg[dest] += r_el
        if cell.material_id == 4:
            return

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
    if fixed_point_q:
        alpha = sp_solver.spsolve(jg, rg)
        u_range = list(range(u_space.n_dof))
        l_range = list(range(u_space.n_dof, u_space.n_dof + l_space.n_dof))
        jg_uu = jg[u_range, :][:, u_range]
        jg_lu = jg[l_range, :][:, u_range]
        jg_ul = jg[u_range, :][:, l_range]
        rg_uu = rg[u_range]

        alpha_l = alpha[l_range]
        gt_p = np.zeros(int(len(l_range) / 2))
        for k in range(3):
            L_k = np.array(np.split(alpha_l, 2))
            Ln_k = L_k[0, :]
            Lt_k = L_k[1, :]

            # step 1: compute u jump
            alpha_u = sp_solver.spsolve(jg_uu, rg_uu - jg_ul * alpha_l)
            u_jump = np.array(np.split(jg_lu * alpha_u, 2))

            # step 2: For all LM
            n = np.empty((0, 2), dtype=float)
            t = np.empty((0, 2), dtype=float)
            for element in l_space.elements:
                el_data = element.data
                ln = el_data.cell.normal[np.array([0, 1])]
                lt = np.array([ln[1], -ln[0]])
                n = np.vstack((n, ln))
                t = np.vstack((t, lt))

            c_fric = 0.0
            theta = 20.0 * np.pi / 180.0
            gn = np.sum(u_jump * n.T, axis=0)
            gt = np.sum(u_jump * t.T, axis=0)

            # step one predict:
            gn0 = 0.0001
            # Tn = Ln_k - gn * (Kv * gn0) / (gn0-gn)
            Tn = Ln_k - Kv * gn
            Tt_trial = Lt_k - Gv * gt
            Phi = c_fric - Tn * np.tan(theta)
            elastic_q = Tt_trial - Phi <= 0.0
            Tt = np.where(elastic_q, Tt_trial, Phi)
            print("gn Tn : ", np.linalg.norm((gn * Tn)))
            print("norm in LM: ", np.linalg.norm((Ln_k - Tn)))
            L_k[0, :] = Tn
            L_k[1, :] = Tt
            alpha_l = L_k.ravel()
            if np.all(elastic_q):
                break
        alpha = np.concatenate((alpha_u, alpha_l))

    else:
        alpha = sp_solver.spsolve(jg, rg)
    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    if write_vtk_q:
        # post-process solution in d
        st = time.time()
        cellid_to_element = dict(zip(u_space.element_ids, u_space.elements))
        # writing solution on mesh points
        vertices = u_space.mesh_topology.entities_by_dimension(0)
        fh_data = np.zeros((len(gmesh.points), n_components))
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
            f_h = (phi_tab[0, :, :, 0] @ alpha_star).T
            fh_data[target_node_id] = f_h.ravel()

        mesh_points = gmesh.points
        con_d = np.array([element.data.cell.node_tags for element in u_space.elements])
        meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
        cells_dict = {meshio_cell_types[u_space.dimension]: con_d}
        p_data_dict = {"u_h": fh_data}

        mesh = meshio.Mesh(
            points=mesh_points,
            cells=cells_dict,
            # Optionally provide extra data on points, cells, etc.
            point_data=p_data_dict,
        )
        prefix = "lm"
        name = prefix + "_elasticity.vtk"
        mesh.write(name)
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

        # post-process solution in d-1 positive skins
        st = time.time()
        cellid_to_element = dict(zip(up_space.element_ids, up_space.elements))
        # writing solution on mesh points
        vertices = up_space.mesh_topology.entities_by_dimension(0)
        fh_data = np.zeros((len(gmesh.points), n_components))
        cell_vertex_map = up_space.mesh_topology.entity_map_by_dimension(0)
        for id in vertices:
            if not cell_vertex_map.has_node(id):
                continue

            pr_ids = list(cell_vertex_map.predecessors(id))
            cell = gmesh.cells[pr_ids[0]]
            if cell.dimension != up_space.dimension:
                continue

            element = cellid_to_element[pr_ids[0]]

            # scattering dof
            dest = up_space.dof_map.destination_indices(cell.id)
            alpha_l = alpha[dest]

            par_points = basix.geometry(up_space.element_type)

            target_node_id = gmesh.cells[id].node_tags[0]
            par_point_id = np.array(
                [
                    i
                    for i, node_id in enumerate(cell.node_tags)
                    if node_id == target_node_id
                ]
            )

            points = gmesh.points[target_node_id]
            if up_space.dimension != 0:
                points = par_points[par_point_id]

            # evaluate mapping
            phi_shapes = evaluate_linear_shapes(points, element.data)
            (x, jac, det_jac, inv_jac) = evaluate_mapping(
                cell.dimension, phi_shapes, gmesh.points[cell.node_tags]
            )
            phi_tab = element.evaluate_basis(points)
            n_phi = phi_tab.shape[2]
            alpha_star = np.array(np.split(alpha_l, n_phi))
            f_h = (phi_tab[0, :, :, 0] @ alpha_star).T
            fh_data[target_node_id] = f_h.ravel()

        mesh_points = gmesh.points
        con_d = np.array([element.data.cell.node_tags for element in up_space.elements])
        meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
        cells_dict = {meshio_cell_types[up_space.dimension]: con_d}
        p_data_dict = {"u_h": fh_data}

        mesh = meshio.Mesh(
            points=mesh_points,
            cells=cells_dict,
            # Optionally provide extra data on points, cells, etc.
            point_data=p_data_dict,
        )
        prefix = "lm_skin_p"
        name = prefix + "_elasticity.vtk"
        mesh.write(name)
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

        # post-process solution in d-1 negative skins
        st = time.time()
        cellid_to_element = dict(zip(um_space.element_ids, um_space.elements))
        # writing solution on mesh points
        vertices = um_space.mesh_topology.entities_by_dimension(0)
        fh_data = np.zeros((len(gmesh.points), n_components))
        cell_vertex_map = um_space.mesh_topology.entity_map_by_dimension(0)
        for id in vertices:
            if not cell_vertex_map.has_node(id):
                continue

            pr_ids = list(cell_vertex_map.predecessors(id))
            cell = gmesh.cells[pr_ids[0]]
            if cell.dimension != um_space.dimension:
                continue

            element = cellid_to_element[pr_ids[0]]

            # scattering dof
            dest = um_space.dof_map.destination_indices(cell.id)
            alpha_l = alpha[dest]

            par_points = basix.geometry(um_space.element_type)

            target_node_id = gmesh.cells[id].node_tags[0]
            par_point_id = np.array(
                [
                    i
                    for i, node_id in enumerate(cell.node_tags)
                    if node_id == target_node_id
                ]
            )

            points = gmesh.points[target_node_id]
            if um_space.dimension != 0:
                points = par_points[par_point_id]

            # evaluate mapping
            phi_shapes = evaluate_linear_shapes(points, element.data)
            (x, jac, det_jac, inv_jac) = evaluate_mapping(
                cell.dimension, phi_shapes, gmesh.points[cell.node_tags]
            )
            phi_tab = element.evaluate_basis(points)
            n_phi = phi_tab.shape[2]
            alpha_star = np.array(np.split(alpha_l, n_phi))
            f_h = (phi_tab[0, :, :, 0] @ alpha_star).T
            fh_data[target_node_id] = f_h.ravel()

        mesh_points = gmesh.points
        con_d = np.array([element.data.cell.node_tags for element in um_space.elements])
        meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
        cells_dict = {meshio_cell_types[um_space.dimension]: con_d}
        p_data_dict = {"u_h": fh_data}

        mesh = meshio.Mesh(
            points=mesh_points,
            cells=cells_dict,
            # Optionally provide extra data on points, cells, etc.
            point_data=p_data_dict,
        )
        prefix = "lm_skin_n"
        name = prefix + "_elasticity.vtk"
        mesh.write(name)
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

        # post-process solution in d-1 LM
        st = time.time()
        cellid_to_element = dict(zip(l_space.element_ids, l_space.elements))
        # writing solution on mesh points
        vertices = l_space.mesh_topology.entities_by_dimension(0)
        fh_data = np.zeros((len(gmesh.points), n_components))
        cell_vertex_map = l_space.mesh_topology.entity_map_by_dimension(0)
        for id in vertices:
            if not cell_vertex_map.has_node(id):
                continue

            pr_ids = list(cell_vertex_map.predecessors(id))
            cell = gmesh.cells[pr_ids[0]]
            if cell.dimension != l_space.dimension:
                continue

            element = cellid_to_element[pr_ids[0]]

            # scattering dof
            dest = l_space.dof_map.destination_indices(cell.id) + u_space.n_dof
            alpha_l = alpha[dest]

            par_points = basix.geometry(l_space.element_type)

            target_node_id = gmesh.cells[id].node_tags[0]
            par_point_id = np.array(
                [
                    i
                    for i, node_id in enumerate(cell.node_tags)
                    if node_id == target_node_id
                ]
            )

            points = gmesh.points[target_node_id]
            if l_space.dimension != 0:
                points = par_points[par_point_id]

            # evaluate mapping
            phi_shapes = evaluate_linear_shapes(points, element.data)
            (x, jac, det_jac, inv_jac) = evaluate_mapping(
                cell.dimension, phi_shapes, gmesh.points[cell.node_tags]
            )
            phi_tab = element.evaluate_basis(points)
            n_phi = phi_tab.shape[2]
            alpha_star = np.array(np.split(alpha_l, n_phi))
            f_h = (phi_tab[0, :, :, 0] @ alpha_star).T
            fh_data[target_node_id] = f_h.ravel()

        mesh_points = gmesh.points
        con_d = np.array([element.data.cell.node_tags for element in l_space.elements])
        meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
        cells_dict = {meshio_cell_types[l_space.dimension]: con_d}
        p_data_dict = {"l_h": fh_data}

        mesh = meshio.Mesh(
            points=mesh_points,
            cells=cells_dict,
            # Optionally provide extra data on points, cells, etc.
            point_data=p_data_dict,
        )
        prefix = "lm_lambda"
        name = prefix + "_elasticity.vtk"
        mesh.write(name)
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")


def lm_h1_cosserat_elasticity(k_order, gmesh, write_vtk_q=False):

    #
    fixed_point_q = False
    skin_stiffness_q = True
    dim = gmesh.dimension
    domain: Domain = gmesh.conformal_mesher.domain
    domain_tags = [1, 2, 3, 4, 5]
    frac_physical_tags = [
        shape.physical_tag
        for shape in domain.shapes[dim - 1]
        if (shape.physical_tag is not None) and (shape.physical_tag not in domain_tags)
    ]
    skin_p_physical_tags = [1000 * p_tag + 1 for p_tag in frac_physical_tags]
    skin_m_physical_tags = [1000 * p_tag - 1 for p_tag in frac_physical_tags]

    # Cosserat Material data
    m_lambda = 8.0e9
    m_mu = 8.0e9
    m_kappa = 0.5 * m_mu
    m_gamma = 1.0e8

    A_n = 0.0
    A_t = 0.0
    A_r = 0.0
    Kv = 1.33333e10
    Gv = m_mu
    s_n = -35.0e6

    hs = 1.0e-1
    m_s_lambda = hs * m_lambda
    m_s_mu = hs * m_mu
    m_s_kappa = hs * m_kappa
    m_s_gamma = hs * m_gamma

    # FESpace: data
    n_components = 3
    if dim == 3:
        n_components = 6

    discontinuous = True
    family = "Lagrange"

    u_space = DiscreteSpace(dim, n_components, family, k_order, gmesh)
    um_space = DiscreteSpace(dim - 1, n_components, family, k_order, gmesh)
    up_space = DiscreteSpace(dim - 1, n_components, family, k_order, gmesh)
    l_space = DiscreteSpace(
        dim - 1,
        n_components,
        family,
        k_order - 2,
        gmesh,
        integration_oder=2 * k_order + 1,
    )

    l_space.make_discontinuous()
    # u_space.build_structures([2, 3])
    if dim == 2:
        u_space.build_structures([2, 4])
    elif dim == 3:
        u_space.build_structures([7])
    up_space.build_structures_on_physical_tags(skin_p_physical_tags)
    um_space.build_structures_on_physical_tags(skin_m_physical_tags)
    l_space.build_structures_on_physical_tags(frac_physical_tags)

    # make entitie maps equal
    up_space.dof_map.vertex_map = u_space.dof_map.vertex_map
    um_space.dof_map.vertex_map = u_space.dof_map.vertex_map
    # l_space.dof_map.vertex_map = u_space.dof_map.vertex_map

    up_space.dof_map.edge_map = u_space.dof_map.edge_map
    um_space.dof_map.edge_map = u_space.dof_map.edge_map
    # l_space.dof_map.edge_map = u_space.dof_map.edge_map

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

    for element in up_space.elements:
        cell = element.data.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs) * n_components
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + 3 * n_dof * n_dof

    for element in um_space.elements:
        cell = element.data.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs) * n_components
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + 3 * n_dof * n_dof

    for element in l_space.elements:
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

    n_dof_g = u_space.dof_map.dof_number() + l_space.dof_map.dof_number()
    rg = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    et = time.time()
    elapsed_time = et - st
    print("Triplets creation time:", elapsed_time, "seconds")

    st = time.time()

    f_rhs = lambda x, y, z: np.array([0.0 * (1 - y), 0.0 * (1 - x), 0.0 * (1 - z)])
    s_load = lambda x, y, z: np.array([0.0 * x, s_n + 0.0 * y, 0.0 * z])
    if dim == 3:
        assert dim == 2
        f_rhs = lambda x, y, z: np.array([(1 - y), -(1 - x), 0.0 * z])

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
            u_space,
            cell_map,
            row,
            col,
            data,
        )
        for element in u_space.elements
    ]

    def scatter_K_skin_form_data(
        element,
        m_s_lambda,
        m_s_mu,
        m_s_kappa,
        m_s_gamma,
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
        block_sequ = (
            np.array(range(0, len(dest) * len(dest))) + c_sequ + 2 * n_dof * n_dof
        )
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

    if skin_stiffness_q:

        [
            scatter_K_skin_form_data(
                element,
                m_s_lambda,
                m_s_mu,
                m_s_kappa,
                m_s_gamma,
                f_rhs,
                up_space,
                cell_map,
                row,
                col,
                data,
            )
            for element in up_space.elements
        ]

        [
            scatter_K_skin_form_data(
                element,
                m_s_lambda,
                m_s_mu,
                m_s_kappa,
                m_s_gamma,
                f_rhs,
                um_space,
                cell_map,
                row,
                col,
                data,
            )
            for element in um_space.elements
        ]

    def scatter_skin_form_data(
        element, up_space, l_space, cell_map, row, col, data, sign
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

        # print("cell.sub_cells_ids: ", cell.sub_cells_ids)
        lm_cell_id = cell.sub_cells_ids[1][1]
        lm_cell_index = l_space.id_to_element[lm_cell_id]
        lm_el_data = l_space.elements[lm_cell_index].data

        lm_points = lm_el_data.quadrature.points
        lm_weights = lm_el_data.quadrature.weights
        lm_phi_tab = lm_el_data.basis.phi
        lm_n = lm_el_data.cell.normal[np.array([0, 1])]
        lm_t = np.array([lm_n[1], -lm_n[0]])

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
        dest_lm = l_space.dof_map.destination_indices(lm_cell_id) + u_space.n_dof

        n_phi = phi_tab.shape[2]
        n_phi_lm = lm_phi_tab.shape[2]
        n_dof = n_phi * n_components
        n_dof_lm = n_phi_lm * n_components
        js = (n_dof, n_dof_lm)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # local blocks
        jac_block = np.zeros((n_phi, n_phi_lm))
        for i, omega in enumerate(weights):
            phi_u = phi_tab[0, i, :, 0]
            phi_l = lm_phi_tab[0, i, :, 0]
            jac_block += sign * det_jac[i] * omega * np.outer(phi_u, phi_l)

        for c in range(n_components):
            b = c
            e = (c + 1) * n_phi * n_components + 1
            j_el[b:e:n_components, b:e:n_components] += jac_block

        # scattering data
        c_sequ = cell_map[cell.id]

        block_size = len(dest) * len(dest_lm)

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest_lm))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest_lm))
        col[block_sequ] += np.tile(dest_lm, len(dest))
        data[block_sequ] += j_el.ravel()

        # transpose contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest_lm))) + c_sequ + block_size
        row[block_sequ] += np.repeat(dest_lm, len(dest))
        col[block_sequ] += np.tile(dest, len(dest_lm))
        data[block_sequ] += j_el.T.ravel()

    [
        scatter_skin_form_data(element, u_space, l_space, cell_map, row, col, data, -1)
        for element in up_space.elements
    ]

    [
        scatter_skin_form_data(element, u_space, l_space, cell_map, row, col, data, +1)
        for element in um_space.elements
    ]

    def scatter_lambda_form_data(
        A_n, A_t, A_r, element, l_space, cell_map, row, col, data
    ):

        n_components = l_space.n_comp
        el_data: ElementData = element.data

        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # destination indexes
        dest = l_space.dof_map.destination_indices(cell.id) + u_space.n_dof

        n_phi = phi_tab.shape[2]
        n_dof = n_phi * n_components
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # local blocks
        jac_block = np.zeros((n_phi, n_phi))
        for i, omega in enumerate(weights):
            phi = phi_tab[0, i, :, 0]
            jac_block += det_jac[i] * omega * np.outer(phi, phi)

        A_data = [A_n, A_t, A_r]
        for c in range(n_components):
            b = c
            e = (c + 1) * n_phi * n_components + 1
            j_el[b:e:n_components, b:e:n_components] += A_data[c] * jac_block

        # scattering data
        c_sequ = cell_map[cell.id]

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

    [
        scatter_lambda_form_data(
            A_n, A_t, A_r, element, l_space, cell_map, row, col, data
        )
        for element in l_space.elements
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
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        scale = 1.0
        if cell.material_id == 2:
            scale = 0.0

        # Partial local vectorization
        f_val_star = scale * s_load(x[:, 0], x[:, 1], x[:, 2])
        phi_s_star = det_jac * weights * phi_tab[0, :, :, 0].T
        for c in range(n_components):
            b = c
            e = (c + 1) * n_phi * n_components + 1
            r_el[b:e:n_components] += phi_s_star @ f_val_star[c]

        # local blocks
        beta = 1.0e12
        jac_block = np.zeros((n_phi, n_phi))
        for i, omega in enumerate(weights):
            phi = phi_tab[0, i, :, 0]
            jac_block += beta * det_jac[i] * omega * np.outer(phi, phi)

        for c in range(n_components - 1):  # free rotations
            b = c
            e = (c + 1) * n_phi * n_components + 1
            j_el[b:e:n_components, b:e:n_components] += jac_block

        # scattering data
        c_sequ = cell_map[cell.id]

        # contribute rhs
        rg[dest] += r_el
        if cell.material_id == 4:
            return

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
    if fixed_point_q:
        alpha = sp_solver.spsolve(jg, rg)
        u_range = list(range(u_space.n_dof))
        l_range = list(range(u_space.n_dof, u_space.n_dof + l_space.n_dof))
        jg_uu = jg[u_range, :][:, u_range]
        jg_lu = jg[l_range, :][:, u_range]
        jg_ul = jg[u_range, :][:, l_range]
        rg_uu = rg[u_range]

        alpha_l = alpha[l_range]
        gt_p = np.zeros(int(len(l_range) / n_components))
        for k in range(3):
            L_k = np.array(np.split(alpha_l, n_components))
            Ln_k = L_k[0, :]
            Lt_k = L_k[1, :]

            # step 1: compute u jump
            alpha_u = sp_solver.spsolve(jg_uu, rg_uu - jg_ul * alpha_l)
            u_jump = np.array(np.split(jg_lu * alpha_u, n_components))

            # step 2: For all LM
            n = np.empty((0, 2), dtype=float)
            t = np.empty((0, 2), dtype=float)
            for element in l_space.elements:
                el_data = element.data
                ln = el_data.cell.normal[np.array([0, 1])]
                lt = np.array([ln[1], -ln[0]])
                n = np.vstack((n, ln))
                t = np.vstack((t, lt))

            c_fric = 0.0
            theta = 20.0 * np.pi / 180.0
            gn = np.sum(u_jump[0:1, :] * n.T, axis=0)
            gt = np.sum(u_jump[0:1, :] * t.T, axis=0)

            # step one predict:
            gn0 = 0.0001
            # Tn = Ln_k - gn * (Kv * gn0) / (gn0-gn)
            Tn = Ln_k - Kv * gn
            Tt_trial = Lt_k - Gv * gt
            Phi = c_fric - Tn * np.tan(theta)
            elastic_q = Tt_trial - Phi <= 0.0
            Tt = np.where(elastic_q, Tt_trial, Phi)
            print("gn Tn : ", np.linalg.norm((gn * Tn)))
            print("norm in LM: ", np.linalg.norm((Ln_k - Tn)))
            L_k[0, :] = Tn
            L_k[1, :] = Tt
            alpha_l = L_k.ravel()
            if np.all(elastic_q):
                break
        alpha = np.concatenate((alpha_u, alpha_l))

    else:
        alpha = sp_solver.spsolve(jg, rg)
    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    if write_vtk_q:
        # post-process solution in d
        st = time.time()
        cellid_to_element = dict(zip(u_space.element_ids, u_space.elements))
        # writing solution on mesh points
        vertices = u_space.mesh_topology.entities_by_dimension(0)
        fh_data = np.zeros((len(gmesh.points), n_components))
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
            f_h = (phi_tab[0, :, :, 0] @ alpha_star).T
            fh_data[target_node_id] = f_h.ravel()

        mesh_points = gmesh.points
        con_d = np.array([element.data.cell.node_tags for element in u_space.elements])
        meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
        cells_dict = {meshio_cell_types[u_space.dimension]: con_d}
        p_data_dict = {"u_h": fh_data}

        mesh = meshio.Mesh(
            points=mesh_points,
            cells=cells_dict,
            # Optionally provide extra data on points, cells, etc.
            point_data=p_data_dict,
        )
        prefix = "lm"
        name = prefix + "_elasticity.vtk"
        mesh.write(name)
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

        # post-process solution in d-1 positive skins
        st = time.time()
        cellid_to_element = dict(zip(up_space.element_ids, up_space.elements))
        # writing solution on mesh points
        vertices = up_space.mesh_topology.entities_by_dimension(0)
        fh_data = np.zeros((len(gmesh.points), n_components))
        cell_vertex_map = up_space.mesh_topology.entity_map_by_dimension(0)
        for id in vertices:
            if not cell_vertex_map.has_node(id):
                continue

            pr_ids = list(cell_vertex_map.predecessors(id))
            cell = gmesh.cells[pr_ids[0]]
            if cell.dimension != up_space.dimension:
                continue

            element = cellid_to_element[pr_ids[0]]

            # scattering dof
            dest = up_space.dof_map.destination_indices(cell.id)
            alpha_l = alpha[dest]

            par_points = basix.geometry(up_space.element_type)

            target_node_id = gmesh.cells[id].node_tags[0]
            par_point_id = np.array(
                [
                    i
                    for i, node_id in enumerate(cell.node_tags)
                    if node_id == target_node_id
                ]
            )

            points = gmesh.points[target_node_id]
            if up_space.dimension != 0:
                points = par_points[par_point_id]

            # evaluate mapping
            phi_shapes = evaluate_linear_shapes(points, element.data)
            (x, jac, det_jac, inv_jac) = evaluate_mapping(
                cell.dimension, phi_shapes, gmesh.points[cell.node_tags]
            )
            phi_tab = element.evaluate_basis(points)
            n_phi = phi_tab.shape[2]
            alpha_star = np.array(np.split(alpha_l, n_phi))
            f_h = (phi_tab[0, :, :, 0] @ alpha_star).T
            fh_data[target_node_id] = f_h.ravel()

        mesh_points = gmesh.points
        con_d = np.array([element.data.cell.node_tags for element in up_space.elements])
        meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
        cells_dict = {meshio_cell_types[up_space.dimension]: con_d}
        p_data_dict = {"u_h": fh_data}

        mesh = meshio.Mesh(
            points=mesh_points,
            cells=cells_dict,
            # Optionally provide extra data on points, cells, etc.
            point_data=p_data_dict,
        )
        prefix = "lm_skin_p"
        name = prefix + "_elasticity.vtk"
        mesh.write(name)
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

        # post-process solution in d-1 negative skins
        st = time.time()
        cellid_to_element = dict(zip(um_space.element_ids, um_space.elements))
        # writing solution on mesh points
        vertices = um_space.mesh_topology.entities_by_dimension(0)
        fh_data = np.zeros((len(gmesh.points), n_components))
        cell_vertex_map = um_space.mesh_topology.entity_map_by_dimension(0)
        for id in vertices:
            if not cell_vertex_map.has_node(id):
                continue

            pr_ids = list(cell_vertex_map.predecessors(id))
            cell = gmesh.cells[pr_ids[0]]
            if cell.dimension != um_space.dimension:
                continue

            element = cellid_to_element[pr_ids[0]]

            # scattering dof
            dest = um_space.dof_map.destination_indices(cell.id)
            alpha_l = alpha[dest]

            par_points = basix.geometry(um_space.element_type)

            target_node_id = gmesh.cells[id].node_tags[0]
            par_point_id = np.array(
                [
                    i
                    for i, node_id in enumerate(cell.node_tags)
                    if node_id == target_node_id
                ]
            )

            points = gmesh.points[target_node_id]
            if um_space.dimension != 0:
                points = par_points[par_point_id]

            # evaluate mapping
            phi_shapes = evaluate_linear_shapes(points, element.data)
            (x, jac, det_jac, inv_jac) = evaluate_mapping(
                cell.dimension, phi_shapes, gmesh.points[cell.node_tags]
            )
            phi_tab = element.evaluate_basis(points)
            n_phi = phi_tab.shape[2]
            alpha_star = np.array(np.split(alpha_l, n_phi))
            f_h = (phi_tab[0, :, :, 0] @ alpha_star).T
            fh_data[target_node_id] = f_h.ravel()

        mesh_points = gmesh.points
        con_d = np.array([element.data.cell.node_tags for element in um_space.elements])
        meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
        cells_dict = {meshio_cell_types[um_space.dimension]: con_d}
        p_data_dict = {"u_h": fh_data}

        mesh = meshio.Mesh(
            points=mesh_points,
            cells=cells_dict,
            # Optionally provide extra data on points, cells, etc.
            point_data=p_data_dict,
        )
        prefix = "lm_skin_n"
        name = prefix + "_elasticity.vtk"
        mesh.write(name)
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

        # post-process solution in d-1 LM
        st = time.time()
        cellid_to_element = dict(zip(l_space.element_ids, l_space.elements))
        # writing solution on mesh points
        vertices = l_space.mesh_topology.entities_by_dimension(0)
        fh_data = np.zeros((len(gmesh.points), n_components))
        cell_vertex_map = l_space.mesh_topology.entity_map_by_dimension(0)
        for id in vertices:
            if not cell_vertex_map.has_node(id):
                continue

            pr_ids = list(cell_vertex_map.predecessors(id))
            cell = gmesh.cells[pr_ids[0]]
            if cell.dimension != l_space.dimension:
                continue

            element = cellid_to_element[pr_ids[0]]

            # scattering dof
            dest = l_space.dof_map.destination_indices(cell.id) + u_space.n_dof
            alpha_l = alpha[dest]

            par_points = basix.geometry(l_space.element_type)

            target_node_id = gmesh.cells[id].node_tags[0]
            par_point_id = np.array(
                [
                    i
                    for i, node_id in enumerate(cell.node_tags)
                    if node_id == target_node_id
                ]
            )

            points = gmesh.points[target_node_id]
            if l_space.dimension != 0:
                points = par_points[par_point_id]

            # evaluate mapping
            phi_shapes = evaluate_linear_shapes(points, element.data)
            (x, jac, det_jac, inv_jac) = evaluate_mapping(
                cell.dimension, phi_shapes, gmesh.points[cell.node_tags]
            )
            phi_tab = element.evaluate_basis(points)
            n_phi = phi_tab.shape[2]
            alpha_star = np.array(np.split(alpha_l, n_phi))
            f_h = (phi_tab[0, :, :, 0] @ alpha_star).T
            fh_data[target_node_id] = f_h.ravel()

        mesh_points = gmesh.points
        con_d = np.array([element.data.cell.node_tags for element in l_space.elements])
        meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
        cells_dict = {meshio_cell_types[l_space.dimension]: con_d}
        p_data_dict = {"l_h": fh_data}

        mesh = meshio.Mesh(
            points=mesh_points,
            cells=cells_dict,
            # Optionally provide extra data on points, cells, etc.
            point_data=p_data_dict,
        )
        prefix = "lm_lambda"
        name = prefix + "_elasticity.vtk"
        mesh.write(name)
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")


def create_md_domain(dimension):

    if dimension == 1:
        box_points = np.array([[0, 0, 0], [1, 0, 0]])
        domain = build_box_1D(box_points)
        return domain
    elif dimension == 2:
        lines_file = "fracture_files/setting_2d_single_fracture.csv"
        box_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        box_points = 2.0 * np.array([[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]])
        domain = build_box_2D_with_lines(box_points, lines_file)
        return domain
    else:
        h_thickness = 1.0
        box_points = np.array(
            [
                [0.0, 0.0, -h_thickness / 2],
                [1.0, 0.0, -h_thickness / 2],
                [1.0, 1.0, -h_thickness / 2],
                [0.0, 1.0, -h_thickness / 2],
                [0.0, 0.0, +h_thickness / 2],
                [1.0, 0.0, +h_thickness / 2],
                [1.0, 1.0, +h_thickness / 2],
                [0.0, 1.0, +h_thickness / 2],
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
    map_fracs_edge = gmesh.cut_conformity_on_embed_shapes()
    # check_q = gmesh.circulate_internal_bc_from_domain()
    # assert check_q[0]
    if write_vtk_q:
        gmesh.write_vtk()
        gmesh.write_data()
    return gmesh


def main():

    dimension = 2
    k_order = 2
    h = 1.0
    l = 0

    domain = create_md_domain(dimension)
    mesher = create_conformal_mesher(domain, h, l)
    gmesh = create_mesh(dimension, mesher, True)

    # lm_h1_elasticity(k_order, gmesh, True)
    lm_h1_cosserat_elasticity(k_order, gmesh, True)
    # h1_cosserat_elasticity(k_order, gmesh, False)
    return


if __name__ == "__main__":
    main()
