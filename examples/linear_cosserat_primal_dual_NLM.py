import copy
import csv
import functools
import marshal
import sys
import time
# from itertools import permutations
from functools import partial, reduce

import auto_diff as ad
import basix
import matplotlib.colors as mcolors
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plot
import meshio
import networkx as nx
import numpy as np
import psutil
import pypardiso as sp_solver
import scipy.sparse as sp
import strong_solution_cosserat_elasticity_example_1 as lce
from auto_diff.vecvalder import VecValDer
from basix import CellType
from numpy import linalg as la
from petsc4py import PETSc
from scipy.sparse import coo_matrix
from shapely.geometry import LineString

import geometry.fracture_network as fn
from basis.element_data import ElementData
from basis.finite_element import FiniteElement
from geometry.domain import Domain
from geometry.domain_market import (build_box_1D, build_box_2D,
                                    build_box_2D_with_lines, build_box_3D,
                                    build_box_3D_with_planes,
                                    build_disjoint_lines, read_fractures_file)
from geometry.edge import Edge
from geometry.geometry_builder import GeometryBuilder
from geometry.geometry_cell import GeometryCell
from geometry.mapping import (evaluate_linear_shapes, evaluate_mapping,
                              store_mapping)
from geometry.shape_manipulation import ShapeManipulation
from geometry.vertex import Vertex
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh
from spaces.discrete_space import DiscreteSpace
from spaces.dof_map import DoFMap
from topology.mesh_topology import MeshTopology

num_cpus = psutil.cpu_count(logical=False)


def matrix_plot(J, sparse_q=True):
    if sparse_q:
        plot.matshow(J.todense())
    else:
        plot.matshow(J)
    plot.colorbar(orientation="vertical")
    plot.set_cmap("seismic")
    plot.show()


def create_md_domain(dimension):
    if dimension == 1:
        box_points = np.array([[0, 0, 0], [1, 0, 0]])
        domain = build_box_1D(box_points)
        return domain
    elif dimension == 2:
        lines_file = "fracture_files/setting_2d_single_fracture.csv"
        box_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        box_points = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
        # domain = build_box_2D_with_lines(box_points, lines_file)
        domain = build_box_2D(box_points)
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
    # map_fracs_edge = gmesh.cut_conformity_on_embed_shapes()
    # check_q = gmesh.circulate_internal_bc_from_domain()
    # assert check_q[0]
    if write_vtk_q:
        gmesh.write_vtk()
        gmesh.write_data()
    return gmesh


def h1_cosserat_elasticity(k_order, gmesh, write_vtk_q=False):
    dim = gmesh.dimension
    # Material data

    sigma_0 = -1.0
    h_bc_ids = [2, 4]
    v_bc_ids = [3, 5]

    m_lambda = 1.0
    m_mu = 1.0
    m_kappa = 1.0
    m_gamma = 1.0

    # FESpace: data
    u_components = 2
    t_components = 1
    if dim == 3:
        u_components = 3
        t_components = 3

    family = "Lagrange"

    u_k_order = k_order + 1
    u_space = DiscreteSpace(
        dim, u_components, family, u_k_order, gmesh, integration_order=2 * u_k_order + 1
    )
    t_space = DiscreteSpace(
        dim, t_components, family, k_order, gmesh, integration_order=2 * u_k_order + 1
    )
    if dim == 2:
        u_space.build_structures([2, 3, 4, 5])
        t_space.build_structures([2, 3, 4, 5])
    elif dim == 3:
        u_space.build_structures([2, 3, 4, 5, 6, 7])
        t_space.build_structures([2, 3, 4, 5, 6, 7])

    u_n_els = len(u_space.elements)
    t_n_els = len(t_space.elements)
    assert u_n_els == t_n_els

    st = time.time()
    # Assembler

    u_n_dof_g = u_space.dof_map.dof_number()
    t_n_dof_g = t_space.dof_map.dof_number()
    n_dof_g = u_n_dof_g + t_n_dof_g
    rg = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    A = PETSc.Mat()
    A.createAIJ([n_dof_g, n_dof_g])

    et = time.time()
    elapsed_time = et - st
    print("Triplets creation time:", elapsed_time, "seconds")

    st = time.time()

    # exact solution
    u_exact = lce.generalized_displacement(m_lambda, m_mu, m_kappa, m_gamma, dim)
    s_exact = lce.stress(m_lambda, m_mu, m_kappa, m_gamma, dim)
    m_exact = lce.couple_stress(m_lambda, m_mu, m_kappa, m_gamma, dim)
    f_rhs = lce.rhs(m_lambda, m_mu, m_kappa, m_gamma, dim)

    def scatter_form_data_ad(
        A, i, m_lambda, m_mu, m_kappa, m_gamma, f_rhs, u_space, t_space
    ):
        u_components = u_space.n_comp
        t_components = t_space.n_comp

        u_data: ElementData = u_space.elements[i].data
        t_data: ElementData = t_space.elements[i].data

        cell = u_data.cell
        points = u_data.quadrature.points
        weights = u_data.quadrature.weights
        x = u_data.mapping.x
        det_jac = u_data.mapping.det_jac
        inv_jac = u_data.mapping.inv_jac

        u_phi_tab = u_data.basis.phi
        t_phi_tab = t_data.basis.phi

        # destination indexes
        dest_u = u_space.dof_map.destination_indices(cell.id)
        dest_t = t_space.dof_map.destination_indices(cell.id) + u_n_dof_g
        dest = np.concatenate([dest_u, dest_t])

        n_u_phi = u_phi_tab.shape[2]
        n_t_phi = t_phi_tab.shape[2]

        n_u_dof = n_u_phi * u_components
        n_t_dof = n_t_phi * t_components

        n_dof = n_u_dof + n_t_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)
        alpha = np.zeros(n_dof)

        # Partial local vectorization
        f_val_star = 0.0 * f_rhs(x[:, 0], x[:, 1], x[:, 2])

        # constant directors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        Imat = np.identity(dim)
        with ad.AutoDiff(alpha) as alpha:
            el_form = np.zeros(n_dof)
            for c in range(u_components):
                b = c
                e = b + n_u_dof
                el_form[b:e:u_components] += (
                    det_jac * weights * u_phi_tab[0, :, :, 0].T @ f_val_star[c]
                )
            for c in range(t_components):
                b = c + n_u_dof
                e = b + n_t_dof
                el_form[b:e:t_components] += (
                    det_jac
                    * weights
                    * t_phi_tab[0, :, :, 0].T
                    @ f_val_star[c + u_components]
                )

            for i, omega in enumerate(weights):
                if dim == 2:
                    inv_jac_m = np.vstack((inv_jac[i] @ e1, inv_jac[i] @ e2))
                    grad_phi_u = (
                        inv_jac_m @ u_phi_tab[1 : u_phi_tab.shape[0] + 1, i, :, 0]
                    ).T
                    grad_phi_t = (
                        inv_jac_m @ t_phi_tab[1 : t_phi_tab.shape[0] + 1, i, :, 0]
                    ).T

                    c = 0
                    a_ux = alpha[:, c : n_u_dof + c : u_components]
                    c = 1
                    a_uy = alpha[:, c : n_u_dof + c : u_components]
                    a_t = alpha[:, n_u_dof : n_t_dof + n_u_dof : t_components]

                    ux_h = a_ux @ u_phi_tab[0, i, :, 0:dim]
                    uy_h = a_uy @ u_phi_tab[0, i, :, 0:dim]
                    grad_uh_x = a_ux @ grad_phi_u
                    grad_uh_y = a_uy @ grad_phi_u
                    grad_uh = VecValDer(
                        np.vstack((grad_uh_x.val, grad_uh_y.val)),
                        np.vstack((grad_uh_x.der, grad_uh_y.der)),
                    )

                    grad_th = a_t @ grad_phi_t
                    th = a_t @ t_phi_tab[0, i, :, 0:t_components]

                    Theta_outer = th * np.array([[0.0, -1.0], [1.0, 0.0]])
                    eh = grad_uh + Theta_outer
                    # Stress decomposition
                    Symm_eh = 0.5 * (eh + eh.T)
                    Skew_eh = 0.5 * (eh - eh.T)
                    tr_eh = VecValDer(eh.val.trace(), eh.der.trace())
                    sh = (
                        2.0 * m_mu * Symm_eh
                        + 2.0 * m_kappa * Skew_eh
                        + m_lambda * tr_eh * Imat
                    )

                    Skew_sh = 0.5 * (sh - sh.T)
                    S_cross = np.array([[Skew_sh[1, 0] - Skew_sh[0, 1]]])
                    k = m_gamma * grad_th
                    strain_energy_h = (grad_phi_u @ sh.T).reshape((n_u_dof,))
                    curvature_energy_h = (
                        grad_phi_t @ k.T + t_phi_tab[0, i, :, 0:t_components] @ S_cross
                    ).reshape((n_t_dof,))

                else:
                    inv_jac_m = np.vstack(
                        (inv_jac[i] @ e1, inv_jac[i] @ e2, inv_jac[i] @ e3)
                    )
                    grad_phi_u = (
                        inv_jac_m @ u_phi_tab[1 : u_phi_tab.shape[0] + 1, i, :, 0]
                    ).T
                    grad_phi_t = (
                        inv_jac_m @ t_phi_tab[1 : t_phi_tab.shape[0] + 1, i, :, 0]
                    ).T

                    c = 0
                    a_ux = alpha[:, c : n_u_dof + c : u_components]
                    a_tx = alpha[:, c + n_u_dof : n_u_dof + n_t_dof + c : t_components]
                    c = 1
                    a_uy = alpha[:, c : n_u_dof + c : u_components]
                    a_ty = alpha[:, c + n_u_dof : n_u_dof + n_t_dof + c : t_components]
                    c = 2
                    a_uz = alpha[:, c : n_u_dof + c : u_components]
                    a_tz = alpha[:, c + n_u_dof : n_u_dof + n_t_dof + c : t_components]

                    ux_h = a_ux @ u_phi_tab[0, i, :, 0:dim]
                    uy_h = a_uy @ u_phi_tab[0, i, :, 0:dim]
                    uz_h = a_uz @ u_phi_tab[0, i, :, 0:dim]
                    grad_uh_x = a_ux @ grad_phi_u
                    grad_uh_y = a_uy @ grad_phi_u
                    grad_uh_z = a_uz @ grad_phi_u
                    grad_uh = VecValDer(
                        np.vstack((grad_uh_x.val, grad_uh_y.val, grad_uh_z.val)),
                        np.vstack((grad_uh_x.der, grad_uh_y.der, grad_uh_z.der)),
                    )

                    tx_h = a_tx @ t_phi_tab[0, i, :, 0:dim]
                    ty_h = a_ty @ t_phi_tab[0, i, :, 0:dim]
                    tz_h = a_tz @ t_phi_tab[0, i, :, 0:dim]
                    grad_th_x = a_tx @ grad_phi_t
                    grad_th_y = a_ty @ grad_phi_t
                    grad_th_z = a_tz @ grad_phi_t
                    grad_th = VecValDer(
                        np.vstack((grad_th_x.val, grad_th_y.val, grad_th_z.val)),
                        np.vstack((grad_th_x.der, grad_th_y.der, grad_th_z.der)),
                    )

                    uh = VecValDer(
                        np.hstack((ux_h.val, uy_h.val, uz_h.val)),
                        np.hstack((ux_h.der, uy_h.der, uz_h.der)),
                    )

                    th = VecValDer(
                        np.hstack((tx_h.val, ty_h.val, tz_h.val)),
                        np.hstack((tx_h.der, ty_h.der, tz_h.der)),
                    )

                    Theta_outer = np.array(
                        [
                            [0.0 * th[0, 0], -th[0, 2], +th[0, 1]],
                            [+th[0, 2], 0.0 * th[0, 0], -th[0, 0]],
                            [-th[0, 1], +th[0, 0], 0.0 * th[0, 0]],
                        ]
                    )

                    eh = grad_uh + Theta_outer
                    # Stress decomposition
                    Symm_eh = 0.5 * (eh + eh.T)
                    Skew_eh = 0.5 * (eh - eh.T)
                    tr_eh = VecValDer(eh.val.trace(), eh.der.trace())
                    sh = (
                        2.0 * m_mu * Symm_eh
                        + 2.0 * m_kappa * Skew_eh
                        + m_lambda * tr_eh * Imat
                    )

                    Skew_sh = 0.5 * (sh - sh.T)
                    S_cross = np.array(
                        [
                            [
                                Skew_sh[2, 1] - Skew_sh[1, 2],
                                Skew_sh[0, 2] - Skew_sh[2, 0],
                                Skew_sh[1, 0] - Skew_sh[0, 1],
                            ]
                        ]
                    )

                    k = m_gamma * grad_th
                    strain_energy_h = (grad_phi_u @ sh.T).reshape((n_u_dof,))
                    curvature_energy_h = (
                        grad_phi_t @ k.T + t_phi_tab[0, i, :, 0:t_components] @ S_cross
                    ).reshape((n_t_dof,))

                multiphysic_integrand = np.zeros((1, n_dof))
                multiphysic_integrand[:, 0:n_u_dof:1] = strain_energy_h
                multiphysic_integrand[
                    :, n_u_dof : n_u_dof + n_t_dof : 1
                ] = curvature_energy_h

                discrete_integrand = (multiphysic_integrand).reshape((n_dof,))
                el_form += det_jac[i] * omega * discrete_integrand

        r_el, j_el = el_form.val, el_form.der.reshape((n_dof, n_dof))

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        for k in range(nnz):
            A.setValue(row=row[k], col=col[k], value=data[k], addv=True)

    [
        scatter_form_data_ad(
            A, i, m_lambda, m_mu, m_kappa, m_gamma, f_rhs, u_space, t_space
        )
        for i in range(u_n_els)
    ]

    def scatter_bc_form_data_ad(A, i, u_space, t_space):
        u_components = u_space.n_comp
        t_components = t_space.n_comp

        u_data: ElementData = u_space.bc_elements[i].data
        t_data: ElementData = t_space.bc_elements[i].data

        cell = u_data.cell
        points = u_data.quadrature.points
        weights = u_data.quadrature.weights
        x = u_data.mapping.x
        det_jac = u_data.mapping.det_jac
        inv_jac = u_data.mapping.inv_jac

        u_phi_tab = u_data.basis.phi
        t_phi_tab = t_data.basis.phi

        # destination indexes
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
        dest_u = u_space.dof_map.bc_destination_indices(neigh_cell_id, cell.id)

        # find high-dimension neigh
        entity_map = t_space.dof_map.mesh_topology.entity_map_by_dimension(
            cell.dimension
        )
        neigh_list = list(entity_map.predecessors(cell.id))
        neigh_check_q = len(neigh_list) > 0
        assert neigh_check_q

        neigh_cell_id = neigh_list[0]
        neigh_cell_index = t_space.id_to_element[neigh_cell_id]
        neigh_cell = t_space.elements[neigh_cell_index].data.cell

        # destination indexes
        dest_neigh = t_space.dof_map.destination_indices(neigh_cell_id)
        dest_t = (
            t_space.dof_map.bc_destination_indices(neigh_cell_id, cell.id) + u_n_dof_g
        )

        dest = np.concatenate([dest_u, dest_t])

        n_u_phi = u_phi_tab.shape[2]
        n_t_phi = t_phi_tab.shape[2]

        n_u_dof = n_u_phi * u_components
        n_t_dof = n_t_phi * t_components

        n_dof = n_u_dof + n_t_dof
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # local blocks
        beta = 1.0e12
        rhs_block_u = np.zeros(n_u_phi)
        jac_block_u = np.zeros((n_u_phi, n_u_phi))
        for i, omega in enumerate(weights):
            phi = u_phi_tab[0, i, :, 0]
            jac_block_u += beta * det_jac[i] * omega * np.outer(phi, phi)

        rhs_block_t = np.zeros(n_t_phi)
        jac_block_t = np.zeros((n_t_phi, n_t_phi))
        for i, omega in enumerate(weights):
            phi = t_phi_tab[0, i, :, 0]
            jac_block_t += beta * det_jac[i] * omega * np.outer(phi, phi)

        # if cell.material_id in v_bc_ids:
        #     u_components_list = [0]
        #     for c in u_components_list:
        #         b = c
        #         e = b + n_u_dof
        #         j_el[b:e:u_components, b:e:u_components] += jac_block_u
        #
        #     for c in range(t_components):
        #         b = c + n_u_dof
        #         e = b + n_t_dof
        #         j_el[b:e:t_components, b:e:t_components] += jac_block_t

        if cell.material_id in h_bc_ids:
            if cell.material_id == 2:
                for c in range(u_components):
                    b = c
                    e = b + n_u_dof
                    j_el[b:e:u_components, b:e:u_components] += jac_block_u

                for c in range(t_components):
                    b = c + n_u_dof
                    e = b + n_t_dof
                    j_el[b:e:t_components, b:e:t_components] += jac_block_t
            else:
                for i, omega in enumerate(weights):
                    phi = u_phi_tab[0, i, :, 0]
                    rhs_block_u += det_jac[i] * omega * sigma_0 * phi

                u_components_list = [1]
                for c in u_components_list:
                    b = c
                    e = b + n_u_dof
                    r_el[b:e:u_components] += rhs_block_u

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        for k in range(nnz):
            A.setValue(row=row[k], col=col[k], value=data[k], addv=True)

    [
        scatter_bc_form_data_ad(A, i, u_space, t_space)
        for i in range(len(u_space.bc_elements))
    ]

    # jg = coo_matrix((data, (row, col)), shape=(n_dof_g, n_dof_g)).tocsr()
    et = time.time()
    elapsed_time = et - st
    print("Assembly time:", elapsed_time, "seconds")

    # solving ls
    A.assemble()

    st = time.time()
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    b = A.createVecLeft()
    b.array[:] = -rg
    x = A.createVecRight()

    petsc_options = {"rtol": 1e-12, "atol": 1e-14}
    ksp = PETSc.KSP().create()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("fgmres")
    ksp.setTolerances(**petsc_options)
    ksp.setConvergenceHistory()
    ksp.getPC().setType("ilu")
    ksp.solve(b, x)
    alpha = x.array

    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    # Computing displacement L2 error
    def compute_u_l2_error(i, u_space, dim):
        l2_error = 0.0
        n_components = u_space.n_comp
        el_data = u_space.elements[i].data
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
    def compute_t_l2_error(i, t_space, dim):
        l2_error = 0.0
        n_components = t_space.n_comp
        el_data = t_space.elements[i].data
        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # scattering dof
        dest = t_space.dof_map.destination_indices(cell.id) + u_n_dof_g
        alpha_l = alpha[dest]

        # for each integration point
        n_phi = phi_tab.shape[2]
        alpha_star = np.array(np.split(alpha_l, n_phi))
        t_e_s = u_exact(x[:, 0], x[:, 1], x[:, 2])[dim : u_components + n_components, :]
        t_h_s = (phi_tab[0, :, :, 0] @ alpha_star[:, 0:n_components]).T
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
    def compute_s_l2_error(i, u_space, t_space, m_mu, m_lambda, m_kappa, dim):
        l2_error = 0.0
        u_components = u_space.n_comp
        t_components = t_space.n_comp
        u_data: ElementData = u_space.elements[i].data
        t_data: ElementData = t_space.elements[i].data

        cell = u_data.cell
        points = u_data.quadrature.points
        weights = u_data.quadrature.weights
        phi_tab = u_data.basis.phi

        x = u_data.mapping.x
        det_jac = u_data.mapping.det_jac
        inv_jac = u_data.mapping.inv_jac

        u_phi_tab = u_data.basis.phi
        t_phi_tab = t_data.basis.phi
        n_u_phi = u_phi_tab.shape[2]
        n_t_phi = t_phi_tab.shape[2]

        n_u_dof = n_u_phi * u_components
        n_t_dof = n_t_phi * t_components

        # destination indexes
        dest_u = u_space.dof_map.destination_indices(cell.id)
        dest_t = t_space.dof_map.destination_indices(cell.id) + u_n_dof_g
        # dest = np.concatenate([dest_u, dest_t])
        alpha_u_l = alpha[dest_u]
        alpha_t_l = alpha[dest_t]

        # for each integration point

        alpha_star_u = np.array(np.split(alpha_u_l, n_u_phi))
        alpha_star_t = np.array(np.split(alpha_t_l, n_t_phi))
        t_h_s = (t_phi_tab[0, :, :, 0] @ alpha_star_t[:, 0:t_components]).T
        for i, omega in enumerate(weights):
            s_e = s_exact(x[i, 0], x[i, 1], x[i, 2])
            grad_phi = inv_jac[i].T @ u_phi_tab[1 : u_phi_tab.shape[0] + 1, i, :, 0]
            grad_uh = grad_phi[0:dim] @ alpha_star_u[:, 0:dim]
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
                + m_lambda * eps_h.trace() * np.identity(dim)
            )
            diff_s = s_e - s_h
            l2_error += det_jac[i] * weights[i] * np.trace(diff_s.T @ diff_s)
        return l2_error

    # Computing couple stress L2 error
    def compute_m_l2_error(i, t_space, m_gamma, dim):
        l2_error = 0.0
        n_components = t_space.n_comp
        el_data = t_space.elements[i].data
        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # scattering dof
        dest = t_space.dof_map.destination_indices(cell.id) + u_n_dof_g
        alpha_l = alpha[dest]

        # for each integration point
        n_phi = phi_tab.shape[2]
        alpha_star = np.array(np.split(alpha_l, n_phi))
        for i, omega in enumerate(weights):
            m_e = m_exact(x[i, 0], x[i, 1], x[i, 2])
            grad_phi = inv_jac[i].T @ phi_tab[1 : phi_tab.shape[0] + 1, i, :, 0]
            grad_th = grad_phi[0:dim] @ alpha_star[:, 0:n_components]
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
    u_error_vec = [compute_u_l2_error(i, u_space, dim) for i in range(u_n_els)]
    t_error_vec = [compute_t_l2_error(i, t_space, dim) for i in range(t_n_els)]
    s_error_vec = [
        compute_s_l2_error(i, u_space, t_space, m_mu, m_lambda, m_kappa, dim)
        for i in range(u_n_els)
    ]
    m_error_vec = [compute_m_l2_error(i, t_space, m_gamma, dim) for i in range(t_n_els)]
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

        # generalized displacements
        cellid_to_u_element = dict(zip(u_space.element_ids, u_space.elements))
        cellid_to_t_element = dict(zip(t_space.element_ids, t_space.elements))
        vertices = u_space.mesh_topology.entities_by_dimension(0)
        cell_vertex_map = u_space.mesh_topology.entity_map_by_dimension(0)
        for id in vertices:
            if not cell_vertex_map.has_node(id):
                continue

            pr_ids = list(cell_vertex_map.predecessors(id))
            cell = gmesh.cells[pr_ids[0]]
            if cell.dimension != u_space.dimension:
                continue

            u_element = cellid_to_u_element[pr_ids[0]]
            t_element = cellid_to_t_element[pr_ids[0]]

            # scattering dof
            dest_u = u_space.dof_map.destination_indices(cell.id)
            dest_t = t_space.dof_map.destination_indices(cell.id) + u_n_dof_g
            alpha_u_l = alpha[dest_u]
            alpha_t_l = alpha[dest_t]

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
            phi_shapes = evaluate_linear_shapes(points, u_element.data)
            (x, jac, det_jac, inv_jac) = evaluate_mapping(
                cell.dimension, phi_shapes, gmesh.points[cell.node_tags]
            )
            u_phi_tab = u_element.evaluate_basis(points)
            t_phi_tab = t_element.evaluate_basis(points)
            n_u_phi = u_phi_tab.shape[2]
            n_t_phi = t_phi_tab.shape[2]

            alpha_star_u = np.array(np.split(alpha_u_l, n_u_phi))
            alpha_star_t = np.array(np.split(alpha_t_l, n_t_phi))

            # Generalized displacement
            u_e = u_exact(x[:, 0], x[:, 1], x[:, 2])[0:dim, :]
            t_e = u_exact(x[:, 0], x[:, 1], x[:, 2])[
                dim : u_components + t_components, :
            ]
            u_h = (u_phi_tab[0, :, :, 0] @ alpha_star_u[:, 0:dim]).T
            t_h = (t_phi_tab[0, :, :, 0] @ alpha_star_t[:, 0:t_components]).T

            # stress and couple stress
            i = 0
            s_e = s_exact(x[i, 0], x[i, 1], x[i, 2])
            grad_phi = inv_jac[i].T @ u_phi_tab[1 : u_phi_tab.shape[0] + 1, i, :, 0]
            grad_uh = grad_phi[0:dim] @ alpha_star_u[:, 0:dim]
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
            grad_phi = inv_jac[i].T @ t_phi_tab[1 : t_phi_tab.shape[0] + 1, i, :, 0]
            grad_th = grad_phi[0:dim] @ alpha_star_t[:, 0:t_components]
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
        mesh.write("fractured_h1_cosserat_elasticity.vtk")
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return np.array([u_l2_error, t_l2_error, s_l2_error, m_l2_error])


def hdiv_cosserat_elasticity(k_order, gmesh, write_vtk_q=False):
    parallel_assembly_q = False

    dim = gmesh.dimension
    # Material data

    domain: Domain = gmesh.conformal_mesher.domain
    domain_tags = [1, 2, 3, 4, 5]
    frac_physical_tags = [
        shape.physical_tag
        for shape in domain.shapes[dim - 1]
        if (shape.physical_tag is not None) and (shape.physical_tag not in domain_tags)
    ]
    skin_p_physical_tags = [1000 * p_tag + 1 for p_tag in frac_physical_tags]
    skin_m_physical_tags = [1000 * p_tag - 1 for p_tag in frac_physical_tags]

    sigma_0 = -1.0
    h_bc_ids = [2, 4]
    v_bc_ids = [3, 5]
    f_bc_ids = [50, 50]  # [skin_p_physical_tags, skin_m_physical_tags]

    m_lambda = 1.0
    m_mu = 1.0
    m_kappa = 1.0
    Lc = 1.0
    m_gamma = m_mu * Lc * Lc

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

    s_k_order = k_order
    m_k_order = k_order
    k_int_order = np.max([s_k_order, m_k_order])
    # stress space
    s_space = DiscreteSpace(
        dim,
        s_components,
        s_family,
        s_k_order,
        gmesh,
        integration_order=2 * k_int_order + 1,
    )
    if dim == 2:
        # s_space.build_structures([2, 3, 4, 5, skin_p_physical_tags, skin_m_physical_tags])
        s_space.build_structures([2, 3, 4, 5])
    elif dim == 3:
        s_space.build_structures([2, 3, 4, 5, 6, 7])

    # couple stress space
    m_space = DiscreteSpace(
        dim,
        m_components,
        m_family,
        m_k_order,
        gmesh,
        integration_order=2 * k_int_order + 1,
    )
    if dim == 2:
        # m_space.build_structures([2, 3, 4, 5, skin_p_physical_tags, skin_m_physical_tags])
        m_space.build_structures([2, 3, 4, 5])
    elif dim == 3:
        m_space.build_structures([2, 3, 4, 5, 6, 7])

    # potential space
    u_space = DiscreteSpace(
        dim,
        u_components,
        u_family,
        s_k_order - 1,
        gmesh,
        integration_order=2 * k_int_order + 1,
    )
    u_space.make_discontinuous()
    u_space.build_structures()

    # rotation space
    t_space = DiscreteSpace(
        dim,
        t_components,
        t_family,
        m_k_order - 1,
        gmesh,
        integration_order=2 * k_int_order + 1,
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
            f_rhs,
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
        f_val_star = 0.0 * f_rhs(x[:, 0], x[:, 1], x[:, 2])
        u_phi_s_star = det_jac * weights * u_phi_tab[0, :, :, 0].T
        t_phi_s_star = det_jac * weights * t_phi_tab[0, :, :, 0].T

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
                el_form[b:e:u_components] += -1.0 * u_phi_s_star @ f_val_star[c]
            for c in range(t_components):
                b = c + n_s_dof + n_m_dof + n_u_dof
                e = b + n_t_dof
                el_form[b:e:t_components] += (
                    -1.0 * t_phi_s_star @ f_val_star[c + u_components]
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
                        Symm_sh
                        - (m_lambda / (2.0 * m_mu + dim * m_lambda)) * tr_s_h * Imat
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
                    S_cross = np.array([[sh[1, 0] - sh[0, 1]]])

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
                        Symm_sh
                        - (m_lambda / (2.0 * m_mu + dim * m_lambda)) * tr_s_h * Imat
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
                                sh[2, 1] - sh[1, 2],
                                sh[0, 2] - sh[2, 0],
                                sh[1, 0] - sh[0, 1],
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

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        for k in range(nnz):
            A.setValue(row=row[k], col=col[k], value=data[k], addv=True)

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
        f_rhs,
    )

    indexes = np.array([i for i in range(s_n_els)])
    collection = np.array_split(indexes, num_cpus)

    def scatter_form_data_on_cells(indexes, args):
        return [scatter_form_data_ad(i, args) for i in indexes]

    results = [scatter_form_data_on_cells(index_set, args) for index_set in collection]

    def scatter_s_bc_form_data(A, i, s_space):
        n_components = s_space.n_comp
        el_data: ElementData = s_space.bc_elements[i].data

        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # find high-dimension neigh
        entity_map = s_space.dof_map.mesh_topology.entity_map_by_dimension(
            cell.dimension
        )
        neigh_list = list(entity_map.predecessors(cell.id))
        neigh_check_q = len(neigh_list) > 0
        assert neigh_check_q

        neigh_cell_id = neigh_list[0]
        neigh_cell_index = s_space.id_to_element[neigh_cell_id]
        neigh_element = s_space.elements[neigh_cell_index]
        neigh_cell = neigh_element.data.cell

        # destination indexes
        dest_neigh = s_space.dof_map.destination_indices(neigh_cell_id)
        dest = s_space.dof_map.bc_destination_indices(neigh_cell_id, cell.id)

        n_phi = phi_tab.shape[2]
        n_dof = n_phi * n_components
        js = (n_dof, n_dof)
        j_el = np.zeros(js)
        r_el = np.zeros(n_dof)

        # compute trace space
        facet_index = neigh_cell.sub_cells_ids[cell.dimension].tolist().index(cell.id)
        vertices = basix.geometry(CellType.triangle)
        facet = basix.cell.sub_entity_connectivity(CellType.triangle)[cell.dimension][
            facet_index
        ][0]
        mapped_points = np.array(
            [vertices[facet[0]] * (1 - x) + vertices[facet[1]] * x for x in points]
        )
        dof_n_index = neigh_element.data.dof.entity_dofs[cell.dimension][facet_index]
        sn_phi_tab = neigh_element.evaluate_basis(mapped_points, False)

        # compute normal
        xc = np.mean(el_data.mesh.points[cell.node_tags], axis=0)
        neigh_xc = np.mean(neigh_element.data.mesh.points[neigh_cell.node_tags], axis=0)
        outward = (xc - neigh_xc) / np.linalg.norm((xc - neigh_xc))
        normal = outward / np.linalg.norm(outward)
        if normal @ outward < 0.0:
            normal *= -1.0

        beta = 1.0e12
        jac_block = np.zeros((n_phi, n_phi))
        for i, omega in enumerate(weights):
            phi = sn_phi_tab[0, i, dof_n_index, 0:s_components] @ normal[0:s_components]
            jac_block += beta * det_jac[i] * omega * np.outer(phi, phi)

        if cell.material_id in h_bc_ids:
            if cell.material_id == 4:
                component_list = [1]
                rhs_block = np.zeros(n_phi)
                for i, omega in enumerate(weights):
                    phi = (
                        sn_phi_tab[0, i, dof_n_index, 0:s_components]
                        @ normal[0:s_components]
                    )
                    rhs_block += beta * sigma_0 * det_jac[i] * omega * phi

                for c in component_list:
                    b = c
                    e = b + n_dof
                    r_el[b:e:n_components] += rhs_block

                for c in range(n_components):
                    b = c
                    e = b + n_dof
                    j_el[b:e:n_components, b:e:n_components] += jac_block

        if cell.material_id in v_bc_ids:
            for c in range(n_components):
                b = c
                e = b + n_dof
                j_el[b:e:n_components, b:e:n_components] += jac_block

        if cell.material_id in f_bc_ids:
            for c in range(n_components):
                b = c
                e = b + n_dof
                j_el[b:e:n_components, b:e:n_components] += jac_block

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        for k in range(nnz):
            A.setValue(row=row[k], col=col[k], value=data[k], addv=True)

    def scatter_m_bc_form_data(A, i, m_space):
        n_components = m_space.n_comp
        el_data: ElementData = m_space.bc_elements[i].data

        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # find high-dimension neigh
        entity_map = m_space.dof_map.mesh_topology.entity_map_by_dimension(
            cell.dimension
        )
        neigh_list = list(entity_map.predecessors(cell.id))
        neigh_check_q = len(neigh_list) > 0
        assert neigh_check_q

        neigh_cell_id = neigh_list[0]
        neigh_cell_index = m_space.id_to_element[neigh_cell_id]
        neigh_element = m_space.elements[neigh_cell_index]
        neigh_cell = neigh_element.data.cell

        # destination indexes
        dest_neigh = m_space.dof_map.destination_indices(neigh_cell_id) + s_n_dof_g
        dest = (
            m_space.dof_map.bc_destination_indices(neigh_cell_id, cell.id) + s_n_dof_g
        )

        n_phi = phi_tab.shape[2]
        n_dof = n_phi * n_components
        js = (n_dof, n_dof)
        j_el = np.zeros(js)
        r_el = np.zeros(n_dof)

        # compute trace space
        facet_index = neigh_cell.sub_cells_ids[cell.dimension].tolist().index(cell.id)
        vertices = basix.geometry(CellType.triangle)
        facet = basix.cell.sub_entity_connectivity(CellType.triangle)[cell.dimension][
            facet_index
        ][0]
        mapped_points = np.array(
            [vertices[facet[0]] * (1 - x) + vertices[facet[1]] * x for x in points]
        )
        dof_n_index = neigh_element.data.dof.entity_dofs[cell.dimension][facet_index]
        mn_phi_tab = neigh_element.evaluate_basis(mapped_points, False)

        # compute normal
        xc = np.mean(el_data.mesh.points[cell.node_tags], axis=0)
        neigh_xc = np.mean(neigh_element.data.mesh.points[neigh_cell.node_tags], axis=0)
        outward = (xc - neigh_xc) / np.linalg.norm((xc - neigh_xc))
        normal = outward / np.linalg.norm(outward)
        if normal @ outward < 0.0:
            normal *= -1.0

        beta = 1.0e12
        jac_block = np.zeros((n_phi, n_phi))
        for i, omega in enumerate(weights):
            phi = mn_phi_tab[0, i, dof_n_index, 0:m_components] @ normal[0:m_components]
            jac_block += beta * det_jac[i] * omega * np.outer(phi, phi)

        if cell.material_id in h_bc_ids:
            if cell.material_id == 4:
                for c in range(n_components):
                    b = c
                    e = b + n_dof
                    j_el[b:e:n_components, b:e:n_components] += jac_block

        if cell.material_id in v_bc_ids:
            for c in range(n_components):
                b = c
                e = b + n_dof
                j_el[b:e:n_components, b:e:n_components] += jac_block

        if cell.material_id in f_bc_ids:
            for c in range(n_components):
                b = c
                e = b + n_dof
                j_el[b:e:n_components, b:e:n_components] += jac_block

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        for k in range(nnz):
            A.setValue(row=row[k], col=col[k], value=data[k], addv=True)

    [scatter_s_bc_form_data(A, i, s_space) for i in range(len(s_space.bc_elements))]

    [scatter_m_bc_form_data(A, i, m_space) for i in range(len(m_space.bc_elements))]

    A.assemble()
    et = time.time()
    elapsed_time = et - st
    print("Assembly time:", elapsed_time, "seconds")

    # solving ls
    st = time.time()

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    b = A.createVecLeft()
    b.array[:] = -rg
    x = A.createVecRight()

    petsc_options = {"rtol": 1e-10, "atol": 1e-12, "divtol": 200, "max_it": 500}
    ksp = PETSc.KSP().create()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("fgmres")
    # ksp.setTolerances(**petsc_options)
    # ksp.setTolerances(1e-10)
    ksp.setTolerances(rtol=1e-10, atol=1e-10, divtol=500, max_it=2000)
    ksp.setConvergenceHistory()
    ksp.getPC().setType("ilu")
    ksp.solve(b, x)
    alpha = x.array

    # viewer = PETSc.Viewer().createASCII("ksp_output.txt")
    # ksp.view(viewer)
    # solver_output = open("ksp_output.txt", "r")
    # for line in solver_output.readlines():
    #     print(line)
    #
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
        mesh.write("fractured_hdiv_cosserat_elasticity.vtk")
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return np.array([u_l2_error, t_l2_error, s_l2_error, m_l2_error])


def main():
    dimension = 2
    k_order = 1
    h = 1.0
    l = 0

    domain = create_md_domain(dimension)
    mesher = create_conformal_mesher(domain, h, l)
    gmesh = create_mesh(dimension, mesher, True)
    # h1_cosserat_elasticity(k_order, gmesh, True)
    hdiv_cosserat_elasticity(k_order, gmesh, True)
    return


if __name__ == "__main__":
    main()
