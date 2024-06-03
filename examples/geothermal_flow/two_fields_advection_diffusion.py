"""
File: two_fields_advection_diffusion.py
Description: This script provide examples for the approach adopted in the contribution:
https://doi.org/10.5540/tema.2017.018.02.0253

Author: Omar Duran
Email: omaryesiduran@gmail.com
Date: 2024-05-30
Version: 1.0.0
License: GPL-3.0 license

"""

import time
from functools import partial
import numpy as np
import scipy
from petsc4py import PETSc

from basis.element_data import ElementData
from geometry.domain import Domain
from geometry.domain_market import build_box_1D, build_box_2D, build_box_3D
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh
from postprocess.l2_error_post_processor import l2_error
from postprocess.projectors import l2_projector
from postprocess.solution_post_processor import write_vtk_file_with_exact_solution
from spaces.product_space import ProductSpace
from TwoFieldsAdvectionDiffusionWeakForm import (
    TwoFieldsDiffusionWeakForm,
    TwoFieldsDiffusionWeakFormBCRobin,
    TwoFieldsAdvectionWeakForm,
    TwoFieldsAdvectionWeakFormBC,
)
import matplotlib.pyplot as plt


def create_product_space(method, gmesh):
    # FESpace: data
    q_k_order = method[1]["q"][1]
    u_k_order = method[1]["u"][1]

    q_components = 1
    u_components = 1

    q_family = method[1]["q"][0]
    u_family = method[1]["u"][0]

    discrete_spaces_data = {
        "q": (gmesh.dimension, q_components, q_family, q_k_order, gmesh),
        "u": (gmesh.dimension, u_components, u_family, u_k_order, gmesh),
    }

    q_disc_Q = False
    u_disc_Q = True
    discrete_spaces_disc = {
        "q": q_disc_Q,
        "u": u_disc_Q,
    }

    if gmesh.dimension == 1:
        q_field_bc_physical_tags = [2, 3]
    elif gmesh.dimension == 2:
        q_field_bc_physical_tags = [2, 3, 4, 5]
    elif gmesh.dimension == 3:
        q_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
    else:
        raise ValueError("Case not available.")

    discrete_spaces_bc_physical_tags = {
        "q": q_field_bc_physical_tags,
    }

    space = ProductSpace(discrete_spaces_data)
    space.make_subspaces_discontinuous(discrete_spaces_disc)
    space.build_structures(discrete_spaces_bc_physical_tags)
    return space


def method_definition(k_order):
    # lower order convention
    method = {
        "q": ("RT", k_order + 1),
        "u": ("Lagrange", k_order),
    }
    return ("mixed_rt", method)


def two_fields_formulation(method, gmesh, write_vtk_q=False):
    dim = gmesh.dimension
    fe_space = create_product_space(method, gmesh)

    # Nonlinear solver data
    n_iterations = 20
    eps_tol = 1.0e-8
    delta_t = 1.0
    t_end = 1.0

    n_dof_g = fe_space.n_dof

    # Material data as scalars

    # constant permeability
    m_kappa = np.pi * 1.0e-2
    # constant velocity
    m_velocity = 0.5  # 1.0e-10
    m_velocity_v = m_velocity * np.ones(3)  # 2.0

    # beta
    m_beta_l = 1.0
    m_beta_r = 1.0e12

    # gamma
    m_gamma_l = 0.0
    m_gamma_r = 0.0

    # c
    m_c_l = 5.0
    m_c_r = 1.0

    def xi_map(x, y, z, m_left, m_right):
        return m_left * (1 - x) + m_right * x

    f_kappa = partial(xi_map, m_left=m_kappa, m_right=m_kappa)
    f_velocity = partial(xi_map, m_left=m_velocity_v, m_right=m_velocity_v)
    f_beta = partial(xi_map, m_left=m_beta_l, m_right=m_beta_r)
    f_gamma = partial(xi_map, m_left=m_gamma_l, m_right=m_gamma_r)
    f_c = partial(xi_map, m_left=m_c_l, m_right=m_c_r)

    st = time.time()
    # exact solution
    if dim == 1:

        def u_exact(x, y, z):
            return np.array(
                [
                    [
                        (
                            m_c_r
                            * (m_beta_r - m_gamma_r)
                            * (
                                m_kappa * m_velocity
                                - (-1 + np.exp((x * m_velocity) / m_kappa))
                                * m_kappa
                                * (m_beta_l - m_velocity)
                            )
                            + m_c_l
                            * (m_beta_l - m_gamma_l)
                            * (
                                np.exp((x * m_velocity) / m_kappa)
                                * m_kappa
                                * (m_beta_r - m_velocity)
                                - np.exp(m_velocity / m_kappa)
                                * (
                                    m_beta_r * m_kappa
                                    + m_kappa * m_velocity
                                    - m_kappa * m_velocity
                                )
                            )
                        )
                        / (
                            (
                                m_beta_l * m_kappa
                                + m_kappa * m_velocity
                                - m_kappa * m_velocity
                            )
                            * (m_beta_r - m_velocity)
                            - np.exp(m_velocity / m_kappa)
                            * (m_beta_l - m_velocity)
                            * (
                                m_beta_r * m_kappa
                                + m_kappa * m_velocity
                                - m_kappa * m_velocity
                            )
                        )
                    ]
                ]
            )

        def q_exact(x, y, z):
            return np.array(
                [
                    -(
                        (
                            m_kappa
                            * (
                                -(
                                    m_c_r
                                    * np.exp((x * m_velocity) / m_kappa)
                                    * (m_beta_r - m_gamma_r)
                                    * m_velocity
                                    * (m_beta_l - m_velocity)
                                )
                                + m_c_l
                                * np.exp((x * m_velocity) / m_kappa)
                                * (m_beta_l - m_gamma_l)
                                * m_velocity
                                * (m_beta_r - m_velocity)
                            )
                        )
                        / (
                            (
                                m_beta_l * m_kappa
                                + m_kappa * m_velocity
                                - m_kappa * m_velocity
                            )
                            * (m_beta_r - m_velocity)
                            - np.exp(m_velocity / m_kappa)
                            * (m_beta_l - m_velocity)
                            * (
                                m_beta_r * m_kappa
                                + m_kappa * m_velocity
                                - m_kappa * m_velocity
                            )
                        )
                    )
                ]
            )

        def f_rhs(x, y, z):
            return np.array([[np.zeros_like(x)]])

    elif dim == 2:
        raise ValueError("Case not available.")
    elif dim == 3:
        raise ValueError("Case not available.")
    else:
        raise ValueError("Invalid dimension.")

    m_functions = {
        "rhs": f_rhs,
        "kappa": f_kappa,
        "velocity": f_velocity,
    }

    exact_functions = {
        "u": u_exact,
        "q": q_exact,
    }

    m_bc_functions = {
        "beta": f_beta,
        "gamma": f_gamma,
        "c": f_c,
        "velocity": f_velocity,
    }

    weak_form = TwoFieldsDiffusionWeakForm(fe_space)
    weak_form.functions = m_functions
    bc_weak_form = TwoFieldsDiffusionWeakFormBCRobin(fe_space)
    bc_weak_form.functions = m_bc_functions

    advection_weak_form = TwoFieldsAdvectionWeakForm(fe_space)
    advection_weak_form.functions = m_bc_functions

    bc_advection_weak_form = TwoFieldsAdvectionWeakFormBC(fe_space)
    bc_advection_weak_form.functions = z = {**exact_functions, **m_bc_functions}

    # retrieve external and internal triplets
    c1_entities = [cell for cell in gmesh.cells if cell.dimension == dim - 1]
    gc0_c1 = gmesh.build_graph(dim, 1)
    c1_triplets = [
        (cell.id, list(gc0_c1.predecessors(cell.id))) for cell in c1_entities
    ]
    c1_itriplets = [triplet for triplet in c1_triplets if len(triplet[1]) == 2]
    c1_epairs = [
        (triplet[0], triplet[1][0]) for triplet in c1_triplets if len(triplet[1]) == 1
    ]

    gidx_midx = fe_space.discrete_spaces["q"].id_to_element
    c1_itriplets = [
        (triplet[0], [gidx_midx[triplet[1][0]], gidx_midx[triplet[1][1]]])
        for triplet in c1_itriplets
    ]
    c1_epairs = [(pair[0], gidx_midx[pair[1]]) for pair in c1_epairs]

    # Initial Guess
    alpha_n = np.zeros(n_dof_g)

    for t in np.arange(delta_t, t_end + delta_t, delta_t):
        print("Current time value: ", t)

        def scatter_form_data(jac_g, i, weak_form, t):
            # destination indexes
            dest = weak_form.space.destination_indexes(i)
            alpha_l_n = alpha_n[dest]
            alpha_l_n_p_1 = alpha_n_p_1[dest]

            r_el, j_el = weak_form.evaluate_form(i, alpha_l_n_p_1, alpha_l_n, t)

            # contribute rhs
            res_g[dest] += r_el

            # contribute lhs
            data = j_el.ravel()
            row = np.repeat(dest, len(dest))
            col = np.tile(dest, len(dest))
            nnz = data.shape[0]
            for k in range(nnz):
                jac_g.setValue(row=row[k], col=col[k], value=data[k], addv=True)

        def scatter_bc_form(jac_g, i, bc_weak_form, t):
            dest = fe_space.bc_destination_indexes(i)
            alpha_l = alpha_n_p_1[dest]
            r_el, j_el = bc_weak_form.evaluate_form(i, alpha_l, t)

            # contribute rhs
            res_g[dest] += r_el

            # contribute lhs
            data = j_el.ravel()
            row = np.repeat(dest, len(dest))
            col = np.tile(dest, len(dest))
            nnz = data.shape[0]
            for k in range(nnz):
                jac_g.setValue(row=row[k], col=col[k], value=data[k], addv=True)

        def scatter_c1_form_data(jac_g, triplet, weak_form):
            # destination indexes
            cell_id, idx_pair = triplet
            i_p, i_n = idx_pair
            dest_p = weak_form.space.destination_indexes(i_p)
            dest_n = weak_form.space.destination_indexes(i_n)
            alpha_pair = (alpha_n_p_1[dest_p], alpha_n_p_1[dest_n])

            r_el, j_el = weak_form.evaluate_form(
                gmesh.cells[cell_id], idx_pair, alpha_pair
            )

            dest = np.concatenate((dest_p, dest_n))
            # contribute rhs
            res_g[dest] += r_el

            # contribute lhs
            data = j_el.ravel()
            row = np.repeat(dest, len(dest))
            col = np.tile(dest, len(dest))
            nnz = data.shape[0]
            for k in range(nnz):
                jac_g.setValue(row=row[k], col=col[k], value=data[k], addv=True)

        def scatter_bc_c1_form(jac_g, pair, weak_form):
            # destination indexes
            cell_id, i = pair
            dest = weak_form.space.destination_indexes(i)
            alpha_l = alpha_n_p_1[dest]

            r_el, j_el = weak_form.evaluate_form(gmesh.cells[cell_id], i, alpha_l)

            # contribute rhs
            res_g[dest] += r_el

            # contribute lhs
            data = j_el.ravel()
            row = np.repeat(dest, len(dest))
            col = np.tile(dest, len(dest))
            nnz = data.shape[0]
            for k in range(nnz):
                jac_g.setValue(row=row[k], col=col[k], value=data[k], addv=True)

        jac_g = PETSc.Mat()
        jac_g.createAIJ([n_dof_g, n_dof_g])

        res_g = np.zeros(n_dof_g)
        print("n_dof: ", n_dof_g)

        # initial guess
        alpha_n_p_1 = alpha_n.copy()
        alpha_n_p_1 = l2_projector(fe_space, exact_functions)

        for iter in range(n_iterations):
            # Assembler
            st = time.time()

            n_els = len(fe_space.discrete_spaces["q"].elements)
            [scatter_form_data(jac_g, i, weak_form, t) for i in range(n_els)]

            n_bc_els = len(fe_space.discrete_spaces["q"].bc_elements)
            [scatter_bc_form(jac_g, i, bc_weak_form, t) for i in range(n_bc_els)]

            [
                scatter_c1_form_data(jac_g, triplet, advection_weak_form)
                for triplet in c1_itriplets
            ]

            [
                scatter_bc_c1_form(jac_g, pair, bc_advection_weak_form)
                for pair in c1_epairs
            ]

            jac_g.assemble()

            et = time.time()
            elapsed_time = et - st
            print("Assembly time:", elapsed_time, "seconds")

            # ai, aj, av = jac_g.getValuesCSR()
            # Asp = scipy.sparse.csr_matrix((av, aj, ai))
            # plt.matshow(Asp.A)

            res_norm = np.linalg.norm(res_g)
            stop_criterion_q = res_norm < eps_tol
            if stop_criterion_q:
                print("Nonlinear solver converged")
                print("Residual norm: ", res_norm)
                print("Number of iterations: ", iter)
                break

            # solving ls
            st = time.time()
            ksp = PETSc.KSP().create()
            ksp.setOperators(jac_g)
            b = jac_g.createVecLeft()
            b.array[:] = -res_g
            x = jac_g.createVecRight()

            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
            ksp.setConvergenceHistory()
            ksp.solve(b, x)
            delta_alpha = x.array

            et = time.time()
            elapsed_time = et - st
            print("Linear solver time:", elapsed_time, "seconds")

            # newton update
            alpha_n_p_1 += delta_alpha

            # Set up to zero lhr and rhs
            res_g = np.zeros_like(res_g)
            jac_g.scale(0.0)

        alpha_n = alpha_n_p_1

    # alpha_n_p_1 = l2_projector(fe_space, exact_functions)
    # p_exact_t_end = lambda x, y, z: p_exact(x, y, z, t_end)
    # mp_exact_t_end = lambda x, y, z: mp_exact(x, y, z, t_end)
    # c_exact_t_end = lambda x, y, z: c_exact(x, y, z, t_end)
    # mc_exact_t_end = lambda x, y, z: mc_exact(x, y, z, t_end)
    #
    # exact_functions_at_t_end = {
    #     "mp": mp_exact_t_end,
    #     "p": p_exact_t_end,
    #     "mc": mc_exact_t_end,
    #     "c": c_exact_t_end,
    # }
    # alpha_n_p_1 = l2_projector(fe_space, exact_functions)
    st = time.time()
    q_l2_error, u_l2_error = l2_error(dim, fe_space, exact_functions, alpha_n_p_1)
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    print("L2-error in q: ", q_l2_error)
    print("L2-error in u: ", u_l2_error)

    if write_vtk_q:
        # post-process solution
        st = time.time()
        file_name = "rates_two_fields.vtk"
        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, exact_functions, alpha_n_p_1, ["u"]
        )
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return u_l2_error + q_l2_error


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
    k_order = 0
    h = 0.5
    n_ref = 6
    dimension = 1

    domain = create_domain(dimension)
    error_data = np.empty((0, 2), float)
    method = method_definition(k_order)
    for l in range(n_ref):
        h_val = h * (2**-l)
        mesher = create_conformal_mesher(domain, h_val, 0)
        gmesh = create_mesh(dimension, mesher, True)
        error_val = two_fields_formulation(method, gmesh, True)
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

    a = error_data[:, 0]
    b = error_data[:, 1]
    plt.loglog(a, b)
    plt.show()
    return


if __name__ == "__main__":
    main()
