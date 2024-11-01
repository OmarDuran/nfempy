"""
File: three_fields_advection_diffusion.py
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
from petsc4py import PETSc

from topology.domain import Domain
from topology.domain_market import build_box_1D, build_box_2D, build_box_3D
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh
from postprocess.l2_error_post_processor import l2_error
from postprocess.solution_post_processor import write_vtk_file_with_exact_solution
from spaces.product_space import ProductSpace
from assembly.SequentialAssembler import SequentialAssembler
from ThreeFieldsAdvectionDiffusionWeakForm import (
    ThreeFieldsDiffusionWeakForm,
    ThreeFieldsDiffusionWeakFormBCRobin,
    ThreeFieldsAdvectionWeakForm,
    ThreeFieldsAdvectionWeakFormBC,
)
import matplotlib.pyplot as plt


def create_product_space(method, gmesh):
    # FESpace: data
    q_k_order = method[1]["q"][1]
    m_k_order = method[1]["m"][1]
    u_k_order = method[1]["u"][1]

    q_components = 1
    m_components = 1
    u_components = 1

    q_family = method[1]["q"][0]
    m_family = method[1]["m"][0]
    u_family = method[1]["u"][0]

    discrete_spaces_data = {
        "q": (gmesh.dimension, q_components, q_family, q_k_order, gmesh),
        "m": (gmesh.dimension, m_components, m_family, m_k_order, gmesh),
        "u": (gmesh.dimension, u_components, u_family, u_k_order, gmesh),
    }

    q_disc_Q = False
    m_disc_Q = False
    u_disc_Q = True
    discrete_spaces_disc = {
        "q": q_disc_Q,
        "m": m_disc_Q,
        "u": u_disc_Q,
    }

    if gmesh.dimension == 1:
        q_field_bc_physical_tags = [2, 3]
        m_field_bc_physical_tags = [2, 3]
    elif gmesh.dimension == 2:
        q_field_bc_physical_tags = [2, 3, 4, 5]
        m_field_bc_physical_tags = [2, 3, 4, 5]
    elif gmesh.dimension == 3:
        q_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
        m_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
    else:
        raise ValueError("Case not available.")

    physical_tags = {
        "q": [1],
        "m": [1],
        "u": [1],
    }

    b_physical_tags = {
        "q": q_field_bc_physical_tags,
        "m": m_field_bc_physical_tags,
    }

    space = ProductSpace(discrete_spaces_data)
    space.make_subspaces_discontinuous(discrete_spaces_disc)
    space.build_structures(physical_tags, b_physical_tags)
    return space


def method_definition(k_order):
    # lower order convention
    method = {
        "q": ("RT", k_order + 1),
        "m": ("RT", k_order + 1),
        "u": ("Lagrange", k_order),
    }
    return ("mixed_rt", method)


def three_fields_formulation(method, gmesh, write_vtk_q=False):
    dim = gmesh.dimension
    fe_space = create_product_space(method, gmesh)

    # Nonlinear solver data
    n_iterations = 20
    eps_tol = 1.0e-4
    delta_t = 1.0
    t_end = 1.0

    n_dof_g = fe_space.n_dof

    # Material data as scalars

    # constant permeability
    m_kappa = np.pi
    # constant velocity
    m_velocity = 100.0  # 1.0e-10
    m_velocity_v = m_velocity * np.ones(3)  # 2.0

    # beta
    m_beta_l = 1.0e12
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

        def m_exact(x, y, z):
            return m_velocity * np.array(
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
        "m": m_exact,
    }

    m_bc_functions = {
        "beta": f_beta,
        "gamma": f_gamma,
        "c": f_c,
        "velocity": f_velocity,
    }

    weak_form = ThreeFieldsDiffusionWeakForm(fe_space)
    weak_form.functions = m_functions
    bc_weak_form = ThreeFieldsDiffusionWeakFormBCRobin(fe_space)
    bc_weak_form.functions = m_bc_functions

    advection_weak_form = ThreeFieldsAdvectionWeakForm(fe_space)
    advection_weak_form.functions = m_bc_functions
    bc_advection_weak_form = ThreeFieldsAdvectionWeakFormBC(fe_space)
    bc_advection_weak_form.functions = {**exact_functions, **m_bc_functions}

    # retrieve external and internal triplets
    c1_entities = [cell for cell in gmesh.cells if cell.dimension == dim - 1]
    gc0_c1 = gmesh.build_graph(dim, 1)
    c1_triplets = [
        (cell.id, list(gc0_c1.predecessors(cell.index()))) for cell in c1_entities
    ]
    c1_itriplets = [triplet for triplet in c1_triplets if len(triplet[1]) == 2]
    c1_epairs = [
        (triplet[0], triplet[1][0][1])
        for triplet in c1_triplets
        if len(triplet[1]) == 1
    ]
    gidx_midx = fe_space.discrete_spaces["q"].id_to_element

    # create sequences
    n_els = len(fe_space.discrete_spaces["q"].elements)
    n_bc_els = len(fe_space.discrete_spaces["q"].bc_elements)
    sequence_domain = [i for i in range(n_els)]
    sequence_bc_domain = [i for i in range(n_bc_els)]
    sequence_c1_itriplets = [
        (triplet[0], [gidx_midx[triplet[1][0]], gidx_midx[triplet[1][1]]])
        for triplet in c1_itriplets
    ]
    sequence_c1_epairs = [(pair[0], gidx_midx[pair[1]]) for pair in c1_epairs]

    # Initial Guess
    alpha_n = np.zeros(n_dof_g)

    for t in np.arange(delta_t, t_end + delta_t, delta_t):
        print("Current time value: ", t)

        jac_g = PETSc.Mat()
        jac_g.createAIJ([n_dof_g, n_dof_g])

        res_g = np.zeros(n_dof_g)
        print("n_dof: ", n_dof_g)

        # initial guess
        alpha_n_p_1 = alpha_n.copy()
        # alpha_n_p_1 = l2_projector(fe_space, exact_functions)

        for iter in range(n_iterations):
            # break

            # Assembler
            assembler = SequentialAssembler(fe_space, jac_g, res_g)
            form_to_input_list = {
                "difussion_form": [
                    "time_dependent_form",
                    sequence_domain,
                    weak_form,
                    (alpha_n_p_1, alpha_n),
                    t,
                ],
                "difussion_bc_form": [
                    "time_dependent_bc_form",
                    sequence_bc_domain,
                    bc_weak_form,
                    (alpha_n_p_1, alpha_n),
                    t,
                ],
                "advection_form": [
                    "interface_form",
                    sequence_c1_itriplets,
                    advection_weak_form,
                    alpha_n_p_1,
                ],
                "advection_bc_form": [
                    "bc_interface_form",
                    sequence_c1_epairs,
                    bc_advection_weak_form,
                    alpha_n_p_1,
                ],
            }
            assembler.form_to_input_list = form_to_input_list
            assembler.scatter_forms(measure_time_q=False)

            jac_g.assemble()

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

    st = time.time()
    q_l2_error, m_l2_error, u_l2_error = l2_error(
        dim, fe_space, exact_functions, alpha_n_p_1
    )
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    print("L2-error in q: ", q_l2_error)
    print("L2-error in m: ", m_l2_error)
    print("L2-error in u: ", u_l2_error)

    if write_vtk_q:
        # post-process solution
        st = time.time()
        file_name = "rates_three_fields.vtk"
        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, exact_functions, alpha_n_p_1, ["u"]
        )
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return u_l2_error + q_l2_error + m_l2_error


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
    n_ref = 9
    dimension = 1

    domain = create_domain(dimension)
    error_data = np.empty((0, 2), float)
    method = method_definition(k_order)
    for l in range(n_ref):
        h_val = h * (2**-l)
        mesher = create_conformal_mesher(domain, h_val, 0)
        gmesh = create_mesh(dimension, mesher, True)
        error_val = three_fields_formulation(method, gmesh, True)
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
