import time

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
from SundusWeakForm import (
    SundusDualWeakForm,
    SundusDualWeakFormBCDirichlet,
)
import matplotlib.pyplot as plt


def create_product_space(method, gmesh):
    # FESpace: data
    mp_k_order = method[1]["mp"][1]
    mc_k_order = method[1]["mc"][1]
    p_k_order = method[1]["p"][1]
    c_k_order = method[1]["c"][1]

    mp_components = 1
    mc_components = 1
    p_components = 1
    c_components = 1

    mp_family = method[1]["mp"][0]
    mc_family = method[1]["mc"][0]
    p_family = method[1]["p"][0]
    c_family = method[1]["c"][0]

    discrete_spaces_data = {
        "mp": (gmesh.dimension, mp_components, mp_family, mp_k_order, gmesh),
        "mc": (gmesh.dimension, mc_components, mc_family, mc_k_order, gmesh),
        "p": (gmesh.dimension, p_components, p_family, p_k_order, gmesh),
        "c": (gmesh.dimension, c_components, c_family, c_k_order, gmesh),
    }

    mp_disc_Q = False
    mc_disc_Q = False
    p_disc_Q = True
    c_disc_Q = True
    discrete_spaces_disc = {
        "mp": mp_disc_Q,
        "mc": mc_disc_Q,
        "p": p_disc_Q,
        "c": c_disc_Q,
    }

    if gmesh.dimension == 1:
        mp_field_bc_physical_tags = [2, 3]
        mc_field_bc_physical_tags = [2, 3]
    elif gmesh.dimension == 2:
        mp_field_bc_physical_tags = [2, 3, 4, 5]
        mc_field_bc_physical_tags = [2, 3, 4, 5]
    elif gmesh.dimension == 3:
        mp_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
        mc_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
    else:
        raise ValueError("Case not available.")

    discrete_spaces_bc_physical_tags = {
        "mp": mp_field_bc_physical_tags,
        "mc": mc_field_bc_physical_tags,
    }

    space = ProductSpace(discrete_spaces_data)
    space.make_subspaces_discontinuous(discrete_spaces_disc)
    space.build_structures(discrete_spaces_bc_physical_tags)
    return space


def method_definition(k_order):
    # lower order convention
    method_1 = {
        "mp": ("RT", k_order + 1),
        "mc": ("RT", k_order + 1),
        "p": ("Lagrange", k_order),
        "c": ("Lagrange", k_order),
    }

    methods = [method_1]
    method_names = ["mixed_rt"]
    return zip(method_names, methods)


def four_fields_formulation(method, gmesh, write_vtk_q=False):
    dim = gmesh.dimension

    fe_space = create_product_space(method, gmesh)

    # Nonlinear solver data
    n_iterations = 20
    eps_tol = 1.0e-10
    delta_t = 0.1
    t_end = 1.0

    n_dof_g = fe_space.n_dof

    # Material data as scalars
    m_kappa = 1.0
    m_delta = 1.0
    m_eta = 1.0

    def f_kappa(x, y, z):
        return m_kappa

    def f_delta(x, y, z):
        return m_delta

    def f_eta(x, y, z):
        return m_eta

    st = time.time()

    # exact solution
    if dim == 1:
        p_exact = lambda x, y, z, t: np.array([t * (1.0 - x) * x])
        mp_exact = lambda x, y, z, t: np.array(
            [
                t * (-1.0 + 2.0 * x),
            ]
        )
        c_exact = lambda x, y, z, t: np.array([t * (1.0 - x) * x])
        mc_exact = lambda x, y, z, t: np.array(
            [
                t * (-1.0 + 2.0 * x),
            ]
        )
        f_rhs = lambda x, y, z, t: np.array([[t * (2.0 + 0.0 * x) + (x * (1 - x))]])

        r_rhs = lambda x, y, z, t: np.array([[t * (2.0 + 0.0 * x) + x * (1 - x)]]) + (
            1.0 + m_eta * c_exact(x, y, z, t) ** 2
        )
    elif dim == 2:
        p_exact = lambda x, y, z, t: np.array([t * ((1.0 - x) * x * (1.0 - y) * y)])
        mp_exact = lambda x, y, z, t: np.array(
            [
                [
                    t * (-((1 - 2 * x) * (1 - y) * y)),
                    -t * ((1 - x) * x * (1 - 2 * y)),
                ]
            ]
        )
        c_exact = lambda x, y, z, t: np.array([t * ((1.0 - x) * x * (1.0 - y) * y)])
        mc_exact = lambda x, y, z, t: np.array(
            [
                [
                    t * (-((1 - 2 * x) * (1 - y) * y)),
                    -t * ((1 - x) * x * (1 - 2 * y)),
                ]
            ]
        )
        f_rhs = lambda x, y, z, t: np.array(
            [[t * (2 * (1 - x) * x + 2 * (1 - y) * y) + (1 - x) * x * (1 - y) * y]]
        )
        r_rhs = lambda x, y, z, t: np.array(
            [[t * (2 * (1 - x) * x + 2 * (1 - y) * y) + (1 - x) * x * (1 - y) * y]]
        ) + (1.0 + m_eta * c_exact(t, x, y, z) ** 2)
    elif dim == 3:
        p_exact = lambda x, y, z, t: np.array(
            [t * ((1.0 - x) * x * (1.0 - y) * y * (1.0 - z) * z)]
        )
        mp_exact = lambda x, y, z, t: np.array(
            [
                [
                    t * (-((1 - 2 * x) * (1 - y) * y * (1 - z) * z)),
                    -t * ((1 - x) * x * (1 - 2 * y) * (1 - z) * z),
                    -t * ((1 - x) * x * (1 - y) * y * (1 - 2 * z)),
                ]
            ]
        )
        c_exact = lambda x, y, z, t: np.array(
            [t * ((1.0 - x) * x * (1.0 - y) * y * (1.0 - z) * z)]
        )
        mc_exact = lambda x, y, z, t: np.array(
            [
                [
                    t * (-((1 - 2 * x) * (1 - y) * y * (1 - z) * z)),
                    -t * ((1 - x) * x * (1 - 2 * y) * (1 - z) * z),
                    -t * ((1 - x) * x * (1 - y) * y * (1 - 2 * z)),
                ]
            ]
        )
        f_rhs = lambda x, y, z, t: np.array(
            [
                t
                * (
                    2 * (1 - x) * x * (1 - y) * y
                    + 2 * (1 - x) * x * (1 - z) * z
                    + 2 * (1 - y) * y * (1 - z) * z
                )
                + (1 - x) * x * (1 - y) * y * (1 - z) * z
            ]
        )
        r_rhs = lambda x, y, z, t: np.array(
            [
                t
                * (
                    2 * (1 - x) * x * (1 - y) * y
                    + 2 * (1 - x) * x * (1 - z) * z
                    + 2 * (1 - y) * y * (1 - z) * z
                )
                + (1 - x) * x * (1 - y) * y * (1 - z) * z
            ]
        ) + (1.0 + m_eta * c_exact(x, y, z, t) ** 2)
        aka = 0
        # raise ValueError("exact sol are nor created for 3D.")
    else:
        raise ValueError("Invalid dimension.")

    m_functions = {
        "rhs_f": f_rhs,
        "rhs_r": r_rhs,
        "kappa": f_kappa,
        "delta": f_delta,
        "eta": f_eta,
        "delta_t": delta_t,
    }

    exact_functions = {
        "mp": mp_exact,
        "p": p_exact,
        "mc": mc_exact,
        "c": c_exact,
    }

    weak_form = SundusDualWeakForm(fe_space)
    weak_form.functions = m_functions
    bc_weak_form = SundusDualWeakFormBCDirichlet(fe_space)
    bc_weak_form.functions = exact_functions

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

        jac_g = PETSc.Mat()
        jac_g.createAIJ([n_dof_g, n_dof_g])

        res_g = np.zeros(n_dof_g)
        print("n_dof: ", n_dof_g)

        # initial guess
        alpha_n_p_1 = alpha_n.copy()

        for iter in range(n_iterations):
            # Assembler
            st = time.time()
            n_els = len(fe_space.discrete_spaces["mp"].elements)
            [scatter_form_data(jac_g, i, weak_form, t) for i in range(n_els)]

            n_bc_els = len(fe_space.discrete_spaces["mp"].bc_elements)
            [scatter_bc_form(jac_g, i, bc_weak_form, t) for i in range(n_bc_els)]

            jac_g.assemble()

            et = time.time()
            elapsed_time = et - st
            print("Assembly time:", elapsed_time, "seconds")

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
            res_g *= 0.0
            jac_g.scale(0.0)

        alpha_n = alpha_n_p_1

    p_exact_t_end = lambda x, y, z: p_exact(x, y, z, t_end)
    mp_exact_t_end = lambda x, y, z: mp_exact(x, y, z, t_end)
    c_exact_t_end = lambda x, y, z: c_exact(x, y, z, t_end)
    mc_exact_t_end = lambda x, y, z: mc_exact(x, y, z, t_end)

    exact_functions_at_t_end = {
        "mp": mp_exact_t_end,
        "p": p_exact_t_end,
        "mc": mc_exact_t_end,
        "c": c_exact_t_end,
    }

    st = time.time()
    mp_l2_error, mc_l2_error, p_l2_error, c_l2_error = l2_error(
        dim, fe_space, exact_functions_at_t_end, alpha_n_p_1
    )
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    print("L2-error in mp: ", mp_l2_error)
    print("L2-error in p : ", p_l2_error)
    print("L2-error in mc: ", mc_l2_error)
    print("L2-error in c : ", c_l2_error)

    if write_vtk_q:
        # post-process solution
        st = time.time()
        file_name = "rates_four_fields.vtk"
        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, exact_functions_at_t_end, alpha_n_p_1
        )
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return p_l2_error + c_l2_error


def create_domain(dimension):
    if dimension == 1:
        box_points = np.array([[0, 0, 0], [1, 0, 0]])
        domain = build_box_1D(box_points)
        return domain
    elif dimension == 2:
        box_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        # box_points = [
        #     point + 0.25 * np.array([-1.0, -1.0, 0.0]) for point in box_points
        # ]
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
        # box_points = [
        #     point + 0.25 * np.array([-1.0, -1.0, -1.0]) for point in box_points
        # ]
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
    n_ref = 4
    dimension = 1

    domain = create_domain(dimension)
    error_data = np.empty((0, 2), float)

    for method in method_definition(k_order):
        for l in range(n_ref):
            h_val = h * (2**-l)
            mesher = create_conformal_mesher(domain, h_val, 0)
            gmesh = create_mesh(dimension, mesher, True)
            error_val = four_fields_formulation(method, gmesh, True)
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

    a = error_data[:, 0]
    b = error_data[:, 1]
    plt.loglog(a, b)
    plt.show()
    return


if __name__ == "__main__":
    main()
