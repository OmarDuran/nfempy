import time
import math
import numpy as np
import scipy
from petsc4py import PETSc

from basis.element_data import ElementData
from geometry.domain import Domain
from geometry.domain_market import build_box_1D, build_box_2D, build_box_3D
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh
from postprocess.l2_error_post_processor import l2_error, l2_error_projected
from postprocess.projectors import l2_projector
from postprocess.solution_post_processor import write_vtk_file_with_exact_solution
from spaces.product_space import ProductSpace
from ArbogastWeakForm import (
    ArbogastDualWeakForm,
    ArbogastDualWeakFormBCDirichlet,
)
from ToPhysicalProjectionWeakForm import ToPhysicalProjectionWeakForm
import matplotlib.pyplot as plt


def create_product_space(method, gmesh):
    # FESpace: data
    mp_k_order = method[1]["v"][1]
    p_k_order = method[1]["q"][1]

    mp_components = 1
    p_components = 1

    mp_family = method[1]["v"][0]
    p_family = method[1]["q"][0]

    discrete_spaces_data = {
        "v": (gmesh.dimension, mp_components, mp_family, mp_k_order, gmesh),
        "q": (gmesh.dimension, p_components, p_family, p_k_order, gmesh),
    }

    mp_disc_Q = False
    p_disc_Q = True
    discrete_spaces_disc = {
        "v": mp_disc_Q,
        "q": p_disc_Q,
    }

    if gmesh.dimension == 1:
        mp_field_bc_physical_tags = [2, 3]
    elif gmesh.dimension == 2:
        mp_field_bc_physical_tags = [2, 3, 4, 5]
    elif gmesh.dimension == 3:
        mp_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
    else:
        raise ValueError("Case not available.")

    discrete_spaces_bc_physical_tags = {
        "v": mp_field_bc_physical_tags,
    }

    space = ProductSpace(discrete_spaces_data)
    space.make_subspaces_discontinuous(discrete_spaces_disc)
    space.build_structures(discrete_spaces_bc_physical_tags)
    return space


def method_definition(k_order):
    # lower order convention
    method_1 = {
        "v": ("RT", k_order + 1),
        "q": ("Lagrange", k_order),
    }

    methods = [method_1]
    method_names = ["mixed_rt"]
    return zip(method_names, methods)


def two_fields_formulation(method, gmesh, write_vtk_q=False):
    dim = gmesh.dimension

    st = time.time()
    fe_space = create_product_space(method, gmesh)
    et = time.time()
    elapsed_time = et - st
    print("Creation of product space:", elapsed_time, "seconds")

    # Nonlinear solver data
    n_iterations = 20
    eps_tol = 1.0e-10

    n_dof_g = fe_space.n_dof

    # Material data as scalars
    m_kappa = 1.0
    m_delta = 1.0
    m_eta = 1.0
    m_mu = 1.0


    st = time.time()

    # exact solution
    if dim == 1:

        r1 = (-3 + np.sqrt(13)) / 2
        r2 = (-3 - np.sqrt(13)) / 2

        beta = 0.5

        def f_porosity(x, y, z):
            return np.array(x ** 2)

        def f_grad_porosity(x, y, z):
            return np.array([2 * x, y * 0.0, z * 0.0])

        # physical variables
        def p_exact(x, y, z):
            return np.where(
                x < 0.0,
                np.array([x * 0.0]),
                np.array(
                    [
                        (
                                -(np.abs(x) ** beta)
                                + 0.5
                                * (3 + np.sqrt(13))
                                * np.abs(x) ** (0.5 * (-3 + np.sqrt(13)))
                                * beta
                        )
                        / (-1 + beta * (3 + beta)),
                    ]
                ),
            )


        def u_exact(x, y, z):
            return np.where(
                x < 0.0,
                np.array([x * 0.0]),
                np.array(
                    [
                        (
                            (
                                -np.abs(x) ** (0.5 * (3 + np.sqrt(13)))
                                + np.abs(x) ** (3 + beta)
                            )
                            * beta
                        )
                        / (-1 + beta * (3 + beta)),
                    ]
                ),
            )

        # unphysical variables
        def q_exact(x, y, z):
            return np.where(
                x < 0.0,
                np.zeros_like(x),
                np.sqrt(f_porosity(x, y, z)) * p_exact(x, y, z),
            )

        def v_exact(x, y, z):
            return np.where(
                x < 0.0, np.array([x * 0.0]), u_exact(x, y, z) / f_d_phi(x, y, z)
            )

        def f_rhs(x, y, z):
            return np.where(
                x < 0.0,
                np.array([[x * 0.0]]),
                np.array([[(np.abs(x) ** beta) * np.sqrt(f_porosity(x, y, z))]]),
            )

        def f_kappa(x, y, z):
            return f_porosity(x, y, z) ** 2

        def f_mu(x, y, z):
            return m_mu

        def f_d_phi(x, y, z):
            return np.sqrt(f_kappa(x, y, z) / f_mu(x, y, z))

        def f_grad_d_phi(x, y, z):
            scalar_part = f_porosity(x, y, z) / (f_mu(x, y, z) * f_d_phi(x, y, z))
            vector_part = f_grad_porosity(x, y, z)
            return scalar_part * vector_part

    elif dim == 2:

        gamma = 1.0

        def f_porosity(x, y, z):
            return np.where(np.logical_or(x <= -3/4, y <= -3/4),
                np.zeros_like(x),
                ((0.75 + x)**gamma)*((0.75 + y)**(2*gamma)),
            )

        def f_grad_porosity(x, y, z):
            return np.where(np.logical_or(x <= -3 / 4, y <= -3 / 4),
                np.array([x * 0.0, y * 0.0]),
                np.array(
                    [
                        ((0.75 + x)**(-1 + gamma))*((0.75 + y)**(2*gamma))*gamma
                        ,
                        2*((0.75 + x)**gamma)*((0.75 + y)**(-1 + 2*gamma))*gamma
                    ]
                ),
            )

        # physical variables
        def p_exact(x, y, z):
            return np.where(np.logical_or(x <= -3/4, y <= -3/4),
                np.array([np.zeros_like(x)]),
                np.array([np.cos(6 * x * (y ** 2))]),
            )

        def u_exact(x, y, z):
            return np.where(np.logical_or(x <= -3/4, y <= -3/4),
                np.array([[x * 0.0,y * 0.0]]),
                np.array(
                    [
                    [
                        6 * ((0.75 + x) ** (2 * gamma)) * (y ** 2) * (
                                    (0.75 + y) ** (4 * gamma)) * np.sin(6 * x * (y ** 2))
                        ,
                        12 * x * ((0.75 + x) ** (2 * gamma)) * y * (
                                    (0.75 + y) ** (4 * gamma)) * np.sin(6 * x * (y ** 2))
                    ]
                    ]
                ),
            )

        # unphysical variables
        def q_exact(x, y, z):
            return np.where(np.logical_or(x <= -3/4, y <= -3/4),
                np.array([np.zeros_like(x)]),
                np.array([np.sqrt(((0.75 + x)**gamma)*((0.75 + y)**(2*gamma)))*np.cos(6*x*(y**2))]),
            )

        def v_exact(x, y, z):
            return np.where(np.logical_or(x <= -3/4, y <= -3/4),
                np.array([[x * 0.0,y * 0.0]]),
                np.array(
                    [
                    [
                        6*(y**2)*np.sqrt(((0.75 + x)**(2*gamma))*((0.75 + y)**(4*gamma)))*np.sin(6*x*(y**2))
                        ,
                        12*x*y*np.sqrt(((0.75 + x)**(2*gamma))*((0.75 + y)**(4*gamma)))*np.sin(6*x*(y**2))
                    ]
                    ]
                ),
            )

        def f_rhs(x, y, z):
            return np.where(np.logical_or(x <= -3/4, y <= -3/4),
                np.array([[x * 0.0]]),
                np.array([[np.sqrt(f_porosity(x, y, z)) *
                            ((1 + 36 * ((0.75 + x) ** gamma) * (y ** 2) * ((0.75 + y) ** (2 * gamma)) * (
                            4 * (x** 2) + (y ** 2))) *
                np.cos(6 * x * (y ** 2)) + (
                            12 * ((0.75 + x) ** gamma) * ((0.75 + y) ** (2 * gamma)) *
                            (x * (3 + 4 * x) * (3 + 4 * y) + 4 * y * (
                                        4 * x * (3 + 4 * x) + y * (
                                            3 + 4 * y)) * gamma) * np.sin(
                        6 * x * (y ** 2)))
                / ((3 + 4 * x) * (3 + 4 * y)))]]),
            )

        def f_kappa(x, y, z):
            return f_porosity(x, y, z) ** 2

        def f_mu(x, y, z):
            return m_mu

        def f_d_phi(x, y, z):
            return np.sqrt(f_kappa(x, y, z) / f_mu(x, y, z))

        def f_grad_d_phi(x, y, z):
            scalar_part = np.where(np.logical_or(x <= -3 / 4, y <= -3 / 4), np.zeros_like(x) ,f_porosity(x, y, z) / (f_mu(x, y, z) * f_d_phi(x, y, z)))
            vector_part = f_grad_porosity(x, y, z)
            return scalar_part * vector_part

    else:
        raise ValueError("Invalid dimension.")


    m_functions = {
        "rhs_f": f_rhs,
        "kappa": f_kappa,
        "porosity": f_porosity,
        "d_phi": f_d_phi,
        "grad_d_phi": f_grad_d_phi,
    }

    bc_functions = {
        "v": v_exact,
        "q": q_exact,
        "d_phi": f_d_phi,
        "porosity": f_porosity,
    }

    exact_functions = {
        "v": v_exact,
        "q": q_exact,
    }


    weak_form = ArbogastDualWeakForm(fe_space)
    weak_form.functions = m_functions
    bc_weak_form = ArbogastDualWeakFormBCDirichlet(fe_space)
    bc_weak_form.functions = bc_functions

    to_physical_weak_form = ToPhysicalProjectionWeakForm(fe_space)
    to_physical_weak_form.functions = m_functions

    def scatter_form_data(jac_g, res_g, i, weak_form):
        # destination indexes
        dest = weak_form.space.destination_indexes(i)
        alpha_l = alpha[dest]
        # print("i: ", i)
        r_el, j_el = weak_form.evaluate_form(i, alpha_l)

        # contribute rhs
        res_g[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        for k in range(nnz):
            jac_g.setValue(row=row[k], col=col[k], value=data[k], addv=True)

    def scatter_form_data_mapping(jac_g, res_g, i, weak_form):
        # destination indexes
        dest = weak_form.space.destination_indexes(i)
        alpha_unscaled_l = alpha_unscaled[dest]
        alpha_l = alpha[dest]
        r_el, j_el = weak_form.evaluate_form(i, alpha_unscaled_l, alpha_l)

        # contribute rhs
        res_g[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        for k in range(nnz):
            jac_g.setValue(row=row[k], col=col[k], value=data[k], addv=True)

    def scatter_bc_form(jac_g, res_g, i, bc_weak_form):
        dest = fe_space.bc_destination_indexes(i)
        alpha_l = alpha[dest]
        r_el, j_el = bc_weak_form.evaluate_form(i, alpha_l)

        # contribute rhs
        res_g[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        for k in range(nnz):
            jac_g.setValue(row=row[k], col=col[k], value=data[k], addv=True)

    # Assembler
    st = time.time()
    jac_g = PETSc.Mat()
    jac_g.createAIJ([n_dof_g, n_dof_g])

    res_g = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    # initial guess
    alpha = np.zeros(n_dof_g)

    for iter in range(n_iterations):
        n_els = len(fe_space.discrete_spaces["v"].elements)
        [scatter_form_data(jac_g, res_g, i, weak_form) for i in range(n_els)]

        n_bc_els = len(fe_space.discrete_spaces["v"].bc_elements)
        [scatter_bc_form(jac_g, res_g, i, bc_weak_form) for i in range(n_bc_els)]

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
        alpha += delta_alpha

        # Set up to zero lhr and rhs
        res_g *= 0.0
        jac_g.scale(0.0)

    st = time.time()
    (
        v_l2_error,
        q_l2_error,
    ) = l2_error(dim, fe_space, exact_functions, alpha)

    alpha_proj = l2_projector(fe_space, exact_functions)
    alpha_e = alpha - alpha_proj
    q_proj_l2_error = l2_error_projected(
        dim, fe_space, alpha_e, ["v"]
    )[0]

    # mapping variables to physical domain
    alpha_unscaled = np.zeros(n_dof_g)
    operator_lhs_g = PETSc.Mat()
    operator_lhs_g.createAIJ([n_dof_g, n_dof_g])
    operator_rhs_g = np.zeros(n_dof_g)
    n_els = len(fe_space.discrete_spaces["v"].elements)
    [scatter_form_data_mapping(operator_lhs_g, operator_rhs_g, i, to_physical_weak_form) for i in range(n_els)]
    operator_lhs_g.assemble()

    # solving ls
    st = time.time()
    ksp = PETSc.KSP().create()
    ksp.setOperators(operator_lhs_g)
    b = operator_lhs_g.createVecLeft()
    b.array[:] = -operator_rhs_g
    x = operator_lhs_g.createVecRight()

    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setConvergenceHistory()
    ksp.solve(b, x)
    alpha_unscaled = x.array

    physical_exact_functions = {
        "v": u_exact,
        "q": p_exact,
    }

    (
        u_l2_error,
        p_l2_error,
    ) = l2_error(dim, fe_space, physical_exact_functions, alpha_unscaled)

    alpha_unscaled_proj = l2_projector(fe_space, physical_exact_functions)
    alpha_unscaled_e = alpha_unscaled - alpha_unscaled_proj
    p_proj_l2_error = l2_error_projected(
        dim, fe_space, alpha_unscaled_e, ["v"]
    )[0]

    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    print("L2-error in v: ", v_l2_error)
    print("L2-error in q: ", q_l2_error)
    print("L2-error in u: ", u_l2_error)
    print("L2-error in p: ", p_l2_error)
    print("L2-error in q projected: ", q_proj_l2_error)
    print("L2-error in p projected: ", p_proj_l2_error)

    if write_vtk_q:
        # post-process solution
        st = time.time()
        file_name = "rates_arbogast_two_fields.vtk"
        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, exact_functions, alpha
        )
        file_name = "rates_arbogast_physical_two_fields.vtk"
        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, physical_exact_functions, alpha_unscaled
        )
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return [q_l2_error, v_l2_error, p_l2_error, u_l2_error, q_proj_l2_error, p_proj_l2_error]


def create_domain(dimension):
    if dimension == 1:
        box_points = np.array([[-1, 0, 0], [1, 0, 0]])
        domain = build_box_1D(box_points)
        return domain
    elif dimension == 2:
        box_points = np.array([[-1.0, -1.0, 0], [1, -1.0, 0], [1, 1, 0], [-1.0, 1, 0]])
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
    h = 2.0
    n_ref = 5
    dimension = 2
    ref_l = 0

    domain = create_domain(dimension)

    n_data = 7
    error_data = np.empty((0, n_data), float)
    for method in method_definition(k_order):
        for l in range(n_ref):
            h_val = h * (2**-l)
            mesher = create_conformal_mesher(domain, h, l)
            gmesh = create_mesh(dimension, mesher, True)
            error_val = two_fields_formulation(method, gmesh, True)
            error_data = np.append(error_data, np.array([[h_val] + error_val]), axis=0)

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

    rates_header = "q,  v,  p,  u,  proj q, proj p "
    error_header = "h,  " + rates_header
    np.savetxt("error_data.txt", error_data, delimiter=',', fmt="%1.4f", header=error_header)
    np.savetxt("rates_data.txt", rates_data, delimiter=',', fmt="%1.4f", header=rates_header)

    x = error_data[:, 0]
    y = error_data[:, 1:n_data]
    lineObjects = plt.loglog(x, y)
    plt.legend(iter(lineObjects), ("q", "v", "p", "u", "projected q", "projected p"))
    plt.title("")
    plt.xlabel("Element size")
    plt.ylabel("L2-error")
    plt.show()


if __name__ == "__main__":
    main()
