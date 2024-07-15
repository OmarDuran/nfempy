import time

import numpy as np
from petsc4py import PETSc
import matplotlib.pyplot as plt

from topology.domain import Domain
from topology.domain_market import build_box_1D, build_box_2D, build_box_3D
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh
from postprocess.l2_error_post_processor import l2_error
from postprocess.solution_post_processor import (
    write_vtk_file_with_exact_solution,
    write_vtk_file_pointwise_l2_error,
)
from spaces.product_space import ProductSpace

from oden_primal_weak_form import OdenPrimalWeakForm, OdenPrimalWeakFormBCDirichlet
from oden_dual_weak_form import OdenDualWeakForm


def h1_model_problem(k_order, gmesh, write_vtk_q=False):
    dim = gmesh.dimension

    # FESpace: data
    u_k_order = k_order + 1

    u_components = 1
    family = "Lagrange"

    discrete_spaces_data = {
        "u": (dim, u_components, family, u_k_order, gmesh),
    }

    u_disc_Q = False
    discrete_spaces_disc = {
        "u": u_disc_Q,
    }

    u_field_bc_physical_tags = [2, 3]
    discrete_spaces_bc_physical_tags = {
        "u": u_field_bc_physical_tags,
    }

    fe_space = ProductSpace(discrete_spaces_data)
    fe_space.make_subspaces_discontinuous(discrete_spaces_disc)
    fe_space.build_structures(discrete_spaces_bc_physical_tags)

    n_dof_g = fe_space.n_dof
    rg = np.zeros(n_dof_g)
    alpha = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    # Assembler
    st = time.time()
    A = PETSc.Mat()
    A.createAIJ([n_dof_g, n_dof_g])

    # Material data
    m_kappa = 1.0

    def f_kappa(x, y, z):
        return m_kappa

    st = time.time()

    # exact solution
    if dim == 1:
        u_exact = lambda x, y, z: np.array(
            [
                -(
                    (
                        -np.e
                        + (np.e ** (1 + 2 * x))
                        + (np.e**x) * x
                        - (np.e ** (2 + x)) * x
                    )
                    / ((np.e**x) * (-1 + (np.e**2)))
                )
            ]
        )
        q_exact = lambda x, y, z: np.array(
            [
                -(
                    (
                        -np.e
                        + (np.e ** (1 + 2 * x))
                        + (np.e**x) * x
                        - (np.e ** (2 + x)) * x
                    )
                    / ((np.e**x) * (-1 + (np.e**2)))
                )
                + (
                    (np.e**x)
                    - (np.e ** (2 + x))
                    + 2 * (np.e ** (1 + 2 * x))
                    + (np.e**x) * x
                    - (np.e ** (2 + x)) * x
                )
                / ((np.e**x) * (-1 + (np.e**2))),
            ]
        )
        f_rhs = lambda x, y, z: np.array([[x]])

    elif dim == 2:
        u_exact = lambda x, y, z: np.array(
            [
                (x - 2 * np.e * np.sinh(x) / (-1 + np.e**2))
                * (y - 2 * np.e * np.sinh(y) / (-1 + np.e**2))
            ]
        )
        q_exact = lambda x, y, z: np.array(
            [
                -(
                    (1 - 2 * np.e * np.cosh(x) / (-1 + np.e**2))
                    * (y - 2 * np.e * np.sinh(y) / (-1 + np.e**2))
                ),
                -(
                    (1 - 2 * np.e * np.cosh(y) / (-1 + np.e**2))
                    * (x - 2 * np.e * np.sinh(x) / (-1 + np.e**2))
                ),
            ]
        )
        f_rhs = lambda x, y, z: np.array(
            [
                [
                    (
                        ((-1 + np.e**2) ** 2 * x * y)
                        - (4 * np.e**2 * np.sinh(x) * np.sinh(y))
                    )
                    / ((-1 + np.e**2) ** 2)
                ]
            ]
        )
    else:
        raise ValueError("Case not implemented.")

    m_functions = {
        "rhs": f_rhs,
        "kappa": f_kappa,
    }

    exact_functions = {
        "u": u_exact,
    }

    weak_form = OdenPrimalWeakForm(fe_space)
    weak_form.functions = m_functions
    bc_weak_form = OdenPrimalWeakFormBCDirichlet(fe_space)
    bc_weak_form.functions = exact_functions

    def scatter_form_data(A, i, weak_form):
        # destination indexes
        dest = weak_form.space.destination_indexes(i)
        alpha_l = alpha[dest]
        r_el, j_el = weak_form.evaluate_form(i, alpha_l)

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        for k in range(nnz):
            A.setValue(row=row[k], col=col[k], value=data[k], addv=True)

    def scatter_bc_form(A, i, bc_weak_form):
        dest = fe_space.bc_destination_indexes(i)
        alpha_l = alpha[dest]
        r_el, j_el = bc_weak_form.evaluate_form(i, alpha_l)

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        for k in range(nnz):
            A.setValue(row=row[k], col=col[k], value=data[k], addv=True)

    n_els = len(fe_space.discrete_spaces["u"].elements)
    [scatter_form_data(A, i, weak_form) for i in range(n_els)]

    n_bc_els = len(fe_space.discrete_spaces["u"].bc_elements)
    [scatter_bc_form(A, i, bc_weak_form) for i in range(n_bc_els)]

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

    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setConvergenceHistory()

    # petsc_options = {"rtol": 1e-10, "atol": 1e-12, "divtol": 200, "max_it": 500}
    # ksp = PETSc.KSP().create()
    # ksp.create(PETSc.COMM_WORLD)
    # ksp.setOperators(A)
    # ksp.setType("fcg")
    # # ksp.setTolerances(**petsc_options)
    # # ksp.setTolerances(1e-10)
    # ksp.setTolerances(rtol=1e-10, atol=1e-10, divtol=500, max_it=2000)
    # ksp.setConvergenceHistory()
    # ksp.getPC().setType("ilu")
    ksp.solve(b, x)
    alpha = x.array

    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    st = time.time()
    u_l2_error = l2_error(dim, fe_space, exact_functions, alpha)
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    print("L2-error: ", u_l2_error)

    if write_vtk_q:
        # post-process solution
        st = time.time()
        file_name = "rates_h1_laplace.vtk"
        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, exact_functions, alpha
        )
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return u_l2_error[0]


def hdiv_model_problem(k_order, gmesh, write_vtk_q=False):
    dim = gmesh.dimension

    # FESpace: data
    q_k_order = k_order + 1
    u_k_order = k_order

    q_components = 1
    u_components = 1
    q_family = "RT"
    u_family = "Lagrange"

    discrete_spaces_data = {
        "q": (dim, q_components, q_family, q_k_order, gmesh),
        "u": (dim, u_components, u_family, u_k_order, gmesh),
    }

    q_disc_Q = False
    u_disc_Q = True
    discrete_spaces_disc = {
        "q": q_disc_Q,
        "u": u_disc_Q,
    }

    q_field_bc_physical_tags = [2, 3, 4, 5]
    if dim == 3:
        q_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]

    physical_tags = {
        "q": [1],
        "u": [1],
    }

    b_physical_tags = {
        "q": q_field_bc_physical_tags,
    }

    fe_space = ProductSpace(discrete_spaces_data)
    fe_space.make_subspaces_discontinuous(discrete_spaces_disc)
    fe_space.build_structures(physical_tags, b_physical_tags)

    n_dof_g = fe_space.n_dof
    rg = np.zeros(n_dof_g)
    alpha = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    # Assembler
    st = time.time()
    A = PETSc.Mat()
    A.createAIJ([n_dof_g, n_dof_g])

    # Material data
    m_kappa = 1.0

    def f_kappa(x, y, z):
        return m_kappa

    st = time.time()

    # exact solution
    if dim == 1:
        u_exact = lambda x, y, z: np.array(
            [
                -(
                    (
                        -np.e
                        + (np.e ** (1 + 2 * x))
                        + (np.e**x) * x
                        - (np.e ** (2 + x)) * x
                    )
                    / ((np.e**x) * (-1 + (np.e**2)))
                )
            ]
        )
        q_exact = lambda x, y, z: np.array(
            [
                -(
                    (
                        -np.e
                        + (np.e ** (1 + 2 * x))
                        + (np.e**x) * x
                        - (np.e ** (2 + x)) * x
                    )
                    / ((np.e**x) * (-1 + (np.e**2)))
                )
                + (
                    (np.e**x)
                    - (np.e ** (2 + x))
                    + 2 * (np.e ** (1 + 2 * x))
                    + (np.e**x) * x
                    - (np.e ** (2 + x)) * x
                )
                / ((np.e**x) * (-1 + (np.e**2))),
            ]
        )
        div_q_exact = lambda x, y, z: np.array([[x]]) - u_exact(x, y, z)
        f_rhs = lambda x, y, z: np.array([[x]])

    elif dim == 2:
        u_exact = lambda x, y, z: np.array(
            [
                (x - 2 * np.e * np.sinh(x) / (-1 + np.e**2))
                * (y - 2 * np.e * np.sinh(y) / (-1 + np.e**2))
            ]
        )
        q_exact = lambda x, y, z: np.array(
            [
                [
                    -(
                        (1 - 2 * np.e * np.cosh(x) / (-1 + np.e**2))
                        * (y - 2 * np.e * np.sinh(y) / (-1 + np.e**2))
                    ),
                    -(
                        (1 - 2 * np.e * np.cosh(y) / (-1 + np.e**2))
                        * (x - 2 * np.e * np.sinh(x) / (-1 + np.e**2))
                    ),
                ]
            ]
        )

        f_rhs = lambda x, y, z: np.array(
            [
                [
                    (
                        ((-1 + np.e**2) ** 2 * x * y)
                        - (4 * np.e**2 * np.sinh(x) * np.sinh(y))
                    )
                    / ((-1 + np.e**2) ** 2)
                ]
            ]
        )

        div_q_exact = lambda x, y, z: np.array([[f_rhs(x, y, z)]]) - u_exact(x, y, z)
    else:
        raise ValueError("Case not implemented.")

    m_functions = {
        "rhs": f_rhs,
        "kappa": f_kappa,
    }

    exact_functions = {
        "q": q_exact,
        "u": u_exact,
        "div_q": div_q_exact,
    }

    weak_form = OdenDualWeakForm(fe_space)
    weak_form.functions = m_functions
    # bc_weak_form = LaplaceDualWeakFormBCDirichlet(fe_space)
    # bc_weak_form.functions = exact_functions

    def scatter_form_data(A, i, weak_form):
        # destination indexes
        dest = weak_form.space.destination_indexes(i)
        alpha_l = alpha[dest]
        r_el, j_el = weak_form.evaluate_form(i, alpha_l)

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        for k in range(nnz):
            A.setValue(row=row[k], col=col[k], value=data[k], addv=True)

    def scatter_bc_form(A, i, bc_weak_form):
        dest = fe_space.bc_destination_indexes(i)
        alpha_l = alpha[dest]
        r_el, j_el = bc_weak_form.evaluate_form(i, alpha_l)

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        for k in range(nnz):
            A.setValue(row=row[k], col=col[k], value=data[k], addv=True)

    n_els = len(fe_space.discrete_spaces["q"].elements)
    [scatter_form_data(A, i, weak_form) for i in range(n_els)]

    # n_bc_els = len(fe_space.discrete_spaces["q"].bc_elements)
    # [scatter_bc_form(A, i, bc_weak_form) for i in range(n_bc_els)]

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

    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setConvergenceHistory()

    ksp.solve(b, x)
    alpha = x.array

    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    st = time.time()
    q_l2_error, p_l2_error = l2_error(dim, fe_space, exact_functions, alpha)
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    print("L2-error in pressure: ", p_l2_error)
    print("L2-error in flux: ", q_l2_error)

    if write_vtk_q:
        # post-process solution
        st = time.time()
        file_name = "rates_hdiv_model_problem.vtk"
        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, exact_functions, alpha
        )

        file_name = "rates_hdiv_model_problem_l2_error.vtk"
        write_vtk_file_pointwise_l2_error(
            file_name, gmesh, fe_space, exact_functions, alpha
        )
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return [p_l2_error, q_l2_error]


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
    k_order = 1
    h = 0.25
    n_ref = 4  # no. of refinement
    dimension = 1
    ref_l = 0

    n_data = 3
    domain = create_domain(dimension)
    error_data = np.empty((0, n_data), float)
    for l in range(n_ref):
        h_val = h * (2**-l)
        mesher = create_conformal_mesher(domain, h_val, 0)
        gmesh = create_mesh(dimension, mesher, True)
        # error_val = h1_model_problem(k_order, gmesh, True)
        error_val = hdiv_model_problem(k_order, gmesh, True)
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

    np.savetxt(
        "rates_data_oden.txt",
        rates_data,
        delimiter=",",
        fmt="%1.5f",
        # header=rates_data_oden,
    )

    np.savetxt(
        "error_data_oden.txt",
        error_data,
        delimiter=",",
        fmt="%1.5f",
        # header=rates_data_oden,
    )

    x = error_data[:, 0]
    y = error_data[:, 1:n_data]
    lineObjects = plt.loglog(x, y)
    plt.legend(iter(lineObjects), ("u", "q"))
    plt.title("")
    plt.xlabel("Element size")
    plt.ylabel("L2-error")
    plt.show()
    return


if __name__ == "__main__":
    main()
