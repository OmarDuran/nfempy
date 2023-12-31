import functools
import gc
import time
import resource

import numpy as np
import strong_solution_cosserat_elasticity_example_3 as lce

import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

from mesh.mesh import Mesh
from mesh.mesh_metrics import mesh_size
from postprocess.l2_error_post_processor import (
    div_error,
    div_scaled_error,
    grad_error,
    l2_error,
)
from postprocess.solution_post_processor import write_vtk_file_with_exact_solution
from spaces.product_space import ProductSpace
from weak_forms.lce_scaled_dual_weak_form import (
    LCEScaledDualWeakForm,
    LCEScaledDualWeakFormBCDirichlet,
)


def four_field_scaled_formulation(method, gmesh, write_vtk_q=False):
    dim = gmesh.dimension

    # FESpace: data
    s_k_order = method[1]["s"][1]
    m_k_order = method[1]["m"][1]
    u_k_order = method[1]["u"][1]
    t_k_order = method[1]["t"][1]

    s_components = 2
    m_components = 1
    u_components = 2
    t_components = 1
    if dim == 3:
        s_components = 3
        m_components = 3
        u_components = 3
        t_components = 3

    s_family = method[1]["s"][0]
    m_family = method[1]["m"][0]
    u_family = method[1]["u"][0]
    t_family = method[1]["t"][0]

    discrete_spaces_data = {
        "s": (dim, s_components, s_family, s_k_order, gmesh),
        "m": (dim, m_components, m_family, m_k_order, gmesh),
        "u": (dim, u_components, u_family, u_k_order, gmesh),
        "t": (dim, t_components, t_family, t_k_order, gmesh),
    }

    s_disc_Q = False
    m_disc_Q = False
    u_disc_Q = True
    t_disc_Q = True
    discrete_spaces_disc = {
        "s": s_disc_Q,
        "m": m_disc_Q,
        "u": u_disc_Q,
        "t": t_disc_Q,
    }

    s_field_bc_physical_tags = [2, 3, 4, 5]
    m_field_bc_physical_tags = [2, 3, 4, 5]
    if dim == 3:
        s_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
        m_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
    discrete_spaces_bc_physical_tags = {
        "s": s_field_bc_physical_tags,
        "m": m_field_bc_physical_tags,
    }

    fe_space = ProductSpace(discrete_spaces_data)
    fe_space.make_subspaces_discontinuous(discrete_spaces_disc)
    fe_space.build_structures(discrete_spaces_bc_physical_tags)

    n_dof_g = fe_space.n_dof
    rg = np.zeros(n_dof_g)
    alpha = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    memory_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Assembler
    st = time.time()

    A = PETSc.Mat()
    A.createAIJ([n_dof_g, n_dof_g])

    # Material data
    m_lambda = 1.0
    m_mu = 1.0
    m_kappa = m_mu

    # exact solution
    u_exact = lce.displacement(m_lambda, m_mu, m_kappa, dim)
    t_exact = lce.rotation(m_lambda, m_mu, m_kappa, dim)
    s_exact = lce.stress(m_lambda, m_mu, m_kappa, dim)
    m_exact = lce.couple_stress_scaled(m_lambda, m_mu, m_kappa, dim)
    div_s_exact = lce.stress_divergence(m_lambda, m_mu, m_kappa, dim)
    div_m_exact = lce.couple_stress_divergence_scaled(m_lambda, m_mu, m_kappa, dim)
    f_rhs = lce.rhs_scaled(m_lambda, m_mu, m_kappa, dim)

    def f_lambda(x, y, z):
        return m_lambda

    def f_mu(x, y, z):
        return m_mu

    def f_kappa(x, y, z):
        return m_kappa

    f_gamma = lce.gamma_s(dim)
    f_grad_gamma = lce.grad_gamma_s(dim)

    m_functions = {
        "rhs": f_rhs,
        "lambda": f_lambda,
        "mu": f_mu,
        "kappa": f_kappa,
        "gamma": f_gamma,
        "grad_gamma": f_grad_gamma,
    }

    exact_functions = {
        "s": s_exact,
        "m": m_exact,
        "u": u_exact,
        "t": t_exact,
        "div_s": div_s_exact,
        "div_m": div_m_exact,
        "gamma": f_gamma,
        "grad_gamma": f_grad_gamma,
    }

    weak_form = LCEScaledDualWeakForm(fe_space)
    weak_form.functions = m_functions
    bc_weak_form = LCEScaledDualWeakFormBCDirichlet(fe_space)
    bc_weak_form.functions = exact_functions

    def scatter_form_data(A, i, weak_form, n_els):
        # destination indexes
        dest = weak_form.space.destination_indexes(i)
        alpha_l = alpha[dest]
        r_el, j_el = weak_form.evaluate_form_vectorized(i, alpha_l)

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        [A.setValue(row=row[k], col=col[k], value=data[k], addv=True) for k in range(nnz)]

        check_points = [(int(k * n_els / 10)) for k in range(11)]
        if i in check_points or i == n_els - 1:
            if i == n_els - 1:
                print("Assembly: progress [%]: ", 100)
            else:
                print("Assembly: progress [%]: ", check_points.index(i)*10)
            print("Assembly: Memory used [GiB] :", (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - memory_start)/1073741824)

    def scatter_bc_form(A, i, bc_weak_form):
        dest = fe_space.bc_destination_indexes(i)
        alpha_l = alpha[dest]
        r_el, j_el = bc_weak_form.evaluate_form(i, alpha_l)

        # contribute rhs
        rg[dest] += r_el

    n_els = len(fe_space.discrete_spaces["s"].elements)
    [scatter_form_data(A, i, weak_form, n_els) for i in range(n_els)]

    n_bc_els = len(fe_space.discrete_spaces["s"].bc_elements)
    [scatter_bc_form(A, i, bc_weak_form) for i in range(n_bc_els)]

    A.assemble()
    print("Assembly: nz_allocated:", int(A.getInfo()['nz_allocated']))
    print("Assembly: nz_used:", int(A.getInfo()['nz_used']))
    print("Assembly: nz_unneeded:", int(A.getInfo()['nz_unneeded']))

    et = time.time()
    elapsed_time = et - st
    print("Assembly: Time:", elapsed_time, "seconds")
    print("Assembly: After PETSc M.assemble: Memory used [GiB] :", (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - memory_start) / 1073741824)

    # memory_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # solving ls
    st = time.time()

    ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
    ksp.setOperators(A)
    b = A.createVecLeft()
    b.array[:] = -rg
    x = A.createVecRight()

    # ksp.setType("preonly")
    # ksp.getPC().setType("lu")
    # ksp.getPC().setFactorSolverType("mumps")
    # ksp.setConvergenceHistory()

    ksp.setType("pgmres")
    ksp.setTolerances(rtol=1e-10, atol=1e-10, divtol=5000, max_it=20000)
    ksp.setConvergenceHistory()
    ksp.getPC().setType("ilu")
    ksp.getPC().setFactorSolverType("superlu")

    ksp.solve(b, x)
    alpha = x.array
    print("Linear solver: After PETSc ksp.solve: Memory used [GiB] :", (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - memory_start) / 1073741824)
    PETSc.KSP.destroy(ksp)
    PETSc.Mat.destroy(A)
    PETSc.Vec.destroy(b)
    PETSc.Vec.destroy(x)

    et = time.time()
    elapsed_time = et - st
    print("Linear solver: Time:", elapsed_time, "seconds")
    print("Linear solver: After PETSc ksp.destroy: Memory used [GiB] :", (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - memory_start) / 1073741824)

    st = time.time()
    s_l2_error, m_l2_error, u_l2_error, t_l2_error = l2_error(
        dim, fe_space, exact_functions, alpha
    )
    div_s_l2_error = div_error(dim, fe_space, exact_functions, alpha, ["m"])[0]
    div_m_l2_error = div_scaled_error(dim, fe_space, exact_functions, alpha, ["s"])[0]

    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    print("L2-error displacement: ", u_l2_error)
    print("L2-error rotation: ", t_l2_error)
    print("L2-error stress: ", s_l2_error)
    print("L2-error couple stress: ", m_l2_error)
    print("L2-error div stress: ", div_s_l2_error)
    print("L2-error div couple stress: ", div_m_l2_error)
    print("")

    if write_vtk_q:
        st = time.time()

        prefix = method[0] + "_k" + str(s_k_order) + "_d" + str(dim)
        file_name = prefix + "_four_fields_scaled_ex_3.vtk"

        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, exact_functions, alpha
        )
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    h_div_s_error = np.sqrt((s_l2_error**2) + (div_s_l2_error**2))
    h_div_m_error = np.sqrt((m_l2_error**2) + (div_m_l2_error**2))

    gc.collect()
    return n_dof_g, np.array(
        [
            u_l2_error,
            t_l2_error,
            s_l2_error,
            m_l2_error,
            div_s_l2_error,
            div_m_l2_error,
            h_div_s_error,
            h_div_m_error,
        ]
    )


def create_mesh_from_file(file_name, dim, write_vtk_q=False):
    gmesh = Mesh(dimension=dim, file_name=file_name)
    gmesh.build_conformal_mesh()
    if write_vtk_q:
        gmesh.write_vtk()
    return gmesh


def perform_convergence_test(configuration: dict):
    # retrieve parameters from dictionary
    k_order = configuration.get("k_order")
    method = configuration.get("method")
    n_ref = configuration.get("n_refinements")
    dimension = configuration.get("dimension")
    dual_form_q = configuration.get("dual_problem_Q", True)
    gamma_value = configuration.get("gamma_value", 1.0)
    write_geometry_vtk = configuration.get("write_geometry_Q", True)
    write_vtk = configuration.get("write_vtk_Q", True)
    report_full_precision_data = configuration.get("report_full_precision_data_Q", True)

    # The initial element size
    h = 1.0 / 3.0

    n_data = 10
    error_data = np.empty((0, n_data), float)
    for lh in range(n_ref):
        mesh_file = "gmsh_files/example_2_" + str(dimension) + "d_l_" + str(lh) + ".msh"
        gmesh = create_mesh_from_file(mesh_file, dimension, write_geometry_vtk)
        h_min, h_mean, h_max = mesh_size(gmesh)
        n_dof, error_vals = four_field_scaled_formulation(method, gmesh, write_vtk)
        chunk = np.concatenate([[n_dof, h_max], error_vals])
        error_data = np.append(error_data, np.array([chunk]), axis=0)

    rates_data = np.empty((0, n_data - 2), float)
    for i in range(error_data.shape[0] - 1):
        chunk_b = np.log(error_data[i])
        chunk_e = np.log(error_data[i + 1])
        h_step = chunk_e[1] - chunk_b[1]
        partial = (chunk_e - chunk_b) / h_step
        rates_data = np.append(rates_data, np.array([list(partial[2:n_data])]), axis=0)

    # minimal report
    if report_full_precision_data:
        print("error data: ", error_data)
        print("error rates data: ", rates_data)

    np.set_printoptions(precision=3)
    print("Dual problem: ", method[0])

    print("Polynomial order: ", k_order)
    print("Dimension: ", dimension)
    print("rounded error data: ", error_data)
    print("rounded error rates data: ", rates_data)
    print(" ")

    str_fields = "u, r, s, o, "
    primal_header = str_fields + "grad_u, grad_r, h_grad_u_norm, h_grad_r_norm"
    dual_header = str_fields + "div_s, div_o, h_div_s_norm, h_div_o_norm"

    base_str_header = primal_header
    if dual_form_q:
        base_str_header = dual_header
    e_str_header = "h, " + base_str_header

    file_name_prefix = method[0] + "_k" + str(k_order) + "_" + str(dimension)
    if report_full_precision_data:
        np.savetxt(
            file_name_prefix + "d_error_ex_3.txt",
            error_data,
            delimiter=",",
            header=e_str_header,
        )
        np.savetxt(
            file_name_prefix + "d_rates_ex_3.txt",
            rates_data,
            delimiter=",",
            header=base_str_header,
        )
    np.savetxt(
        file_name_prefix + "d_error_ex_3_rounded.txt",
        error_data,
        fmt="%1.3e",
        delimiter=",",
        header=e_str_header,
    )
    np.savetxt(
        file_name_prefix + "d_rates_ex_3_rounded.txt",
        rates_data,
        fmt="%1.3f",
        delimiter=",",
        header=base_str_header,
    )

    return


def method_definition(k_order):
    method_2_dnc = {
        "s": ("BDM", k_order),
        "m": ("RT", k_order),
        "u": ("Lagrange", k_order - 1),
        "t": ("Lagrange", k_order - 1),
    }
    methods = [method_2_dnc]
    method_names = ["m2_dnc"]
    return zip(method_names, methods)


def main():
    n_refinements = 3
    for k in [1]:
        for method in method_definition(k):
            configuration = {
                "n_refinements": n_refinements,
                "method": method,
            }

            for d in [3]:
                configuration.__setitem__("k_order", k)
                configuration.__setitem__("dimension", d)
                perform_convergence_test(configuration)


if __name__ == "__main__":
    main()
