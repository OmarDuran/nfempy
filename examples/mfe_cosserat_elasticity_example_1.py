import functools
import gc
import time
import resource

import numpy as np
import strong_solution_cosserat_elasticity_example_1 as lce
from petsc4py import PETSc

from geometry.domain import Domain
from geometry.domain_market import build_box_1D, build_box_2D, build_box_3D
from mesh.conformal_mesher import ConformalMesher
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
from weak_forms.lce_dual_weak_form import LCEDualWeakForm, LCEDualWeakFormBCDirichlet
from weak_forms.lce_scaled_dual_weak_form import (
    LCEScaledDualWeakForm,
    LCEScaledDualWeakFormBCDirichlet,
)


def compose_file_name(method, k_order, ref_l, dim, material_data, suffix):
    prefix = (
        method[0]
        + "_k"
        + str(k_order)
        + "_l"
        + str(ref_l)
        + "_d"
        + str(dim)
        + "_gamma_"
        + str(material_data["gamma"])
    )
    file_name = prefix + suffix
    return file_name


def four_field_formulation(material_data, method, gmesh, write_vtk_q=False):
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
    m_lambda = material_data["lambda"]
    m_mu = material_data["mu"]
    m_kappa = material_data["kappa"]
    m_gamma = material_data["gamma"]

    # exact solution
    u_exact = lce.displacement(m_lambda, m_mu, m_kappa, m_gamma, dim)
    t_exact = lce.rotation(m_lambda, m_mu, m_kappa, m_gamma, dim)
    s_exact = lce.stress(m_lambda, m_mu, m_kappa, m_gamma, dim)
    m_exact = lce.couple_stress(m_lambda, m_mu, m_kappa, m_gamma, dim)
    div_s_exact = lce.stress_divergence(m_lambda, m_mu, m_kappa, m_gamma, dim)
    div_m_exact = lce.couple_stress_divergence(m_lambda, m_mu, m_kappa, m_gamma, dim)
    f_rhs = lce.rhs(m_lambda, m_mu, m_kappa, m_gamma, dim)

    def f_lambda(x, y, z):
        return m_lambda

    def f_mu(x, y, z):
        return m_mu

    def f_kappa(x, y, z):
        return m_kappa

    def f_gamma(x, y, z):
        return m_gamma

    m_functions = {
        "rhs": f_rhs,
        "lambda": f_lambda,
        "mu": f_mu,
        "kappa": f_kappa,
        "gamma": f_gamma,
    }

    exact_functions = {
        "s": s_exact,
        "m": m_exact,
        "u": u_exact,
        "t": t_exact,
        "div_s": div_s_exact,
        "div_m": div_m_exact,
    }

    weak_form = LCEDualWeakForm(fe_space)
    weak_form.functions = m_functions
    bc_weak_form = LCEDualWeakFormBCDirichlet(fe_space)
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
        nnz_idx = np.where(np.logical_not(np.isclose(data, 1.0e-16)))[0]
        [
            A.setValue(row=row[idx], col=col[idx], value=data[idx], addv=True)
            for idx in nnz_idx
        ]
        # Petsc ILU requires explicit existence of diagonal zeros
        [
            A.setValue(row=idx, col=idx, value=0.0, addv=True)
            for idx in dest
        ]

        check_points = [(int(k * n_els / 10)) for k in range(11)]
        if i in check_points or i == n_els - 1:
            if i == n_els - 1:
                print("Assembly: progress [%]: ", 100)
            else:
                print("Assembly: progress [%]: ", check_points.index(i) * 10)
                print(
                            "Assembly: Memory used :",
                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - memory_start,
                        )

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

    et = time.time()
    elapsed_time = et - st
    print("Assembly time:", elapsed_time, "seconds")

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

    ksp.setType("tfqmr")
    ksp.setTolerances(rtol=0.0, atol=1e-7, divtol=5000, max_it=20000)
    ksp.setConvergenceHistory()
    ksp.getPC().setType("ilu")

    ksp.solve(b, x)
    alpha = x.array
    residuals_history = ksp.getConvergenceHistory()

    PETSc.KSP.destroy(ksp)
    PETSc.Mat.destroy(A)
    PETSc.Vec.destroy(b)
    PETSc.Vec.destroy(x)

    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    st = time.time()
    s_l2_error, m_l2_error, u_l2_error, t_l2_error = l2_error(
        dim, fe_space, exact_functions, alpha
    )
    div_s_l2_error, div_m_l2_error = div_error(dim, fe_space, exact_functions, alpha)
    h_div_s_error = np.sqrt((s_l2_error**2) + (div_s_l2_error**2))
    h_div_m_error = np.sqrt((m_l2_error**2) + (div_m_l2_error**2))
    et = time.time()
    elapsed_time = et - st
    print("Error time:", elapsed_time, "seconds")
    print("L2-error displacement: ", u_l2_error)
    print("L2-error rotation: ", t_l2_error)
    print("L2-error stress: ", s_l2_error)
    print("L2-error couple stress: ", m_l2_error)
    print("L2-error div stress: ", div_s_l2_error)
    print("L2-error div couple stress: ", div_m_l2_error)
    print("")

    if write_vtk_q:
        st = time.time()

        lambda_value = material_data["lambda"]
        gamma_value = material_data["gamma"]

        prefix = method[0] + "_k" + str(s_k_order) + "_d" + str(dim)
        prefix += "_lambda_" + str(lambda_value) + "_gamma_" + str(gamma_value)
        file_name = prefix + "_four_fields_ex_1.vtk"

        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, exact_functions, alpha
        )
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

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
    ), residuals_history


def four_field_scaled_formulation(material_data, method, gmesh, write_vtk_q=False):
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
    m_lambda = material_data["lambda"]
    m_mu = material_data["mu"]
    m_kappa = material_data["kappa"]
    m_gamma = material_data["gamma"]

    # exact solution
    u_exact = lce.displacement(m_lambda, m_mu, m_kappa, m_gamma, dim)
    t_exact = lce.rotation(m_lambda, m_mu, m_kappa, m_gamma, dim)
    s_exact = lce.stress(m_lambda, m_mu, m_kappa, m_gamma, dim)
    m_exact = lce.couple_stress_scaled(m_lambda, m_mu, m_kappa, m_gamma, dim)
    div_s_exact = lce.stress_divergence(m_lambda, m_mu, m_kappa, m_gamma, dim)
    div_m_exact = lce.couple_stress_divergence_scaled(
        m_lambda, m_mu, m_kappa, m_gamma, dim
    )
    f_rhs = lce.rhs_scaled(m_lambda, m_mu, m_kappa, m_gamma, dim)

    def f_lambda(x, y, z):
        return m_lambda

    def f_mu(x, y, z):
        return m_mu

    def f_kappa(x, y, z):
        return m_kappa

    def f_gamma(x, y, z):
        return m_gamma * np.ones_like(x)

    def f_grad_gamma(x, y, z):
        d_gamma_x = 0.0 * x
        d_gamma_y = 0.0 * y
        return np.array([d_gamma_x, d_gamma_y])

    if dim == 3:

        def f_grad_gamma(x, y, z):
            d_gamma_x = 0.0 * x
            d_gamma_y = 0.0 * y
            d_gamma_z = 0.0 * z
            return np.array([d_gamma_x, d_gamma_y, d_gamma_z])

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
        nnz_idx = np.where(np.logical_not(np.isclose(data, 1.0e-16)))[0]
        [
            A.setValue(row=row[idx], col=col[idx], value=data[idx], addv=True)
            for idx in nnz_idx
        ]
        # Petsc ILU requires explicit existence of diagonal zeros
        [
            A.setValue(row=idx, col=idx, value=0.0, addv=True)
            for idx in dest
        ]

        check_points = [(int(k * n_els / 10)) for k in range(11)]
        if i in check_points or i == n_els - 1:
            if i == n_els - 1:
                print("Assembly: progress [%]: ", 100)
            else:
                print("Assembly: progress [%]: ", check_points.index(i) * 10)
            print(
                "Assembly: Memory used :",
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - memory_start,
            )

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

    et = time.time()
    elapsed_time = et - st
    print("Assembly time:", elapsed_time, "seconds")

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

    ksp.setType("tfqmr")
    ksp.setTolerances(rtol=0.0, atol=1e-7, divtol=5000, max_it=20000)
    ksp.setConvergenceHistory()
    ksp.getPC().setType("ilu")

    ksp.solve(b, x)
    alpha = x.array
    residuals_history = ksp.getConvergenceHistory()

    PETSc.KSP.destroy(ksp)
    PETSc.Mat.destroy(A)
    PETSc.Vec.destroy(b)
    PETSc.Vec.destroy(x)

    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    st = time.time()
    s_l2_error, m_l2_error, u_l2_error, t_l2_error = l2_error(
        dim, fe_space, exact_functions, alpha
    )
    div_s_l2_error = div_error(dim, fe_space, exact_functions, alpha, ["m"])[0]
    div_m_l2_error = div_scaled_error(dim, fe_space, exact_functions, alpha, ["s"])[0]
    h_div_s_error = np.sqrt((s_l2_error**2) + (div_s_l2_error**2))
    h_div_m_error = np.sqrt((m_l2_error**2) + (div_m_l2_error**2))
    et = time.time()
    elapsed_time = et - st
    print("Error time:", elapsed_time, "seconds")
    print("L2-error displacement: ", u_l2_error)
    print("L2-error rotation: ", t_l2_error)
    print("L2-error stress: ", s_l2_error)
    print("L2-error couple stress star: ", m_l2_error)
    print("L2-error div stress: ", div_s_l2_error)
    print("L2-error div couple stress: ", div_m_l2_error)
    print(" ", div_m_l2_error)

    if write_vtk_q:
        st = time.time()

        lambda_value = material_data["lambda"]
        gamma_value = material_data["gamma"]

        prefix = method[0] + "_k" + str(s_k_order) + "_d" + str(dim)
        prefix += "_lambda_" + str(lambda_value) + "_gamma_" + str(gamma_value)
        file_name = prefix + "_four_fields_scaled_ex_1.vtk"

        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, exact_functions, alpha
        )
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

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
    ), residuals_history


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


def create_mesh_from_file(file_name, dim, write_vtk_q=False):
    gmesh = Mesh(dimension=dim, file_name=file_name)
    gmesh.build_conformal_mesh()
    if write_vtk_q:
        gmesh.write_vtk()
    return gmesh


def perform_convergence_test(configuration: dict):
    # retrieve parameters from given configuration
    k_order = configuration.get("k_order")
    method = configuration.get("method")
    n_ref = configuration.get("n_refinements")
    dimension = configuration.get("dimension")
    dual_form_q = configuration.get("dual_problem_Q", True)
    material_data = configuration.get("material_data", {})
    write_geometry_vtk = configuration.get("write_geometry_Q", True)
    write_vtk = configuration.get("write_vtk_Q", True)
    report_full_precision_data = configuration.get("report_full_precision_data_Q", True)

    # The initial element size
    h = 1.0

    # Create a unit squared or a unit cube
    domain = create_domain(dimension)

    n_data = 10
    error_data = np.empty((0, n_data), float)
    for lh in range(n_ref):
        h_val = h * (2**-lh)
        mesher = create_conformal_mesher(domain, h, lh)
        gmesh = create_mesh(dimension, mesher, write_geometry_vtk)
        h_min, h_mean, h_max = mesh_size(gmesh)
        if method[0] == "m2_dnc":
            n_dof, error_vals, res_history = four_field_scaled_formulation(
                material_data, method, gmesh, write_vtk
            )
        else:
            n_dof, error_vals, res_history = four_field_formulation(
                material_data, method, gmesh, write_vtk
            )
        file_name_res = compose_file_name(
            method, k_order, lh, gmesh.dimension, material_data, "_res_history_ex_1.txt"
        )
        np.savetxt(file_name_res,res_history,delimiter=",",)
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
    e_str_header = "n_dof, h, " + base_str_header

    lambda_value = material_data["lambda"]
    gamma_value = material_data["gamma"]

    file_name_prefix = (
        method[0]
        + "_lambda_"
        + str(lambda_value)
        + "_gamma_"
        + str(gamma_value)
        + "_k"
        + str(k_order)
        + "_"
        + str(dimension)
    )
    if report_full_precision_data:
        np.savetxt(
            file_name_prefix + "d_error_ex_1.txt",
            error_data,
            delimiter=",",
            header=e_str_header,
        )
        np.savetxt(
            file_name_prefix + "d_rates_ex_1.txt",
            rates_data,
            delimiter=",",
            header=base_str_header,
        )
    np.savetxt(
        file_name_prefix + "d_error_ex_1_rounded.txt",
        error_data,
        fmt="%1.3e",
        delimiter=",",
        header=e_str_header,
    )
    np.savetxt(
        file_name_prefix + "d_rates_ex_1_rounded.txt",
        rates_data,
        fmt="%1.3f",
        delimiter=",",
        header=base_str_header,
    )

    return


def method_definition(k_order):
    method_1_dc = {
        "s": ("RT", k_order),
        "m": ("RT", k_order + 1),
        "u": ("Lagrange", k_order - 1),
        "t": ("Lagrange", k_order),
    }

    method_2_dnc = {
        "s": ("BDM", k_order),
        "m": ("RT", k_order),
        "u": ("Lagrange", k_order - 1),
        "t": ("Lagrange", k_order - 1),
    }

    method_3_dc = {
        "s": ("BDM", k_order),
        "m": ("BDM", k_order + 1),
        "u": ("Lagrange", k_order - 1),
        "t": ("Lagrange", k_order),
    }

    methods = [method_1_dc, method_2_dnc, method_3_dc]
    method_names = ["m1_dc", "m2_dnc", "m3_dnc"]
    return zip(method_names, methods)


def material_data_definition():
    # Material data for example 1
    case_0 = {"lambda": 1.0, "mu": 1.0, "kappa": 1.0, "gamma": 1.0}
    case_1 = {"lambda": 1.0, "mu": 1.0, "kappa": 1.0, "gamma": 1.0e-2}
    case_2 = {"lambda": 1.0, "mu": 1.0, "kappa": 1.0, "gamma": 1.0e-4}
    cases = [case_0, case_1, case_2]

    return cases


def main():
    refinements = {1: 4, 2: 4}
    case_data = material_data_definition()
    for k in [2, 1]:
        n_ref = refinements[k]
        methods = method_definition(k)
        for i, method in enumerate(methods):
            for material_data in case_data:
                configuration = {
                    "n_refinements": n_ref,
                    "method": method,
                    "material_data": material_data,
                }

                for d in [3]:
                    configuration.__setitem__("k_order", k)
                    configuration.__setitem__("dimension", d)
                    perform_convergence_test(configuration)


if __name__ == "__main__":
    main()
