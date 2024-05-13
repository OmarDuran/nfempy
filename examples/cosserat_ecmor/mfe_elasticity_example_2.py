import resource
import time

import numpy as np
import strong_solution_elasticity_example_2 as le
from petsc4py import PETSc

from geometry.domain import Domain
from geometry.domain_market import build_box_1D, build_box_2D, build_box_3D
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh
from mesh.mesh_metrics import mesh_size
from postprocess.l2_error_post_processor import div_error, l2_error, l2_error_projected
from postprocess.projectors import l2_projector
from postprocess.solution_norms_post_processor import div_norm, l2_norm
from postprocess.solution_post_processor import write_vtk_file_with_exact_solution
from spaces.product_space import ProductSpace
from weak_forms.le_dual_weak_form import LEDualWeakForm, LEDualWeakFormBCDirichlet
from weak_forms.le_riesz_map_weak_form import LERieszMapWeakForm


def compose_file_name(method, k_order, ref_l, dim, material_data, suffix):
    prefix = (
        method[0]
        + "_lambda_"
        + str(material_data["lambda"])
        + "_mu_"
        + str(material_data["mu"])
        + "_k"
        + str(k_order)
        + "_l"
        + str(ref_l)
        + "_"
        + str(dim)
        + "d"
    )
    file_name = prefix + suffix
    return file_name


def create_product_space(method, gmesh):
    # FESpace: data
    s_k_order = method[1]["s"][1]
    u_k_order = method[1]["u"][1]
    t_k_order = method[1]["t"][1]

    s_components = 2
    u_components = 2
    t_components = 1
    if gmesh.dimension == 3:
        s_components = 3
        u_components = 3
        t_components = 3

    s_family = method[1]["s"][0]
    u_family = method[1]["u"][0]
    t_family = method[1]["t"][0]

    discrete_spaces_data = {
        "s": (gmesh.dimension, s_components, s_family, s_k_order, gmesh),
        "u": (gmesh.dimension, u_components, u_family, u_k_order, gmesh),
        "t": (gmesh.dimension, t_components, t_family, t_k_order, gmesh),
    }

    s_disc_Q = False
    u_disc_Q = True
    t_disc_Q = True
    discrete_spaces_disc = {
        "s": s_disc_Q,
        "u": u_disc_Q,
        "t": t_disc_Q,
    }

    s_field_bc_physical_tags = [2, 3, 4, 5]
    m_field_bc_physical_tags = [2, 3, 4, 5]
    if gmesh.dimension == 3:
        s_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
        m_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
    discrete_spaces_bc_physical_tags = {
        "s": s_field_bc_physical_tags,
        "m": m_field_bc_physical_tags,
    }

    space = ProductSpace(discrete_spaces_data)
    space.make_subspaces_discontinuous(discrete_spaces_disc)
    space.build_structures(discrete_spaces_bc_physical_tags)
    return space


def three_field_approximation(material_data, method, gmesh, symmetric_solver_q=True):
    dim = gmesh.dimension

    fe_space = create_product_space(method, gmesh)

    n_dof_g = fe_space.n_dof
    rg = np.zeros(n_dof_g)
    alpha = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    memory_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Assembler
    st = time.time()

    A = PETSc.Mat()
    A.createAIJ([n_dof_g, n_dof_g])
    if symmetric_solver_q:
        A.setType("sbaij")

    P = PETSc.Mat()
    P.createAIJ([n_dof_g, n_dof_g])
    if symmetric_solver_q:
        P.setType("sbaij")

    # Material data
    m_lambda = material_data["lambda"]
    m_mu = material_data["mu"]

    # exact solution
    u_exact = le.displacement(m_lambda, m_mu, dim)
    t_exact = le.rotation(m_lambda, m_mu, dim)
    s_exact = le.stress(m_lambda, m_mu, dim)
    f_rhs = le.rhs(m_lambda, m_mu, dim)

    def f_lambda(x, y, z):
        return m_lambda

    def f_mu(x, y, z):
        return m_mu

    m_functions = {
        "rhs": f_rhs,
        "lambda": f_lambda,
        "mu": f_mu,
    }

    exact_functions = {
        "s": s_exact,
        "u": u_exact,
        "t": t_exact,
    }

    weak_form = LEDualWeakForm(fe_space)
    weak_form.functions = m_functions
    bc_weak_form = LEDualWeakFormBCDirichlet(fe_space)
    bc_weak_form.functions = exact_functions

    riesz_map_weak_form = LERieszMapWeakForm(fe_space)
    riesz_map_weak_form.functions = m_functions

    def scatter_form_data(A, i, weak_form, n_els):
        # destination indexes
        dest = weak_form.space.destination_indexes(i)
        alpha_l = alpha[dest]
        # r_el_e, j_el_e = weak_form.evaluate_form(i, alpha_l)
        r_el, j_el = weak_form.evaluate_form_vectorized(i, alpha_l)

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz_idx = np.where(np.logical_not(np.isclose(data, 1.0e-16)))[0]
        if symmetric_solver_q:
            [
                A.setValue(row=row[idx], col=col[idx], value=data[idx], addv=True)
                for idx in nnz_idx
                if row[idx] <= col[idx]
            ]
        else:
            [
                A.setValue(row=row[idx], col=col[idx], value=data[idx], addv=True)
                for idx in nnz_idx
            ]
        # Petsc ILU requires explicit existence of diagonal zeros
        [A.setValue(row=idx, col=idx, value=0.0, addv=True) for idx in dest]

        check_points = [(int(k * n_els / 10)) for k in range(11)]
        if i in check_points or i == n_els - 1:
            if i == n_els - 1:
                print("Assembly: progress [%]: ", 100)
            else:
                print("Assembly: progress [%]: ", check_points.index(i) * 10)
                print(
                    "Assembly: Memory used [Byte] :",
                    (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - memory_start),
                )

    def scatter_riesz_form_data(P, i, riesz_map_weak_form, n_els):
        # destination indexes
        dest = riesz_map_weak_form.space.destination_indexes(i)
        alpha_l = alpha[dest]

        r_el, j_el = riesz_map_weak_form.evaluate_form_vectorized(i, alpha_l)

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz_idx = np.where(np.logical_not(np.isclose(data, 1.0e-16)))[0]
        if symmetric_solver_q:
            [
                P.setValue(row=row[idx], col=col[idx], value=data[idx], addv=True)
                for idx in nnz_idx
                if row[idx] <= col[idx]
            ]
        else:
            [
                P.setValue(row=row[idx], col=col[idx], value=data[idx], addv=True)
                for idx in nnz_idx
            ]

        check_points = [(int(k * n_els / 10)) for k in range(11)]
        if i in check_points or i == n_els - 1:
            if i == n_els - 1:
                print("Assembly: progress [%]: ", 100)
            else:
                print("Assembly: progress [%]: ", check_points.index(i) * 10)
            print(
                "Assembly: Memory used [Byte] :",
                (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - memory_start),
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

    print("")
    print("Assembly preconditioner")
    n_els = len(fe_space.discrete_spaces["s"].elements)
    [scatter_riesz_form_data(P, i, riesz_map_weak_form, n_els) for i in range(n_els)]

    A.assemble()
    P.assemble()
    print("Assembly: nz_allocated:", int(A.getInfo()["nz_allocated"]))
    print("Assembly: nz_used:", int(A.getInfo()["nz_used"]))
    print("Assembly: nz_unneeded:", int(A.getInfo()["nz_unneeded"]))

    et = time.time()
    elapsed_time = et - st
    print("Assembly: Time:", elapsed_time, "seconds")
    print(
        "Assembly: After PETSc M.assemble: Memory used [Byte] :",
        (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - memory_start),
    )

    # solving ls
    st = time.time()

    ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
    ksp.setOperators(A, P)
    b = A.createVecLeft()
    b.array[:] = -rg
    x = A.createVecRight()

    ksp.setType("minres")
    ksp.getPC().setType("fieldsplit")
    is_general_sigma = PETSc.IS()
    is_general_u = PETSc.IS()
    fields_idx = np.add.accumulate([0] + list(fe_space.discrete_spaces_dofs.values()))
    general_sigma_idx = np.array(range(fields_idx[0], fields_idx[1]), dtype=np.int32)
    general_u_idx = np.array(range(fields_idx[1], fields_idx[3]), dtype=np.int32)
    is_general_sigma.createGeneral(general_sigma_idx)
    is_general_u.createGeneral(general_u_idx)

    ksp.getPC().setFieldSplitIS(
        ("gen_sigma", is_general_sigma), ("gen_u", is_general_u)
    )
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
    ksp_s, ksp_u = ksp.getPC().getFieldSplitSubKSP()
    ksp_s.setType("preonly")
    if symmetric_solver_q:
        ksp_s.getPC().setType("cholesky")
    else:
        ksp_s.getPC().setType("lu")
    ksp_s.getPC().setFactorSolverType("mumps")
    ksp_u.setType("preonly")
    if symmetric_solver_q:
        ksp_u.getPC().setType("icc")
    else:
        ksp_u.getPC().setType("ilu")
    ksp.setTolerances(rtol=0.0, atol=1e-9, divtol=5000, max_it=20000)
    ksp.setConvergenceHistory()
    ksp.setFromOptions()

    ksp.solve(b, x)
    alpha = x.array
    residuals_history = ksp.getConvergenceHistory()
    print(
        "Linear solver: After PETSc ksp.solve: Memory used [Byte] :",
        (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - memory_start),
    )
    PETSc.KSP.destroy(ksp)
    PETSc.Mat.destroy(A)
    PETSc.Vec.destroy(b)
    PETSc.Vec.destroy(x)

    et = time.time()
    elapsed_time = et - st
    print("Linear solver: Time:", elapsed_time, "seconds")
    print(
        "Linear solver: After PETSc ksp.destroy: Memory used [Byte] :",
        (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - memory_start),
    )

    return alpha, residuals_history


def three_field_postprocessing(
    k_order, material_data, method, gmesh, alpha, write_vtk_q=False
):
    dim = gmesh.dimension

    fe_space = create_product_space(method, gmesh)
    n_dof_g = fe_space.n_dof

    # Material data
    m_lambda = material_data["lambda"]
    m_mu = material_data["mu"]

    # exact solution
    u_exact = le.displacement(m_lambda, m_mu, dim)
    t_exact = le.rotation(m_lambda, m_mu, dim)
    s_exact = le.stress(m_lambda, m_mu, dim)
    div_s_exact = le.stress_divergence(m_lambda, m_mu, dim)

    exact_functions = {
        "s": s_exact,
        "u": u_exact,
        "t": t_exact,
        "div_s": div_s_exact,
    }

    if write_vtk_q:
        st = time.time()

        lambda_value = material_data["lambda"]
        mu_value = material_data["mu"]

        prefix = method[0] + "_k" + str(k_order) + "_d" + str(dim)
        prefix += "_lambda_" + str(lambda_value) + "_mu_" + str(mu_value)
        file_name = prefix + "_four_fields_ex_2.vtk"

        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, exact_functions, alpha
        )
        et = time.time()
        elapsed_time = et - st
        print("VTK post-processing time:", elapsed_time, "seconds")

    st = time.time()
    s_l2_error, u_l2_error, t_l2_error = l2_error(
        dim, fe_space, exact_functions, alpha
    )
    div_s_l2_error = div_error(dim, fe_space, exact_functions, alpha)[0]

    s_h_div_error = np.sqrt((s_l2_error**2) + (div_s_l2_error**2))

    alpha_proj = l2_projector(fe_space, exact_functions)
    alpha_e = alpha - alpha_proj
    u_proj_l2_error, t_proj_l2_error = l2_error_projected(
        dim, fe_space, alpha_e, ["s"]
    )

    et = time.time()
    elapsed_time = et - st
    print("Error time:", elapsed_time, "seconds")
    print("L2-error displacement: ", u_l2_error)
    print("L2-error rotation: ", t_l2_error)
    print("L2-error stress: ", s_l2_error)
    print("L2-error div stress: ", div_s_l2_error)
    print("L2-error projected displacement: ", u_proj_l2_error)
    print("L2-error projected rotation: ", t_proj_l2_error)
    print("")

    return n_dof_g, np.array(
        [
            u_l2_error,
            t_l2_error,
            s_l2_error,
            div_s_l2_error,
            s_h_div_error,
            u_proj_l2_error,
            t_proj_l2_error,
        ]
    )


def three_field_solution_norms(material_data, method, gmesh):
    dim = gmesh.dimension

    fe_space = create_product_space(method, gmesh)

    # Material data
    m_lambda = material_data["lambda"]
    m_mu = material_data["mu"]

    # exact solution
    u_exact = le.displacement(m_lambda, m_mu, dim)
    t_exact = le.rotation(m_lambda, m_mu, dim)
    s_exact = le.stress(m_lambda, m_mu, dim)
    div_s_exact = le.stress_divergence(m_lambda, m_mu, dim)

    exact_functions = {
        "s": s_exact,
        "u": u_exact,
        "t": t_exact,
        "div_s": div_s_exact,
    }

    st = time.time()
    s_norm, u_norm, t_norm = l2_norm(dim, fe_space, exact_functions)
    div_s_norm = div_norm(dim, fe_space, exact_functions)[0]
    h_div_s_norm = np.sqrt((s_norm**2) + (div_s_norm**2))
    et = time.time()
    elapsed_time = et - st
    print("Solution norms time:", elapsed_time, "seconds")
    print("Displacement norm: ", u_norm)
    print("Rotation norm: ", t_norm)
    print("Stress norm: ", s_norm)
    print("div stress norm: ", div_s_norm)
    print("Stress hdiv-norm: ", h_div_s_norm)
    print(" ")

    return np.array(
        [
            [
                u_norm,
                t_norm,
                s_norm,
                div_s_norm,
                h_div_s_norm,
            ]
        ]
    )


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


def perform_convergence_approximations(configuration: dict):
    # retrieve parameters from given configuration
    k_order = configuration.get("k_order")
    method = configuration.get("method")
    n_ref = configuration.get("n_refinements")
    dimension = configuration.get("dimension")
    material_data = configuration.get("material_data", {})
    write_geometry_vtk = configuration.get("write_geometry_Q", True)

    # The initial element size
    h = 1.0

    # Create a unit squared or a unit cube
    domain = create_domain(dimension)

    for lh in range(n_ref):
        mesher = create_conformal_mesher(domain, h, lh)
        gmesh = create_mesh(dimension, mesher, write_geometry_vtk)
        alpha, res_history = three_field_approximation(material_data, method, gmesh)
        file_name = compose_file_name(
            method, k_order, lh, gmesh.dimension, material_data, "_alpha_ex_2.npy"
        )
        with open(file_name, "wb") as f:
            np.save(f, alpha)
        file_name_res = compose_file_name(
            method, k_order, lh, gmesh.dimension, material_data, "_res_history_ex_2.txt"
        )
        # First position includes n_dof
        np.savetxt(
            file_name_res,
            np.concatenate((np.array([len(alpha)]), res_history)),
            delimiter=",",
        )

    return


def perform_convergence_postprocessing(configuration: dict):
    # retrieve parameters from given configuration
    k_order = configuration.get("k_order")
    method = configuration.get("method")
    n_ref = configuration.get("n_refinements")
    dimension = configuration.get("dimension")
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
        mesher = create_conformal_mesher(domain, h, lh)
        gmesh = create_mesh(dimension, mesher, write_geometry_vtk)
        h_min, h_mean, h_max = mesh_size(gmesh)

        file_name = compose_file_name(
            method, k_order, lh, gmesh.dimension, material_data, "_alpha_ex_2.npy"
        )
        with open(file_name, "rb") as f:
            alpha = np.load(f)
        n_dof, error_vals = three_field_postprocessing(
            k_order, material_data, method, gmesh, alpha, write_vtk
        )

        file_name_res = compose_file_name(
            method, k_order, lh, gmesh.dimension, material_data, "_res_history_ex_2.txt"
        )
        res_data = np.genfromtxt(file_name_res, dtype=None, delimiter=",")
        n_iterations = res_data.shape[0] - 1  # First position includes n_dof
        chunk = np.concatenate([[n_dof, n_iterations, h_max], error_vals])
        error_data = np.append(error_data, np.array([chunk]), axis=0)

        # compute solution norms for the last refinement level
        if lh == n_ref - 1:
            sol_norms = three_field_solution_norms(material_data, method, gmesh)

    rates_data = np.empty((0, n_data - 3), float)
    for i in range(error_data.shape[0] - 1):
        chunk_b = np.log(error_data[i])
        chunk_e = np.log(error_data[i + 1])
        h_step = chunk_e[2] - chunk_b[2]
        partial = (chunk_e - chunk_b) / h_step
        rates_data = np.append(rates_data, np.array([list(partial[3:n_data])]), axis=0)

    # minimal report
    np.set_printoptions(precision=3)
    print("Dual problem: ", method[0])
    print("Polynomial order: ", k_order)
    print("Dimension: ", dimension)
    print("rounded error data: ", error_data)
    print("rounded error rates data: ", rates_data)
    print(" ")

    str_fields = "u, r, s,"
    dual_header = str_fields + "div_s, s_h_div_norm, Pu, Pr"
    base_str_header = dual_header
    e_str_header = "n_dof, n_iter, h, " + base_str_header

    lambda_value = material_data["lambda"]
    mu_value = material_data["mu"]

    file_name_prefix = (
        method[0]
        + "_lambda_"
        + str(lambda_value)
        + "_mu_"
        + str(mu_value)
        + "_k"
        + str(k_order)
        + "_"
        + str(dimension)
    )
    if report_full_precision_data:
        np.savetxt(
            file_name_prefix + "d_error_ex_2.txt",
            error_data,
            delimiter=",",
            header=e_str_header,
        )
        np.savetxt(
            file_name_prefix + "d_rates_ex_2.txt",
            rates_data,
            delimiter=",",
            header=base_str_header,
        )
        np.savetxt(
            file_name_prefix + "d_solution_norms_ex_2.txt",
            sol_norms,
            delimiter=",",
            header=base_str_header,
        )
    np.savetxt(
        file_name_prefix + "d_error_ex_2_rounded.txt",
        error_data,
        fmt="%1.3e",
        delimiter=",",
        header=e_str_header,
    )
    np.savetxt(
        file_name_prefix + "d_rates_ex_2_rounded.txt",
        rates_data,
        fmt="%1.3f",
        delimiter=",",
        header=base_str_header,
    )
    np.savetxt(
        file_name_prefix + "d_solution_norms_ex_2_rounded.txt",
        sol_norms,
        fmt="%1.10f",
        delimiter=",",
        header=base_str_header,
    )

    return


def method_definition(k_order):

    method_1 = {
        "s": ("BDM", k_order + 1),
        "u": ("Lagrange", k_order),
        "t": ("Lagrange", k_order),
    }

    methods = [method_1]
    method_names = ["wc_afw"]
    return zip(method_names, methods)


def material_data_definition():
    # Material data for example 2
    case_0 = {"lambda": 1.0, "mu": 1.0}
    case_1 = {"lambda": 1.0e2, "mu": 1.0}
    case_2 = {"lambda": 1.0e4, "mu": 1.0}
    case_3 = {"lambda": 1.0e8, "mu": 1.0}
    # cases = [case_0, case_1, case_2, case_3]
    cases = [case_0]
    return cases


def main():
    dimension = 2
    approximation_q = False
    postprocessing_q = True
    refinements = {0: 8, 1: 8}
    case_data = material_data_definition()
    for k in [0]:
        methods = method_definition(k)
        for i, method in enumerate(methods):
            for material_data in case_data:
                configuration = {
                    "k_order": k,
                    "dimension": dimension,
                    "n_refinements": refinements[k],
                    "method": method,
                    "material_data": material_data,
                }
                if approximation_q:
                    perform_convergence_approximations(configuration)
                if postprocessing_q:
                    perform_convergence_postprocessing(configuration)



if __name__ == "__main__":
    main()
