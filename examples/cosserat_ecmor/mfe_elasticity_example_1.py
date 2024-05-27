import resource
import time

import numpy as np
import strong_solution_elasticity_example_1 as le
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
from mesh.mesh_metrics import cell_centroid


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
    if gmesh.dimension == 3:
        s_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
    discrete_spaces_bc_physical_tags = {
        "s": s_field_bc_physical_tags,
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

        prefix = "ex_1_" + method[0] + "_lambda_" + str(lambda_value)
        file_name = prefix + ".vtk"

        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, exact_functions, alpha, ["u", "t"]
        )
        et = time.time()
        elapsed_time = et - st
        print("VTK post-processing time:", elapsed_time, "seconds")

    st = time.time()
    s_l2_error, u_l2_error, t_l2_error = l2_error(dim, fe_space, exact_functions, alpha)
    div_s_l2_error = div_error(dim, fe_space, exact_functions, alpha)[0]

    s_h_div_error = np.sqrt((s_l2_error**2) + (div_s_l2_error**2))

    alpha_proj = l2_projector(fe_space, exact_functions)
    alpha_e = alpha - alpha_proj
    u_proj_l2_error, t_proj_l2_error = l2_error_projected(dim, fe_space, alpha_e, ["s"])

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


def ecmor_fv_postprocessing(
    material_data, method, gmesh, alpha, geometry_data, write_vtk_q=False
):
    dim = gmesh.dimension
    fe_space = create_product_space(method, gmesh)
    dof_per_field = fe_space.discrete_spaces_dofs
    fields_idx = np.add.accumulate([0] + list(dof_per_field.values()))
    alpha = np.concatenate((alpha, np.zeros((dof_per_field["t"]))))

    def compute_permutation(origin_xc, target_xc):
        "Compute permutation indices for reorder origin_xc to match target_xc"

        hashr = lambda x: hash(str(x[0]) + str(x[1]) + str(x[0]))
        origin = np.around(origin_xc, 12)
        target = np.around(target_xc, 12)
        ho = np.fromiter(map(hashr, origin), dtype=np.int64)
        ht = np.fromiter(map(hashr, target), dtype=np.int64)
        perm = np.argsort(ho)[np.argsort(np.argsort(ht))]
        return perm

    def compute_normal(points):
        v = (points[1] - points[0]) / np.linalg.norm((points[1] - points[0]))
        n = np.array([-v[1], v[0]])
        return n

    xc_c0 = np.array(
        [cell_centroid(cell, gmesh) for cell in gmesh.cells if cell.dimension == dim]
    )
    cells_c1 = [cell for cell in gmesh.cells if cell.dimension == dim - 1]
    n_c1 = np.array(
        [compute_normal(gmesh.points[np.sort(cell.node_tags)]) for cell in cells_c1]
    )
    xc_c1 = np.array([cell_centroid(cell, gmesh) for cell in cells_c1])

    dxc_c0 = np.linalg.norm(xc_c0, axis=1)
    dxc_c1 = np.linalg.norm(xc_c1, axis=1)
    edxc_c0 = np.linalg.norm(geometry_data["xc_c0"], axis=1)
    edxc_c1 = np.linalg.norm(geometry_data["xc_c1"], axis=1)
    node_perm = compute_permutation(geometry_data["points"], gmesh.points)
    assert np.all(np.isclose(geometry_data["points"][node_perm], gmesh.points))

    # compute dof permutations
    cell_perm = compute_permutation(geometry_data["xc_c0"], xc_c0)
    if np.all(np.isclose(np.arange(0, xc_c0.shape[0]), cell_perm)):
        print("Meshes does not need cell permutation.")
    else:
        aka = 0
    face_perm = compute_permutation(geometry_data["xc_c1"], xc_c1)

    assert np.all(np.isclose(geometry_data["xc_c0"][cell_perm], xc_c0))
    assert np.all(np.isclose(geometry_data["xc_c1"][face_perm], xc_c1))
    n_sign = np.sign(np.sum(geometry_data["n_c1"][face_perm, 0:2] * n_c1, axis=1))

    alpha_s = np.array(np.split(alpha[fields_idx[0] : fields_idx[1]], xc_c1.shape[0])) * np.vstack((n_sign,n_sign)).T
    alpha_u = np.array(np.split(alpha[fields_idx[1] : fields_idx[2]], xc_c0.shape[0]))

    alpha[fields_idx[0] : fields_idx[1]] = alpha_s[face_perm].flatten()
    alpha[fields_idx[1] : fields_idx[2]] = alpha_u[cell_perm].flatten()

    n_dof_g = fe_space.n_dof

    # Material data
    m_lambda = material_data["lambda"]
    m_mu = material_data["mu"]

    # exact solution
    u_exact = le.displacement(m_lambda, m_mu, dim)
    t_exact = le.rotation(m_lambda, m_mu, dim)
    s_exact = le.stress(m_lambda, m_mu, dim)

    exact_functions = {
        "s": s_exact,
        "u": u_exact,
        "t": t_exact,
    }

    if write_vtk_q:
        st = time.time()

        lambda_value = material_data["lambda"]

        prefix = "ex_1_" + method[0] + "_lambda_" + str(lambda_value)
        file_name = prefix + "_ecmor.vtk"

        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, exact_functions, alpha, ["u", "t"]
        )
        et = time.time()
        elapsed_time = et - st
        print("VTK post-processing time:", elapsed_time, "seconds")

    st = time.time()
    s_l2_error, u_l2_error, t_l2_error = l2_error(dim, fe_space, exact_functions, alpha)

    alpha_proj = l2_projector(fe_space, exact_functions)
    alpha_e = alpha - alpha_proj
    u_proj_l2_error, t_proj_l2_error = l2_error_projected(dim, fe_space, alpha_e, ["s"])

    et = time.time()
    elapsed_time = et - st
    print("Error time:", elapsed_time, "seconds")
    print("L2-error displacement: ", u_l2_error)
    print("L2-error rotation: ", t_l2_error)
    print("L2-error stress: ", s_l2_error)
    print("L2-error projected displacement: ", u_proj_l2_error)
    print("L2-error projected rotation: ", t_proj_l2_error)
    print("")

    return n_dof_g, np.array(
        [
            u_l2_error,
            t_l2_error,
            s_l2_error,
            u_proj_l2_error,
            t_proj_l2_error,
        ]
    )


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

    for lh in range(n_ref):
        mesh_file = "gmsh_files/ex_1/partition_ex_1_l_" + str(lh) + ".msh"
        gmesh = create_mesh_from_file(mesh_file, dimension, write_geometry_vtk)
        alpha, res_history = three_field_approximation(material_data, method, gmesh)
        file_name = compose_file_name(
            method, k_order, lh, gmesh.dimension, material_data, "_alpha.npy"
        )
        with open(file_name, "wb") as f:
            np.save(f, alpha)
        file_name_res = compose_file_name(
            method, k_order, lh, gmesh.dimension, material_data, "_res_history.txt"
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

    n_data = 10
    error_data = np.empty((0, n_data), float)
    for lh in range(n_ref):
        mesh_file = "gmsh_files/ex_1/partition_ex_1_l_" + str(lh) + ".msh"
        gmesh = create_mesh_from_file(mesh_file, dimension, write_geometry_vtk)
        h_min, h_mean, h_max = mesh_size(gmesh)

        file_name = compose_file_name(
            method, k_order, lh, gmesh.dimension, material_data, "_alpha.npy"
        )
        with open(file_name, "rb") as f:
            alpha = np.load(f)
        n_dof, error_vals = three_field_postprocessing(
            k_order, material_data, method, gmesh, alpha, write_vtk
        )

        file_name_res = compose_file_name(
            method, k_order, lh, gmesh.dimension, material_data, "_res_history.txt"
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

    file_name_prefix = "ex_1_" + method[0] + "_lambda_" + str(lambda_value)
    if report_full_precision_data:
        np.savetxt(
            file_name_prefix + "_error.txt",
            error_data,
            delimiter=",",
            header=e_str_header,
        )
        np.savetxt(
            file_name_prefix + "_rates.txt",
            rates_data,
            delimiter=",",
            header=base_str_header,
        )
        np.savetxt(
            file_name_prefix + "_solution_norms.txt",
            sol_norms,
            delimiter=",",
            header=base_str_header,
        )
    np.savetxt(
        file_name_prefix + "_error_rounded.txt",
        error_data,
        fmt="%1.3e",
        delimiter=",",
        header=e_str_header,
    )
    np.savetxt(
        file_name_prefix + "_rates_rounded.txt",
        rates_data,
        fmt="%1.3f",
        delimiter=",",
        header=base_str_header,
    )
    np.savetxt(
        file_name_prefix + "_solution_norms_rounded.txt",
        sol_norms,
        fmt="%1.10f",
        delimiter=",",
        header=base_str_header,
    )

    return


def perform_ecmor_postprocessing(configuration: dict):
    # retrieve parameters from given configuration
    k_order = configuration.get("k_order", 0)
    method = configuration.get("method")
    n_ref = configuration.get("n_refinements")
    dimension = configuration.get("dimension")
    material_data = configuration.get("material_data", {})
    write_geometry_vtk = configuration.get("write_geometry_Q", True)
    write_vtk = configuration.get("write_vtk_Q", True)
    report_full_precision_data = configuration.get("report_full_precision_data_Q", True)
    fv_tpsa_folder = configuration.get("tpsa_data", None)

    if fv_tpsa_folder is None:
        raise ValueError("TPSA folder is not provided.")

    l_map = {0: 0.25, 1: 0.125, 2: 0.0625, 3: 0.03125, 4: 0.015625, 5: 0.0078125}

    n_data = 7
    error_data = np.empty((0, n_data), float)
    for lh in range(n_ref):
        mesh_file = "gmsh_files/ex_1/partition_ex_1_l_" + str(lh) + ".msh"
        gmesh = create_mesh_from_file(mesh_file, dimension, write_geometry_vtk)
        h_min, h_mean, h_max = mesh_size(gmesh)

        # loading displacements
        u_suffix = "_displacement_" + str(lh) + "_" + str(l_map[lh]) + ".npy"
        file_name = compose_file_name_fv(method, material_data, u_suffix)
        file_name_u = fv_tpsa_folder + "/" + file_name
        with open(file_name_u, "rb") as f:
            alpha_u = np.load(f)

        # loading stress
        s_suffix = "_stress_" + str(lh) + "_" + str(l_map[lh]) + ".npy"
        file_name = compose_file_name_fv(method, material_data, s_suffix)
        file_name_s = fv_tpsa_folder + "/" + file_name
        with open(file_name_s, "rb") as f:
            alpha_s = np.load(f)

        # loading geometrical information
        file_cell_centroid = (
            fv_tpsa_folder
            + "/"
            + "ex_1_cell_centroid"
            + "_"
            + str(lh)
            + "_"
            + str(l_map[lh])
            + ".npy"
        )
        with open(file_cell_centroid, "rb") as f:
            cell_centroid = np.load(file_cell_centroid).T

        file_face_centroid = (
            fv_tpsa_folder
            + "/"
            + "ex_1_face_centroid"
            + "_"
            + str(lh)
            + "_"
            + str(l_map[lh])
            + ".npy"
        )
        with open(file_face_centroid, "rb") as f:
            face_centroid = np.load(file_face_centroid).T

        file_mesh_points = (
                fv_tpsa_folder
                + "/"
                + "ex_1_node"
                + "_"
                + str(lh)
                + "_"
                + str(l_map[lh])
                + ".npy"
        )
        with open(file_mesh_points, "rb") as f:
            mesh_points = np.load(file_mesh_points).T

        file_cell_node = (
                fv_tpsa_folder
                + "/"
                + "ex_1_cell_node"
                + "_"
                + str(lh)
                + "_"
                + str(l_map[lh])
                + ".npy"
        )
        with open(file_cell_node, "rb") as f:
            cell_node = np.load(file_cell_node).T

        file_face_node = (
                fv_tpsa_folder
                + "/"
                + "ex_1_face_node"
                + "_"
                + str(lh)
                + "_"
                + str(l_map[lh])
                + ".npy"
        )
        with open(file_face_node, "rb") as f:
            face_node = np.load(file_face_node).T

        file_face_normal = (
            fv_tpsa_folder
            + "/"
            + "ex_1_face_normal"
            + "_"
            + str(lh)
            + "_"
            + str(l_map[lh])
            + ".npy"
        )
        with open(file_face_normal, "rb") as f:
            face_normnal = np.load(file_face_normal).T

        file_face_length = (
            fv_tpsa_folder
            + "/"
            + "ex_1_face_length"
            + "_"
            + str(lh)
            + "_"
            + str(l_map[lh])
            + ".npy"
        )
        with open(file_face_length, "rb") as f:
            face_length = np.load(file_face_length).T

        geometry_data = {
            "xc_c0": cell_centroid,
            "xc_c1": face_centroid,
            "xs_c0": cell_node,
            "xs_c1": face_node,
            "points": mesh_points,
            "n_c1": face_normnal,
            "measure_c1": face_length,
        }

        alpha = np.concatenate((alpha_s, alpha_u))
        n_dof, error_vals = ecmor_fv_postprocessing(
            material_data, method, gmesh, alpha, geometry_data, write_vtk
        )
        n_dof = np.concatenate((alpha_s, alpha_u)).shape[0]

        chunk = np.concatenate([[n_dof, h_max], error_vals])
        error_data = np.append(error_data, np.array([chunk]), axis=0)

    rates_data = np.empty((0, n_data - 2), float)
    for i in range(error_data.shape[0] - 1):
        chunk_b = np.log(error_data[i])
        chunk_e = np.log(error_data[i + 1])
        h_step = chunk_e[2] - chunk_b[2]
        partial = (chunk_e - chunk_b) / h_step
        rates_data = np.append(rates_data, np.array([list(partial[2:n_data])]), axis=0)

    # minimal report
    np.set_printoptions(precision=3)
    print("Dual problem: ", method[0])
    print("Polynomial order: ", k_order)
    print("Dimension: ", dimension)
    print("rounded error data: ", error_data)
    print("rounded error rates data: ", rates_data)
    print(" ")

    str_fields = "u, r, s,"
    dual_header = str_fields + " Pu, Pr"
    base_str_header = dual_header
    e_str_header = "n_dof, h, " + base_str_header

    lambda_value = material_data["lambda"]

    file_name_prefix = "ex_1_" + method[0] + "_lambda_" + str(lambda_value)
    if report_full_precision_data:
        np.savetxt(
            file_name_prefix + "_error.txt",
            error_data,
            delimiter=",",
            header=e_str_header,
        )
        np.savetxt(
            file_name_prefix + "_rates.txt",
            rates_data,
            delimiter=",",
            header=base_str_header,
        )
    np.savetxt(
        file_name_prefix + "_error_rounded.txt",
        error_data,
        fmt="%1.3e",
        delimiter=",",
        header=e_str_header,
    )
    np.savetxt(
        file_name_prefix + "_rates_rounded.txt",
        rates_data,
        fmt="%1.3f",
        delimiter=",",
        header=base_str_header,
    )

    return


def fv_method_definition(method_name):
    k_order = 0
    method_1 = {
        "s": ("RT", 1),
        "u": ("Lagrange", 0),
        "t": ("Lagrange", 0),
    }
    methods = [method_1]
    method_names = [method_name]
    return zip(method_names, methods)


def method_definition(k_order):
    method_1 = {
        "s": ("BDM", k_order + 1),
        "u": ("Lagrange", k_order),
        "t": ("Lagrange", k_order),
    }

    methods = [method_1]
    method_names = ["three_field_MFEM"]
    return zip(method_names, methods)


def material_data_definition():
    # Material data for example 1
    case_0 = {"lambda": 1.0, "mu": 1.0}
    case_1 = {"lambda": 1.0e2, "mu": 1.0}
    case_2 = {"lambda": 1.0e4, "mu": 1.0}
    case_3 = {"lambda": 1.0e8, "mu": 1.0}
    cases = [case_0, case_1, case_2, case_3]
    cases = [case_0]
    return cases


def compose_file_name(method, k_order, ref_l, dim, material_data, suffix):
    prefix = (
        "ex_1_"
        + method[0]
        + "_lambda_"
        + str(material_data["lambda"])
        + "_l_"
        + str(ref_l)
    )
    file_name = prefix + suffix
    return file_name


def compose_file_name_fv(method, material_data, suffix):
    formatted_number = "{:.1e}".format(material_data["lambda"])
    _, exponent = formatted_number.split("e")

    prefix = "ex_1_" + method[0] + "_lambda_" + "1e" + str(int(exponent))
    file_name = prefix + suffix
    return file_name


def main():
    dimension = 2
    approximation_q = False
    postprocessing_q = False
    refinements = {0: 5, 1: 6}
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

    # Postprocessing FV results
    postprocessing_ecmor_q = True
    fv_tpsa_folder = "output_ecmor_fv/tpsa_mpsa_results_v4"
    for method_name in ["TPSA"]:
        methods = fv_method_definition(method_name)
        for i, method in enumerate(methods):
            for material_data in case_data:
                configuration = {
                    "method_name": method_name,
                    "dimension": dimension,
                    "n_refinements": refinements[0],
                    "method": method,
                    "material_data": material_data,
                    "tpsa_data": fv_tpsa_folder,
                }
                if postprocessing_ecmor_q:
                    perform_ecmor_postprocessing(configuration)


if __name__ == "__main__":
    main()
