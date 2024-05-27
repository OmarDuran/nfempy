import resource
import time

import numpy as np
import strong_solution_cosserat_elasticity_example_3 as lce
from petsc4py import PETSc

from mesh.mesh import Mesh
from mesh.mesh_metrics import mesh_size
from postprocess.l2_error_post_processor import (
    div_error,
    div_scaled_error,
    l2_error,
    l2_error_projected,
)
from postprocess.projectors import l2_projector
from postprocess.solution_norms_post_processor import div_norm, l2_norm
from postprocess.solution_post_processor import (
    write_vtk_file_exact_solution,
    write_vtk_file_with_exact_solution,
    write_vtk_file,
)
from spaces.product_space import ProductSpace
from weak_forms.lce_scaled_dual_weak_form import (
    LCEScaledDualWeakForm,
    LCEScaledDualWeakFormBCDirichlet,
)
from weak_forms.lce_scaled_riesz_map_weak_form import LCEScaledRieszMapWeakForm
from mesh.mesh_metrics import cell_centroid


def create_product_space(method, gmesh):
    # FESpace: data
    s_k_order = method[1]["s"][1]
    m_k_order = method[1]["m"][1]
    u_k_order = method[1]["u"][1]
    t_k_order = method[1]["t"][1]

    s_components = 2
    m_components = 1
    u_components = 2
    t_components = 1
    if gmesh.dimension == 3:
        s_components = 3
        m_components = 3
        u_components = 3
        t_components = 3

    s_family = method[1]["s"][0]
    m_family = method[1]["m"][0]
    u_family = method[1]["u"][0]
    t_family = method[1]["t"][0]

    discrete_spaces_data = {
        "s": (gmesh.dimension, s_components, s_family, s_k_order, gmesh),
        "m": (gmesh.dimension, m_components, m_family, m_k_order, gmesh),
        "u": (gmesh.dimension, u_components, u_family, u_k_order, gmesh),
        "t": (gmesh.dimension, t_components, t_family, t_k_order, gmesh),
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

    s_field_bc_physical_tags = [9, 10, 11, 12]
    m_field_bc_physical_tags = [9, 10, 11, 12]
    discrete_spaces_bc_physical_tags = {
        "s": s_field_bc_physical_tags,
        "m": m_field_bc_physical_tags,
    }

    space = ProductSpace(discrete_spaces_data)
    space.make_subspaces_discontinuous(discrete_spaces_disc)
    space.build_structures(discrete_spaces_bc_physical_tags)
    return space


def four_field_scaled_approximation(method, gmesh, symmetric_solver_q=True):
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

    riesz_map_weak_form = LCEScaledRieszMapWeakForm(fe_space)
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
    general_sigma_idx = np.array(range(fields_idx[0], fields_idx[2]), dtype=np.int32)
    general_u_idx = np.array(range(fields_idx[2], fields_idx[4]), dtype=np.int32)
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
    ksp.setTolerances(rtol=0.0, atol=1e-12, divtol=5000, max_it=20000)
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


def four_field_scaled_postprocessing(k_order, method, gmesh, alpha, write_vtk_q=False):
    dim = gmesh.dimension
    fe_space = create_product_space(method, gmesh)
    n_dof_g = fe_space.n_dof

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
    f_gamma = lce.gamma_s(dim)
    f_grad_gamma = lce.grad_gamma_s(dim)

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

    if write_vtk_q:
        st = time.time()

        prefix = "ex_3_" + method[0]
        file_name = prefix + "_scaled_formulation.vtk"

        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, exact_functions, alpha, ["u", "t"]
        )

        prefix = "ex_3_" + method[0]
        file_name = prefix + "_gamma_scale_formulation.vtk"

        name_to_fields = {"gamma": 1, "grad_gamma": dim}
        write_vtk_file_exact_solution(file_name, gmesh, name_to_fields, exact_functions)

        et = time.time()
        elapsed_time = et - st
        print("VTK post-processing time:", elapsed_time, "seconds")

    st = time.time()
    s_l2_error, m_l2_error, u_l2_error, t_l2_error = l2_error(
        dim, fe_space, exact_functions, alpha
    )
    div_s_l2_error = div_error(dim, fe_space, exact_functions, alpha, ["m"])[0]
    div_m_l2_error = div_scaled_error(dim, fe_space, exact_functions, alpha, ["s"])[0]
    s_h_div_error = np.sqrt((s_l2_error**2) + (div_s_l2_error**2))
    m_h_div_error = np.sqrt((m_l2_error**2) + (div_m_l2_error**2))

    alpha_proj = l2_projector(fe_space, exact_functions)
    alpha_e = alpha - alpha_proj
    u_proj_l2_error, t_proj_l2_error = l2_error_projected(
        dim, fe_space, alpha_e, ["s", "m"]
    )

    et = time.time()
    elapsed_time = et - st
    print("Error time:", elapsed_time, "seconds")
    print("L2-error displacement: ", u_l2_error)
    print("L2-error rotation: ", t_l2_error)
    print("L2-error stress: ", s_l2_error)
    print("L2-error couple stress: ", m_l2_error)
    print("L2-error div stress: ", div_s_l2_error)
    print("L2-error div couple stress: ", div_m_l2_error)
    print("L2-error projected displacement: ", u_proj_l2_error)
    print("L2-error projected rotation: ", t_proj_l2_error)
    print("")

    return n_dof_g, np.array(
        [
            u_l2_error,
            t_l2_error,
            s_l2_error,
            m_l2_error,
            div_s_l2_error,
            div_m_l2_error,
            s_h_div_error,
            m_h_div_error,
            u_proj_l2_error,
            t_proj_l2_error,
        ]
    )


def four_field_scaled_solution_norms(method, gmesh):
    dim = gmesh.dimension
    fe_space = create_product_space(method, gmesh)

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
    f_gamma = lce.gamma_s(dim)
    f_grad_gamma = lce.grad_gamma_s(dim)

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

    st = time.time()
    s_norm, m_norm, u_norm, t_norm = l2_norm(dim, fe_space, exact_functions)
    div_s_norm, div_m_norm = div_norm(dim, fe_space, exact_functions)
    h_div_s_norm = np.sqrt((s_norm**2) + (div_s_norm**2))
    h_div_m_norm = np.sqrt((m_norm**2) + (div_m_norm**2))
    et = time.time()
    elapsed_time = et - st
    print("Solution norms time:", elapsed_time, "seconds")
    print("Displacement norm: ", u_norm)
    print("Rotation norm: ", t_norm)
    print("Stress norm: ", s_norm)
    print("Couple stress star norm: ", m_norm)
    print("div stress norm: ", div_s_norm)
    print("div couple stress norm: ", div_m_norm)
    print("Stress hdiv-norm: ", h_div_s_norm)
    print("Couple stress star hdiv-norm: ", h_div_m_norm)
    print(" ")

    return np.array(
        [
            [
                u_norm,
                t_norm,
                s_norm,
                m_norm,
                div_s_norm,
                div_m_norm,
                h_div_s_norm,
                h_div_m_norm,
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
    alpha = np.concatenate(
        (
            alpha[0 : dof_per_field["s"]],
            np.zeros((dof_per_field["m"])),
            alpha[dof_per_field["s"] : dof_per_field["u"] + dof_per_field["s"]],
            np.zeros((dof_per_field["t"])),
        )
    )

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

    def compute_centroid(points):
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

    # check if the mesh points are the same
    node_perm = compute_permutation(geometry_data["points"], gmesh.points)
    assert np.all(np.isclose(geometry_data["points"][node_perm], gmesh.points))

    def entity_centroid(points):
        xc = np.mean(points, axis=0)
        return xc

    points_c0 = geometry_data["points"][geometry_data["node_tags_c0"]]
    points_c1 = geometry_data["points"][geometry_data["node_tags_c1"]]
    exc_c0 = np.array(list(map(entity_centroid, points_c0)))
    exc_c1 = np.array(list(map(entity_centroid, points_c1)))

    # compute dof permutations
    cell_perm = compute_permutation(exc_c0, xc_c0)
    face_perm = compute_permutation(exc_c1, xc_c1)

    assert np.all(np.isclose(exc_c0[cell_perm], xc_c0))
    assert np.all(np.isclose(exc_c1[face_perm], xc_c1))
    n_sign = np.sign(np.sum(geometry_data["n_c1"][face_perm, 0:2] * n_c1, axis=1))

    alpha_s = (
        np.array(np.split(alpha[fields_idx[0] : fields_idx[1]], xc_c1.shape[0]))
        * np.vstack((n_sign, n_sign)).T
    )
    alpha_u = np.array(np.split(alpha[fields_idx[2] : fields_idx[3]], xc_c0.shape[0]))

    alpha[fields_idx[0] : fields_idx[1]] = alpha_s[face_perm].flatten()
    alpha[fields_idx[2] : fields_idx[3]] = alpha_u[cell_perm].flatten()

    n_dof_g = fe_space.n_dof

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
    f_gamma = lce.gamma_s(dim)
    f_grad_gamma = lce.grad_gamma_s(dim)

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

    if write_vtk_q:
        st = time.time()

        prefix = "ex_3_" + method[0]
        file_name = prefix + "_ecmor.vtk"

        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, exact_functions, alpha, ["u", "t"]
        )
        et = time.time()
        elapsed_time = et - st
        print("VTK post-processing time:", elapsed_time, "seconds")

    st = time.time()
    s_l2_error, m_l2_error, u_l2_error, t_l2_error = l2_error(
        dim, fe_space, exact_functions, alpha
    )

    alpha_proj = l2_projector(fe_space, exact_functions)
    alpha_e = alpha - alpha_proj
    u_proj_l2_error, t_proj_l2_error = l2_error_projected(
        dim, fe_space, alpha_e, ["s", "m"]
    )

    et = time.time()
    elapsed_time = et - st
    print("Error time:", elapsed_time, "seconds")
    print("L2-error displacement: ", u_l2_error)
    print("L2-error rotation: ", t_l2_error)
    print("L2-error stress: ", s_l2_error)
    print("L2-error couple stress: ", m_l2_error)
    print("L2-error projected displacement: ", u_proj_l2_error)
    print("L2-error projected rotation: ", t_proj_l2_error)
    print("")

    return n_dof_g, np.array(
        [
            u_l2_error,
            t_l2_error,
            s_l2_error,
            m_l2_error,
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
    # retrieve parameters from dictionary
    k_order = configuration.get("k_order")
    method = configuration.get("method")
    n_ref = configuration.get("n_refinements")
    dimension = configuration.get("dimension")
    write_geometry_vtk = configuration.get("write_geometry_Q", True)

    for lh in range(n_ref):
        mesh_file = "gmsh_files/ex_3/partition_ex_3_l_" + str(lh) + ".msh"
        gmesh = create_mesh_from_file(mesh_file, dimension, write_geometry_vtk)
        alpha, res_history = four_field_scaled_approximation(method, gmesh)
        file_name = compose_file_name(
            method, k_order, lh, gmesh.dimension, "_alpha.npy"
        )
        with open(file_name, "wb") as f:
            np.save(f, alpha)
        file_name_res = compose_file_name(
            method, k_order, lh, gmesh.dimension, "_res_history.txt"
        )
        # First position includes n_dof
        np.savetxt(
            file_name_res,
            np.concatenate((np.array([len(alpha)]), res_history)),
            delimiter=",",
        )

    return


def perform_convergence_postprocessing(configuration: dict):
    # retrieve parameters from dictionary
    k_order = configuration.get("k_order")
    method = configuration.get("method")
    n_ref = configuration.get("n_refinements")
    dimension = configuration.get("dimension")
    write_geometry_vtk = configuration.get("write_geometry_Q", True)
    write_vtk = configuration.get("write_vtk_Q", True)
    report_full_precision_data = configuration.get("report_full_precision_data_Q", True)

    n_data = 13
    error_data = np.empty((0, n_data), float)
    for lh in range(n_ref):
        mesh_file = "gmsh_files/ex_3/partition_ex_3_l_" + str(lh) + ".msh"
        gmesh = create_mesh_from_file(mesh_file, dimension, write_geometry_vtk)
        h_min, h_mean, h_max = mesh_size(gmesh)

        file_name = compose_file_name(
            method, k_order, lh, gmesh.dimension, "_alpha.npy"
        )
        with open(file_name, "rb") as f:
            alpha = np.load(f)
        n_dof, error_vals = four_field_scaled_postprocessing(
            k_order, method, gmesh, alpha, write_vtk
        )

        file_name_res = compose_file_name(
            method, k_order, lh, gmesh.dimension, "_res_history.txt"
        )
        res_data = np.genfromtxt(file_name_res, dtype=None, delimiter=",")
        n_iterations = res_data.shape[0] - 1  # First position includes n_dof
        chunk = np.concatenate([[n_dof, n_iterations, h_max], error_vals])
        error_data = np.append(error_data, np.array([chunk]), axis=0)

        # compute solution norms for the last refinement level
        if lh == n_ref - 1:
            sol_norms = four_field_scaled_solution_norms(method, gmesh)

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

    str_fields = "u, r, s, o, "
    dual_header = str_fields + "div_s, div_o, s_h_div_norm, o_h_div_norm, Pu, Pr"
    base_str_header = dual_header
    e_str_header = "n_dof, n_iter, h, " + base_str_header

    file_name_prefix = "ex_3_" + method[0]
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

    n_data = 8
    error_data = np.empty((0, n_data), float)
    for lh in range(n_ref):
        mesh_file = "gmsh_files/ex_3/partition_ex_3_l_" + str(lh) + ".msh"
        gmesh = create_mesh_from_file(mesh_file, dimension, write_geometry_vtk)
        h_min, h_mean, h_max = mesh_size(gmesh)

        # loading displacements
        u_suffix = "__displacement_" + str(lh) + "_" + str(l_map[lh]) + ".npy"
        file_name = compose_file_name_fv(method, u_suffix)
        file_name_u = fv_tpsa_folder + "/" + file_name
        with open(file_name_u, "rb") as f:
            alpha_u = np.load(f)

        # loading stress
        s_suffix = "__stress_" + str(lh) + "_" + str(l_map[lh]) + ".npy"
        file_name = compose_file_name_fv(method, s_suffix)
        file_name_s = fv_tpsa_folder + "/" + file_name
        with open(file_name_s, "rb") as f:
            alpha_s = np.load(f)

        file_cell_centroid = (
            fv_tpsa_folder
            + "/"
            + "ex_3_cell_centroid"
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
            + "ex_3_face_centroid"
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
            + "ex_3_node"
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
            + "ex_3_cell_node"
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
            + "ex_3_face_node"
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
            + "ex_3_face_normal"
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
            + "ex_3_face_length"
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
            "node_tags_c0": cell_node,
            "node_tags_c1": face_node,
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

    file_name_prefix = "ex_3_" + method[0]
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
        "m": ("RT", 1),
        "u": ("Lagrange", 0),
        "t": ("Lagrange", 0),
    }
    methods = [method_1]
    method_names = [method_name]
    return zip(method_names, methods)


def method_definition(k_order):
    method_1 = {
        "s": ("BDM", k_order + 1),
        "m": ("RT", k_order + 1),
        "u": ("Lagrange", k_order),
        "t": ("Lagrange", k_order),
    }
    methods = [method_1]
    method_names = ["four_field_MFEM"]

    return zip(method_names, methods)


def compose_file_name(method, k_order, ref_l, dim, suffix):
    prefix = "ex_3_" + method[0] + "_l" + str(ref_l)
    file_name = prefix + suffix
    return file_name


def compose_file_name_fv(method, suffix):
    prefix = "ex_3_" + method[0]
    file_name = prefix + suffix
    return file_name


def main():
    dimension = 2
    approximation_q = True
    postprocessing_q = True
    refinements = {0: 6, 1: 6}
    for k in [0]:
        for method in method_definition(k):
            configuration = {
                "k_order": k,
                "dimension": dimension,
                "n_refinements": refinements[k],
                "method": method,
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
            configuration = {
                "dimension": dimension,
                "n_refinements": refinements[k],
                "method": method,
                "tpsa_data": fv_tpsa_folder,
            }
            if postprocessing_ecmor_q:
                perform_ecmor_postprocessing(configuration)


if __name__ == "__main__":
    main()
