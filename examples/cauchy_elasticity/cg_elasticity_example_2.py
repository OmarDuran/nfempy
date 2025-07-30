import resource
import time
from functools import partial

import numpy as np
from mesh.mesh import Mesh
from mesh.mesh_metrics import mesh_size
from petsc4py import PETSc
from postprocess.l2_error_post_processor import l2_error
from postprocess.solution_post_processor import write_vtk_file_with_exact_solution
from spaces.product_space import ProductSpace
from weak_forms.le_primal_weak_form import LEPrimalWeakForm, LEPrimalWeakFormBCDirichlet

import strong_solution_elasticity_example_2 as le


def create_product_space(method, gmesh):
    # FESpace: data
    u_k_order = method[1]["u"][1]
    u_components = 2 if gmesh.dimension == 2 else 3
    u_family = method[1]["u"][0]

    discrete_spaces_data = {
        "u": (gmesh.dimension, u_components, u_family, u_k_order, gmesh),
    }

    u_disc_Q = False  # We want continuous elements for CG
    discrete_spaces_disc = {
        "u": u_disc_Q,
    }

    physical_tags = {
        "u": [1, 2, 3, 4],
    }

    b_physical_tags = {
        "u": [5, 6, 7, 8],
    }

    space = ProductSpace(discrete_spaces_data)
    space.make_subspaces_discontinuous(discrete_spaces_disc)
    space.build_structures(physical_tags, b_physical_tags)
    return space


def primal_approximation(material_data, method, gmesh):
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
    A.setType("sbaij")  # Symmetric matrix for CG formulation

    # Material data
    m_lambda = material_data["lambda"]
    m_mu = material_data["mu"]
    m_kappa = material_data["kappa"]

    # exact solution and functions
    u_exact = le.displacement(m_lambda, m_mu, m_kappa, dim)
    f_rhs = le.rhs(m_lambda, m_mu, m_kappa, dim)
    f_xi = partial(le.xi, m_kappa=m_kappa, dim=dim)

    def f_lambda(x, y, z):
        return f_xi(x, y, z)

    def f_mu(x, y, z):
        return f_xi(x, y, z)

    m_functions = {
        "rhs": f_rhs,
        "lambda": f_lambda,
        "mu": f_mu,
    }

    exact_functions = {
        "u": u_exact,
    }

    weak_form = LEPrimalWeakForm(fe_space)
    weak_form.functions = m_functions
    bc_weak_form = LEPrimalWeakFormBCDirichlet(fe_space)
    bc_weak_form.functions = exact_functions

    def scatter_form_data(A, i, weak_form, n_els):
        dest = weak_form.space.destination_indexes(i)
        alpha_l = alpha[dest]
        r_el, j_el = weak_form.evaluate_form(i, alpha_l)

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
            if row[idx] <= col[idx]
        ]
        # Diagonal zeros for PETSc ILU
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

    def scatter_bc_form(A, i, bc_weak_form):
        dest = fe_space.bc_destination_indexes(i)
        alpha_l = alpha[dest]
        r_el, j_el = bc_weak_form.evaluate_form(i, alpha_l)
        rg[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz_idx = np.where(np.logical_not(np.isclose(data, 1.0e-16)))[0]
        [
            A.setValue(row=row[idx], col=col[idx], value=data[idx], addv=True)
            for idx in nnz_idx
            if row[idx] <= col[idx]
        ]


    n_els = len(fe_space.discrete_spaces["u"].elements)
    [scatter_form_data(A, i, weak_form, n_els) for i in range(n_els)]

    n_bc_els = len(fe_space.discrete_spaces["u"].bc_elements)
    [scatter_bc_form(A, i, bc_weak_form) for i in range(n_bc_els)]

    A.assemble()
    print("Assembly: nz_allocated:", int(A.getInfo()["nz_allocated"]))
    print("Assembly: nz_used:", int(A.getInfo()["nz_used"]))
    print("Assembly: nz_unneeded:", int(A.getInfo()["nz_unneeded"]))

    et = time.time()
    elapsed_time = et - st
    print("Assembly: Time:", elapsed_time, "seconds")

    # solving linear system
    st = time.time()

    ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
    ksp.setOperators(A)
    b = A.createVecLeft()
    b.array[:] = -rg
    x = A.createVecRight()

    ksp.setType("cg")  # Conjugate gradient for symmetric positive definite system
    ksp.getPC().setType("icc")  # Incomplete Cholesky preconditioner
    ksp.setTolerances(rtol=0.0, atol=1e-10, divtol=5000, max_it=1000)
    ksp.setFromOptions()

    ksp.solve(b, x)
    alpha = x.array
    residuals_history = ksp.getConvergenceHistory()

    PETSc.KSP.destroy(ksp)
    PETSc.Mat.destroy(A)
    PETSc.Vec.destroy(b)
    PETSc.Vec.destroy(x)

    et = time.time()
    elapsed_time = et - st
    print("Linear solver: Time:", elapsed_time, "seconds")

    return alpha, residuals_history


def primal_postprocessing(material_data, method, gmesh, alpha, write_vtk_q=False):
    dim = gmesh.dimension
    fe_space = create_product_space(method, gmesh)

    # Material data
    m_lambda = material_data["lambda"]
    m_mu = material_data["mu"]
    m_kappa = material_data["kappa"]

    # exact solution
    u_exact = le.displacement(m_lambda, m_mu, m_kappa, dim)

    exact_functions = {
        "u": u_exact,
    }

    if write_vtk_q:
        st = time.time()
        prefix = "cg_ex_2_" + method[0] + "_kappa_" + str(material_data["kappa"])
        file_name = prefix + ".vtk"
        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, exact_functions, alpha
        )
        et = time.time()
        elapsed_time = et - st
        print("VTK post-processing time:", elapsed_time, "seconds")

    st = time.time()
    u_l2_error = l2_error(dim, fe_space, exact_functions, alpha)[0]

    et = time.time()
    elapsed_time = et - st
    print("Error computation time:", elapsed_time, "seconds")
    print("L2-error displacement: ", u_l2_error)
    print("")

    return fe_space.n_dof, np.array([u_l2_error])


def create_mesh_from_file(file_name, dim, write_vtk_q=False):
    gmesh = Mesh(dimension=dim, file_name=file_name)
    gmesh.build_conformal_mesh()
    if write_vtk_q:
        gmesh.write_vtk()
    return gmesh


def method_definition(k_order):
    method_cg = {
        "u": ("Lagrange", k_order),
    }
    methods = [method_cg]
    method_names = ["FEM"]
    return zip(method_names, methods)


def material_data_definition():
    # Material data for example 2
    case_0 = {"lambda": 1.0, "mu": 1.0, "kappa": 1.0e-6}
    case_1 = {"lambda": 1.0, "mu": 1.0, "kappa": 1.0}
    case_2 = {"lambda": 1.0, "mu": 1.0, "kappa": 1.0e6}
    cases = [case_0, case_1, case_2]
    cases = [case_1]
    return cases


def compose_file_name(method, ref_l, material_data, suffix):
    prefix = (
        "ex_2_"
        + method[0]
        + "_kappa_"
        + str(material_data["kappa"])
        + "_l_"
        + str(ref_l)
    )
    file_name = prefix + suffix
    return file_name


def perform_convergence_approximations(configuration: dict):
    # retrieve parameters from given configuration
    method = configuration.get("method")
    n_ref = configuration.get("n_refinements")
    dimension = configuration.get("dimension")
    material_data = configuration.get("material_data", {})
    write_geometry_vtk = configuration.get("write_geometry_Q", True)

    for lh in range(n_ref):
        mesh_file = f"gmsh_files/ex_2/partition_ex_2_l_{lh}.msh"
        gmesh = create_mesh_from_file(mesh_file, dimension, write_geometry_vtk)
        alpha, res_history = primal_approximation(material_data, method, gmesh)
        file_name = compose_file_name(method, lh, material_data, "_alpha.npy")
        with open(file_name, "wb") as f:
            np.save(f, alpha)
        file_name_res = compose_file_name(method, lh, material_data, "_res_history.txt")
        # First position includes n_dof
        np.savetxt(
            file_name_res,
            np.concatenate((np.array([len(alpha)]), res_history)),
            delimiter=",",
        )
    return


def perform_convergence_postprocessing(configuration: dict):
    # retrieve parameters from given configuration
    method = configuration.get("method")
    n_ref = configuration.get("n_refinements")
    dimension = configuration.get("dimension")
    material_data = configuration.get("material_data", {})
    write_geometry_vtk = configuration.get("write_geometry_Q", True)
    write_vtk = configuration.get("write_vtk_Q", True)
    report_full_precision_data = configuration.get("report_full_precision_data_Q", True)

    n_data = 4  # n_dof, n_iterations, h_max, displacement error
    error_data = np.empty((0, n_data), float)

    for lh in range(n_ref):
        mesh_file = f"gmsh_files/ex_2/partition_ex_2_l_{lh}.msh"
        gmesh = create_mesh_from_file(mesh_file, dimension, write_geometry_vtk)
        h_min, h_mean, h_max = mesh_size(gmesh)

        file_name = compose_file_name(method, lh, material_data, "_alpha.npy")
        with open(file_name, "rb") as f:
            alpha = np.load(f)
        n_dof, errors = primal_postprocessing(
            material_data, method, gmesh, alpha, write_vtk
        )

        chunk = np.array([n_dof, 1, h_max, errors[0]]) # 0 for n_iterations since we use direct solver
        error_data = np.append(error_data, np.array([chunk]), axis=0)

    # Calculate convergence rates (only for displacement error)
    rates_data = np.empty((0, 1), float)  # Only one rate for displacement
    for i in range(error_data.shape[0] - 1):
        chunk_b = np.log(error_data[i])
        chunk_e = np.log(error_data[i + 1])
        h_step = chunk_e[2] - chunk_b[2]  # using h_max
        partial = (chunk_e[3] - chunk_b[3]) / h_step  # rate for displacement error
        rates_data = np.append(rates_data, np.array([[partial]]), axis=0)

    # Print minimal report
    np.set_printoptions(precision=3)
    print(f"CG formulation: {method[0]}")
    print(f"Dimension: {dimension}")
    print("Error data (DOFs, iterations, h_max, L2-error):")
    print(error_data)
    print("Convergence rates:")
    print(rates_data)
    print("")

    # Save data to files
    kappa_value = material_data["kappa"]
    file_name_prefix = f"cg_ex_2_{method[0]}_kappa_{kappa_value}"

    e_str_header = "n_dof, n_iter, h, u_L2_error"
    r_str_header = "u_convergence_rate"

    if report_full_precision_data:
        np.savetxt(
            file_name_prefix + "_error.txt",
            error_data,
            delimiter=",",
            header=e_str_header
        )
        np.savetxt(
            file_name_prefix + "_rates.txt",
            rates_data,
            delimiter=",",
            header=r_str_header
        )

    # Save rounded data
    np.savetxt(
        file_name_prefix + "_error_rounded.txt",
        error_data,
        fmt="%1.3e",
        delimiter=",",
        header=e_str_header
    )
    np.savetxt(
        file_name_prefix + "_rates_rounded.txt",
        rates_data,
        fmt="%1.3f",
        delimiter=",",
        header=r_str_header
    )


def main():
    dimension = 2
    approximation_q = True
    postprocessing_q = True
    refinements = {2: 2}
    case_data = material_data_definition()

    for k in [2]:  # polynomial degree
        methods = method_definition(k)
        for method in methods:
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
