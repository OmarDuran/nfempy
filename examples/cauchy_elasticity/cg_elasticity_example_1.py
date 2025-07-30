import resource
import time
from functools import partial

import numpy as np
from mesh.mesh import Mesh
from mesh.mesh_metrics import mesh_size
from petsc4py import PETSc
from postprocess.l2_error_post_processor import l2_error
from postprocess.solution_norms_post_processor import l2_norm
from postprocess.solution_post_processor import write_vtk_file_with_exact_solution
from spaces.product_space import ProductSpace
from weak_forms.le_primal_weak_form import LEPrimalWeakForm, LEPrimalWeakFormBCDirichlet

import strong_solution_elasticity_example_1 as le


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

    # exact solution and functions
    u_exact = le.displacement(m_lambda, m_mu, dim)  # kappa=0 for classical elasticity
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
    ksp.getPC().setType("icc")  # Algebraic multigrid preconditioner
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

    # exact solution
    u_exact = le.displacement(m_lambda, m_mu, dim)

    exact_functions = {
        "u": u_exact,
    }

    if write_vtk_q:
        st = time.time()
        prefix = "cg_ex_" + method[0] + "_lambda_" + str(material_data["lambda"])
        file_name = prefix + ".vtk"
        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, exact_functions, alpha,
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
        "u": ("Lagrange", k_order),  # k+1 for optimal convergence
    }
    methods = [method_cg]
    method_names = ["CG"]
    return zip(method_names, methods)


def material_data_definition():
    case_0 = {"lambda": 1.0, "mu": 1.0}
    case_1 = {"lambda": 10.0, "mu": 1.0}
    case_2 = {"lambda": 100.0, "mu": 1.0}
    # cases = [case_0, case_1, case_2]
    cases = [case_0]
    return cases


def compose_file_name(method, ref_l, material_data, suffix):
    prefix = (
        "cg_ex_"
        + method[0]
        + "_lambda_"
        + str(material_data["lambda"])
        + "_l_"
        + str(ref_l)
    )
    file_name = prefix + suffix
    return file_name


def main():
    dimension = 2
    approximation_q = True
    postprocessing_q = True
    refinements = {1: 2}  # 4 refinement levels for k=1
    case_data = material_data_definition()

    for k in [1]:  # Could extend to higher orders
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

                print(f"\nRunning CG elasticity with:")
                print(f"Method: {method[0]}, k={k}")
                print(f"lambda={material_data['lambda']}, mu={material_data['mu']}")

                for lh in range(refinements[k]):
                    mesh_file = f"gmsh_files/ex_1/partition_ex_1_l_{lh}.msh"
                    gmesh = create_mesh_from_file(mesh_file, dimension, True)
                    h_min, h_mean, h_max = mesh_size(gmesh)
                    print(f"\nMesh level {lh}, h_max = {h_max}")

                    if approximation_q:
                        alpha, res_history = primal_approximation(material_data, method, gmesh)
                        file_name = compose_file_name(method, lh, material_data, "_alpha.npy")
                        with open(file_name, "wb") as f:
                            np.save(f, alpha)

                    if postprocessing_q:
                        file_name = compose_file_name(method, lh, material_data, "_alpha.npy")
                        with open(file_name, "rb") as f:
                            alpha = np.load(f)
                        n_dof, errors = primal_postprocessing(
                            material_data, method, gmesh, alpha, True
                        )
                        print(f"DOFs: {n_dof}")
                        print(f"L2 error: {errors[0]}")


if __name__ == "__main__":
    main()
