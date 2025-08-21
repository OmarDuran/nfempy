import resource
import time
import numpy as np

from mesh.mesh import Mesh
from petsc4py import PETSc
from postprocess.solution_post_processor import write_vtk_file
from spaces.product_space import ProductSpace
from weak_forms.le_primal_weak_form import LEPrimalWeakForm, LEPrimalWeakFormBCDirichlet, LEPrimalWeakFormBCNeumann


def create_product_space(method, gmesh, mat_ids):
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
        "u": mat_ids['under'] + mat_ids['reservoir'] + mat_ids['over'],  # domain tags
    }
    # Boundary tags: 5=left, 6=right, 7=bottom, 8=top (assumed from mesh)
    b_physical_tags = {
        "u": mat_ids['bc_bottom'] + mat_ids['bc_top'] + mat_ids['bc_west'] + mat_ids['bc_east'],
    }

    space = ProductSpace(discrete_spaces_data)
    space.make_subspaces_discontinuous(discrete_spaces_disc)
    space.build_structures(physical_tags, b_physical_tags)
    return space


def primal_approximation_with_load(material_data, method, gmesh, mat_ids):
    dim = gmesh.dimension
    fe_space = create_product_space(method, gmesh, mat_ids)
    n_dof_g = fe_space.n_dof

    rg = np.zeros(n_dof_g)
    alpha = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    memory_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    st = time.time()

    A = PETSc.Mat()
    A.createAIJ([n_dof_g, n_dof_g])
    A.setType("sbaij")

    m_lambda = material_data["lambda"]
    m_mu = material_data["mu"]

    # Zero body force
    def f_rhs(x, y, z):
        return np.zeros((2, x.shape[0]))

    def f_lambda(x, y, z):
        return m_lambda * np.ones_like(x)

    def f_mu(x, y, z):
        return m_mu * np.ones_like(x)

    m_functions = {
        "rhs": f_rhs,
        "lambda": f_lambda,
        "mu": f_mu,
    }

    weak_form = LEPrimalWeakForm(fe_space)
    weak_form.functions = m_functions

    # Dirichlet BC: zero displacement on left, right, bottom (tags 5, 6, 7)
    def zero_disp(x, y, z):
        return np.array(
            [
                np.zeros_like(x),
                np.zeros_like(y),
            ]
        )

    bc_dirichlet = LEPrimalWeakFormBCDirichlet(fe_space)
    bc_dirichlet.functions = {"u": zero_disp}

    # Neumann BC: vertical load on top (tag 8)
    def vertical_load(x, y, z):
        # Apply a vertical load of magnitude 1.0 in y direction
        return np.array([np.zeros_like(x), 0.1 * np.ones_like(x)])

    bc_neumann = LEPrimalWeakFormBCNeumann(fe_space)
    bc_neumann.functions = {"t": vertical_load}


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

    def scatter_bc_dirichlet(A, i, bc_weak_form):
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

    def scatter_bc_neumann(i, bc_weak_form):
        dest = fe_space.bc_destination_indexes(i)
        alpha_l = alpha[dest]
        r_el, _ = bc_weak_form.evaluate_form(i, alpha_l)
        rg[dest] += r_el

    n_els = len(fe_space.discrete_spaces["u"].elements)
    [scatter_form_data(A, i, weak_form, n_els) for i in range(n_els)]

    # get bounday elements
    bc_elements = fe_space.discrete_spaces["u"].bc_elements
    bc_top_idx = [i for i, bel in enumerate(bc_elements) if bel.data.cell.material_id in mat_ids['bc_top']]
    bc_bottom_idx = [i for i, bel in enumerate(bc_elements) if bel.data.cell.material_id in mat_ids['bc_bottom']]
    bc_laterals_idx = [i for i, bel in enumerate(bc_elements) if bel.data.cell.material_id in mat_ids['bc_west']+mat_ids['bc_east']]

    [scatter_bc_dirichlet(A, i, bc_dirichlet) for i in bc_bottom_idx]
    [scatter_bc_dirichlet(A, i, bc_dirichlet) for i in bc_laterals_idx]
    [scatter_bc_neumann(i, bc_neumann) for i in bc_top_idx]

    A.assemble()
    print("Assembly: nz_allocated:", int(A.getInfo()["nz_allocated"]))
    print("Assembly: nz_used:", int(A.getInfo()["nz_used"]))
    print("Assembly: nz_unneeded:", int(A.getInfo()["nz_unneeded"]))

    et = time.time()
    print("Assembly: Time:", et - st, "seconds")

    st = time.time()
    ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
    ksp.setOperators(A)
    b = A.createVecLeft()
    b.array[:] = -rg
    x = A.createVecRight()

    ksp.setType("cg")
    ksp.getPC().setType("icc")
    ksp.setTolerances(rtol=0.0, atol=1e-10, divtol=5000, max_it=1000)
    ksp.setFromOptions()

    ksp.solve(b, x)
    alpha = x.array

    PETSc.KSP.destroy(ksp)
    PETSc.Mat.destroy(A)
    PETSc.Vec.destroy(b)
    PETSc.Vec.destroy(x)

    et = time.time()
    print("Linear solver: Time:", et - st, "seconds")

    return alpha


def main():

    mat_ids = {
        'under': [1],
        'reservoir': [2],
        'over': [3],
        'bc_bottom': [4],
        'bc_top': [5],
        'bc_west': [6],
        'bc_east': [7],
    }

    dimension = 2
    mesh_file = "gmsh_files/ex_1/example_1_2d.msh"
    gmesh = Mesh(dimension=dimension, file_name=mesh_file)
    gmesh.build_conformal_mesh()
    gmesh.write_vtk()
    material_data = {"lambda": 1.0, "mu": 1.0}
    method = ("FEM", {"u": ("Lagrange", 2)})

    alpha = primal_approximation_with_load(material_data, method, gmesh, mat_ids)

    # Output VTK file
    fe_space = create_product_space(method, gmesh, mat_ids)
    file_name = "constrained_le.vtk"
    write_vtk_file(
        file_name, gmesh, fe_space, alpha,
    )
    print(f"Results written to {file_name}")


if __name__ == "__main__":
    main()
