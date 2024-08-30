import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc
import time

from exact_functions import get_exact_functions_by_co_dimension
from exact_functions import get_rhs_by_co_dimension
from postprocess.l2_error_post_processor import l2_error
from postprocess.solution_post_processor import write_vtk_file_with_exact_solution
from spaces.product_space import ProductSpace
from spaces.md_product_space import MDProductSpace
from mesh.mesh import Mesh
from topology.domain_market import create_md_box_2D
from mesh.discrete_domain import DiscreteDomain
from mesh.mesh_operations import cut_conformity_along_c1_lines

# simple weak form
from weak_forms.lce_primal_weak_form import LCEPrimalWeakForm as PrimalWeakForm
from weak_forms.lce_primal_weak_form import LCEPrimalWeakFormBCDirichlet as WeakFormBCDir
from ContactWeakForm import ContactWeakForm
import matplotlib.pyplot as plt


def method_definition(dimension, k_order, displacement_name, rotation_name):

    method_1 = {
        displacement_name: ("Lagrange", k_order),
        rotation_name: ("Lagrange", k_order),
    }

    methods = [method_1]
    method_names = ["primal"]
    return zip(method_names, methods)


def create_product_space(dimension, method, gmesh, displacement_name, rotation_name):

    # FESpace: data
    u_k_order = method[1][displacement_name][1]
    r_k_order = method[1][rotation_name][1]

    u_components = 2
    r_components = 1

    u_family = method[1][displacement_name][0]
    p_family = method[1][rotation_name][0]

    discrete_spaces_data = {
        displacement_name: (dimension, u_components, u_family, u_k_order, gmesh),
        rotation_name: (dimension, r_components, p_family, r_k_order, gmesh),
    }

    u_disc_Q = False
    p_disc_Q = False
    discrete_spaces_disc = {
        displacement_name: u_disc_Q,
        rotation_name: p_disc_Q,
    }

    if gmesh.dimension == 2:
        u_field_physical_tags = [[], [50], [1]]
        r_field_physical_tags = [[], [50], [1]]
        u_field_bc_physical_tags = [[], [2, 4], [2, 3, 4, 5]]
        r_field_bc_physical_tags = [[], [2, 4], [2, 3, 4, 5]]
    else:
        raise ValueError("Case not available.")

    physical_tags = {
        displacement_name: u_field_physical_tags[dimension],
        rotation_name: r_field_physical_tags[dimension],
    }

    b_physical_tags = {
        displacement_name: u_field_bc_physical_tags[dimension],
        rotation_name: r_field_bc_physical_tags[dimension],
    }

    space = ProductSpace(discrete_spaces_data)
    space.make_subspaces_discontinuous(discrete_spaces_disc)
    space.build_structures(physical_tags, b_physical_tags)
    return space


def fracture_disjoint_set():
    fracture_0 = np.array([[0.5, 0.0, 0.0], [0.5, 1.0, 0.0]])
    fractures = [fracture_0]
    return np.array(fractures)


def generate_conformal_mesh(md_domain, h_val, n_ref, fracture_physical_tags):

    # For simplicity use h_val to control fracture refinement
    n_points = int(1.5/h_val) + 1

    physical_tags = [fracture_physical_tags["line"]]
    transfinite_agruments = {"n_points": n_points, "meshType": "Bump", "coef": 1.0}
    mesh_arguments = {
        "lc": h_val,
        "n_refinements": n_ref,
        "curves_refinement": (physical_tags, transfinite_agruments),
    }

    domain_h = DiscreteDomain(dimension=md_domain.dimension)
    domain_h.domain = md_domain
    domain_h.generate_mesh(mesh_arguments)
    domain_h.write_mesh("gmesh.msh")

    # Mesh representation
    gmesh = Mesh(dimension=md_domain.dimension, file_name="gmesh.msh")
    gmesh.build_conformal_mesh()
    gmesh.write_vtk()
    return gmesh


def md_two_fields_approximation(config, write_vtk_q=False):

    k_order = config["k_order"]
    displacement_name, rotation_name = config["var_names"]

    m_lambda = config["m_lambda"]
    m_mu = config["m_mu"]
    m_kappa = config["m_kappa"]
    m_gamma = config["m_gamma"]

    m_lambda_c1 = config["m_lambda_c1"]
    m_mu_c1 = config["m_mu_c1"]
    m_kappa_c1 = config["m_kappa_c1"]
    m_gamma_c1 = config["m_gamma_c1"]

    m_cu_normal = config["m_cu_normal"]
    m_cu_tangential = config["m_cu_tangential"]
    m_cr_normal = config["m_cr_normal"]
    m_cr_tangential = config["m_cr_tangential"]

    domain_physical_tags = {"area": 1, "bc_0": 2, "bc_1": 3, "bc_2": 4, "bc_3": 5}
    box_points = np.array(
        [
            [0, 0, 0],
            [config["lx"], 0, 0],
            [config["lx"], config["ly"], 0],
            [0, config["ly"], 0],
        ]
    )

    # fracture data
    lines = fracture_disjoint_set()
    fracture_physical_tags = {"line": 10, "internal_bc": 20, "point": 30}
    md_domain = create_md_box_2D(
        box_points, domain_physical_tags, lines, fracture_physical_tags
    )

    # Conformal gmsh discrete representation
    gmesh = generate_conformal_mesh(
        md_domain, config["mesh_size"], config["n_ref"], fracture_physical_tags
    )

    physical_tags = fracture_physical_tags
    physical_tags["line_clones"] = 50
    physical_tags["point_clones"] = 100
    interfaces = cut_conformity_along_c1_lines(lines, physical_tags, gmesh, False)
    gmesh.write_vtk()

    md_produc_space = []
    for d in [2, 1]:
        methods = method_definition(d, k_order, displacement_name, rotation_name)
        for method in methods:
            fe_space = create_product_space(d, method, gmesh, displacement_name, rotation_name)
            md_produc_space.append(fe_space)

    print("Surface: Number of dof: ", md_produc_space[0].n_dof)
    print("Line: Number of dof: ", md_produc_space[1].n_dof)

    m_lambda = config["m_lambda"]
    m_mu = config["m_mu"]
    m_kappa = config["m_kappa"]
    m_gamma = config["m_gamma"]

    m_lambda_c1 = config["m_lambda_c1"]
    m_mu_c1 = config["m_mu_c1"]
    m_kappa_c1 = config["m_kappa_c1"]
    m_gamma_c1 = config["m_gamma_c1"]

    def f_lambda_c0(x, y, z):
        return m_lambda
    def f_mu_c0(x, y, z):
        return m_mu
    def f_kappa_c0(x, y, z):
        return m_kappa
    def f_gamma_c0(x, y, z):
        return m_gamma
    def f_rhs_c0(x, y, z):
        return np.ones_like([x,x,x,x])

    def f_lambda_c1(x, y, z):
        return m_lambda_c1
    def f_mu_c1(x, y, z):
        return m_mu_c1
    def f_kappa_c1(x, y, z):
        return m_kappa_c1
    def f_gamma_c1(x, y, z):
        return m_gamma_c1
    def f_rhs_c1(x, y, z):
        return np.ones_like([x,x,x,x])

    def f_cu_normal(x, y, z):
        return m_cu_normal
    def f_cu_tangential(x, y, z):
        return m_cu_tangential
    def f_cr_normal(x, y, z):
        return m_cr_normal
    def f_cr_tangential(x, y, z):
        return m_cr_tangential

    # primitive assembly
    dof_seq = np.array([0, md_produc_space[0].n_dof, md_produc_space[1].n_dof])
    global_dof = np.add.accumulate(dof_seq)
    md_produc_space[0].dof_shift = global_dof[0]
    md_produc_space[1].dof_shift = global_dof[1]
    n_dof_g = np.sum(dof_seq)
    rg = np.zeros(n_dof_g)
    alpha = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    # Assembler
    st = time.time()
    A = PETSc.Mat()
    A.createAIJ([n_dof_g, n_dof_g])

    m_functions_c0 = {
        "rhs": f_rhs_c0,
        "lambda": f_lambda_c0,
        "mu": f_mu_c0,
        "kappa": f_kappa_c0,
        "gamma": f_gamma_c0,
    }

    m_functions_c1 = {
        "rhs": f_rhs_c1,
        "lambda": f_lambda_c1,
        "mu": f_mu_c1,
        "kappa": f_kappa_c1,
        "gamma": f_gamma_c1,
    }

    def uD_c0(x, y, z):
        return np.zeros_like([x,x])
    def rD_c0(x, y, z):
        return np.zeros_like([x,x])

    def uD_c1(x, y, z):
        return np.zeros_like([x, x])
    def rD_c1(x, y, z):
        return np.zeros_like([x, x])

    bc_functions_c0 = {
        "u": uD_c0,
        "r": rD_c0,
    }

    bc_functions_c1 = {
        "u": uD_c1,
        "r": rD_c1,
    }

    m_functions_contact = {
        "cu_normal": f_cu_normal,
        "cu_tangential": f_cu_tangential,
        "cr_normal": f_cr_normal,
        "cr_tangential": f_cr_tangential,
    }

    weak_form_c0 = PrimalWeakForm(md_produc_space[0])
    weak_form_c0.functions = m_functions_c0

    weak_form_c1 = PrimalWeakForm(md_produc_space[1])
    weak_form_c1.functions = m_functions_c1

    bc_weak_form_c0 = WeakFormBCDir(md_produc_space[0])
    bc_weak_form_c0.functions = bc_functions_c0

    bc_weak_form_c1 = WeakFormBCDir(md_produc_space[1])
    bc_weak_form_c1.functions = bc_functions_c1

    int_contact_weak_form = ContactWeakForm(md_produc_space)
    int_contact_weak_form.functions = m_functions_contact

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

        dest = bc_weak_form.space.bc_destination_indexes(i)
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

    def scatter_contact_form_data(A, c0_idx_p, c0_idx_n, int_weak_form):

        dest_p = int_weak_form.space[0].bc_destination_indexes(c0_idx_p)
        dest_n = int_weak_form.space[0].bc_destination_indexes(c0_idx_n)
        dest = np.concatenate([dest_p, dest_n])
        alpha_l = alpha[dest]
        r_el, j_el = int_weak_form.evaluate_form(c0_idx_p, c0_idx_n, alpha_l)

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        for k in range(nnz):
            A.setValue(row=row[k], col=col[k], value=data[k], addv=True)

    n_els_c0 = len(md_produc_space[0].discrete_spaces["u"].elements)
    n_els_c1 = len(md_produc_space[1].discrete_spaces["u"].elements)
    [scatter_form_data(A, i, weak_form_c0) for i in range(n_els_c0)]
    [scatter_form_data(A, i, weak_form_c1) for i in range(n_els_c1)]

    all_b_cell_c0_ids = md_produc_space[0].discrete_spaces["q"].bc_element_ids
    eb_c0_ids = [
        id
        for id in all_b_cell_c0_ids
        if gmesh.cells[id].material_id != physical_tags["line_clones"]
    ]
    eb_c0_el_idx = [
        md_produc_space[0].discrete_spaces["q"].id_to_bc_element[id] for id in eb_c0_ids
    ]
    [scatter_bc_form(A, i, bc_weak_form_c0) for i in eb_c0_el_idx]

    n_bc_els_c1 = len(md_produc_space[1].discrete_spaces["q"].bc_elements)
    [scatter_bc_form(A, i, bc_weak_form_c1) for i in range(n_bc_els_c1)]

    # Interface weak forms
    for interface in interfaces:
        c1_data = interface["c1"]
        c1_el_idx = [
            md_produc_space[1].discrete_spaces["q"].id_to_element[cell.id]
            for cell in c1_data[0]
        ]
        c0_pel_idx = [
            md_produc_space[0].discrete_spaces["q"].id_to_bc_element[cell.id]
            for cell in c1_data[1]
        ]
        c0_nel_idx = [
            md_produc_space[0].discrete_spaces["q"].id_to_bc_element[cell.id]
            for cell in c1_data[2]
        ]
        for c1_idx, p_c0_idx, n_c0_idx in zip(c1_el_idx, c0_pel_idx, c0_nel_idx):
            scatter_contact_form_data(
                A, p_c0_idx, c1_idx, int_contact_weak_form
            )  # positive - negative side

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

    ai, aj, av = A.getValuesCSR()
    jac_sp = sp.csr_matrix((av, aj, ai))
    alpha = sp.linalg.spsolve(jac_sp, -rg)

    # ksp.solve(b, x)
    # alpha = x.array

    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    # L2 error for mixed-dimensional solution
    errors_by_co_dim = []
    for co_dim in [0, 1]:
        print("Computing L2-error for co-dimension: ", co_dim)
        st = time.time()
        q_l2_error, p_l2_error = l2_error(
            gmesh.dimension - co_dim,
            md_produc_space[co_dim],
            exact_functions[co_dim],
            alpha,
        )
        errors_by_co_dim.append((q_l2_error, p_l2_error))
        et = time.time()
        elapsed_time = et - st
        print("L2-error time:", elapsed_time, "seconds")
        print("L2-error in q: ", q_l2_error)
        print("L2-error in p: ", p_l2_error)
        print("")

    for co_dim in [0, 1]:
        if write_vtk_q:
            print("Post-processing solution for co-dimension: ", co_dim)
            st = time.time()
            file_name = "md_elliptic_two_fields_c" + str(co_dim) + ".vtk"
            write_vtk_file_with_exact_solution(
                file_name,
                gmesh,
                md_produc_space[co_dim],
                exact_functions[co_dim],
                alpha,
            )
            et = time.time()
            elapsed_time = et - st
            print("Post-processing time:", elapsed_time, "seconds")
            print("")

    return errors_by_co_dim


def main():

    # plot_rates_q = True
    config = {}
    # domain and discrete domain data
    config["lx"] = 1.0
    config["ly"] = 1.0

    # Material data
    config["m_lambda"] = 1.0
    config["m_mu"] = 1.0
    config["m_kappa"] = 1.0
    config["m_gamma"] = 1.0

    config["m_lambda_c1"] = 1.0
    config["m_mu_c1"] = 1.0
    config["m_kappa_c1"] = 1.0
    config["m_gamma_c1"] = 1.0

    config["m_cu_normal"] = 1.0
    config["m_cu_tangential"] = 1.0
    config["m_cr_normal"] = 1.0
    config["m_cr_tangential"] = 1.0

    # function space data
    config["n_ref"] = 0
    config["k_order"] = 1
    config["var_names"] = ("u", "t")

    # errors_data = []
    # h_sizes = []
    for h_size in [0.5]:
        config["mesh_size"] = h_size
        # h_sizes.append(h_size)
        error_data = md_two_fields_approximation(config, True)
        # errors_data.append(np.array(error_data))
    # errors_data = np.array(errors_data)

    # if plot_rates_q:
    #     x = np.array(h_sizes)
    #     y = np.hstack(
    #         (
    #             errors_data[:, 0:2:2, 0], # q_c0
    #             errors_data[:, 0:2:2, 1], # p_c0
    #             errors_data[:, 1:2:2, 0], # q_c1
    #             errors_data[:, 1:2:2, 1], # p_c1
    #         )
    #     )
    #     lineObjects = plt.loglog(x, y)
    #     plt.legend(
    #         iter(lineObjects),
    #         ("q_c0", "p_c0", "q_c1", "p_c1"),
    #     )
    #     plt.title("")
    #     plt.xlabel("Element size")
    #     plt.ylabel("L2-error")
    #     plt.show()


if __name__ == "__main__":
    main()
