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
from weak_forms.laplace_dual_weak_form import LaplaceDualWeakForm as MixedWeakForm
from weak_forms.laplace_dual_weak_form import (
    LaplaceDualWeakFormBCDirichlet as WeakFormBCDir,
)
from CouplingWeakForm import CouplingWeakForm
import matplotlib.pyplot as plt


def method_definition(dimension, k_order, flux_name, potential_name):

    # lower order convention
    if dimension in [1, 2, 3]:
        method_1 = {
            flux_name: ("RT", k_order + 1),
            potential_name: ("Lagrange", k_order),
        }
    else:
        method_1 = {
            potential_name: ("Lagrange", k_order),
        }

    methods = [method_1]
    method_names = ["mixed_rt"]
    return zip(method_names, methods)


def create_product_space(dimension, method, gmesh, flux_name, potential_name):

    # FESpace: data
    mp_k_order = method[1][flux_name][1]
    p_k_order = method[1][potential_name][1]

    mp_components = 1
    p_components = 1

    mp_family = method[1][flux_name][0]
    p_family = method[1][potential_name][0]

    discrete_spaces_data = {
        flux_name: (dimension, mp_components, mp_family, mp_k_order, gmesh),
        potential_name: (dimension, p_components, p_family, p_k_order, gmesh),
    }

    mp_disc_Q = False
    p_disc_Q = True
    discrete_spaces_disc = {
        flux_name: mp_disc_Q,
        potential_name: p_disc_Q,
    }

    if gmesh.dimension == 2:
        md_field_physical_tags = [[], [10], [1]]
        mp_field_bc_physical_tags = [[], [2, 4], [2, 3, 4, 5, 50]]
    else:
        raise ValueError("Case not available.")

    physical_tags = {
        flux_name: md_field_physical_tags[dimension],
        potential_name: md_field_physical_tags[dimension],
    }

    b_physical_tags = {
        flux_name: mp_field_bc_physical_tags[dimension],
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
    flux_name, potential_name = config["var_names"]

    m_c = config["m_c"]
    m_kappa = config["m_kappa"]
    m_kappa_normal = config["m_kappa_normal"]
    m_delta = config["m_delta"]

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
        methods = method_definition(d, k_order, flux_name, potential_name)
        for method in methods:
            fe_space = create_product_space(d, method, gmesh, flux_name, potential_name)
            md_produc_space.append(fe_space)

    exact_functions_c0 = get_exact_functions_by_co_dimension(
        0, flux_name, potential_name, m_c, m_kappa, m_delta
    )
    exact_functions_c1 = get_exact_functions_by_co_dimension(
        1, flux_name, potential_name, m_c, m_kappa, m_delta
    )
    exact_functions = [exact_functions_c0, exact_functions_c1]

    rhs_c0 = get_rhs_by_co_dimension(0, "rhs", m_c, m_kappa, m_delta)
    rhs_c1 = get_rhs_by_co_dimension(1, "rhs", m_c, m_kappa, m_delta)

    print("Surface: Number of dof: ", md_produc_space[0].n_dof)
    print("Line: Number of dof: ", md_produc_space[1].n_dof)

    def f_kappa_c0(x, y, z):
        return m_kappa

    def f_kappa_c1(x, y, z):
        return m_kappa * m_delta

    def f_kappa_normal_c1(x, y, z):
        return m_kappa_normal

    def f_delta(x, y, z):
        return m_delta

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
        "rhs": rhs_c0["rhs"],
        "kappa": f_kappa_c0,
    }

    m_functions_c1 = {
        "rhs": rhs_c1["rhs"],
        "kappa": f_kappa_c1,
    }

    m_functions_coupling = {
        "delta": f_delta,
        "kappa_normal": f_kappa_normal_c1,
    }

    weak_form_c0 = MixedWeakForm(md_produc_space[0])
    weak_form_c0.functions = m_functions_c0

    weak_form_c1 = MixedWeakForm(md_produc_space[1])
    weak_form_c1.functions = m_functions_c1

    bc_weak_form_c0 = WeakFormBCDir(md_produc_space[0])
    bc_weak_form_c0.functions = exact_functions_c0

    bc_weak_form_c1 = WeakFormBCDir(md_produc_space[1])
    bc_weak_form_c1.functions = exact_functions_c1

    int_coupling_weak_form = CouplingWeakForm(md_produc_space)
    int_coupling_weak_form.functions = m_functions_coupling

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

    def scatter_coupling_form_data(A, c0_idx, c1_idx, int_weak_form):

        dest_c0 = int_weak_form.space[0].bc_destination_indexes(c0_idx, "u")
        dest_c1 = int_weak_form.space[1].destination_indexes(c1_idx, "p")
        dest = np.concatenate([dest_c0, dest_c1])
        alpha_l = alpha[dest]
        r_el, j_el = int_weak_form.evaluate_form(c0_idx, c1_idx, alpha_l)

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

    all_b_cell_c0_ids = md_produc_space[0].discrete_spaces["u"].bc_element_ids
    eb_c0_ids = [
        id
        for id in all_b_cell_c0_ids
        if gmesh.cells[id].material_id != physical_tags["line_clones"]
    ]
    eb_c0_el_idx = [
        md_produc_space[0].discrete_spaces["u"].id_to_bc_element[id] for id in eb_c0_ids
    ]
    [scatter_bc_form(A, i, bc_weak_form_c0) for i in eb_c0_el_idx]

    n_bc_els_c1 = len(md_produc_space[1].discrete_spaces["u"].bc_elements)
    [scatter_bc_form(A, i, bc_weak_form_c1) for i in range(n_bc_els_c1)]

    # Interface weak forms
    for interface in interfaces:
        c1_data = interface["c1"]
        c1_el_idx = [
            md_produc_space[1].discrete_spaces["u"].id_to_element[cell.id]
            for cell in c1_data[0]
        ]
        c0_pel_idx = [
            md_produc_space[0].discrete_spaces["u"].id_to_bc_element[cell.id]
            for cell in c1_data[1]
        ]
        c0_nel_idx = [
            md_produc_space[0].discrete_spaces["u"].id_to_bc_element[cell.id]
            for cell in c1_data[2]
        ]
        for c1_idx, p_c0_idx, n_c0_idx in zip(c1_el_idx, c0_pel_idx, c0_nel_idx):
            scatter_coupling_form_data(
                A, p_c0_idx, c1_idx, int_coupling_weak_form
            )  # positive side
            scatter_coupling_form_data(
                A, n_c0_idx, c1_idx, int_coupling_weak_form
            )  # negative side

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
        u_l2_error, p_l2_error = l2_error(
            gmesh.dimension - co_dim,
            md_produc_space[co_dim],
            exact_functions[co_dim],
            alpha,
        )
        errors_by_co_dim.append((u_l2_error, p_l2_error))
        et = time.time()
        elapsed_time = et - st
        print("L2-error time:", elapsed_time, "seconds")
        print("L2-error in u: ", u_l2_error)
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
    plot_rates_q = True
    config = {}
    # domain and discrete domain data
    config["lx"] = 1.0
    config["ly"] = 1.0

    # Material data
    config["m_c"] = 1.0
    config["m_kappa"] = 1.0
    config["m_kappa_normal"] = 1.0
    config["m_delta"] = 1.0e-3

    # function space data
    config["n_ref"] = 0
    config["k_order"] = 1
    config["var_names"] = ("u", "p")

    errors_data = []
    h_sizes = []
    for h_size in [0.5, 0.25, 0.125, 0.0625, 0.03125]:
        config["mesh_size"] = h_size
        h_sizes.append(h_size)
        error_data = md_two_fields_approximation(config, True)
        errors_data.append(np.array(error_data))
    errors_data = np.array(errors_data)

    if plot_rates_q:
        x = np.array(h_sizes)
        y = np.hstack(
            (
                errors_data[:, 0:2:2, 0], # u_c0
                errors_data[:, 0:2:2, 1], # p_c0
                errors_data[:, 1:2:2, 0], # u_c1
                errors_data[:, 1:2:2, 1], # p_c1
            )
        )
        lineObjects = plt.loglog(x, y)
        plt.legend(
            iter(lineObjects),
            ("u_c0", "p_c0", "u_c1", "p_c1"),
        )
        plt.title("")
        plt.xlabel("Element size")
        plt.ylabel("L2-error")
        plt.show()


if __name__ == "__main__":
    main()
