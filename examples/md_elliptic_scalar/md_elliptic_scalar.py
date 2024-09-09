import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc
import time

from exact_functions import get_exact_functions_by_co_dimension
from exact_functions import get_rhs_by_co_dimension
from postprocess.projectors import l2_projector
from postprocess.l2_error_post_processor import l2_error, l2_error_projected
from postprocess.solution_post_processor import write_vtk_file_with_exact_solution
from spaces.product_space import ProductSpace
from mesh.mesh import Mesh
from mesh.mesh_metrics import mesh_size
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
    fracture_0 = np.array([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]])
    fractures = [fracture_0]
    return np.array(fractures)


def generate_conformal_mesh(md_domain, h_val, n_ref, fracture_physical_tags):

    # For simplicity use h_val to control fracture refinement
    n_points = int(1.5 / h_val) + 1

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

    m_c1 = config["m_c1"]
    m_c2 = config["m_c2"]
    m_kappa_c0 = config["m_kappa_c0"]
    m_kappa_c1 = config["m_kappa_c1"]
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

    # compute mesh sizes per codimension
    h_sizes = []
    physical_tags = [1, 10]  # 1 for triangles (Rock) and 10 for lines (Fractures)
    for co_dim in [0, 1]:
        dim = gmesh.dimension - co_dim
        _, _, h_max = mesh_size(gmesh, dim=dim, physical_tag=physical_tags[co_dim])
        h_sizes.append(h_max)

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
        0, flux_name, potential_name, m_c1, m_c2, m_kappa_c0, m_kappa_c1, m_delta
    )
    exact_functions_c1 = get_exact_functions_by_co_dimension(
        1, flux_name, potential_name, m_c1, m_c2, m_kappa_c0, m_kappa_c1, m_delta
    )
    exact_functions = [exact_functions_c0, exact_functions_c1]

    rhs_c0 = get_rhs_by_co_dimension(
        0, "rhs", m_c1, m_c2, m_kappa_c0, m_kappa_c1, m_delta
    )
    rhs_c1 = get_rhs_by_co_dimension(
        1, "rhs", m_c1, m_c2, m_kappa_c0, m_kappa_c1, m_delta
    )

    print("Surface: Number of dof: ", md_produc_space[0].n_dof)
    print("Line: Number of dof: ", md_produc_space[1].n_dof)

    def f_kappa_c0(x, y, z):
        return m_kappa_c0

    def f_kappa_c1(x, y, z):
        return m_kappa_c1 * m_delta

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

    def scatter_coupling_form_data(A, c1_idx, c0_p_idx, c0_n_idx, int_weak_form):

        dest_c0_p = int_weak_form.space[0].bc_destination_indexes(c0_p_idx, "u")
        dest_c0_n = int_weak_form.space[0].bc_destination_indexes(c0_n_idx, "u")
        dest_c1 = int_weak_form.space[1].destination_indexes(c1_idx, "p")
        dest = np.concatenate([dest_c0_p, dest_c0_n, dest_c1])
        alpha_l = alpha[dest]
        r_el, j_el = int_weak_form.evaluate_form(c1_idx, c0_p_idx, c0_n_idx, alpha_l)

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
        for c1_idx, c0_p_idx, c0_n_idx in zip(c1_el_idx, c0_pel_idx, c0_nel_idx):
            scatter_coupling_form_data(
                A, c1_idx, c0_p_idx, c0_n_idx, int_coupling_weak_form
            )  # positive and negative at once

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

    # Some issue with PETSC solver
    # ksp.solve(b, x)
    # alpha = x.array

    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    # L2 error for mixed-dimensional solution
    errors_by_co_dim = []
    for co_dim in [0, 1]:
        dim = gmesh.dimension - co_dim
        print("Computing L2-error for co-dimension: ", co_dim)
        st = time.time()
        u_l2_error, p_l2_error = l2_error(
            gmesh.dimension - co_dim,
            md_produc_space[co_dim],
            exact_functions[co_dim],
            alpha,
        )

        # l2_error for projected pressure
        dof_shift = md_produc_space[co_dim].dof_shift
        n_dof = md_produc_space[co_dim].n_dof
        # compute projection on co-dimension co_dim
        alpha_proj = l2_projector(
            md_produc_space[co_dim], exact_functions[co_dim], -dof_shift
        )
        alpha_e = alpha[0 + dof_shift : n_dof + dof_shift : 1] - alpha_proj
        # compute l2_error of projected exact solution on co-dimension co_dim
        p_proj_l2_error = l2_error_projected(
            dim, md_produc_space[co_dim], alpha_e, ["u"], -dof_shift
        )[0]

        errors_by_co_dim.append((u_l2_error, p_l2_error, p_proj_l2_error))
        et = time.time()
        elapsed_time = et - st
        print("L2-error time:", elapsed_time, "seconds")
        print("L2-error in u: ", u_l2_error)
        print("L2-error in p: ", p_l2_error)
        print("L2-error in p projected: ", p_proj_l2_error)
        print("")

    case_names_by_co_dim = compose_case_name(config)
    for co_dim in [0, 1]:
        if write_vtk_q:
            print("Post-processing solution for co-dimension: ", co_dim)
            st = time.time()
            prefix = case_names_by_co_dim[co_dim]
            file_name = (
                prefix
                + "mesh_size_"
                + str(config["mesh_size"])
                + "_md_elliptic_two_fields.vtk"
            )
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

    h_size_and_error_data_by_co_dim = {
        0: (h_sizes[0], errors_by_co_dim[0]),
        1: (h_sizes[1], errors_by_co_dim[1]),
    }
    return h_size_and_error_data_by_co_dim


def compose_case_name(config):

    max_dim = 2
    case_names_by_co_dim = []

    k_order = config["k_order"]
    flux_name, potential_name = config["var_names"]
    m_c1 = config["m_c1"]
    m_c2 = config["m_c2"]
    m_kappa_c0 = config["m_kappa_c0"]
    m_kappa_c1 = config["m_kappa_c1"]
    m_kappa_normal = config["m_kappa_normal"]
    m_delta = config["m_delta"]
    folder_name = config.get("folder_name", None)
    for co_dim in [0, 1]:
        d = max_dim - co_dim
        methods = method_definition(d, k_order, flux_name, potential_name)
        for method in methods:
            case_name = (
                method[0]
                + "_"
                + "c_"
                + str(co_dim)
                + "_material_parameters_"
                + str(m_c1)
                + "_"
                + str(m_c2)
                + "_"
                + str(m_kappa_c0)
                + "_"
                + str(m_kappa_c1)
                + "_"
                + str(m_kappa_normal)
                + "_"
                + str(m_delta)
                + "_"
            )
            if folder_name is not None:
                import os

                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                case_name = folder_name + "/" + case_name
            case_names_by_co_dim.append(case_name)
    return case_names_by_co_dim


def compute_approximations(config):

    # Variable naming implemented in weakform
    config["var_names"] = ("u", "p")

    save_plot_rates_q = config["save_plot_rates_q"]
    errors_data = []
    h_sizes = []
    for h_size in config["mesh_sizes"]:
        config["mesh_size"] = h_size
        h_size_and_error_data_by_co_dim = md_two_fields_approximation(config, True)
        h_sizes.append(
            np.array(
                [
                    h_size_and_error_data_by_co_dim[0][0],
                    h_size_and_error_data_by_co_dim[1][0],
                ]
            )
        )
        errors_chunk = np.array(
            [
                np.array(h_size_and_error_data_by_co_dim[0][1]),
                np.array(h_size_and_error_data_by_co_dim[1][1]),
            ]
        )
        errors_data.append(errors_chunk)
    h_sizes = np.array(h_sizes)
    errors_data = np.array(errors_data)

    case_names_by_co_dim = compose_case_name(config)

    n_data = 4
    for co_dim in [0, 1]:
        case_name = case_names_by_co_dim[co_dim]
        h_data = np.array(h_sizes[:, co_dim])
        error_data = np.vstack([h_data, errors_data[:, co_dim].T]).T
        rates_data = np.empty((0, n_data - 1), float)
        for i in range(error_data.shape[0] - 1):
            chunk_b = np.log(error_data[i])
            chunk_e = np.log(error_data[i + 1])
            h_step = chunk_e[0] - chunk_b[0]
            partial = (chunk_e - chunk_b) / h_step
            rates_data = np.append(
                rates_data, np.array([list(partial[1:n_data])]), axis=0
            )

        rates_data = np.vstack((np.array([np.nan] * rates_data.shape[1]), rates_data))

        assert error_data.shape[0] == rates_data.shape[0]
        raw_data = np.zeros(
            (
                error_data.shape[0],
                error_data.shape[1] + rates_data.shape[1],
            ),
            dtype=error_data.dtype,
        )
        raw_data[:, 0] = error_data[:, 0]
        raw_data[:, 1::2] = error_data[:, 1 : error_data.shape[1]]
        raw_data[:, 2::2] = rates_data

        normal_conv_data = raw_data[:, 0 : raw_data.shape[1] - 2]
        enhanced_conv_data = raw_data[
            :,
            np.insert(np.arange(raw_data.shape[1] - 2, raw_data.shape[1]), 0, 0),
        ]

        np.set_printoptions(precision=5)
        print("normal convergence data: ", normal_conv_data)
        print("enhanced convergence data: ", enhanced_conv_data)

        normal_header = "h, u,  rate,   p,  rate"
        enhanced_header = "h,   proj p, rate"
        np.savetxt(
            case_name + "normal_conv_data.txt",
            normal_conv_data,
            delimiter=",",
            fmt="%1.6f",
            header=normal_header,
        )
        np.savetxt(
            case_name + "enhanced_conv_data.txt",
            enhanced_conv_data,
            delimiter=",",
            fmt="%1.6f",
            header=enhanced_header,
        )

    if save_plot_rates_q:
        for co_dim in [0, 1]:
            x = np.array(h_sizes[:, co_dim])
            y = errors_data[:, co_dim]  # u, p, p_proj
            lineObjects = plt.loglog(x, y, marker="o")
            plt.legend(
                iter(lineObjects),
                ("u", "p", "p_projected"),
            )
            plt.title("Errors on omega with co-dimension: " + str(co_dim))
            plt.xlabel("Element size")
            plt.ylabel("L2-error")
            figure_name = case_names_by_co_dim[co_dim] + "l2_error_plot.png"
            plt.savefig(figure_name)
            plt.clf()


def main():
    deltas_frac = [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5]
    for delta_frac in deltas_frac:
        config = {}
        # domain and discrete domain data
        config["lx"] = 1.0
        config["ly"] = 1.0

        # Material data
        config["m_c1"] = 1.0
        config["m_c2"] = 1.0
        config["m_kappa_c0"] = 1.0
        config["m_kappa_c1"] = 1 / delta_frac
        config["m_kappa_normal"] = 1.0
        config["m_delta"] = delta_frac

        # function space data
        config["n_ref"] = 0
        config["k_order"] = 0
        config["mesh_sizes"] = [
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
            0.0078125,
            0.00390625,
            0.001953125,
            0.0009765625,
        ]

        # output data
        config["folder_name"] = "output"
        config["save_plot_rates_q"] = True

        compute_approximations(config)


if __name__ == "__main__":
    main()
