import time

import numpy as np
from petsc4py import PETSc

from topology.domain import Domain
from domain_builder import build_line_1D, build_surface_2D
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh
from mesh.mesh_metrics import mesh_size
from postprocess.l2_error_post_processor import l2_error, l2_error_projected
from postprocess.projectors import l2_projector
from postprocess.solution_post_processor import (
    write_fe_spaces_on_vtk_file_with_exact_solution
)
from spaces.product_space import ProductSpace
from weak_forms.unscaled_elliptic_weak_form import (
    UnscaledEllipticWeakForm,
    UnscaledEllipticWeakFormBCDirichlet,
)
import strong_solutions_TArbogast as exact_funcs
from functools import partial
import matplotlib.pyplot as plt


def create_product_spaces(method, gmesh, flux_names, potential_names):
    # FESpace: data
    u_k_order = method[1][flux_names["unscaled"]][1]
    p_k_order = method[1][potential_names["unscaled"]][1]

    u_family = method[1][flux_names["unscaled"]][0]
    p_family = method[1][potential_names["unscaled"]][0]

    flux_components = 1
    potential_components = 1

    # Unscaled space data
    discrete_spaces_data_unscaled = {
        flux_names["unscaled"]: (gmesh.dimension, flux_components, u_family, u_k_order, gmesh),
        potential_names["unscaled"]: (gmesh.dimension, potential_components, p_family, p_k_order, gmesh),
    }

    flux_disc_Q = False
    potential_disc_Q = True

    # Unscaled space discontinuity data
    discrete_spaces_disc_unscaled = {
        flux_names["unscaled"]: flux_disc_Q,
        potential_names["unscaled"]: potential_disc_Q,
    }

    if gmesh.dimension == 1:
        flux_unscale_bc_physical_tags = [3, 4]
    elif gmesh.dimension == 2:
        flux_unscale_bc_physical_tags = [3, 4, 5, 6, 7, 8]
    else:
        raise ValueError("Case not available.")

    # Unscaled space physical tags
    physical_tags_unscaled = {
        flux_names["unscaled"]: [1, 2],
        potential_names["unscaled"]: [1, 2],
    }

    b_physical_tags_unscaled = {
        flux_names["unscaled"]: flux_unscale_bc_physical_tags,
    }

    # Create unscaled space
    space_unscaled = ProductSpace(discrete_spaces_data_unscaled)
    space_unscaled.make_subspaces_discontinuous(discrete_spaces_disc_unscaled)
    space_unscaled.build_structures(physical_tags_unscaled, b_physical_tags_unscaled)

    return space_unscaled


def method_definition(k_order):
    # lower order convention
    method_1 = {
        "u": ("RT", k_order + 1),
        "p": ("Lagrange", k_order),
        "v": ("RT", k_order + 1),
        "q": ("Lagrange", k_order),
        "l": ("Lagrange", k_order),
    }

    method_2 = {
        "u": ("BDM", k_order + 1),
        "p": ("Lagrange", k_order),
        "v": ("BDM", k_order + 1),
        "q": ("Lagrange", k_order),
        "l": ("Lagrange", k_order),
    }

    # methods = [method_1, method_2]
    methods = [method_1]
    method_names = ["mixed_rt", "mixed_bdm"]
    return zip(method_names, methods)


def two_fields_formulation(method, material, gmesh, case_name, write_vtk_q=True):
    dim = gmesh.dimension

    # tag material interface for Lagrange multipliers


    st = time.time()
    print("Creating unscaled variable fe space.")
    fe_space_unscaled = create_product_spaces(method, gmesh, flux_names={"scaled":"v","unscaled":"u"},potential_names={"scaled":"q","unscaled":"p"})
    et = time.time()
    elapsed_time = et - st
    print("Creation of product space:", elapsed_time, "seconds")


    # Nonlinear solver data
    n_iterations = 2
    eps_tol = 1.0e-10
    n_dof_g = fe_space_unscaled.n_dof
    fe_space_unscaled.dof_shift = 0

    st = time.time()

    # Material data as scalars
    m_mu = 1.0
    m_par = material["m_par"]

    # retrieve material functions
    assert exact_funcs.test_degeneracy(m_par, m_mu, dim)
    f_porosity = partial(exact_funcs.f_porosity, m_par=m_par, dim=dim)
    f_d_phi = partial(exact_funcs.f_d_phi, m_par=m_par, m_mu=m_mu, dim=dim)
    f_grad_d_phi = partial(exact_funcs.f_grad_d_phi, m_par=m_par, m_mu=m_mu, dim=dim)
    f_kappa = partial(exact_funcs.f_kappa, m_par=m_par, dim=dim)

    # retrieve exact functions
    u_exact = partial(exact_funcs.u_exact, m_par=m_par, dim=dim)
    p_exact = partial(exact_funcs.p_exact, m_par=m_par, dim=dim)
    f_rhs = partial(exact_funcs.f_rhs, m_par=m_par, dim=dim)

    m_functions = {
        "rhs": f_rhs,
        "kappa": f_kappa,
        "porosity": f_porosity,
        "d_phi": f_d_phi,
        "grad_d_phi": f_grad_d_phi,
    }

    bc_functions = {
        "u": u_exact,
        "p": p_exact,
        "d_phi": f_d_phi,
        "porosity": f_porosity,
    }

    exact_functions = {
        "u": u_exact,
        "p": p_exact,
    }

    weak_form_unscaled = UnscaledEllipticWeakForm(fe_space_unscaled)
    weak_form_unscaled.functions = m_functions
    weak_form_unscaled.phi_threshold = 1.0e-4

    bc_weak_form_unscaled = UnscaledEllipticWeakFormBCDirichlet(fe_space_unscaled)
    bc_weak_form_unscaled.functions = bc_functions


    def scatter_form_data(jac_g, res_g, i, weak_form):
        # destination indexes
        dest = weak_form.space.destination_indexes(i)
        alpha_l = alpha[dest]

        r_el, j_el = weak_form.evaluate_form(i, alpha_l)

        # contribute rhs
        res_g[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        for k in range(nnz):
            jac_g.setValue(row=row[k], col=col[k], value=data[k], addv=True)

    def scatter_bc_form(jac_g, res_g, i, bc_weak_form, fe_space):
        dest = fe_space.bc_destination_indexes(i)
        alpha_l = alpha[dest]
        r_el, j_el = bc_weak_form.evaluate_form(i, alpha_l)

        # contribute rhs
        res_g[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        for k in range(nnz):
            jac_g.setValue(row=row[k], col=col[k], value=data[k], addv=True)


    # Assembler
    st = time.time()
    jac_g = PETSc.Mat()
    jac_g.createAIJ([n_dof_g, n_dof_g])

    res_g = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    # initial guess
    alpha = np.zeros(n_dof_g)

    idx_unscale = np.arange(len(fe_space_unscaled.discrete_spaces["u"].elements))

    for iter in range(n_iterations):
        [scatter_form_data(jac_g, res_g, i, weak_form_unscaled) for i in idx_unscale]
        n_bc_els = len(fe_space_unscaled.discrete_spaces["u"].bc_elements)
        [scatter_bc_form(jac_g, res_g, i, bc_weak_form_unscaled, fe_space_unscaled) for i in range(n_bc_els)]


        jac_g.assemble()

        et = time.time()
        elapsed_time = et - st
        print("Assembly time:", elapsed_time, "seconds")

        res_norm = np.linalg.norm(res_g)
        stop_criterion_q = res_norm < eps_tol
        if stop_criterion_q:
            print("Nonlinear solver converged")
            print("Residual norm: ", res_norm)
            print("Number of iterations: ", iter)
            break

        # solving ls
        st = time.time()
        ksp = PETSc.KSP().create()
        ksp.setOperators(jac_g)
        b = jac_g.createVecLeft()
        b.array[:] = -res_g
        x = jac_g.createVecRight()

        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.setConvergenceHistory()
        ksp.solve(b, x)
        delta_alpha = x.array

        et = time.time()
        elapsed_time = et - st
        print("Linear solver time:", elapsed_time, "seconds")

        # newton update
        alpha += delta_alpha

        # Set up to zero lhr and rhs
        res_g *= 0.0
        jac_g.scale(0.0)

    st = time.time()
    (
        u_l2_error,
        p_l2_error,
    ) = l2_error(dim, fe_space_unscaled, exact_functions, alpha)

    alpha_proj = l2_projector(fe_space_unscaled, exact_functions)
    alpha_e = alpha - alpha_proj
    p_proj_l2_error = l2_error_projected(dim, fe_space_unscaled, alpha_e, ["u"])[0]

    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    print("L2-error in u: ", u_l2_error)
    print("L2-error in p: ", p_l2_error)
    print("L2-error in p projected: ", p_proj_l2_error)

    if write_vtk_q:
        # post-process solution
        st = time.time()
        file_name = case_name + "two_fields.vtk"
        write_fe_spaces_on_vtk_file_with_exact_solution(
            file_name, gmesh, [fe_space_unscaled], exact_functions, alpha
        )
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return [
        p_l2_error,
        u_l2_error,
        p_proj_l2_error,
    ]


def create_domain(dimension, make_fitted_q):
    if dimension == 1:
        offset = 1.0 / 3.0
        if make_fitted_q:
            offset = 0.0
        points = np.array([[-1, 0, 0], [-offset, 0, 0], [1, 0, 0]])
        domain = build_line_1D(points)
        return domain
    elif dimension == 2:
        offset = 0.5
        if make_fitted_q:
            offset = 0.75 # original
        points = np.array(
            [
                [-1.0, -1.0, 0],
                [1, -1.0, 0],
                [1, -offset, 0],
                [1, 1, 0],
                [-offset, 1, 0],
                [-1.0, 1, 0],
                [-offset, -offset, 0],
            ]
        )
        domain = build_surface_2D(points, skip_e_tags=[])
        return domain
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")


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
        gmesh.write_vtk(file_name="geometry_scenario_2")
    return gmesh


def material_data_definition(dim):
    # Material data for example 1 and 2
    if dim == 1:
        case_0 = {"m_par": +0.5}
        case_1 = {"m_par": -0.5}
        case_2 = {"m_par": -1.0}
        case_3 = {"m_par": -1.5}
    elif dim == 2:
        case_0 = {"m_par": 2.0}
        case_1 = {"m_par": 1.0}
        case_2 = {"m_par": 0.25}
        case_3 = {"m_par": 0.125}
    else:
        raise ValueError("Only 1D and 2D settings are supported by this script.")
    cases = [case_0, case_1, case_2, case_3]
    cases = [case_0]
    return cases


def compose_case_name(method, dimension, domain, material, folder_name=None):
    if folder_name is None:
        case_name = (
            method[0]
            + "_"
            + str(dimension)
            + "d_"
            + domain[0]
            + "_"
            + "material_parameter_"
            + str(material["m_par"])
            + "_"
        )
    else:
        import os

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        case_name = (
            folder_name
            + "/"
            + method[0]
            + "_"
            + str(dimension)
            + "d_"
            + domain[0]
            + "_"
            + "material_parameter_"
            + str(material["m_par"])
            + "_"
        )
    return case_name


def main():
    # fixed directives
    k_order = 0
    h = 0.5
    n_ref = 6
    dimensions = [2]
    folder_name = "output_scenario_1"
    plot_rates_q = True

    # method variants
    methods = method_definition(k_order)
    for method in methods:
        for dimension in dimensions:
            # dimension dependent variants

            if dimension == 1 and method[0] == "mixed_bdm":
                continue

            fitted_domain = create_domain(dimension, make_fitted_q=True)
            unfitted_domain = create_domain(dimension, make_fitted_q=False)
            # domains = {"fitted": fitted_domain, "unfitted": unfitted_domain}
            domains = {"fitted": fitted_domain}
            materials = material_data_definition(dimension)
            for domain in domains.items():
                for material in materials:
                    # composing name prefix
                    case_name = compose_case_name(
                        method, dimension, domain, material, folder_name
                    )

                    n_data = 4
                    error_data = np.empty((0, n_data), float)
                    for l in range(n_ref):
                        h_val = h * (2**-l)
                        case_name_with_level = case_name + "l_" + str(l) + "_"
                        mesher = create_conformal_mesher(domain[1], h_val, 0)
                        gmesh = create_mesh(dimension, mesher, True)
                        h_min, h_mean, h_max = mesh_size(gmesh)
                        error_val = two_fields_formulation(
                            method, material, gmesh, case_name_with_level, True
                        )
                        error_data = np.append(
                            error_data, np.array([[h_max] + error_val]), axis=0
                        )

                    rates_data = np.empty((0, n_data - 1), float)
                    for i in range(error_data.shape[0] - 1):
                        chunk_b = np.log(error_data[i])
                        chunk_e = np.log(error_data[i + 1])
                        h_step = chunk_e[0] - chunk_b[0]
                        partial = (chunk_e - chunk_b) / h_step
                        rates_data = np.append(
                            rates_data, np.array([list(partial[1:n_data])]), axis=0
                        )

                    rates_data = np.vstack(
                        (np.array([np.nan] * rates_data.shape[1]), rates_data)
                    )

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

                    np.set_printoptions(precision=5)
                    print("convergence data: ", raw_data)

                    header = "h, p, rate, u, rate, proj p, rate"
                    np.savetxt(
                        case_name + "conv_data.txt",
                        raw_data,
                        delimiter=",",
                        fmt="%1.6f",
                        header=header,
                    )

                    if plot_rates_q:
                        x = error_data[:, 0]
                        y = error_data[:, 1:n_data]
                        lineObjects = plt.loglog(x, y, marker="o")
                        plt.legend(
                            lineObjects,
                            ("p", "u", "projected p"),
                        )
                        plt.title("")
                        plt.xlabel("Element size")
                        plt.ylabel("L2-error")

                        # Save the figure
                        filename = case_name + "convergence_plot.png"
                        plt.savefig(filename, dpi=300, bbox_inches='tight')
                        # plt.show()
                        plt.close()


if __name__ == "__main__":
    main()
