import functools
import time

import numpy as np
import strong_solution_cosserat_elasticity as lce
from petsc4py import PETSc

from basis.element_data import ElementData
from geometry.domain import Domain
from geometry.domain_market import build_box_1D, build_box_2D, build_box_3D
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh
from postprocess.l2_error_post_processor import l2_error, grad_error, div_error
from postprocess.solution_post_processor import write_vtk_file_with_exact_solution
from spaces.product_space import ProductSpace
from weak_forms.lce_dual_weak_form import LCEDualWeakForm, LCEDualWeakFormBCDirichlet
from weak_forms.lce_primal_weak_form import (
    LCEPrimalWeakForm,
    LCEPrimalWeakFormBCDirichlet,
)

from postprocess.projectors import l2_projector


def h1_cosserat_elasticity(epsilon, method, gmesh, write_vtk_q=False):
    dim = gmesh.dimension

    # FESpace: data
    u_k_order = method[1]["u"][1]
    t_k_order = method[1]["t"][1]

    u_components = 2
    t_components = 1
    if dim == 3:
        u_components = 3
        t_components = 3
    u_family = method[1]["u"][0]
    t_family = method[1]["t"][0]

    discrete_spaces_data = {
        "u": (dim, u_components, u_family, u_k_order, gmesh),
        "t": (dim, t_components, t_family, t_k_order, gmesh),
    }

    u_disc_Q = False
    t_disc_Q = False
    discrete_spaces_disc = {
        "u": u_disc_Q,
        "t": t_disc_Q,
    }

    u_field_bc_physical_tags = [2, 3, 4, 5]
    t_field_bc_physical_tags = [2, 3, 4, 5]
    if dim == 3:
        u_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
        t_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
    discrete_spaces_bc_physical_tags = {
        "u": u_field_bc_physical_tags,
        "t": t_field_bc_physical_tags,
    }

    fe_space = ProductSpace(discrete_spaces_data)
    fe_space.make_subspaces_discontinuous(discrete_spaces_disc)
    fe_space.build_structures(discrete_spaces_bc_physical_tags)

    n_dof_g = fe_space.n_dof
    rg = np.zeros(n_dof_g)
    alpha = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    # Assembler
    st = time.time()
    A = PETSc.Mat()
    A.createAIJ([n_dof_g, n_dof_g])

    # Material data
    m_lambda = 1.0
    m_mu = 1.0
    m_kappa = m_mu
    m_gamma = epsilon

    # exact solution
    u_exact = lce.displacement(m_lambda, m_mu, m_kappa, m_gamma, dim)
    t_exact = lce.rotation(m_lambda, m_mu, m_kappa, m_gamma, dim)
    grad_u_exact = lce.displacement_gradient(m_lambda, m_mu, m_kappa, m_gamma, dim)
    grad_t_exact = lce.rotation_gradient(m_lambda, m_mu, m_kappa, m_gamma, dim)
    s_exact = lce.stress(m_lambda, m_mu, m_kappa, m_gamma, dim)
    m_exact = lce.couple_stress(m_lambda, m_mu, m_kappa, m_gamma, dim)
    f_rhs = lce.rhs(m_lambda, m_mu, m_kappa, m_gamma, dim)

    def f_lambda(x, y, z):
        return m_lambda

    def f_mu(x, y, z):
        return m_mu

    def f_kappa(x, y, z):
        return m_kappa

    def f_gamma(x, y, z):
        return m_gamma

    m_functions = {
        "rhs": f_rhs,
        "lambda": f_lambda,
        "mu": f_mu,
        "kappa": f_kappa,
        "gamma": f_gamma,
    }

    exact_functions = {
        "u": u_exact,
        "t": t_exact,
        "grad_u": grad_u_exact,
        "grad_t": grad_t_exact,
    }

    weak_form = LCEPrimalWeakForm(fe_space)
    weak_form.functions = m_functions
    bc_weak_form = LCEPrimalWeakFormBCDirichlet(fe_space)
    bc_weak_form.functions = exact_functions

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
        dest = fe_space.bc_destination_indexes(i)
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

    n_els = len(fe_space.discrete_spaces["u"].elements)
    [scatter_form_data(A, i, weak_form) for i in range(n_els)]

    n_bc_els = len(fe_space.discrete_spaces["u"].bc_elements)
    [scatter_bc_form(A, i, bc_weak_form) for i in range(n_bc_els)]

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

    ksp = PETSc.KSP().create()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("fcg")
    ksp.setTolerances(rtol=1e-10, atol=1e-10, divtol=500, max_it=2000)
    ksp.setConvergenceHistory()
    ksp.getPC().setType("ilu")
    ksp.solve(b, x)
    alpha = x.array

    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    # Computing stress L2 error
    def compute_s_l2_error(i, fe_space, m_mu, m_lambda, m_kappa, dim):
        l2_error = 0.0
        u_space = fe_space.discrete_spaces["u"]
        t_space = fe_space.discrete_spaces["t"]

        t_components = t_space.n_comp
        u_data: ElementData = u_space.elements[i].data
        t_data: ElementData = t_space.elements[i].data

        weights = u_data.quadrature.weights

        x = u_data.mapping.x
        det_jac = u_data.mapping.det_jac
        inv_jac = u_data.mapping.inv_jac

        u_phi_tab = u_data.basis.phi
        t_phi_tab = t_data.basis.phi
        n_u_phi = u_phi_tab.shape[2]
        n_t_phi = t_phi_tab.shape[2]

        # destination indexes
        dest_u = fe_space.discrete_spaces_destination_indexes(i)["u"]
        dest_t = fe_space.discrete_spaces_destination_indexes(i)["t"]
        alpha_u_l = alpha[dest_u]
        alpha_t_l = alpha[dest_t]

        # for each integration point

        alpha_star_u = np.array(np.split(alpha_u_l, n_u_phi))
        alpha_star_t = np.array(np.split(alpha_t_l, n_t_phi))
        t_h_s = (t_phi_tab[0, :, :, 0] @ alpha_star_t[:, 0:t_components]).T
        for i, omega in enumerate(weights):
            s_e = s_exact(x[i, 0], x[i, 1], x[i, 2])
            grad_phi = inv_jac[i].T @ u_phi_tab[1 : u_phi_tab.shape[0] + 1, i, :, 0]
            grad_uh = grad_phi[0:dim] @ alpha_star_u[:, 0:dim]
            if dim == 2:
                t_h = np.array([[0.0, -t_h_s[0, i]], [t_h_s[0, i], 0.0]])
            else:
                t_h = np.array(
                    [
                        [0.0, -t_h_s[2, i], +t_h_s[1, i]],
                        [+t_h_s[2, i], 0.0, -t_h_s[0, i]],
                        [-t_h_s[1, i], +t_h_s[0, i], 0.0],
                    ]
                )
            eps_h = grad_uh.T + t_h
            symm_eps = 0.5 * (eps_h + eps_h.T)
            skew_eps = 0.5 * (eps_h - eps_h.T)
            s_h = (
                2.0 * m_mu * symm_eps
                + 2.0 * m_kappa * skew_eps
                + m_lambda * eps_h.trace() * np.identity(dim)
            )
            diff_s = s_e - s_h
            l2_error += det_jac[i] * weights[i] * np.trace(diff_s.T @ diff_s)
        return l2_error

    # Computing couple stress L2 error
    def compute_m_l2_error(i, fe_space, m_gamma, dim):
        l2_error = 0.0
        t_space = fe_space.discrete_spaces["t"]

        n_components = t_space.n_comp
        el_data = t_space.elements[i].data

        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # scattering dof
        dest = fe_space.discrete_spaces_destination_indexes(i)["t"]
        alpha_l = alpha[dest]

        # for each integration point
        n_phi = phi_tab.shape[2]
        alpha_star = np.array(np.split(alpha_l, n_phi))
        for i, omega in enumerate(weights):
            m_e = m_exact(x[i, 0], x[i, 1], x[i, 2])
            grad_phi = inv_jac[i].T @ phi_tab[1 : phi_tab.shape[0] + 1, i, :, 0]
            grad_th = grad_phi[0:dim] @ alpha_star[:, 0:n_components]
            m_h = m_gamma * grad_th.T
            diff_m = m_e - m_h
            l2_error += det_jac[i] * weights[i] * np.trace(diff_m.T @ diff_m)
        return l2_error

    st = time.time()
    u_l2_error, t_l2_error = l2_error(dim, fe_space, exact_functions, alpha)
    grad_u_l2_error, grad_t_l2_error = grad_error(dim, fe_space, exact_functions, alpha)
    s_error_vec = [
        compute_s_l2_error(i, fe_space, m_mu, m_lambda, m_kappa, dim)
        for i in range(n_els)
    ]
    m_error_vec = [compute_m_l2_error(i, fe_space, m_gamma, dim) for i in range(n_els)]
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    s_l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, s_error_vec))
    m_l2_error = np.sqrt(functools.reduce(lambda x, y: x + y, m_error_vec))
    print("L2-error displacement: ", u_l2_error)
    print("L2-error rotation: ", u_l2_error)
    print("L2-error stress: ", s_l2_error)
    print("L2-error couple stress: ", m_l2_error)
    print("L2-error grad displacement: ", grad_u_l2_error)
    print("L2-error grad rotation: ", grad_t_l2_error)

    if write_vtk_q:
        st = time.time()
        file_name = method[0] + "_rates_h1_cosserat_elasticity.vtk"
        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, exact_functions, alpha
        )
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return np.array(
        [
            u_l2_error,
            t_l2_error,
            s_l2_error,
            m_l2_error,
            grad_u_l2_error,
            grad_t_l2_error,
        ]
    )


def hdiv_cosserat_elasticity(epsilon, method, gmesh, write_vtk_q=False):
    dim = gmesh.dimension

    # FESpace: data
    s_k_order = method[1]["s"][1]
    m_k_order = method[1]["m"][1]
    u_k_order = method[1]["u"][1]
    t_k_order = method[1]["t"][1]

    s_components = 2
    m_components = 1
    u_components = 2
    t_components = 1
    if dim == 3:
        s_components = 3
        m_components = 3
        u_components = 3
        t_components = 3

    s_family = method[1]["s"][0]
    m_family = method[1]["m"][0]
    u_family = method[1]["u"][0]
    t_family = method[1]["t"][0]

    discrete_spaces_data = {
        "s": (dim, s_components, s_family, s_k_order, gmesh),
        "m": (dim, m_components, m_family, m_k_order, gmesh),
        "u": (dim, u_components, u_family, u_k_order, gmesh),
        "t": (dim, t_components, t_family, t_k_order, gmesh),
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

    s_field_bc_physical_tags = [2, 3, 4, 5]
    m_field_bc_physical_tags = [2, 3, 4, 5]
    if dim == 3:
        s_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
        m_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
    discrete_spaces_bc_physical_tags = {
        "s": s_field_bc_physical_tags,
        "m": m_field_bc_physical_tags,
    }

    fe_space = ProductSpace(discrete_spaces_data)
    fe_space.make_subspaces_discontinuous(discrete_spaces_disc)
    fe_space.build_structures(discrete_spaces_bc_physical_tags)

    n_dof_g = fe_space.n_dof
    rg = np.zeros(n_dof_g)
    alpha = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    # Assembler
    st = time.time()

    A = PETSc.Mat()
    A.createAIJ([n_dof_g, n_dof_g])

    # Material data
    m_lambda = 1.0
    m_mu = 1.0
    m_kappa = m_mu
    m_gamma = epsilon

    # exact solution
    u_exact = lce.displacement(m_lambda, m_mu, m_kappa, m_gamma, dim)
    t_exact = lce.rotation(m_lambda, m_mu, m_kappa, m_gamma, dim)
    s_exact = lce.stress(m_lambda, m_mu, m_kappa, m_gamma, dim)
    m_exact = lce.couple_stress(m_lambda, m_mu, m_kappa, m_gamma, dim)
    div_s_exact = lce.stress_divergence(m_lambda, m_mu, m_kappa, m_gamma, dim)
    div_m_exact = lce.couple_stress_divergence(m_lambda, m_mu, m_kappa, m_gamma, dim)
    f_rhs = lce.rhs(m_lambda, m_mu, m_kappa, m_gamma, dim)

    def f_lambda(x, y, z):
        return m_lambda

    def f_mu(x, y, z):
        return m_mu

    def f_kappa(x, y, z):
        return m_kappa

    def f_gamma(x, y, z):
        return m_gamma

    m_functions = {
        "rhs": f_rhs,
        "lambda": f_lambda,
        "mu": f_mu,
        "kappa": f_kappa,
        "gamma": f_gamma,
    }

    exact_functions = {
        "s": s_exact,
        "m": m_exact,
        "u": u_exact,
        "t": t_exact,
        "div_s": div_s_exact,
        "div_m": div_m_exact,
    }

    weak_form = LCEDualWeakForm(fe_space)
    weak_form.functions = m_functions
    bc_weak_form = LCEDualWeakFormBCDirichlet(fe_space)
    bc_weak_form.functions = exact_functions

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
        dest = fe_space.bc_destination_indexes(i)
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

    # alpha_p = l2_projector(fe_space, exact_functions)
    # alpha = alpha_p

    n_els = len(fe_space.discrete_spaces["s"].elements)
    [scatter_form_data(A, i, weak_form) for i in range(n_els)]

    n_bc_els = len(fe_space.discrete_spaces["s"].bc_elements)
    [scatter_bc_form(A, i, bc_weak_form) for i in range(n_bc_els)]

    A.assemble()

    print("residual norm:", np.linalg.norm(rg))

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

    ksp = PETSc.KSP().create()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("fgmres")
    ksp.setTolerances(rtol=1e-12, atol=1e-12, divtol=500, max_it=2000)
    ksp.setConvergenceHistory()
    ksp.getPC().setType("ilu")
    ksp.solve(b, x)
    alpha = x.array

    # viewer = PETSc.Viewer().createASCII("ksp_output.txt")
    # ksp.view(viewer)
    # solver_output = open("ksp_output.txt", "r")
    # for line in solver_output.readlines():
    #     print(line)
    #
    # residuals = ksp.getConvergenceHistory()
    # plt.semilogy(residuals)

    # alpha_p = l2_projector(fe_space, exact_functions)
    # alpha = alpha_p

    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    st = time.time()
    s_l2_error, m_l2_error, u_l2_error, t_l2_error = l2_error(
        dim, fe_space, exact_functions, alpha
    )
    div_s_l2_error, div_m_l2_error = div_error(dim, fe_space, exact_functions, alpha)
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    print("L2-error displacement: ", u_l2_error)
    print("L2-error rotation: ", t_l2_error)
    print("L2-error stress: ", s_l2_error)
    print("L2-error couple stress: ", m_l2_error)
    print("L2-error div stress: ", div_s_l2_error)
    print("L2-error div couple stress: ", div_m_l2_error)

    if write_vtk_q:
        st = time.time()
        file_name = "rates_hdiv_cosserat_elasticity.vtk"
        write_vtk_file_with_exact_solution(
            file_name, gmesh, fe_space, exact_functions, alpha
        )
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return np.array(
        [u_l2_error, t_l2_error, s_l2_error, m_l2_error, div_s_l2_error, div_m_l2_error]
    )


def create_domain(dimension):
    if dimension == 1:
        box_points = np.array([[0, 0, 0], [1, 0, 0]])
        domain = build_box_1D(box_points)
        return domain
    elif dimension == 2:
        box_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        # box_points = [
        #     point + 0.25 * np.array([-1.0, -1.0, 0.0]) for point in box_points
        # ]
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
        # box_points = [
        #     point + 0.25 * np.array([-1.0, -1.0, -1.0]) for point in box_points
        # ]
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

    # npts = np.array([pt + 0.25 * np.array([-1.0, -1.0, -1.0]) for pt in gmesh.points])
    # gmesh.points = npts

    if write_vtk_q:
        gmesh.write_vtk()
    return gmesh


def perform_convergence_test(configuration: dict):
    # retrieve parameters from dictionary
    k_order = configuration.get("k_order")
    method = configuration.get("method")
    n_ref = configuration.get("n_refinements")
    dimension = configuration.get("dimension")
    dual_form_q = configuration.get("dual_problem_Q", False)
    epsilon_value = configuration.get("epsilon_value", 1.0)
    write_geometry_vtk = configuration.get("write_geometry_Q", False)
    write_vtk = configuration.get("write_vtk_Q", False)
    report_full_precision_data = configuration.get(
        "report_full_precision_data_Q", False
    )

    # The initial element size
    h = 1.0

    # Create a unit squared or a unit cube
    domain = create_domain(dimension)

    n_data = 7
    error_data = np.empty((0, n_data), float)
    for lh in range(n_ref):
        h_val = h * (2**-lh)
        mesher = create_conformal_mesher(domain, h, lh)
        gmesh = create_mesh(dimension, mesher, write_geometry_vtk)
        if dual_form_q:
            error_vals = hdiv_cosserat_elasticity(
                epsilon_value, method, gmesh, write_vtk
            )
        else:
            error_vals = h1_cosserat_elasticity(epsilon_value, method, gmesh, write_vtk)
        chunk = np.concatenate([[h_val], error_vals])
        error_data = np.append(error_data, np.array([chunk]), axis=0)

    rates_data = np.empty((0, n_data - 1), float)
    for i in range(error_data.shape[0] - 1):
        chunk_b = np.log(error_data[i])
        chunk_e = np.log(error_data[i + 1])
        h_step = chunk_e[0] - chunk_b[0]
        partial = (chunk_e - chunk_b) / h_step
        rates_data = np.append(rates_data, np.array([list(partial[1:n_data])]), axis=0)

    # minimal report
    if report_full_precision_data:
        print("error data: ", error_data)
        print("error rates data: ", rates_data)

    np.set_printoptions(precision=3)
    if dual_form_q:
        print("Dual problem")
    else:
        print("Primal problem")

    print("Polynomial order: ", k_order)
    print("Dimension: ", dimension)
    print("rounded error data: ", error_data)
    print("rounded error rates data: ", rates_data)
    print(" ")

    primal_header = "h, u, r, sigma, omega, grad_u, grad_r"
    dual_header = "h, u, r, sigma, omega, div_sigma, div_omega"
    str_header = primal_header
    if dual_form_q:
        str_header = dual_header

    if report_full_precision_data:
        np.savetxt(
            method[0] + "_k" + str(k_order) + "_" + str(dimension) + "d_error_data.txt",
            error_data,
            delimiter=",",
            header=str_header,
        )
        np.savetxt(
            method[0]
            + "_k"
            + str(k_order)
            + "_"
            + str(dimension)
            + "d_expected_order_convergence.txt",
            rates_data,
            delimiter=",",
            header=str_header,
        )
    np.savetxt(
        method[0]
        + "_k"
        + str(k_order)
        + "_"
        + str(dimension)
        + "d_error_data_rounded.txt",
        error_data,
        fmt="%1.3e",
        delimiter=",",
        header=str_header,
    )
    np.savetxt(
        method[0]
        + "_k"
        + str(k_order)
        + "_"
        + str(dimension)
        + "d_expected_order_convergence_rounded.txt",
        rates_data,
        fmt="%1.3f",
        delimiter=",",
        header=str_header,
    )

    return


def method_definition(k_order):
    method_1_pc = {"u": ("Lagrange", k_order + 1), "t": ("Lagrange", k_order)}
    method_2_pnc = {"u": ("Lagrange", k_order), "t": ("Lagrange", k_order)}
    method_3_dc = {
        "s": ("RT", k_order),
        "m": ("RT", k_order + 1),
        "u": ("Lagrange", k_order - 1),
        "t": ("Lagrange", k_order),
    }
    method_4_dc = {
        "s": ("BDM", k_order),
        "m": ("BDM", k_order + 1),
        "u": ("Lagrange", k_order - 1),
        "t": ("Lagrange", k_order),
    }
    method_5_dnc = {
        "s": ("BDM", k_order),
        "m": ("RT", k_order),
        "u": ("Lagrange", k_order - 1),
        "t": ("Lagrange", k_order - 1),
    }

    methods = [method_1_pc, method_2_pnc, method_3_dc, method_4_dc, method_5_dnc]
    method_names = ["m1_pc", "m2_pnc", "m3_dc", "m4_dc", "m5_dnc"]
    return zip(method_names, methods)


def main():
    epsilon_value = 1.0
    write_vtk_files_Q = True
    report_full_precision_data_Q = False

    for k in [1]:
        methods = method_definition(k)
        for i, method in enumerate(methods):
            dual_problem_q = False
            if i in [2, 3, 4]:
                dual_problem_q = True
            if i in [0, 1]:
                continue
            configuration = {
                "n_refinements": 4,
                "dual_problem_Q": dual_problem_q,
                "write_geometry_Q": write_vtk_files_Q,
                "write_vtk_Q": write_vtk_files_Q,
                "method": method,
                "epsilon_value": epsilon_value,
                "report_full_precision_data_Q": report_full_precision_data_Q,
            }

            for d in [2]:
                configuration.__setitem__("k_order", k)
                configuration.__setitem__("dimension", d)
                perform_convergence_test(configuration)


if __name__ == "__main__":
    main()
