import copy
import csv
import functools
import marshal
import sys
import time
# from itertools import permutations
from functools import partial, reduce

import auto_diff as ad
import basix
import matplotlib.colors as mcolors
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plot
import meshio
import networkx as nx
import numpy as np
import psutil
import scipy.sparse as sp
# from numba import njit, types
import strong_solution_cosserat_elasticity as lce
from auto_diff.vecvalder import VecValDer
from numpy import linalg as la
from petsc4py import PETSc
from scipy.sparse import coo_matrix
from shapely.geometry import LineString

import geometry.fracture_network as fn
from basis.element_data import ElementData
from basis.finite_element import FiniteElement
from geometry.domain import Domain
from geometry.domain_market import (build_box_1D, build_box_2D,
                                    build_box_2D_with_lines, build_box_3D,
                                    build_box_3D_with_planes,
                                    build_disjoint_lines, read_fractures_file)
from geometry.edge import Edge
from geometry.geometry_builder import GeometryBuilder
from geometry.geometry_cell import GeometryCell
from geometry.mapping import (evaluate_linear_shapes, evaluate_mapping,
                              store_mapping)
from geometry.shape_manipulation import ShapeManipulation
from geometry.vertex import Vertex
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh
from spaces.discrete_space import DiscreteSpace
from spaces.dof_map import DoFMap
from spaces.product_space import ProductSpace
from topology.mesh_topology import MeshTopology
from weak_forms.lce_primal_weak_form import (LCEPrimalWeakForm,
                                             LCEPrimalWeakFormBCDirichlet)
from weak_forms.lce_dual_weak_form import LCEDualWeakForm

from postprocess.l2_error_post_processor import l2_error
from postprocess.solution_post_processor import write_vtk_file_with_exact_solution
num_cpus = psutil.cpu_count(logical=False)


def h1_cosserat_elasticity(k_order, gmesh, write_vtk_q=False):
    dim = gmesh.dimension

    # FESpace: data
    u_k_order = k_order + 1
    t_k_order = k_order

    u_components = 2
    t_components = 1
    if dim == 3:
        u_components = 3
        t_components = 3
    family = "Lagrange"

    discrete_spaces_data = {
        "u": (dim, u_components, family, u_k_order, gmesh),
        "t": (dim, t_components, family, t_k_order, gmesh),
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
    m_kappa = 1.0
    m_gamma = 1.0

    # exact solution
    u_exact = lce.displacement(m_lambda, m_mu, m_kappa, m_gamma, dim)
    t_exact = lce.rotation(m_lambda, m_mu, m_kappa, m_gamma, dim)
    s_exact = lce.stress(m_lambda, m_mu, m_kappa, m_gamma, dim)
    m_exact = lce.couple_stress(m_lambda, m_mu, m_kappa, m_gamma, dim)
    f_rhs = lce.rhs(m_lambda, m_mu, m_kappa, m_gamma, dim)

    f_lambda = lambda x, y, z: m_lambda
    f_mu = lambda x, y, z: m_mu
    f_kappa = lambda x, y, z: m_kappa
    f_gamma = lambda x, y, z: m_gamma

    m_functions = {
        "rhs": f_rhs,
        "lambda": f_lambda,
        "mu": f_mu,
        "kappa": f_kappa,
        "gamma": f_gamma,
    }

    exact_functions = {
        'u': u_exact,
        't': t_exact,
    }

    weak_form = LCEPrimalWeakForm(fe_space)
    weak_form.functions = m_functions
    bc_weak_form = LCEPrimalWeakFormBCDirichlet(fe_space)
    bc_weak_form.functions = exact_functions

    def scatter_form_data(A, i, weak_form):
        # destination indexes
        dest = weak_form.space.destination_indexes(i)
        alpha_l = alpha[dest]
        r_el, j_el = weak_form.evaluate_form(i, alpha)

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
        r_el, j_el = bc_weak_form.evaluate_form(i, alpha)

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

    petsc_options = {"rtol": 1e-12, "atol": 1e-14}
    ksp = PETSc.KSP().create()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("fgmres")
    ksp.setTolerances(**petsc_options)
    ksp.setConvergenceHistory()
    ksp.getPC().setType("ilu")
    ksp.solve(b, x)
    alpha = x.array

    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    u_l2_error, t_l2_error = l2_error(dim, fe_space, exact_functions, alpha)

    # Computing stress L2 error
    def compute_s_l2_error(i, fe_space, m_mu, m_lambda, m_kappa, dim):
        l2_error = 0.0
        u_space = fe_space.discrete_spaces["u"]
        t_space = fe_space.discrete_spaces["t"]

        u_components = u_space.n_comp
        t_components = t_space.n_comp
        u_data: ElementData = u_space.elements[i].data
        t_data: ElementData = t_space.elements[i].data

        cell = u_data.cell
        points = u_data.quadrature.points
        weights = u_data.quadrature.weights
        phi_tab = u_data.basis.phi

        x = u_data.mapping.x
        det_jac = u_data.mapping.det_jac
        inv_jac = u_data.mapping.inv_jac

        u_phi_tab = u_data.basis.phi
        t_phi_tab = t_data.basis.phi
        n_u_phi = u_phi_tab.shape[2]
        n_t_phi = t_phi_tab.shape[2]

        n_u_dof = n_u_phi * u_components
        n_t_dof = n_t_phi * t_components

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
        cell = el_data.cell
        points = el_data.quadrature.points
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

    if write_vtk_q:
        st = time.time()
        file_name = "rates_h1_cosserat_elasticity.vtk"
        write_vtk_file_with_exact_solution(file_name, gmesh,fe_space,exact_functions,alpha)
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return np.array([u_l2_error, t_l2_error, s_l2_error, m_l2_error])


def hdiv_cosserat_elasticity(k_order, gmesh, write_vtk_q=False):

    dim = gmesh.dimension

    # FESpace: data
    s_k_order = k_order
    m_k_order = k_order
    u_k_order = s_k_order - 1
    t_k_order = m_k_order - 1

    s_components = 2
    m_components = 1
    u_components = 2
    t_components = 1
    if dim == 3:
        s_components = 3
        m_components = 3
        u_components = 3
        t_components = 3

    s_family = "BDM"
    m_family = "RT"
    u_family = "Lagrange"
    t_family = "Lagrange"

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
    m_kappa = 1.0
    Lc = 1.0
    m_gamma = m_mu * Lc * Lc

    # exact solution
    u_exact = lce.displacement(m_lambda, m_mu, m_kappa, m_gamma, dim)
    t_exact = lce.rotation(m_lambda, m_mu, m_kappa, m_gamma, dim)
    s_exact = lce.stress(m_lambda, m_mu, m_kappa, m_gamma, dim)
    m_exact = lce.couple_stress(m_lambda, m_mu, m_kappa, m_gamma, dim)
    f_rhs = lce.rhs(m_lambda, m_mu, m_kappa, m_gamma, dim)

    f_lambda = lambda x, y, z: m_lambda
    f_mu = lambda x, y, z: m_mu
    f_kappa = lambda x, y, z: m_kappa
    f_gamma = lambda x, y, z: m_gamma

    m_functions = {
        "rhs": f_rhs,
        "lambda": f_lambda,
        "mu": f_mu,
        "kappa": f_kappa,
        "gamma": f_gamma,
    }

    exact_functions = {
        's': s_exact,
        'm': m_exact,
        'u': u_exact,
        't': t_exact,
    }

    weak_form = LCEDualWeakForm(fe_space)
    weak_form.functions = m_functions
    # bc_weak_form = LCEDualWeakFormBCDirichlet(fe_space)
    # bc_weak_form.functions = exact_functions

    def scatter_form_data(A, i, weak_form):
        # destination indexes
        dest = weak_form.space.destination_indexes(i)
        alpha_l = alpha[dest]
        r_el, j_el = weak_form.evaluate_form(i, alpha)

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
        r_el, j_el = bc_weak_form.evaluate_form(i, alpha)

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        data = j_el.ravel()
        row = np.repeat(dest, len(dest))
        col = np.tile(dest, len(dest))
        nnz = data.shape[0]
        for k in range(nnz):
            A.setValue(row=row[k], col=col[k], value=data[k], addv=True)

    n_els = len(fe_space.discrete_spaces["s"].elements)
    [scatter_form_data(A, i, weak_form) for i in range(n_els)]

    # n_bc_els = len(fe_space.discrete_spaces["s"].bc_elements)
    # [scatter_bc_form(A, i, bc_weak_form) for i in range(n_bc_els)]

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

    petsc_options = {"rtol": 1e-10, "atol": 1e-12, "divtol": 200, "max_it": 500}
    ksp = PETSc.KSP().create()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("fgmres")
    # ksp.setTolerances(**petsc_options)
    # ksp.setTolerances(1e-10)
    ksp.setTolerances(rtol=1e-10, atol=1e-10, divtol=500, max_it=2000)
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

    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    st = time.time()
    s_l2_error, m_l2_error, u_l2_error, t_l2_error = l2_error(dim, fe_space,
                                                              exact_functions, alpha)
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    print("L2-error displacement: ", u_l2_error)
    print("L2-error rotation: ", t_l2_error)
    print("L2-error stress: ", s_l2_error)
    print("L2-error couple stress: ", m_l2_error)

    if write_vtk_q:

        st = time.time()
        file_name = "rates_hdiv_cosserat_elasticity.vtk"
        write_vtk_file_with_exact_solution(file_name, gmesh,fe_space,exact_functions,alpha)
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")


    return np.array([u_l2_error, t_l2_error, s_l2_error, m_l2_error])


def create_domain(dimension):
    if dimension == 1:
        box_points = np.array([[0, 0, 0], [1, 0, 0]])
        domain = build_box_1D(box_points)
        return domain
    elif dimension == 2:
        box_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
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


def perform_convergence_test(configuration: dict):
    # retrieve parameters from dictionary
    k_order = configuration.get("k_order")
    n_ref = configuration.get("n_refinements")
    dimension = configuration.get("dimension")
    mixed_form_q = configuration.get("dual_problem_Q", False)
    write_geometry_vtk = configuration.get("write_geometry_Q", False)
    write_vtk = configuration.get("write_vtk_Q", False)
    report_full_precision_data = configuration.get(
        "report_full_precision_data_Q", False
    )

    # The initial element size
    h = 1.0

    # Create a unit squared or a unit cube
    domain = create_domain(dimension)

    n_data = 5
    error_data = np.empty((0, n_data), float)
    for l in range(n_ref):
        h_val = h * (2**-l)
        mesher = create_conformal_mesher(domain, h, l)
        gmesh = create_mesh(dimension, mesher, write_geometry_vtk)
        if mixed_form_q:
            error_vals = hdiv_cosserat_elasticity(k_order, gmesh, write_vtk)
        else:
            error_vals = h1_cosserat_elasticity(k_order, gmesh, write_vtk)
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
    if mixed_form_q:
        print("Dual problem")
    else:
        print("Primal problem")
    print("Polynomial order: ", k_order)
    print("Dimension: ", dimension)
    print("rounded error data: ", error_data)
    print("rounded error rates data: ", rates_data)
    print(" ")
    if mixed_form_q:
        if report_full_precision_data:
            np.savetxt(
                "dual_problem_k"
                + str(k_order)
                + "_"
                + str(dimension)
                + "d_l2_error_data.txt",
                error_data,
                delimiter=",",
                header="element size, displacement, rotation, stress, couple stress",
            )
            np.savetxt(
                "dual_problem_k"
                + str(k_order)
                + "_"
                + str(dimension)
                + "d_l2_expected_order_convergence.txt",
                rates_data,
                delimiter=",",
                header="displacement, rotation, stress, couple stress",
            )
        np.savetxt(
            "dual_problem_k"
            + str(k_order)
            + "_"
            + str(dimension)
            + "d_l2_error_data_rounded.txt",
            error_data,
            fmt="%1.3e",
            delimiter=",",
            header="element size, displacement, rotation, stress, couple stress",
        )
        np.savetxt(
            "dual_problem_k"
            + str(k_order)
            + "_"
            + str(dimension)
            + "d_l2_expected_order_convergence_rounded.txt",
            rates_data,
            fmt="%1.3f",
            delimiter=",",
            header="displacement, rotation, stress, couple stress",
        )

    else:
        if report_full_precision_data:
            np.savetxt(
                "primal_problem_k"
                + str(k_order)
                + "_"
                + str(dimension)
                + "d_l2_error_data.txt",
                error_data,
                delimiter=",",
                header="element size, displacement, rotation, stress, couple stress",
            )
            np.savetxt(
                "primal_problem_k"
                + str(k_order)
                + "_"
                + str(dimension)
                + "d_l2_expected_order_convergence.txt",
                rates_data,
                delimiter=",",
                header="displacement, rotation, stress, couple stress",
            )
        np.savetxt(
            "primal_problem_k"
            + str(k_order)
            + "_"
            + str(dimension)
            + "d_l2_error_data_rounded.txt",
            error_data,
            fmt="%1.3e",
            delimiter=",",
            header="element size, displacement, rotation, stress, couple stress",
        )
        np.savetxt(
            "primal_problem_k"
            + str(k_order)
            + "_"
            + str(dimension)
            + "d_l2_expected_order_convergence_rounded.txt",
            rates_data,
            fmt="%1.3f",
            delimiter=",",
            header="displacement, rotation, stress, couple stress",
        )

    return


def main():

    write_vtk_files_Q = True
    report_full_precision_data_Q = False

    primal_configuration = {
        "n_refinements": 1,
        "write_geometry_Q": write_vtk_files_Q,
        "write_vtk_Q": write_vtk_files_Q,
        "report_full_precision_data_Q": report_full_precision_data_Q,
    }

    # primal problem
    for k in [1]:
        for d in [2]:
            primal_configuration.__setitem__("k_order", k)
            primal_configuration.__setitem__("dimension", d)
            # perform_convergence_test(primal_configuration)

    dual_configuration = {
        "n_refinements": 3,
        "dual_problem_Q": True,
        "write_geometry_Q": write_vtk_files_Q,
        "write_vtk_Q": write_vtk_files_Q,
        "report_full_precision_data_Q": report_full_precision_data_Q,
    }

    # dual problem
    for k in [2]:
        for d in [2]:
            dual_configuration.__setitem__("k_order", k)
            dual_configuration.__setitem__("dimension", d)
            perform_convergence_test(dual_configuration)


if __name__ == "__main__":
    main()
