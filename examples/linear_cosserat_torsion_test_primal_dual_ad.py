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
from postprocess.l2_error_post_processor import l2_error
from postprocess.solution_post_processor import write_vtk_file
from spaces.product_space import ProductSpace
from weak_forms.lce_dual_weak_form import LCEDualWeakForm, LCEDualWeakFormBCDirichlet, LCEDualWeakFormBCNeumann
from weak_forms.lce_primal_weak_form import (
    LCEPrimalWeakForm,
    LCEPrimalWeakFormBCDirichlet,
)

import scipy as sp

def torsion_h1_cosserat_elasticity(L_c, k_order, gmesh, write_vtk_q=False):

    dim = gmesh.dimension

    # FESpace: data
    u_k_order = k_order + 1
    t_k_order = k_order

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
    m_lambda = 0.5769
    m_mu = 0.3846
    m_kappa = m_mu
    m_gamma = m_mu * L_c * L_c

    def f_rhs(x, y, z):
        return np.array([0.0 * x, 0.0 * y, 0.0 * z, 0.0 * x, 0.0 * y, 0.0 * z])

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

    weak_form = LCEPrimalWeakForm(fe_space)
    weak_form.functions = m_functions

    # BC functions
    def u_null(x, y, z):
        return np.array([0.0 * x, 0.0 * y, 0.0 * z])
    def t_null(x, y, z):
        return np.array([0.0 * x, 0.0 * y, 0.0 * z])

    def u_exact_rotation(x, y, z):
        angle = 13.0 * np.pi / 180.0
        xc = np.array([x, y, z])
        R_mat = np.array(
            [
                [np.cos(angle), np.sin(angle), 0.0],
                [-np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        u_D = R_mat @ xc - xc
        return z * u_D / 10.0

    def t_exact(x, y, z):
        angle = 13.0 * np.pi / 180.0
        return angle * np.array([-y * z/10.0, x * z/10.0, z/10.0])

    bot_bc_functions = {
        "u": u_null,
        "t": t_null,
    }

    top_bc_functions = {
        "u": u_exact_rotation,
        "t": t_exact,
    }

    bc_weak_form_top = LCEPrimalWeakFormBCDirichlet(fe_space)
    bc_weak_form_top.functions = top_bc_functions

    bc_weak_form_bot = LCEPrimalWeakFormBCDirichlet(fe_space)
    bc_weak_form_bot.functions = bot_bc_functions

    def scatter_form(A, i, weak_form):
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
    [scatter_form(A, i, weak_form) for i in range(n_els)]

    # filter dirichlet faces
    bc_elements = fe_space.discrete_spaces["u"].bc_elements
    top_bc_els = [i for i, element in enumerate(bc_elements) if
                  element.data.cell.material_id == 7]
    bot_bc_els = [i for i, element in enumerate(bc_elements) if
                  element.data.cell.material_id == 6]

    # filter neumann faces
    lat_bc_els = [i for i, element in enumerate(bc_elements) if
                  element.data.cell.material_id in [2, 3, 4, 5]]

    [scatter_bc_form(A, i, bc_weak_form_top) for i in top_bc_els]
    [scatter_bc_form(A, i, bc_weak_form_top) for i in bot_bc_els]
    [scatter_bc_form(A, i, bc_weak_form_top) for i in lat_bc_els]

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
    ksp.setType("fcg")
    ksp.setTolerances(**petsc_options)
    ksp.setConvergenceHistory()
    ksp.getPC().setType("ilu")
    ksp.solve(b, x)
    alpha = x.array

    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")


    if write_vtk_q:
        st = time.time()
        file_name = "torsion_h1_cosserat_elasticity.vtk"
        write_vtk_file(file_name, gmesh, fe_space, alpha)
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return 0.0, 0.0


def torsion_hdiv_cosserat_elasticity(L_c, k_order, gmesh, write_vtk_q=False):
    dim = gmesh.dimension

    # FESpace: data
    s_k_order = k_order
    m_k_order = k_order
    u_k_order = s_k_order - 1
    t_k_order = m_k_order - 1

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
    m_lambda = 0.5769
    m_mu = 0.3846
    m_kappa = m_mu
    m_gamma = m_mu * L_c * L_c

    def f_rhs(x, y, z):
        return np.array([0.0 * x, 0.0 * y, 0.0 * z, 0.0 * x, 0.0 * y, 0.0 * z])

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

    weak_form = LCEDualWeakForm(fe_space)
    weak_form.functions = m_functions

    # BC functions
    def u_null(x, y, z):
        return np.array([0.0 * x, 0.0 * y, 0.0 * z])

    def t_null(x, y, z):
        return np.array([0.0 * x, 0.0 * y, 0.0 * z])

    def u_exact_rotation(x, y, z):
        angle = 13.0 * np.pi / 180.0
        xc = np.array([x, y, z])
        R_mat = np.array(
            [
                [np.cos(angle), np.sin(angle), 0.0],
                [-np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        u_D = R_mat @ xc - xc
        return z * u_D / 10.0

    def t_exact(x, y, z):
        angle = 13.0 * np.pi / 180.0
        return angle * np.array([-y * z/10.0, x * z/10.0, z/10.0])

    bot_bc_functions = {
        "u": u_null,
        "t": t_null,
    }

    top_bc_functions = {
        "u": u_exact_rotation,
        "t": t_exact,
    }

    def Sn_null(x, y, z):
        return np.array([0.0 * x, 0.0 * y, 0.0 * z])

    def Mn_null(x, y, z):
        return np.array([0.0 * x, 0.0 * y, 0.0 * z])

    lat_bc_functions = {
        "s": Sn_null,
        "m": Mn_null,
    }

    bc_weak_form_top = LCEDualWeakFormBCDirichlet(fe_space)
    bc_weak_form_top.functions = top_bc_functions

    bc_weak_form_bot = LCEDualWeakFormBCDirichlet(fe_space)
    bc_weak_form_bot.functions = bot_bc_functions

    bc_weak_form_lat = LCEDualWeakFormBCNeumann(fe_space)
    bc_weak_form_lat.functions = lat_bc_functions

    def scatter_form(A, i, weak_form):
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

    n_els = len(fe_space.discrete_spaces["s"].elements)
    [scatter_form(A, i, weak_form) for i in range(n_els)]

    # filter dirichlet faces
    bc_elements = fe_space.discrete_spaces["s"].bc_elements
    top_bc_els = [i for i, element in enumerate(bc_elements) if
                  element.data.cell.material_id == 7]
    bot_bc_els = [i for i, element in enumerate(bc_elements) if
                  element.data.cell.material_id == 6]

    # filter neumann faces
    lat_bc_els = [i for i, element in enumerate(bc_elements) if
                  element.data.cell.material_id in [2, 3, 4, 5]]

    [scatter_bc_form(A, i, bc_weak_form_top) for i in top_bc_els]
    [scatter_bc_form(A, i, bc_weak_form_top) for i in bot_bc_els]
    [scatter_bc_form(A, i, bc_weak_form_top) for i in lat_bc_els]

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
    ksp.setType("fgmres")
    ksp.setTolerances(rtol=1e-10, atol=1e-10, divtol=500, max_it=2000)
    ksp.setConvergenceHistory()
    ksp.getPC().setType("ilu")
    ksp.solve(b, x)
    alpha = x.array

    ai, aj, av = A.getValuesCSR()
    Asp = sp.sparse.csr_matrix((av, aj, ai))
    alpha = sp.sparse.linalg.spsolve(Asp, -rg)


    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    # def integrate_M_t_strain(i, s_space):
    #     n_components = s_space.n_comp
    #     el_data: ElementData = s_space.bc_elements[i].data
    #
    #     cell = el_data.cell
    #     points = el_data.quadrature.points
    #     weights = el_data.quadrature.weights
    #     phi_tab = el_data.basis.phi
    #
    #     x = el_data.mapping.x
    #     det_jac = el_data.mapping.det_jac
    #     inv_jac = el_data.mapping.inv_jac
    #
    #     # find high-dimension neigh
    #     entity_map = s_space.dof_map.mesh_topology.entity_map_by_dimension(
    #         cell.dimension
    #     )
    #     neigh_list = list(entity_map.predecessors(cell.id))
    #     neigh_check_q = len(neigh_list) > 0
    #     assert neigh_check_q
    #
    #     neigh_cell_id = neigh_list[0]
    #     neigh_cell_index = s_space.id_to_element[neigh_cell_id]
    #     neigh_element = s_space.elements[neigh_cell_index]
    #     neigh_cell = neigh_element.data.cell
    #
    #     # destination indexes
    #     dest_neigh = s_space.dof_map.destination_indices(neigh_cell_id)
    #     dest = s_space.dof_map.bc_destination_indices(neigh_cell_id, cell.id)
    #
    #     n_phi = phi_tab.shape[2]
    #     n_dof = n_phi * n_components
    #     js = (n_dof, n_dof)
    #     j_el = np.zeros(js)
    #     r_el = np.zeros(n_dof)
    #
    #     # compute trace space
    #     facet_index = neigh_cell.sub_cells_ids[2].tolist().index(cell.id)
    #     vertices = basix.geometry(CellType.tetrahedron)
    #     facet = basix.cell.sub_entity_connectivity(CellType.tetrahedron)[
    #         cell.dimension
    #     ][facet_index][0]
    #     mapped_points = np.array(
    #         [
    #             vertices[facet[0]] * (1 - x - y)
    #             + vertices[facet[1]] * x
    #             + vertices[facet[2]] * y
    #             for x, y in points
    #         ]
    #     )
    #     dof_n_index = neigh_element.data.dof.entity_dofs[cell.dimension][facet_index]
    #     sn_phi_tab = neigh_element.evaluate_basis(mapped_points, False)
    #     n_phi = sn_phi_tab.shape[2]
    #     M_t = 0.0
    #     if cell.material_id in [7]:
    #         alpha_l = alpha[dest_neigh]
    #         alpha_star = np.array(np.split(alpha_l, n_phi))
    #         for i, omega in enumerate(weights):
    #             xv = x[i, 0], x[i, 1], x[i, 2]
    #             s_h = np.vstack(
    #                 tuple(
    #                     [
    #                         sn_phi_tab[0, i, :, 0:dim].T @ alpha_star[:, d]
    #                         for d in range(dim)
    #                     ]
    #                 )
    #             )
    #             M_t += det_jac[i] * omega * (xv[0] * s_h[2, 1] - xv[1] * s_h[2, 0])
    #
    #     return M_t
    #
    # def integrate_M_t_curvature(i, m_space):
    #     n_components = m_space.n_comp
    #     el_data: ElementData = m_space.bc_elements[i].data
    #
    #     cell = el_data.cell
    #     points = el_data.quadrature.points
    #     weights = el_data.quadrature.weights
    #     phi_tab = el_data.basis.phi
    #
    #     x = el_data.mapping.x
    #     det_jac = el_data.mapping.det_jac
    #     inv_jac = el_data.mapping.inv_jac
    #
    #     # find high-dimension neigh
    #     entity_map = m_space.dof_map.mesh_topology.entity_map_by_dimension(
    #         cell.dimension
    #     )
    #     neigh_list = list(entity_map.predecessors(cell.id))
    #     neigh_check_q = len(neigh_list) > 0
    #     assert neigh_check_q
    #
    #     neigh_cell_id = neigh_list[0]
    #     neigh_cell_index = m_space.id_to_element[neigh_cell_id]
    #     neigh_element = m_space.elements[neigh_cell_index]
    #     neigh_cell = neigh_element.data.cell
    #
    #     # destination indexes
    #     dest_neigh = m_space.dof_map.destination_indices(neigh_cell_id) + s_n_dof_g
    #     dest = (
    #         m_space.dof_map.bc_destination_indices(neigh_cell_id, cell.id) + s_n_dof_g
    #     )
    #
    #     n_phi = phi_tab.shape[2]
    #     n_dof = n_phi * n_components
    #     js = (n_dof, n_dof)
    #     j_el = np.zeros(js)
    #     r_el = np.zeros(n_dof)
    #
    #     # compute trace space
    #     facet_index = neigh_cell.sub_cells_ids[2].tolist().index(cell.id)
    #     vertices = basix.geometry(CellType.tetrahedron)
    #     facet = basix.cell.sub_entity_connectivity(CellType.tetrahedron)[
    #         cell.dimension
    #     ][facet_index][0]
    #     mapped_points = np.array(
    #         [
    #             vertices[facet[0]] * (1 - x - y)
    #             + vertices[facet[1]] * x
    #             + vertices[facet[2]] * y
    #             for x, y in points
    #         ]
    #     )
    #     dof_n_index = neigh_element.data.dof.entity_dofs[cell.dimension][facet_index]
    #     mn_phi_tab = neigh_element.evaluate_basis(mapped_points, False)
    #     n_phi = mn_phi_tab.shape[2]
    #     M_t = 0.0
    #     if cell.material_id in [7]:
    #         alpha_l = alpha[dest_neigh]
    #         alpha_star = np.array(np.split(alpha_l, n_phi))
    #         for i, omega in enumerate(weights):
    #             xv = x[i, 0], x[i, 1], x[i, 2]
    #             m_h = np.vstack(
    #                 tuple(
    #                     [
    #                         mn_phi_tab[0, i, :, 0:dim].T @ alpha_star[:, d]
    #                         for d in range(dim)
    #                     ]
    #                 )
    #             )
    #             M_t += det_jac[i] * omega * (m_h[2, 2])
    #
    #     return M_t
    #
    # st = time.time()
    # M_t_strain_vec = [
    #     integrate_M_t_strain(i, s_space) for i in range(len(m_space.bc_elements))
    # ]
    # M_t_curvature_vec = [
    #     integrate_M_t_curvature(i, m_space) for i in range(len(m_space.bc_elements))
    # ]
    # et = time.time()
    # elapsed_time = et - st
    # print("Integrate M_t time:", elapsed_time, "seconds")
    M_t_strain = 0.0 #functools.reduce(lambda x, y: x + y, M_t_strain_vec)
    M_t_curvature = 0.0 #functools.reduce(lambda x, y: x + y, M_t_curvature_vec)

    if write_vtk_q:
        st = time.time()
        file_name = "torsion_hdiv_cosserat_elasticity.vtk"
        write_vtk_file(file_name, gmesh, fe_space, alpha)
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return M_t_strain, M_t_curvature


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


def create_conformal_mesher_from_file(file_name, dim):
    mesher = ConformalMesher(dimension=dim)
    mesher.write_mesh(file_name)
    return mesher


def create_mesh_from_file(file_name, dim, write_vtk_q=False):
    gmesh = Mesh(dimension=dim, file_name=file_name)
    gmesh.build_conformal_mesh()
    if write_vtk_q:
        gmesh.write_vtk()
    return gmesh


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

def main():

    k_order = 2
    write_geometry_vtk = True
    write_vtk = True
    mesh_file = "gmsh_files/cylinder.msh"
    gmesh = create_mesh_from_file(mesh_file, 3, write_geometry_vtk)

    l_cvalues = np.logspace(-5, 5, num=20, endpoint=True)
    l_cvalues = [1.0e-5]
    m_t_values = []
    for L_c in l_cvalues:
        m_t_val = torsion_h1_cosserat_elasticity(L_c, k_order, gmesh, write_vtk)
        m_t_val = torsion_hdiv_cosserat_elasticity(L_c, k_order, gmesh, write_vtk)
        m_t_values.append(m_t_val)
    scatter_data = np.insert(np.array(m_t_values), 0, np.array(l_cvalues), axis=1)
    np.savetxt(
        "torsion_test_k" + str(k_order) + "_lc_torque_diagram.txt",
        scatter_data,
        delimiter=",",
        header="Lc, M_t_strain, M_t_curvature",
    )

    return


if __name__ == "__main__":
    main()
