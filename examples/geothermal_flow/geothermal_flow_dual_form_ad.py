import time

import numpy as np
import scipy
from petsc4py import PETSc

from basis.element_data import ElementData
from geometry.domain import Domain
from geometry.domain_market import build_box_1D, build_box_2D, build_box_3D
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh
from postprocess.l2_error_post_processor import l2_error
from postprocess.projectors import l2_projector
from postprocess.solution_post_processor import write_vtk_file
from spaces.product_space import ProductSpace
from TwoComponentMultiphaseFlowWeakForm import (
    TwoCompMultiPhaseFlowWeakForm,
    TwoCompMultiPhaseFlowWeakFormBCDirichlet,
)
import matplotlib.pyplot as plt


def create_product_space(method, gmesh):
    # FESpace: data
    q_mass_k_order = method[1]["q_mass"][1]
    q_energy_k_order = method[1]["q_energy"][1]
    p_k_order = method[1]["p"][1]
    h_k_order = method[1]["h"][1]
    z_k_order = method[1]["z"][1]

    n_components = 1

    q_mass_family = method[1]["q_mass"][0]
    q_energy_family = method[1]["q_energy"][0]
    p_family = method[1]["p"][0]
    h_family = method[1]["h"][0]
    z_family = method[1]["z"][0]

    discrete_spaces_data = {
        "q_mass": (gmesh.dimension, n_components, q_mass_family, q_mass_k_order, gmesh),
        "q_energy": (gmesh.dimension, n_components, q_energy_family, q_energy_k_order, gmesh),
        "p": (gmesh.dimension, n_components, p_family, p_k_order, gmesh),
        "h": (gmesh.dimension, n_components, h_family, h_k_order, gmesh),
        "z": (gmesh.dimension, n_components, z_family, z_k_order, gmesh),
    }

    q_mass_disc_Q = False
    q_energy_disc_Q = False
    p_disc_Q = True
    h_disc_Q = True
    z_disc_Q = True
    discrete_spaces_disc = {
        "q_mass": q_mass_disc_Q,
        "q_energy": q_energy_disc_Q,
        "p": p_disc_Q,
        "h": h_disc_Q,
        "z": z_disc_Q,
    }

    if gmesh.dimension == 1:
        q_mass_field_bc_physical_tags = [2, 3]
        q_energy_field_bc_physical_tags = [2, 3]
    elif gmesh.dimension == 2:
        q_mass_field_bc_physical_tags = [2, 3, 4, 5]
        q_energy_field_bc_physical_tags = [2, 3, 4, 5]
    elif gmesh.dimension == 3:
        q_mass_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
        q_energy_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
    else:
        raise ValueError("Case not available.")

    discrete_spaces_bc_physical_tags = {
        "q_mass": q_mass_field_bc_physical_tags,
        "q_energy": q_energy_field_bc_physical_tags,
    }

    space = ProductSpace(discrete_spaces_data)
    space.make_subspaces_discontinuous(discrete_spaces_disc)
    space.build_structures(discrete_spaces_bc_physical_tags)
    return space


def method_definition(k_order):
    # lower order convention
    method_1 = {
        "q_mass": ("BDM", k_order + 1),
        "q_energy": ("BDM", k_order + 1),
        "p": ("Lagrange", k_order),
        "h": ("Lagrange", k_order),
        "z": ("Lagrange", k_order),
    }

    methods = [method_1]
    method_names = ["dual_rt"]
    return zip(method_names, methods)


def geothermal_flow_formulation(method, gmesh, write_vtk_q=False):
    dim = gmesh.dimension

    fe_space = create_product_space(method, gmesh)

    n_dof_g = fe_space.n_dof
    rg = np.zeros(n_dof_g)
    alpha = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    # Assembler
    st = time.time()
    A = PETSc.Mat()
    A.createAIJ([n_dof_g, n_dof_g])

    # Material data as scalars
    m_kappa = 1.0
    m_delta = 1.0

    def f_kappa(x, y, z):
        return m_kappa

    def f_delta(x, y, z):
        return m_delta

    st = time.time()

    rhs_p = lambda x, y, z: np.array([+2.0 + 0.0 * x])
    rhs_h = lambda x, y, z: np.array([+2.0 + 0.0 * x])
    rhs_z = lambda x, y, z: np.array([0.0 * x])

    m_functions = {
        "rhs_p": rhs_p,
        "rhs_h": rhs_h,
        "rhs_z": rhs_z,
        "kappa": f_kappa,
        "delta": f_delta,
    }

    bc_p = lambda x, y, z: np.array([0.0 * x])
    bc_h = lambda x, y, z: np.array([0.0 * x])
    bc_z = lambda x, y, z: np.array([0.0 * x])

    bc_functions = {
        "bc_p": bc_p,
        "bc_h": bc_h,
        "bc_z": bc_z,
    }

    weak_form = TwoCompMultiPhaseFlowWeakForm(fe_space)
    weak_form.functions = m_functions
    bc_weak_form = TwoCompMultiPhaseFlowWeakFormBCDirichlet(fe_space)
    bc_weak_form.functions = bc_functions

    # building interfaces at provided codimension
    codim = 1
    cids_codim_1 = [cell.id for cell in gmesh.cells if cell.dimension == dim - codim]
    g_codim_1 = gmesh.build_graph(dim, codim)
    neighs = [list(g_codim_1.predecessors(id)) for id in cids_codim_1]

    aka = 0
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

    assert len(fe_space.discrete_spaces["q_mass"].elements) == len(
        fe_space.discrete_spaces["q_energy"].elements)
    n_els = len(fe_space.discrete_spaces["q_mass"].elements)
    [scatter_form_data(A, i, weak_form) for i in range(n_els)]

    assert len(fe_space.discrete_spaces["q_mass"].bc_elements) == len(
        fe_space.discrete_spaces["q_energy"].bc_elements)
    n_bc_els = len(fe_space.discrete_spaces["q_mass"].bc_elements)
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

    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setConvergenceHistory()

    # petsc_options = {"rtol": 1e-12, "atol": 1e-14}
    # ksp = PETSc.KSP().create()
    # ksp.create(PETSc.COMM_WORLD)
    # ksp.setOperators(A)
    # ksp.setType("fgmres")
    # ksp.setTolerances(**petsc_options)
    # ksp.setConvergenceHistory()
    # ksp.getPC().setType("ilu")
    ksp.solve(b, x)
    alpha = x.array

    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    if write_vtk_q:
        # post-process solution
        st = time.time()
        file_name = "five_fields.vtk"
        write_vtk_file(
            file_name, gmesh, fe_space, alpha
        )
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    return


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


def main():

    k_order = 0
    h = 0.5
    dimension = 2


    domain = create_domain(dimension)
    error_data = np.empty((0, 2), float)
    mesher = create_conformal_mesher(domain, h, 0)
    gmesh = create_mesh(dimension, mesher, True)

    for method in method_definition(k_order):
        geothermal_flow_formulation(method, gmesh, True)

    return


if __name__ == "__main__":
    main()
