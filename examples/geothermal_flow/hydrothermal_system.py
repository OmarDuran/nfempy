"""
File: three_fields_advection_diffusion.py
Description: Hydrothermal Flow Model in the System H2O–NaCl:
https://doi.org/10.1007/s11242-020-01499-6

Author: Omar Duran
Email: omaryesiduran@gmail.com
Date: 2024-06-11
Version: 1.0.0
License: GPL-3.0 license

"""

import time
from functools import partial
import numpy as np
from petsc4py import PETSc

from geometry.domain import Domain
from geometry.domain_market import build_box_1D, build_box_2D, build_box_3D
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh
from postprocess.l2_error_post_processor import l2_error
from postprocess.projectors import l2_projector
from postprocess.solution_post_processor import write_vtk_file
from spaces.product_space import ProductSpace
from assembly.SequentialAssembler import SequentialAssembler
from HydrothermalWeakForms import (
    DiffusionWeakForm,
    DiffusionWeakFormBCRobin,
    AdvectionWeakForm,
    AdvectionWeakFormBC,
)
import matplotlib.pyplot as plt


def create_product_space(method, gmesh):
    if gmesh.dimension == 1:
        field_bc_physical_tags = [2, 3]
    elif gmesh.dimension == 2:
        field_bc_physical_tags = [2, 3, 4, 5]
    elif gmesh.dimension == 3:
        field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
    else:
        raise ValueError("Case not available.")

    # FESpace: data
    discrete_spaces_data = {}
    discrete_spaces_disc = {}
    discrete_spaces_bc_physical_tags = {}
    n_components = 1
    for item in method[1].items():
        field, (family, k_order) = item
        discrete_spaces_data[field] = (
            gmesh.dimension,
            n_components,
            family,
            k_order,
            gmesh,
        )
        if family in ["RT", "BDM"]:
            discrete_spaces_disc[field] = False
            discrete_spaces_bc_physical_tags[field] = field_bc_physical_tags
        else:
            discrete_spaces_disc[field] = True

        aka = 0

    space = ProductSpace(discrete_spaces_data)
    space.make_subspaces_discontinuous(discrete_spaces_disc)
    space.build_structures(discrete_spaces_bc_physical_tags)
    return space


def method_definition():
    # lower order method
    k_order = 0
    method = {
        "md": ("RT", k_order + 1),
        "ca": ("RT", k_order + 1),
        "qd": ("RT", k_order + 1),
        "qa": ("RT", k_order + 1),
        "p": ("Lagrange", k_order),
        "z": ("Lagrange", k_order),
        "h": ("Lagrange", k_order),
        "t": ("Lagrange", k_order),
        "sv": ("Lagrange", k_order),
        "x_H2O_l": ("Lagrange", k_order),
        "x_H2O_v": ("Lagrange", k_order),
        "x_NaCl_l": ("Lagrange", k_order),
        "x_NaCl_v": ("Lagrange", k_order),
    }
    return ("mixed_rt", method)


def hydrothermal_mixed_formulation(method, gmesh, write_vtk_q=False):

    dim = gmesh.dimension
    fe_space = create_product_space(method, gmesh)

    # Nonlinear solver data
    n_iterations = 20
    eps_tol = 1.0e-4
    delta_t = 1.0
    t_end = 1.0

    n_dof_g = fe_space.n_dof

    # Material data as scalars

    # Constant material properties
    m_K_thermal = 3.0
    m_kappa = 1.0
    m_porosity = 0.1
    m_rho_r = 2650.0
    m_cp_r = 1000.0

    def f_K_thermal(x, y, z):
        return m_K_thermal

    def f_kappa(x, y, z):
        return m_kappa

    def f_porosity(x, y, z):
        return m_porosity

    def f_rho_r(x, y, z):
        return m_rho_r

    def f_cp_r(x, y, z):
        return m_cp_r

    m_functions = {
        "K_thermal": f_K_thermal,
        "kappa": f_kappa,
        "porosity": f_porosity,
        "rho_r": f_rho_r,
        "cp_r": f_cp_r,
    }

    def xi_map(x, y, z, m_west, m_east):
        return m_west * (1 - x) + m_east * x

    def eta_map(x, y, z, m_south, m_north):
        return m_south * (1 - y) + m_north * y



    f_p = partial(xi_map, m_west=25.0, m_east=3.56)
    f_beta_md = partial(xi_map, m_west=1.0e10, m_east=1.0e10)
    f_gamma_md = partial(xi_map, m_west=0.0, m_east=0.0)
    f_t = partial(eta_map, m_south=200.0, m_north=200.0)
    f_beta_qd = partial(eta_map, m_south=1.0e10, m_north=1.0e10)
    f_gamma_qd = partial(eta_map, m_south=0.0, m_north=0.0)
    f_z = partial(xi_map, m_west=0.0, m_east=0.0)

    m_bc_functions = {
        "p_D": f_p,
        "t_D": f_t,
        "z_inlet": f_z,
        "beta_md": f_beta_md,
        "gamma_md": f_gamma_md,
        "beta_qd": f_beta_qd,
        "gamma_qd": f_gamma_qd,
    }

    weak_form = DiffusionWeakForm(fe_space)
    weak_form.functions = m_functions
    bc_weak_form = DiffusionWeakFormBCRobin(fe_space)
    bc_weak_form.functions = m_bc_functions

    advection_weak_form = AdvectionWeakForm(fe_space)
    advection_weak_form.functions = m_bc_functions
    bc_advection_weak_form = AdvectionWeakFormBC(fe_space)
    bc_advection_weak_form.functions = m_bc_functions

    # retrieve external and internal triplets
    c1_entities = [cell for cell in gmesh.cells if cell.dimension == dim - 1]
    gc0_c1 = gmesh.build_graph(dim, 1)
    c1_triplets = [
        (cell.id, list(gc0_c1.predecessors(cell.id))) for cell in c1_entities
    ]
    c1_itriplets = [triplet for triplet in c1_triplets if len(triplet[1]) == 2]
    c1_epairs = [
        (triplet[0], triplet[1][0]) for triplet in c1_triplets if len(triplet[1]) == 1
    ]
    gidx_midx = fe_space.discrete_spaces["md"].id_to_element

    # create sequences
    n_els = len(fe_space.discrete_spaces["md"].elements)
    n_bc_els = len(fe_space.discrete_spaces["md"].bc_elements)
    sequence_domain = [i for i in range(n_els)]
    sequence_bc_domain = [i for i in range(n_bc_els)]
    sequence_c1_itriplets = [
        (triplet[0], [gidx_midx[triplet[1][0]], gidx_midx[triplet[1][1]]])
        for triplet in c1_itriplets
    ]
    sequence_c1_epairs = [(pair[0], gidx_midx[pair[1]]) for pair in c1_epairs]

    # Initial Guess
    alpha = np.zeros(n_dof_g)

    for t in np.arange(delta_t, t_end + delta_t, delta_t):
        print("Current time value: ", t)

        jac_g = PETSc.Mat()
        jac_g.createAIJ([n_dof_g, n_dof_g])

        res_g = np.zeros(n_dof_g)
        print("n_dof: ", n_dof_g)

        # initial guess
        alpha_n = alpha.copy()
        # alpha_n = l2_projector(fe_space, exact_functions)

        for iter in range(n_iterations):
            # break

            # Assembler
            assembler = SequentialAssembler(fe_space, jac_g, res_g)
            form_to_input_list = {
                "difussion_form": [
                    "time_dependent_form",
                    sequence_domain,
                    weak_form,
                    (alpha_n, alpha),
                    t,
                ],
                "difussion_bc_form": [
                    "time_dependent_bc_form",
                    sequence_bc_domain,
                    bc_weak_form,
                    (alpha_n, alpha),
                    t,
                ],
                # "advection_form": [
                #     "interface_form",
                #     sequence_c1_itriplets,
                #     advection_weak_form,
                #     alpha_n_p_1,
                # ],
                # "advection_bc_form": [
                #     "bc_interface_form",
                #     sequence_c1_epairs,
                #     bc_advection_weak_form,
                #     alpha_n_p_1,
                # ],
            }
            assembler.form_to_input_list = form_to_input_list
            assembler.scatter_forms(measure_time_q=False)

            jac_g.assemble()

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
            alpha_n += delta_alpha

            # Set up to zero lhr and rhs
            res_g = np.zeros_like(res_g)
            jac_g.scale(0.0)

        alpha = alpha_n

    if write_vtk_q:
        # post-process solution
        st = time.time()
        file_name = "hydrothermal_system.vtk"
        cell_centered_fields = []
        for item in method[1].items():
            field, (family, k_order) = item
            if family not in ['RT', 'BDM']:
                cell_centered_fields.append(field)

        write_vtk_file(
            file_name, gmesh, fe_space, alpha_n, cell_centered_fields
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


def main():

    h = 0.1
    dimension = 2

    domain = create_domain(dimension)
    error_data = np.empty((0, 2), float)
    method = method_definition()
    mesher = create_conformal_mesher(domain, h, 0)
    gmesh = create_mesh(dimension, mesher, True)
    hydrothermal_mixed_formulation(method, gmesh, True)
    return


if __name__ == "__main__":
    main()
