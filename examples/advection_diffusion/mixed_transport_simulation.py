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
from SundusWeakForm import (
    SundusDualWeakForm,
    SundusDualWeakFormBCDirichlet,
)
from topology.topological_queries import find_neighbors_by_codimension_1
import matplotlib.pyplot as plt


def create_product_space(method, gmesh):
    # FESpace: data
    mp_k_order = method[1]["mp"][1]
    mc_k_order = method[1]["mc"][1]
    p_k_order  = method[1]["p"][1]
    c_k_order  = method[1]["c"][1]

    mp_components = 1
    mc_components = 1
    p_components  = 1
    c_components  = 1

    mp_family = method[1]["mp"][0]
    mc_family = method[1]["mc"][0]
    p_family  = method[1]["p"][0]
    c_family  = method[1]["c"][0]

    discrete_spaces_data = {
        "mp": (gmesh.dimension, mp_components, mp_family, mp_k_order, gmesh),
        "mc": (gmesh.dimension, mc_components, mc_family, mc_k_order, gmesh),
        "p" : (gmesh.dimension, p_components, p_family, p_k_order, gmesh),
        "c" : (gmesh.dimension, c_components, c_family, c_k_order, gmesh),
    }

    mp_disc_Q = False
    mc_disc_Q = False
    p_disc_Q  = True
    c_disc_Q  = True
    discrete_spaces_disc = {
        "mp": mp_disc_Q,
        "mc": mc_disc_Q,
        "p" : p_disc_Q,
        "c" : c_disc_Q,
    }

    if gmesh.dimension == 1:
        mp_field_bc_physical_tags = [2, 3]
        mc_field_bc_physical_tags = [2, 3]
    elif gmesh.dimension == 2:
        mp_field_bc_physical_tags = [2, 3, 4, 5]
        mc_field_bc_physical_tags = [2, 3, 4, 5]
    elif gmesh.dimension == 3:
        mp_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
        mc_field_bc_physical_tags = [2, 3, 4, 5, 6, 7]
    else:
        raise ValueError("Case not available.")

    discrete_spaces_bc_physical_tags = {
        "mp": mp_field_bc_physical_tags,
        "mc": mc_field_bc_physical_tags,
    }

    space = ProductSpace(discrete_spaces_data)
    space.make_subspaces_discontinuous(discrete_spaces_disc)
    space.build_structures(discrete_spaces_bc_physical_tags)
    return space


def method_definition():

    # Polynomial order is fixed to 0
    k_order = 0

    # lower order convention
    method_1 = {
        "mp": ("RT", k_order + 1),
        "mc": ("RT", k_order + 1),
        "p" : ("Lagrange", k_order),
        "c" : ("Lagrange", k_order),
    }

    methods = [method_1]
    method_names = ["mixed_rt"]
    return zip(method_names, methods)

def simulation_mixed_transport(method, gmesh, write_vtk_q=False):
    dim = gmesh.dimension

    fe_space = create_product_space(method, gmesh)

    # Nonlinear solver data
    n_iterations = 20
    eps_tol = 1.0e-5
    t_0 = 0.0
    delta_t = 0.1
    t_end = 1.0

    n_dof_g = fe_space.n_dof

    # Material data as scalars
    m_kappa = 1.0
    m_delta = 1.0
    m_eta = 0.0


    def f_kappa(x, y, z):
        return m_kappa

    def f_delta(x, y, z):
        return m_delta

    def f_eta(x, y, z):
        return m_eta


    st = time.time()

    # IC
    p_init = 10.0
    c_init = 0.0
    p_0 = lambda x, y, z: np.array([0.0*x + p_init])
    mp_0 = lambda x, y, z: np.array([[0.0*x,0.0*y]])
    c_0 = lambda x, y, z: np.array([0.0*x + c_init])
    mc_0 = lambda x, y, z: np.array([[0.0*x,0.0*y]])

    # Source terms and boundary data
    p_inlet = 20
    p_outlet = 10
    p_D = lambda x, y, z, t: np.array([p_inlet * (1.0-x/10.0) + p_outlet * (x/10.0)])

    c_inlet = 0
    c_outlet = 0
    c_D = lambda x, y, z, t: np.array([c_inlet * (1.0-x/10.0) + c_outlet * (x/10.0)])

    f_rhs = lambda x, y, z, t: np.array([[0.0*x]])
    r_rhs = lambda x, y, z, t: np.array([[0.0*x]])

    m_functions = {
        "rhs_f": f_rhs,
        "rhs_r": r_rhs,
        "kappa": f_kappa,
        "delta": f_delta,
        "eta": f_eta,
        "delta_t": delta_t
    }

    BC_functions = {
        "p": p_D,
        "c": c_D,
    }

    weak_form = SundusDualWeakForm(fe_space)
    weak_form.functions = m_functions
    bc_weak_form = SundusDualWeakFormBCDirichlet(fe_space)
    bc_weak_form.functions = BC_functions


    # Computing initial condition
    IC_functions = {
        "mp": mp_0,
        "mc": mc_0,
        "p": p_0,
        "c": c_0,
    }

    # initial condition is initialized with a L2 projector of given functions
    alpha_n = l2_projector(fe_space,IC_functions)
    if write_vtk_q:
        # post-process solution
        st = time.time()
        prefix = 'mixed_transport_'
        suffix = '.vtk'
        file_name = prefix + str(dim) + 'd_' + 't_' + str(0) + suffix
        write_vtk_file(file_name, gmesh, fe_space, alpha_n)
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

    for t_idx, t in enumerate(np.arange(delta_t, t_end + delta_t, delta_t)):
        print("Current time value: ", t)

        def scatter_form_data(jac_g, i, weak_form, t):

            neighs_map = find_neighbors_by_codimension_1(gmesh)

            # destination indexes
            dest = weak_form.space.destination_indexes(i)
            alpha_l_n = alpha_n[dest]
            alpha_l_n_p_1 = alpha_n_p_1[dest]

            r_el, j_el = weak_form.evaluate_form(i, alpha_l_n_p_1, alpha_l_n, t)

            # contribute rhs
            res_g[dest] += r_el

            # contribute lhs
            data = j_el.ravel()
            row = np.repeat(dest, len(dest))
            col = np.tile(dest, len(dest))
            nnz = data.shape[0]
            for k in range(nnz):
                jac_g.setValue(row=row[k], col=col[k], value=data[k], addv=True)

        def scatter_bc_form(jac_g, i, bc_weak_form, t ):
            dest = fe_space.bc_destination_indexes(i)
            alpha_l = alpha_n_p_1[dest]
            r_el, j_el = bc_weak_form.evaluate_form(i, alpha_l, t)

            # contribute rhs
            res_g[dest] += r_el

            # contribute lhs
            data = j_el.ravel()
            row = np.repeat(dest, len(dest))
            col = np.tile(dest, len(dest))
            nnz = data.shape[0]
            for k in range(nnz):
                jac_g.setValue(row=row[k], col=col[k], value=data[k], addv=True)

        jac_g = PETSc.Mat()
        jac_g.createAIJ([n_dof_g, n_dof_g])

        res_g = np.zeros(n_dof_g)
        print("n_dof: ", n_dof_g)

        # initial guess
        alpha_n_p_1 = alpha_n.copy()

        for iter in range(n_iterations):

            # Assembler
            st = time.time()
            n_els = len(fe_space.discrete_spaces["mp"].elements)
            [scatter_form_data(jac_g, i, weak_form, t) for i in range(n_els)]

            n_bc_els = len(fe_space.discrete_spaces["mp"].bc_elements)
            [scatter_bc_form(jac_g, i, bc_weak_form, t) for i in range(n_bc_els)]

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
            alpha_n_p_1 += delta_alpha

            # Set up to zero lhr and rhs
            res_g *= 0.0
            jac_g.scale(0.0)

        alpha_n = alpha_n_p_1.copy()
        if write_vtk_q:
            # post-process solution
            st = time.time()
            prefix = 'mixed_transport_'
            suffix = '.vtk'
            file_name = prefix + str(dim) + 'd_' + 't_' + str(t_idx+1) + suffix
            write_vtk_file(file_name, gmesh, fe_space, alpha_n_p_1)
            et = time.time()
            elapsed_time = et - st
            print("Post-processing time:", elapsed_time, "seconds")

    return


def create_mesh_from_file(file_name, dim, write_vtk_q=False):
    gmesh = Mesh(dimension=dim, file_name=file_name)
    gmesh.build_conformal_mesh()
    if write_vtk_q:
        gmesh.write_vtk()
    return gmesh


def main():

    # Problem data


    # Mesh data
    ref_level = 0
    dimension = 2
    write_geometry_vtk = True

    # load geometry
    mesh_file = (
            "gmsh_files/ex_1/example_1_" + str(dimension) + "d_l_" + str(ref_level) + ".msh"
    )
    gmesh = create_mesh_from_file(mesh_file, dimension, write_geometry_vtk)

    # run simulation for each registered method
    for method in method_definition():
            simulation_mixed_transport(method, gmesh, True)

    print("Simulation completed.")
    return

if __name__ == "__main__":
    main()
