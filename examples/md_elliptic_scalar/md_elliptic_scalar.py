import numpy as np
from petsc4py import PETSc
import time

from exact_functions import get_exact_functions_by_co_dimension
from exact_functions import get_rhs_by_co_dimension
from postprocess.projectors import l2_projector
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
from RobinCouplingWeakForm import RobinCouplingWeakForm


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
        mp_field_bc_physical_tags = [[], [20], [2, 3, 4, 5, 50]]
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
    fracture_1 = np.array([[0.2, 0.5, 0.0], [0.8, 0.5, 0.0]])
    fracture_2 = np.array([[0.2, 0.8, 0.0], [0.8, 0.4, 0.0]])
    fractures = [fracture_0, fracture_1, fracture_2]
    fractures = [fracture_0]
    return np.array(fractures)


def generate_conformal_mesh(md_domain, h_val, fracture_physical_tags):

    physical_tags = [fracture_physical_tags["line"]]
    transfinite_agruments = {"n_points": 15, "meshType": "Bump", "coef": 1.0}
    mesh_arguments = {
        "lc": h_val,
        "n_refinements": 0,
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


# Material data as scalars
m_c = 1.0
m_kappa = 1.0
m_kappa_normal = 1.0e14
m_delta = 1.0e-3

# rock domain
lx = 1.0
ly = 1.0
domain_physical_tags = {"area": 1, "bc_0": 2, "bc_1": 3, "bc_2": 4, "bc_3": 5}
box_points = np.array([[0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0]])

# fracture data
lines = fracture_disjoint_set()
fracture_physical_tags = {"line": 10, "internal_bc": 20, "point": 30}
md_domain = create_md_box_2D(
    box_points, domain_physical_tags, lines, fracture_physical_tags
)

# Conformal gmsh discrete representation
h_val = 0.075
gmesh = generate_conformal_mesh(md_domain, h_val, fracture_physical_tags)

physical_tags = {"c1": 10, "c1_clones": 50}
physical_tags = fracture_physical_tags
physical_tags["line_clones"] = 50
physical_tags["point_clones"] = 100
interfaces = cut_conformity_along_c1_lines(lines, physical_tags, gmesh, False)
gmesh.write_vtk()


k_order = 0
write_vtk_q = True
case_name = "md_elliptic_"
flux_name = "q"
potential_name = "p"

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


# First assembly trial
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

weak_form_c0 = MixedWeakForm(md_produc_space[0])
weak_form_c0.functions = m_functions_c0

weak_form_c1 = MixedWeakForm(md_produc_space[1])
weak_form_c1.functions = m_functions_c1

bc_weak_form_c0 = WeakFormBCDir(md_produc_space[0])
bc_weak_form_c0.functions = exact_functions_c0

bc_weak_form_c1 = WeakFormBCDir(md_produc_space[1])
bc_weak_form_c1.functions = exact_functions_c1

m_functions_int_robin = {
    "delta": f_delta,
    "kappa_normal": f_kappa_normal_c1,
}

int_robin_weak_form = RobinCouplingWeakForm(md_produc_space)
int_robin_weak_form.functions = m_functions_int_robin

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

def scatter_robin_form_data(A, c0_idx, c1_idx, int_weak_form):

    dest_c0 = int_weak_form.space[0].bc_destination_indexes(c0_idx, 'q')
    dest_c1 = int_weak_form.space[1].destination_indexes(c1_idx, 'p')
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


n_els_c0 = len(md_produc_space[0].discrete_spaces["q"].elements)
n_els_c1 = len(md_produc_space[1].discrete_spaces["q"].elements)
[scatter_form_data(A, i, weak_form_c0) for i in range(n_els_c0)]
[scatter_form_data(A, i, weak_form_c1) for i in range(n_els_c1)]

n_bc_els_c0 = len(md_produc_space[0].discrete_spaces["q"].bc_elements)
n_bc_els_c1 = len(md_produc_space[1].discrete_spaces["q"].bc_elements)
[scatter_bc_form(A, i, bc_weak_form_c0) for i in range(n_bc_els_c0)]
[scatter_bc_form(A, i, bc_weak_form_c1) for i in range(n_bc_els_c1)]

# Interface weak forms
for interface in interfaces:
    c1_data = interface['c1']
    c1_el_idx = [md_produc_space[1].discrete_spaces["q"].id_to_element[cell.id] for cell in c1_data[0]]
    c0_pel_idx = [md_produc_space[0].discrete_spaces["q"].id_to_bc_element[cell.id] for cell in c1_data[1]]
    c0_nel_idx = [md_produc_space[0].discrete_spaces["q"].id_to_bc_element[cell.id] for cell in c1_data[2]]
    for c1_idx, p_c0_idx, n_c0_idx in zip(c1_el_idx, c0_pel_idx, c0_nel_idx):
        scatter_robin_form_data(A, p_c0_idx, c1_idx, int_robin_weak_form) # positive side
        scatter_robin_form_data(A, n_c0_idx, c1_idx, int_robin_weak_form) # negative side

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

ksp.solve(b, x)
alpha = x.array

et = time.time()
elapsed_time = et - st
print("Linear solver time:", elapsed_time, "seconds")

for co_dim in [0, 1]:
    if write_vtk_q:
        # post-process solution
        st = time.time()
        file_name = "md_elliptic_two_fields_c" + str(co_dim) + ".vtk"
        write_vtk_file_with_exact_solution(
            file_name, gmesh, md_produc_space[co_dim], exact_functions[co_dim], alpha
        )
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")

# post-process projection
for co_dim in [0, 1]:
    md_produc_space[0].dof_shift = 0
    md_produc_space[1].dof_shift = 0
    alpha = l2_projector(md_produc_space[co_dim], exact_functions[co_dim])
    if write_vtk_q:
        # post-process solution
        st = time.time()
        file_name = "projection_md_elliptic_two_fields_c" + str(co_dim) + ".vtk"
        write_vtk_file_with_exact_solution(
            file_name, gmesh, md_produc_space[co_dim], exact_functions[co_dim], alpha
        )
        et = time.time()
        elapsed_time = et - st
        print("Post-processing time:", elapsed_time, "seconds")
