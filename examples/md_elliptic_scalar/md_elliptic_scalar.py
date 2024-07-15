
import numpy as np
import time

from exact_functions import get_exact_functions_by_co_dimension
from postprocess.projectors import l2_projector
from postprocess.solution_post_processor import write_vtk_file_with_exact_solution
from spaces.product_space import ProductSpace
from spaces.md_product_space import MDProductSpace
from mesh.mesh import Mesh
from topology.domain_market import create_md_box_2D
from mesh.discrete_domain import DiscreteDomain
from mesh.mesh_operations import cut_conformity_along_c1_lines


def method_definition(dimension, k_order, flux_name, potential_name):

    # lower order convention
    if dimension in [1,2,3]:
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
        mp_field_bc_physical_tags = [[], [20], [2, 3, 4, 5]]
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
    fracture_1 = np.array([[0.5, 0.2, 0.0], [0.5, 0.8, 0.0]])
    fracture_2 = np.array([[0.1, 0.5, 0], [0.75, 0.5, 0]])
    fractures = [fracture_1]
    return np.array(fractures)

def generate_conformal_mesh(md_domain, h_val, fracture_physical_tags):

    physical_tags = [fracture_physical_tags['line']]
    transfinite_agruments = {'n_points': 10, 'meshType': "Bump", 'coef': 1.0}
    mesh_arguments = {'lc': h_val, 'n_refinements': 0,
                      'curves_refinement': (physical_tags, transfinite_agruments)}

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
m_delta = 10 ** -3

# rock domain
lx = 1.0
ly = 1.0
domain_physical_tags = {"area": 1, "bc_0": 2, "bc_1": 3, "bc_2": 4, "bc_3": 5}
box_points = np.array([[0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0]])

# fracture data
lines = fracture_disjoint_set()
fracture_physical_tags = {"line": 10, "internal_bc": 20, "point": 30}
md_domain = create_md_box_2D(box_points, domain_physical_tags, lines, fracture_physical_tags)

# Conformal gmsh discrete representation
h_val = 0.1
gmesh = generate_conformal_mesh(md_domain, h_val, fracture_physical_tags)

physical_tags = {'c1': 10, 'c1_clones': 50}
physical_tags = fracture_physical_tags
physical_tags['line_clones'] = 50
physical_tags['point_clones'] = 100
# cut_conformity_along_c1_lines(lines, physical_tags, gmesh)
# gmesh.write_vtk()



k_order = 0
co_dim = 1
write_vtk_q = True
case_name = 'md_elliptic_'
flux_name = 'u'
potential_name = 'p'

md_produc_space = []
for d in [2, 1]:
    methods = method_definition(d, k_order, flux_name, potential_name)
    for method in methods:
        fe_space = create_product_space(d, method, gmesh, flux_name, potential_name)
        md_produc_space.append(fe_space)

aka = 0

exact_functions_c0 = get_exact_functions_by_co_dimension(0, flux_name, potential_name, m_c, m_kappa, m_delta)
exact_functions_c1 = get_exact_functions_by_co_dimension(1, flux_name, potential_name, m_c, m_kappa, m_delta)
exact_functions = [exact_functions_c0, exact_functions_c1]

co_dim = 1
alpha = l2_projector(md_produc_space[co_dim],exact_functions[co_dim])
if write_vtk_q:
    # post-process solution
    st = time.time()
    file_name = "md_elliptic_two_fields_c" + str(co_dim) + ".vtk"
    write_vtk_file_with_exact_solution(
        file_name, gmesh, md_produc_space[co_dim],exact_functions[co_dim], alpha
    )
    et = time.time()
    elapsed_time = et - st
    print("Post-processing time:", elapsed_time, "seconds")



