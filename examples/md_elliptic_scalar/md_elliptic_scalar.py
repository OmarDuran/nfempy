
import numpy as np
import time

from exact_functions import get_exact_functions_by_co_dimension
from postprocess.projectors import l2_projector
from postprocess.solution_post_processor import write_vtk_file_with_exact_solution
from spaces.product_space import ProductSpace
from geometry.geometry_builder import GeometryBuilder
import geometry.fracture_network as fn
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh


def method_definition(k_order,  flux_names, potential_names):
    # lower order convention
    method_1 = {
        flux_names[0]: ("RT", k_order + 1),
        flux_names[1]: ("RT", k_order + 1),
        potential_names[0]: ("Lagrange", k_order),
        potential_names[1]: ("Lagrange", k_order),
    }

    methods = [method_1]
    method_names = ["mixed_rt"]
    return zip(method_names, methods)

def create_product_space(method, gmesh, flux_names, potential_names):

    assert method[1][flux_names[0]][1] == method[1][flux_names[1]][1]
    assert method[1][potential_names[0]][1] == method[1][potential_names[1]][1]

    # FESpace: data
    mp_k_order = method[1][flux_names[0]][1]
    p_k_order = method[1][potential_names[0]][1]

    mp_components = 1
    p_components = 1

    assert method[1][flux_names[0]][0] == method[1][flux_names[1]][0]
    assert method[1][potential_names[0]][0] == method[1][potential_names[1]][0]

    mp_family = method[1][flux_names[1]][0]
    p_family = method[1][potential_names[1]][0]

    discrete_spaces_data = {
        # flux_names[0]: (gmesh.dimension, mp_components, mp_family, mp_k_order, gmesh),
        flux_names[1]: (gmesh.dimension - 1, mp_components, mp_family, mp_k_order, gmesh),
        # potential_names[0]: (gmesh.dimension, p_components, p_family, p_k_order, gmesh),
        potential_names[1]: (gmesh.dimension-1, p_components, p_family, p_k_order, gmesh),
    }

    mp_disc_Q = False
    p_disc_Q = True
    discrete_spaces_disc = {
        # flux_names[0]: mp_disc_Q,
        flux_names[1]: mp_disc_Q,
        # potential_names[0]: p_disc_Q,
        potential_names[1]: p_disc_Q,
    }

    if gmesh.dimension == 2:
        mp_field_bc_physical_tags = [[2, 3, 4, 5, 99, 101], [10]]
    else:
        raise ValueError("Case not available.")

    discrete_spaces_bc_physical_tags = {
        # flux_names[0]: mp_field_bc_physical_tags[0],
        flux_names[1]: mp_field_bc_physical_tags[1],
    }

    space = ProductSpace(discrete_spaces_data)
    space.make_subspaces_discontinuous(discrete_spaces_disc)
    space.build_structures(discrete_spaces_bc_physical_tags)
    return space



# Material data as scalars
m_c = 1.0
m_kappa = 1.0
m_delta = 10 ** -3





def generate_geometry_2d():
    box_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    g_builder = GeometryBuilder(dimension=2)
    g_builder.build_box_2D(box_points)
    return g_builder


def fracture_disjoint_set():
    fracture_1 = np.array([[0.5, 0.1, 0], [0.5, 0.75, 0]])
    fracture_2 = np.array([[0.1, 0.5, 0], [0.75, 0.5, 0]])
    fractures = [fracture_1, fracture_2]
    return fractures


def generate_fracture_network(fractures):
    fracture_network = fn.FractureNetwork(dimension=2, physical_tag_shift=10)
    fracture_network.intersect_1D_fractures(fractures)
    fracture_network.build_grahp(all_fixed_d_cells_q=True)
    return fracture_network


def generate_conformal_mesh():
    mesher = ConformalMesher(dimension=2)
    mesher.set_geometry_builder(generate_geometry_2d())
    fractures = fracture_disjoint_set()
    mesher.set_fracture_network(generate_fracture_network(fractures))
    mesher.set_points()
    mesher.generate(0.1)
    mesher.write_mesh("gmesh.msh")
    return mesher


def generate_mesh(write_vtk_q = False):
    conformal_mesh = generate_conformal_mesh()
    gmesh = Mesh(dimension=2, file_name="gmesh.msh")
    gmesh.set_conformal_mesher(conformal_mesh)
    gmesh.build_conformal_mesh()
    # gmesh.cut_conformity_on_fractures_mds_ec()
    if write_vtk_q:
        gmesh.write_vtk()
    return gmesh


gmesh = generate_mesh(True)
# check_q = gmesh.circulate_internal_bc()
# assert check_q[0]


# cells_c0 = [cell for cell in gmesh.cells if cell.dimension == 2]
#
# for cell in gmesh.cells:
#     if cell.dimension == 1 and cell.material_id == 10:
#         print("Fracture coordinates: ", gmesh.points[cell.node_tags])
# cells_c1 = [cell for cell in gmesh.cells if cell.dimension == 1 and cell.material_id == 10]


k_order = 0
co_dim = 1
write_vtk_q = True
case_name = 'md_elliptic_'
flux_names = ['u_c0', 'u_c1']
potential_names = ['p_c0', 'p_c1']
methods = method_definition(k_order, flux_names, potential_names)
for method in methods:
    fe_space = create_product_space(method, gmesh, flux_names, potential_names)
    aka = 0

exact_functions_c0 = get_exact_functions_by_co_dimension(0, flux_names, potential_names, m_c, m_kappa, m_delta)
exact_functions_c1 = get_exact_functions_by_co_dimension(1, flux_names, potential_names, m_c, m_kappa, m_delta)
exact_functions = [exact_functions_c0,exact_functions_c1] # {**exact_functions_c0, **exact_functions_c1}

alpha = l2_projector(fe_space,exact_functions[co_dim])
if write_vtk_q:
    # post-process solution
    st = time.time()
    file_name = case_name + "two_fields.vtk"
    write_vtk_file_with_exact_solution(
        file_name, gmesh, fe_space, exact_functions[co_dim], alpha
    )
    et = time.time()
    elapsed_time = et - st
    print("Post-processing time:", elapsed_time, "seconds")



