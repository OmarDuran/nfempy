
import numpy as np
from geometry.geometry_builder import GeometryBuilder
import geometry.fracture_network as fn
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh






# Material data as scalars
m_kappa = 1.0

c = 1
m_delta = 10 ** -3


def p_fracture(x, y, z):
    return np.sin(2 * np.pi * y)

def p_grad_fracture(x, y, z):
    return 2 * np.pi * np.cos(2 * np.pi * y)

def f_kappa(x, y, z):
    return m_kappa

def f_delta(x, y, z):
    return m_delta



# exact solution

p_exact = lambda x, y, z: np.array([(c * x + f_delta(x, y, z)) * p_fracture(x, y, z)])
u_exact = lambda x, y, z: np.array(
    [
        [
            -c * p_fracture(x, y, z),
            - (c * x + f_delta(x, y, z)) * p_grad_fracture(x, y, z),
        ]
    ]
)

p_f_exact = lambda x, y, z: np.array(
    [
        np.sin(2 * np.pi * y),
    ]
)
u_f_exact = lambda x, y, z: np.array([- f_kappa(x, y, z) *  f_delta(x, y, z) * p_grad_fracture(x, y, z)])

f_rhs = lambda x, y, z: np.array([[4 * (np.pi**2) * (c * x + f_delta(x, y, z)) * p_fracture(x, y, z)]])
r_rhs = lambda x, y, z: np.array([[4 * (np.pi**2) * f_kappa(x, y, z) *  f_delta(x, y, z)  * np.sin(2 * np.pi * y)]])


fracture_set_tags = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]]


def generate_geometry_2d():
    box_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    g_builder = GeometryBuilder(dimension=2)
    g_builder.build_box_2D(box_points)
    return g_builder


def fracture_2d_set():
    fracture_1 = np.array([[0.5, 0.25, 0], [0.5, 0.75, 0]])
    fracture_2 = np.array([[0.25, 0.5, 0], [0.75, 0.5, 0]])
    fracture_3 = np.array([[0.2, 0.35, 0], [0.85, 0.35, 0]])
    fracture_4 = np.array([[0.15, 0.15, 0], [0.85, 0.85, 0]])
    fracture_5 = np.array([[0.15, 0.85, 0], [0.85, 0.15, 0]])
    fractures = [fracture_1, fracture_2, fracture_3, fracture_4, fracture_5]
    return fractures


def generate_fracture_network(fractures):
    fracture_network = fn.FractureNetwork(dimension=2, physical_tag_shift=10)
    fracture_network.intersect_1D_fractures(fractures)
    fracture_network.build_grahp(all_fixed_d_cells_q=True)
    return fracture_network


def generate_conformal_mesh(fracture_tags):
    mesher = ConformalMesher(dimension=2)
    mesher.set_geometry_builder(generate_geometry_2d())
    fractures = []
    for tag in fracture_tags:
        fractures.append(fracture_2d_set()[tag])
    mesher.set_fracture_network(generate_fracture_network(fractures))
    mesher.set_points()
    mesher.generate(0.1)
    mesher.write_mesh("gmesh.msh")
    return mesher


def generate_mesh(fracture_tags, write_vtk_q = False):
    conformal_mesh = generate_conformal_mesh(fracture_tags)
    gmesh = Mesh(dimension=2, file_name="gmesh.msh")
    gmesh.set_conformal_mesher(conformal_mesh)
    gmesh.build_conformal_mesh()
    gmesh.cut_conformity_on_fractures_mds_ec()
    if write_vtk_q:
        gmesh.write_vtk()
    return gmesh


fracture_tags = fracture_set_tags[1]
gmesh = generate_mesh(fracture_tags, True)
check_q = gmesh.circulate_internal_bc()
assert check_q[0]







