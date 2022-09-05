import numpy as np
from numpy import linalg as la

from shapely.geometry import LineString

import geometry.fracture_network as fn
import networkx as nx

import matplotlib.pyplot as plt

from geometry.cell import Cell
from geometry.geometry_builder import GeometryBuilder

import basix
from basix import ElementFamily, CellType, LagrangeVariant



def polygon_polygon_intersection():

    fracture_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

    fracture_2 = np.array([[0.5, 0., 0.5], [0.5, 0., -0.5], [0.5, 1., -0.5], [0.5, 1., 0.5]])
    fracture_3 = np.array([[0., 0.5, -0.5], [1., 0.5, -0.5], [1., 0.5, 0.5],
     [0., 0.5, 0.5]])

    # fracture_2 = np.array([[0.6, 0., 0.5], [0.6, 0., -0.5], [0.6, 1., -0.5], [0.6, 1., 0.5]])
    # fracture_3 = np.array([[0.25, 0., 0.5], [0.914463, 0.241845, -0.207107], [0.572443, 1.18154, -0.207107],
    #  [-0.0920201, 0.939693, 0.5]])

    fractures = [fracture_1,fracture_2,fracture_3]

    fracture_network = fn.FractureNetwork(dimension=3)
    # fracture_network.render_fractures(fractures)
    fracture_network.intersect_2D_fractures(fractures, True)
    fracture_network.build_grahp()
    fracture_network.draw_grahp()
    ika = 0

def examples_fiat():


    points, weights = basix.make_quadrature(
        basix.QuadratureType.gauss_jacobi, CellType.triangle, 4)


    aka = 0


    # element = FiniteElement("Lagrange", triangle, 2)
    #
    # u = TrialFunction(element)
    # v = TestFunction(element)
    #
    # x = SpatialCoordinate(triangle)
    # d_x = x[0] - 0.5
    # d_y = x[1] - 0.5
    # f = 10.0 * exp(-(d_x * d_x + d_y * d_y) / 0.02)
    # g = sin(5.0 * x[0])
    #
    # a = inner(grad(u), grad(v)) * dx
    # L = f * v * dx + g * v * ds

    # V = FiniteElement("CG", interval, 1)
    # v = TestFunction(V)
    # u = TrialFunction(V)
    # w = Argument(V, 2)  # This was 0, not sure why
    # f = Coefficient(V)
    # x = SpatialCoordinate(interval)
    # # pts = CellVertices(interval)

    F0 = f * u * v * w * dx
    a, L = system(F0)
    assert (len(a.integrals()) == 0)
    assert (len(L.integrals()) == 0)

    F1 = derivative(F0, f)
    a, L = system(F1)
    assert (len(a.integrals()) == 0)
    assert (len(L.integrals()) == 0)

    F2 = action(F0, f)
    a, L = system(F2)
    assert (len(a.integrals()) == 1)
    assert (len(L.integrals()) == 0)

    F3 = action(F2, f)
    a, L = system(F3)
    assert (len(L.integrals()) == 1)

    cell = Cell("triangle")
    cell = triangle
    aka = 0

    # from FIAT import Lagrange, quadrature, shapes
    # shape = shapes.TRIANGLE
    # degree = 2
    # U = Lagrange.Lagrange(shape, degree)
    # Q = quadrature.make_quadrature(shape, 2)
    # Ufs = U.function_space()
    # Ufs.tabulate(Q.get_points())

def main():

    examples_fiat()
    return 0

    # polygon_polygon_intersection()
    # return 0

    # Higher dimension geometry
    s = 1.0
    box_points = s * np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    g_builder = GeometryBuilder(dimension=2)
    g_builder.build_box_2D(box_points)
    g_builder.build_grahp()

    # insert base fractures
    fracture_1 = np.array([[0.25, 0.25], [0.75, 0.75]])
    fracture_2 = np.array([[0.25, 0.75], [0.75, 0.25]])
    fracture_3 = np.array([[0.5, 0.25], [0.5, 0.75]])
    fracture_4 = np.array([[0.65, 0.25], [0.65, 0.75]])
    fracture_5 = np.array([[0.25, 0.5], [0.75, 0.5]])

    fracture_1 = np.array([[0.5, 0.25], [0.5, 0.75]])
    fracture_2 = np.array([[0.25, 0.5], [0.75, 0.5]])

    fractures = [fracture_1,fracture_2]

    fracture_network = fn.FractureNetwork(dimension=2)
    fracture_network.intersect_1D_fractures(fractures, render_intersection_q = False)
    fracture_network.build_grahp(all_fixed_d_cells_q = True)
    # fracture_network.draw_grahp()


    mesher = Mesher(dimension=2)
    mesher.set_geometry_builder(g_builder)
    mesher.set_fracture_network(fracture_network)
    mesher.set_points()
    mesher.generate(1.0)
    mesher.write_mesh("gmesh.msh")


    gmesh = Mesh(dimension=2, file_name="gmesh.msh")
    gmesh.set_Mesher(mesher)
    gmesh.transfer_conformal_mesh()

    gd2c1 = gmesh.build_graph(2, 1)
    gd2c2 = gmesh.build_graph(2, 2)
    gd1c1 = gmesh.build_graph(1, 1)
    # gmesh.draw_graph(gd1c1)
    gmesh.cut_conformity_on_fractures()
    gmesh.write_vtk()
    cgd2c1 = gmesh.build_graph_on_materials(2, 1)
    cgd2c2 = gmesh.build_graph_on_materials(2, 2)
    cgd1c1 = gmesh.build_graph_on_materials(1, 1)
    # gmesh.draw_graph(gd1c1)
    aka = 0

if __name__ == '__main__':
    main()



