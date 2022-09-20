
import numpy as np

from numpy import linalg as la

from shapely.geometry import LineString

import geometry.fracture_network as fn
import networkx as nx

import matplotlib.pyplot as plt

from geometry.cell import Cell
from geometry.geometry_builder import GeometryBuilder
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh

import basix
from basix import ElementFamily, CellType, LagrangeVariant, LatticeType



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


def dof_permutations_tranformations():

    # Degree 2 Lagrange element
    # =========================
    #
    # We create a degree 2 Lagrange element on a triangle, then print the
    # values of the attributes `dof_transformations_are_identity` and
    # `dof_transformations_are_permutations`.
    #
    # The value of `dof_transformations_are_identity` is False: this tells
    # us that permutations or transformations are needed for this element.
    #
    # The value of `dof_transformations_are_permutations` is True: this
    # tells us that for this element, all the corrections we need to apply
    # permutations. This is the simpler case, and means we make the
    # orientation corrections by applying permutations when creating the
    # DOF map.

    lagrange = basix.create_element(
        ElementFamily.P, CellType.triangle, 2, LagrangeVariant.equispaced)
    print(lagrange.dof_transformations_are_identity)
    print(lagrange.dof_transformations_are_permutations)

    # We can apply permutations by using the matrices returned by the
    # method `base_transformations`. This method will return one matrix
    # for each edge of the cell (for 2D and 3D cells), and two matrices
    # for each face of the cell (for 3D cells). These describe the effect
    # of reversing the edge or reflecting and rotating the face.
    #
    # For this element, we know that the base transformations will be
    # permutation matrices.

    print(lagrange.base_transformations())

    # The matrices returned by `base_transformations` are quite large, and
    # are equal to the identity matrix except for a small block of the
    # matrix. It is often easier and more efficient to use the matrices
    # returned by the method `entity_transformations` instead.
    #
    # `entity_transformations` returns a dictionary that maps the type
    # of entity (`"interval"`, `"triangle"`, `"quadrilateral"`) to a
    # matrix describing the effect of permuting that entity on the DOFs
    # on that entity.
    #
    # For this element, we see that this method returns one matrix for
    # an interval: this matrix reverses the order of the four DOFs
    # associated with that edge.

    print(lagrange.entity_transformations())

    # In orders to work out which DOFs are associated with each edge,
    # we use the attribute `entity_dofs`. For example, the following can
    # be used to see which DOF numbers are associated with edge (dim 1)
    # number 2:

    print(lagrange.entity_dofs[1][2])

    # Degree 2 Lagrange element
    # =========================
    #
    # For a degree 2 Lagrange element, no permutations or transformations
    # are needed. We can verify this by checking that
    # `dof_transformations_are_identity` is `True`. To confirm that the
    # transformations are identity matrices, we also print the base
    # transformations.

    lagrange_degree_2 = basix.create_element(
        ElementFamily.P, CellType.triangle, 2, LagrangeVariant.equispaced)
    print(lagrange_degree_2.dof_transformations_are_identity)
    print(lagrange_degree_2.base_transformations())

    # Degree 2 Nédélec element
    # ========================
    #
    # For a degree 2 Nédélec (first kind) element on a tetrahedron, the
    # corrections are not all permutations, so both
    # `dof_transformations_are_identity` and
    # `dof_transformations_are_permutations` are `False`.

    nedelec = basix.create_element(ElementFamily.N1E, CellType.tetrahedron, 2)
    print(nedelec.dof_transformations_are_identity)
    print(nedelec.dof_transformations_are_permutations)

    # For this element, `entity_transformations` returns a dictionary
    # with two entries: a matrix for an interval that describes
    # the effect of reversing the edge; and an array of two matrices
    # for a triangle. The first matrix for the triangle describes
    # the effect of rotating the triangle. The second matrix describes
    # the effect of reflecting the triangle.
    #
    # For this element, the matrix describing the effect of rotating
    # the triangle is
    #
    # .. math::
    #    \left(\begin{array}{cc}-1&-1\\1&0\end{array}\right).
    #
    # This is not a permutation, so this must be applied when assembling
    # a form and cannot be applied to the DOF numbering in the DOF map.

    print(nedelec.entity_transformations())

    # To demonstrate how these transformations can be used, we create a
    # lattice of points where we will tabulate the element.

    points = basix.create_lattice(
        CellType.tetrahedron, 3, LatticeType.equispaced, True)

    # If (for example) the direction of edge 2 in the physical cell does
    # not match its direction on the reference, then we need to adjust the
    # tabulated data.
    #
    # As the cell sub-entity that we are correcting is an interval, we
    # get the `"interval"` item from the entity transformations dictionary.
    # We use `entity_dofs[1][2]` (1 is the dimension of an edge, 2 is the
    # index of the edge we are reversing) to find out which dofs are on
    # our edge.
    #
    # To adjust the tabulated data, we loop over each point in the lattice
    # and over the value size. For each of these values, we apply the
    # transformation matrix to the relevant DOFs.

    data = nedelec.tabulate(0, points)

    transformation = nedelec.entity_transformations()["interval"][0]
    dofs = nedelec.entity_dofs[1][2]

    for point in range(data.shape[1]):
        for dim in range(data.shape[3]):
            data[0, point, dofs, dim] = np.dot(transformation, data[0, point, dofs, dim])

    print(data)

def examples_fiat():


    points, weights = basix.make_quadrature(
        basix.QuadratureType.gauss_jacobi, CellType.triangle, 3)

    lagrange = basix.create_element(
        ElementFamily.P, CellType.triangle, 3, LagrangeVariant.equispaced)

    lagrange.base_transformations()
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

    # F0 = f * u * v * w * dx
    # a, L = system(F0)
    # assert (len(a.integrals()) == 0)
    # assert (len(L.integrals()) == 0)
    #
    # F1 = derivative(F0, f)
    # a, L = system(F1)
    # assert (len(a.integrals()) == 0)
    # assert (len(L.integrals()) == 0)
    #
    # F2 = action(F0, f)
    # a, L = system(F2)
    # assert (len(a.integrals()) == 1)
    # assert (len(L.integrals()) == 0)
    #
    # F3 = action(F2, f)
    # a, L = system(F3)
    # assert (len(L.integrals()) == 1)
    #
    # cell = Cell("triangle")
    # cell = triangle
    # aka = 0

    # from FIAT import Lagrange, quadrature, shapes
    # shape = shapes.TRIANGLE
    # degree = 2
    # U = Lagrange.Lagrange(shape, degree)
    # Q = quadrature.make_quadrature(shape, 2)
    # Ufs = U.function_space()
    # Ufs.tabulate(Q.get_points())

def domain_with_fractures():
    # Higher dimension geometry
    s = 1.0
    box_points = s * np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    g_builder = GeometryBuilder(dimension=2)
    g_builder.build_box_2D(box_points)
    g_builder.build_grahp()

    # insert base fractures
    fracture_1 = np.array([[0.5, 0.25], [0.5, 0.75]])
    fracture_2 = np.array([[0.25, 0.5], [0.75, 0.5]])
    fracture_3 = np.array([[0.2, 0.35], [0.85, 0.35]])
    fracture_4 = np.array([[0.15, 0.15], [0.85, 0.85]])
    fracture_5 = np.array([[0.15, 0.85], [0.85, 0.15]])

    fractures = [fracture_1]

    fracture_network = fn.FractureNetwork(dimension=2, physical_tag_shift=10)
    fracture_network.intersect_1D_fractures(fractures, render_intersection_q=False)
    fracture_network.build_grahp(all_fixed_d_cells_q=True)
    # fracture_network.draw_grahp()

    mesher = ConformalMesher(dimension=2)
    mesher.set_geometry_builder(g_builder)
    mesher.set_fracture_network(fracture_network)
    mesher.set_points()
    mesher.generate(1.0)
    mesher.write_mesh("gmesh.msh")

    gmesh = Mesh(dimension=2, file_name="gmesh.msh")
    gmesh.set_conformal_mesher(mesher)
    gmesh.build_conformal_mesh()

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

    check_q = gmesh.circulate_internal_bc()
    if check_q[0]:
        print("Internal bc is closed.")

    aka = 0

def main():

    s = 1.0
    box_points = s * np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    g_builder = GeometryBuilder(dimension=2)
    g_builder.build_box_2D(box_points)
    g_builder.build_grahp()


    mesher = ConformalMesher(dimension=2)
    mesher.set_geometry_builder(g_builder)
    mesher.set_points()
    mesher.generate(1.0)
    mesher.write_mesh("gmesh.msh")


    gmesh = Mesh(dimension=2, file_name="gmesh.msh")
    gmesh.set_conformal_mesher(mesher)
    gmesh.build_conformal_mesh()
    gmesh.write_vtk()


    # Create conformity
    gd2c1 = gmesh.build_graph(2, 1)
    gd2c2 = gmesh.build_graph(2, 2)

    cells_ids = list(gd2c2.nodes())
    vertices_ids = [id for id in cells_ids if gmesh.cells[id].dimension == 0]
    n_vertices = len(vertices_ids)
    k_order = 1

    fields = [1]
    n_fields = len(fields)


    for cell in gmesh.cells:
        if cell.dimension != 2:
            continue

        points, weights = basix.make_quadrature(basix.QuadratureType.gauss_jacobi, CellType.triangle, k_order + 1)
        lagrange = basix.create_element(ElementFamily.P, CellType.triangle, k_order, LagrangeVariant.equispaced)

        # phi_tab = lagrange.tabulate(0, points)
        # print(phi_tab)
        # v, np, n_dof, _ = phi_tab.shape
        # s = (n_dof, n_dof)
        # k_el = np.zeros(s)
        # for i, omega in enumerate(weights):
        #     k_el = k_el + omega * np.outer(phi_tab[0,i,:,0],phi_tab[0,i,:,0])

        # lagrange.base_transformations()
        ako = 0

    aka = 0

if __name__ == '__main__':
    main()



