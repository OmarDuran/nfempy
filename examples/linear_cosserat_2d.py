
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

import scipy.sparse as sp
from scipy.sparse import coo_matrix

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import meshio

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

def matrix_plot(A):
    norm = mcolors.TwoSlopeNorm(vmin=-10.0, vcenter=0, vmax=10.0)
    plt.matshow(A.todense(),norm=norm,cmap='RdBu_r');
    plt.colorbar()
    plt.show()

def main():

    s = 1.0
    box_points = s * np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    g_builder = GeometryBuilder(dimension=2)
    g_builder.build_box_2D(box_points)
    g_builder.build_grahp()


    mesher = ConformalMesher(dimension=2)
    mesher.set_geometry_builder(g_builder)
    mesher.set_points()
    mesher.generate(0.1)
    mesher.write_mesh("gmesh.msh")


    gmesh = Mesh(dimension=2, file_name="gmesh.msh")
    gmesh.set_conformal_mesher(mesher)
    gmesh.build_conformal_mesh() # expensive method
    # gmesh.write_data()
    gmesh.write_vtk()

    # Create conformity
    gd2c1 = gmesh.build_graph(2, 1)
    gd2c2 = gmesh.build_graph(2, 2)

    cells_ids = list(gd2c2.nodes())
    vertices_ids = [id for id in cells_ids if gmesh.cells[id].dimension == 0]
    n_vertices = len(vertices_ids)
    vertex_map = dict(zip(vertices_ids, list(range(n_vertices))))

    k_order = 1

    fields = [1]
    n_fields = len(fields)
    n_dof_g = n_vertices*n_fields
    rgs = (n_dof_g)
    rg = np.zeros(rgs)


    c_size = 0
    cell_map = {}
    for cell in gmesh.cells:
        if cell.dimension != 2:
            continue

        lagrange = basix.create_element(ElementFamily.P, CellType.triangle, k_order, LagrangeVariant.equispaced)
        n_dof = 0
        for n_entity_dofs in lagrange.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs)
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof*n_dof


    row_l = [0] *  c_size
    col_l = [0] *  c_size
    data_l = [0] *  c_size

    for cell in gmesh.cells:
        if cell.dimension != 2:
            continue


        points, weights = basix.make_quadrature(basix.QuadratureType.gauss_jacobi, CellType.triangle, 2 * k_order + 1)
        lagrange = basix.create_element(ElementFamily.P, CellType.triangle, k_order,
                                        LagrangeVariant.equispaced)

        phi_tab = lagrange.tabulate(0, points)
        # print(phi_tab)
        n_dof = phi_tab.shape[2]
        js = (n_dof, n_dof)
        rs = (n_dof)
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        fun = lambda x, y, z: 16 * x * (1.0 - x) * y * (1.0 - y)

        # evaluate mapping
        linear_base = basix.create_element(ElementFamily.P, CellType.triangle, 1,
                                           LagrangeVariant.equispaced)
        g_phi_tab = linear_base.tabulate(1, points)
        cell_points = gmesh.points[cell.node_tags]

        aka = 0
        # linear_base
        for i, omega in enumerate(weights):


            # QR-decomposition is not unique
            # It's only unique up to the signs of the rows of R
            xmap = np.dot(g_phi_tab[0, i, :, 0],cell_points)
            grad_xmap = np.dot(g_phi_tab[[1, 2], i, :, 0],cell_points).T
            q_axes, r_jac = np.linalg.qr(grad_xmap)
            r_sign = np.diag(np.sign(np.diag(r_jac)), 0)
            q_axes = np.dot(q_axes, r_sign)
            r_jac = np.dot(r_sign, r_jac)

            det_g_jac = np.linalg.det(r_jac)
            if det_g_jac < 0.0:
                print('Negative det jac: ', det_g_jac)


            f_val = fun(xmap[0],xmap[1],xmap[2])
            r_el = r_el + det_g_jac * omega * f_val * phi_tab[0, i, :, 0]
            j_el = j_el + det_g_jac * omega * np.outer(phi_tab[0,i,:,0],phi_tab[0,i,:,0])

        # lagrange.base_transformations()
        # b_transformations = lagrange.base_transformations()
        # e_transformations = lagrange.entity_transformations()
        # print(lagrange.dof_transformations_are_identity)
        # print(lagrange.dof_transformations_are_permutations)
        # scattering dof
        dof_supports = list(gd2c2.successors(cell.id))
        dest = np.array([vertex_map.get(dof_s) for dof_s in dof_supports])

        c_sequ = cell_map[cell.id]
        for i, g_i in enumerate(dest):
            rg[g_i] += r_el[i]
            for j, g_j in enumerate(dest):
                row_l[c_sequ] = g_i
                col_l[c_sequ] = g_j
                data_l[c_sequ] = j_el[i,j]
                c_sequ = c_sequ + 1


        ako = 0

    row = np.array(row_l)
    col = np.array(col_l)
    data = np.array(data_l)
    jg = coo_matrix((data, (row, col)), shape=(n_dof_g, n_dof_g)).tocsr()

    # solving ls
    alpha = sp.linalg.spsolve(jg, rg)

    # writing solution on mesh points
    cell_0d_ids = [cell.id for cell in gmesh.cells if cell.dimension == 0]
    ph_data = np.zeros(len(gmesh.points))
    pe_data = np.zeros(len(gmesh.points))
    for id in cell_0d_ids:
        pr_ids = list(gd2c2.predecessors(id))
        cell = gmesh.cells[pr_ids[0]]
        if cell.dimension != 2:
            continue
        # scattering dof
        dof_supports = list(gd2c2.successors(cell.id))
        dest = np.array([vertex_map.get(dof_s) for dof_s in dof_supports])
        alpha_l = alpha[dest]

        cell_type = getattr(basix.CellType, "triangle")
        par_points = basix.geometry(cell_type)

        vertex_id = np.array([i for i, cid in enumerate(cell.cells_ids[0]) if cid == id])
        lagrange = basix.create_element(ElementFamily.P, CellType.triangle, k_order,
                                        LagrangeVariant.equispaced)

        points = par_points[vertex_id]
        phi_tab = lagrange.tabulate(0, points)

        # dof transformations
        aka = 0
        # lagrange.basic_transform()

        linear_base = basix.create_element(ElementFamily.P, CellType.triangle, 1,
                                           LagrangeVariant.equispaced)
        g_phi_tab = linear_base.tabulate(1, points)

        # evaluate mapping
        cell_points = gmesh.points[cell.node_tags]
        xmap = np.dot(g_phi_tab[0, 0, :, 0], cell_points)
        p_e = fun(xmap[0], xmap[1], xmap[2])
        p_h = np.dot(alpha_l, phi_tab[0, 0, :, 0])
        # print("p_e,p_h: ", [p_e,p_h])
        cell_0d = gmesh.cells[id]
        ph_data[cell_0d.node_tags] = p_h
        pe_data[cell_0d.node_tags] = p_e



    file_name = "h1_projector.xdmf"

    mesh_points = gmesh.points
    con_2d = np.array(
        [
            cell.node_tags
            for cell in gmesh.cells
            if cell.dimension == 2 and cell.id != None
        ]
    )
    cells_dict = {"triangle": con_2d}
    p_data_dict = {"ph": ph_data, "pe": pe_data}

    mesh = meshio.Mesh(
        points= mesh_points,
        cells = cells_dict,
        # Optionally provide extra data on points, cells, etc.
        point_data=p_data_dict
    )
    mesh.write("h1_projector.vtk")

    # this depends on pip install meshio[all] which is not working on macOS
    # time_data = [0.0]
    # with meshio.XdmfTimeSeriesWriter(file_name) as writer:
    #     writer.write_points_cells(mesh_points, cells_dict)
    #     for t in time_data:
    #         writer.write_data(t, point_data=p_data_dict)

    aka = 0

if __name__ == '__main__':
    main()



