
import numpy as np

from numpy import linalg as la

from shapely.geometry import LineString
from numba import njit, prange

import geometry.fracture_network as fn
import networkx as nx

import matplotlib.pyplot as plt

from geometry.geometry_cell import GeometryCell
from geometry.geometry_builder import GeometryBuilder
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh

import basix
from basix import ElementFamily, CellType, LagrangeVariant, LatticeType
import functools
from functools import partial

import scipy.sparse as sp
from scipy.sparse import coo_matrix

import matplotlib.colors as mcolors
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plot
import meshio
import itertools


import time
import sys

def polygon_polygon_intersection():

    fracture_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

    fracture_2 = np.array([[0.5, 0., 0.5], [0.5, 0., -0.5], [0.5, 1., -0.5], [0.5, 1., 0.5]])
    fracture_3 = np.array([[0., 0.5, -0.5], [1., 0.5, -0.5], [1., 0.5, 0.5],
     [0., 0.5, 0.5]])

    fracture_2 = np.array([[0.6, 0., 0.5], [0.6, 0., -0.5], [0.6, 1., -0.5], [0.6, 1., 0.5]])
    fracture_3 = np.array([[0.25, 0., 0.5], [0.914463, 0.241845, -0.207107], [0.572443, 1.18154, -0.207107],
     [-0.0920201, 0.939693, 0.5]])

    fractures = [fracture_1,fracture_2,fracture_3]

    fracture_network = fn.FractureNetwork(dimension=3)
    fracture_network.render_fractures(fractures)
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


class FiniteElement:

    def __init__(self, id, mesh, k_order, family, int_order = 0):
        self.mesh = mesh
        self.cell = mesh.cells[id]
        self.k_order = k_order
        self.discontinuous = discontinuous
        self.int_order = int_order
        self.family = family
        self.basis_variant = None
        self.basis_generator = None
        self.dof_ordering = None
        self.quadrature = None
        self.mapping = None
        self.phi = None
        self._build_structures()

    def _build_structures(self):
        self._set_integration_order()
        family = self._element_family()
        b_variant = self._element_variant()
        cell_type = self._element_type()
        self.basis_generator = basix.create_element(family, cell_type, self.k_order, b_variant, self.discontinuous)
        self.dof_ordering = self._dof_premutations()
        self.quadrature = basix.make_quadrature(basix.QuadratureType.gauss_jacobi, cell_type, self.int_order)
        self.evaluate_basis(self.quadrature[0],True)

    def _set_integration_order(self):
        if self.int_order == 0:
            self.int_order = 2 * self.k_order + 1


    def _element_variant(self):
        return LagrangeVariant.gll_centroid

    def _element_family(self):
        families = {"Lagrange": ElementFamily.P, "BDM": ElementFamily.BDM,
                    "RT": ElementFamily.RT, "N1E": ElementFamily.N1E,
                    "N2E": ElementFamily.N1E}
        family = self.family
        return families[family]

    def _element_type(self):
        element_types = {0: CellType.point, 1: CellType.interval, 2: CellType.triangle, 3: CellType.tetrahedron}
        dimension = self.cell.dimension
        return element_types[dimension]

    def _dof_premutations(self):
        # this permutation is because basix uses different entity ordering.
        if self.cell.dimension == 0:
            return self._dof_perm_point()
        elif self.cell.dimension == 1:
            return self._dof_perm_interval()
        elif self.cell.dimension == 2:
            return self._dof_perm_triangle()
        elif self.cell.dimension == 3:
            return self._dof_perm_triangle()

    def _dof_perm_point(self):
        indices = np.array([0], dtype=int)
        return indices

    def _dof_perm_interval(self):
        indices = np.array([], dtype=int)
        for dim, entity_dof in enumerate(self.basis_generator.entity_dofs):
            indices = np.append(indices, np.array(entity_dof, dtype=int))
        return indices.ravel()

    def _dof_perm_triangle(self):
        indices = np.array([], dtype=int)
        e_perms = np.array([1, 2, 0])
        for dim, entity_dof in enumerate(self.basis_generator.entity_dofs):
            if dim == 1:
                # for i, chunk in enumerate(entity_dof):
                #     if i == 2:
                #         chunk.reverse()
                entity_dof = [entity_dof[i] for i in e_perms]
                indices = np.append(indices, np.array(entity_dof, dtype=int))

            else:
                indices = np.append(indices, np.array(entity_dof, dtype=int))
        return indices.ravel()

    def _dof_perm_tetrahedron(self):

        raise ValueError("ERROR: perm_tetrahedron is not implemented.")

        indices = np.array([], dtype=int)
        e_perms = np.array([1, 2, 0])
        for dim, entity_dof in enumerate(self.basis_generator.entity_dofs):
            if dim == 1:
                for i, chunk in enumerate(entity_dof):
                    if i == 2:
                        chunk.reverse()
                entity_dof = [entity_dof[i] for i in e_perms]
                indices = np.append(indices, np.array(entity_dof, dtype=int))

            else:
                indices = np.append(indices, np.array(entity_dof, dtype=int))
        return indices.ravel()

    def compute_mapping(self, points, storage = False):
        cell_points = self.mesh.points[self.cell.node_tags]
        cell_type = self._element_type()
        linear_element = basix.create_element(ElementFamily.P, cell_type, 1, LagrangeVariant.equispaced)
        # points, weights = self.quadrature
        phi = linear_element.tabulate(1, points)
        dim = self.cell.dimension
        # Compute geometrical transformations
        x = []
        jac = []
        det_jac = []
        inv_jac = []

        for i, point in enumerate(points):

            # QR-decomposition is not unique
            xmap = np.dot(phi[0, i, :, 0], cell_points)
            grad_xmap = np.dot(phi[list(range(1,dim+1)), i, :, 0], cell_points).T
            q_axes, r_jac = np.linalg.qr(grad_xmap)
            det_g_jac = np.linalg.det(r_jac)

            # It's only unique up to the signs of the rows of R
            r_sign = np.diag(np.sign(np.diag(r_jac)), 0)
            q_axes = np.dot(q_axes, r_sign)
            r_jac = np.dot(r_sign, r_jac)
            det_g_jac = np.linalg.det(r_jac)
            if det_g_jac < 0.0:
                print('Negative det jac: ', det_g_jac)


            x.append(xmap)
            jac.append(grad_xmap)
            det_jac.append(det_g_jac)
            inv_jac.append(np.dot(np.linalg.inv(r_jac),q_axes.T))

        x = np.array(x)
        jac = np.array(jac)
        det_jac = np.array(det_jac)
        inv_jac = np.array(inv_jac)
        if storage:
            self.mapping = (x,jac,det_jac,inv_jac)
        return (x,jac,det_jac,inv_jac)

    def _validate_orientation(self):
        connectiviy = np.array([[0, 1], [1, 2], [2, 0]])
        e_perms = np.array([1, 2, 0])
        orientation = [False, False, False]
        for i, con in enumerate(connectiviy):
            edge = self.cell.node_tags[con]
            v_edge = self.mesh.cells[self.cell.sub_cells_ids[1][i]].node_tags
            if np.any(edge == v_edge):
                orientation[i] = True
        orientation = [orientation[i] for i in e_perms]
        return orientation

    def evaluate_basis(self, points, storage = False):

        (x, jac, det_jac, inv_jac) = self.compute_mapping(points, storage)
        # map functions
        phi_hat_tab = self.basis_generator.tabulate(1, points)
        phi_tab = self.basis_generator.push_forward(phi_hat_tab[0], jac, det_jac, inv_jac)

        # make functions outward
        if not self.basis_generator.dof_transformations_are_identity:
            n_dof = int(np.mean(self.basis_generator.num_entity_dofs[1]))
            for index in [0,2]:
                transformation = self.basis_generator.entity_transformations()["interval"][0]
                dofs = self.basis_generator.entity_dofs[1][index]
                for point in range(phi_tab.shape[0]):
                    for dim in range(phi_tab.shape[2]):
                        phi_tab[point, dofs, dim] = np.dot(transformation,
                                                           phi_tab[point, dofs, dim])

        # triangle ref connectivity
        if not self.basis_generator.dof_transformations_are_identity:
            oriented_q = self._validate_orientation()
            for index, check in enumerate(oriented_q):
                if check:
                    continue
                transformation = self.basis_generator.entity_transformations()["interval"][0]
                dofs = self.basis_generator.entity_dofs[1][index]
                for point in range(phi_tab.shape[0]):
                    for dim in range(phi_tab.shape[2]):
                        phi_tab[point, dofs, dim] = np.dot(transformation,phi_tab[point, dofs, dim])

        if storage:
            self.phi = phi_tab
        return phi_tab

class DoFMap:
    def __init__(self, mesh, conformity):
        self.mesh = mesh
        self.conformity_type = self.conformity_type(conformity)
        self.vertex_map = {}
        self.edge_map = {}
        self.face_map = {}
        self.volume_map = {}

    def build_entity_maps(self):
        dim = self.mesh.dimension
        self.d2_cd1_map = self.mesh.build_graph(dim, dim-1)
        self.d2_cd2_map = self.mesh.build_graph(dim, dim)

    @staticmethod
    def conformity_type(dimension):
        types = {'l-2': 0, 'h-1': 1, 'h-div': 2, 'h-curl': 3}
        return types[dimension]

def h1_projector(gmesh):

    # polynomial order
    k_order = 3

    # Create conformity
    st = time.time()
    # Entities by codimension
    # https://defelement.com/ciarlet.html
    dim = gmesh.dimension
    # collect maps
    codimension_maps = []
    for d in range(dim):
        codimension_maps.append(gmesh.build_graph(dim, dim-d))
    gd2c2 = codimension_maps[0]
    gd2c1 = codimension_maps[1]

    # Collecting ids
    entities_ids = {}
    for d in range(dim):
        entities_ids[d] = [id for id in list(codimension_maps[d].nodes()) if gmesh.cells[id].dimension == d]
    entities_ids[dim] = [id for id in list(codimension_maps[dim-1].nodes()) if gmesh.cells[id].dimension == dim]

    vertices_ids = entities_ids[0]
    edges_ids = entities_ids[1]
    faces_ids = entities_ids[2]
    n_vertices = len(vertices_ids)
    n_edges = len(edges_ids)
    n_faces = len(faces_ids)

    et = time.time()
    elapsed_time = et - st
    print('Preprocessing I time:', elapsed_time, 'seconds')

    st = time.time()
    elements = list(map(partial(FiniteElement, mesh=gmesh, k_order=k_order, family="Lagrange"),faces_ids))
    et = time.time()
    elapsed_time = et - st
    n_d_cells = len(elements)
    print('Number of processed elements:', n_d_cells)
    print('Element construction time:', elapsed_time, 'seconds')


    st = time.time()
    conformity = "h-1"

    # H1 functionality

    # conformity needs to be defined
    # Computing cell_id -> local_size map

    # Assembler
    # Triplets data
    c_size = 0
    n_dof_g = 0
    cell_map = {}
    for element in elements:
        cell = element.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs)
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    row = np.zeros((c_size), dtype=np.int64)
    col = np.zeros((c_size), dtype=np.int64)
    data = np.zeros((c_size), dtype=np.float64)

    n_fields = 1
    b_variant = LagrangeVariant.gll_centroid
    ref_element = basix.create_element(ElementFamily.P, CellType.triangle, k_order, b_variant)
    entity_support = [n_vertices,n_edges,n_faces]
    entity_dofs = [0, 0, 0, 0]
    for dim, n_entity_dofs in enumerate(ref_element.num_entity_dofs):
        e_dofs = int(np.mean(n_entity_dofs))
        entity_dofs[dim] = e_dofs
        entity_support[dim] *= e_dofs

    # Enumerates DoF
    dof_indices = np.array([0, entity_support[0], entity_support[1], entity_support[2]])
    global_indices = np.add.accumulate(dof_indices)
    # Computing cell mappings
    vertex_map = dict(zip(vertices_ids, np.split(np.array(range(global_indices[0],global_indices[1])),len(vertices_ids))))
    edge_map = dict(zip(edges_ids, np.split(np.array(range(global_indices[1],global_indices[2])),len(edges_ids))))
    face_map = dict(zip(faces_ids, np.split(np.array(range(global_indices[2],global_indices[3])),len(faces_ids))))


    n_dof_g = sum(entity_support)
    rg = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    lagrange = basix.create_element(ElementFamily.P, CellType.triangle, k_order,
                                    b_variant)

    # fun = lambda x, y, z: 16 * x * (1.0 - x) * y * (1.0 - y)
    # fun = lambda x, y, z: x + y
    # fun = lambda x, y, z: x * (1.0 - x) + y * (1.0 - y)
    fun = lambda x, y, z: x * (1.0 - x) * x + y * (1.0 - y) * y
    # fun = lambda x, y, z: x * (1.0 - x) * x * x + y * (1.0 - y) * y * y
    # fun = lambda x, y, z: x * (1.0 - x) * x * x * x + y * (1.0 - y) * y * y * y
    et = time.time()
    elapsed_time = et - st
    print('Preprocessing II time:', elapsed_time, 'seconds')


    st = time.time()
    xd = [(gd2c2.successors(faces_ids[0]),gd2c1.successors(faces_ids[0])) for i in range(10)]
    et = time.time()
    elapsed_time = et - st
    print('Search in grahps  time:', elapsed_time, 'seconds')

    st = time.time()
    def scatter_el_data(element, fun, gd2c2, gd2c1, cell_map, row, col, data ):

        # compute basis
        points, weights = element.quadrature
        phi_tab = element.phi
        (x, jac, det_jac, inv_jac) = element.mapping

        cell = element.cell

        n_dof = element.phi.shape[2]
        js = (n_dof, n_dof)
        rs = (n_dof)
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # linear_base
        for i, omega in enumerate(weights):
            f_val = fun(x[i, 0], x[i, 1], x[i, 2])
            r_el = r_el + det_jac[i] * omega * f_val * phi_tab[i, :, 0]
            j_el = j_el + det_jac[i] * omega * np.outer(phi_tab[i, :, 0],
                                                        phi_tab[i, :, 0])

        # scattering dof
        dof_vertex_supports = list(gd2c2.successors(cell.id))
        dof_edge_supports = list(gd2c1.successors(cell.id))
        dest_vertex = np.array([vertex_map.get(dof_s) for dof_s in dof_vertex_supports],
                               dtype=int).ravel()
        dest_edge = np.array([edge_map.get(dof_s) for dof_s in dof_edge_supports],
                             dtype=int).ravel()
        dest_faces = np.array([face_map.get(cell.id)], dtype=int).ravel()
        dest = np.concatenate((dest_vertex, dest_edge, dest_faces))[element.dof_ordering]

        c_sequ = cell_map[cell.id]

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

    aka = 0
    [scatter_el_data(element, fun, gd2c2, gd2c1, cell_map, row, col, data) for element in elements]


    jg = coo_matrix((data, (row, col)), shape=(n_dof_g, n_dof_g)).tocsr()
    et = time.time()
    elapsed_time = et - st
    print('Assembly time:', elapsed_time, 'seconds')

    # solving ls
    st = time.time()
    alpha = sp.linalg.spsolve(jg, rg)
    et = time.time()
    elapsed_time = et - st
    print('Linear solver time:', elapsed_time, 'seconds')

    # Computing L2 error
    def compute_l2_error(element, gd2c2, gd2c1):
        l2_error = 0.0
        cell = element.cell
        # scattering dof
        dof_vertex_supports = list(gd2c2.successors(cell.id))
        dof_edge_supports = list(gd2c1.successors(cell.id))
        dest_vertex = np.array([vertex_map.get(dof_s) for dof_s in dof_vertex_supports],
                               dtype=int).ravel()
        dest_edge = np.array([edge_map.get(dof_s) for dof_s in dof_edge_supports],
                             dtype=int).ravel()
        dest_faces = np.array([face_map.get(cell.id)], dtype=int).ravel()
        dest = np.concatenate((dest_vertex, dest_edge, dest_faces))[element.dof_ordering]
        alpha_l = alpha[dest]

        (x, jac, det_jac, inv_jac) = element.mapping
        points, weights = element.quadrature
        phi_tab = element.phi
        for i, pt in enumerate(points):
            p_e = fun(x[i, 0], x[i, 1], x[i, 2])
            p_h = np.dot(alpha_l, phi_tab[i, :, 0])
            l2_error += det_jac[i] * weights[i] * (p_h - p_e) * (p_h - p_e)

        return l2_error

    st = time.time()
    error_vec = [compute_l2_error(element, gd2c2, gd2c1) for element in elements]
    et = time.time()
    elapsed_time = et - st
    print('L2-error time:', elapsed_time, 'seconds')

    l2_error = functools.reduce(lambda x, y: x + y, error_vec)
    print("L2-error: ",np.sqrt(l2_error))
    assert np.isclose(np.sqrt(l2_error), 0.0, atol=1.e-14)


    # post-process solution
    st = time.time()
    cellid_to_element = dict(zip(faces_ids, elements))
    # writing solution on mesh points
    cell_0d_ids = [cell.node_tags[0] for cell in gmesh.cells if cell.dimension == 0]
    ph_data = np.zeros(len(gmesh.points))
    pe_data = np.zeros(len(gmesh.points))
    for id in list(vertex_map.keys()):
        if not gd2c2.has_node(id):
            continue

        pr_ids = list(gd2c2.predecessors(id))
        # pr_ids = [id for id in pr_ids if gmesh.cells[id].dimension == 2]
        cell = gmesh.cells[pr_ids[0]]
        if cell.dimension != 2:
            continue

        element = cellid_to_element[pr_ids[0]]

        # scattering dof
        dof_vertex_supports = list(gd2c2.successors(cell.id))
        dof_edge_supports = list(gd2c1.successors(cell.id))
        dest_vertex = np.array([vertex_map.get(dof_s) for dof_s in dof_vertex_supports],dtype=int).ravel()
        dest_edge = np.array([edge_map.get(dof_s) for dof_s in dof_edge_supports],dtype=int).ravel()
        dest_faces = np.array([face_map.get(cell.id)],dtype=int).ravel()
        dest = np.concatenate((dest_vertex,dest_edge,dest_faces))[element.dof_ordering]
        alpha_l = alpha[dest]

        cell_type = getattr(basix.CellType, "triangle")
        par_points = basix.geometry(cell_type)

        target_node_id = gmesh.cells[id].node_tags[0]
        par_point_id = np.array([i for i, node_id in enumerate(cell.node_tags) if node_id == target_node_id])

        points = par_points[par_point_id]

        # evaluate mapping
        (x, jac, det_jac, inv_jac) = element.compute_mapping(points)
        phi_tab = element.evaluate_basis(points)
        p_e = fun(x[0,0], x[0,1], x[0,2])
        p_h = np.dot(alpha_l, phi_tab[0, :, 0])
        ph_data[target_node_id] = p_h
        pe_data[target_node_id] = p_e


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
    et = time.time()
    elapsed_time = et - st
    print('Post-processing time:', elapsed_time, 'seconds')

def matrix_plot(J, sparse_q = True):

    if sparse_q:
        plot.matshow(J.todense())
    else:
        plot.matshow(J)
    plot.colorbar(orientation="vertical")
    plot.set_cmap("seismic")
    plot.show()

def hdiv_projector(gmesh):


    # polynomial order
    k_order = 4

    # Create conformity
    st = time.time()
    # Entities by codimension
    # https://defelement.com/ciarlet.html
    dim = gmesh.dimension
    # collect maps
    codimension_maps = []
    for d in range(dim):
        codimension_maps.append(gmesh.build_graph(dim, dim-d))
    gd2c2 = codimension_maps[0]
    gd2c1 = codimension_maps[1]

    # Collecting ids
    entities_ids = {}
    for d in range(dim):
        entities_ids[d] = [id for id in list(codimension_maps[d].nodes()) if gmesh.cells[id].dimension == d]
    entities_ids[dim] = [id for id in list(codimension_maps[dim-1].nodes()) if gmesh.cells[id].dimension == dim]

    vertices_ids = entities_ids[0]
    edges_ids = entities_ids[1]
    faces_ids = entities_ids[2]
    n_vertices = len(vertices_ids)
    n_edges = len(edges_ids)
    n_faces = len(faces_ids)

    et = time.time()
    elapsed_time = et - st
    print('Preprocessing I time:', elapsed_time, 'seconds')

    st = time.time()
    elements = list(map(partial(FiniteElement, mesh=gmesh, k_order=k_order, family="BDM"),faces_ids))
    et = time.time()
    elapsed_time = et - st
    n_d_cells = len(elements)
    print('Number of processed elements:', n_d_cells)
    print('Element construction time:', elapsed_time, 'seconds')

    st = time.time()
    conformity = "h-div"


    # Assembler
    # Triplets data
    c_size = 0
    n_dof_g = 0
    cell_map = {}
    for element in elements:
        cell = element.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs)
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    row = np.zeros((c_size), dtype=np.int64)
    col = np.zeros((c_size), dtype=np.int64)
    data = np.zeros((c_size), dtype=np.float64)

    n_fields = 1
    b_variant = LagrangeVariant.gll_centroid
    ref_element = basix.create_element(ElementFamily.BDM, CellType.triangle, k_order, b_variant)

    entity_support = [n_vertices,n_edges,n_faces]
    entity_dofs = [0, 0, 0, 0]
    for dim, n_entity_dofs in enumerate(ref_element.num_entity_dofs):
        e_dofs = int(np.mean(n_entity_dofs))
        entity_dofs[dim] = e_dofs
        entity_support[dim] *= e_dofs

    # Enumerates DoF
    dof_indices = np.array([0, entity_support[0], entity_support[1], entity_support[2]])
    global_indices = np.add.accumulate(dof_indices)
    # Computing cell mappings
    vertex_map = dict(zip(vertices_ids, np.split(np.array(range(global_indices[0],global_indices[1])),len(vertices_ids))))
    edge_map = dict(zip(edges_ids, np.split(np.array(range(global_indices[1],global_indices[2])),len(edges_ids))))
    face_map = dict(zip(faces_ids, np.split(np.array(range(global_indices[2],global_indices[3])),len(faces_ids))))


    n_dof_g = sum(entity_support)
    rg = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    # fun = lambda x, y, z: np.array([y, -x, 0])
    fun = lambda x, y, z: np.array([y*(1-y)*y*y, -x*(1-x)*x*x, 0])
    et = time.time()
    elapsed_time = et - st
    print('Preprocessing II time:', elapsed_time, 'seconds')

    st = time.time()
    xd = [(gd2c2.successors(faces_ids[0]),gd2c1.successors(faces_ids[0])) for i in range(10)]
    et = time.time()
    elapsed_time = et - st
    print('Search in grahps  time:', elapsed_time, 'seconds')

    st = time.time()
    def scatter_el_data(element, fun, gd2c2, gd2c1, cell_map, row, col, data):

        # compute basis
        points, weights = element.quadrature
        phi_tab = element.phi
        (x, jac, det_jac, inv_jac) = element.mapping

        cell = element.cell

        n_dof = element.phi.shape[1]
        js = (n_dof, n_dof)
        rs = (n_dof)
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # linear_base
        for i, omega in enumerate(weights):
            f_val = fun(x[i, 0], x[i, 1], x[i, 2])
            r_el = r_el + det_jac[i] * omega * phi_tab[i, :, :]@f_val
            for d in range(3):
                j_el = j_el + det_jac[i] * omega * np.outer(phi_tab[i, :, d],
                                                        phi_tab[i, :, d])

        # scattering dof
        dof_vertex_supports = list(gd2c2.successors(cell.id))
        dof_edge_supports = list(gd2c1.successors(cell.id))
        dest_vertex = np.array([vertex_map.get(dof_s) for dof_s in dof_vertex_supports],
                               dtype=int).ravel()
        dest_edge = np.array([edge_map.get(dof_s) for dof_s in dof_edge_supports],
                             dtype=int).ravel()
        dest_faces = np.array([face_map.get(cell.id)], dtype=int).ravel()
        dest = np.concatenate((dest_vertex, dest_edge, dest_faces))[element.dof_ordering]

        c_sequ = cell_map[cell.id]

        # check local projection
        check_proj_q = False
        if check_proj_q:
            l2_error = 0.0
            al = np.linalg.solve(j_el, r_el)
            for i, pt in enumerate(points):
                u_e = fun(x[i, 0], x[i, 1], x[i, 2])
                u_h = np.dot(al, phi_tab[i, :, :])
                l2_error += det_jac[i] * weights[i] * np.dot(u_h - u_e , u_h - u_e)
            aka = 0

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

    aka = 0
    [scatter_el_data(element, fun, gd2c2, gd2c1, cell_map, row, col, data) for element in elements]


    jg = coo_matrix((data, (row, col)), shape=(n_dof_g, n_dof_g)).tocsr()
    et = time.time()
    elapsed_time = et - st
    print('Assembly time:', elapsed_time, 'seconds')

    # solving ls
    st = time.time()
    alpha = sp.linalg.spsolve(jg, rg)
    et = time.time()
    elapsed_time = et - st
    print('Linear solver time:', elapsed_time, 'seconds')

    # Computing L2 error
    def compute_l2_error(element, gd2c2, gd2c1):
        l2_error = 0.0
        cell = element.cell
        # scattering dof
        dof_vertex_supports = list(gd2c2.successors(cell.id))
        dof_edge_supports = list(gd2c1.successors(cell.id))
        dest_vertex = np.array([vertex_map.get(dof_s) for dof_s in dof_vertex_supports],
                               dtype=int).ravel()
        dest_edge = np.array([edge_map.get(dof_s) for dof_s in dof_edge_supports],
                             dtype=int).ravel()
        dest_faces = np.array([face_map.get(cell.id)], dtype=int).ravel()
        dest = np.concatenate((dest_vertex, dest_edge, dest_faces))[element.dof_ordering]
        alpha_l = alpha[dest]

        (x, jac, det_jac, inv_jac) = element.mapping
        points, weights = element.quadrature
        phi_tab = element.phi
        for i, pt in enumerate(points):
            u_e = fun(x[i, 0], x[i, 1], x[i, 2])
            u_h = np.dot(alpha_l, phi_tab[i, :, :])
            l2_error += det_jac[i]  * weights[i] * np.dot((u_h - u_e) , (u_h - u_e))

        return l2_error

    st = time.time()
    error_vec = [compute_l2_error(element, gd2c2, gd2c1) for element in elements]
    et = time.time()
    elapsed_time = et - st
    print('L2-error time:', elapsed_time, 'seconds')

    l2_error = functools.reduce(lambda x, y: x + y, error_vec)
    print("L2-error: ",np.sqrt(l2_error))
    assert np.isclose(np.sqrt(l2_error), 0.0, atol=1.e-14)


    # post-process solution
    st = time.time()
    cellid_to_element = dict(zip(faces_ids, elements))
    # writing solution on mesh points
    # cell_0d_ids = [cell.node_tags[0] for cell in gmesh.cells if cell.dimension == 0]
    uh_data = np.zeros((len(gmesh.points),3))
    ue_data = np.zeros((len(gmesh.points),3))
    for id in list(vertex_map.keys()):
        if not gd2c2.has_node(id):
            continue

        pr_ids = list(gd2c2.predecessors(id))
        # pr_ids = [id for id in pr_ids if gmesh.cells[id].dimension == 2]
        cell = gmesh.cells[pr_ids[0]]
        if cell.dimension != 2:
            continue

        element = cellid_to_element[pr_ids[0]]

        # scattering dof
        dof_vertex_supports = list(gd2c2.successors(cell.id))
        dof_edge_supports = list(gd2c1.successors(cell.id))
        dest_vertex = np.array([vertex_map.get(dof_s) for dof_s in dof_vertex_supports],dtype=int).ravel()
        dest_edge = np.array([edge_map.get(dof_s) for dof_s in dof_edge_supports],dtype=int).ravel()
        dest_faces = np.array([face_map.get(cell.id)],dtype=int).ravel()
        dest = np.concatenate((dest_vertex,dest_edge,dest_faces))[element.dof_ordering]
        alpha_l = alpha[dest]

        cell_type = getattr(basix.CellType, "triangle")
        par_points = basix.geometry(cell_type)

        target_node_id = gmesh.cells[id].node_tags[0]
        par_point_id = np.array([i for i, node_id in enumerate(cell.node_tags) if node_id == target_node_id])

        points = par_points[par_point_id]
        # points = np.array([np.mean(par_points,axis=0)])

        # evaluate mapping
        (x, jac, det_jac, inv_jac) = element.compute_mapping(points)
        phi_tab = element.evaluate_basis(points)
        u_e = fun(x[0,0], x[0,1], x[0,2])
        u_h = np.dot(alpha_l, phi_tab[0, :, :])
        # aka = 0
        ue_data[target_node_id] = u_e
        uh_data[target_node_id] = u_h


    mesh_points = gmesh.points
    con_2d = np.array(
        [
            cell.node_tags
            for cell in gmesh.cells
            if cell.dimension == 2 and cell.id != None
        ]
    )
    cells_dict = {"triangle": con_2d}
    u_data_dict = {"uh": uh_data, "ue": ue_data}

    mesh = meshio.Mesh(
        points= mesh_points,
        cells = cells_dict,
        # Optionally provide extra data on points, cells, etc.
        point_data=u_data_dict
    )
    mesh.write("hdiv_projector.vtk")
    et = time.time()
    elapsed_time = et - st
    print('Post-processing time:', elapsed_time, 'seconds')

def l2_projector(gmesh):

    # polynomial order
    k_order = 3

    # basix.ufl.convert_ufl_element(e1)

    # Create conformity
    st = time.time()
    # Entities by codimension
    # https://defelement.com/ciarlet.html
    dim = gmesh.dimension
    # collect maps
    codimension_maps = []
    for d in range(dim):
        codimension_maps.append(gmesh.build_graph(dim, dim-d))
    gd2c2 = codimension_maps[0]
    gd2c1 = codimension_maps[1]

    # Collecting ids
    entities_ids = {}
    for d in range(dim):
        entities_ids[d] = [id for id in list(codimension_maps[d].nodes()) if gmesh.cells[id].dimension == d]
    entities_ids[dim] = [id for id in list(codimension_maps[dim-1].nodes()) if gmesh.cells[id].dimension == dim]

    vertices_ids = entities_ids[0]
    edges_ids = entities_ids[1]
    faces_ids = entities_ids[2]
    n_vertices = len(vertices_ids)
    n_edges = len(edges_ids)
    n_faces = len(faces_ids)

    et = time.time()
    elapsed_time = et - st
    print('Preprocessing I time:', elapsed_time, 'seconds')

    st = time.time()
    elements = list(map(partial(FiniteElement, mesh=gmesh, k_order=k_order, family="Lagrange", discontinuous=True),faces_ids))
    et = time.time()
    elapsed_time = et - st
    n_d_cells = len(elements)
    print('Number of processed elements:', n_d_cells)
    print('Element construction time:', elapsed_time, 'seconds')


    st = time.time()
    conformity = "l-2"

    # L2 functionality

    # conformity needs to be defined
    # Computing cell_id -> local_size map

    # Assembler
    # Triplets data
    c_size = 0
    n_dof_g = 0
    cell_map = {}
    for element in elements:
        cell = element.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs)
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    row = np.zeros((c_size), dtype=np.int64)
    col = np.zeros((c_size), dtype=np.int64)
    data = np.zeros((c_size), dtype=np.float64)

    n_fields = 1
    b_variant = LagrangeVariant.gll_centroid
    ref_element = basix.create_element(ElementFamily.P, CellType.triangle, k_order, b_variant, True)
    entity_support = [n_vertices,n_edges,n_faces]
    entity_dofs = [0, 0, 0, 0]
    for dim, n_entity_dofs in enumerate(ref_element.num_entity_dofs):
        e_dofs = int(np.mean(n_entity_dofs))
        entity_dofs[dim] = e_dofs
        entity_support[dim] *= e_dofs

    # Enumerates DoF
    dof_indices = np.array([0, entity_support[0], entity_support[1], entity_support[2]])
    global_indices = np.add.accumulate(dof_indices)
    # Computing cell mappings
    vertex_map = dict(zip(vertices_ids, np.split(np.array(range(global_indices[0],global_indices[1])),len(vertices_ids))))
    edge_map = dict(zip(edges_ids, np.split(np.array(range(global_indices[1],global_indices[2])),len(edges_ids))))
    face_map = dict(zip(faces_ids, np.split(np.array(range(global_indices[2],global_indices[3])),len(faces_ids))))


    n_dof_g = sum(entity_support)
    rg = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    lagrange = basix.create_element(ElementFamily.P, CellType.triangle, k_order,
                                    b_variant)

    # fun = lambda x, y, z: 16 * x * (1.0 - x) * y * (1.0 - y)
    # fun = lambda x, y, z: x + y
    # fun = lambda x, y, z: x * (1.0 - x) + y * (1.0 - y)
    fun = lambda x, y, z: x * (1.0 - x) * x + y * (1.0 - y) * y
    # fun = lambda x, y, z: x * (1.0 - x) * x * x + y * (1.0 - y) * y * y
    # fun = lambda x, y, z: x * (1.0 - x) * x * x * x + y * (1.0 - y) * y * y * y
    et = time.time()
    elapsed_time = et - st
    print('Preprocessing II time:', elapsed_time, 'seconds')


    st = time.time()
    xd = [(gd2c2.successors(faces_ids[0]),gd2c1.successors(faces_ids[0])) for i in range(10)]
    et = time.time()
    elapsed_time = et - st
    print('Search in grahps  time:', elapsed_time, 'seconds')

    st = time.time()
    def scatter_el_data(element, fun, gd2c2, gd2c1, cell_map, row, col, data ):

        # compute basis
        points, weights = element.quadrature
        phi_tab = element.phi
        (x, jac, det_jac, inv_jac) = element.mapping

        cell = element.cell

        n_dof = element.phi.shape[2]
        js = (n_dof, n_dof)
        rs = (n_dof)
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # linear_base
        for i, omega in enumerate(weights):
            f_val = fun(x[i, 0], x[i, 1], x[i, 2])
            r_el = r_el + det_jac[i] * omega * f_val * phi_tab[i, :, 0]
            j_el = j_el + det_jac[i] * omega * np.outer(phi_tab[i, :, 0],
                                                        phi_tab[i, :, 0])

        # scattering dof
        dof_vertex_supports = list(gd2c2.successors(cell.id))
        dof_edge_supports = list(gd2c1.successors(cell.id))
        dest_vertex = np.array([vertex_map.get(dof_s) for dof_s in dof_vertex_supports],
                               dtype=int).ravel()
        dest_edge = np.array([edge_map.get(dof_s) for dof_s in dof_edge_supports],
                             dtype=int).ravel()
        dest_faces = np.array([face_map.get(cell.id)], dtype=int).ravel()
        dest = np.concatenate((dest_vertex, dest_edge, dest_faces))[element.dof_ordering]

        c_sequ = cell_map[cell.id]

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

    aka = 0
    [scatter_el_data(element, fun, gd2c2, gd2c1, cell_map, row, col, data) for element in elements]


    jg = coo_matrix((data, (row, col)), shape=(n_dof_g, n_dof_g)).tocsr()
    et = time.time()
    elapsed_time = et - st
    print('Assembly time:', elapsed_time, 'seconds')

    # solving ls
    st = time.time()
    alpha = sp.linalg.spsolve(jg, rg)
    et = time.time()
    elapsed_time = et - st
    print('Linear solver time:', elapsed_time, 'seconds')

    # Computing L2 error
    def compute_l2_error(element, gd2c2, gd2c1):
        l2_error = 0.0
        cell = element.cell
        # scattering dof
        dof_vertex_supports = list(gd2c2.successors(cell.id))
        dof_edge_supports = list(gd2c1.successors(cell.id))
        dest_vertex = np.array([vertex_map.get(dof_s) for dof_s in dof_vertex_supports],
                               dtype=int).ravel()
        dest_edge = np.array([edge_map.get(dof_s) for dof_s in dof_edge_supports],
                             dtype=int).ravel()
        dest_faces = np.array([face_map.get(cell.id)], dtype=int).ravel()
        dest = np.concatenate((dest_vertex, dest_edge, dest_faces))[element.dof_ordering]
        alpha_l = alpha[dest]

        (x, jac, det_jac, inv_jac) = element.mapping
        points, weights = element.quadrature
        phi_tab = element.phi
        for i, pt in enumerate(points):
            p_e = fun(x[i, 0], x[i, 1], x[i, 2])
            p_h = np.dot(alpha_l, phi_tab[i, :, 0])
            l2_error += det_jac[i] * weights[i] * (p_h - p_e) * (p_h - p_e)

        return l2_error

    st = time.time()
    error_vec = [compute_l2_error(element, gd2c2, gd2c1) for element in elements]
    et = time.time()
    elapsed_time = et - st
    print('L2-error time:', elapsed_time, 'seconds')

    l2_error = functools.reduce(lambda x, y: x + y, error_vec)
    print("L2-error: ",np.sqrt(l2_error))
    assert np.isclose(np.sqrt(l2_error), 0.0, atol=1.e-14)

    # post-process solution
    st = time.time()
    cellid_to_element = dict(zip(faces_ids, elements))
    # writing solution on mesh points
    cell_0d_ids = [cell.node_tags[0] for cell in gmesh.cells if cell.dimension == 0]
    ph_data = np.zeros(len(gmesh.points))
    pe_data = np.zeros(len(gmesh.points))
    for id in list(vertex_map.keys()):
        if not gd2c2.has_node(id):
            continue

        pr_ids = list(gd2c2.predecessors(id))
        # pr_ids = [id for id in pr_ids if gmesh.cells[id].dimension == 2]
        cell = gmesh.cells[pr_ids[0]]
        if cell.dimension != 2:
            continue

        element = cellid_to_element[pr_ids[0]]

        # scattering dof
        dof_vertex_supports = list(gd2c2.successors(cell.id))
        dof_edge_supports = list(gd2c1.successors(cell.id))
        dest_vertex = np.array([vertex_map.get(dof_s) for dof_s in dof_vertex_supports],dtype=int).ravel()
        dest_edge = np.array([edge_map.get(dof_s) for dof_s in dof_edge_supports],dtype=int).ravel()
        dest_faces = np.array([face_map.get(cell.id)],dtype=int).ravel()
        dest = np.concatenate((dest_vertex,dest_edge,dest_faces))[element.dof_ordering]
        alpha_l = alpha[dest]

        cell_type = getattr(basix.CellType, "triangle")
        par_points = basix.geometry(cell_type)

        target_node_id = gmesh.cells[id].node_tags[0]
        par_point_id = np.array([i for i, node_id in enumerate(cell.node_tags) if node_id == target_node_id])

        points = par_points[par_point_id]

        # evaluate mapping
        (x, jac, det_jac, inv_jac) = element.compute_mapping(points)
        phi_tab = element.evaluate_basis(points)
        p_e = fun(x[0,0], x[0,1], x[0,2])
        p_h = np.dot(alpha_l, phi_tab[0, :, 0])
        ph_data[target_node_id] = p_h
        pe_data[target_node_id] = p_e


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
    mesh.write("l2_projector.vtk")
    et = time.time()
    elapsed_time = et - st
    print('Post-processing time:', elapsed_time, 'seconds')


def generate_mesh():

    # higher dimension domain geometry
    s = 1.0
    box_points = s * np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    g_builder = GeometryBuilder(dimension=2)
    g_builder.build_box_2D(box_points)
    g_builder.build_grahp()

    gmesh = None
    fractures_q = True
    if fractures_q:
        # polygon_polygon_intersection()
        h_cell = 1.0/4.0
        fracture_tags = [0,1,2,3,4,5]
        fracture_1 = np.array([[0.5, 0.2], [0.5, 0.8]])
        fracture_1 = np.array([[0.5, 0.4], [0.5, 0.6]])
        fracture_2 = np.array([[0.25, 0.5], [0.75, 0.5]])
        fracture_3 = np.array([[0.2, 0.35], [0.85, 0.35]])
        fracture_4 = np.array([[0.15, 0.15], [0.85, 0.85]])
        fracture_5 = np.array([[0.15, 0.85], [0.85, 0.15]])
        fracture_6 = np.array([[0.22, 0.62], [0.92, 0.22]])
        disjoint_fractures = [fracture_1, fracture_2, fracture_3, fracture_4, fracture_5, fracture_6]

        mesher = ConformalMesher(dimension=2)
        mesher.set_geometry_builder(g_builder)
        fractures = []
        for tag in fracture_tags:
            fractures.append(disjoint_fractures[tag])
        fracture_network = fn.FractureNetwork(dimension=2, physical_tag_shift=10)
        fracture_network.intersect_1D_fractures(fractures, render_intersection_q = False)
        fracture_network.build_grahp(all_fixed_d_cells_q=True)
        # mesher.set_fracture_network(fracture_network)
        mesher.set_points()
        mesher.generate(h_cell)
        mesher.write_mesh("gmesh.msh")

        gmesh = Mesh(dimension=2, file_name="gmesh.msh")
        gmesh.set_conformal_mesher(mesher)
        gmesh.build_conformal_mesh_II()
        # map_fracs_edge = gmesh.cut_conformity_on_fractures_mds_ec()
        # factor = 0.025
        # gmesh.apply_visual_opening(map_fracs_edge, factor)


        gmesh.write_data()
        gmesh.write_vtk()
        # print("Skin boundary is closed Q:", gmesh.circulate_internal_bc())
        print("h-size: ", h_cell)
    else:
        # polygon_polygon_intersection()
        h_cell = 1.0 / (1.0)
        mesher = ConformalMesher(dimension=2)
        mesher.set_geometry_builder(g_builder)
        mesher.set_points()
        mesher.generate(h_cell)
        mesher.write_mesh("gmesh.msh")

        gmesh = Mesh(dimension=2, file_name="gmesh.msh")
        gmesh.set_conformal_mesher(mesher)
        gmesh.build_conformal_mesh_II()
        gmesh.write_data()
        gmesh.write_vtk()
        print("h-size: ", h_cell)

    return gmesh


def main():

    gmesh = generate_mesh()
    # pojectors
    h1_projector(gmesh)
    hdiv_projector(gmesh)
    l2_projector(gmesh)


if __name__ == '__main__':
    main()




