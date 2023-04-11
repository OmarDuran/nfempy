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
from topology.mesh_topology import MeshTopology
from basis.finite_element import FiniteElement

import basix
from basix import ElementFamily, CellType, LagrangeVariant, LatticeType
import functools
from functools import partial
from itertools import permutations
from functools import reduce

# import operator

import scipy.sparse as sp
from scipy.sparse import coo_matrix
import pypardiso as sp_solver

import matplotlib.colors as mcolors

# import matplotlib.pyplot as plt
import matplotlib.pyplot as plot
import meshio
import itertools


import time
import sys


def polygon_polygon_intersection():

    fracture_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

    fracture_2 = np.array(
        [[0.5, 0.0, 0.5], [0.5, 0.0, -0.5], [0.5, 1.0, -0.5], [0.5, 1.0, 0.5]]
    )
    fracture_3 = np.array(
        [[0.0, 0.5, -0.5], [1.0, 0.5, -0.5], [1.0, 0.5, 0.5], [0.0, 0.5, 0.5]]
    )

    fracture_2 = np.array(
        [[0.6, 0.0, 0.5], [0.6, 0.0, -0.5], [0.6, 1.0, -0.5], [0.6, 1.0, 0.5]]
    )
    fracture_3 = np.array(
        [
            [0.25, 0.0, 0.5],
            [0.914463, 0.241845, -0.207107],
            [0.572443, 1.18154, -0.207107],
            [-0.0920201, 0.939693, 0.5],
        ]
    )

    fractures = [fracture_1, fracture_2, fracture_3]

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
        ElementFamily.P, CellType.triangle, 2, LagrangeVariant.equispaced
    )
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
        ElementFamily.P, CellType.triangle, 2, LagrangeVariant.equispaced
    )
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

    points = basix.create_lattice(CellType.tetrahedron, 3, LatticeType.equispaced, True)

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
            data[0, point, dofs, dim] = np.dot(
                transformation, data[0, point, dofs, dim]
            )

    print(data)


def examples_fiat():

    points, weights = basix.make_quadrature(
        basix.QuadratureType.gauss_jacobi, CellType.triangle, 3
    )

    lagrange = basix.create_element(
        ElementFamily.P, CellType.triangle, 3, LagrangeVariant.equispaced
    )

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
    plt.matshow(A.todense(), norm=norm, cmap="RdBu_r")
    plt.colorbar()
    plt.show()


class DoFMap:

    def __init__(self, mesh_topology, family, element_type, k_order, basis_variant, discontinuous=False):
        self.mesh_topology = mesh_topology
        self.ref_element = basix.create_element(family, element_type, k_order, basis_variant, discontinuous)
        self.vertex_map = {}
        self.edge_map = {}
        self.face_map = {}
        self.volume_map = {}
        self.n_dof = 0

    def build_entity_maps(self, n_dof_shift = 0):

        dim = self.mesh_topology.mesh.dimension
        n_fields = 1
        vertex_ids = []
        edge_ids = []
        face_ids = []
        volume_ids = []

        if dim == 1:
            vertex_ids = self.mesh_topology.entities_by_codimension(1)
            edge_ids = self.mesh_topology.entities_by_codimension(0)
        elif dim == 2:
            vertex_ids = self.mesh_topology.entities_by_codimension(2)
            edge_ids = self.mesh_topology.entities_by_codimension(1)
            face_ids = self.mesh_topology.entities_by_codimension(0)
        elif dim == 3:
            vertex_ids = self.mesh_topology.entities_by_codimension(3)
            edge_ids = self.mesh_topology.entities_by_codimension(2)
            face_ids = self.mesh_topology.entities_by_codimension(1)
            volume_ids = self.mesh_topology.entities_by_codimension(0)
        else:
            raise ValueError("Case not implemented for dimension: " % dim)

        n_vertices = len(vertex_ids)
        n_edges = len(edge_ids)
        n_faces = len(face_ids)
        n_volumes = len(volume_ids)

        entity_support = [n_vertices, n_edges, n_faces, n_volumes]
        for dim, n_entity_dofs in enumerate(self.ref_element.num_entity_dofs):
            e_dofs = int(np.mean(n_entity_dofs))
            entity_support[dim] *= e_dofs * n_fields

        # Enumerates DoF
        dof_indices = np.array(
            [0, entity_support[0], entity_support[1], entity_support[2],
             entity_support[3]]
        )
        global_indices = np.add.accumulate(dof_indices)
        global_indices += n_dof_shift
        # Computing cell mappings
        if len(vertex_ids) != 0:
            self.vertex_map = dict(
                zip(
                    vertex_ids,
                    np.split(
                        np.array(range(global_indices[0], global_indices[1])),
                        len(vertex_ids)
                    ),
                )
            )

        if len(edge_ids) != 0:
            self.edge_map = dict(
                zip(
                    edge_ids,
                    np.split(
                        np.array(range(global_indices[1], global_indices[2])), len(edge_ids)
                    ),
                )
            )

        if len(face_ids) != 0:
            self.face_map = dict(
                zip(
                    face_ids,
                    np.split(
                        np.array(range(global_indices[2], global_indices[3])), len(face_ids)
                    ),
                )
            )

        if len(volume_ids) != 0:
            self.volume_map = dict(
                zip(
                    volume_ids,
                    np.split(
                        np.array(range(global_indices[3], global_indices[4])),
                        len(volume_ids)
                    ),
                )
            )
        self.n_dof = sum(entity_support)

    def dof_number(self):
        return self.n_dof

    def destination_indices(self, cell_id):

        dim = self.mesh_topology.mesh.dimension
        entity_maps = [self.vertex_map,self.edge_map,self.face_map,self.volume_map]
        dest_by_dim = []
        for d in range(dim+1):
            entity_map = self.mesh_topology.entity_map_by_dimension(d)
            dof_supports = list(entity_map.successors(cell_id))
            entity_dest = np.array(
                [entity_maps[d].get(dof_s) for dof_s in dof_supports], dtype=int
            ).ravel()
            dest_by_dim.append(entity_dest)
        dest = np.concatenate(dest_by_dim)
        return dest

        # gd2c2 = self.mesh_topology.entity_map_by_codimension(2)
        # gd2c1 = mesh_topology.entity_map_by_codimension(1)
        # gd2c0 = mesh_topology.entity_map_by_codimension(0)
        #
        # dof_vertex_supports = list(gd3c3.successors(cell_id))
        # dof_edge_supports = list(gd3c2.successors(cell_id))
        # dof_face_supports = list(gd3c1.successors(cell_id))
        # dof_volume_supports = list(gd3c1.successors(cell_id))
        #
        # dest_vertex = np.array(
        #     [self.vertex_map.get(dof_s) for dof_s in dof_vertex_supports], dtype=int
        # ).ravel()
        # dest_edge = np.array(
        #     [self.edge_map.get(dof_s) for dof_s in dof_edge_supports], dtype=int
        # ).ravel()
        # dest_face = np.array(
        #     [self.face_map.get(dof_s) for dof_s in dof_face_supports], dtype=int
        # ).ravel()
        # dest_volume = np.array(
        #     [self.volume_map.get(dof_s) for dof_s in dof_volume_supports], dtype=int
        # ).ravel()
        # dest = np.concatenate((dest_vertex, dest_edge, dest_face, dest_volume))
        # return dest



def h1_projector(gmesh):

    # FESpace: data
    # polynomial order
    dim = gmesh.dimension
    conformity = "h-1"
    discontinuous = False
    k_order = 5
    family = "BDM"
    element_type = FiniteElement.type_by_dimension(dim)
    basis_family = FiniteElement.basis_family(family)
    basis_variant = FiniteElement.basis_variant()

    # scalar
    # fun = lambda x, y, z: 16 * x * (1.0 - x) * y * (1.0 - y)
    # fun = lambda x, y, z: x + y
    # fun = lambda x, y, z: x * (1.0 - x) + y * (1.0 - y)
    # fun = lambda x, y, z: x * (1.0 - x) * x + y * (1.0 - y) * y
    # fun = lambda x, y, z: x * (1.0 - x) * x * x + y * (1.0 - y) * y * y
    # fun = lambda x, y, z: x * (1.0 - x) * x * x * x + y * (1.0 - y) * y * y * y

    # vectorial
    # fun = lambda x, y, z: np.array([y, -x, -z])
    fun = lambda x, y, z: np.array([y * (1 - y), -x * (1 - x), -z * (1 - z)])
    # fun = lambda x, y, z: np.array([y * (1 - y) * y * y, -x * (1 - x) * x * x, -z*(1-z)*z*z])


    st = time.time()
    # Entities by codimension
    # https://defelement.com/ciarlet.html
    mesh_topology = MeshTopology(gmesh)
    cell_ids = mesh_topology.entities_by_codimension(0)

    et = time.time()
    elapsed_time = et - st
    print("Preprocessing I time:", elapsed_time, "seconds")

    st = time.time()
    elements = list(
        map(
            partial(FiniteElement, mesh=gmesh, k_order=k_order, family=family, discontinuous=discontinuous),
            cell_ids,
        )
    )
    et = time.time()
    elapsed_time = et - st
    n_d_cells = len(elements)
    print("Number of processed elements:", n_d_cells)
    print("Element construction time:", elapsed_time, "seconds")

    st = time.time()
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

    # DoF map for a variable supported on the element type
    dof_map = DoFMap(mesh_topology,basis_family,element_type,k_order,basis_variant,discontinuous=discontinuous)
    dof_map.build_entity_maps()
    n_dof_g = dof_map.dof_number()

    rg = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    et = time.time()
    elapsed_time = et - st
    print("Preprocessing II time:", elapsed_time, "seconds")

    st = time.time()
    def scatter_el_data(element, fun, dof_map, cell_map, row, col, data):

        cell = element.cell
        points, weights = element.quadrature
        phi_tab = element.phi
        (x, jac, det_jac, inv_jac) = element.mapping

        n_dof = element.phi.shape[2]
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # linear_base
        for i, omega in enumerate(weights):
            f_val = fun(x[i, 0], x[i, 1], x[i, 2])
            r_el = r_el + det_jac[i] * omega * f_val * phi_tab[i, :, 0]
            j_el = j_el + det_jac[i] * omega * np.outer(
                phi_tab[i, :, 0], phi_tab[i, :, 0]
            )

        # scattering dof
        dest = dof_map.destination_indices(cell.id)
        dest = dest[element.dof_ordering]

        c_sequ = cell_map[cell.id]

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

    [
        scatter_el_data(element, fun, dof_map, cell_map, row, col, data)
        for element in elements
    ]

    jg = coo_matrix((data, (row, col)), shape=(n_dof_g, n_dof_g)).tocsr()
    et = time.time()
    elapsed_time = et - st
    print("Assembly time:", elapsed_time, "seconds")

    # solving ls
    st = time.time()
    alpha = sp.linalg.spsolve(jg, rg)
    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    # Computing L2 error
    def compute_l2_error(element, dof_map):
        l2_error = 0.0
        cell = element.cell
        # scattering dof
        dest = dof_map.destination_indices(cell.id)
        dest = dest[element.dof_ordering]
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
    error_vec = [compute_l2_error(element, dof_map) for element in elements]
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")

    l2_error = functools.reduce(lambda x, y: x + y, error_vec)
    print("L2-error: ", np.sqrt(l2_error))
    assert np.isclose(np.sqrt(l2_error), 0.0, atol=1.0e-14)

    # post-process solution
    st = time.time()
    cellid_to_element = dict(zip(cell_ids, elements))
    # writing solution on mesh points
    ph_data = np.zeros(len(gmesh.points))
    pe_data = np.zeros(len(gmesh.points))
    vertices = mesh_topology.entities_dimension(0)
    cell_vertex_map = mesh_topology.entity_map_by_dimension(0)
    for id in vertices:
        if not cell_vertex_map.has_node(id):
            continue

        pr_ids = list(cell_vertex_map.predecessors(id))
        # pr_ids = [id for id in pr_ids if gmesh.cells[id].dimension == 2]
        cell = gmesh.cells[pr_ids[0]]
        if cell.dimension != gmesh.dimension:
            continue

        element = cellid_to_element[pr_ids[0]]

        # scattering dof
        dest = dof_map.destination_indices(cell.id)
        dest = dest[element.dof_ordering]
        alpha_l = alpha[dest]

        par_points = basix.geometry(element_type)

        target_node_id = gmesh.cells[id].node_tags[0]
        par_point_id = np.array(
            [i for i, node_id in enumerate(cell.node_tags) if node_id == target_node_id]
        )

        points = par_points[par_point_id]

        # evaluate mapping
        (x, jac, det_jac, inv_jac) = element.compute_mapping(points)
        phi_tab = element.evaluate_basis(points)
        p_e = fun(x[0, 0], x[0, 1], x[0, 2])
        p_h = np.dot(alpha_l, phi_tab[0, :, 0])
        ph_data[target_node_id] = p_h
        pe_data[target_node_id] = p_e

    mesh_points = gmesh.points
    con_d = np.array(
        [
            cell.node_tags
            for cell in gmesh.cells
            if cell.dimension == gmesh.dimension and cell.id != None and cell.material_id != None
        ]
    )
    meshio_cell_types = { 0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
    cells_dict = {meshio_cell_types[gmesh.dimension]: con_d}
    p_data_dict = {"ph": ph_data, "pe": pe_data}

    mesh = meshio.Mesh(
        points=mesh_points,
        cells=cells_dict,
        # Optionally provide extra data on points, cells, etc.
        point_data=p_data_dict,
    )
    mesh.write("h1_projector.vtk")
    et = time.time()
    elapsed_time = et - st
    print("Post-processing time:", elapsed_time, "seconds")


def matrix_plot(J, sparse_q=True):

    if sparse_q:
        plot.matshow(J.todense())
    else:
        plot.matshow(J)
    plot.colorbar(orientation="vertical")
    plot.set_cmap("seismic")
    plot.show()


def hdiv_projector(gmesh):

    # FESpace: data
    # polynomial order
    dim = gmesh.dimension
    conformity = "h-div"
    discontinuous = False
    k_order = 3
    family = "BDM"
    element_type = FiniteElement.type_by_dimension(dim)
    basis_family = FiniteElement.basis_family(family)
    basis_variant = FiniteElement.basis_variant()

    # vectorial
    # fun = lambda x, y, z: np.array([y, -x, -z])
    # fun = lambda x, y, z: np.array([y * (1 - y), -x * (1 - x), -z * (1 - z)])
    fun = lambda x, y, z: np.array([y * (1 - y) *y, -x * (1 - x) *x, -z * (1 - z)* z])
    # fun = lambda x, y, z: np.array([y * (1 - y) * y * y, -x * (1 - x) * x * x, -z*(1-z)*z*z])


    st = time.time()
    # Entities by codimension
    # https://defelement.com/ciarlet.html
    mesh_topology = MeshTopology(gmesh)
    cell_ids = mesh_topology.entities_by_codimension(0)

    et = time.time()
    elapsed_time = et - st
    print("Preprocessing I time:", elapsed_time, "seconds")

    st = time.time()
    elements = list(
        map(
            partial(FiniteElement, mesh=gmesh, k_order=k_order, family=family, discontinuous=discontinuous),
            cell_ids,
        )
    )
    et = time.time()
    elapsed_time = et - st
    n_d_cells = len(elements)
    print("Number of processed elements:", n_d_cells)
    print("Element construction time:", elapsed_time, "seconds")

    st = time.time()
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

    # DoF map for a variable supported on the element type
    dof_map = DoFMap(mesh_topology,basis_family,element_type,k_order,basis_variant,discontinuous=discontinuous)
    dof_map.build_entity_maps()
    n_dof_g = dof_map.dof_number()

    rg = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    et = time.time()
    elapsed_time = et - st
    print("Preprocessing II time:", elapsed_time, "seconds")

    st = time.time()

    def scatter_el_data(element, fun, dof_map, cell_map, row, col, data):

        cell = element.cell
        points, weights = element.quadrature
        phi_tab = element.phi
        (x, jac, det_jac, inv_jac) = element.mapping

        n_dof = element.phi.shape[1]
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # linear_base
        for i, omega in enumerate(weights):
            f_val = fun(x[i, 0], x[i, 1], x[i, 2])
            r_el = r_el + det_jac[i] * omega * phi_tab[i, :, :] @ f_val
            for d in range(3):
                j_el = j_el + det_jac[i] * omega * np.outer(
                    phi_tab[i, :, d], phi_tab[i, :, d]
                )

        # scattering dof
        dest = dof_map.destination_indices(cell.id)
        dest = dest[element.dof_ordering]

        c_sequ = cell_map[cell.id]

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

    [
        scatter_el_data(element, fun, dof_map, cell_map, row, col, data)
        for element in elements
    ]

    jg = coo_matrix((data, (row, col)), shape=(n_dof_g, n_dof_g)).tocsr()
    et = time.time()
    elapsed_time = et - st
    print("Assembly time:", elapsed_time, "seconds")

    # solving ls
    st = time.time()
    alpha = sp.linalg.spsolve(jg, rg)
    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    # Computing L2 error
    def compute_l2_error(element, dof_map):
        l2_error = 0.0
        cell = element.cell
        # scattering dof
        dest = dof_map.destination_indices(cell.id)
        dest = dest[element.dof_ordering]
        alpha_l = alpha[dest]

        (x, jac, det_jac, inv_jac) = element.mapping
        points, weights = element.quadrature
        phi_tab = element.phi
        for i, pt in enumerate(points):
            u_e = fun(x[i, 0], x[i, 1], x[i, 2])
            u_h = np.dot(alpha_l, phi_tab[i, :, :])
            l2_error += det_jac[i] * weights[i] * np.dot((u_h - u_e), (u_h - u_e))

        return l2_error

    st = time.time()
    error_vec = [compute_l2_error(element, dof_map) for element in elements]
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")

    l2_error = functools.reduce(lambda x, y: x + y, error_vec)
    print("L2-error: ", np.sqrt(l2_error))
    assert np.isclose(np.sqrt(l2_error), 0.0, atol=1.0e-14)

    # post-process solution
    # writing solution on mesh points

    st = time.time()
    cellid_to_element = dict(zip(cell_ids, elements))
    uh_data = np.zeros((len(gmesh.points), 3))
    ue_data = np.zeros((len(gmesh.points), 3))

    vertices = mesh_topology.entities_dimension(0)
    cell_vertex_map = mesh_topology.entity_map_by_dimension(0)
    for id in vertices:
        if not cell_vertex_map.has_node(id):
            continue

        pr_ids = list(cell_vertex_map.predecessors(id))
        cell = gmesh.cells[pr_ids[0]]
        if cell.dimension != gmesh.dimension:
            continue

        element = cellid_to_element[pr_ids[0]]

        # scattering dof
        dest = dof_map.destination_indices(cell.id)
        dest = dest[element.dof_ordering]
        alpha_l = alpha[dest]

        par_points = basix.geometry(element_type)

        target_node_id = gmesh.cells[id].node_tags[0]
        par_point_id = np.array(
            [i for i, node_id in enumerate(cell.node_tags) if node_id == target_node_id]
        )

        points = par_points[par_point_id]

        # evaluate mapping
        (x, jac, det_jac, inv_jac) = element.compute_mapping(points)
        phi_tab = element.evaluate_basis(points)
        u_e = fun(x[0, 0], x[0, 1], x[0, 2])
        u_h = np.dot(alpha_l, phi_tab[0, :, :])

        ue_data[target_node_id] = u_e
        uh_data[target_node_id] = u_h

    mesh_points = gmesh.points
    con_d = np.array(
        [
            cell.node_tags
            for cell in gmesh.cells
            if cell.dimension == gmesh.dimension and cell.id != None and cell.material_id != None
        ]
    )
    meshio_cell_types = { 0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
    cells_dict = {meshio_cell_types[gmesh.dimension]: con_d}
    u_data_dict = {"uh": uh_data, "ue": ue_data}

    mesh = meshio.Mesh(
        points=mesh_points,
        cells=cells_dict,
        # Optionally provide extra data on points, cells, etc.
        point_data=u_data_dict,
    )
    mesh.write("hdiv_projector.vtk")
    et = time.time()
    elapsed_time = et - st
    print("Post-processing time:", elapsed_time, "seconds")


def generate_mesh():

    h_cell = 1.0 / (4.0)
    # higher dimension domain geometry
    s = 1.0

    box_points = s * np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    g_builder = GeometryBuilder(dimension=2)
    g_builder.build_box_2D(box_points)
    g_builder.build_grahp()

    gmesh = None
    fractures_q = False
    if fractures_q:
        # polygon_polygon_intersection()
        # h_cell = 1.0 / 4.0
        fracture_tags = [0, 1, 2, 3, 4, 5]
        fracture_1 = np.array([[0.5, 0.2], [0.5, 0.8]])
        fracture_1 = np.array([[0.5, 0.4], [0.5, 0.6]])
        fracture_2 = np.array([[0.25, 0.5], [0.75, 0.5]])
        fracture_3 = np.array([[0.2, 0.35], [0.85, 0.35]])
        fracture_4 = np.array([[0.15, 0.15], [0.85, 0.85]])
        fracture_5 = np.array([[0.15, 0.85], [0.85, 0.15]])
        fracture_6 = np.array([[0.22, 0.62], [0.92, 0.22]])
        disjoint_fractures = [
            fracture_1,
            fracture_2,
            fracture_3,
            fracture_4,
            fracture_5,
            fracture_6,
        ]

        mesher = ConformalMesher(dimension=2)
        mesher.set_geometry_builder(g_builder)
        fractures = []
        for tag in fracture_tags:
            fractures.append(disjoint_fractures[tag])
        fracture_network = fn.FractureNetwork(dimension=2, physical_tag_shift=10)
        fracture_network.intersect_1D_fractures(fractures, render_intersection_q=False)
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

        mesher = ConformalMesher(dimension=2)
        mesher.set_geometry_builder(g_builder)
        mesher.set_points()
        mesher.generate(h_cell)
        mesher.write_mesh("gmesh.msh")

        gmesh = Mesh(dimension=2, file_name="gmesh.msh")
        gmesh.set_conformal_mesher(mesher)
        gmesh.build_conformal_mesh_II()

        # gmesh.write_data()
        gmesh.write_vtk()
        print("h-size: ", h_cell)

    return gmesh

def generate_mesh_3d():

    h_cell = 1.0 / (4.0)

    # higher dimension domain geometry
    s = 1.0

    box_points = s * np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )
    g_builder = GeometryBuilder(dimension=3)
    g_builder.build_box(box_points)
    g_builder.build_grahp()

    mesher = ConformalMesher(dimension=3)
    mesher.set_geometry_builder(g_builder)
    mesher.set_points()
    mesher.generate(h_cell)
    mesher.write_mesh("gmesh.msh")

    gmesh = Mesh(dimension=3, file_name="gmesh.msh")
    gmesh.set_conformal_mesher(mesher)
    gmesh.build_conformal_mesh_II()

    # gmesh.write_data()
    gmesh.write_vtk()
    print("h-size: ", h_cell)


    return gmesh

def main():

    gmesh = generate_mesh()

    # # pojectors
    gmesh_3d = generate_mesh_3d()
    # h1_projector(gmesh)
    hdiv_projector(gmesh_3d)

    # l2_projector(gmesh)

    # gmesh_3d = generate_mesh_3d()
    # h1_projector_3d(gmesh_3d)
    # hdiv_projector_3d(gmesh_3d)
    # l2_projector_3d(gmesh_3d)


if __name__ == "__main__":
    main()
