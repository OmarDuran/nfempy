import numpy as np
from numpy import linalg as la
import quadpy
import gmsh

from shapely.geometry import LineString
import geometry.triangle_triangle_intersection_test as tt_intersector
import networkx as nx

import matplotlib.pyplot as plt

# Topology for Geometric Design
# It is required to find a better reference
#https://www.maruf.ca/files/caadoc/CAATobTechArticles/TopoConcepts.htm#Manifold

def cell_type(dimension):
    types = ("Vertex","Edge","Face","Volume")
    # types = ("0-cell", "1-cell", "2-cell", "3-cell")
    return types[dimension]

class cell:

    def __init__(self, dimension, id, point_id = None):

        self.dimension = dimension
        self.type = cell_type(dimension)
        self.id = id
        self.boundary_cells = np.array([],dtype=cell)
        self.immersed_cells = np.array([],dtype=cell)

        # A vertex has the point id attribute
        self.point_id = None
        if dimension == 0:
            if point_id is None:
                self.point_id = id
            else:
                self.point_id = point_id


class fracture_cell(cell):

    def __init__(self, dimension, id):
        super(fracture_cell, self).__init__(dimension, id)
        self.dimension = dimension
        self.type = cell_type(dimension)
        self.is_bc_cell = False
        self.boundary_cells = np.array([],dtype=cell)
        self.immersed_cells = np.array([],dtype=cell)


# geometry representation
class Network:

    # The network object represents a mesh of intersecting objects.
    # For 2d the mesh includes (d-f)-facets f = {0,1}:
    #   (0)-facets vertices
    #   (1)-facets edges

    # For 3d the mesh includes (d-f)-facets f = {0,1,2}:
    #   (0)-facets vertices
    #   (1)-facets edges
    #   (2)-facets faces

    def __init__(self, dimension, eps: float = 1.0e-12):

        self.eps = eps
        self.cells = np.array([], dtype=cell)
        self.points: np.array = None
        self.vertices = None
        self.dimension = dimension
        self.grahp = None
        self.connectivity = None
        self.fractures_indices = None
        self.fractures_bcs = None

    def render_fractures(
        self, fractures: np.array
    ):
        axes = plt.axes(projection="3d")
        axes.set_xlabel('$X$', fontsize=12)
        axes.set_ylabel('$Y$', fontsize=12)
        axes.set_zlabel('$Z$', fontsize=12)

        for fracture in fractures:
            ox = fracture[[0, 1, 2, 3, 0]].T[0]
            oy = fracture[[0, 1, 2, 3, 0]].T[1]
            oz = fracture[[0, 1, 2, 3, 0]].T[2]
            axes.plot(ox, oy, oz, color="green", marker="o", lw=2)

        plt.show()

    def insert_vertex_cell(self, point):
        R = self.points - point
        r_norm = la.norm(R, axis=1)
        index = np.where(r_norm < self.eps)
        vertex = None
        if index[0].size == 0:
            point_id = len(self.points)
            self.points = np.append(self.points, [point], axis=0)
            cell_id = len(self.cells)
            vertex = cell(0, cell_id, point_id)
            self.cells = np.append(self.cells, vertex)
        else:
            target_point_id = index[0][0]
            for cell_i in self.cells:
                if cell_i.dimension == 0 and cell_i.point_id == target_point_id:
                    vertex = cell_i
                    break
        return vertex

    def insert_fracture_cell(self, cell_id, fracture):

        self.points = np.append(self.points, np.array([point for point in fracture]), axis=0)
        loop = [i + cell_id for i in range(len(fracture))]
        self.cells = np.append(self.cells, np.array([cell(0, index) for index in loop]))

        loop.append(loop[0])
        connectivities = np.array([[loop[index], loop[index + 1]] for index in
                                   range(len(loop) - 1)])

        cell_id = cell_id + len(fracture)
        edges_indices = []
        for con in connectivities:
            edge = cell(1, cell_id)
            edge.boundary_cells = self.cells[con]
            self.cells = np.append(self.cells, edge)
            edges_indices.append(cell_id)
            cell_id = cell_id + 1

        edges_indices = np.array(edges_indices)
        surface = cell(2, cell_id)
        surface.boundary_cells = self.cells[edges_indices]
        self.cells = np.append(self.cells, surface)

        return cell_id

    def intersect_2D_fractures(self, fractures, render_intersection_q = False):

        self.cells = np.array([], dtype=cell)
        self.points = np.empty((0, 3), dtype=float)

        cell_id = 0
        n_fractures = len(fractures)
        for fracture in fractures:
            cell_id = self.insert_fracture_cell(cell_id, fracture)
            cell_id = cell_id + 1

        intersection_data = [[None for f in fractures] for f in fractures]

        cells_2d = [cell_i for cell_i in self.cells if cell_i.dimension == 2]
        for i, cell_i in enumerate(cells_2d):
            f_i = fractures[i]
            for j, cell_j in enumerate(cells_2d):
                f_j = fractures[j]
                if i >= j:
                    intersection_data[i][j] = (False, np.array, np.array)
                    continue

                obj = pp_intersector.PolygonPolygonIntersectionTest()
                intersection_q = obj.polygon_polygon_intersection(f_i, f_j, render_intersection_q)
                intersection_data[i][j] = intersection_q

        for i, cell_i in enumerate(cells_2d):

            i_results = [chunk[0] for chunk in intersection_data[i]]
            n_intersections = np.count_nonzero(i_results)

            if n_intersections > 0:

                p, q, r = fractures[i][[0, 1, 3]]
                dir_cross = np.cross(p - q, r - q)
                n_t = dir_cross / np.linalg.norm(dir_cross)
                drop = np.argmax(np.abs(n_t))
                pos = np.array(
                    list(
                        {
                            0,
                            1,
                            2,
                        }
                        - {drop}
                    )
                )

                fractures_1d = []
                for chunk in intersection_data[i]:
                    if chunk[0]:
                        fractures_1d.append(np.array([chunk[1], chunk[2]]))

                fracture_network = Network(dimension=2)
                fracture_network.intersect_1D_fractures(fractures_1d, pos,
                                                        render_intersection_q)
                # fracture_network.build_grahp()
                # fracture_network.draw_grahp()

                # insert cells
                point_id = len(self.points)
                cells_id = len(self.cells)
                self.points = np.append(self.points,
                                   np.array([point for point in fracture_network.points]),
                                   axis=0)

                for f_cell in fracture_network.cells:
                    f_cell.id = f_cell.id + cells_id
                    if f_cell.dimension == 0:
                        f_cell.point_id = f_cell.point_id + point_id
                    self.cells = np.append(self.cells, f_cell)

                f_cell_1d = []
                if n_intersections == 1:
                    f_cell_1d = [f_cell for f_cell in fracture_network.cells if f_cell.dimension == 1]
                else:
                    f_cell_1d = [f_cell for f_cell in fracture_network.cells if
                                 f_cell.dimension == 1 and len(f_cell.immersed_cells) > 0]

                cj = 0
                for j, chunk in enumerate(intersection_data[i]):
                    if chunk[0]:
                        f_cell_intersection = f_cell_1d[cj]
                        cell_j = cells_2d[j]
                        cell_i.immersed_cells = np.append(cell_i.immersed_cells, f_cell_intersection)
                        cell_j.immersed_cells = np.append(cell_j.immersed_cells, f_cell_intersection)
                        cj = cj + 1

    def intersect_1D_fractures(self, fractures, pos = np.array([0, 1]), render_intersection_q = False):

        self.cells = np.array([], dtype=cell)

        # build dfn disjoint geometrical description
        self.points = np.array([point for fracture in fractures for point in fracture])
        n_points = len(self.points)
        self.cells = np.append(self.cells,
                              np.array(
                                  [cell(0, i) for i, point in enumerate(self.points)]))

        connectivities = np.array([[index, index+1] for index in range(0,n_points,2)])

        # insert fracture cells
        cell_id = n_points
        for con in connectivities:
            edge = cell(1, cell_id)
            edge.boundary_cells = self.cells[con]
            self.cells = np.append(self.cells, edge)
            cell_id = cell_id + 1

        tt_test = tt_intersector.TriangleTriangleIntersectionTest()

        cells_1d = [cell_1d for cell_1d in self.cells if cell_1d.dimension == 1]
        for i, cell_i in enumerate(cells_1d):
            cell_i_bc_ids = [bc_cell.id for bc_cell in cell_i.boundary_cells]
            a, b = [self.points[id] for id in cell_i_bc_ids]
            for j, cell_j in enumerate(cells_1d):
                cell_j_bc_ids = [bc_cell.id for bc_cell in cell_j.boundary_cells]
                p, q = [self.points[id] for id in cell_j_bc_ids]
                if i >= j:
                    continue
                intersection_data = tt_test.line_line_intersection(a, b, p, q,
                                                                   pos,
                                                                   render_intersection_q)
                if intersection_data[0]:
                    point = intersection_data[1]
                    vertex = self.insert_vertex_cell(point)
                    cell_i.immersed_cells = np.append(cell_i.immersed_cells, vertex)
                    cell_j.immersed_cells = np.append(cell_j.immersed_cells, vertex)

        cell_id = len(self.cells)
        for cell_i in cells_1d:
            b, e = [bc_cell.point_id for bc_cell in cell_i.boundary_cells]
            i = [immersed_cell.point_id for immersed_cell in cell_i.immersed_cells]
            if len(i) > 0 :
                R = self.points[i] - self.points[b]
                r_norm = la.norm(R, axis=1)
                perm = np.argsort(r_norm)
                cell_indices = [cell_i.immersed_cells[k].id for k in perm.tolist()]
                cell_indices = list(dict.fromkeys(cell_indices))
                cell_indices.insert(0, b)
                cell_indices.append(e)
                connectivities = np.array([[cell_indices[index],cell_indices[index+1]] for index in range(len(cell_indices) - 1)])
                cell_i.immersed_cells = np.array([], dtype=cell)
                for con in connectivities:
                    edge = cell(1, cell_id)
                    edge.boundary_cells = self.cells[con]
                    self.cells = np.append(self.cells, edge)
                    cell_i.immersed_cells = np.append(cell_i.immersed_cells, edge)
                    cell_id = cell_id + 1


    def gather_edges(self, g_cell: cell, tuple_id_list):
        for bc_cell in g_cell.boundary_cells:
            tuple_id_list.append((g_cell.id, bc_cell.id))
            if bc_cell.dimension == 0:
                print("BC: Vertex with id: ", bc_cell.id)
            else:
                self.gather_edges(bc_cell, tuple_id_list)
        for immersed_cell in g_cell.immersed_cells:
            tuple_id_list.append((g_cell.id, immersed_cell.id))
            if immersed_cell.dimension == 0:
                print("IM: Vertex with id: ", immersed_cell.id)
            else:
                self.gather_edges(immersed_cell, tuple_id_list)

    def build_grahp(self):

        disjoint_cells = [cell_i for cell_i in self.cells if cell_i.dimension == self.dimension - 1 and len(cell_i.immersed_cells) != 0]
        tuple_id_list = []
        for cell_1d  in disjoint_cells:
            self.gather_edges(cell_1d, tuple_id_list)

        self.graph = nx.from_edgelist(tuple_id_list, create_using=nx.DiGraph)

    def draw_grahp(self):
        nx.draw(self.graph, pos=nx.circular_layout(self.graph), with_labels=True, node_color="skyblue")

class geometry_builder:

    def __init__(self, dimension):
        self.dimension = dimension

    def set_boundary(self, vertices, connectivity, material_id):
        self.vertices = vertices
        self.connectivity = connectivity
        self.material_id = material_id

    def build_internal_bc(self, Network, normal_expansion = 1.0e-1):
        assert Network.dimension == self.dimension, f"Geometry and network dimension are not equal {Network.dimension}"

        # classify intersections
        nodes = list(Network.grahp.nodes)
        node_neighs = [[] for _ in nodes]
        for i in range(len(nodes)):
            neighs = list(nx.all_neighbors(Network.grahp, nodes[i]))
            node_neighs[i].append(neighs)

        xd = 0


class Fracture:

    def __init__(self, vertices: np.array, connectivity: np.array, dimension):

        self.vertices = vertices
        self.connectivity = connectivity
        self.dimension = dimension
        self.boundary = connectivity
        self.normal: np.array = None
        self.tangent: np.array = None



import geometry.polygon_polygon_intersection_test as pp_intersector

def polygon_polygon_intersection():

    fracture_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

    fracture_2 = np.array([[0.5, 0., 0.5], [0.5, 0., -0.5], [0.5, 1., -0.5], [0.5, 1., 0.5]])
    fracture_3 = np.array([[0., 0.5, -0.5], [1., 0.5, -0.5], [1., 0.5, 0.5],
     [0., 0.5, 0.5]])

    fracture_2 = np.array([[0.6, 0., 0.5], [0.6, 0., -0.5], [0.6, 1., -0.5], [0.6, 1., 0.5]])
    fracture_3 = np.array([[0.25, 0., 0.5], [0.914463, 0.241845, -0.207107], [0.572443, 1.18154, -0.207107],
     [-0.0920201, 0.939693, 0.5]])

    fractures = [fracture_1,fracture_2,fracture_3]

    fracture_network = Network(dimension=3)
    fracture_network.render_fractures(fractures)
    fracture_network.intersect_2D_fractures(fractures, True)
    fracture_network.build_grahp()
    fracture_network.draw_grahp()
    ika = 0



# geometry method
def build_box(cells, box_points):

    cells = np.append(cells, np.array([cell(0, i) for i, point in enumerate(box_points)]))

    edge = cell(1, 8)
    edge.boundary_cells = cells[[0, 1]]
    cells = np.append(cells, edge)

    edge = cell(1, 9)
    edge.boundary_cells = cells[[1, 2]]
    cells = np.append(cells, edge)

    edge = cell(1, 10)
    edge.boundary_cells = cells[[2, 3]]
    cells = np.append(cells, edge)

    edge = cell(1, 11)
    edge.boundary_cells = cells[[3, 0]]
    cells = np.append(cells, edge)

    edge = cell(1, 12)
    edge.boundary_cells = cells[[4, 5]]
    cells = np.append(cells, edge)

    edge = cell(1, 13)
    edge.boundary_cells = cells[[5, 6]]
    cells = np.append(cells, edge)

    edge = cell(1, 14)
    edge.boundary_cells = cells[[6, 7]]
    cells = np.append(cells, edge)

    edge = cell(1, 15)
    edge.boundary_cells = cells[[7, 4]]
    cells = np.append(cells, edge)

    edge = cell(1, 16)
    edge.boundary_cells = cells[[0, 4]]
    cells = np.append(cells, edge)

    edge = cell(1, 17)
    edge.boundary_cells = cells[[1, 5]]
    cells = np.append(cells, edge)

    edge = cell(1, 18)
    edge.boundary_cells = cells[[2, 6]]
    cells = np.append(cells, edge)

    edge = cell(1, 19)
    edge.boundary_cells = cells[[3, 7]]
    cells = np.append(cells, edge)

    surface = cell(2, 20)
    surface.boundary_cells = cells[[8, 17, 12, 16]]
    cells = np.append(cells, surface)

    surface = cell(2, 21)
    surface.boundary_cells = cells[[9, 18, 13, 17]]
    cells = np.append(cells, surface)

    surface = cell(2, 22)
    surface.boundary_cells = cells[[10, 13, 14, 19]]
    cells = np.append(cells, surface)

    surface = cell(2, 23)
    surface.boundary_cells = cells[[11, 16, 15, 19]]
    cells = np.append(cells, surface)

    surface = cell(2, 24)
    surface.boundary_cells = cells[[8, 9, 10, 11]]
    cells = np.append(cells, surface)

    surface = cell(2, 25)
    surface.boundary_cells = cells[[12, 13, 14, 15]]
    cells = np.append(cells, surface)

    volume = cell(3, 26)
    volume.boundary_cells = cells[[20, 21, 22, 23, 24, 25]]
    cells = np.append(cells, volume)
    return cells

def build_box_2D(cells, box_points):
    cells = np.append(cells, np.array([cell(0, i) for i, point in enumerate(box_points)]))

    edge = cell(1, 4)
    edge.boundary_cells = cells[[0, 1]]
    cells = np.append(cells, edge)

    edge = cell(1, 5)
    edge.boundary_cells = cells[[1, 2]]
    cells = np.append(cells, edge)

    edge = cell(1, 6)
    edge.boundary_cells = cells[[2, 3]]
    cells = np.append(cells, edge)

    edge = cell(1, 7)
    edge.boundary_cells = cells[[3, 0]]
    cells = np.append(cells, edge)

    surface = cell(2, 8)
    surface.boundary_cells = cells[[4, 5, 6, 7]]
    cells = np.append(cells, surface)

    return cells



def insert_graph_edge(g_cell: cell, tuple_id_list):
    for bc_cell in g_cell.boundary_cells:
        tuple_id_list.append((g_cell.id, bc_cell.id))
        if bc_cell.dimension == 0:
            print("BC: Vertex with id: ",bc_cell.id)
        else:
            insert_graph_edge(bc_cell, tuple_id_list)
    for immersed_cell in g_cell.immersed_cells:
        tuple_id_list.append((g_cell.id, immersed_cell.id))
        if immersed_cell.dimension == 0:
            print("IM: Vertex with id: ",immersed_cell.id)
        else:
            insert_graph_edge(immersed_cell, tuple_id_list)

def draw_graph(G):
    # pos = nx.bipartite_layout(G, top)
    nx.draw(G, pos=nx.circular_layout(G), with_labels=True, node_color="skyblue")

def build_geometry_graph(cells):

    return graph


def main():


    # polygon_polygon_intersection()
    # return 0

    # cells = np.array([],dtype=cell)

    s = 1.0;

    # points = s * np.array([[-1, -1, 0], [+1, -1, 0], [+1, +1, 0], [-1, +1, 0],
    #                    [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1]])
    # create volume cell
    # cells = build_box(cells,points)
    # tuple_id_list = []
    # insert_graph_edge(cells[26],tuple_id_list)
    # graph = nx.from_edgelist(tuple_id_list, create_using=nx.DiGraph)
    # draw_graph(graph)
    # gbuilder = geometry_builder(dimension=3)
    # polygon_polygon_intersection()
    # return 0

    # surface cell
    points = s * np.array([[-1, -1], [+1, -1], [+1, +1], [-1, +1]])
    # cells = build_box_2D(cells,points)

    # insert base fractures
    fracture_1 = np.array([[0.25, 0.25], [0.75, 0.75]])
    fracture_2 = np.array([[0.25, 0.75], [0.75, 0.25]])
    fracture_3 = np.array([[0.5, 0.25], [0.5, 0.75]])

    fractures = [fracture_1,fracture_2,fracture_3]

    fracture_network = Network(dimension=2)
    fracture_network.intersect_1D_fractures(fractures)
    fracture_network.build_grahp()

    pre_cells = fracture_network.graph.pred[6]

    fracture_network.draw_grahp()

    gbuilder = geometry_builder(dimension=2)
    gbuilder.build_internal_bc(fracture_network)


    # neigh = list(nx.all_neighbors(G, 8))

if __name__ == '__main__':
    main()



