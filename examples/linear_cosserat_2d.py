import numpy as np
from numpy import linalg as la
import quadpy
import gmsh

from shapely.geometry import LineString
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

    def __init__(self, dimension, id):

        self.dimension = dimension
        self.type = cell_type(dimension)
        self.id = id
        self.boundary_cells = None
        self.immersed_cells = None


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

    def __init__(self, dimension):

        self.network = [] # list of facets
        self.dimension = dimension
        self.grahp = None
        self.vertices = None
        self.connectivity = None
        self.fractures_indices = None
        self.fractures_bcs = None

    def intersect_2D_fractures(self, vertices, connectivity):
        k = 0

    def intersect_1D_fractures(self, vertices, connectivity):

        # preserve original data
        self.fractures_indices = [[] for _ in connectivity]
        self.fractures_bcs = connectivity

        self.vertices = vertices
        self.connectivity = [[] for _ in connectivity]
        c = vertices.shape[0]
        for i in range(len(connectivity)):
            for j in range(i + 1, len(connectivity)):
                fi = LineString(vertices[connectivity[i]])
                fj = LineString(vertices[connectivity[j]])
                point = fi.intersection(fj)
                if not point.is_empty:
                    print("p: ", point)
                    self.vertices = np.append(self.vertices, np.array([[point.x, point.y]]), axis=0)
                    self.connectivity[i].append(c)
                    self.connectivity[j].append(c)
                    c = c + 1

        for i in range(len(connectivity)):
            R = self.vertices[self.connectivity[i]] - self.vertices[connectivity[i, 0]]
            r_norm = la.norm(R, axis=1)
            perm = np.argsort(r_norm)
            indices = [self.connectivity[i][k] for k in perm.tolist()]
            indices.insert(0, connectivity[i, 0])
            indices.append(connectivity[i, 1])
            self.connectivity[i] = [(indices[i], indices[i + 1]) for i in
                          range(0, len(indices) - 1)]

    def build_grahp(self):
        fedge_list = [item for sublist in self.connectivity for item in sublist]
        self.grahp = nx.from_edgelist(fedge_list, create_using=nx.DiGraph)

    def draw_grahp(self):

        if self.grahp is None:
            self.build_grahp()

        nodes = list(self.grahp.nodes)
        pos = {nodes[i]: self.vertices[nodes[i]] for i in range(len(nodes))}

        # add axis
        fig, ax = plt.subplots()
        nx.draw(self.grahp, pos=pos, node_color='k', ax=ax)
        nx.draw(self.grahp, pos=pos, node_size=1500, ax=ax)  # draw nodes and edges
        nx.draw_networkx_labels(self.grahp, pos=pos)  # draw node labels/names
        # draw edge weights
        labels = nx.get_edge_attributes(self.grahp, 'weight')
        nx.draw_networkx_edge_labels(self.grahp, pos, edge_labels=labels, ax=ax)
        plt.axis("on")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()

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
    fracture_2 = np.array([[0.6, 0., 0.5], [0.6, 0., -0.5], [0.6, 1., -0.5], [0.6, 1., 0.5]])
    fracture_3 = np.array([[0.25, 0., 0.5], [0.914463, 0.241845, -0.207107], [0.572443, 1.18154, -0.207107],
     [-0.0920201, 0.939693, 0.5]])

    fractures = [fracture_1,fracture_2,fracture_3]

    intersection_data = 3*[3*[None]]
    for i, f_o in enumerate(fractures):
        for j,  f_t in enumerate(fractures):
            if i == j:
                intersection_data[i][j] = (False,np.array,np.array)
                continue
            obj = pp_intersector.PolygonPolygonIntersectionTest()
            intersection_q = obj.polygon_polygon_intersection(f_o, f_t, False)
            intersection_data[i][j] = intersection_q

    i = 0
    i_results = [chunk[0] for chunk in intersection_data[i]]
    n_intersections = np.count_nonzero(i_results)
    if n_intersections > 1:

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

        points = []
        for chunk in intersection_data[i]:
            if chunk[0]:
                points.append(chunk[1])
                points.append(chunk[2])
        points = np.array(points)

        pts = points[:,pos]
        c_map = np.array([[0,1],[2,3]])

        fracture_network = Network(dimension=2)
        fracture_network.intersect_1D_fractures(pts,c_map)
        fracture_network.build_grahp()
        fracture_network.draw_grahp()




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

def insert_graph_edge(g_cell: cell, tuple_id_list):
    for bc_cell in g_cell.boundary_cells:
        tuple_id_list.append((g_cell.id, bc_cell.id))
        if bc_cell.dimension == 0:
            print("Vertex with id: ",bc_cell.id)
        else:
            insert_graph_edge(bc_cell, tuple_id_list)

def draw_graph(G):
    # pos = nx.bipartite_layout(G, top)
    nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color="skyblue")

def build_geometry_graph(cells):

    return graph


def main():

    cells = np.array([],dtype=cell)
    s = 1.0;
    points = s * np.array([[-1, -1, 0], [+1, -1, 0], [+1, +1, 0], [-1, +1, 0],
                       [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1]])

    # create volume cell
    cells = build_box(cells,points)
    tuple_id_list = []
    insert_graph_edge(cells[26],tuple_id_list)
    gbuilder = geometry_builder(dimension=3)

    graph = nx.from_edgelist(tuple_id_list, create_using=nx.DiGraph)
    # draw_graph(graph)

    polygon_polygon_intersection()
    return 0

    pts = np.array([[0.25, 0.25], [0.75, 0.75],[0.25, 0.75],[0.75, 0.25]])
    c_map = np.array([[0,1],[2,3]])

    fracture_network = Network(dimension=2)
    fracture_network.intersect_1D_fractures(pts,c_map)
    fracture_network.build_grahp()
    fracture_network.draw_grahp()

    gbuilder = geometry_builder(dimension=2)
    gbuilder.build_internal_bc(fracture_network)


    # neigh = list(nx.all_neighbors(G, 8))

if __name__ == '__main__':
    main()



