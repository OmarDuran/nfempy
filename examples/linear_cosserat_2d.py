import numpy as np
from numpy import linalg as la
import quadpy
import gmsh

from shapely.geometry import LineString

import geometry.fracture_network as fn
import networkx as nx

import matplotlib.pyplot as plt


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


def build_geometry_graph(cells):

    return graph


def main():


    polygon_polygon_intersection()
    return 0

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
    # fracture_3 = np.array([[0.5, 0.25], [0.5, 0.75]])
    fracture_3 = np.array([[0.65, 0.25], [0.65, 0.75]])

    fractures = [fracture_1,fracture_2,fracture_3]

    fracture_network = fn.FractureNetwork(dimension=2)
    fracture_network.intersect_1D_fractures(fractures, render_intersection_q = True)
    fracture_network.build_grahp()

    pre_cells = fracture_network.graph.pred[6]

    fracture_network.draw_grahp()

    gbuilder = geometry_builder(dimension=2)
    gbuilder.build_internal_bc(fracture_network)


    # neigh = list(nx.all_neighbors(G, 8))

if __name__ == '__main__':
    main()



