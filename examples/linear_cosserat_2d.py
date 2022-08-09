import numpy as np
from numpy import linalg as la
import quadpy
import gmsh

from shapely.geometry import LineString
import networkx as nx

import matplotlib.pyplot as plt


class facet:

    def __init__(self, dimension):
        self.connectivity = None

class cell:

    def __init__(self, dimension):

        self.dimension = dimension
        self.facets = None
        self.connectivy_grahp = None


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

        self.network = [] # list of cells
        self.dimension = dimension
        self.grahp = None
        self.vertices = None
        self.connectivity = None
        self.fractures_indices = None
        self.fractures_bcs = None

    def intersect_fractures(self, vertices, connectivity):

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

import geometry.computational_geometry as cgeo

def main():
    import os

    print('getcwd:      ', os.getcwd())
    print('__file__:    ', __file__)

    obj = cgeo.ComputationalGeometry()
    obj.TVolume([0.25, 0.25], [0.75, 0.75],[0.25, 0.75],[0.75, 0.25])

    # cd.TVolume([0.25, 0.25], [0.75, 0.75],[0.25, 0.75],[0.75, 0.25])
    print("dir: ",dir())
    pts = np.array([[0.25, 0.25], [0.75, 0.75],[0.25, 0.75],[0.75, 0.25]])
    c_map = np.array([[0,1],[2,3]])

    fracture_network = Network(dimension=2)
    fracture_network.intersect_fractures(pts,c_map)
    fracture_network.build_grahp()
    fracture_network.draw_grahp()

    gbuilder = geometry_builder(dimension=2)
    gbuilder.build_internal_bc(fracture_network)


    # neigh = list(nx.all_neighbors(G, 8))

if __name__ == '__main__':
    main()



