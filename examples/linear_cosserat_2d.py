import numpy as np
from numpy import linalg as la
import quadpy
import gmsh

from shapely.geometry import LineString
import networkx as nx

import matplotlib.pyplot as plt


class Fracture:

    def __init__(self, vertices: np.array, connectivity: np.array, dimension):

        self.vertices = vertices
        self.connectivity = connectivity
        self.dimension = dimension
        self.boundary = connectivity


def main():

    pts = np.array([[0.25, 0.25], [0.75, 0.75],[0.25, 0.75],[0.75, 0.25],[0.55, 0.25],[0.55, 0.75],[0.1, 0.35],[0.8, 0.35]])
    c_map = np.array([[0,1],[2,3],[4,5],[6,7]])

    n_pts = pts
    f_c_map = [[] for _ in c_map]
    c = pts.shape[0]
    for i in range(len(c_map)):
        ids = []
        for j in range(i + 1,len(c_map)):
            fi = LineString(pts[c_map[i]])
            fj = LineString(pts[c_map[j]])
            point = fi.intersection(fj)
            if not point.is_empty:
                print("p: ", point)
                n_pts = np.append(n_pts,np.array([[point.x, point.y]]),axis=0)
                f_c_map[i].append(c)
                f_c_map[j].append(c)
                c = c + 1

    # for i in range(len(c_map)):
    for i in range(len(c_map)):
        R = n_pts[f_c_map[i]] - n_pts[c_map[i, 0]]
        r_norm = la.norm(R, axis=1)
        perm = np.argsort(r_norm)
        findexes = [f_c_map[i][k] for k in perm.tolist()]
        findexes.insert(0,c_map[i, 0])
        findexes.append(c_map[i, 1])
        f_c_map[i] = [(findexes[i], findexes[i+1]) for i in range(0, len(findexes)-1)]

    fedge_list = [item for sublist in f_c_map for item in sublist]
    G = nx.from_edgelist(fedge_list, create_using=nx.DiGraph)
    # nx.draw_planar(
    #     G,
    #     with_labels=True,
    #     node_size=1000,
    #     node_color="#ffff8f",
    #     width=0.8,
    #     font_size=14,
    # )

    # you want your own layout
    pos = nx.spring_layout(G)
    nodes  =  list(G.nodes)
    pos = {nodes[i]: n_pts[nodes[i]] for i in range(len(nodes))}

    # add axis
    fig, ax = plt.subplots()
    nx.draw(G, pos=pos, node_color='k', ax=ax)
    nx.draw(G, pos=pos, node_size=1500, ax=ax)  # draw nodes and edges
    nx.draw_networkx_labels(G, pos=pos)  # draw node labels/names
    # draw edge weights
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)
    plt.axis("on")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()

    neigh = list(nx.all_neighbors(G, 8))

if __name__ == '__main__':
    main()



