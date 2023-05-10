import numpy as np
import networkx as nx
from geometry.shape import Shape


class Domain:
    def __init__(self, dimension):
        self.dimension = dimension
        self.shapes = [np.array([], dtype=Shape) for i in range(dimension + 1)]
        self.graph = None
        self.graph_embed = None

    def append_shapes(self, shapes):
        dims = [shape.dimension for shape in shapes]
        dim, counts = np.unique(dims, return_counts=True)
        assert counts[0] == len(shapes)
        self.shapes[dim[0]] = np.append(self.shapes[dim[0]], shapes, axis=0)

    def gather_graph_edges(self, shape: Shape, tuple_id_list):
        for bc_shape in shape.boundary_shapes:
            tuple_id_list.append(
                (
                    (self.dimension - shape.dimension, shape.tag),
                    (self.dimension - bc_shape.dimension, bc_shape.tag),
                )
            )
            if bc_shape.dimension != 0:
                self.gather_graph_edges(bc_shape, tuple_id_list)

        for immersed_shape in shape.immersed_shapes:
            tuple_id_list.append(
                (
                    (self.dimension - shape.dimension, shape.tag),
                    (self.dimension - immersed_shape.dimension, immersed_shape.tag),
                )
            )
            if immersed_shape.dimension != 0:
                self.gather_graph_edges(immersed_shape, tuple_id_list)

    def build_grahp(self):
        disjoint_shapes = [shape_i for shape_i in self.shapes[self.dimension]]
        tuple_id_list = []
        for shape in disjoint_shapes:
            self.gather_graph_edges(shape, tuple_id_list)

        self.graph = nx.from_edgelist(tuple_id_list, create_using=nx.DiGraph)

    def build_graph_embed(self):
        disjoint_shapes = [shape_i for shape_i in self.shapes[self.dimension]]
        tuple_id_list = []
        for shape in disjoint_shapes:
            self.gather_graph_edges(shape, tuple_id_list)

        self.graph = nx.from_edgelist(tuple_id_list, create_using=nx.DiGraph)

    def draw_grahp(self):
        nx.draw(
            self.graph,
            pos=nx.circular_layout(self.graph),
            with_labels=True,
            node_color="skyblue",
        )
