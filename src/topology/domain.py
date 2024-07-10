import networkx as nx
import numpy as np

from topology.shape import Shape


class Domain:
    def __init__(self, dimension):
        self.dimension = dimension
        self.shapes = [np.array([], dtype=Shape) for i in range(dimension + 1)]
        self.graph = None
        self.graph_embed = None

    def append_shapes(self, shapes):
        if len(shapes) == 0:
            return
        dims = [shape.dimension for shape in shapes]
        dim, counts = np.unique(dims, return_counts=True)
        assert counts[0] == len(shapes)
        self.shapes[dim[0]] = np.append(self.shapes[dim[0]], shapes, axis=0)

    def max_tag(self):
        tags = [
            shape.tag
            for shapes_d in self.shapes
            for shape in shapes_d
        ]
        max = np.max(np.array(tags))
        return max

    def max_physical_tag(self):
        physical_tags = [
            shape.physical_tag
            for shapes_d in self.shapes
            for shape in shapes_d
            if shape.physical_tag is not None
        ]
        max = np.max(np.array(physical_tags))
        return max

    def shapes_with_physical_tag(self, physical_tag):
        shapes = np.array(
            [
                shape
                for shapes_d in self.shapes
                for shape in shapes_d
                if shape.physical_tag == physical_tag
            ]
        )
        return shapes

    def refresh_wires(self):
        [shape.orient_immersed_edges() for shape in self.shapes[1] if shape.composite]

    def retag_shapes_with_dimension(self, dim):
        tag = 0
        for i, shape in enumerate(self.shapes[dim]):
            shape.tag = tag
            tag += 1
            if len(shape.immersed_shapes):
                for j, ishape in enumerate(shape.immersed_shapes):
                    if shape.dimension != dim:
                        continue
                    ishape.tag = tag
                    tag += 1

    def retag_shapes(self):
        for d in range(self.dimension + 1):
            self.retag_shapes_with_dimension(d)

    def lookup_vertex(self, point):
        indices = []
        for i, vertex in enumerate(self.shapes[0]):
            if np.all(np.isclose(vertex.point, point)):
                indices.append(i)
        if len(indices) > 0:
            assert len(indices) == 1
            return self.shapes[0][indices][0]
        else:
            return None

    def remove_vertex(self):
        shapes = []
        for shape in self.shapes[0]:
            if self.graph.has_node((self.dimension, shape.tag)):
                shapes = shapes + [shape]
        self.shapes[0] = np.array(shapes)

    def shape_index(self, shape):
        return shape.index(max_dimension=self.dimension)

    def gather_graph_edges(self, shape: Shape, tuple_id_list):
        for bc_shape in shape.boundary_shapes:
            tuple_id_list.append(
                (
                    self.shape_index(shape),
                    self.shape_index(bc_shape),
                )
            )
            if bc_shape.dimension != 0:
                self.gather_graph_edges(bc_shape, tuple_id_list)

        for immersed_shape in shape.immersed_shapes:
            tuple_id_list.append(
                (
                    self.shape_index(shape),
                    self.shape_index(immersed_shape),
                )
            )
            if immersed_shape.dimension != 0:
                self.gather_graph_edges(immersed_shape, tuple_id_list)

    def build_grahp(self, dimension = None):
        if dimension is None:
            dimension = self.dimension
        disjoint_shapes = [shape_i for shape_i in self.shapes[dimension]]
        tuple_id_list = []
        for shape in disjoint_shapes:
            self.gather_graph_edges(shape, tuple_id_list)

        self.graph = nx.from_edgelist(tuple_id_list, create_using=nx.DiGraph)

        # case where a domain is composed of disjoint vertices
        if dimension == 0:
            if len(list(self.graph.nodes())) == 0:
                d0_nodes = [shape.index(self.dimension) for shape in disjoint_shapes]
                self.graph.add_nodes_from(d0_nodes)


    def draw_grahp(self):
        nx.draw(
            self.graph,
            pos=nx.circular_layout(self.graph),
            with_labels=True,
            node_color="skyblue",
        )

