import numpy as np
from geometry.shape import Shape


class Wire(Shape):
    def __init__(self, tag, edges, vertices):
        super().__init__()
        self._dimension = 1
        self.tag = tag
        self.composite = True
        self.immersed_shapes = edges
        self.boundary_shapes = vertices
        self.orient_immersed_edges()

    def admissible_dimensions(self):
        return [0, 1]

    # @staticmethod
    # def validate_edge(edge):
    #     perm = np.argsort([vertex.tag for vertex in edge.boundary_shapes])
    #     edge.boundary_shapes = edge.boundary_shapes[perm]
    #     return edge
    #
    # @staticmethod
    # def reverse_edge(edge):
    #     edge.boundary_shapes = np.reverse([vertex.tag for vertex in edge.boundary_shapes])
    #     return edge

    def orient_immersed_edges(self):
        edges = self.immersed_shapes
        seed_vertex = edges[0].boundary_shapes[0]
        if seed_vertex.tag > edges[0].boundary_shapes[1].tag:
            seed_vertex = edges[0].boundary_shapes[1]

        self.orientation = []
        for edge in edges:
            if seed_vertex == edge.boundary_shapes[0]:
                self.orientation.append(1)
                seed_vertex = edge.boundary_shapes[1]
            else:
                self.orientation.append(-1)
                seed_vertex = edge.boundary_shapes[0]
        self.orientation = np.array(self.orientation)
