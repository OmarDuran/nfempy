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

    @property
    def immersed_shapes(self):
        shapes = np.array([], dtype=Shape)
        for shape in self._immersed_shapes:
            if len(shape.immersed_shapes) > 0:
                for ishape in shape.immersed_shapes:
                    shapes = np.append(shapes, np.array([ishape]), axis=0)
            else:
                shapes = np.append(shapes, np.array([shape]), axis=0)
        return shapes

    @immersed_shapes.setter
    def immersed_shapes(self, shapes):

        check_shapes = np.array(
            [shape.dimension in self.admissible_dimensions() for shape in shapes]
        )
        if np.all(check_shapes):
            self._immersed_shapes = shapes
        else:
            raise ValueError(
                "This shape can only contain these dimensions: ",
                self.admissible_dimensions(),
            )

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
