import numpy as np
from geometry.shape import Shape


class Edge(Shape):
    def __init__(self, tag, vertices):
        super().__init__()
        self._dimension = 1
        self.tag = tag
        self.boundary_shapes = vertices

    def admissible_dimensions(self):
        return [0, 1]
