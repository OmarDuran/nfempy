import numpy as np
from geometry.shape import Shape


class Face(Shape):
    def __init__(self, tag, wires):
        super().__init__()
        self._dimension = 2
        self.tag = tag
        self.boundary_shapes = wires

    def admissible_dimensions(self):
        return [0, 1, 2]
