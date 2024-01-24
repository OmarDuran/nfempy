import numpy as np

from geometry.shape import Shape


class Solid(Shape):
    def __init__(self, tag, shells):
        super().__init__()
        self._dimension = 3
        self.tag = tag
        self.boundary_shapes = shells

    def admissible_dimensions(self):
        return [0, 1, 2, 3]
