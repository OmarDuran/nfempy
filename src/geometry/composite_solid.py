import numpy as np
from geometry.shape import Shape


class CompositeSolid(Shape):
    def __init__(self, tag, solids, shells):
        super().__init__()
        self._dimension = 3
        self.tag = tag
        self.composite = True
        self.immersed_shapes = solids
        self.boundary_shapes = shells

    def admissible_dimensions(self):
        return [0, 1, 2, 3]
