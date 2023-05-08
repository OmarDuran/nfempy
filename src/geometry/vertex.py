import numpy as np
from geometry.shape import Shape


class Vertex(Shape):
    def __init__(self, tag, point):
        super().__init__()
        self._dimension = 0
        self.tag = tag
        self.point = point

    def admissible_dimensions(self):
        return [0]

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, point: np.ndarray):
        self._point: np.ndarray = point
