import numpy as np

from topology.shape import Shape


class Vertex(Shape):
    def __init__(self, tag, point):
        super().__init__()
        self._dimension = 0
        self.tag = tag
        self.point = point

    def admissible_dimensions(self):
        return [0]

    def __eq__(self, other):
        identity_check = (self.dimension, self.tag) == (other.dimension, other.tag)
        geometry_check = np.all(np.isclose(self.point, other.point))
        return identity_check and geometry_check

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, point: np.ndarray):
        self._point: np.ndarray = point

    def shape_assignment(self, other):
        super().shape_assignment(other)
        self.point = other.point