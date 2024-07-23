import numpy as np
from topology.shape import Shape


class Face(Shape):
    def __init__(self, tag, wires):
        super().__init__()
        self._dimension = 2
        self.tag = tag
        self.boundary_shapes = wires

    def admissible_dimensions(self):
        return [0, 1, 2]

    def boundary_points(self):
        points = []
        for wire in self.boundary_shapes:
            for i, edge in enumerate(wire.immersed_shapes):
                if wire.orientation[i] > 0:
                    points.append(edge.boundary_points()[0])
                else:
                    points.append(edge.boundary_points()[1])
        points = np.array(points)
        return points