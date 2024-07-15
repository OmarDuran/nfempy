from topology.shape import Shape


class Shell(Shape):
    def __init__(self, tag, faces, wires):
        super().__init__()
        self._dimension = 2
        self.tag = tag
        self.composite = True
        self.immersed_shapes = faces
        self.boundary_shapes = wires

    def admissible_dimensions(self):
        return [0, 1, 2]
