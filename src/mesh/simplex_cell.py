import numpy as np
from .mesh_cell import MeshCell


class SimplexCell(MeshCell):
    def __init__(self, dimension, id):
        super(SimplexCell, self).__init__(dimension, id)

    def set_cells_0d(self, tags):
        pass

    def set_cells_1d(self, tags):
        pass

    def set_cells_2d(self, tags):
        pass

    def set_cells_3d(self, tags):
        pass
