import abc
import numpy as np


class MeshCell(abc.ABC):
    def __init__(self, dimension, id, material_id):
        self.dimension = dimension
        self.type = mesh_cell_type(dimension)
        self.id = id
        self.material_id = material_id
        self.perm = np.array([], dtype=int)
        self.cells_0d = np.array([], dtype=MeshCell)
        self.cells_1d = np.array([], dtype=MeshCell)
        self.cells_2d = np.array([], dtype=MeshCell)
        self.cells_3d = np.array([], dtype=MeshCell)

    def set_material_id(self, material_id):
        self.material_id = material_id

    def get_material_id(self):
        return self.material_id

    @abc.abstractmethod
    def set_cells_0d(self, tags):
        pass

    @abc.abstractmethod
    def set_cells_1d(self, tags):
        pass

    @abc.abstractmethod
    def set_cells_2d(self, tags):
        pass

    @abc.abstractmethod
    def set_cells_3d(self, tags):
        pass

    @staticmethod
    def mesh_cell_type(dimension):
        types = ("0d-cell", "1d-cell", "2d-cell", "3d-cell")
        return types[dimension]
