import abc
import numpy as np


# class MeshCell(abc.ABC):
class MeshCell:
    def __init__(self, dimension):
        self.dimension = dimension
        self.type = self.mesh_cell_type(dimension)
        self.id = None
        self.material_id = None
        self.node_tags = np.array([], dtype=int)
        self.perm = np.array([], dtype=int)
        self.cells_0d = np.array([], dtype=MeshCell)
        self.cells_1d = np.array([], dtype=MeshCell)
        self.cells_2d = np.array([], dtype=MeshCell)

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def set_material_id(self, material_id):
        self.material_id = material_id

    def get_material_id(self):
        return self.material_id

    def set_node_tags(self, node_tags):
        self.node_tags = node_tags
        self.perm = np.argsort(self.node_tags)

    def get_node_tags(self):
        return self.node_tags

    # @abc.abstractmethod
    def set_cells_0d(self, cells_0d):
        self.cells_0d = cells_0d

    # @abc.abstractmethod
    def set_cells_1d(self, cells_1d):
        self.cells_1d = cells_1d

    # @abc.abstractmethod
    def set_cells_2d(self, cells_2d):
        self.cells_2d = cells_2d


    @staticmethod
    def mesh_cell_type(dimension):
        types = ("0d-cell", "1d-cell", "2d-cell", "3d-cell")
        return types[dimension]
