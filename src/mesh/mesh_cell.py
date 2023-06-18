import numpy as np


def barycenter(points):
    xc = np.mean(points, axis=0)
    return xc


def x_rotation(vector, theta):
    R = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    return np.dot(R, vector)


def y_rotation(vector, theta):
    R = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    return np.dot(R, vector)


def z_rotation(vector, theta):
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return np.dot(R, vector)


def rotate_vector(vector, theta, axis=2):
    if axis == 0:
        return x_rotation(vector, theta)
    if axis == 1:
        return y_rotation(vector, theta)
    if axis == 2:
        return z_rotation(vector, theta)


# class MeshCell(abc.ABC):
class MeshCell:
    def __init__(self, dimension):
        self.dimension = dimension
        self.type = self.mesh_cell_type(dimension)
        self.id = None
        self.material_id = None
        self.physical_name = "Unnamed"
        self.node_tags = np.array([], dtype=int)
        self.perm = np.array([], dtype=int)
        data = [np.array([], dtype=int) for i in range(dimension + 1)]
        self.sub_cells_ids = data

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def set_physical_name(self, physical_name):
        self.physical_name = physical_name

    def get_physical_name(self):
        return self.physical_name

    def set_material_id(self, material_id):
        self.material_id = material_id

    def get_material_id(self):
        return self.material_id

    def set_node_tags(self, node_tags):
        self.node_tags = node_tags
        self.perm = np.argsort(self.node_tags)

    def get_node_tags(self):
        return self.node_tags

    def set_sub_cells_ids(self, dim, cells_ids):
        assert self.dimension >= dim
        self.sub_cells_ids[dim] = cells_ids

    @staticmethod
    def mesh_cell_type(dimension):
        types = ("0d-cell", "1d-cell", "2d-cell", "3d-cell")
        return types[dimension]
