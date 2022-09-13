import numpy as np


# Topology for Geometric Design
# It is required to find a better reference
# https://www.maruf.ca/files/caadoc/CAATobTechArticles/TopoConcepts.htm#Manifold
def cell_type(dimension):
    types = ("Vertex", "Edge", "Face", "Volume")
    # types = ("0-cell", "1-cell", "2-cell", "3-cell")
    return types[dimension]


# TODO rename to GeometryCell
class Cell:
    def __init__(self, dimension, id, physical_tag=None, point_id=None):

        self.dimension = dimension
        self.type = cell_type(dimension)
        self.id = id
        self.physical_tag = physical_tag
        self.boundary_cells = np.array([], dtype=Cell)
        self.immersed_cells = np.array([], dtype=Cell)

        # A vertex has the point id attribute
        self.point_id = None
        if dimension == 0:
            if point_id is None:
                self.point_id = id
            else:
                self.point_id = point_id
