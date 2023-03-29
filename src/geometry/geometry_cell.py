import numpy as np

# Topology for Geometric Design
# It is required to find a better reference
# https://www.maruf.ca/files/caadoc/CAATobTechArticles/TopoConcepts.htm#Manifold
def geo_cell_type(dimension):
    types = ("Vertex", "Edge", "Face", "Volume")
    return types[dimension]


class GeometryCell:
    def __init__(self, dimension, id, physical_tag=None, point_id=None):

        self.dimension = dimension
        self.type = geo_cell_type(dimension)
        self.id = id
        self.physical_tag = physical_tag
        self.boundary_cells = np.array([], dtype=GeometryCell)
        self.immersed_cells = np.array([], dtype=GeometryCell)
        self.point_id = None

        # A vertex id is point_id
        if self.type == "Vertex":
            if point_id is None:
                self.point_id = id
            else:
                self.point_id = point_id

