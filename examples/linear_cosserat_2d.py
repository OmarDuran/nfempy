import numpy as np
import quadpy
import gmsh

from shapely.geometry import LineString


class Fracture:

    def __init__(self, vertices: np.array, connectivity: np.array):

        self.vertices = vertices
        self.connectivity = connectivity
        self.dimension = connectivity.shape[1]
        self.boundary = connectivity

def main():

    line1 = LineString([(0, 0), (1, 0), (1, 1)])
    line2 = LineString([(0, 1), (1, 1)])
    print(line1.intersection(line2))

if __name__ == '__main__':
    main()



