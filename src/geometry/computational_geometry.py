import numpy as np


class ComputationalGeometry:
    """Worker class for computational geometry.

    This class is mainly dedicated to compute polygon-polygon intersections

    """

    def __init__(self):
        pass

    @staticmethod
    def TVolume(self, a, b, c, p) -> float:
        tequ = np.array([a, b, c, p])
        print(tequ)
