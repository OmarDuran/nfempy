import numpy as np


class ComputationalGeometry:
    """Worker class for computational geometry.

    This class is mainly dedicated to compute polygon-polygon intersections

    """

    def __init__(self):
        pass

    # TODO: name it to TMeasurement
    def TVolume(self, a: np.array, b: np.array, c: np.array, p: np.array) -> float:
        tequ = np.hstack((np.array([a, b, c, p]), np.ones((4, 1))))
        measurement = np.linalg.det(tequ)
        return measurement