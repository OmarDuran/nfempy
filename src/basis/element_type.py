from basix import CellType


def type_by_dimension(dimension):
    element_types = {
        0: CellType.point,
        1: CellType.interval,
        2: CellType.triangle,
        3: CellType.tetrahedron,
    }
    return element_types[dimension]
