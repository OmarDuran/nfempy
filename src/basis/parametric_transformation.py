import basix
import numpy as np
from basix import CellType

from basis.element_data import ElementData

def _R0_to_R1(facet_index, points, data_c1: ElementData, data_c0: ElementData):
    dim = data_c1.cell.dimension

    line_vertices = basix.geometry(CellType.interval)
    line_connectivities = basix.cell.sub_entity_connectivity(CellType.interval)
    line_facet = line_connectivities[dim][facet_index][0]
    line_face_subentities = data_c0.cell.node_tags[line_facet]

    point_facet = data_c1.cell.node_tags

    permutation_q = tuple(point_facet) != tuple(line_face_subentities)

    if permutation_q:
        assert tuple(np.sort(line_facet)) == tuple(np.sort(line_face_subentities))
        index_map = dict(zip(line_face_subentities, line_facet))
        line_facet = [index_map[k] for k in line_facet]

    # perform linear map
    mapped_points = np.array(
        [
            line_vertices[line_facet[0]] * xi
            for xi in points
        ]
    )
    return mapped_points

def _R1_to_R2(facet_index, points, data_c1: ElementData, data_c0: ElementData):
    dim = data_c1.cell.dimension

    triangle_vertices = basix.geometry(CellType.triangle)
    triangle_connectivities = basix.cell.sub_entity_connectivity(CellType.triangle)
    triangle_facet = triangle_connectivities[dim][facet_index][0]
    triangle_face_subentities = data_c0.cell.node_tags[triangle_facet]

    line_facet = data_c1.cell.node_tags

    permutation_q = tuple(line_facet) != tuple(triangle_face_subentities)

    if permutation_q:
        assert tuple(np.sort(line_facet)) == tuple(np.sort(triangle_face_subentities))
        index_map = dict(zip(triangle_face_subentities, triangle_facet))
        triangle_facet = [index_map[k] for k in line_facet]

    # perform linear map
    mapped_points = np.array(
        [
            triangle_vertices[triangle_facet[0]] * (1 - xi)
            + triangle_vertices[triangle_facet[1]] * xi
            for xi in points
        ]
    )
    return mapped_points


def _R2_to_R3(facet_index, points, data_c1: ElementData, data_c0: ElementData):
    dim = data_c1.cell.dimension
    tetrahedron_vertices = basix.geometry(CellType.tetrahedron)
    tetrahedron_connectivities = basix.cell.sub_entity_connectivity(
        CellType.tetrahedron
    )
    tetrahedron_facet = tetrahedron_connectivities[dim][facet_index][0]
    tetrahedron_face_subentities = data_c0.cell.node_tags[tetrahedron_facet]

    triangle_facet = data_c1.cell.node_tags
    permutation_q = tuple(triangle_facet) != tuple(tetrahedron_face_subentities)

    if permutation_q:
        assert tuple(np.sort(triangle_facet)) == tuple(
            np.sort(tetrahedron_face_subentities)
        )
        index_map = dict(zip(tetrahedron_face_subentities, tetrahedron_facet))
        tetrahedron_facet = [index_map[k] for k in triangle_facet]

    # perform linear map
    mapped_points = np.array(
        [
            tetrahedron_vertices[tetrahedron_facet[0]] * (1 - xi - eta)
            + tetrahedron_vertices[tetrahedron_facet[1]] * xi
            + tetrahedron_vertices[tetrahedron_facet[2]] * eta
            for xi, eta in points
        ]
    )
    return mapped_points


def transform_lower_to_higher(points, data_c1: ElementData, data_c0: ElementData):
    cell = data_c1.cell
    neigh_cell = data_c0.cell
    facet_index = neigh_cell.sub_cells_ids[cell.dimension].tolist().index(cell.id)
    if cell.dimension == 0:
        return _R0_to_R1(facet_index, points, data_c1, data_c0)
    elif cell.dimension == 1:
        return _R1_to_R2(facet_index, points, data_c1, data_c0)
    elif cell.dimension == 2:
        return _R2_to_R3(facet_index, points, data_c1, data_c0)

