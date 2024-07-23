import numpy as np
import warnings
from functools import partial

from globals import geometry_collapse_precision as collapse_precision
from globals import geometry_line_polygon_incidence_tol as s_incidence_tol
from geometry.operations.point_geometry_operations import coplanar_measurement
from geometry.operations.line_geometry_operations import lines_triangle_intersection
from geometry.operations.polygon_operations import triangulate_polygon


def triangle_triangle_intersection(
    triangle_tool: np.array, triangle_object: np.array, eps: float = s_incidence_tol
) -> np.array:

    p, q, r = triangle_tool
    a, b, c = triangle_object
    p_measurement = coplanar_measurement(p, a, b, c)
    q_measurement = coplanar_measurement(q, a, b, c)
    r_measurement = coplanar_measurement(r, a, b, c)

    p_coplanar_q = np.isclose(p_measurement, 0.0, rtol=eps, atol=eps)
    q_coplanar_q = np.isclose(q_measurement, 0.0, rtol=eps, atol=eps)
    r_coplanar_q = np.isclose(r_measurement, 0.0, rtol=eps, atol=eps)

    # perform lines - triangle intersection
    lines_tool = np.array([[p, q], [q, r], [r, p]])
    intx_data, intx_q = lines_triangle_intersection(lines_tool, a, b, c, eps)

    t_are_coplanar_q = np.all([p_coplanar_q, q_coplanar_q, r_coplanar_q])
    if t_are_coplanar_q and np.count_nonzero(intx_q) > 1:
        # Case for triangle intersection rendering an area it will be ignored.
        warnings.warn("Warning:: Intersection of coplanar triangles forms an area.")
        return None

    if np.count_nonzero(intx_q) == 0:

        sym_lines_tool = np.array([[a, b], [b, c], [c, a]])
        sym_intx_data, sym_intx_q = lines_triangle_intersection(
            sym_lines_tool, p, q, r, eps
        )

        if np.count_nonzero(sym_intx_q) == 0:
            return None
        else:
            # Define domains to be subtracted
            sym_intx_data_rounded = np.round(sym_intx_data, decimals=collapse_precision)
            _, idx = np.unique(sym_intx_data_rounded, axis=0, return_index=True)
            sym_intx_data = sym_intx_data[idx]
            if len(sym_intx_data.shape) == 3:
                sym_intx_data = sym_intx_data[:, 0, :]
            return sym_intx_data
    elif np.count_nonzero(intx_q) == 1:
        sym_lines_tool = np.array([[a, b], [b, c], [c, a]])
        sym_intx_data, sym_intx_q = lines_triangle_intersection(
            sym_lines_tool, p, q, r, eps
        )
        if np.count_nonzero(sym_intx_q) == 1:
            # Define domains to be subtracted
            intx_data_rounded = np.round(intx_data, decimals=collapse_precision)
            _, idx = np.unique(intx_data_rounded, axis=0, return_index=True)
            intx_data = intx_data[idx]
            if len(intx_data.shape) == 3:
                intx_data = intx_data[:, 0, :]

            # Define domains to be subtracted
            sym_intx_data_rounded = np.round(sym_intx_data, decimals=collapse_precision)
            _, idx = np.unique(sym_intx_data_rounded, axis=0, return_index=True)
            sym_intx_data = sym_intx_data[idx]
            if len(sym_intx_data.shape) == 3:
                sym_intx_data = sym_intx_data[:, 0, :]

            intx_data = np.concatenate([intx_data, sym_intx_data])
            # Define domains to be subtracted
            intx_data_rounded = np.round(intx_data, decimals=collapse_precision)
            _, idx = np.unique(intx_data_rounded, axis=0, return_index=True)
            intx_data = intx_data[idx]
            if len(intx_data.shape) == 3:
                intx_data = intx_data[:, 0, :]
            return intx_data

    # Define domains to be subtracted
    intx_data_rounded = np.round(intx_data, decimals=collapse_precision)
    _, idx = np.unique(intx_data_rounded, axis=0, return_index=True)
    intx_data = intx_data[idx]
    if len(intx_data.shape) == 3:
        intx_data = intx_data[:, 0, :]
    return intx_data


def triangle_polygon_intersection(
    triangle_tool: np.array, polygon_object: np.array, eps: float = s_incidence_tol
) -> np.array:

    triangles_obj = triangulate_polygon(polygon_object)
    intx_data = []
    for triangle_obj in triangles_obj:
        out = triangle_triangle_intersection(triangle_tool, triangle_obj, eps)
        if out is None:
            continue
        intx_data.append(out)
    if len(intx_data) == 0:
        return None
    intx_data = np.concatenate(intx_data)
    intx_data_rounded = np.round(intx_data, decimals=collapse_precision)
    _, idx = np.unique(intx_data_rounded, axis=0, return_index=True)
    intx_data = intx_data[idx]
    if len(intx_data) <= 2:
        return intx_data

    xc = np.mean(intx_data, axis=0)
    dirs = intx_data - xc
    v = intx_data[0] - xc
    idx = np.argsort(np.dot(dirs, v))
    intx_data = intx_data[idx[np.array([0, -1])]]
    return intx_data


def polygon_polygon_intersection(
    polygon_tool: np.array, polygon_object: np.array, eps: float = s_incidence_tol
) -> np.array:

    # compute intersections
    triangle_poly_int = partial(
        triangle_polygon_intersection, polygon_object=polygon_object, eps=eps
    )
    triangles_tool = triangulate_polygon(polygon_tool)
    intx_data = list(map(triangle_poly_int, triangles_tool))
    filter_none = list(filter(lambda x: x is not None, intx_data))
    if len(filter_none) == 0:
        return None
    intx_data = np.concatenate(filter_none)
    intx_data_rounded = np.round(intx_data, decimals=collapse_precision)
    _, idx = np.unique(intx_data_rounded, axis=0, return_index=True)
    intx_data = intx_data[idx]
    if len(intx_data) <= 2:
        return intx_data

    xc = np.mean(intx_data, axis=0)
    dirs = intx_data - xc
    v = intx_data[0] - xc
    idx = np.argsort(np.dot(dirs, v))
    intx_data = intx_data[idx[np.array([0, -1])]]
    return intx_data


def polygons_polygon_intersection(
    polygons_tool: np.array, polygon_object: np.array, eps: float = s_incidence_tol
) -> np.array:
    # compute intersections
    polygon_polygon_intx = partial(polygon_polygon_intersection, polygon_object=polygon_object, eps=eps)
    result = list(map(polygon_polygon_intx,polygons_tool))

    # filter lines outside segment
    result = np.array(list(filter(lambda x: x is not None, result)))
    return result

def polygons_polygons_intersection(
    polygons_tools: np.array,
    polygons_objects: np.array,
    deduplicate_lines_q: bool = False,
    eps: float = s_incidence_tol,
) -> float:
    # compute intersections

    output = [None for _ in range(polygons_objects.shape[0])]
    for i, polygon_obj in enumerate(polygons_objects):
        polygon_equality = [
            np.all(np.isclose(polygon_tool, polygon_obj)) for polygon_tool in polygons_tools if polygon_tool.shape == polygon_obj.shape
        ]
        tuple_idx = np.where(polygon_equality)
        if len(tuple_idx) > 0:
            idx = tuple_idx[0]
            filtered_polygons_tools = np.delete(polygons_tools, idx, axis=0)
            output[i] = polygons_polygon_intersection(
                filtered_polygons_tools, polygon_obj, eps
            )
        else:
            output[i] = polygons_polygon_intersection(
                polygons_tools, polygon_obj, eps
            )

    if deduplicate_lines_q:
        assert False # not implemented yet
        lines = np.empty((0, 2, 3), dtype=float)
        for lines_in_polygon in output:
            for line in lines_in_polygon:
                lines = np.append(lines, np.array([line]), axis=0)
        # Define domains to be subtracted
        lines_rounded = np.round(lines, decimals=collapse_precision)
        _, idx = np.unique(lines_rounded, axis=0, return_index=True)
        unique_lines = lines[idx]
        return unique_lines
    else:
        return output