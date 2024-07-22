import numpy as np
from scipy.spatial import Delaunay
from functools import partial

from globals import geometry_collapse_precision as collapse_precision
from globals import geometry_line_polygon_incidence_tol as s_incidence_tol
from geometry.operations.point_geometry_operations import coplanar_measurement
from geometry.operations.line_geometry_operations import lines_triangle_intersection
from geometry.operations.polygon_operations import triangulate_polygon

def triangle_triangle_intersection(
    triangle_tool: np.array, triangle_object: np.array, eps: float = s_incidence_tol
) -> np.array:

    p, q ,r = triangle_tool
    a, b, c = triangle_object
    p_measurement = coplanar_measurement(p, a, b, c)
    q_measurement = coplanar_measurement(q, a, b, c)
    r_measurement = coplanar_measurement(r, a, b, c)

    p_coplanar_q = np.isclose(p_measurement,0.0, rtol=eps, atol=eps)
    q_coplanar_q = np.isclose(q_measurement, 0.0, rtol=eps, atol=eps)
    r_coplanar_q = np.isclose(r_measurement, 0.0, rtol=eps, atol=eps)

    # perform lines - triangle intersection
    lines_tool = np.array([[p, q], [q, r], [r, p]])
    intx_data, intx_q = lines_triangle_intersection(lines_tool,a,b,c,eps)

    t_are_coplanar_q = np.all([p_coplanar_q,q_coplanar_q,r_coplanar_q])
    if t_are_coplanar_q and np.count_nonzero(intx_q) > 1:
        raise ValueError("Intersection of coplanar triangles forms an area.")

    if np.count_nonzero(intx_q) == 0:
        return None

    # Define domains to be subtracted
    intx_data_rounded = np.round(intx_data, decimals=collapse_precision)
    _, idx = np.unique(intx_data_rounded, axis=0, return_index=True)
    intx_data = intx_data[idx]
    if len(intx_data.shape) == 3:
        intx_data = intx_data[:,0,:]
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

    xc = np.mean(intx_data,axis=0)
    dirs = intx_data - xc
    v = intx_data[0] - xc
    idx = np.argsort(np.dot(dirs, v))
    intx_data = intx_data[idx[np.array([0, -1])]]
    return intx_data
