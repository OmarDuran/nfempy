import numpy as np
from scipy.spatial import Delaunay
from functools import partial


def __normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def __edge_direction(v0: np.array, v1: np.array, v2: np.array, normal: np.array):
    edge1 = v1 - v0
    edge2 = v2 - v1

    # Calculate the cross product of adjacent edges
    cross_product = np.cross(edge1, edge2)
    cross_product = __normalize(cross_product)
    direction = np.dot(normal, cross_product)
    return direction


def __projection_directions(points):
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])
    normal = polygon_normal(points)
    pos = np.argmax(
        np.abs([np.dot(normal, e1), np.dot(normal, e2), np.dot(normal, e3)])
    )
    return np.delete(np.array([0, 1, 2]), pos)


def polygon_normal(points: np.array):
    # centroid
    xc = np.mean(points, axis=0)

    # get two relevant in-plane directions
    dxc = np.array([np.linalg.norm(point - xc) for point in points])
    if np.isclose(np.max(dxc), np.min(dxc)):
        pa = points[0]
        pb = points[1]
        pc = points[2]
    else:
        pa = points[np.argmax(dxc)]
        points_red = np.delete(points, np.argmax(dxc), axis=0)
        dxc_red = np.delete(dxc, np.argmax(dxc), axis=0)
        pb = points_red[np.argmax(dxc_red)]
        pc = points_red[np.argmin(dxc_red)]

    # normal vector of the polygon
    normal = __normalize(np.cross(pb - pa, pc - pa))
    return normal


def convex_q(points: np.array):

    n_points = len(points)
    if n_points < 4:
        return True  # A polygon with less than 4 vertices is always convex

    normal = polygon_normal(points)

    # Check convexity by ensuring all edges turn in the same direction
    eged_directions = np.array(
        [
            __edge_direction(
                points[i],
                points[(i + 1) % n_points],
                points[(i + 2) % n_points],
                normal,
            )
            for i in range(n_points)
        ]
    )
    count = np.abs(np.sum(np.sign(eged_directions)))
    if int(count) == n_points:
        return True
    else:
        return False


def winding_number(p: np.array, polygon_points: np.array):
    """
    Compute the winding number of a point with respect to a polygon (planar angle method).
    """

    shifted_polygon = polygon_points - p

    # Compute the angles of each vertex with respect to the origin
    angles = np.arctan2(shifted_polygon[:, 1], shifted_polygon[:, 0])

    # Compute the difference between consecutive angles
    angle_diffs = np.diff(np.unwrap(angles))

    # Sum the differences to get the winding number
    winding_number = np.sum(angle_diffs) / (2 * np.pi)
    return int(round(winding_number))


def triangulate_convex_polygon(points: np.array):

    n_points = len(points)
    if n_points < 3:
        raise ValueError("A polygon must have at least 3 vertices.")

    triangles = []

    # Use the centroid as first point
    xc = np.mean(points, axis=0)

    # Create triangles with v0 and each pair of subsequent vertices
    for i in range(0, n_points - 1):
        v1 = points[i]
        v2 = points[i + 1]
        triangle = np.array([xc, v1, v2])
        triangles.append(triangle)
    # append last edge
    v1 = points[n_points - 1]
    v2 = points[0]
    triangle = np.array([xc, v1, v2])
    triangles.append(triangle)
    return np.array(triangles)


def triangulate_polygon(points: np.array):

    if convex_q(points):
        return triangulate_convex_polygon(points)

    # Drop a direction to project the 3D vertices to 2D
    dir_idx = __projection_directions(points)

    # Perform Delaunay triangulation on the 2D projected vertices
    delaunay = Delaunay(points[:, dir_idx])

    # Get the triangles
    triangles_R2 = delaunay.simplices

    # delete triangles outside the original polygon
    triangle_xc = np.mean(points[:, dir_idx][triangles_R2], axis=1)
    polygon_winding = partial(winding_number, polygon_points=points[:, dir_idx])
    valid_triangles = np.argwhere(list(map(polygon_winding, triangle_xc))).ravel()
    triangles_R2 = triangles_R2[valid_triangles]

    # Map the triangles back to 3D
    triangles = points[triangles_R2]
    return triangles
