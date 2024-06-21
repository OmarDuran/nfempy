import numpy as np
from numpy import linalg as la

from globals import topology_collapse_tol as collapse_tol
from topology.vertex import Vertex

import topology.polygon_polygon_intersection_test as pp_intersector
import topology.triangle_triangle_intersection_test as tt_intersector
from topology.edge import Edge
from topology.wire import Wire
from topology.face import Face
from topology.shell import Shell
from topology.shape import Shape


def vertex_with_same_geometry(
    vertex_a: Vertex, vertex_b: Vertex, eps: float = collapse_tol
):
    point_a = vertex_a.point
    point_b = vertex_b.point
    same_geometry_q = np.all(np.isclose(point_a, point_b, rtol=eps, atol=eps))
    return same_geometry_q


def vertex_strong_equality(vertex_a: Vertex, vertex_b: Vertex):
    shape_equality_q = vertex_a == vertex_b
    geometry_equality_q = vertex_with_same_geometry(vertex_a, vertex_b)
    strong_equality = shape_equality_q and geometry_equality_q
    return strong_equality


def collapse_vertex(
    v_tool: Vertex, v_object: Vertex, eps: float = collapse_tol
) -> Vertex:
    if vertex_strong_equality(v_tool, v_object):
        return v_tool
    elif vertex_with_same_geometry(v_tool, v_object, eps):
        v_object.shape_assignment(v_tool)
        return v_object
    else:
        return v_object


def vertex_edge_boundary_intersection(
    v_tool: Vertex, edge_object: Edge, eps: float = collapse_tol
):
    check = [
        vertex_with_same_geometry(v_tool, e_vertex, eps)
        for e_vertex in edge_object.boundary_shapes
    ]
    if check[0]:
        return edge_object.boundary_shapes[0]
    elif check[1]:
        return edge_object.boundary_shapes[1]
    else:
        return None


def point_line_intersection(A, B, P):
    x1, y1, z1 = A
    x2, y2, z2 = B
    x, y, z = P

    # If A and B are the same point, check if P is also the same point
    if (x1 == x2) and (y1 == y2) and (z1 == z2):
        return (x == x1) and (y == y1) and (z == z1)

    # Avoid division by zero by checking if the line segment is vertical in any dimension
    if x2 != x1:
        t_x = (x - x1) / (x2 - x1)
    else:
        t_x = None

    if y2 != y1:
        t_y = (y - y1) / (y2 - y1)
    else:
        t_y = None

    if z2 != z1:
        t_z = (z - z1) / (z2 - z1)
    else:
        t_z = None

    # Compare t values while ignoring None
    t_values = [t for t in [t_x, t_y, t_z] if t is not None]

    # If there are no valid t values, A and B are the same point and P is not that point
    if not t_values:
        return False

    # Check if all non-None t values are the same
    return all(t == t_values[0] for t in t_values)


def vertex_edge_intersection(
    v_tool: Vertex, edge_object: Edge, eps: float = collapse_tol
):

    bc_out = vertex_edge_boundary_intersection(v_tool, edge_object, eps)
    if bc_out is not None:
        return bc_out

    check = [
        vertex_with_same_geometry(v_tool, e_vertex)
        for e_vertex in edge_object.boundary_shapes
    ]
    if check[0]:
        return edge_object.boundary_shapes[0]
    elif check[1]:
        return edge_object.boundary_shapes[1]
    else:
        return None


def point_on_edge_boundary(point, edge):
    check = [np.all(np.isclose(point, vertex.point)) for vertex in edge.boundary_shapes]
    if check[0]:
        return edge.boundary_shapes[0]
    elif check[1]:
        return edge.boundary_shapes[1]
    else:
        return None


def points_on_face_boundary(points, face):
    wires = face.boundary_shapes
    for wire in wires:
        for f_edge in wire.immersed_shapes:
            f_points = f_edge.boundary_points()
            if np.all(np.isclose(f_points[[0, 1]], points)):
                return f_edge
            elif np.all(np.isclose(f_points[[1, 0]], points)):
                return f_edge
    return None


def embed_vertex_in_edge_boundary(vertex, edge):
    vertices_bc = [vertex_bc for vertex_bc in edge.boundary_shapes]
    check = [
        np.all(np.isclose(vertex.point, vertex_bc.point)) for vertex_bc in vertices_bc
    ]
    if check[0]:
        edge.boundary_shapes[0] = vertex
    elif check[1]:
        edge.boundary_shapes[1] = vertex


def embed_vertex_in_edge(vertices: np.ndarray, edge: Edge, tag_shift=0):
    # unify vertices
    points = np.array([vertex.point for vertex in vertices])
    unique_points, indices = np.unique(points, return_index=True, axis=0)
    vertices = vertices[indices]

    indices_on_bc = []
    for i, vertex in enumerate(vertices):
        obj_out = point_on_edge_boundary(vertex.point, edge)
        if obj_out is not None:
            indices_on_bc.append(i)
            embed_vertex_in_edge_boundary(vertex, edge)
            continue

    a: Vertex = edge.boundary_shapes[0]
    b: Vertex = edge.boundary_shapes[1]
    translation = a.point
    a.point = a.point - translation
    b.point = b.point - translation

    ra_norm = la.norm(a.point, axis=0)
    rb_norm = la.norm(b.point, axis=0)

    r_norms = [ra_norm, rb_norm]
    valid_indices = []
    for i, vertex in enumerate(vertices):
        if i in indices_on_bc:
            continue
        vertex.point = vertex.point - translation
        r_norm = la.norm(vertex.point, axis=0)
        if ra_norm < r_norm and r_norm < rb_norm:
            is_a = np.isclose(r_norm, ra_norm)
            is_b = np.isclose(r_norm, rb_norm)
            if not is_a and not is_b:
                valid_indices.append(i)
                r_norms.append(r_norm)
        vertex.point = vertex.point + translation
    a.point = a.point + translation
    b.point = b.point + translation
    if len(r_norms) > 2:
        perm = np.argsort(r_norms)
        vertex_list = [a, b] + list(vertices[valid_indices])
        vertex_list = [vertex_list[i] for i in perm]
        for i in range(len(vertex_list) - 1):
            e = Edge(i + tag_shift, np.array([vertex_list[i], vertex_list[i + 1]]))
            edge.immersed_shapes = np.append(edge.immersed_shapes, e)
    return vertices[valid_indices]


def intersect_edges(
    edges_obj: np.ndarray,
    edges_tool: np.ndarray,
    axis=2,
    e_tag_shift=0,
    v_tag_shift=0,
    p_tag_shift=0,
    render_intersection_q=False,
):
    pos = np.array([0, 1])
    if axis == 0:
        pos = np.array([1, 2])
    elif axis == 1:
        pos = np.array([0, 2])

    vertices = np.array([], dtype=Shape)
    points = np.empty(shape=(0, 3), dtype=float)
    case_line_indices = np.empty(shape=(0, 2), dtype=int)
    edges = np.array([], dtype=Shape)
    # Collects intersection vertices

    tt_test = tt_intersector.TriangleTriangleIntersectionTest()

    v_tag = v_tag_shift
    e_tag = e_tag_shift

    physical_tag = p_tag_shift + 1
    vertex_map = {}
    for i, edge_i in enumerate(edges_obj):
        if isinstance(edge_i, Wire):
            continue
        a, b = [shape.point for shape in edge_i.boundary_shapes]
        for j, edge_j in enumerate(edges_tool):
            if isinstance(edge_j, Wire):
                continue
            p, q = [shape.point for shape in edge_j.boundary_shapes]
            if i >= j:
                continue
            intersection_data = tt_test.line_line_intersection(
                a, b, p, q, pos, render_intersection_q
            )
            if intersection_data[0]:
                new_point = intersection_data[1]
                obj_vertex = point_on_edge_boundary(new_point, edge_i)
                tool_vertex = point_on_edge_boundary(new_point, edge_j)
                # Connected edges by a common vertex are ignored
                if obj_vertex is not None and tool_vertex is not None:
                    if obj_vertex == tool_vertex:
                        continue

                existence_check = np.array(
                    [np.all(np.isclose(new_point, point)) for point in points]
                )

                # new approach
                if len(existence_check) != 0 and np.any(existence_check):
                    v_index = np.argwhere(existence_check)[0, 0]
                    points = np.vstack((points, points[v_index]))
                else:
                    points = np.vstack((points, new_point))
                case_line_indices = np.vstack((case_line_indices, np.array([i, j])))

    if len(points) > 0:
        # map recurrences and make geometry unique
        unique_points, indices, inv_indices = np.unique(
            points, return_index=True, return_inverse=True, axis=0
        )

        # create unique vertices
        for point in unique_points:
            v = Vertex(v_tag, point)
            v.physical_tag = physical_tag
            v_tag += 1
            physical_tag += 1
            vertices = np.append(vertices, np.array([v]), axis=0)

        cases_idx = list(range(len(points)))
        vertex_map = dict(zip(cases_idx, vertices[inv_indices]))
        for i, edge_i in enumerate(edges_obj):
            # print("edge index: ", i)
            if isinstance(edge_i, Wire):
                continue
            case_indices = np.argwhere(case_line_indices == i)[:, 0]
            if len(case_indices) == 0:
                continue
            vertices_to_embed = np.array([vertex_map[i] for i in case_indices])
            _ = embed_vertex_in_edge(vertices_to_embed, edge_i, tag_shift=e_tag)
            if len(edge_i.immersed_shapes) > 0:
                new_e_tags = np.array([shape.tag for shape in edge_i.immersed_shapes])
                e_tag = np.max(new_e_tags) + 1
                edges = np.append(edges, edge_i.immersed_shapes)

    return (edges, vertices)


def embed_edge_in_face(edges: np.ndarray, face: Face):
    face.immersed_shapes = np.append(face.immersed_shapes, np.array([edges]))


def intersect_faces(
    faces_obj: np.ndarray,
    faces_tool: np.ndarray,
    f_tag_shift=0,
    e_tag_shift=0,
    v_tag_shift=0,
    p_tag_shift=0,
    render_intersection_q=False,
):
    vertices = np.array([], dtype=Shape)
    edges = np.array([], dtype=Shape)
    point_pairs = np.empty(shape=(0, 2, 3), dtype=float)
    case_plane_indices = np.empty(shape=(0, 2), dtype=int)

    # Collects intersection vertices and edges
    pp_test = pp_intersector.PolygonPolygonIntersectionTest()

    v_tag = v_tag_shift
    e_tag = e_tag_shift
    f_tag = f_tag_shift

    physical_tag = p_tag_shift + 1
    vertex_map = {}
    for i, face_i in enumerate(faces_obj):
        if isinstance(face_i, Shell):
            continue
        face_i_vertices = face_i.boundary_shapes[0].orient_immersed_vertices()
        face_i_pts = np.array([vertex.point for vertex in face_i_vertices])
        for j, face_j in enumerate(faces_tool):
            if isinstance(face_j, Shell):
                continue
            if i >= j:
                continue
            face_j_vertices = face_j.boundary_shapes[0].orient_immersed_vertices()
            face_j_pts = np.array([vertex.point for vertex in face_j_vertices])

            intersection_data = pp_test.polygon_polygon_intersection(
                face_i_pts, face_j_pts, render_intersection_q
            )
            if intersection_data[0]:
                new_points = np.array([intersection_data[1], intersection_data[2]])
                obj_edge = points_on_face_boundary(new_points, face_i)
                tool_edge = points_on_face_boundary(new_points, face_j)
                # Connected faces by a common edge are ignored
                if obj_edge is not None and tool_edge is not None:
                    if obj_edge == tool_edge:
                        continue

                existence_check = np.array(
                    [
                        np.all(np.isclose(new_points, point_pair))
                        for point_pair in point_pairs
                    ]
                )

                # new approach
                if len(existence_check) != 0 and np.any(existence_check):
                    v_index = np.argwhere(existence_check)[0, 0]
                    point_pairs = np.append((point_pairs, point_pairs[v_index]), axis=0)
                else:
                    point_pairs = np.append(point_pairs, np.array([new_points]), axis=0)
                case_plane_indices = np.vstack((case_plane_indices, np.array([i, j])))

    if len(point_pairs) > 0:
        # map recurrences and make geometry unique
        unique_point_pairs, indices, inv_indices = np.unique(
            point_pairs, return_index=True, return_inverse=True, axis=0
        )

        # create unique egdes
        for point_pair in unique_point_pairs:
            v0 = Vertex(v_tag, point_pair[0])
            vertices = np.append(vertices, np.array([v0]), axis=0)
            v_tag += 1

            v1 = Vertex(v_tag, point_pair[1])
            vertices = np.append(vertices, np.array([v1]), axis=0)
            v_tag += 1

            e = Edge(e_tag, np.array([v0, v1]))
            e.physical_tag = physical_tag
            e_tag += 1
            physical_tag += 1
            edges = np.append(edges, np.array([e]), axis=0)

        cases_idx = list(range(len(point_pairs)))
        edge_map = dict(zip(cases_idx, edges[inv_indices]))

        for i, face_i in enumerate(faces_obj):
            # print("face index: ", i)
            if isinstance(face_i, Shell):
                continue
            case_indices = np.argwhere(case_plane_indices == i)[:, 0]
            if len(case_indices) == 0:
                continue
            # line line intersections
            face_i_edges = face_i.boundary_shapes[0].immersed_shapes
            face_i_vertices = face_i.boundary_shapes[0].orient_immersed_vertices()
            face_i_pts = np.array([vertex.point for vertex in face_i_vertices])
            dir = (face_i_pts[2] - face_i_pts[0]) / np.linalg.norm(
                face_i_pts[2] - face_i_pts[0]
            )
            axis_dir = np.argmin(np.abs(dir))

            edges_to_embed = np.array([edge_map[i] for i in case_indices])

            # performing multiple intersection of connected and disjointed edges
            edges_obj = np.insert(edges_to_embed, 0, face_i_edges, axis=0)
            edges_tool = np.insert(edges_to_embed, 0, face_i_edges, axis=0)
            (frag_edges, frag_vertices) = intersect_edges(
                edges_obj,
                edges_tool,
                axis=axis_dir,
                v_tag_shift=v_tag,
                e_tag_shift=e_tag,
                p_tag_shift=physical_tag,
            )

            embed_edge_in_face(edges_to_embed, face_i)

            if len(frag_vertices) > 0:
                new_v_tags = np.array([shape.tag for shape in frag_vertices])
                v_tag = np.max(new_v_tags) + 1
                vertices = np.append(vertices, frag_vertices)

            if len(frag_edges) > 0:
                new_e_tags = np.array([shape.tag for shape in frag_edges])
                e_tag = np.max(new_e_tags) + 1
                edges = np.append(edges, frag_edges)
    return (edges, vertices)
