import networkx as nx
import numpy as np
from numpy import linalg as la

import geometry.polygon_polygon_intersection_test as pp_intersector
import geometry.triangle_triangle_intersection_test as tt_intersector

from geometry.shape import Shape
from geometry.vertex import Vertex
from geometry.edge import Edge
from geometry.wire import Wire
from geometry.face import Face
from geometry.shell import Shell
from geometry.solid import Solid


class ShapeManipulation:
    def __init__(self, dimension):
        self.dimension = dimension

    @staticmethod
    def point_on_edge_boundary(point, edge):
        check = [
            np.all(np.isclose(point, vertex.point)) for vertex in edge.boundary_shapes
        ]
        if check[0]:
            return edge.boundary_shapes[0]
        elif check[1]:
            return edge.boundary_shapes[1]
        else:
            return None

    @staticmethod
    def embed_vertex_in_edge_boundary(vertex, edge):
        vertices_bc = [vertex_bc for vertex_bc in edge.boundary_shapes]
        check = [
            np.all(np.isclose(vertex.point, vertex_bc.point))
            for vertex_bc in vertices_bc
        ]
        if check[0]:
            edge.boundary_shapes[0] = vertex
        elif check[1]:
            edge.boundary_shapes[1] = vertex

    @staticmethod
    def embed_vertex_in_edge(vertices: np.ndarray, edge: Edge, tag_shift=0):

        # unify vertices
        points = np.array([vertex.point for vertex in vertices])
        unique_points, indices = np.unique(points, return_index=True, axis=0)
        vertices = vertices[indices]

        indices_on_bc = []
        for i, vertex in enumerate(vertices):
            obj_out = ShapeManipulation.point_on_edge_boundary(vertex.point, edge)
            if obj_out is not None:
                indices_on_bc.append(i)
                ShapeManipulation.embed_vertex_in_edge_boundary(vertex, edge)
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

    @staticmethod
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
        case_idx = 0
        vertex_idx = 0
        vertex_map = {}
        new_vertices_idx = []
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
                    obj_vertex = ShapeManipulation.point_on_edge_boundary(
                        new_point, edge_i
                    )
                    tool_vertex = ShapeManipulation.point_on_edge_boundary(
                        new_point, edge_j
                    )
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

                    # v = Vertex(v_tag, point)
                    # existence_check = np.array(
                    #     [
                    #         np.all(np.isclose(v.point, vertex.point))
                    #         for vertex in vertices
                    #     ]
                    # )
                    # if len(existence_check) != 0 and np.any(existence_check):
                    #     v_index = np.argwhere(existence_check)[0, 0]
                    #     vertex_map.__setitem__(case_idx, v_index)
                    #     v = vertices[v_index]
                    #     if obj_vertex is not None and tool_vertex is not None:
                    #         v.immersed_shapes = np.append(
                    #             v.immersed_shapes,
                    #             np.array([obj_vertex, tool_vertex]),
                    #             axis=0,
                    #         )
                    #     if obj_vertex is not None and tool_vertex is not None:
                    #         v.immersed_shapes = np.append(
                    #             v.immersed_shapes,
                    #             np.array([obj_vertex, tool_vertex]),
                    #             axis=0,
                    #         )
                    #     elif obj_vertex is not None:
                    #         v.immersed_shapes = np.append(
                    #             v.immersed_shapes,
                    #             np.array([obj_vertex]),
                    #             axis=0,
                    #         )
                    #     elif tool_vertex is not None:
                    #         v.immersed_shapes = np.append(
                    #             v.immersed_shapes,
                    #             np.array([tool_vertex]),
                    #             axis=0,
                    #         )
                    #     # else:
                    #     #     v.physical_tag = physical_tag
                    #     #     physical_tag += 1
                    #     #     vertices = np.append(vertices, np.array([v]), axis=0)
                    #     #     new_vertices_idx.append(vertex_idx)
                    #     #     v_tag += 1
                    # else:
                    #     # Edges with geometrically common vertices
                    #     if obj_vertex is not None and tool_vertex is not None:
                    #         # The directive in this case is to embed existing vertices in
                    #         # new vertex v with a physical tag representing intersections
                    #         v.physical_tag = physical_tag
                    #         physical_tag += 1
                    #         v.immersed_shapes = np.append(
                    #             v.immersed_shapes,
                    #             np.array([obj_vertex, tool_vertex]),
                    #             axis=0,
                    #         )
                    #         vertices = np.append(vertices, np.array([v]), axis=0)
                    #         new_vertices_idx.append(vertex_idx)
                    #         v_tag += 1
                    #     elif obj_vertex is not None:
                    #         # The directive in this case is to embed existing vertex in
                    #         # new vertex v with a physical tag representing intersections
                    #         v.physical_tag = physical_tag
                    #         physical_tag += 1
                    #         v.immersed_shapes = np.append(
                    #             v.immersed_shapes,
                    #             np.array([obj_vertex]),
                    #             axis=0,
                    #         )
                    #         vertices = np.append(vertices, np.array([v]), axis=0)
                    #         new_vertices_idx.append(vertex_idx)
                    #         v_tag += 1
                    #     elif tool_vertex is not None:
                    #         v.physical_tag = physical_tag
                    #         physical_tag += 1
                    #         v.immersed_shapes = np.append(
                    #             v.immersed_shapes,
                    #             np.array([tool_vertex]),
                    #             axis=0,
                    #         )
                    #         vertices = np.append(vertices, np.array([v]), axis=0)
                    #         new_vertices_idx.append(vertex_idx)
                    #         v_tag += 1
                    #     else:
                    #         v.physical_tag = physical_tag
                    #         physical_tag += 1
                    #         vertices = np.append(vertices, np.array([v]), axis=0)
                    #         new_vertices_idx.append(vertex_idx)
                    #         v_tag += 1
                    #     vertex_map.__setitem__(case_idx, vertex_idx)
                    #     vertex_idx += 1
                    # line_indices = np.vstack((line_indices, np.array([i, j])))
                    # case_idx += 1

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
                local_vertices = ShapeManipulation.embed_vertex_in_edge(
                    vertices_to_embed, edge_i, tag_shift=e_tag
                )
                if len(edge_i.immersed_shapes) > 0:
                    new_e_tags = np.array(
                        [shape.tag for shape in edge_i.immersed_shapes]
                    )
                    e_tag = np.max(new_e_tags) + 1
                    edges = np.append(edges, edge_i.immersed_shapes)

        return (edges, vertices)

    @staticmethod
    def embed_edge_in_face(edges: np.ndarray, face: Face):
        # wire_edges = [
        #     edge for wire in face.boundary_shapes for edge in wire.immersed_shapes
        # ]
        # for edge in edges:
        #     for i, vertex in enumerate(edge.boundary_shapes):
        #         for wire_edge in wire_edges:
        #             wire_vertex = ShapeManipulation.point_on_edge_boundary(
        #                 vertex.point, wire_edge
        #             )
        #             if wire_vertex is not None:
        #                 edge.boundary_shapes[i] = wire_vertex
        #                 break
        face.immersed_shapes = np.append(face.immersed_shapes, np.array([edges]))
