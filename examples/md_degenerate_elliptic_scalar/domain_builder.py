import numpy as np

from globals import topology_tag_shape_info
from globals import geometry_collapse_precision as collapse_precision

from topology.domain import Domain
from topology.edge import Edge
from topology.face import Face
from topology.vertex import Vertex
from topology.wire import Wire

from topology.operations.domain_operations import create_domain
from topology.operations.domain_operations import domain_difference
from topology.operations.domain_operations import domain_union

from geometry.operations.point_geometry_operations import point_line_intersection
from geometry.operations.point_geometry_operations import points_line_intersection
from geometry.operations.point_geometry_operations import points_line_argsort
from geometry.operations.point_geometry_operations import points_polygon_intersection
from geometry.operations.line_geometry_operations import lines_lines_intersection


def build_surface_2D(surface_points, physical_tags=None):

    domain = Domain(dimension=2)
    vertices = np.array(
        [Vertex(tag, point) for tag, point in enumerate(surface_points)]
    )
    domain.append_shapes(vertices)

    edges_connectivities = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 0],
        [4, 1],
    ]
    edges_list = []
    e_physical_tags = {
        0: 3,
        1: 3,
        2: 4,
        3: 5,
        4: 5,
        5: 6,
        6: None,
    }
    for e_tag, edge_con in enumerate(edges_connectivities):
        edge = Edge(e_tag, domain.shapes[0][edge_con])
        edge.physical_tag = e_physical_tags[e_tag]
        edges_list.append(edge)

    edges = np.array(edges_list)
    domain.append_shapes(edges)

    edge_idxs = np.array([0, 6, 4, 5])
    vertices = np.array([domain.shapes[0][2], domain.shapes[0][2]])
    e_tag += 1
    wire = Wire(e_tag, edges[edge_idxs], vertices)
    domain.append_shapes(np.array([wire]))
    surface = Face(0, np.array([wire]))
    surface.physical_tag = physical_tags.get("area_0", None)
    domain.append_shapes(np.array([surface]))

    edge_idxs = np.array([1, 2, 3, 6])
    vertices = np.array([domain.shapes[0][0], domain.shapes[0][0]])
    e_tag += 1
    wire = Wire(e_tag, edges[edge_idxs], vertices)
    domain.append_shapes(np.array([wire]))
    surface = Face(1, np.array([wire]))
    surface.physical_tag = physical_tags.get("area_1", None)
    domain.append_shapes(np.array([surface]))

    domain.build_grahp()
    return domain


def create_md_box_2D(
    box_points: np.array,
    domain_physical_tags: dict,
    lines: np.array = None,
    fracture_physical_tags: dict = None,
    make_fitted_q: bool = False,
):
    # processing
    if domain_physical_tags is None:
        domain_physical_tags = {
            "area_0": 1,
            "area_1": 2,
            "bc_0": 3,
            "bc_1": 4,
            "bc_2": 5,
            "bc_3": 6,
        }
    offset = 1.0 / 6.0
    if make_fitted_q:
        offset = 0.0
    base_domain_points = np.insert(box_points, 1, np.array([offset, -1.0, 0]), axis=0)
    base_domain_points = np.insert(
        base_domain_points, 4, np.array([offset, +1.0, 0]), axis=0
    )

    rock_domain = build_surface_2D(base_domain_points, domain_physical_tags)
    if rock_domain.dimension != 2:
        raise ValueError(
            "Only 2D with 1D fractures settings are supported by this script."
        )

    if lines is None:
        return rock_domain
    if fracture_physical_tags is None:
        fracture_physical_tags = {"line": 100, "internal_bc": 200, "point": 300}

    rock_domain_vertices = [shape for shape in rock_domain.shapes[0]]
    rock_domain_faces = [shape for shape in rock_domain.shapes[2]]
    rock_domain_edges = [
        shape for shape in rock_domain.shapes[1] if not shape.composite
    ]
    rock_domain_lines = np.array(
        [shape.boundary_points() for shape in rock_domain_edges]
    )
    boundary_intx = lines_lines_intersection(
        lines_tools=lines, lines_objects=rock_domain_lines, deduplicate_points_q=True
    )
    fracture_intx = lines_lines_intersection(
        lines_tools=lines, lines_objects=lines, deduplicate_points_q=True
    )

    boundary_vertices = {}
    for i, line in enumerate(rock_domain_lines):
        a, b = line
        out, intx_idx = points_line_intersection(boundary_intx, a, b)
        if len(out) == 0:
            continue
        boundary_vertices[i] = out

    raw_vertices = []
    for item in boundary_vertices.items():
        i, intx_points = item
        for point in intx_points:
            v: Vertex = Vertex(topology_tag_shape_info.min, point)
            v.physical_tag = rock_domain_edges[i].physical_tag
            raw_vertices.append(v)

    # Eliminate duplicates values since face boundaries can not share a vertex (line boundary)
    b_points = np.array([vertex.point for vertex in raw_vertices])
    b_points_rounded = np.round(b_points, decimals=collapse_precision)
    _, unique_idx = np.unique(b_points_rounded, axis=0, return_index=True)

    tag = rock_domain.max_tag() + 1
    vertices = []
    for i, vertex in enumerate(raw_vertices):
        if i in unique_idx:
            vertex.tag = tag
            vertices.append(vertex)
            tag += 1

    for point in fracture_intx:
        v: Vertex = Vertex(tag, point)
        v.physical_tag = fracture_physical_tags["point"]
        vertices.append(v)
        tag += 1

    domain_c1 = create_domain(dimension=1, shapes=np.array(vertices))
    # domain_c1.build_grahp(0)
    # domain_c1.draw_grahp()

    edges = []
    for line in lines:
        edge_bc = []
        for point in line:
            v: Vertex = Vertex(tag, point)
            v.physical_tag = fracture_physical_tags["internal_bc"]
            vertices.append(v)
            edge_bc.append(v)
            tag += 1
        e: Edge = Edge(tag, np.array(edge_bc))
        e.physical_tag = fracture_physical_tags["line"]
        edges.append(e)
        tag += 1

    vertices = np.array(vertices)
    edges = np.array(edges)
    domain = create_domain(dimension=1, shapes=[])
    domain.append_shapes(rock_domain_vertices)
    domain.append_shapes(vertices)
    domain.append_shapes(rock_domain_edges)
    domain.append_shapes(edges)

    domain_c0 = domain_difference(domain, domain_c1, tag)
    md_domain_c1 = domain_union(domain_c0, domain_c1)

    # remove shapes outside the original domain
    vertices = np.array([vertex for vertex in md_domain_c1.shapes[0]])
    edges = np.array([edge for edge in md_domain_c1.shapes[1]])
    points = np.array([vertex.point for vertex in vertices])
    v_out, v_intx_q = points_polygon_intersection(points, box_points)
    edges_xcs = np.array([np.mean(edge.boundary_points(), axis=0) for edge in edges])
    e_out, e_intx_q = points_polygon_intersection(edges_xcs, box_points)

    vertices = vertices[v_intx_q]
    edges = edges[e_intx_q]

    faces = []
    wires = []
    fracture_edges = np.array(
        [edge for edge in edges if edge.physical_tag == fracture_physical_tags["line"]]
    )
    fracture_edges_xcs = np.array(
        [np.mean(edge.boundary_points(), axis=0) for edge in fracture_edges]
    )
    name_to_base_points = {
        "area_0": np.array([0, 1, 4, 5]),
        "area_1": np.array([1, 2, 3, 4]),
    }
    face_idx = 0
    for domain_face in rock_domain_faces:
        domain_name = "area_" + str(face_idx)
        wire_edges = []
        for boundary_edge in domain_face.boundary_shapes[0].immersed_shapes:
            a, b = boundary_edge.boundary_points()
            sub_edges = np.array(
                [
                    edge
                    for edge in edges
                    if point_line_intersection(
                        np.mean(edge.boundary_points(), axis=0), a, b
                    )
                    is not None
                ]
            )
            sub_edges_xcs = np.array(
                [np.mean(sub_edge.boundary_points(), axis=0) for sub_edge in sub_edges]
            )
            idx = points_line_argsort(sub_edges_xcs, a, b)
            wire_edges.append(sub_edges[idx])
        wire_edges = np.concatenate(wire_edges)
        wire_boundary = wire_edges[0].boundary_shapes[[0]]  # closed
        wire_i: Wire = Wire(tag, wire_edges, wire_boundary)
        tag += 1
        face_i: Face = Face(tag, np.array([wire_i]))
        # compute immersed_edges
        face_points = base_domain_points[name_to_base_points[domain_name]]
        v_out, v_intx_q = points_polygon_intersection(fracture_edges_xcs, face_points)
        face_i.immersed_shapes = fracture_edges[v_intx_q]

        face_i.physical_tag = domain_physical_tags[domain_name]
        tag += 1

        wires.append(wire_i)
        faces.append(face_i)
        face_idx += 1

    md_domain = create_domain(dimension=2, shapes=[])
    md_domain.append_shapes(vertices)
    md_domain.append_shapes(edges)
    md_domain.append_shapes(np.array(wires))
    md_domain.append_shapes(np.array(faces))
    return md_domain
