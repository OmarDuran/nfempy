import numpy as np
import os
from topology.vertex import Vertex
from topology.domain import Domain
from topology.domain_market import (
    build_box_1D,
    build_box_2D,
)
from topology.domain_market import build_box_2D_with_lines
from topology.shape_manipulation import ShapeManipulation

from mesh.conformal_mesher import ConformalMesher


def generate_domain_1d():
    box_points = np.array([[0, 0, 0], [1, 0, 0]])
    domain = build_box_1D(box_points)
    domain.build_grahp()
    return domain


def generate_domain_2d():
    box_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    domain = build_box_2D(box_points)
    domain.build_grahp()
    return domain


def generate_domain_2d_with_lines(lines_file):
    file_name = os.getcwd() + "/tests/test_geometry/" + lines_file
    box_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    domain = build_box_2D_with_lines(box_points, file_name)
    domain.build_grahp()
    return domain


def test_domain_1d():
    domain: Domain = generate_domain_1d()
    vertex_0 = domain.shapes[0][0]
    vertex_1 = domain.shapes[0][1]
    edge_0 = domain.shapes[1][0]

    predecessors_vertex_0 = list(domain.graph.predecessors((1, vertex_0.tag)))[0]
    predecessors_vertex_1 = list(domain.graph.predecessors((1, vertex_1.tag)))[0]
    assert predecessors_vertex_0 == predecessors_vertex_1

    edge_0_successors = list(domain.graph.successors((0, edge_0.tag)))
    assert (1, vertex_0.tag) == edge_0_successors[0]
    assert (1, vertex_1.tag) == edge_0_successors[1]


def test_domain_1d_vertex_in_edge():
    domain: Domain = generate_domain_1d()
    edge_0 = domain.shapes[1][0]

    # max tag per dimension
    max_vertex_tag = len(domain.shapes[0])
    max_edge_tag = len(domain.shapes[1])

    immersed_points = np.array([[0.25, 0, 0], [0.75, 0, 0], [1.75, 0, 0]])

    physical_tags = [4, 5, 6, 7, 8]
    vertices = np.array(
        [
            Vertex(tag + max_vertex_tag, point)
            for tag, point in enumerate(immersed_points)
        ]
    )
    for vertex, physical_tag in zip(vertices, physical_tags):
        vertex.physical_tag = physical_tag

    # embed_vertex_in_edge will filter points not in the line
    vertices = ShapeManipulation.embed_vertex_in_edge(
        vertices, edge_0, tag_shift=max_edge_tag
    )
    domain.append_shapes(vertices)
    domain.append_shapes(edge_0.immersed_shapes)
    domain.build_grahp()

    vertex_0 = domain.shapes[0][0]
    vertex_1 = domain.shapes[0][1]
    vertex_2 = domain.shapes[0][2]
    vertex_3 = domain.shapes[0][3]
    edge_1 = domain.shapes[1][1]
    edge_2 = domain.shapes[1][2]
    edge_3 = domain.shapes[1][3]

    predecessors_vertex_0 = list(domain.graph.predecessors((1, vertex_0.tag)))
    predecessors_vertex_1 = list(domain.graph.predecessors((1, vertex_1.tag)))
    assert predecessors_vertex_0[0] == predecessors_vertex_1[0]
    assert (0, edge_1.tag) == predecessors_vertex_0[1]
    assert (0, edge_3.tag) == predecessors_vertex_1[1]

    predecessors_vertex_2 = list(domain.graph.predecessors((1, vertex_2.tag)))
    assert (0, edge_1.tag) == predecessors_vertex_2[0]
    assert (0, edge_2.tag) == predecessors_vertex_2[1]
    predecessors_vertex_3 = list(domain.graph.predecessors((1, vertex_3.tag)))
    assert (0, vertex_2.tag) == predecessors_vertex_3[0]
    assert (0, vertex_3.tag) == predecessors_vertex_3[1]

    edge_0_successors = list(domain.graph.successors((0, edge_0.tag)))
    assert (1, vertex_0.tag) == edge_0_successors[0]
    assert (1, vertex_1.tag) == edge_0_successors[1]
    assert (0, edge_1.tag) == edge_0_successors[2]
    assert (0, edge_2.tag) == edge_0_successors[3]
    assert (0, edge_3.tag) == edge_0_successors[4]


def test_domain_2d():
    domain: Domain = generate_domain_2d()
    vertex_0 = domain.shapes[0][0]
    vertex_1 = domain.shapes[0][1]
    edge_0 = domain.shapes[1][0]
    edge_1 = domain.shapes[1][1]
    edge_2 = domain.shapes[1][2]
    edge_3 = domain.shapes[1][3]
    wire_4 = domain.shapes[1][4]
    face_0 = domain.shapes[0][0]

    predecessors_edge_0 = list(domain.graph.predecessors((1, edge_0.tag)))[0]
    predecessors_edge_1 = list(domain.graph.predecessors((1, edge_1.tag)))[0]
    predecessors_edge_2 = list(domain.graph.predecessors((1, edge_2.tag)))[0]
    predecessors_edge_3 = list(domain.graph.predecessors((1, edge_3.tag)))[0]
    assert (1, wire_4.tag) == predecessors_edge_0
    assert (1, wire_4.tag) == predecessors_edge_1
    assert (1, wire_4.tag) == predecessors_edge_2
    assert (1, wire_4.tag) == predecessors_edge_3

    face_0_successors = list(domain.graph.successors((0, face_0.tag)))
    assert (1, wire_4.tag) == face_0_successors[0]


def test_domain_2d_line_in_face():
    lines_file = "lines/single_shape.csv"
    domain: Domain = generate_domain_2d_with_lines(lines_file)
    vertex_0 = domain.shapes[0][0]
    vertex_1 = domain.shapes[0][1]
    vertex_2 = domain.shapes[0][2]
    vertex_3 = domain.shapes[0][3]
    vertex_4 = domain.shapes[0][4]
    vertex_5 = domain.shapes[0][5]
    edge_0 = domain.shapes[1][0]
    edge_1 = domain.shapes[1][1]
    edge_2 = domain.shapes[1][2]
    edge_3 = domain.shapes[1][3]
    wire_4 = domain.shapes[1][4]
    edge_5 = domain.shapes[1][5]
    face_0 = domain.shapes[0][0]

    predecessors_edge_0 = list(domain.graph.predecessors((1, edge_0.tag)))[0]
    predecessors_edge_1 = list(domain.graph.predecessors((1, edge_1.tag)))[0]
    predecessors_edge_2 = list(domain.graph.predecessors((1, edge_2.tag)))[0]
    predecessors_edge_3 = list(domain.graph.predecessors((1, edge_3.tag)))[0]
    assert (1, wire_4.tag) == predecessors_edge_0
    assert (1, wire_4.tag) == predecessors_edge_1
    assert (1, wire_4.tag) == predecessors_edge_2
    assert (1, wire_4.tag) == predecessors_edge_3

    predecessors_vertex_4 = list(domain.graph.predecessors((2, vertex_4.tag)))[0]
    predecessors_vertex_5 = list(domain.graph.predecessors((2, vertex_5.tag)))[0]
    assert predecessors_vertex_4 == predecessors_vertex_5

    face_0_successors = list(domain.graph.successors((0, face_0.tag)))
    assert (1, wire_4.tag) == face_0_successors[0]
    assert (1, edge_5.tag) == face_0_successors[1]


def test_domain_2d_line_in_face_bc_intersection():
    lines_file = "lines/single_shape_bc_intersection.csv"
    domain: Domain = generate_domain_2d_with_lines(lines_file)
    edge_0 = domain.shapes[1][0]
    edge_1 = domain.shapes[1][1]
    edge_2 = domain.shapes[1][2]
    edge_3 = domain.shapes[1][3]
    wire_4 = domain.shapes[1][4]
    edge_5 = domain.shapes[1][5]
    face_0 = domain.shapes[0][0]

    # wire features
    assert len(wire_4.immersed_shapes) == 6
    assert len(wire_4.orientation) == 6

    predecessors_wire_4 = list(domain.graph.successors((1, wire_4.tag)))
    assert (1, edge_1.immersed_shapes[0].tag) == predecessors_wire_4[2]
    assert (1, edge_1.immersed_shapes[1].tag) == predecessors_wire_4[3]
    assert (1, edge_2.tag) == predecessors_wire_4[4]
    assert (1, edge_3.immersed_shapes[0].tag) == predecessors_wire_4[5]
    assert (1, edge_3.immersed_shapes[1].tag) == predecessors_wire_4[6]

    vertex_4 = domain.lookup_vertex(np.array([0.0, 0.5, 0.0]))
    vertex_5 = domain.lookup_vertex(np.array([1.0, 0.5, 0.0]))
    predecessors_vertex_4 = list(domain.graph.predecessors((2, vertex_4.tag)))
    predecessors_vertex_5 = list(domain.graph.predecessors((2, vertex_5.tag)))
    assert (1, edge_3.immersed_shapes[0].tag) == predecessors_vertex_4[0]
    assert (1, edge_3.immersed_shapes[1].tag) == predecessors_vertex_4[1]
    assert (1, edge_1.immersed_shapes[0].tag) == predecessors_vertex_5[0]
    assert (1, edge_1.immersed_shapes[1].tag) == predecessors_vertex_5[1]
    assert predecessors_vertex_4[2] == predecessors_vertex_5[2]

    face_0_successors = list(domain.graph.successors((0, face_0.tag)))
    assert (1, wire_4.tag) == face_0_successors[0]
    assert (1, edge_5.tag) == face_0_successors[1]


def test_domain_2d_cross_shape_in_face():
    lines_file = "lines/cross_shape_lines.csv"
    domain: Domain = generate_domain_2d_with_lines(lines_file)
    vertex_0 = domain.shapes[0][0]
    vertex_1 = domain.shapes[0][1]
    vertex_2 = domain.shapes[0][2]
    vertex_3 = domain.shapes[0][3]
    vertex_4 = domain.shapes[0][4]
    vertex_5 = domain.shapes[0][5]
    vertex_6 = domain.shapes[0][6]
    vertex_7 = domain.shapes[0][7]
    vertex_8 = domain.shapes[0][8]
    edge_0 = domain.shapes[1][0]
    edge_1 = domain.shapes[1][1]
    edge_2 = domain.shapes[1][2]
    edge_3 = domain.shapes[1][3]
    wire_4 = domain.shapes[1][4]
    edge_5 = domain.shapes[1][5]
    edge_6 = domain.shapes[1][6]
    face_0 = domain.shapes[0][0]

    # wire features
    assert len(wire_4.immersed_shapes) == 4
    assert len(wire_4.orientation) == 4

    predecessors_wire_4 = list(domain.graph.successors((1, wire_4.tag)))
    assert (2, vertex_0.tag) == predecessors_wire_4[0]
    assert (1, edge_0.tag) == predecessors_wire_4[1]
    assert (1, edge_1.tag) == predecessors_wire_4[2]
    assert (1, edge_2.tag) == predecessors_wire_4[3]
    assert (1, edge_3.tag) == predecessors_wire_4[4]

    predecessors_vertex_8 = list(domain.graph.predecessors((2, vertex_8.tag)))
    assert (1, edge_5.immersed_shapes[0].tag) == predecessors_vertex_8[0]
    assert (1, edge_5.immersed_shapes[1].tag) == predecessors_vertex_8[1]
    assert (1, edge_6.immersed_shapes[0].tag) == predecessors_vertex_8[2]
    assert (1, edge_6.immersed_shapes[1].tag) == predecessors_vertex_8[3]

    face_0_successors = list(domain.graph.successors((0, face_0.tag)))
    assert (1, wire_4.tag) == face_0_successors[0]
    assert (1, edge_5.tag) == face_0_successors[1]
    assert (1, edge_6.tag) == face_0_successors[2]


def test_domain_2d_cross_shape_in_face_bc_intersection():
    lines_file = "lines/cross_shape_bc_intersection.csv"
    domain: Domain = generate_domain_2d_with_lines(lines_file)
    edge_0 = domain.shapes[1][0]
    edge_1 = domain.shapes[1][1]
    edge_2 = domain.shapes[1][2]
    edge_3 = domain.shapes[1][3]
    wire_4 = domain.shapes[1][4]
    edge_5 = domain.shapes[1][5]
    edge_6 = domain.shapes[1][6]
    face_0 = domain.shapes[0][0]

    # wire features
    assert len(wire_4.immersed_shapes) == 8
    assert len(wire_4.orientation) == 8

    predecessors_wire_4 = list(domain.graph.successors((1, wire_4.tag)))
    assert (1, edge_0.immersed_shapes[0].tag) == predecessors_wire_4[1]
    assert (1, edge_0.immersed_shapes[1].tag) == predecessors_wire_4[2]
    assert (1, edge_1.immersed_shapes[0].tag) == predecessors_wire_4[3]
    assert (1, edge_1.immersed_shapes[1].tag) == predecessors_wire_4[4]
    assert (1, edge_2.immersed_shapes[0].tag) == predecessors_wire_4[5]
    assert (1, edge_2.immersed_shapes[1].tag) == predecessors_wire_4[6]
    assert (1, edge_3.immersed_shapes[0].tag) == predecessors_wire_4[7]
    assert (1, edge_3.immersed_shapes[1].tag) == predecessors_wire_4[8]

    vertex_4 = domain.lookup_vertex(np.array([0.0, 0.5, 0.0]))
    predecessors_vertex_4 = list(domain.graph.predecessors((2, vertex_4.tag)))
    assert (1, edge_3.immersed_shapes[0].tag) == predecessors_vertex_4[0]
    assert (1, edge_3.immersed_shapes[1].tag) == predecessors_vertex_4[1]
    assert (1, edge_5.tag) == predecessors_vertex_4[2]
    assert (1, edge_5.immersed_shapes[0].tag) == predecessors_vertex_4[3]

    vertex_5 = domain.lookup_vertex(np.array([0.5, 0.0, 0.0]))
    predecessors_vertex_5 = list(domain.graph.predecessors((2, vertex_5.tag)))
    assert (1, edge_0.immersed_shapes[0].tag) == predecessors_vertex_5[0]
    assert (1, edge_0.immersed_shapes[1].tag) == predecessors_vertex_5[1]
    assert (1, edge_6.tag) == predecessors_vertex_5[2]
    assert (1, edge_6.immersed_shapes[0].tag) == predecessors_vertex_5[3]

    vertex_6 = domain.lookup_vertex(np.array([0.5, 0.5, 0.0]))
    predecessors_vertex_6 = list(domain.graph.predecessors((2, vertex_6.tag)))
    assert (1, edge_5.immersed_shapes[0].tag) == predecessors_vertex_6[0]
    assert (1, edge_5.immersed_shapes[1].tag) == predecessors_vertex_6[1]
    assert (1, edge_6.immersed_shapes[0].tag) == predecessors_vertex_6[2]
    assert (1, edge_6.immersed_shapes[1].tag) == predecessors_vertex_6[3]

    vertex_7 = domain.lookup_vertex(np.array([0.5, 1.0, 0.0]))
    predecessors_vertex_7 = list(domain.graph.predecessors((2, vertex_7.tag)))
    assert (1, edge_2.immersed_shapes[0].tag) == predecessors_vertex_7[0]
    assert (1, edge_2.immersed_shapes[1].tag) == predecessors_vertex_7[1]
    assert (1, edge_6.tag) == predecessors_vertex_7[2]
    assert (1, edge_6.immersed_shapes[1].tag) == predecessors_vertex_7[3]

    vertex_8 = domain.lookup_vertex(np.array([1.0, 0.5, 0.0]))
    predecessors_vertex_8 = list(domain.graph.predecessors((2, vertex_8.tag)))
    assert (1, edge_1.immersed_shapes[0].tag) == predecessors_vertex_8[0]
    assert (1, edge_1.immersed_shapes[1].tag) == predecessors_vertex_8[1]
    assert (1, edge_5.tag) == predecessors_vertex_8[2]
    assert (1, edge_5.immersed_shapes[1].tag) == predecessors_vertex_8[3]

    face_0_successors = list(domain.graph.successors((0, face_0.tag)))
    assert (1, wire_4.tag) == face_0_successors[0]
    assert (1, edge_5.tag) == face_0_successors[1]
    assert (1, edge_6.tag) == face_0_successors[2]


def test_domain_2d_t_shape_in_face_bc_intersection():
    lines_file = "lines/t_shape_bc_intersection.csv"
    domain: Domain = generate_domain_2d_with_lines(lines_file)
    edge_0 = domain.shapes[1][0]
    edge_1 = domain.shapes[1][1]
    edge_2 = domain.shapes[1][2]
    edge_3 = domain.shapes[1][3]
    wire_4 = domain.shapes[1][4]
    edge_5 = domain.shapes[1][5]
    edge_6 = domain.shapes[1][6]
    face_0 = domain.shapes[0][0]

    # wire features
    assert len(wire_4.immersed_shapes) == 7
    assert len(wire_4.orientation) == 7

    predecessors_wire_4 = list(domain.graph.successors((1, wire_4.tag)))
    assert (1, edge_0.immersed_shapes[0].tag) == predecessors_wire_4[1]
    assert (1, edge_0.immersed_shapes[1].tag) == predecessors_wire_4[2]
    assert (1, edge_1.immersed_shapes[0].tag) == predecessors_wire_4[3]
    assert (1, edge_1.immersed_shapes[1].tag) == predecessors_wire_4[4]
    assert (1, edge_2.tag) == predecessors_wire_4[5]
    assert (1, edge_3.immersed_shapes[0].tag) == predecessors_wire_4[6]
    assert (1, edge_3.immersed_shapes[1].tag) == predecessors_wire_4[7]

    vertex_4 = domain.lookup_vertex(np.array([0.0, 0.75, 0.0]))
    predecessors_vertex_4 = list(domain.graph.predecessors((2, vertex_4.tag)))
    assert (1, edge_3.immersed_shapes[0].tag) == predecessors_vertex_4[0]
    assert (1, edge_3.immersed_shapes[1].tag) == predecessors_vertex_4[1]
    assert (1, edge_5.tag) == predecessors_vertex_4[2]
    assert (1, edge_5.immersed_shapes[0].tag) == predecessors_vertex_4[3]

    vertex_5 = domain.lookup_vertex(np.array([0.75, 0.0, 0.0]))
    predecessors_vertex_5 = list(domain.graph.predecessors((2, vertex_5.tag)))
    assert (1, edge_0.immersed_shapes[0].tag) == predecessors_vertex_5[0]
    assert (1, edge_0.immersed_shapes[1].tag) == predecessors_vertex_5[1]
    assert (1, edge_6.tag) == predecessors_vertex_5[2]

    vertex_6 = domain.lookup_vertex(np.array([0.75, 0.75, 0.0]))
    predecessors_vertex_6 = list(domain.graph.predecessors((2, vertex_6.tag)))
    assert (1, edge_5.immersed_shapes[0].tag) == predecessors_vertex_6[0]
    assert (1, edge_5.immersed_shapes[1].tag) == predecessors_vertex_6[1]
    assert (1, edge_6.tag) == predecessors_vertex_6[2]

    vertex_7 = domain.lookup_vertex(np.array([1.0, 0.75, 0.0]))
    predecessors_vertex_7 = list(domain.graph.predecessors((2, vertex_7.tag)))
    assert (1, edge_1.immersed_shapes[0].tag) == predecessors_vertex_7[0]
    assert (1, edge_1.immersed_shapes[1].tag) == predecessors_vertex_7[1]
    assert (1, edge_5.tag) == predecessors_vertex_7[2]
    assert (1, edge_5.immersed_shapes[1].tag) == predecessors_vertex_7[3]

    face_0_successors = list(domain.graph.successors((0, face_0.tag)))
    assert (1, wire_4.tag) == face_0_successors[0]
    assert (1, edge_5.tag) == face_0_successors[1]
    assert (1, edge_6.tag) == face_0_successors[2]


def test_domain_2d_y_shape_in_face_bc_intersection():
    lines_file = "lines/y_shape_bc_intersection.csv"
    domain: Domain = generate_domain_2d_with_lines(lines_file)
    edge_0 = domain.shapes[1][0]
    edge_1 = domain.shapes[1][1]
    edge_2 = domain.shapes[1][2]
    edge_3 = domain.shapes[1][3]
    wire_4 = domain.shapes[1][4]
    edge_5 = domain.shapes[1][5]
    edge_6 = domain.shapes[1][6]
    edge_7 = domain.shapes[1][7]
    edge_8 = domain.shapes[1][8]
    edge_9 = domain.shapes[1][9]
    face_0 = domain.shapes[0][0]

    # wire features
    assert len(wire_4.immersed_shapes) == 6
    assert len(wire_4.orientation) == 6

    predecessors_wire_4 = list(domain.graph.successors((1, wire_4.tag)))
    assert (1, edge_0.tag) == predecessors_wire_4[1]
    assert (1, edge_1.immersed_shapes[0].tag) == predecessors_wire_4[2]
    assert (1, edge_1.immersed_shapes[1].tag) == predecessors_wire_4[3]
    assert (1, edge_2.immersed_shapes[0].tag) == predecessors_wire_4[4]
    assert (1, edge_2.immersed_shapes[1].tag) == predecessors_wire_4[5]
    assert (1, edge_3.tag) == predecessors_wire_4[6]

    vertex_7 = domain.lookup_vertex(np.array([1.0, 0.6, 0]))
    predecessors_vertex_7 = list(domain.graph.predecessors((2, vertex_7.tag)))
    assert (1, edge_1.immersed_shapes[0].tag) == predecessors_vertex_7[0]
    assert (1, edge_1.immersed_shapes[1].tag) == predecessors_vertex_7[1]
    assert (1, edge_6.tag) == predecessors_vertex_7[2]

    vertex_5 = domain.lookup_vertex(np.array([0.5, 1.0, 0]))
    predecessors_vertex_5 = list(domain.graph.predecessors((2, vertex_5.tag)))
    assert (1, edge_2.immersed_shapes[0].tag) == predecessors_vertex_5[0]
    assert (1, edge_2.immersed_shapes[1].tag) == predecessors_vertex_5[1]
    assert (1, edge_7.tag) == predecessors_vertex_5[2]

    vertex_6 = domain.lookup_vertex(np.array([0.6, 0.6, 0]))
    predecessors_vertex_6 = list(domain.graph.predecessors((2, vertex_6.tag)))
    assert (1, edge_5.tag) == predecessors_vertex_6[0]
    assert (1, edge_6.tag) == predecessors_vertex_6[1]
    assert (1, edge_7.tag) == predecessors_vertex_6[2]

    face_0_successors = list(domain.graph.successors((0, face_0.tag)))
    assert (1, wire_4.tag) == face_0_successors[0]
    assert (1, edge_5.tag) == face_0_successors[1]
    assert (1, edge_6.tag) == face_0_successors[2]
    assert (1, edge_7.tag) == face_0_successors[3]


def test_domain_2d_18_shapes_in_face():
    lines_file = "lines/18_shapes_in_unit_square.csv"
    domain: Domain = generate_domain_2d_with_lines(lines_file)
    wire_4 = domain.shapes[1][4]
    face_0 = domain.shapes[2][0]

    assert len(domain.shapes[0]) == 63
    assert len(domain.shapes[1]) == 87
    assert len(face_0.immersed_shapes) == 18
    assert len(wire_4.immersed_shapes) == 4
    assert len(wire_4.orientation) == 4

    mesher = ConformalMesher(dimension=domain.dimension)
    mesher.domain = domain
    mesher.generate_from_domain(0.05)
    mesher.write_mesh("gmesh.msh")
