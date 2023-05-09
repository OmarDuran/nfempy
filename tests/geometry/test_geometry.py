import pytest
import numpy as np

from geometry.vertex import Vertex
from geometry.edge import Edge
from geometry.wire import Wire
from geometry.face import Face
from geometry.shell import Shell
from geometry.solid import Solid
from geometry.domain import Domain
from geometry.domain_market import build_box_1D, build_box_2D, build_box_3D
from geometry.shape_manipulation import ShapeManipulation


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
    vertex_0 = domain.shapes[0][0]
    vertex_1 = domain.shapes[0][1]
    edge_0 = domain.shapes[1][0]

    # max tag per dimension
    max_vertex_tag = len(domain.shapes[0])
    max_edge_tag = len(domain.shapes[1])

    immersed_points = np.array(
        [[0.25, 0, 0], [0.75, 0, 0], [0.99999, 0, 0], [1.75, 0, 0]]
    )

    physical_tags = [4, 5, 6, 7, 8]
    vertices = np.array(
        [
            Vertex(tag + max_vertex_tag, point)
            for tag, point in enumerate(immersed_points)
        ]
    )
    for vertex, physical_tag in zip(vertices, physical_tags):
        vertex.physical_tag = physical_tag

    # embed_vertex_in_edge will filter points not in the interval
    vertices = ShapeManipulation.embed_vertex_in_edge(
        vertices, edge_0, tag_shift=max_edge_tag
    )
    domain.append_shapes(vertices)
    domain.append_shapes(edge_0.immersed_shapes)
    domain.build_grahp()

    predecessors_vertex_0 = list(domain.graph.predecessors((1, vertex_0.tag)))[0]
    predecessors_vertex_1 = list(domain.graph.predecessors((1, vertex_1.tag)))[0]
    assert predecessors_vertex_0 == predecessors_vertex_1

    vertex_2 = domain.shapes[0][2]
    vertex_3 = domain.shapes[0][3]
    edge_1 = domain.shapes[1][1]
    edge_2 = domain.shapes[1][2]
    edge_3 = domain.shapes[1][3]

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
