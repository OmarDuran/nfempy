import pytest
import numpy as np
from topology.vertex import Vertex
from topology.edge import Edge
from topology.wire import Wire
from topology.face import Face
from topology.shell import Shell
from topology.solid import Solid
from topology.composite_solid import CompositeSolid

from topology.graphical_shape import draw_vertex_sequence


def build_solid(points):

    vertices = np.array([Vertex(tag, point) for tag, point in enumerate(points)])
    edge_connectivities = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    edges = []
    for tag, con in enumerate(edge_connectivities):
        edge = Edge(tag, vertices[con])
        edges.append(edge)

    edges = np.array(edges)
    tag += 1

    wire_0 = Wire(tag, edges[[0, 1, 2, 3]], vertices[[0]])
    tag += 1

    wire_1 = Wire(tag, edges[[4, 5, 6, 7]], vertices[[4]])
    tag += 1

    wire_2 = Wire(tag, edges[[0, 9, 4, 8]], vertices[[0]])
    tag += 1

    wire_3 = Wire(tag, edges[[1, 10, 5, 9]], vertices[[1]])
    tag += 1

    wire_4 = Wire(tag, edges[[2, 11, 6, 10]], vertices[[2]])
    tag += 1

    wire_5 = Wire(tag, edges[[3, 11, 7, 8]], vertices[[3]])
    tag += 1

    wires = np.array([wire_0, wire_1, wire_2, wire_3, wire_4, wire_5])

    surfaces = []
    for wire in wires:
        surface = Face(tag, np.array([wire]))
        surfaces.append(surface)
        tag += 1
    surfaces = np.array(surfaces)

    shell = Shell(
        tag,
        surfaces,
        wires,
    )
    tag += 1

    solid = Solid(tag, np.array([shell]))
    return shell, solid


def test_vertex():

    v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1: Vertex = Vertex(1, np.array([0.0, 0.0, 0.0]))
    v2: Vertex = Vertex(2, np.array([0.5, 0.0, 0.0]))

    assert v0.dimension == 0
    assert v1.dimension == 0
    assert v2.dimension == 0
    assert not v0.composite
    assert not v1.composite
    assert not v2.composite

    v0.immersed_shapes = np.array([v2])
    v1.immersed_shapes = np.array([v2])
    assert v1.immersed_shapes[0] == v0.immersed_shapes[0]

    assert hash((v0.dimension, v0.tag)) == v0.hash()
    assert hash((v1.dimension, v1.tag)) == v1.hash()

    assert v0 == v0
    assert v0 != v1
    assert v2 != v0

    # test immersed_shapes
    assert v2 in v0
    assert v2 in v1


def test_edge():

    v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1: Vertex = Vertex(1, np.array([0.5, 0.5, 0.5]))
    v2: Vertex = Vertex(2, np.array([1.0, 1.0, 1.0]))

    e0: Edge = Edge(0, np.array([v0, v1]))
    e1: Edge = Edge(1, np.array([v1, v2]))
    e2: Edge = Edge(2, np.array([v0, v2]))

    assert e0.dimension == 1
    assert e1.dimension == 1
    assert e2.dimension == 1
    assert not e0.composite
    assert not e1.composite
    assert not e2.composite

    e2.immersed_shapes = np.array([e0, e1])

    assert hash((e0.dimension, e0.tag)) == e0.hash()
    assert hash((e1.dimension, e1.tag)) == e1.hash()

    assert e0 == e0
    assert e0 != e1
    assert e2 != e0

    # test immersed_shapes
    assert e0 in e2
    assert e1 in e2


def test_wire():

    v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1: Vertex = Vertex(1, np.array([1.0, 0.0, 0.0]))
    v2: Vertex = Vertex(2, np.array([1.0, 1.0, 1.0]))
    v3: Vertex = Vertex(3, np.array([0.0, 1.0, 1.0]))

    e0: Edge = Edge(0, np.array([v0, v1]))
    e1: Edge = Edge(1, np.array([v1, v2]))
    e2: Edge = Edge(2, np.array([v2, v3]))
    e3: Edge = Edge(3, np.array([v3, v0]))

    w0: Wire = Wire(0, np.array([e0, e1, e2, e3]), np.array([v0]))
    w1: Wire = Wire(0, np.array([e1, e2, e3, e0]), np.array([v1]))
    w2: Wire = Wire(1, np.array([e0, e1, e2, e3]), np.array([v0]))

    assert w0.dimension == 1
    assert w1.dimension == 1
    assert w1.dimension == 1
    assert hash((w0.dimension, w0.tag)) == w0.hash()
    assert hash((w1.dimension, w1.tag)) == w1.hash()
    assert hash((w2.dimension, w2.tag)) == w2.hash()

    assert w0.composite
    assert w1.composite
    assert w2.composite

    assert np.all(w0.orientation == np.array([1, 1, 1, -1]))
    assert np.all(w1.orientation == np.array([1, 1, -1, 1]))
    assert np.all(w2.orientation == np.array([1, 1, 1, -1]))

    assert w0 == w0
    assert w0 != w1
    assert w2 != w0

    # test immersed_shapes
    assert v0 in w0
    assert e0 in w0
    assert e1 in w0
    assert e2 in w0
    assert e3 in w0


def test_face():

    v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1: Vertex = Vertex(1, np.array([1.0, 0.0, 0.0]))
    v2: Vertex = Vertex(2, np.array([1.0, 1.0, 1.0]))
    v3: Vertex = Vertex(3, np.array([0.0, 1.0, 1.0]))

    e0: Edge = Edge(0, np.array([v0, v1]))
    e1: Edge = Edge(1, np.array([v1, v2]))
    e2: Edge = Edge(2, np.array([v2, v3]))
    e3: Edge = Edge(3, np.array([v3, v0]))

    w0: Wire = Wire(0, np.array([e0, e1, e2, e3]), np.array([v0]))
    w1: Wire = Wire(0, np.array([e1, e2, e3, e0]), np.array([v1]))
    w2: Wire = Wire(1, np.array([e0, e1, e2, e3]), np.array([v0]))

    f0: Face = Face(0, np.array([w0]))
    f1: Face = Face(0, np.array([w1]))
    f2: Face = Face(1, np.array([w2]))

    assert f0.dimension == 2
    assert f1.dimension == 2
    assert f1.dimension == 2
    assert hash((f0.dimension, w0.tag)) == f0.hash()
    assert hash((f1.dimension, w1.tag)) == f1.hash()
    assert hash((f2.dimension, w2.tag)) == f2.hash()

    assert not f0.composite
    assert not f1.composite
    assert not f2.composite

    assert f0 == f0
    assert f0 != f1
    assert f2 != f0

    # test immersed_shapes
    assert w0 in f0
    assert w1 not in f0


def test_shell():

    v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1: Vertex = Vertex(1, np.array([1.0, 0.0, 0.0]))
    v2: Vertex = Vertex(2, np.array([0.0, 1.0, 0.0]))
    v3: Vertex = Vertex(3, np.array([2.0, 1.0, 1.0]))
    v4: Vertex = Vertex(4, np.array([1.0, 2.0, 1.0]))

    e0: Edge = Edge(0, np.array([v0, v1]))
    e1: Edge = Edge(1, np.array([v1, v2]))
    e2: Edge = Edge(2, np.array([v2, v0]))

    e3: Edge = Edge(3, np.array([v1, v2]))
    e4: Edge = Edge(4, np.array([v2, v4]))
    e5: Edge = Edge(5, np.array([v4, v3]))
    e6: Edge = Edge(6, np.array([v3, v1]))

    w0: Wire = Wire(0, np.array([e0, e1, e2]), np.array([v0]))
    w1: Wire = Wire(1, np.array([e3, e4, e5, e6]), np.array([v1]))

    f0: Face = Face(0, np.array([w0]))
    f1: Face = Face(1, np.array([w1]))

    s0: Shell = Shell(0, np.array([f0, f1]), np.array([w0, w1]))
    s1: Shell = Shell(0, np.array([f1, f0]), np.array([w1, w0]))

    # vertex_seq = [wire.orient_immersed_vertices() for wire in s0.boundary_shapes]
    # draw_vertex_sequence(0, vertex_seq, True)

    assert s0.dimension == 2
    assert hash((s0.dimension, s0.tag)) == s0.hash()
    assert s0.composite

    assert s0 == s0
    assert s0 != s1

    # test immersed_shapes
    assert f0 in s0
    assert f1 in s0
    assert w0 in s0
    assert w1 in s0


def test_solid():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    shell, solid = build_solid(points)

    # wires = solid.boundary_shapes[0].boundary_shapes
    # vertex_seq = [wire.orient_immersed_vertices() for wire in wires]
    # draw_vertex_sequence(0, vertex_seq, True)

    assert solid.dimension == 3
    assert hash((solid.dimension, solid.tag)) == solid.hash()
    assert not solid.composite

    assert solid == solid

    # test immersed_shapes
    assert shell in solid


def test_commposite_solid():

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    shell0, solid0 = build_solid(points)

    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [1.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ]
    )
    shell1, solid1 = build_solid(points)
    csolid0 = CompositeSolid(0, np.array([solid0, solid1]), np.array([shell0, shell1]))
    csolid1 = CompositeSolid(0, np.array([solid1, solid0]), np.array([shell1, shell0]))

    # wires0 = csolid.boundary_shapes[0].boundary_shapes
    # wires1 = csolid.boundary_shapes[1].boundary_shapes
    # wires = np.concatenate((wires0,wires1))
    # vertex_seq = [wire.orient_immersed_vertices() for wire in wires]
    # draw_vertex_sequence(0, vertex_seq, True)

    assert csolid0.dimension == 3
    assert hash((csolid0.dimension, csolid0.tag)) == csolid0.hash()
    assert csolid0.composite

    assert csolid0 == csolid0
    assert csolid0 != csolid1

    # test immersed_shapes
    assert solid0 in csolid0
    assert solid1 in csolid0
    assert shell0 in csolid0
    assert shell1 in csolid0


def test_shape_assignment():

    v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1: Vertex = Vertex(1, np.array([1.0, 0.0, 0.0]))
    v2: Vertex = Vertex(2, np.array([1.0, 1.0, 1.0]))
    v3: Vertex = Vertex(3, np.array([0.0, 1.0, 1.0]))

    e0: Edge = Edge(0, np.array([v0, v1]))
    e1: Edge = Edge(1, np.array([v1, v2]))
    e2: Edge = Edge(2, np.array([v2, v3]))
    e3: Edge = Edge(3, np.array([v3, v0]))

    w0: Wire = Wire(0, np.array([e0, e1, e2, e3]), np.array([v0]))
    w1: Wire = Wire(1, np.array([e1, e2, e3, e0]), np.array([v1]))

    assert v0 != v1
    v0.shape_assignment(v1)
    assert v0 == v1

    assert e0 != e1
    e0.shape_assignment(e1)
    assert e0 == e1

    assert w0 != w1
    w0.shape_assignment(w1)
    assert w0 == w1

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    shell0, solid0 = build_solid(points)
    solid0.tag = 0

    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [1.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ]
    )
    shell1, solid1 = build_solid(points)
    solid0.tag = 1

    assert shell0 != shell1
    shell0.shape_assignment(shell1)
    assert shell0 == shell1

    assert solid0 != solid1
    solid0.shape_assignment(solid1)
    assert solid0 == solid1
