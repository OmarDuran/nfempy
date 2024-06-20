import pytest
import numpy as np
from topology.vertex import Vertex
from topology.edge import Edge
from topology.wire import Wire
from topology.face import Face
from topology.shell import Shell

from topology.graphical_shape import draw_vertex_sequence

def test_vertex():

    v0: Vertex = Vertex(0, np.array([0.0,0.0,0.0]))
    v1: Vertex = Vertex(1, np.array([0.0,0.0,0.0]))
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

    v0: Vertex = Vertex(0, np.array([0.0,0.0,0.0]))
    v1: Vertex = Vertex(1, np.array([0.5,0.5,0.5]))
    v2: Vertex = Vertex(2, np.array([1.0,1.0,1.0]))

    e0: Edge = Edge(0, np.array([v0, v1]))
    e1: Edge = Edge(1, np.array([v1, v2]))
    e2: Edge = Edge(2, np.array([v0, v2]))

    assert e0.dimension == 1
    assert e1.dimension == 1
    assert e2.dimension == 1
    assert not e0.composite
    assert not e1.composite
    assert not e2.composite

    e2.immersed_shapes = np.array([e0,e1])

    assert hash((e0.dimension, e0.tag)) == e0.hash()
    assert hash((e1.dimension, e1.tag)) == e1.hash()

    assert e0 == e0
    assert e0 != e1
    assert e2 != e0

    # test immersed_shapes
    assert e0 in e2
    assert e1 in e2

def test_wire():

    v0: Vertex = Vertex(0, np.array([0.0,0.0,0.0]))
    v1: Vertex = Vertex(1, np.array([1.0,0.0,0.0]))
    v2: Vertex = Vertex(2, np.array([1.0,1.0,1.0]))
    v3: Vertex = Vertex(3, np.array([0.0,1.0,1.0]))

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

    assert np.all(w0.orientation == np.array([ 1,  1,  1, -1]))
    assert np.all(w1.orientation == np.array([ 1,  1, -1,  1]))
    assert np.all(w2.orientation == np.array([ 1,  1,  1, -1]))

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

    v0: Vertex = Vertex(0, np.array([0.0,0.0,0.0]))
    v1: Vertex = Vertex(1, np.array([1.0,0.0,0.0]))
    v2: Vertex = Vertex(2, np.array([1.0,1.0,1.0]))
    v3: Vertex = Vertex(3, np.array([0.0,1.0,1.0]))

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

    v0: Vertex = Vertex(0, np.array([0.0,0.0,0.0]))
    v1: Vertex = Vertex(1, np.array([1.0,0.0,0.0]))
    v2: Vertex = Vertex(2, np.array([0.0,1.0,0.0]))
    v3: Vertex = Vertex(3, np.array([2.0,1.0,1.0]))
    v4: Vertex = Vertex(4, np.array([1.0,2.0,1.0]))

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

    s0: Shell = Shell(0, np.array([f0,f1]), np.array([w0,w1]))
    s1: Shell = Shell(0, np.array([f1,f0]), np.array([w1,w0]))

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