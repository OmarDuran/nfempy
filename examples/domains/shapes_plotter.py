import numpy as np
from topology.graphical_shape import draw_vertex_sequence
from topology.domain import Domain

from topology.domain_market import build_box_3D

box_points = np.array(
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
domain: Domain = build_box_3D(box_points)
solid = domain.shapes[3][0]
wires = solid.boundary_shapes[0].boundary_shapes
vertex_sequence = np.array([wire.orient_immersed_vertices() for wire in wires])
draw_vertex_sequence(0, vertex_sequence, True)
