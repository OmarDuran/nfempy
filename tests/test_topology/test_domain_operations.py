import pytest
import subprocess
import numpy as np
import networkx as nx
from topology.domain_operations import create_domain
from topology.domain_operations  import domain_difference
from topology.domain_operations  import domain_union
from topology.vertex import Vertex
from topology.edge import Edge

from topology.line_line_incidence import lines_lines_intersection

from mesh.discrete_domain import DiscreteDomain
from mesh.mesh import Mesh

def __transformation_matrix(theta, tx, ty, tz):
    Rx = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )
    Ry = np.array(
        [
            [np.cos(theta), 0, -np.sin(theta), 0],
            [0, 1, 0, 0],
            [np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )
    Rz = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    T = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])
    return T @ Rx @ Ry @ Rz


def __transform_points(points, transformation_matrix):
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones))
    transformed_points_homogeneous = np.dot(points_homogeneous, transformation_matrix.T)
    transformed_points = transformed_points_homogeneous[:, :3]
    return transformed_points

def __mesh_md_domain(md_domain):
    # Conformal gmsh discrete representation
    h_val = 0.1
    domain_h = DiscreteDomain(dimension=md_domain.dimension)
    domain_h.domain = md_domain
    domain_h.generate_mesh(h_val, 0)
    domain_h.write_mesh("gmesh.msh")

    # Mesh representation
    gmesh = Mesh(dimension=md_domain.dimension, file_name="gmesh.msh")
    gmesh.build_conformal_mesh()
    gmesh.write_vtk()
    assert True

    # # Delete temporary files
    # command = "rm -r *.vtk *.msh"
    # subprocess.run(command, shell=True, capture_output=True, text=True)

def md_domain_single_line(case):

    selector= {
        'c0': {'edge_data': [np.array([0, 1])], 'vertex_data': np.array([])},
        'c1': {'edge_data': [np.array([0, 1])], 'vertex_data': np.array([2])},
        'c2': {'edge_data': [np.array([0, 1])], 'vertex_data': np.array([2, 3])},
        'c3': {'edge_data': [np.array([0, 1])], 'vertex_data': np.array([2, 4, 5])},
        'c4': {'edge_data': [np.array([0, 1])], 'vertex_data': np.array([2, 3, 4, 5])},
    }
    cdata = selector[case]

    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
        [0.75, 0.75, 0.75],
        [-0.5, -0.5, -0.5],
        [+1.5, +1.5, +1.5],
    ])

    vertices = np.array([], dtype=Vertex)
    tag = 0
    for point in points:
        v: Vertex = Vertex(tag, point)
        vertices = np.append(vertices,np.array([v]),axis=0)
        tag += 1

    # physical tag on c0_shape boundary
    for vertex in vertices[cdata['edge_data'][0]]:
        vertex.physical_tag = 2

    # physical tag on c1_shapes
    if cdata['vertex_data'].shape[0] != 0:
        for vertex in vertices[cdata['vertex_data']]:
            vertex.physical_tag = 20

    edges = np.array([], dtype=Edge)
    for e_con in cdata['edge_data']:
        e: Edge = Edge(tag, vertices[e_con])
        e.physical_tag = 1
        edges = np.append(edges,np.array([e]),axis=0)
        tag += 1
    edges = np.array(edges)

    max_dim = 1
    shapes = np.concatenate([vertices[cdata['edge_data'][0]], edges])
    domain = create_domain(dimension=max_dim, shapes=shapes)

    if cdata['vertex_data'].shape[0] != 0:
        c1_vertices = vertices[cdata['vertex_data']]
    else:
        c1_vertices = np.array([])
    domain_c1 = create_domain(dimension=max_dim, shapes=c1_vertices)

    # compute difference
    if cdata['vertex_data'].shape[0] != 0:
        tag = np.max([domain.max_tag(), domain_c1.max_tag()]) + 1
    else:
        tag = domain.max_tag() + 1

    domain_c0 = domain_difference(domain, domain_c1, tag)

    # compute union
    md_domain = domain_union(domain_c0, domain_c1)
    return domain, domain_c1, domain_c0, md_domain

@pytest.mark.parametrize("case", ['c0','c1','c2','c3', 'c4'])
def test_operations_on_single_line(case):

    domain, domain_c1, domain_c0, md_domain = md_domain_single_line(case)

    domain.build_grahp()
    domain_c1.build_grahp(0)
    domain_c0.build_grahp()
    md_domain.build_grahp()

    if case == 'c0':
        assert list(domain.graph.nodes()) == list(domain_c0.graph.nodes())
        assert list(domain_c0.graph.nodes()) == list(md_domain.graph.nodes())
        assert list(domain_c1.graph.nodes()) == []
    else:
        intx_0 = nx.intersection(domain.graph, domain_c0.graph)
        intx_1 = nx.intersection(md_domain.graph, domain_c0.graph)
        intx_2 = nx.intersection(md_domain.graph, domain_c1.graph)

        # reconstruct domain_c0.graph (domain_c0 = md_domain - domain_c1)
        c0_graph = md_domain.graph.copy()
        internal_bc_graph = domain_c0.graph.copy()
        c1_graph = domain_c1.graph.copy()
        c1_nodes = list(c1_graph.nodes())

        edges_associated = [(u, v) for u, v in c0_graph.edges() if u in c1_nodes or v in c1_nodes]
        c0_graph.remove_edges_from(edges_associated)
        disconnected_nodes = list(nx.isolates(c0_graph))
        c0_graph.remove_nodes_from(disconnected_nodes)
        internal_bc_graph.remove_edges_from(intx_1.edges())
        c0_graph.add_edges_from(internal_bc_graph.edges())

        assert list(intx_0.nodes()) == [(1, 0), (1, 1)]
        assert list(intx_2.nodes())[0] in list(domain_c1.graph.nodes())
        assert set(domain_c0.graph.nodes()) == set(c0_graph.nodes())
        assert set(domain_c0.graph.edges()) == set(c0_graph.edges())

    # Conformal mesh representation
    __mesh_md_domain(md_domain)


def md_domain_multiple_lines(case, transform_points_q):

    selector = {
        'c0': {'edge_data': [np.array([0, 4]), np.array([2, 6])]},
        'c1': {'edge_data': [np.array([5, 7]), np.array([7, 1]), np.array([1, 3]), np.array([3, 5])]},
        'c2': {'edge_data': [np.array([0, 4]), np.array([2, 6]), np.array([5, 7]), np.array([7, 1]), np.array([1, 3]), np.array([3, 5])]},
        'c3': {'edge_data': [np.array([5, 7]), np.array([7, 1]), np.array([1, 3]),
                             np.array([3, 5]), np.array([10, 9]), np.array([10, 8]),
                             np.array([10, 9]), np.array([10, 12])]},
        'c4': {'edge_data': [np.array([5, 7]), np.array([7, 1]), np.array([1, 3]),
                             np.array([3, 5]), np.array([8, 11])]},
        'c5': {'edge_data': [np.array([5, 2]), np.array([2, 7]), np.array([1, 7]),
                             np.array([1, 3]), np.array([3, 6]), np.array([0, 4])]},
        'c6': {'edge_data': [np.array([5, 7]), np.array([7, 1]), np.array([1, 3]),
                             np.array([3, 5]), np.array([0, 3]), np.array([0, 5]), np.array([2, 5]), np.array([2, 7]),
                             np.array([4, 1]), np.array([4, 7]), np.array([6, 1]), np.array([6, 3])]},
    }
    cdata = selector[case]

    points = np.array([
        [+1.5, +0.0, 0.0],
        [+1.0, +1.0, 0.0],
        [+0.0, +1.5, 0.0],
        [-1.0, +1.0, 0.0],
        [-1.5, +0.0, 0.0],
        [-1.0, -1.0, 0.0],
        [-0.0, -1.5, 0.0],
        [+1.0, -1.0, 0.0],
        [+0.0, +1.0, 0.0],
        [+1.0, +0.5, 0.0],
        [+0.0, +0.0, 0.0],
        [+0.0, -1.0, 0.0],
        [-0.5, -0.5, 0.0],
    ])

    if transform_points_q:
        theta = np.pi / 4
        tx, ty, tz = 1, 2, 3
        trans_matrix = __transformation_matrix(theta, tx, ty, tz)
        points = __transform_points(points, trans_matrix)

    vertices = np.array([], dtype=Vertex)
    tag = 0
    for point in points:
        v: Vertex = Vertex(tag, point)
        v.physical_tag = 2  # physical tag on c0_shapes boundaries
        vertices = np.append(vertices, np.array([v]), axis=0)
        tag += 1

    c0_edges = np.array([], dtype=Edge)
    c0_vertices = np.array([], dtype=Vertex)
    for e_con in cdata['edge_data']:
        for v in vertices[e_con]:
            c0_vertices = np.append(c0_vertices, np.array([v]), axis=0)
        e: Edge = Edge(tag, vertices[e_con])
        e.physical_tag = 1
        c0_edges = np.append(c0_edges, np.array([e]), axis=0)
        tag += 1
    c0_edges = np.array(c0_edges)

    max_dim = 1
    shapes = np.concatenate([c0_vertices, c0_edges])
    domain = create_domain(dimension=max_dim, shapes=shapes)

    lines = np.array([e.boundary_points() for e in c0_edges])
    unique_points = lines_lines_intersection(lines, lines, deduplicate_points_q=True)

    c1_vertices = np.array([], dtype=Vertex)
    tag = domain.max_tag() + 1
    for point in unique_points:
        v: Vertex = Vertex(tag, point)
        v.physical_tag = 20
        c1_vertices = np.append(c1_vertices, np.array([v]), axis=0)
        tag += 1
    domain_c1 = create_domain(dimension=max_dim, shapes=c1_vertices)

    # compute difference
    tag = np.max([domain.max_tag(), domain_c1.max_tag()]) + 1
    domain_c0 = domain_difference(domain, domain_c1, tag)

    # compute union
    md_domain = domain_union(domain_c0, domain_c1)
    return domain, domain_c1, domain_c0, md_domain

@pytest.mark.parametrize("case, transform_points_q", [
    ('c0', False),
    ('c0', True),
    ('c1', False),
    ('c1', True),
    ('c2', False),
    ('c2', True),
    ('c3', False),
    ('c3', True),
    ('c4', False),
    ('c4', True),
    ('c5', False),
    ('c5', True),
    ('c6', False),
    ('c6', True),
])
def test_operations_on_multiple_lines(case, transform_points_q):

    domain, domain_c1, domain_c0, md_domain = md_domain_multiple_lines(case, transform_points_q)

    domain.build_grahp()
    domain_c1.build_grahp(0)
    domain_c0.build_grahp()
    md_domain.build_grahp()

    intx_0 = nx.intersection(domain.graph, domain_c0.graph)
    intx_1 = nx.intersection(md_domain.graph, domain_c0.graph)
    intx_2 = nx.intersection(md_domain.graph, domain_c1.graph)

    # reconstruct domain_c0.graph (domain_c0 = md_domain - domain_c1)
    c0_graph = md_domain.graph.copy()
    internal_bc_graph = domain_c0.graph.copy()
    c1_graph = domain_c1.graph.copy()
    c1_nodes = list(c1_graph.nodes())

    edges_associated = [(u, v) for u, v in c0_graph.edges() if u in c1_nodes or v in c1_nodes]
    c0_graph.remove_edges_from(edges_associated)
    disconnected_nodes = list(nx.isolates(c0_graph))
    c0_graph.remove_nodes_from(disconnected_nodes)
    internal_bc_graph.remove_edges_from(intx_1.edges())
    c0_graph.add_edges_from(internal_bc_graph.edges())

    # assert list(intx_0.nodes()) == [(1, 0), (1, 1)]
    # assert list(intx_2.nodes())[0] in list(domain_c1.graph.nodes())
    # assert set(domain_c0.graph.nodes()) == set(c0_graph.nodes())
    # assert set(domain_c0.graph.edges()) == set(c0_graph.edges())

    # Conformal mesh representation
    __mesh_md_domain(md_domain)