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

    # Delete temporary files
    command = "rm -r *.vtk *.msh"
    subprocess.run(command, shell=True, capture_output=True, text=True)

