import networkx as nx
import numpy as np

from mesh.mesh_topology import MeshTopology


def coloring_mesh_by_co_dimension(n_colors, mesh, dim, sub_entity_co_dim):
    mesh_topology = MeshTopology(mesh, dim)
    mesh_topology.build_data()

    map = mesh_topology.entity_map_by_codimension(sub_entity_co_dim)
    reverse_map = map.reverse(copy=False)

    # coloring graph
    rcoloring = nx.coloring.equitable_color(reverse_map, num_colors=n_colors)
    c0_entities = [cell.id for cell in mesh.cells if cell.dimension == dim]
    colored_elements = [item for item in rcoloring.items() if item[0] in c0_entities]
    colored_elements = np.vstack(colored_elements)

    # coloring mesh
    base_colors, idx_inv = np.unique(colored_elements[:, 1], return_inverse=True)
    colors = np.array(list(range(len(base_colors))))
    colored_elements[:, 1] = colors[np.array([idx_inv])]

    # building color maps
    entity_to_color = dict(colored_elements)
    color_to_entities = {}
    for item in entity_to_color.items():
        existing_key = color_to_entities.get(item[1], None)
        if existing_key is None:
            color_to_entities.__setitem__(item[1], [item[0]])
        else:
            chunk = existing_key + [item[0]]
            color_to_entities.__setitem__(item[1], chunk)

    return entity_to_color, color_to_entities
