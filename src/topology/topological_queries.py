from mesh.mesh import Mesh
from mesh.mesh_cell import MeshCell
from topology.mesh_topology import MeshTopology


def find_higher_dimension_neighs(cell: MeshCell, mesh_topology: MeshTopology):
    entity_map = mesh_topology.entity_map_by_dimension(cell.dimension)
    neigh_list = list(entity_map.predecessors(cell.id))
    return neigh_list


def find_lower_dimension_neighs(cell: MeshCell, mesh_topology: MeshTopology):
    entity_map = mesh_topology.entity_map_by_dimension(cell.dimension)
    neigh_list = list(entity_map.successors(cell.id))
    return neigh_list

def find_neighbors_by_codimension_1(gmesh: Mesh):
    co_dim = 1
    cell_idx_to_neigh_idxs = {}
    g = gmesh.build_graph(gmesh.dimension, co_dim)
    cells_c0 = [cell for cell in gmesh.cells if cell.dimension == gmesh.dimension]
    for cell_c0 in cells_c0:
        cells_co_dim = cell_c0.sub_cells_ids[gmesh.dimension - co_dim]
        cell_id = cell_c0.id
        neigh_idxs = []
        for cell_co_dim in cells_co_dim:
            neighs = list(g.predecessors(cell_co_dim))
            if len(neighs) == 2:
                neighs.remove(cell_id)
                neigh_idxs.append(neighs[0])
            else:
                neigh_idxs.append(cell_co_dim)
        cell_idx_to_neigh_idxs[cell_id] = neigh_idxs
    return cell_idx_to_neigh_idxs