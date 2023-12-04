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
