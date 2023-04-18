import basix
import numpy as np


class DoFMap:
    def __init__(
        self,
        mesh_topology,
        family,
        element_type,
        k_order,
        basis_variant,
        discontinuous=False,
    ):
        self.mesh_topology = mesh_topology
        self.ref_element = basix.create_element(
            family, element_type, k_order, basis_variant, discontinuous
        )
        self.vertex_map = {}
        self.edge_map = {}
        self.face_map = {}
        self.volume_map = {}
        self.n_dof = 0
        self.dimension = self.mesh_topology.mesh.dimension

    def set_topological_dimension(self, dimension):
        if self.mesh_topology.mesh.dimension < dimension:
            raise ValueError(
                "DoFMap:: max dimension available is "
                % self.mesh_topology.mesh.dimension
            )
        self.dimension = dimension

    def build_entity_maps(self, n_components=1, n_dof_shift=0):
        dim = self.dimension

        vertex_ids = []
        edge_ids = []
        face_ids = []
        volume_ids = []

        if dim == 0:
            vertex_ids = self.mesh_topology.entities_by_dimension(0)
        elif dim == 1:
            vertex_ids = self.mesh_topology.entities_by_dimension(0)
            edge_ids = self.mesh_topology.entities_by_dimension(1)
        elif dim == 2:
            vertex_ids = self.mesh_topology.entities_by_dimension(0)
            edge_ids = self.mesh_topology.entities_by_dimension(1)
            face_ids = self.mesh_topology.entities_by_dimension(2)
        elif dim == 3:
            vertex_ids = self.mesh_topology.entities_by_dimension(0)
            edge_ids = self.mesh_topology.entities_by_dimension(1)
            face_ids = self.mesh_topology.entities_by_dimension(2)
            volume_ids = self.mesh_topology.entities_by_dimension(3)
        else:
            raise ValueError("Case not implemented for dimension: " % dim)

        n_vertices = len(vertex_ids)
        n_edges = len(edge_ids)
        n_faces = len(face_ids)
        n_volumes = len(volume_ids)

        entity_support = [n_vertices, n_edges, n_faces, n_volumes]
        for dim, n_entity_dofs in enumerate(self.ref_element.num_entity_dofs):
            e_dofs = int(np.mean(n_entity_dofs))
            entity_support[dim] *= e_dofs * n_components

        # Enumerates DoF
        dof_indices = np.array(
            [
                0,
                entity_support[0],
                entity_support[1],
                entity_support[2],
                entity_support[3],
            ]
        )
        global_indices = np.add.accumulate(dof_indices)
        global_indices += n_dof_shift
        # Computing cell mappings
        if len(vertex_ids) != 0:
            self.vertex_map = dict(
                zip(
                    vertex_ids,
                    np.split(
                        np.array(range(global_indices[0], global_indices[1])),
                        len(vertex_ids),
                    ),
                )
            )

        if len(edge_ids) != 0:
            self.edge_map = dict(
                zip(
                    edge_ids,
                    np.split(
                        np.array(range(global_indices[1], global_indices[2])),
                        len(edge_ids),
                    ),
                )
            )

        if len(face_ids) != 0:
            self.face_map = dict(
                zip(
                    face_ids,
                    np.split(
                        np.array(range(global_indices[2], global_indices[3])),
                        len(face_ids),
                    ),
                )
            )

        if len(volume_ids) != 0:
            self.volume_map = dict(
                zip(
                    volume_ids,
                    np.split(
                        np.array(range(global_indices[3], global_indices[4])),
                        len(volume_ids),
                    ),
                )
            )
        self.n_dof = sum(entity_support)

    def dof_number(self):
        return self.n_dof

    def destination_indices(self, cell_id):
        dim = self.dimension
        entity_maps = [self.vertex_map, self.edge_map, self.face_map, self.volume_map]
        dest_by_dim = []
        for d in range(dim + 1):
            entity_map = self.mesh_topology.entity_map_by_dimension(d)
            dof_supports = list(entity_map.successors(cell_id))
            entity_dest = np.array(
                [entity_maps[d].get(dof_s) for dof_s in dof_supports], dtype=int
            ).ravel()
            dest_by_dim.append(entity_dest)
        dest = np.concatenate(dest_by_dim)
        return dest

    def bc_destination_indices(self, cell_id, bc_cell_id):
        bc_cells_ids = self.mesh_topology.mesh.cells[bc_cell_id].sub_cells_ids
        dim = self.dimension
        entity_maps = [self.vertex_map, self.edge_map, self.face_map, self.volume_map]
        dest_by_dim = []
        for d in range(dim):
            entity_map = self.mesh_topology.entity_map_by_dimension(d)
            dof_supports = list(entity_map.successors(cell_id))
            dof_supports = [
                dof_support
                for dof_support in dof_supports
                if dof_support in list(bc_cells_ids[d])
            ]
            if int(np.mean(self.ref_element.num_entity_dofs[d])) == 0:
                dof_supports = []
            entity_dest = np.array(
                [entity_maps[d].get(dof_s) for dof_s in dof_supports], dtype=int
            ).ravel()
            dest_by_dim.append(entity_dest)
        dest = np.concatenate(dest_by_dim)
        return dest
