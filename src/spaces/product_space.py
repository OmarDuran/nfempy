import numpy as np

from spaces.discrete_space import DiscreteSpace
import basix
from basis.element_family import basis_variant, family_by_name
from basis.element_type import type_by_dimension


class ProductSpace:
    # The discrete product space representation
    def __init__(self, discrete_spaces_data):
        self._define_fields_names(discrete_spaces_data)
        self._define_integration_order_and_quadrature(discrete_spaces_data)
        self._define_discrete_spaces(discrete_spaces_data)

    def _define_fields_physical_tags(self, fields_physical_tags):
        self.fields_physical_tags = fields_physical_tags

    def _define_fields_names(self, discrete_spaces_data):
        self.names = list(discrete_spaces_data.keys())

    def _define_integration_order_and_quadrature(self, discrete_spaces_data):
        k_orders = []
        dims = []
        for item in discrete_spaces_data.items():
            field, data = item
            k_orders.append(data[3])
            dims.append(data[0])

        self.integration_order = 2 * np.max(k_orders) + 1
        self.quadrature = [None, None, None, None]
        for dim in range(np.max(dims) + 1):
            if dim == 0:
                self.quadrature[dim] = (np.array([1.0]), np.array([1.0]))
            else:
                cell_type = type_by_dimension(dim)
                self.quadrature[dim] = basix.make_quadrature(
                    cell_type, self.integration_order, basix.QuadratureType.gauss_jacobi
                )
        self.bc_quadrature = [None, None, None, None]
        for dim in dims:
            dim -= 1
            if dim == 0:
                self.bc_quadrature[dim] = (np.array([1.0]), np.array([1.0]))
            else:
                cell_type = type_by_dimension(dim)
                self.bc_quadrature[dim] = basix.make_quadrature(
                    cell_type, self.integration_order, basix.QuadratureType.gauss_jacobi
                )

    def _define_discrete_spaces(self, discrete_spaces_data):
        self.discrete_spaces = {}
        for item in discrete_spaces_data.items():
            field, data = item
            dim, components, family, k_order, gmesh = data
            discrete_space = DiscreteSpace(
                dim,
                components,
                family,
                k_order,
                gmesh,
                integration_order=self.integration_order,
            )
            discrete_space.name = field
            self.discrete_spaces.__setitem__(field, discrete_space)

    def _define_discrete_spaces_dof(self):
        self.discrete_spaces_dofs = {}
        for item in self.discrete_spaces.items():
            name, space = item
            dofs = space.dof_map.dof_number()
            self.discrete_spaces_dofs.__setitem__(name, dofs)

    def _define_n_dof(self):
        self.n_dof = 0
        for item in self.discrete_spaces_dofs.items():
            name, dofs = item
            self.n_dof += dofs

    def dimension(self):
        dims = []
        for item in self.discrete_spaces.items():
            _, space = item
            dims.append(space.dimension)
        product_space_dim = np.max(dims)
        return product_space_dim

    def make_subspaces_discontinuous(self, discrete_space_disc):
        for name in self.names:
            disc_Q = discrete_space_disc.get(name, False)
            if disc_Q:
                self.discrete_spaces[name].make_discontinuous()

    def build_structures(self, physical_tags, b_physical_tags):
        self._define_fields_physical_tags(physical_tags)
        for item in self.discrete_spaces.items():
            name, space = item
            list_physical_tags = physical_tags.get(name, [None])
            list_b_physical_tags = b_physical_tags.get(name, [])
            space.build_structures(list_physical_tags)
            space.build_boundary_structures(list_b_physical_tags)

        self._define_discrete_spaces_dof()
        self._define_n_dof()

    @staticmethod
    def _retrieve_space_destination_indexes(space, cell_index):
        cell = space.elements[cell_index].data.cell
        cell_id = cell.id
        field_dest = space.dof_map.destination_indices(cell_id)
        return field_dest

    @staticmethod
    def _retrieve_space_bc_destination_indexes(space, cell_index):
        cell = space.bc_elements[cell_index].data.cell
        cell_id = cell.id

        # find high-dimension neigh
        entity_map = space.dof_map.mesh_topology.entity_map_by_dimension(cell.dimension)
        neigh_list = list(entity_map.predecessors(cell.index()))
        neigh_check_q = len(neigh_list) > 0
        assert neigh_check_q

        # destination indexes
        neigh_cell_id = neigh_list[0][1]
        field_dest = space.dof_map.bc_destination_indices(neigh_cell_id, cell_id)
        return field_dest

    def discrete_spaces_destination_indexes(self, cell_index):
        dofs_list = list(self.discrete_spaces_dofs.values())
        dofs_list.insert(0, 0)
        dofs_list = np.add.accumulate(dofs_list)

        discrete_spaces_dest = {}
        for i, item in enumerate(self.discrete_spaces.items()):
            name, space = item
            stride = dofs_list[i]
            field_dest = (
                self._retrieve_space_destination_indexes(space, cell_index) + stride
            )
            discrete_spaces_dest.__setitem__(name, field_dest)

        return discrete_spaces_dest

    def destination_indexes(self, cell_index):
        dest = list(self.discrete_spaces_destination_indexes(cell_index).values())
        dest = np.concatenate(dest)
        return dest

    def discrete_spaces_bc_destination_indexes(self, cell_index):
        dofs_list = list(self.discrete_spaces_dofs.values())
        dofs_list.insert(0, 0)
        dofs_list = np.add.accumulate(dofs_list)

        discrete_spaces_bc_dest = {}
        for i, item in enumerate(self.discrete_spaces.items()):
            name, space = item

            if len(space.bc_elements) == 0:
                continue

            stride = dofs_list[i]
            field_dest = (
                self._retrieve_space_bc_destination_indexes(space, cell_index) + stride
            )
            discrete_spaces_bc_dest.__setitem__(name, field_dest)

        return discrete_spaces_bc_dest

    def bc_destination_indexes(self, cell_index):
        dest = list(self.discrete_spaces_bc_destination_indexes(cell_index).values())
        dest = np.concatenate(dest)
        return dest
