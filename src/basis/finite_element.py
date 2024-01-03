import basix
import numpy as np

from basis.element_data import ElementData
from basis.element_family import basis_variant, family_by_name
from basis.element_type import type_by_dimension
from basis.permute_and_transform import permute_and_transform
from geometry.mapping import evaluate_linear_shapes, evaluate_mapping


class FiniteElement:
    def __init__(
        self, cell_id, family, k_order, mesh, discontinuous=False, integration_order=0
    ):
        self.family = family
        self.k_order = k_order
        self.mesh = mesh
        self.discontinuous = discontinuous
        self.integration_order = integration_order
        self.basis_generator = None
        self.data = ElementData(
            dimension=mesh.cells[cell_id].dimension, cell=mesh.cells[cell_id], mesh=mesh
        )
        self._build_structures()

    def _build_structures(self):
        # fetch information from static methods
        family = self.family
        cell_type = type_by_dimension(self.data.cell.dimension)
        variant = basis_variant()
        self._set_integration_order()

        # create generator
        quadrature = np.empty(0)
        if self.data.cell.dimension == 0:
            # Can only create order 0 Lagrange on a point
            self.k_order = 0
            self.family = "Lagrange"
            family = family_by_name(self.family)
            self.basis_generator = basix.create_element(
                family,
                cell_type,
                self.k_order,
                variant,
                self.discontinuous,
            )
            # quadrature = (np.array([1.0]), np.array([1.0]))
        else:
            if self.data.cell.dimension == 1 and self.family in ["RT", "BDM"]:
                self.family = "Lagrange"
                family = family_by_name(self.family)
                self.basis_generator = basix.create_element(
                    family,
                    cell_type,
                    self.k_order,
                    variant,
                    self.discontinuous,
                )
            else:
                self.basis_generator = basix.create_element(
                    family,
                    cell_type,
                    self.k_order,
                    variant,
                    self.discontinuous,
                )
            # quadrature = basix.make_quadrature(
            #     basix.QuadratureType.gauss_jacobi, cell_type, self.integration_order
            # )
        # Partially fill element data
        self._fill_element_dof_data()
        self._fill_element_bc_entity_data()
        # self._fill_element_data(quadrature)
        # self.evaluate_mapping(quadrature[0])

    def _fill_element_quadratue(self, quadrature):
        self.data.quadrature.points = quadrature[0]
        self.data.quadrature.weights = quadrature[1]

    def _fill_element_dof_data(self):
        self.data.dof.entity_dofs = self.basis_generator.entity_dofs
        # self.data.dof.num_entity_dofs = self.basis_generator.num_entity_dofs
        self.data.dof.transformations_are_identity = (
            self.basis_generator.dof_transformations_are_identity
        )
        self.data.dof.transformations = self.basis_generator.entity_transformations()

    def _fill_element_bc_entity_data(self):
        c1_sub_cells_ids = self.data.cell.sub_cells_ids[self.data.dimension - 1]
        self.data.bc_entities = np.array(
            [
                self.mesh.cells[id].get_material_id() is not None
                for id in c1_sub_cells_ids
            ]
        )

    def _set_integration_order(self):
        if self.integration_order == 0:
            self.integration_order = 2 * self.k_order + 1

    def storage_basis(self):
        self.evaluate_basis(self.data.quadrature.points, storage=True)

    def evaluate_mapping(self, points, storage=False):
        if self.data.dimension == 0:
            phi_shape = np.ones((1))
            cell_points = self.data.mesh.points[self.data.cell.node_tags]
            (x, jac, det_jac, inv_jac) = evaluate_mapping(
                self.data.dimension, phi_shape, cell_points
            )
            if storage:
                self.data.mapping.x = x
                self.data.mapping.jac = jac
                self.data.mapping.det_jac = det_jac
                self.data.mapping.inv_jac = inv_jac
            return (x, jac, det_jac, inv_jac)

        phi_shape = evaluate_linear_shapes(points, self.data)
        self.data.mapping.phi = phi_shape
        cell_points = self.data.mesh.points[self.data.cell.node_tags]
        (x, jac, det_jac, inv_jac) = evaluate_mapping(
            self.data.dimension, phi_shape, cell_points
        )
        if storage:
            self.data.mapping.x = x
            self.data.mapping.jac = jac
            self.data.mapping.det_jac = det_jac
            self.data.mapping.inv_jac = inv_jac

        return (x, jac, det_jac, inv_jac)

    def evaluate_basis(self, points, jac, det_jac, inv_jac):
        if self.data.dimension == 0:
            phi_tab = np.ones((1, 1, 1, 1))
            return phi_tab

        # tabulate
        phi_tab = self.basis_generator.tabulate(1, points)
        phi_mapped = self.basis_generator.push_forward(
            phi_tab[0], jac, det_jac, inv_jac
        )
        if phi_tab[0].shape == phi_mapped.shape:
            phi_tab[0] = phi_mapped
        else:
            phi_tab = np.insert(phi_tab, 2, 0, axis=3)
            phi_tab[0] = phi_mapped

        phi_tab = permute_and_transform(phi_tab, self.data)
        return phi_tab
