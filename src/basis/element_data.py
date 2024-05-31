import numpy as np

from mesh.mesh import Mesh
from mesh.mesh_cell import MeshCell


class QuadratureData:
    def __init__(
        self, points: np.ndarray = np.empty(0), weights: np.ndarray = np.empty(0)
    ):
        # Mesh and mesh entity
        self.points = points
        self.weights = weights

    @classmethod
    def copy(self):
        points = self.points
        weights = self.weights
        return QuadratureData(points, weights)


class MappingData:
    def __init__(
        self,
        phi: np.ndarray = np.empty(0),
        x: np.ndarray = np.empty(0),
        jac: np.ndarray = np.empty(0),
        det_jac: np.ndarray = np.empty(0),
        inv_jac: np.ndarray = np.empty(0),
    ):
        # Geometrical mapping data
        phi: np.ndarray = phi
        x: np.ndarray = x
        jac: np.ndarray = jac
        det_jac: np.ndarray = det_jac
        inv_jac: np.ndarray = inv_jac

    @classmethod
    def copy(self):
        phi = self.phi
        cell_points: np.ndarray = np.empty(0)
        x: np.ndarray = np.empty(0)
        jac: np.ndarray = np.empty(0)
        det_jac: np.ndarray = np.empty(0)
        inv_jac: np.ndarray = np.empty(0)
        return MappingData(phi, cell_points, x, jac, det_jac, inv_jac)


class DoFData:
    def __init__(
        self,
        entity_dofs: np.ndarray = np.empty(0),
        transformations_are_identity: bool = True,
        transformations: dict = {},
        num_entity_dofs: np.ndarray = np.empty(0),
    ):
        # entity dofs
        self.entity_dofs: np.ndarray = entity_dofs

        # entity dofs
        self.transformations_are_identity: bool = transformations_are_identity

        # transformations
        self.transformations: dict = transformations

        # num entity dofs
        self.num_entity_dofs: np.ndarray = num_entity_dofs

    @classmethod
    def copy(self):
        entity_dofs = self.entity_dofs
        transformations_are_identity = self.transformations_are_identity
        transformations = self.transformations
        num_entity_dofs = self.num_entity_dofs
        return DoFData(
            entity_dofs, transformations_are_identity, transformations, num_entity_dofs
        )


class BCEntities:
    def __init__(
        self,
        bc_entities: np.ndarray = np.empty(0),
    ):
        # basis
        self.bc_entities: np.ndarray = bc_entities

    @classmethod
    def copy(self):
        bc_entities: np.ndarray = np.empty(0)
        return BCEntities(bc_entities)


class BasisData:
    def __init__(
        self,
        phi: np.ndarray = np.empty(0),
    ):
        # basis
        self.phi: np.ndarray = phi

    @classmethod
    def copy(self):
        phi: np.ndarray = np.empty(0)
        return BasisData(phi)


class ElementData:
    def __init__(self, cell: MeshCell = None, mesh: Mesh = None):
        self.cell: MeshCell = cell
        self.mesh: Mesh = mesh
        self.quadrature: QuadratureData = QuadratureData()
        self.mapping: MappingData = MappingData()
        self.dof: DoFData = DoFData()
        self.bc_entities: BCEntities = BCEntities()
        self.basis: BasisData = BasisData()

    @classmethod
    def copy(self):
        cell = None
        mesh = None
        quadrature = self.quadrature.copy()
        mapping = self.mapping.copy()
        dof = self.dof.copy()
        basis = self.basis.copy()
        return ElementData(cell, mesh, quadrature, mapping, dof, basis)
