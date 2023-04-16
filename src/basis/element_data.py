from dataclasses import dataclass, field

import numpy as np

from mesh.mesh import Mesh
from mesh.mesh_cell import MeshCell


@dataclass
class QuadratureData:
    # Mesh and mesh entity
    points: np.ndarray = np.empty(0)
    weights: np.ndarray = np.empty(0)

    @classmethod
    def copy(self):
        points = self.points
        weights = self.weights
        return QuadratureData(points, weights)


@dataclass
class MappingData:

    # Geometrical mapping data
    phi: np.ndarray = np.empty(0)
    x: np.ndarray = np.empty(0)
    jac: np.ndarray = np.empty(0)
    det_jac: np.ndarray = np.empty(0)
    inv_jac: np.ndarray = np.empty(0)

    @classmethod
    def copy(self):
        phi = self.phi
        cell_points: np.ndarray = np.empty(0)
        x: np.ndarray = np.empty(0)
        jac: np.ndarray = np.empty(0)
        det_jac: np.ndarray = np.empty(0)
        inv_jac: np.ndarray = np.empty(0)
        return MappingData(phi, cell_points, x, jac, det_jac, inv_jac)


@dataclass
class DoFData:

    # entity dofs
    entity_dofs: np.ndarray = np.empty(0)

    # entity dofs
    transformations_are_identity: bool = True

    # transformations
    transformations: dict = field(default_factory=dict)

    # destination indexes
    dest: np.ndarray = np.empty(0)

    @classmethod
    def copy(self):
        entity_dofs = self.entity_dofs
        transformations_are_identity = self.transformations_are_identity
        transformations = self.transformations
        dest: np.ndarray = np.empty(0)
        return DoFData(entity_dofs, transformations_are_identity, transformations, dest)


@dataclass
class BasisData:

    # basis
    phi: np.ndarray = np.empty(0)

    @classmethod
    def copy(self):
        phi: np.ndarray = np.empty(0)
        return BasisData(phi)


@dataclass
class ElementData:
    dimension: int = -1
    cell: MeshCell = -1
    mesh: Mesh = None
    quadrature: QuadratureData = QuadratureData
    mapping: MappingData = MappingData
    dof: DoFData = DoFData
    basis: BasisData = BasisData

    @classmethod
    def copy(self):
        dimension = self.dimension
        cell = self.cell
        mesh = self.mesh
        quadrature = self.quadrature.copy()
        mapping = self.mapping.copy()
        dof = self.dof.copy()
        basis = self.basis.copy()
        return ElementData(dimension, cell, mesh, quadrature, mapping, dof, basis)
