from dataclasses import dataclass, field

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
        dest: np.ndarray = np.empty(0),
    ):

        # entity dofs
        self.entity_dofs: np.ndarray = entity_dofs

        # entity dofs
        self.transformations_are_identity: bool = transformations_are_identity

        # transformations
        self.transformations: dict = transformations

        # destination indexes
        self.dest: np.ndarray = dest

    @classmethod
    def copy(self):
        entity_dofs = self.entity_dofs
        transformations_are_identity = self.transformations_are_identity
        transformations = self.transformations
        dest: np.ndarray = np.empty(0)
        return DoFData(entity_dofs, transformations_are_identity, transformations, dest)


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
    def __init__(self, dimension: int = -1, cell: MeshCell = None, mesh: Mesh = None):

        self.dimension: int = dimension
        self.cell: MeshCell = cell
        self.mesh: Mesh = mesh
        self.quadrature: QuadratureData = QuadratureData()
        self.mapping: MappingData = MappingData()
        self.dof: DoFData = DoFData()
        self.basis: BasisData = BasisData()

    @classmethod
    def copy(self):
        dimension = self.dimension
        cell = None
        mesh = None
        quadrature = self.quadrature.copy()
        mapping = self.mapping.copy()
        dof = self.dof.copy()
        basis = self.basis.copy()
        return ElementData(dimension, cell, mesh, quadrature, mapping, dof, basis)
