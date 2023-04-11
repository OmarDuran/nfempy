from itertools import permutations

import basix
import numpy as np
from basix import CellType, ElementFamily, LagrangeVariant


class FiniteElement:
    def __init__(self, id, mesh, k_order, family, discontinuous=False, int_order=0):
        self.mesh = mesh
        self.cell = mesh.cells[id]
        self.k_order = k_order
        self.discontinuous = discontinuous
        self.int_order = int_order
        self.family = family
        self.variant = None
        self.basis_generator = None
        self.dof_ordering = None
        self.quadrature = None
        self.mapping = None
        self.phi = None
        self._build_structures()

    def _build_structures(self):

        # fecth information from static methods
        element_family = self.basis_family(self.family)
        cell_type = self.type_by_dimension(self.cell.dimension)
        self.variant = self.basis_variant()

        self._set_integration_order()
        self.basis_generator = basix.create_element(
            element_family, cell_type, self.k_order, self.variant, self.discontinuous
        )
        self.dof_ordering = self._dof_premutations()
        self.quadrature = basix.make_quadrature(
            basix.QuadratureType.gauss_jacobi, cell_type, self.int_order
        )
        self.evaluate_basis(self.quadrature[0], True)

    @staticmethod
    def basis_variant():
        return LagrangeVariant.gll_centroid

    @staticmethod
    def basis_family(family):
        families = {
            "Lagrange": ElementFamily.P,
            "BDM": ElementFamily.BDM,
            "RT": ElementFamily.RT,
            "N1E": ElementFamily.N1E,
            "N2E": ElementFamily.N1E,
        }
        return families[family]

    @staticmethod
    def type_by_dimension(dimension):
        element_types = {
            0: CellType.point,
            1: CellType.interval,
            2: CellType.triangle,
            3: CellType.tetrahedron,
        }
        return element_types[dimension]

    def _set_integration_order(self):
        if self.int_order == 0:
            self.int_order = 2 * self.k_order + 1

    def _dof_premutations(self):
        # this permutation is because basix uses different entity ordering.
        if self.cell.dimension == 0:
            return self._dof_perm_point()
        elif self.cell.dimension == 1:
            return self._dof_perm_interval()
        elif self.cell.dimension == 2:
            return self._dof_perm_triangle()
        elif self.cell.dimension == 3:
            return self._dof_perm_tetrahedron()

    def _dof_perm_point(self):
        indices = np.array([0], dtype=int)
        return indices

    def _dof_perm_interval(self):
        indices = np.array([], dtype=int)
        for dim, entity_dof in enumerate(self.basis_generator.entity_dofs):
            indices = np.append(indices, np.array(entity_dof, dtype=int))
        return indices.ravel()

    def _dof_perm_triangle(self):
        indices = np.array([], dtype=int)
        e_perms = np.array([1, 2, 0])
        for dim, entity_dof in enumerate(self.basis_generator.entity_dofs):
            if dim == 1:
                entity_dof = [entity_dof[i] for i in e_perms]
                indices = np.append(indices, np.array(entity_dof, dtype=int))
            else:
                indices = np.append(indices, np.array(entity_dof, dtype=int))
        return indices.ravel()

    def _dof_perm_tetrahedron(self):

        indices = np.array([], dtype=int)
        e_edge_perms = np.array([0, 1, 2, 3, 4, 5])
        e_face_perms = np.array([0, 1, 2, 3])
        for dim, entity_dof in enumerate(self.basis_generator.entity_dofs):
            if dim == 2:
                entity_dof = [entity_dof[i] for i in e_face_perms]
                indices = np.append(indices, np.array(entity_dof, dtype=int))
            elif dim == 1:
                entity_dof = [entity_dof[i] for i in e_edge_perms]
                indices = np.append(indices, np.array(entity_dof, dtype=int))
            else:
                indices = np.append(indices, np.array(entity_dof, dtype=int))
        return indices.ravel()

    def compute_mapping(self, points, storage=False):
        cell_points = self.mesh.points[self.cell.node_tags]
        cell_type = self.type_by_dimension(self.cell.dimension)
        linear_element = basix.create_element(
            ElementFamily.P, cell_type, 1, LagrangeVariant.equispaced
        )
        # points, weights = self.quadrature
        phi = linear_element.tabulate(1, points)
        dim = self.cell.dimension

        # Compute geometrical transformations
        x = phi[0, :, :, 0] @ cell_points
        jac = np.rollaxis(phi[list(range(1, dim + 1)), :, :, 0], 1) @ cell_points
        jac = np.transpose(jac, (0, 2, 1))

        def compute_det_and_pseudo_inverse(grad_xmap):
            # QR-decomposition is not unique
            q_axes, r_jac = np.linalg.qr(grad_xmap)
            det_g_jac = np.linalg.det(r_jac)

            # It's only unique up to the signs of the rows of R
            r_sign = np.diag(np.sign(np.diag(r_jac)), 0)
            q_axes = np.dot(q_axes, r_sign)
            r_jac = np.dot(r_sign, r_jac)
            det_g_jac = np.linalg.det(r_jac)
            if det_g_jac < 0.0:
                print("Negative det jac: ", det_g_jac)
            inv_g_jac = np.dot(np.linalg.inv(r_jac), q_axes.T)
            return det_g_jac, inv_g_jac

        map_result = list(map(compute_det_and_pseudo_inverse, jac))
        det_jac, inv_jac = zip(*map_result)
        det_jac = np.array(det_jac)
        inv_jac = np.array(inv_jac)

        if storage:
            self.mapping = (x, jac, det_jac, inv_jac)
        return (x, jac, det_jac, inv_jac)

    def _validate_edge_orientation_2d(self):
        connectiviy = np.array([[0, 1], [1, 2], [2, 0]])
        e_perms = np.array([1, 2, 0])
        orientation = [False, False, False]
        for i, con in enumerate(connectiviy):
            edge = self.cell.node_tags[con]
            v_edge = self.mesh.cells[self.cell.sub_cells_ids[1][i]].node_tags
            if np.any(edge == v_edge):
                orientation[i] = True
        orientation = [orientation[i] for i in e_perms]
        return orientation

    def _permute_and_transform_basis_2d(self, phi_tab):

        # make functions outward
        if not self.basis_generator.dof_transformations_are_identity:
            # n_dof = int(np.mean(self.basis_generator.num_entity_dofs[1]))
            for index in [0, 2]:
                transformation = self.basis_generator.entity_transformations()[
                    "interval"
                ][0]
                dofs = self.basis_generator.entity_dofs[1][index]
                for dim in range(phi_tab.shape[2]):
                    phi_tab[:, dofs, dim] = phi_tab[:, dofs, dim] @ transformation.T

        # triangle ref connectivity
        if not self.basis_generator.dof_transformations_are_identity:
            oriented_q = self._validate_edge_orientation_2d()
            for index, check in enumerate(oriented_q):
                if check:
                    continue
                transformation = self.basis_generator.entity_transformations()[
                    "interval"
                ][0]
                dofs = self.basis_generator.entity_dofs[1][index]
                for dim in range(phi_tab.shape[2]):
                    phi_tab[:, dofs, dim] = phi_tab[:, dofs, dim] @ transformation.T
        return phi_tab

    def _validate_edge_orientation_3d(self):

        edge_0 = np.array([0, 1])
        edge_1 = np.array([1, 2])
        edge_2 = np.array([2, 0])
        edge_3 = np.array([0, 3])
        edge_4 = np.array([2, 3])
        edge_5 = np.array([1, 3])

        edge_0 = np.array([2, 3])
        edge_1 = np.array([1, 3])
        edge_2 = np.array([1, 2])
        edge_3 = np.array([0, 3])
        edge_4 = np.array([0, 2])
        edge_5 = np.array([0, 1])
        edges = [edge_0, edge_1, edge_2, edge_3, edge_4, edge_5]

        # e_perms = np.array([5, 2, 4, 3, 0, 1])
        e_perms = np.array([0, 1, 2, 3, 4, 5])
        orientation = [False, False, False, False, False, False]
        for i, egde_con in enumerate(edges):
            edge = self.cell.node_tags[egde_con]
            edge_node_tags = self.mesh.cells[self.cell.sub_cells_ids[1][i]].node_tags
            v_edge = np.sort(edge_node_tags)
            if np.all(edge == v_edge):
                orientation[i] = True
        orientation = [orientation[i] for i in e_perms]
        return orientation

    def _validate_face_orientation_3d(self):

        face_0 = np.array([1, 2, 3])
        face_1 = np.array([0, 2, 3])
        face_2 = np.array([0, 1, 3])
        face_3 = np.array([0, 1, 2])
        faces = [face_0, face_1, face_2, face_3]
        face_perms = [0, 1, 2, 3]

        orientation = [False, False, False, False]
        rotations = [0, 0, 0, 0]
        reflections = [0, 0, 0, 0]
        for i, face_con in enumerate(faces):
            volume_node_tags = self.cell.node_tags
            face = volume_node_tags[face_con]
            face_ref = np.sort(face)
            face_cell = self.mesh.cells[self.cell.sub_cells_ids[2][i]]

            valid_orientation = np.all(face_ref == face)
            if valid_orientation:
                orientation[i] = True
            else:
                perms = list(permutations(list(face_ref)))
                pos_search = [i for i, perm in enumerate(perms) if perm == tuple(face)]
                assert len(pos_search) == 1
                assert pos_search[0] != 0
                position = pos_search[0]

                # cases
                if position == 0:
                    rotations[i] = 0
                    reflections[i] = 0
                elif position == 1:
                    rotations[i] = 0
                    reflections[i] = 1
                elif position == 2:
                    rotations[i] = 2
                    reflections[i] = 1
                elif position == 3:
                    rotations[i] = 1
                    reflections[i] = 0
                elif position == 4:
                    rotations[i] = 2
                    reflections[i] = 0
                elif position == 5:
                    rotations[i] = 1
                    reflections[i] = 1

        orientation = [orientation[i] for i in face_perms]
        rotations = [rotations[i] for i in face_perms]
        reflections = [reflections[i] for i in face_perms]
        return orientation, rotations, reflections

    def _permute_and_transform_basis_3d(self, phi_tab):

        if not self.basis_generator.dof_transformations_are_identity:
            oriented_q, rot_data, ref_data = self._validate_face_orientation_3d()
            for index, check in enumerate(oriented_q):
                if check:
                    continue
                rotate_t, reflect_t = self.basis_generator.entity_transformations()[
                    "triangle"
                ]

                transformation = np.identity(len(rotate_t))
                for i in range(rot_data[index]):
                    transformation = rotate_t @ transformation
                for i in range(ref_data[index]):
                    transformation = reflect_t @ transformation

                dofs = self.basis_generator.entity_dofs[2][index]
                for dim in range(phi_tab.shape[2]):
                    phi_tab[:, dofs, dim] = phi_tab[:, dofs, dim] @ transformation.T

            # reflect edge orientation
            oriented_q = self._validate_edge_orientation_3d()
            for index, check in enumerate(oriented_q):
                if check:
                    continue
                reflect_t = self.basis_generator.entity_transformations()["interval"][0]
                dofs = self.basis_generator.entity_dofs[1][index]
                for dim in range(phi_tab.shape[2]):
                    phi_tab[:, dofs, dim] = phi_tab[:, dofs, dim] @ reflect_t.T

        return phi_tab

    def _permute_and_transform_basis(self, phi_tab):
        if self.cell.dimension == 2:
            return self._permute_and_transform_basis_2d(phi_tab)
        elif self.cell.dimension == 3:
            return self._permute_and_transform_basis_3d(phi_tab)
        else:
            return phi_tab

    def evaluate_basis(self, points, storage=False):

        (x, jac, det_jac, inv_jac) = self.compute_mapping(points, storage)
        # map functions
        phi_hat_tab = self.basis_generator.tabulate(1, points)
        phi_tab = self.basis_generator.push_forward(
            phi_hat_tab[0], jac, det_jac, inv_jac
        )
        phi_tab = self._permute_and_transform_basis(phi_tab)
        if storage:
            self.phi = phi_tab
        return phi_tab
