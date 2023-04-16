from itertools import permutations

import numpy as np

from basis.element_data import ElementData
from mesh.mesh import Mesh
from mesh.mesh_cell import MeshCell


def _validate_edge_orientation_2d(data: ElementData):
    edge_0 = data.cell.node_tags[np.array([1, 2])]
    edge_1 = data.cell.node_tags[np.array([0, 2])]
    edge_2 = data.cell.node_tags[np.array([0, 1])]
    edges = [edge_0, edge_1, edge_2]

    orientation = [False, False, False]
    for i, edge in enumerate(edges):
        v_edge = data.mesh.cells[data.cell.sub_cells_ids[1][i]].node_tags
        if np.all(edge == v_edge):
            orientation[i] = True
    return orientation


def _permute_and_transform_2d(phi, data: ElementData):

    identity_transformation_q = data.dof.transformations_are_identity
    # triangle reflections
    if not identity_transformation_q:
        oriented_q = _validate_edge_orientation_2d(data)
        for index, check in enumerate(oriented_q):
            if check:
                continue
            transformation = data.dof.transformations["interval"][0]
            dofs = data.dof.entity_dofs[1][index]
            for d in range(phi.shape[0]):
                for dim in range(phi.shape[3]):
                    phi[d, :, dofs, dim] = transformation @ phi[d, :, dofs, dim]
    return phi


def _validate_edge_orientation_3d(data: ElementData):
    edge_0 = np.array([2, 3])
    edge_1 = np.array([1, 3])
    edge_2 = np.array([1, 2])
    edge_3 = np.array([0, 3])
    edge_4 = np.array([0, 2])
    edge_5 = np.array([0, 1])
    edges = [edge_0, edge_1, edge_2, edge_3, edge_4, edge_5]

    orientation = [False, False, False, False, False, False]
    for i, egde_con in enumerate(edges):
        edge = data.cell.node_tags[egde_con]
        edge_node_tags = data.mesh.cells[data.cell.sub_cells_ids[1][i]].node_tags
        v_edge = np.sort(edge_node_tags)
        if np.all(edge == v_edge):
            orientation[i] = True
    return orientation


def _validate_face_orientation_3d(data: ElementData):

    face_0 = np.array([1, 2, 3])
    face_1 = np.array([0, 2, 3])
    face_2 = np.array([0, 1, 3])
    face_3 = np.array([0, 1, 2])
    faces = [face_0, face_1, face_2, face_3]

    orientation = [False, False, False, False]
    rotations = [0, 0, 0, 0]
    reflections = [0, 0, 0, 0]
    for i, face_con in enumerate(faces):
        volume_node_tags = data.cell.node_tags
        face = volume_node_tags[face_con]
        face_ref = np.sort(face)
        face_cell = data.mesh.cells[data.cell.sub_cells_ids[2][i]]

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
    return orientation, rotations, reflections


def _permute_and_transform_3d(phi, data: ElementData):

    identity_transformation_q = data.dof.transformations_are_identity
    if not identity_transformation_q:
        oriented_q, rot_data, ref_data = _validate_face_orientation_3d(data)
        for index, check in enumerate(oriented_q):
            if check:
                continue
            rotate_t, reflect_t = data.dof.transformations["triangle"]

            transformation = np.identity(len(rotate_t))
            for i in range(rot_data[index]):
                transformation = rotate_t @ transformation
            for i in range(ref_data[index]):
                transformation = reflect_t @ transformation

            dofs = data.dof.entity_dofs[2][index]
            for d in range(phi.shape[0]):
                for dim in range(phi.shape[3]):
                    phi[d, :, dofs, dim] = transformation @ phi[d, :, dofs, dim]

        # reflect edge orientation
        oriented_q = _validate_edge_orientation_3d(data)
        for index, check in enumerate(oriented_q):
            if check:
                continue
            reflect_t = data.dof.transformations["interval"][0]
            dofs = data.dof.entity_dofs[1][index]
            for d in range(phi.shape[0]):
                for dim in range(phi.shape[3]):
                    phi[d, :, dofs, dim] = reflect_t @ phi[d, :, dofs, dim]
        return phi


def permute_and_transform(phi, data: ElementData):

    if data.dimension == 2:
        return _permute_and_transform_2d(phi, data)
    elif data.dimension == 3:
        return _permute_and_transform_3d(phi, data)
    else:
        return phi
