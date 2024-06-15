import basix
import numpy as np

from basis.element_data import ElementData
from basis.element_family import basis_variant, family_by_name
from basis.element_type import type_by_dimension


def evaluate_linear_shapes(points, data: ElementData):
    cell_type = type_by_dimension(data.cell.dimension)
    family = family_by_name("Lagrange")
    variant = basis_variant()
    linear_element = basix.create_element(
        family=family,
        celltype=cell_type,
        degree=1,
        lagrange_variant=variant,
    )
    phi = linear_element.tabulate(1, points)
    return phi


def _compute_det_and_pseudo_inverse(grad_xmap):
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


def evaluate_mapping(dimension, phi, cell_points):
    if dimension == 0:
        x = cell_points
        jac = det_jac = inv_jac = np.array([1.0])
        return (x, jac, det_jac, inv_jac)

    # Compute geometrical transformations
    x = phi[0, :, :, 0] @ cell_points
    jac = np.rollaxis(phi[list(range(1, dimension + 1)), :, :, 0], 1) @ cell_points
    jac = np.transpose(jac, (0, 2, 1))

    map_result = list(map(_compute_det_and_pseudo_inverse, jac))
    det_jac, inv_jac = zip(*map_result)
    det_jac = np.array(det_jac)
    inv_jac = np.array(inv_jac)

    return (x, jac, det_jac, inv_jac)


def store_mapping(data: ElementData):
    (x, jac, det_jac, inv_jac) = evaluate_mapping(
        data.cell.dimension, data.mapping.phi, data.mapping.cell_points
    )
    data.mapping.x = x
    data.mapping.jac = jac
    data.mapping.det_jac = det_jac
    data.mapping.inv_jac = inv_jac
