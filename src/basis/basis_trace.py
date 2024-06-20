from spaces.product_space import ProductSpace
from spaces.discrete_space import DiscreteSpace
from basis.element_data import ElementData
from basis.finite_element import FiniteElement
from basis.parametric_transformation import transform_lower_to_higher
from mesh.topological_queries import find_higher_dimension_neighs


def _trace(points, c1_data: ElementData, c0_element: FiniteElement, return_all_dof_q):
    c1_cell = c1_data.cell
    c0_cell = c0_element.data.cell

    mapped_points = transform_lower_to_higher(points, c1_data, c0_element.data)
    _, jac_c0, det_jac_c0, inv_jac_c0 = c0_element.evaluate_mapping(mapped_points)
    tr_phi_tab = c0_element.evaluate_basis(
        mapped_points, jac_c0, det_jac_c0, inv_jac_c0
    )
    if return_all_dof_q:
        tr_phi = tr_phi_tab[:, :, :, :]
    else:
        c1_entity_index = (
            c0_cell.sub_cells_ids[c1_cell.dimension].tolist().index(c1_cell.id)
        )
        dof_n_index = c0_element.data.dof.entity_dofs[c1_cell.dimension][
            c1_entity_index
        ]
        tr_phi = tr_phi_tab[:, :, dof_n_index, :]
    return tr_phi


def trace_discrete_space(
    sub_space: DiscreteSpace,
    points,
    c1_data: ElementData,
    return_all_dof_q,
    neigh_idx=0,
):

    c1_cell = c1_data.cell
    neigh_list = find_higher_dimension_neighs(c1_cell, sub_space.dof_map.mesh_topology)
    neigh_check = len(neigh_list) > 0
    assert neigh_check

    # select neighbor
    c0_cell_id = neigh_list[neigh_idx]
    c0_element_idx = sub_space.id_to_element[c0_cell_id]
    c0_element = sub_space.elements[c0_element_idx]

    tr_phi = _trace(points, c1_data, c0_element, return_all_dof_q)
    return tr_phi


def trace_product_space(
    fields,
    space: ProductSpace,
    points,
    c1_data: ElementData,
    return_all_dof_q,
    neigh_idx=0,
):
    traces_map = {}
    sub_spaces = {field: space.discrete_spaces[field] for field in fields}
    for item in sub_spaces.items():
        field, sub_space = item
        tr_phi = trace_discrete_space(
            sub_space, points, c1_data, return_all_dof_q, neigh_idx
        )
        traces_map[field] = tr_phi
    return traces_map
