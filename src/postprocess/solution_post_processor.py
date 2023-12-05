import basix
import meshio
import numpy as np

from basis.element_family import family_by_name
from geometry.mapping import evaluate_linear_shapes, evaluate_mapping
from spaces.product_space import ProductSpace


def write_vtk_file(file_name, gmesh, fe_space, alpha):
    dim = gmesh.dimension
    vec_families = [
        family_by_name("RT"),
        family_by_name("BDM"),
        family_by_name("N1E"),
        family_by_name("N2E"),
    ]
    p_data_dict = {}

    for item in fe_space.discrete_spaces.items():
        name, space = item
        n_comp = space.n_comp
        n_data = n_comp
        if space.family in vec_families:
            n_data *= dim

        fh_data = np.zeros((len(gmesh.points), n_data))
        cellid_to_element = dict(zip(space.element_ids, space.elements))

        vertices = space.mesh_topology.entities_by_dimension(0)
        cell_vertex_map = space.mesh_topology.entity_map_by_dimension(0)
        for id in vertices:
            if not cell_vertex_map.has_node(id):
                continue

            pr_ids = list(cell_vertex_map.predecessors(id))
            cell = gmesh.cells[pr_ids[0]]
            if cell.dimension != space.dimension:
                continue

            element = cellid_to_element[pr_ids[0]]

            # scattering dof
            index = fe_space.discrete_spaces[name].id_to_element[cell.id]
            dest = fe_space.discrete_spaces_destination_indexes(index)[name]
            alpha_l = alpha[dest]

            par_points = basix.geometry(space.element_type)

            target_node_id = gmesh.cells[id].node_tags[0]
            par_point_id = np.array(
                [
                    i
                    for i, node_id in enumerate(cell.node_tags)
                    if node_id == target_node_id
                ]
            )

            points = gmesh.points[target_node_id]
            if space.dimension != 0:
                points = par_points[par_point_id]

            # evaluate mapping
            phi_shapes = evaluate_linear_shapes(points, element.data)
            (x, jac, det_jac, inv_jac) = evaluate_mapping(
                cell.dimension, phi_shapes, gmesh.points[cell.node_tags]
            )
            phi_tab = element.evaluate_basis(points)
            n_phi = phi_tab.shape[2]

            alpha_star = np.array(np.split(alpha_l, n_phi))

            # Generalized displacement
            if space.family is family_by_name("Lagrange"):
                f_h = (phi_tab[0, :, :, 0] @ alpha_star[:, 0:dim]).T
            else:
                f_h = np.vstack(
                    tuple(
                        [
                            phi_tab[0, 0, :, 0:dim].T @ alpha_star[:, c]
                            for c in range(n_comp)
                        ]
                    )
                )
            fh_data[target_node_id] = f_h.ravel()

        p_data_dict.__setitem__(name + "_h", fh_data)

    mesh_points = gmesh.points
    cells = [cell for cell in gmesh.cells if cell.dimension == gmesh.dimension]
    con_d = np.array([cell.node_tags for cell in cells])
    meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
    cells_dict = {meshio_cell_types[gmesh.dimension]: con_d}

    mesh = meshio.Mesh(
        points=mesh_points,
        cells=cells_dict,
        # Optionally provide extra data on points, cells, etc.
        point_data=p_data_dict,
    )
    mesh.write(file_name)


def write_vtk_file_with_exact_solution(file_name, gmesh, fe_space, functions, alpha):
    dim = gmesh.dimension
    vec_families = [
        family_by_name("RT"),
        family_by_name("BDM"),
        family_by_name("N1E"),
        family_by_name("N2E"),
    ]
    p_data_dict = {}

    for item in fe_space.discrete_spaces.items():
        name, space = item
        n_comp = space.n_comp
        f_exact = functions[name]

        n_data = n_comp
        if space.family in vec_families:
            n_data *= dim

        fh_data = np.zeros((len(gmesh.points), n_data))
        fe_data = np.zeros((len(gmesh.points), n_data))
        cellid_to_element = dict(zip(space.element_ids, space.elements))

        vertices = space.mesh_topology.entities_by_dimension(0)
        cell_vertex_map = space.mesh_topology.entity_map_by_dimension(0)
        for id in vertices:
            if not cell_vertex_map.has_node(id):
                continue

            pr_ids = list(cell_vertex_map.predecessors(id))
            cell = gmesh.cells[pr_ids[0]]
            if cell.dimension != space.dimension:
                continue

            element = cellid_to_element[pr_ids[0]]

            # scattering dof
            index = fe_space.discrete_spaces[name].id_to_element[cell.id]
            dest = fe_space.discrete_spaces_destination_indexes(index)[name]
            alpha_l = alpha[dest]

            par_points = basix.geometry(space.element_type)

            target_node_id = gmesh.cells[id].node_tags[0]
            par_point_id = np.array(
                [
                    i
                    for i, node_id in enumerate(cell.node_tags)
                    if node_id == target_node_id
                ]
            )

            points = gmesh.points[target_node_id]
            if space.dimension != 0:
                points = par_points[par_point_id]

            # evaluate mapping
            phi_shapes = evaluate_linear_shapes(points, element.data)
            (x, jac, det_jac, inv_jac) = evaluate_mapping(
                cell.dimension, phi_shapes, gmesh.points[cell.node_tags]
            )
            phi_tab = element.evaluate_basis(points)
            n_phi = phi_tab.shape[2]

            alpha_star = np.array(np.split(alpha_l, n_phi))

            # Generalized displacement
            if space.family is family_by_name("Lagrange"):
                f_e = f_exact(x[:, 0], x[:, 1], x[:, 2])
                f_h = (phi_tab[0, :, :, 0] @ alpha_star[:, 0:dim]).T
            else:
                f_e = f_exact(x[0, 0], x[0, 1], x[0, 2])
                f_h = np.vstack(
                    tuple(
                        [
                            phi_tab[0, 0, :, 0:dim].T @ alpha_star[:, c]
                            for c in range(n_comp)
                        ]
                    )
                )
            fh_data[target_node_id] = f_h.ravel()
            fe_data[target_node_id] = f_e.ravel()

        p_data_dict.__setitem__(name + "_h", fh_data)
        p_data_dict.__setitem__(name + "_e", fe_data)

    mesh_points = gmesh.points
    cells = [cell for cell in gmesh.cells if cell.dimension == gmesh.dimension]
    con_d = np.array([cell.node_tags for cell in cells])
    meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
    cells_dict = {meshio_cell_types[gmesh.dimension]: con_d}

    mesh = meshio.Mesh(
        points=mesh_points,
        cells=cells_dict,
        # Optionally provide extra data on points, cells, etc.
        point_data=p_data_dict,
    )
    mesh.write(file_name)
