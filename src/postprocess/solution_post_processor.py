import basix
import meshio
import numpy as np

from mesh.mesh_metrics import cell_centroid
from basis.element_family import family_by_name
from geometry.mapping import evaluate_linear_shapes, evaluate_mapping
from mesh.mesh_topology import MeshTopology


def node_average_quatity(node_idx, cell_idxs, fe_space, name, alpha):
    space = fe_space.discrete_spaces[name]
    n_comp = space.n_comp

    gmesh = space.mesh_topology.mesh
    dim = gmesh.dimension

    cells = [gmesh.cells[dim_id[1]] for dim_id in cell_idxs]

    f_h_values = []
    for cell in cells:
        if cell.material_id not in fe_space.fields_physical_tags[name]:
            continue
        element_idx = space.id_to_element[cell.id]
        element = space.elements[element_idx]

        dest = fe_space.discrete_spaces_destination_indexes(element_idx)[name]
        alpha_l = alpha[dest]

        par_points = basix.geometry(space.element_type)

        par_point_id = np.array(
            [i for i, node_id in enumerate(cell.node_tags) if node_id == node_idx]
        )
        assert len(par_point_id) == 1

        points = gmesh.points[node_idx]
        if space.dimension != 0:
            points = par_points[par_point_id]

        # evaluate mapping
        phi_shapes = evaluate_linear_shapes(points, element.data)
        (x, jac, det_jac, inv_jac) = evaluate_mapping(
            cell.dimension, phi_shapes, gmesh.points[cell.node_tags]
        )
        phi_tab = element.evaluate_basis(points, jac, det_jac, inv_jac)
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
        f_h_values.append(f_h)

    # compute average
    f_h = np.mean(np.array(f_h_values), axis=0)
    return f_h


def cell_centered_quatity(cell, fe_space, name, alpha):
    space = fe_space.discrete_spaces[name]
    n_comp = space.n_comp

    gmesh = space.mesh_topology.mesh
    dim = gmesh.dimension

    element_idx = space.id_to_element[cell.id]
    element = space.elements[element_idx]

    dest = fe_space.discrete_spaces_destination_indexes(element_idx)[name]
    alpha_l = alpha[dest]

    par_points = basix.geometry(space.element_type)
    point = np.array([np.mean(par_points, axis=0)])

    # evaluate mapping
    phi_shapes = evaluate_linear_shapes(point, element.data)
    (x, jac, det_jac, inv_jac) = evaluate_mapping(
        cell.dimension, phi_shapes, gmesh.points[cell.node_tags]
    )
    phi_tab = element.evaluate_basis(point, jac, det_jac, inv_jac)
    n_phi = phi_tab.shape[2]

    alpha_star = np.array(np.split(alpha_l, n_phi))

    # Generalized displacement
    if space.family is family_by_name("Lagrange"):
        f_h = (phi_tab[0, :, :, 0] @ alpha_star[:, 0:dim]).T
    else:
        f_h = np.vstack(
            tuple([phi_tab[0, 0, :, 0:dim].T @ alpha_star[:, c] for c in range(n_comp)])
        )
    return f_h


def write_vtk_file(file_name, gmesh, fe_space, alpha, cell_centered=[]):
    dim = gmesh.dimension
    vec_families = [
        family_by_name("RT"),
        family_by_name("BDM"),
        family_by_name("N1E"),
        family_by_name("N2E"),
    ]
    p_data_dict = {}
    c_data_dict = {}

    for item in fe_space.discrete_spaces.items():
        name, space = item
        n_comp = space.n_comp
        n_data = n_comp
        if space.family in vec_families:
            n_data *= dim

        if name in cell_centered:
            cells = [cell for cell in gmesh.cells if cell.dimension == dim]
            fh_data = np.zeros((len(cells), n_data))

            for cell_idx, cell in enumerate(cells):
                f_h = cell_centered_quatity(cell, fe_space, name, alpha)
                fh_data[cell_idx] = f_h.ravel()

            c_data_dict[name + "_h"] = [fh_data]
        else:
            fh_data = np.zeros((len(gmesh.points), n_data))
            vertices = space.mesh_topology.entities_by_dimension(0)
            cell_vertex_map = space.mesh_topology.entity_map_by_dimension(0)
            for vertex_idx in vertices:
                vertex_g_index = (0, vertex_idx)
                if not cell_vertex_map.has_node(vertex_g_index):
                    continue
                node_idx = gmesh.cells[vertex_idx].node_tags[0]
                cell_idxs = list(cell_vertex_map.predecessors(vertex_g_index))

                f_h = node_average_quatity(node_idx, cell_idxs, fe_space, name, alpha)
                fh_data[node_idx] = f_h.ravel()

            p_data_dict[name + "_h"] = fh_data

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
        cell_data=c_data_dict,
    )
    mesh.write(file_name)


def write_vtk_file_with_exact_solution(
    file_name, gmesh, fe_space, functions, alpha, cell_centered=[]
):
    dim = gmesh.dimension
    vec_families = [
        family_by_name("RT"),
        family_by_name("BDM"),
        family_by_name("N1E"),
        family_by_name("N2E"),
    ]
    p_data_dict = {}
    c_data_dict = {}

    # dims = []
    # for item in fe_space.discrete_spaces.items():
    #     _, space = item
    #     dims.append(space.dimension)
    # product_space_dim = np.max(dims)
    # mesh_topology: MeshTopology = MeshTopology(gmesh, product_space_dim)
    # mesh_topology.build_data()

    for item in fe_space.discrete_spaces.items():
        name, space = item
        n_comp = space.n_comp
        f_exact = functions[name]

        n_data = n_comp
        if space.family in vec_families:
            n_data *= dim

        if name in cell_centered:
            cells = [cell for cell in gmesh.cells if cell.dimension == dim]
            fh_data = np.zeros((len(cells), n_data))
            fe_data = np.zeros((len(cells), n_data))

            for cell_idx, cell in enumerate(cells):
                x = cell_centroid(cell, gmesh)
                f_e = f_exact(x[0], x[1], x[2])
                f_h = cell_centered_quatity(cell, fe_space, name, alpha)
                fh_data[cell_idx] = f_h.ravel()
                fe_data[cell_idx] = f_e.ravel()

            c_data_dict[name + "_h"] = [fh_data]
            c_data_dict[name + "_e"] = [fe_data]
        else:
            fh_data = np.zeros((len(gmesh.points), n_data))
            fe_data = np.zeros((len(gmesh.points), n_data))

            vertices = space.mesh_topology.entities_by_dimension(0)
            cell_vertex_map = space.mesh_topology.entity_map_by_dimension(0)
            for vertex_idx in vertices:
                vertex_g_index = (0, vertex_idx)
                if not cell_vertex_map.has_node(vertex_g_index):
                    continue
                node_idx = gmesh.cells[vertex_idx].node_tags[0]
                cell_idxs = list(cell_vertex_map.predecessors(vertex_g_index))

                x = gmesh.points[node_idx]
                f_e = f_exact(x[0], x[1], x[2])
                f_h = node_average_quatity(node_idx, cell_idxs, fe_space, name, alpha)
                fh_data[node_idx] = f_h.ravel()
                fe_data[node_idx] = f_e.ravel()

            p_data_dict[name + "_h"] = fh_data
            p_data_dict[name + "_e"] = fe_data

    mesh_points = gmesh.points
    cells = [
        cell
        for cell in gmesh.cells
        if cell.material_id in fe_space.fields_physical_tags[name]
    ]
    con_d = np.array([cell.node_tags for cell in cells])
    meshio_cell_types = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
    cells_dict = {meshio_cell_types[fe_space.dimension()]: con_d}

    mesh = meshio.Mesh(
        points=mesh_points,
        cells=cells_dict,
        # Optionally provide extra data on points, cells, etc.
        point_data=p_data_dict,
        cell_data=c_data_dict,
    )
    mesh.write(file_name)


def write_vtk_file_pointwise_l2_error(
    file_name, gmesh, fe_space, functions, alpha, cell_centered=[]
):
    dim = gmesh.dimension
    vec_families = [
        family_by_name("RT"),
        family_by_name("BDM"),
        family_by_name("N1E"),
        family_by_name("N2E"),
    ]
    p_data_dict = {}
    c_data_dict = {}

    for item in fe_space.discrete_spaces.items():
        name, space = item
        n_comp = space.n_comp
        f_exact = functions[name]

        if name in cell_centered:
            cells = [cell for cell in gmesh.cells if cell.dimension == dim]
            eh_data = np.zeros((len(cells), 1))

            for cell_idx, cell in enumerate(cells):
                x = cell_centroid(cell, gmesh)
                f_e = f_exact(x[0], x[1], x[2])
                f_h = cell_centered_quatity(cell, fe_space, name, alpha)
                e_f = np.linalg.norm(f_e - f_h)
                eh_data[cell_idx] = e_f.ravel()

            c_data_dict[name + "_error"] = [eh_data]
        else:
            eh_data = np.zeros((len(gmesh.points), 1))

            vertices = space.mesh_topology.entities_by_dimension(0)
            cell_vertex_map = space.mesh_topology.entity_map_by_dimension(0)
            for vertex_idx in vertices:
                vertex_g_index = (0, vertex_idx)
                if not cell_vertex_map.has_node(vertex_g_index):
                    continue
                node_idx = gmesh.cells[vertex_idx].node_tags[0]
                cell_idxs = list(cell_vertex_map.predecessors(vertex_g_index))

                x = gmesh.points[node_idx]
                f_e = f_exact(x[0], x[1], x[2])
                f_h = node_average_quatity(node_idx, cell_idxs, fe_space, name, alpha)
                e_f = np.linalg.norm(f_e - f_h)
                eh_data[node_idx] = e_f.ravel()

            p_data_dict[name + "_error"] = eh_data

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
        cell_data=c_data_dict,
    )
    mesh.write(file_name)


def write_vtk_file_exact_solution(file_name, gmesh, name_to_fields, functions):
    dim = gmesh.dimension
    p_data_dict = {}
    for item in name_to_fields.items():
        name, n_data = item
        f_exact = functions[name]
        fe_data = np.zeros((len(gmesh.points), n_data))
        for ip, x in enumerate(gmesh.points):
            f_e = f_exact(x[0], x[1], x[2])
            fe_data[ip] = f_e.ravel()
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
