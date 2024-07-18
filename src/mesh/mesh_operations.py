import numpy as np
import networkx as nx
import itertools
from functools import partial
from mesh.mesh import Mesh
from mesh.mesh_metrics import cell_centroid
from geometry.operations.point_geometry_operations import points_line_intersection
from geometry.operations.point_geometry_operations import points_line_argsort


# def cut_conformity_along_c1_line(line: np.array, physical_tags, mesh: Mesh):
#
#     assert mesh.dimension == 2
#
#     a, b = line
#     tangent_dir = (b - a) / np.linalg.norm(b - a)
#     normal_dir = tangent_dir[np.array([1, 0, 2])]
#     normal_dir[0] *= -1.0
#
#     # cut conformity on c1 objects
#     gd2c1 = mesh.build_graph(2, 1)
#     raw_c1_cells = np.array(
#         [cell for cell in mesh.cells if cell.material_id == physical_tags["line"]]
#     )
#     raw_c1_cell_xcs = np.array([cell_centroid(cell, mesh) for cell in raw_c1_cells])
#
#     out, intx_q = points_line_intersection(raw_c1_cell_xcs, a, b)
#     c1_cells = raw_c1_cells[intx_q]
#
#     # operate only on graph
#     cell_id = mesh.max_cell_id() + 1
#     new_cells = []
#     old_edges = []
#     new_edges = []
#     for c1_cell in c1_cells:
#         c1_idx = c1_cell.index()
#         c0_cells_idx = list(gd2c1.predecessors(c1_idx))
#
#         for c0_idx in c0_cells_idx:
#             old_edges.append((c0_idx, c1_idx))
#
#             c1_cell_clone = c1_cell.clone()
#             c1_cell_clone.id = cell_id
#             c1_cell_clone.material_id = physical_tags["line_clones"]
#             new_cells.append(c1_cell_clone)
#
#             new_edges.append((c0_idx, c1_cell_clone.index()))
#
#             cell_id += 1
#
#     mesh.append_cells(np.array(new_cells))
#
#     # update cells in place
#     for i, graph_edge in enumerate(new_edges):
#         c0_idx, c1_idx = graph_edge
#         o_c0_idx, o_c1_idx = old_edges[i]
#         assert c0_idx == o_c0_idx
#         c0_cells = mesh.cells[c0_idx[1]]
#         idx = c0_cells.sub_cell_index(c1_idx[0], o_c1_idx[1])
#         assert idx is not None
#         c0_cells.sub_cells_ids[c1_idx[0]][idx] = c1_idx[1]
#
#     # cut conformity on c2 objects
#     gd2c2 = mesh.build_graph(2, 2)
#     gd1c1 = mesh.build_graph(1, 1)
#     raw_c2_cells_idx = np.unique(
#         np.concatenate([cell.sub_cells_ids[0] for cell in c1_cells])
#     )
#     c2_cells = np.array(
#         [
#             mesh.cells[idx]
#             for idx in raw_c2_cells_idx
#             if mesh.cells[idx].material_id != physical_tags["internal_bc"]
#         ]
#     )
#
#     if c2_cells.shape[0] == 0:  # no internal points to process
#         return
#
#     # operate only on graph
#     cell_id = mesh.max_cell_id() + 1
#     node_tag = mesh.max_node_tag() + 1
#     new_nodes = []
#     new_points = []
#     new_cells = []
#     old_edges = []
#     new_edges = []
#     for c2_cell in c2_cells:
#         c2_idx = c2_cell.index()
#         c0_cells_idx = np.array(list(gd2c2.predecessors(c2_idx)))
#         # c1_cells_idx = np.array(list(gd1c1.predecessors(c2_idx)))
#
#         c0_cells = [mesh.cells[c0_idx[1]] for c0_idx in c0_cells_idx]
#         c0_xcs = np.array([cell_centroid(cell, mesh) for cell in c0_cells])
#         c0_dirs = c0_xcs - mesh.points[c2_cell.node_tags]
#         positive_side = np.where(np.dot(c0_dirs, normal_dir) > 0, True, False)
#         negative_side = np.where(np.dot(c0_dirs, normal_dir) < 0, True, False)
#         # c1_cells = [mesh.cells[c1_idx[1]] for c1_idx in c1_cells_idx if mesh.cells[c1_idx[1]].material_id != physical_tags['line']]
#
#         # classify cells
#         for c0_idx in c0_cells_idx[positive_side]:
#             point_clone = mesh.points[c2_cell.node_tags].copy()
#             c2_cell_clone = c2_cell.clone()
#             c2_cell_clone.id = cell_id
#             c2_cell_clone.node_tags = np.array([node_tag])
#             if c2_cell.material_id == physical_tags["point"]:
#                 c2_cell_clone.material_id = physical_tags["point_clones"]
#             new_cells.append(c2_cell_clone)
#             new_points.append(point_clone)
#             cell_id += 1
#             node_tag += 1
#
#             new_nodes.append((c2_cell.node_tags[0], c2_cell_clone.node_tags[0]))
#             old_edges.append((c0_idx, c2_idx))
#             new_edges.append((c0_idx, c2_cell_clone.index()))
#
#         for c0_idx in c0_cells_idx[negative_side]:
#             point_clone = mesh.points[c2_cell.node_tags].copy()
#             c2_cell_clone = c2_cell.clone()
#             c2_cell_clone.id = cell_id
#             c2_cell_clone.node_tags = np.array([node_tag])
#             if c2_cell.material_id == physical_tags["point"]:
#                 c2_cell_clone.material_id = physical_tags["point_clones"]
#             new_cells.append(c2_cell_clone)
#             new_points.append(point_clone)
#             cell_id += 1
#             node_tag += 1
#
#             new_nodes.append((c2_cell.node_tags[0], c2_cell_clone.node_tags[0]))
#             old_edges.append((c0_idx, c2_idx))
#             new_edges.append((c0_idx, c2_cell_clone.index()))
#
#     mesh.append_cells(np.array(new_cells))
#     mesh.append_points(np.concatenate(new_points))
#
#     assert len(new_edges) == len(new_nodes)
#     # update cells in place
#     for i, graph_edge in enumerate(new_edges):
#         c0_idx, c2_idx = graph_edge
#         o_c0_idx, o_c2_idx = old_edges[i]
#         o_node_tag, n_node_tag = new_nodes[i]
#         assert np.all(c0_idx == o_c0_idx)
#         c0_cells = mesh.cells[c0_idx[1]]
#         for d in [2, 1]:
#             for sub_cell_id in c0_cells.sub_cells_ids[d]:
#                 sub_cell = mesh.cells[sub_cell_id]
#                 idx = sub_cell.sub_cell_index(c2_idx[0], o_c2_idx[1])
#                 if idx is None:
#                     continue
#                 sub_cell.sub_cells_ids[c2_idx[0]][idx] = c2_idx[1]
#                 if d == 1:
#                     sub_cell.set_sub_cells_ids(0, np.sort(sub_cell.sub_cells_ids[0]))
#                 idx = sub_cell.node_tag_index(o_node_tag)
#                 if idx is None:
#                     continue
#                 sub_cell.node_tags[idx] = n_node_tag


def __duplicate_cells(cell_id, cells, physical_tag_map={}):
    new_cells = []
    for cell in cells:
        dim = cell.dimension
        cell_clone = cell.clone()
        cell_clone.id = cell_id
        cell_clone.sub_cells_ids[dim] = np.array([cell_id])
        material_id = cell_clone.material_id
        cell_clone.material_id = physical_tag_map.get(material_id, material_id)
        new_cells.append(cell_clone)
        cell_id += 1
    return new_cells, cell_id


def duplicate_mesh_points_from_0d_cells(cells_0d, mesh, bump_args=None):
    node_tags = np.concatenate([cell.node_tags for cell in cells_0d])
    new_points = mesh.points[node_tags].copy()
    if bump_args is not None:
        a, b = bump_args["line"]
        normal = bump_args["normal"]
        side = bump_args["side"]
        scale = bump_args.get("scale", 0.1)
        idx = points_line_argsort(new_points, a, b)
        sv = np.linalg.norm(new_points[idx] - a, axis=1)
        nv = side * scale * (sv * (1.0 - sv) + 1.0e-1)
        new_points[idx] += (np.tile(normal, (nv.shape[0], 1)).T * nv).T

    node_tag = mesh.max_node_tag()
    for cell in cells_0d:
        cell.node_tags = np.array([node_tag])
        node_tag += 1
    mesh.append_points(new_points)
    assert node_tag == mesh.max_node_tag()


# def __update_cell(cell, sub_cells):
#     new_cells = []
#     for cell in cells:
#         cell_clone = cell.clone()
#         cell_clone.id = cell_id
#         material_id  = cell_clone.material_id
#         cell_clone.material_id = physical_tag_map.get(material_id, material_id)
#         new_cells.append(cell_clone)
#         cell_id += 1
#     return new_cells, cell_id


def replace_node_in_grahp(node_pair, graph):
    old_node, new_node = node_pair

    if not graph.has_node(old_node):
        return

    # Add the new node and connect it to the predecessors of the old node
    predecessors = list(graph.predecessors(old_node))

    old_edges = []
    new_edges = []
    for predecessor in predecessors:
        old_edges.append((predecessor, old_node))
        new_edges.append((predecessor, new_node))
    # Remove old and add new edges
    graph.remove_edges_from(old_edges)
    graph.add_edges_from(new_edges)
    graph.remove_node(old_node)


def replace_nodes_in_grahp(node_pairs, graph):
    [replace_node_in_grahp(node_pair, graph) for node_pair in node_pairs]


def update_cells_with_cell_pairs(cells, cell_pairs, mesh):
    cells_dims = np.array([cell.dimension for cell in cells])
    max_dim = np.max(cells_dims)
    min_dim = np.min(cells_dims)
    if max_dim != min_dim:
        raise ValueError("cells should contain cells with the same dimension.")

    patch_g = nx.DiGraph()
    cells_ids = np.array([cell.id for cell in cells])
    cell_idx_maps = []
    for d in range(0, max_dim):
        co_dim = max_dim - d
        local_graph = mesh.build_graph_from_cell_ids(cells_ids, max_dim, co_dim)
        patch_g.add_edges_from(local_graph.edges())
        pairs = [
            (o_cell.index(), t_cell.index()) for o_cell, t_cell in cell_pairs[co_dim]
        ]
        cell_idx_maps.append(dict(pairs))

    # update sub_cells
    for d in range(0, max_dim):
        for item in cell_idx_maps[d].items():
            old_cell_idx, new_cell_idx = item
            predecessors = list(patch_g.predecessors(old_cell_idx))
            for predecessor in predecessors:
                cell_dim, cell_id = predecessor
                cell = mesh.cells[cell_id]
                out = cell.sub_cell_index(old_cell_idx[0], old_cell_idx[1])
                if out is None:
                    continue
                cell.sub_cells_ids[new_cell_idx[0]][out] = new_cell_idx[1]
                # # ensure edge orientation
                # if cell.dimension == 1:
                #     cell.sub_cells_ids[new_cell_idx[0]] = np.sort(cell.sub_cells_ids[new_cell_idx[0]])

                # update node_tags
                if old_cell_idx[0] == 0 and new_cell_idx[0] == 0:
                    old_cell = mesh.cells[old_cell_idx[1]]
                    new_cell = mesh.cells[new_cell_idx[1]]
                    tag_pos = cell.node_tag_index(old_cell.node_tags)
                    assert tag_pos is not None
                    cell.node_tags[tag_pos] = new_cell.node_tags[0]


def cut_conformity_along_c1_line(
    line: np.array, physical_tags, mesh: Mesh, visual_frac_q=False
):

    assert mesh.dimension == 2

    a, b = line
    tangent_dir = (b - a) / np.linalg.norm(b - a)
    normal_dir = tangent_dir[np.array([1, 0, 2])]
    normal_dir[0] *= -1.0

    bump_args = {}
    bump_args["line"] = (a, b)
    bump_args["normal"] = normal_dir
    if not visual_frac_q:
        bump_args["scale"] = 0.0
    else:
        bump_args["scale"] = 0.05

    # cut conformity on c1 objects
    raw_c1_cells = np.array(
        [cell for cell in mesh.cells if cell.material_id == physical_tags["line"]]
    )
    raw_c1_cell_xcs = np.array([cell_centroid(cell, mesh) for cell in raw_c1_cells])

    out, intx_q = points_line_intersection(raw_c1_cell_xcs, a, b)
    c1_cells = raw_c1_cells[intx_q]

    # classify associated c0 cells
    g_d_0d = mesh.build_graph(mesh.dimension, mesh.dimension)
    g_d1_0d = mesh.build_graph(mesh.dimension - 1, mesh.dimension - 1)

    # collect c2 cells
    c2_cells_ids = np.unique(
        np.concatenate([cell.sub_cells_ids[0] for cell in c1_cells])
    )
    c2_cells = [
        mesh.cells[cell_id]
        for cell_id in c2_cells_ids
        if mesh.cells[cell_id].material_id != physical_tags["internal_bc"]
    ]

    c0_cells_data = [list(g_d_0d.predecessors(cell.index())) for cell in c2_cells]
    c0_cells_idx = np.array(list(itertools.chain(*c0_cells_data)))
    c0_cell_ids = np.unique([cell_idx[1] for cell_idx in c0_cells_idx])
    c0_cells = np.array([mesh.cells[c0_id] for c0_id in c0_cell_ids])
    c0_xcs = np.array([cell_centroid(cell, mesh) for cell in c0_cells])
    c0_dirs = c0_xcs - np.mean([a, b], axis=0)
    positive_side = np.where(np.dot(c0_dirs, normal_dir) > 0, True, False)
    negative_side = np.where(np.dot(c0_dirs, normal_dir) < 0, True, False)

    # duplicate c1 entities
    physical_tag_map = {
        physical_tags["line"]: physical_tags["line_clones"],
        physical_tags["point"]: physical_tags["point_clones"],
    }
    cell_id = mesh.max_cell_id()
    positive_c0_cells = c0_cells[positive_side]
    positive_c1_cells, cell_id = __duplicate_cells(cell_id, c1_cells, physical_tag_map)
    mesh.append_cells(np.array(positive_c1_cells))
    assert cell_id == mesh.max_cell_id()
    positive_c2_cells, cell_id = __duplicate_cells(cell_id, c2_cells, physical_tag_map)
    bump_args["side"] = +1.0
    duplicate_mesh_points_from_0d_cells(
        positive_c2_cells, mesh, bump_args=bump_args
    )  # update node_tags
    mesh.append_cells(np.array(positive_c2_cells))
    assert cell_id == mesh.max_cell_id()
    positive_cell_pairs = [
        None,
        zip(c1_cells, positive_c1_cells),
        zip(c2_cells, positive_c2_cells),
    ]

    negative_c0_cells = c0_cells[negative_side]
    negative_c1_cells, cell_id = __duplicate_cells(cell_id, c1_cells, physical_tag_map)
    mesh.append_cells(np.array(negative_c1_cells))
    assert cell_id == mesh.max_cell_id()
    negative_c2_cells, cell_id = __duplicate_cells(cell_id, c2_cells, physical_tag_map)
    bump_args["side"] = -1.0
    duplicate_mesh_points_from_0d_cells(
        negative_c2_cells, mesh, bump_args=bump_args
    )  # update node_tags
    mesh.append_cells(np.array(negative_c2_cells))
    assert cell_id == mesh.max_cell_id()
    negative_cell_pairs = [
        None,
        zip(c1_cells, negative_c1_cells),
        zip(c2_cells, negative_c2_cells),
    ]

    # update c0 cells
    update_cells_with_cell_pairs(positive_c0_cells, positive_cell_pairs, mesh)
    update_cells_with_cell_pairs(negative_c0_cells, negative_cell_pairs, mesh)

    # update c1 cells
    positive_cell_pairs = [None, zip(c2_cells, positive_c2_cells)]
    negative_cell_pairs = [None, zip(c2_cells, negative_c2_cells)]
    update_cells_with_cell_pairs(positive_c1_cells, positive_cell_pairs, mesh)
    update_cells_with_cell_pairs(negative_c1_cells, negative_cell_pairs, mesh)

    interface = {}
    interface["c1"] = [c1_cells, positive_c1_cells, negative_c1_cells]
    interface["c2"] = [c2_cells, positive_c2_cells, negative_c2_cells]

    # update c1 cells
    # # classify associated c1 cells
    # g_d1_0d = mesh.build_graph(mesh.dimension - 1, mesh.dimension - 1)
    c1_cells_data = [list(g_d1_0d.predecessors(cell.index())) for cell in c2_cells]
    c1_cells_idx = np.array(list(itertools.chain(*c1_cells_data)))
    c1_cell_ids = np.unique([cell_idx[1] for cell_idx in c1_cells_idx])
    c1_cells = np.array([mesh.cells[c1_id] for c1_id in c1_cell_ids])
    c1_xcs = np.array([cell_centroid(cell, mesh) for cell in c1_cells])
    c1_dirs = c1_xcs - np.mean([a, b], axis=0)
    positive_side = np.where(np.dot(c1_dirs, normal_dir) > 0, True, False)
    negative_side = np.where(np.dot(c1_dirs, normal_dir) < 0, True, False)
    positive_c1_cells = c1_cells[positive_side]
    negative_c1_cells = c1_cells[negative_side]

    positive_cell_pairs = [None, zip(c2_cells, positive_c2_cells)]
    negative_cell_pairs = [None, zip(c2_cells, negative_c2_cells)]
    update_cells_with_cell_pairs(positive_c1_cells, positive_cell_pairs, mesh)
    update_cells_with_cell_pairs(negative_c1_cells, negative_cell_pairs, mesh)

    return interface


def cut_conformity_along_c1_lines(
    lines: np.array, physical_tags, mesh: Mesh, visual_frac_q=False
):
    cut_conformity = partial(
        cut_conformity_along_c1_line,
        physical_tags=physical_tags,
        mesh=mesh,
        visual_frac_q=visual_frac_q,
    )
    interfaces = list(map(cut_conformity, lines))
    return interfaces
