import numpy as np
from functools import partial
from mesh.mesh import Mesh
from mesh.mesh_metrics import cell_centroid
from geometry.operations.point_geometry_operations import points_line_intersection

def cut_conformity_along_c1_line(line: np.array, physical_tags, mesh:Mesh):

    assert mesh.dimension == 2

    a, b = line
    tangent_dir = (b - a) / np.linalg.norm(b-a)
    normal_dir = tangent_dir[np.array([1,0,2])]
    normal_dir[0] *= -1.0

    # cut conformity on c1 objects
    gd2c1 = mesh.build_graph(2, 1)
    raw_c1_cells = np.array([cell for cell in mesh.cells if cell.material_id == physical_tags['line']])
    raw_c1_cell_xcs = np.array([cell_centroid(cell, mesh) for cell in raw_c1_cells])

    out, intx_q = points_line_intersection(raw_c1_cell_xcs, a, b)
    c1_cells = raw_c1_cells[intx_q]

    # operate only on graph
    cell_id = mesh.max_cell_id() + 1
    new_cells = []
    old_edges = []
    new_edges = []
    for c1_cell in c1_cells:
        c1_idx = c1_cell.index()
        c0_cells_idx = list(gd2c1.predecessors(c1_idx))

        for c0_idx in c0_cells_idx:
            old_edges.append((c0_idx, c1_idx))

            c1_cell_clone = c1_cell.clone()
            c1_cell_clone.id = cell_id
            c1_cell_clone.material_id = physical_tags['line_clones']
            new_cells.append(c1_cell_clone)

            new_edges.append((c0_idx, c1_cell_clone.index()))

            cell_id += 1

    mesh.append_cells(np.array(new_cells))

    # update cells in place
    for i, graph_edge in enumerate(new_edges):
        c0_idx, c1_idx = graph_edge
        o_c0_idx, o_c1_idx = old_edges[i]
        assert c0_idx == o_c0_idx
        c0_cells = mesh.cells[c0_idx[1]]
        idx = c0_cells.sub_cell_index(c1_idx[0], o_c1_idx[1])
        assert idx is not None
        c0_cells.sub_cells_ids[c1_idx[0]][idx] = c1_idx[1]

    # cut conformity on c2 objects
    gd2c2 = mesh.build_graph(2, 2)
    gd1c1 = mesh.build_graph(1, 1)
    raw_c2_cells_idx = np.unique(np.concatenate([cell.sub_cells_ids[0] for cell in c1_cells]))
    c2_cells = np.array([mesh.cells[idx] for idx in raw_c2_cells_idx if mesh.cells[idx].material_id != physical_tags['internal_bc']])

    if c2_cells.shape[0] == 0: # no internal points to process
        return

    # operate only on graph
    cell_id = mesh.max_cell_id() + 1
    node_tag = mesh.max_node_tag() + 1
    new_nodes = []
    new_points = []
    new_cells = []
    old_edges = []
    new_edges = []
    for c2_cell in c2_cells:
        c2_idx = c2_cell.index()
        c0_cells_idx = np.array(list(gd2c2.predecessors(c2_idx)))
        # c1_cells_idx = np.array(list(gd1c1.predecessors(c2_idx)))

        c0_cells = [mesh.cells[c0_idx[1]] for c0_idx in c0_cells_idx]
        c0_xcs = np.array([cell_centroid(cell, mesh) for cell in c0_cells])
        c0_dirs = c0_xcs - mesh.points[c2_cell.node_tags]
        positive_side = np.where(np.dot(c0_dirs, normal_dir) > 0, True, False)
        negative_side = np.where(np.dot(c0_dirs, normal_dir) < 0, True, False)
        # c1_cells = [mesh.cells[c1_idx[1]] for c1_idx in c1_cells_idx if mesh.cells[c1_idx[1]].material_id != physical_tags['line']]

        # classify cells
        for c0_idx in c0_cells_idx[positive_side]:
            point_clone = mesh.points[c2_cell.node_tags].copy()
            c2_cell_clone = c2_cell.clone()
            c2_cell_clone.id = cell_id
            c2_cell_clone.node_tags = np.array([node_tag])
            if c2_cell.material_id == physical_tags['point']:
                c2_cell_clone.material_id = physical_tags['point_clones']
            new_cells.append(c2_cell_clone)
            new_points.append(point_clone)
            cell_id += 1
            node_tag += 1

            new_nodes.append((c2_cell.node_tags[0], c2_cell_clone.node_tags[0]))
            old_edges.append((c0_idx, c2_idx))
            new_edges.append((c0_idx, c2_cell_clone.index()))


        for c0_idx in c0_cells_idx[negative_side]:
            point_clone = mesh.points[c2_cell.node_tags].copy()
            c2_cell_clone = c2_cell.clone()
            c2_cell_clone.id = cell_id
            c2_cell_clone.node_tags = np.array([node_tag])
            if c2_cell.material_id == physical_tags['point']:
                c2_cell_clone.material_id = physical_tags['point_clones']
            new_cells.append(c2_cell_clone)
            new_points.append(point_clone)
            cell_id += 1
            node_tag += 1

            new_nodes.append((c2_cell.node_tags[0], c2_cell_clone.node_tags[0]))
            old_edges.append((c0_idx, c2_idx))
            new_edges.append((c0_idx, c2_cell_clone.index()))

    mesh.append_cells(np.array(new_cells))
    mesh.append_points(np.concatenate(new_points))

    assert len(new_edges) == len(new_nodes)
    # update cells in place
    for i, graph_edge in enumerate(new_edges):
        c0_idx, c2_idx = graph_edge
        o_c0_idx, o_c2_idx = old_edges[i]
        o_node_tag, n_node_tag = new_nodes[i]
        assert np.all(c0_idx == o_c0_idx)
        c0_cells = mesh.cells[c0_idx[1]]
        for d in [2, 1]:
            for sub_cell_id in c0_cells.sub_cells_ids[d]:
                sub_cell = mesh.cells[sub_cell_id]
                idx = sub_cell.sub_cell_index(c2_idx[0], o_c2_idx[1])
                if idx is None:
                    continue
                sub_cell.sub_cells_ids[c2_idx[0]][idx] = c2_idx[1]

                idx = sub_cell.node_tag_index(o_node_tag)
                if idx is None:
                    continue
                sub_cell.node_tags[idx] = n_node_tag

def cut_conformity_along_c1_lines(lines: np.array, physical_tags, mesh:Mesh):
    cut_conformity = partial(cut_conformity_along_c1_line, physical_tags=physical_tags,mesh=mesh)
    list(map(cut_conformity,lines))