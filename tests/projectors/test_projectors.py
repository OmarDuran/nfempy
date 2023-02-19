
import pytest
import numpy as np
from geometry.geometry_builder import GeometryBuilder
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh



import basix
from basix import ElementFamily, CellType, LagrangeVariant, LatticeType

import scipy.sparse as sp
from scipy.sparse import coo_matrix

import time

k_orders = [1,2,3,4,5]
functions = [lambda x, y, z: x + y,
lambda x, y, z: x * (1.0 - x) + y * (1.0 - y),
lambda x, y, z: (x**2) * (1.0 - x) + (y**2) * (1.0 - y),
lambda x, y, z: (x**3) * (1.0 - x) + (y**3) * (1.0 - y),
lambda x, y, z: (x**4) * (1.0 - x) + (y**4) * (1.0 - y)]

def generate_geometry_2d():
    box_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    g_builder = GeometryBuilder(dimension=2)
    g_builder.build_box_2D(box_points)
    return g_builder

def generate_conformal_mesh(h_cell):

    mesher = ConformalMesher(dimension=2)
    mesher.set_geometry_builder(generate_geometry_2d())
    mesher.set_points()
    mesher.generate(h_cell)
    mesher.write_mesh("gmesh.msh")
    return mesher

def generate_mesh(h_cell):
    conformal_mesh = generate_conformal_mesh(h_cell)
    gmesh = Mesh(dimension=2, file_name="gmesh.msh")
    gmesh.set_conformal_mesher(conformal_mesh)
    gmesh.build_conformal_mesh()
    gmesh.write_vtk()
    return gmesh

def permute_edges(element):

    indices = np.array([], dtype=int)
    rindices = np.array([], dtype=int)
    e_perms = np.array([1, 2, 0])
    e_rperms = np.array([2, 0, 1])
    for dim , entity_dof in enumerate(element.entity_dofs):
        if dim == 1:
            for i, chunk in enumerate(entity_dof):
                if i == 2:
                    chunk.reverse()
            entity_dof_r = [entity_dof[i] for i in e_rperms]
            entity_dof = [entity_dof[i] for i in e_perms]
            rindices = np.append(rindices, np.array(entity_dof_r, dtype=int))
            indices = np.append(indices,np.array(entity_dof, dtype=int))

        else:
            rindices = np.append(rindices, np.array(entity_dof, dtype=int))
            indices = np.append(indices, np.array(entity_dof, dtype=int))
    return (indices.ravel(),rindices.ravel())

def validate_orientation(gmesh, cell):
    connectiviy = np.array([[0, 1], [1, 2], [2, 0]])
    e_perms = np.array([1, 2, 0])
    orientation = [False,False,False]
    for i, con in enumerate(connectiviy):
        edge = cell.node_tags[con]
        v_edge = gmesh.cells[cell.cells_ids[1][i]].node_tags
        if np.any(edge == v_edge):
            orientation[i] = True
    orientation = [orientation[i] for i in e_perms]
    return orientation

@pytest.mark.parametrize("k_order", k_orders)
def test_h1_projector(k_order):
    
    h_cell = 0.5
    gmesh = generate_mesh(h_cell)
    fun = functions[k_order - 1]
    # Create conformity
    st = time.time()
    gd2c2 = gmesh.build_graph(2, 2)
    gd2c1 = gmesh.build_graph(2, 1)

    cells_ids_c2 = list(gd2c2.nodes())
    cells_ids_c1 = list(gd2c1.nodes())

    # Collecting ids
    vertices_ids = [id for id in cells_ids_c2 if gmesh.cells[id].dimension == 0]
    n_vertices = len(vertices_ids)
    edges_ids = [id for id in cells_ids_c1 if gmesh.cells[id].dimension == 1]
    n_edges = len(edges_ids)
    faces_ids = [id for id in cells_ids_c2 if gmesh.cells[id].dimension == 2]
    n_faces = len(faces_ids)

    conformity = "h-1"
    b_variant = LagrangeVariant.gll_centroid

    # H1 functionality

    # Conformity needs to be defined
    lagrange = basix.create_element(ElementFamily.P, CellType.triangle, k_order,
                                    b_variant)

    c_size = 0
    n_dof_g = 0
    cell_map = {}
    for cell in gmesh.cells:
        if cell.dimension != 2:
            continue
        n_dof = 0
        for n_entity_dofs in lagrange.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs)
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    row = np.zeros((c_size), dtype=np.int64)
    col = np.zeros((c_size), dtype=np.int64)
    data = np.zeros((c_size), dtype=np.float64)

    n_fields = 1
    entity_support = [n_vertices,n_edges,n_faces,0]
    entity_dofs = [0, 0, 0, 0]
    for dim, n_entity_dofs in enumerate(lagrange.num_entity_dofs):
        e_dofs = int(np.mean(n_entity_dofs))
        entity_dofs[dim] = e_dofs
        entity_support[dim] *= e_dofs

    global_indices = np.add.accumulate([0, entity_support[0], entity_support[1], entity_support[2]])
    # Computing cell mappings
    vertex_map = dict(zip(vertices_ids, np.split(np.array(range(global_indices[0],global_indices[1])),len(vertices_ids))))
    edge_map = dict(zip(edges_ids, np.split(np.array(range(global_indices[1],global_indices[2])),len(edges_ids))))
    face_map = dict(zip(faces_ids, np.split(np.array(range(global_indices[2],global_indices[3])),len(faces_ids))))

    lagrange.base_transformations()
    b_transformations = lagrange.base_transformations()
    e_transformations = lagrange.entity_transformations()
    print(lagrange.dof_transformations_are_identity)
    print(lagrange.dof_transformations_are_permutations)

    n_dof_g = sum(entity_support)
    rg = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    # Fixed parametric basis and data
    points, weights = basix.make_quadrature(basix.QuadratureType.gauss_jacobi,
                                            CellType.triangle, 2 * k_order + 2)
    lagrange = basix.create_element(ElementFamily.P, CellType.triangle, k_order,
                                    b_variant)
    phi_hat_tab = lagrange.tabulate(1, points)

    linear_base = basix.create_element(ElementFamily.P, CellType.triangle, 1,
                                       LagrangeVariant.equispaced)
    geo_phi_tab = linear_base.tabulate(1, points)

    # permute functions
    perms = permute_edges(lagrange)

 
    et = time.time()
    elapsed_time = et - st
    print('Preprocessing time:', elapsed_time, 'seconds')

    st = time.time()
    for cell in gmesh.cells:
        if cell.dimension != 2:
            continue

        # print(phi_tab)
        n_dof = phi_hat_tab.shape[2]
        js = (n_dof, n_dof)
        rs = (n_dof)
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # For a given cell compute geometrical information
        # Evaluate mappings
        # linear_base = basix.create_element(ElementFamily.P, CellType.triangle, 1,
        #                                    LagrangeVariant.equispaced)
        # g_phi_tab = linear_base.tabulate(1, points)
        cell_points = gmesh.points[cell.node_tags]

        # Compute geometrical transformations
        xa = []
        Ja = []
        detJa = []
        invJa = []

        for i, point in enumerate(points):

            # QR-decomposition is not unique
            # It's only unique up to the signs of the rows of R
            xmap = np.dot(geo_phi_tab[0, i, :, 0],cell_points)
            grad_xmap = np.dot(geo_phi_tab[[1, 2], i, :, 0],cell_points).T
            q_axes, r_jac = np.linalg.qr(grad_xmap)
            r_sign = np.diag(np.sign(np.diag(r_jac)), 0)
            q_axes = np.dot(q_axes, r_sign)
            r_jac = np.dot(r_sign, r_jac)

            det_g_jac = np.linalg.det(r_jac)
            if det_g_jac < 0.0:
                print('Negative det jac: ', det_g_jac)

            xa.append(xmap)
            Ja.append(r_jac)
            detJa.append(det_g_jac)
            invJa.append(np.linalg.inv(r_jac))

        xa = np.array(xa)
        Ja = np.array(Ja)
        detJa = np.array(detJa)
        invJa = np.array(invJa)

        # map functions
        phi_tab = lagrange.push_forward(phi_hat_tab[0], Ja, detJa, invJa)

        # triangle ref connectivity
        if not lagrange.dof_transformations_are_identity:
            oriented_q =  validate_orientation(gmesh,cell)
            for index, check in enumerate(oriented_q):
                if check:
                    continue
                transformation = lagrange.entity_transformations()["interval"][0]
                dofs = lagrange.entity_dofs[1][index]
                for point in range(phi_tab.shape[0]):
                    for dim in range(phi_tab.shape[2]):
                        phi_tab[point, dofs, dim] = np.dot(transformation,phi_tab[point, dofs, dim])


        # linear_base
        for i, omega in enumerate(weights):

            f_val = fun(xa[i,0],xa[i,1],xa[i,2])
            r_el = r_el + detJa[i] * omega * f_val * phi_tab[ i, :, 0]
            j_el = j_el + detJa[i] * omega * np.outer(phi_tab[i,:,0],phi_tab[i,:,0])

        # scattering dof
        dof_vertex_supports = list(gd2c2.successors(cell.id))
        dof_edge_supports = list(gd2c1.successors(cell.id))
        dest_vertex = np.array([vertex_map.get(dof_s) for dof_s in dof_vertex_supports],dtype=int).ravel()
        dest_edge = np.array([edge_map.get(dof_s) for dof_s in dof_edge_supports],dtype=int).ravel()
        dest_faces = np.array([face_map.get(cell.id)],dtype=int).ravel()
        dest = np.concatenate((dest_vertex,dest_edge,dest_faces))[perms[0]]
        # dest = dest_vertex
        c_sequ = cell_map[cell.id]

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

        ako = 0

    jg = coo_matrix((data, (row, col)), shape=(n_dof_g, n_dof_g)).tocsr()
    et = time.time()
    elapsed_time = et - st
    print('Assembly time:', elapsed_time, 'seconds')

    # solving ls
    st = time.time()
    alpha = sp.linalg.spsolve(jg, rg)
    et = time.time()
    elapsed_time = et - st
    print('Linear solver time:', elapsed_time, 'seconds')

    # Computing L2 error

    l2_error = 0.0
    cell_2d_ids = [cell.id for cell in gmesh.cells if cell.dimension == 2]
    for id in cell_2d_ids:
        cell = gmesh.cells[id]
        if cell.dimension != 2:
            continue

        # scattering dof
        dof_vertex_supports = list(gd2c2.successors(cell.id))
        dof_edge_supports = list(gd2c1.successors(cell.id))
        dest_vertex = np.array([vertex_map.get(dof_s) for dof_s in dof_vertex_supports],dtype=int).ravel()
        dest_edge = np.array([edge_map.get(dof_s) for dof_s in dof_edge_supports],dtype=int).ravel()
        dest_faces = np.array([face_map.get(cell.id)],dtype=int).ravel()
        dest = np.concatenate((dest_vertex,dest_edge,dest_faces))[perms[0]]
        alpha_l = alpha[dest]

        # Compute geometrical transformations
        xa = []
        Ja = []
        detJa = []
        invJa = []
        cell_points = gmesh.points[cell.node_tags]
        for i, point in enumerate(points):

            # QR-decomposition is not unique
            # It's only unique up to the signs of the rows of R
            xmap = np.dot(geo_phi_tab[0, i, :, 0], cell_points)
            grad_xmap = np.dot(geo_phi_tab[[1, 2], i, :, 0], cell_points).T
            q_axes, r_jac = np.linalg.qr(grad_xmap)
            r_sign = np.diag(np.sign(np.diag(r_jac)), 0)
            q_axes = np.dot(q_axes, r_sign)
            r_jac = np.dot(r_sign, r_jac)

            det_g_jac = np.linalg.det(r_jac)
            if det_g_jac < 0.0:
                print('Negative det jac: ', det_g_jac)

            xa.append(xmap)
            Ja.append(r_jac)
            detJa.append(det_g_jac)
            invJa.append(np.linalg.inv(r_jac))

        xa = np.array(xa)
        Ja = np.array(Ja)
        detJa = np.array(detJa)
        invJa = np.array(invJa)

        # map functions
        phi_tab = lagrange.push_forward(phi_hat_tab[0], Ja, detJa, invJa)

        # triangle ref connectivity
        if not lagrange.dof_transformations_are_identity:
            oriented_q =  validate_orientation(gmesh,cell)
            for index, check in enumerate(oriented_q):
                if check:
                    continue
                transformation = lagrange.entity_transformations()["interval"][0]
                dofs = lagrange.entity_dofs[1][index]
                for point in range(phi_tab.shape[0]):
                    for dim in range(phi_tab.shape[2]):
                        phi_tab[point, dofs, dim] = np.dot(transformation,phi_tab[point, dofs, dim])

        for i, pt in enumerate(points):
            p_e = fun(xa[i,0],xa[i,1],xa[i,2])
            p_h = np.dot(alpha_l, phi_tab[i, :, 0])
            l2_error += detJa[i] * weights[i] * (p_h-p_e) * (p_h-p_e)


    print("L2-error: ",np.sqrt(l2_error))

    
    l2_error_q = np.isclose(l2_error,0.0,rtol=1e-14)
    assert l2_error_q

