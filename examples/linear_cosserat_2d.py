import numpy as np

from numpy import linalg as la

from shapely.geometry import LineString
from numba import njit, prange

import geometry.fracture_network as fn
import networkx as nx

import matplotlib.pyplot as plt

from geometry.geometry_cell import GeometryCell
from geometry.geometry_builder import GeometryBuilder
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh
from topology.mesh_topology import MeshTopology
from basis.finite_element import FiniteElement
from spaces.dof_map import DoFMap
from spaces.discrete_field import DiscreteField

import basix
import functools
from functools import partial

# from itertools import permutations
from functools import reduce


import scipy.sparse as sp
from scipy.sparse import coo_matrix
import pypardiso as sp_solver

import matplotlib.colors as mcolors

# import matplotlib.pyplot as plt
import matplotlib.pyplot as plot
import meshio


import jax
import jax.numpy as jnp
import time
import sys


def polygon_polygon_intersection():

    fracture_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

    fracture_2 = np.array(
        [[0.5, 0.0, 0.5], [0.5, 0.0, -0.5], [0.5, 1.0, -0.5], [0.5, 1.0, 0.5]]
    )
    fracture_3 = np.array(
        [[0.0, 0.5, -0.5], [1.0, 0.5, -0.5], [1.0, 0.5, 0.5], [0.0, 0.5, 0.5]]
    )

    fracture_2 = np.array(
        [[0.6, 0.0, 0.5], [0.6, 0.0, -0.5], [0.6, 1.0, -0.5], [0.6, 1.0, 0.5]]
    )
    fracture_3 = np.array(
        [
            [0.25, 0.0, 0.5],
            [0.914463, 0.241845, -0.207107],
            [0.572443, 1.18154, -0.207107],
            [-0.0920201, 0.939693, 0.5],
        ]
    )

    fractures = [fracture_1, fracture_2, fracture_3]

    fracture_network = fn.FractureNetwork(dimension=3)
    fracture_network.render_fractures(fractures)
    fracture_network.intersect_2D_fractures(fractures, True)
    fracture_network.build_grahp()
    fracture_network.draw_grahp()
    ika = 0

def domain_with_fractures():
    # Higher dimension geometry
    s = 1.0
    box_points = s * np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    g_builder = GeometryBuilder(dimension=2)
    g_builder.build_box_2D(box_points)
    g_builder.build_grahp()

    # insert base fractures
    fracture_1 = np.array([[0.5, 0.25], [0.5, 0.75]])
    fracture_2 = np.array([[0.25, 0.5], [0.75, 0.5]])
    fracture_3 = np.array([[0.2, 0.35], [0.85, 0.35]])
    fracture_4 = np.array([[0.15, 0.15], [0.85, 0.85]])
    fracture_5 = np.array([[0.15, 0.85], [0.85, 0.15]])

    fractures = [fracture_1]

    fracture_network = fn.FractureNetwork(dimension=2, physical_tag_shift=10)
    fracture_network.intersect_1D_fractures(fractures, render_intersection_q=False)
    fracture_network.build_grahp(all_fixed_d_cells_q=True)
    # fracture_network.draw_grahp()

    mesher = ConformalMesher(dimension=2)
    mesher.set_geometry_builder(g_builder)
    mesher.set_fracture_network(fracture_network)
    mesher.set_points()
    mesher.generate(1.0)
    mesher.write_mesh("gmesh.msh")

    gmesh = Mesh(dimension=2, file_name="gmesh.msh")
    gmesh.set_conformal_mesher(mesher)
    gmesh.build_conformal_mesh()

    gd2c1 = gmesh.build_graph(2, 1)
    gd2c2 = gmesh.build_graph(2, 2)
    gd1c1 = gmesh.build_graph(1, 1)
    # gmesh.draw_graph(gd1c1)
    gmesh.cut_conformity_on_fractures()
    gmesh.write_vtk()
    cgd2c1 = gmesh.build_graph_on_materials(2, 1)
    cgd2c2 = gmesh.build_graph_on_materials(2, 2)
    cgd1c1 = gmesh.build_graph_on_materials(1, 1)
    # gmesh.draw_graph(gd1c1)

    check_q = gmesh.circulate_internal_bc()
    if check_q[0]:
        print("Internal bc is closed.")

    aka = 0


def matrix_plot(A):
    norm = mcolors.TwoSlopeNorm(vmin=-10.0, vcenter=0, vmax=10.0)
    plt.matshow(A.todense(), norm=norm, cmap="RdBu_r")
    plt.colorbar()
    plt.show()

def h1_vec_projector(gmesh):

    # FESpace: data
    # polynomial order
    n_components = 3
    dim = gmesh.dimension
    discontinuous = True
    k_order = 1
    family = "Lagrange"

    u_field = DiscreteField(dim,n_components,family,k_order,gmesh)
    u_field.make_discontinuous()
    u_field.build_dof_map()
    u_field.build_elements()

    #  n-components field
    # fun = lambda x, y, z: np.array([y, -x, -z])
    fun = lambda x, y, z: np.array([y * (1 - y), -x * (1 - x), -z * (1 - z)])
    # fun = lambda x, y, z: np.array([y * (1 - y) *y, -x * (1 - x) *x, -z * (1 - z)* z])
    # fun = lambda x, y, z: np.array([y * (1 - y) * y * y, -x * (1 - x) * x * x, -z*(1-z)*z*z])
    # fun = lambda x, y, z: np.array([-y/(x*x+y*y + 1 ), +x/(x*x+y*y + 1), z/(x*x+y*y + 1)])


    st = time.time()
    # Assembler
    # Triplets data
    c_size = 0
    n_dof_g = 0
    cell_map = {}
    for element in u_field.elements:
        cell = element.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs) * n_components
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    row = np.zeros((c_size), dtype=np.int64)
    col = np.zeros((c_size), dtype=np.int64)
    data = np.zeros((c_size), dtype=np.float64)

    n_dof_g = u_field.dof_map.dof_number()
    rg = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    et = time.time()
    elapsed_time = et - st
    print("Preprocessing II time:", elapsed_time, "seconds")

    st = time.time()
    def scatter_el_data(element, fun, u_field, cell_map, row, col, data):

        n_components = u_field.n_comp
        cell = element.cell
        points, weights = element.quadrature
        phi_tab = element.phi
        (x, jac, det_jac, inv_jac) = element.mapping

        # destination indexes
        dest = u_field.dof_map.destination_indices(cell.id)
        dest = np.array(np.split(dest,len(element.dof_ordering)))[element.dof_ordering].ravel()

        n_phi = element.phi.shape[1]
        n_dof = n_phi * n_components
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # Partial local vectorization
        f_val_star = fun(x[:, 0], x[:, 1], x[:, 2])
        phi_s_star = (det_jac * weights * phi_tab[:, :, 0].T)

        # local blocks
        indices = np.array(np.split(np.array(range(n_phi * n_components)), n_phi)).T
        jac_block = np.zeros((n_phi, n_phi))
        for i, omega in enumerate(weights):
            phi_star = phi_tab[i, :, 0]
            jac_block = jac_block + det_jac[i] * omega * np.outer(phi_star, phi_star)

        for c in range(n_components):
            b = c
            e = (c + 1) * n_phi * n_components + 1
            r_el[b:e:n_components] += phi_s_star @ f_val_star[c]
            j_el[b:e:n_components, b:e:n_components] += jac_block

        # scattering data
        c_sequ = cell_map[cell.id]

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

    [
        scatter_el_data(element, fun, u_field, cell_map, row, col, data)
        for element in u_field.elements
    ]

    jg = coo_matrix((data, (row, col)), shape=(n_dof_g, n_dof_g)).tocsr()
    et = time.time()
    elapsed_time = et - st
    print("Assembly time:", elapsed_time, "seconds")

    # solving ls
    st = time.time()
    # alpha = sp.linalg.spsolve(jg, rg)
    alpha = sp_solver.spsolve(jg, rg)
    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    # Computing L2 error
    def compute_l2_error(element, u_field):
        l2_error = 0.0
        n_components = u_field.n_comp
        cell = element.cell
        # scattering dof
        dest = u_field.dof_map.destination_indices(cell.id)
        dest = np.array(np.split(dest,len(element.dof_ordering)))[element.dof_ordering].ravel()
        alpha_l = alpha[dest]

        (x, jac, det_jac, inv_jac) = element.mapping
        points, weights = element.quadrature
        phi_tab = element.phi

        # vectorization
        n_phi = phi_tab.shape[1]
        p_e_s = fun(x[:, 0], x[:, 1], x[:, 2])
        alpha_star = np.array(np.split(alpha_l, n_phi))
        p_h_s = (phi_tab[:, :, 0] @ alpha_star).T
        l2_error = np.sum(det_jac * weights * (p_e_s - p_h_s) * (p_e_s - p_h_s))
        return l2_error

    st = time.time()
    error_vec = [compute_l2_error(element, u_field) for element in u_field.elements]
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")

    l2_error = functools.reduce(lambda x, y: x + y, error_vec)
    print("L2-error: ", np.sqrt(l2_error))
    # assert np.isclose(np.sqrt(l2_error), 0.0, atol=1.0e-14)

    # post-process solution
    st = time.time()
    cellid_to_element = dict(zip(u_field.element_ids, u_field.elements))
    # writing solution on mesh points
    vertices = u_field.mesh_topology.entities_by_dimension(0)
    fh_data = np.zeros((len(gmesh.points),n_components))
    fe_data = np.zeros((len(gmesh.points),n_components))
    cell_vertex_map = u_field.mesh_topology.entity_map_by_dimension(0)
    for id in vertices:
        if not cell_vertex_map.has_node(id):
            continue

        pr_ids = list(cell_vertex_map.predecessors(id))
        cell = gmesh.cells[pr_ids[0]]
        if cell.dimension != u_field.dimension:
            continue

        element = cellid_to_element[pr_ids[0]]

        # scattering dof
        dest = u_field.dof_map.destination_indices(cell.id)
        dest = np.array(np.split(dest,len(element.dof_ordering)))[element.dof_ordering].ravel()
        alpha_l = alpha[dest]

        par_points = basix.geometry(u_field.element_type)

        target_node_id = gmesh.cells[id].node_tags[0]
        par_point_id = np.array(
            [i for i, node_id in enumerate(cell.node_tags) if node_id == target_node_id]
        )

        points = gmesh.points[target_node_id]
        if u_field.dimension != 0:
            points = par_points[par_point_id]


        # evaluate mapping
        (x, jac, det_jac, inv_jac) = element.compute_mapping(points)
        phi_tab = element.evaluate_basis(points)
        n_phi = phi_tab.shape[1]
        f_e = fun(x[:, 0], x[:, 1], x[:, 2])
        alpha_star = np.array(np.split(alpha_l, n_phi))
        f_h = (phi_tab[:, :, 0] @ alpha_star).T
        fh_data[target_node_id] = f_h.ravel()
        fe_data[target_node_id] = f_e.ravel()

    mesh_points = gmesh.points
    con_d = np.array(
        [
            element.cell.node_tags
            for element in u_field.elements
        ]
    )
    meshio_cell_types = { 0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
    cells_dict = {meshio_cell_types[u_field.dimension]: con_d}
    p_data_dict = {"f_h": fh_data, "f_exact": fe_data}

    mesh = meshio.Mesh(
        points=mesh_points,
        cells=cells_dict,
        # Optionally provide extra data on points, cells, etc.
        point_data=p_data_dict,
    )
    mesh.write("h1_projector.vtk")
    et = time.time()
    elapsed_time = et - st
    print("Post-processing time:", elapsed_time, "seconds")


def matrix_plot(J, sparse_q=True):

    if sparse_q:
        plot.matshow(J.todense())
    else:
        plot.matshow(J)
    plot.colorbar(orientation="vertical")
    plot.set_cmap("seismic")
    plot.show()


def hdiv_projector(gmesh):

    # FESpace: data
    # polynomial order
    dim = gmesh.dimension
    conformity = "h-div"
    discontinuous = False
    k_order = 1
    family = "N2E"
    element_type = FiniteElement.type_by_dimension(dim)
    basis_family = FiniteElement.basis_family(family)
    basis_variant = FiniteElement.basis_variant()

    # vectorial
    # fun = lambda x, y, z: np.array([y, -x, -z])
    # fun = lambda x, y, z: np.array([y * (1 - y), -x * (1 - x), -z * (1 - z)])
    # fun = lambda x, y, z: np.array([y * (1 - y) *y, -x * (1 - x) *x, -z * (1 - z)* z])
    # fun = lambda x, y, z: np.array([y * (1 - y) * y * y, -x * (1 - x) * x * x, -z*(1-z)*z*z])
    fun = lambda x, y, z: np.array([-y/(x*x+y*y + 1 ), +x/(x*x+y*y + 1), z/(x*x+y*y + 1)])


    st = time.time()
    # Entities by codimension
    # https://defelement.com/ciarlet.html
    mesh_topology = MeshTopology(gmesh)
    cell_ids = mesh_topology.entities_by_codimension(0)

    et = time.time()
    elapsed_time = et - st
    print("Preprocessing I time:", elapsed_time, "seconds")

    st = time.time()
    elements = list(
        map(
            partial(FiniteElement, mesh=gmesh, k_order=k_order, family=family, discontinuous=discontinuous),
            cell_ids,
        )
    )
    et = time.time()
    elapsed_time = et - st
    n_d_cells = len(elements)
    print("Number of processed elements:", n_d_cells)
    print("Element construction time:", elapsed_time, "seconds")

    st = time.time()
    # Assembler
    # Triplets data
    c_size = 0
    n_dof_g = 0
    cell_map = {}
    for element in elements:
        cell = element.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs)
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    row = np.zeros((c_size), dtype=np.int64)
    col = np.zeros((c_size), dtype=np.int64)
    data = np.zeros((c_size), dtype=np.float64)

    # DoF map for a variable supported on the element type
    dof_map = DoFMap(mesh_topology,basis_family,element_type,k_order,basis_variant,discontinuous=discontinuous)
    dof_map.build_entity_maps()
    n_dof_g = dof_map.dof_number()

    rg = np.zeros(n_dof_g)
    print("n_dof: ", n_dof_g)

    et = time.time()
    elapsed_time = et - st
    print("Preprocessing II time:", elapsed_time, "seconds")

    st = time.time()

    def scatter_el_data(element, fun, dof_map, cell_map, row, col, data):

        cell = element.cell
        points, weights = element.quadrature
        phi_tab = element.phi
        (x, jac, det_jac, inv_jac) = element.mapping

        n_dof = element.phi.shape[1]
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # linear_base
        for i, omega in enumerate(weights):
            f_val = fun(x[i, 0], x[i, 1], x[i, 2])
            r_el = r_el + det_jac[i] * omega * phi_tab[i, :, :] @ f_val
            for d in range(3):
                j_el = j_el + det_jac[i] * omega * np.outer(
                    phi_tab[i, :, d], phi_tab[i, :, d]
                )

        # scattering dof
        dest = dof_map.destination_indices(cell.id)
        dest = dest[element.dof_ordering]

        c_sequ = cell_map[cell.id]

        # contribute rhs
        rg[dest] += r_el

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

    [
        scatter_el_data(element, fun, dof_map, cell_map, row, col, data)
        for element in elements
    ]

    jg = coo_matrix((data, (row, col)), shape=(n_dof_g, n_dof_g)).tocsr()
    et = time.time()
    elapsed_time = et - st
    print("Assembly time:", elapsed_time, "seconds")

    # solving ls
    st = time.time()
    alpha = sp.linalg.spsolve(jg, rg)
    et = time.time()
    elapsed_time = et - st
    print("Linear solver time:", elapsed_time, "seconds")

    # Computing L2 error
    def compute_l2_error(element, dof_map):
        l2_error = 0.0
        cell = element.cell
        # scattering dof
        dest = dof_map.destination_indices(cell.id)
        dest = dest[element.dof_ordering]
        alpha_l = alpha[dest]

        (x, jac, det_jac, inv_jac) = element.mapping
        points, weights = element.quadrature
        phi_tab = element.phi
        for i, pt in enumerate(points):
            u_e = fun(x[i, 0], x[i, 1], x[i, 2])
            u_h = np.dot(alpha_l, phi_tab[i, :, :])
            l2_error += det_jac[i] * weights[i] * np.dot((u_h - u_e), (u_h - u_e))

        return l2_error

    st = time.time()
    error_vec = [compute_l2_error(element, dof_map) for element in elements]
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")

    l2_error = functools.reduce(lambda x, y: x + y, error_vec)
    print("L2-error: ", np.sqrt(l2_error))
    # assert np.isclose(np.sqrt(l2_error), 0.0, atol=1.0e-14)

    # post-process solution
    # writing solution on mesh points

    st = time.time()
    cellid_to_element = dict(zip(cell_ids, elements))
    uh_data = np.zeros((len(gmesh.points), 3))
    ue_data = np.zeros((len(gmesh.points), 3))

    vertices = mesh_topology.entities_dimension(0)
    cell_vertex_map = mesh_topology.entity_map_by_dimension(0)
    for id in vertices:
        if not cell_vertex_map.has_node(id):
            continue

        pr_ids = list(cell_vertex_map.predecessors(id))
        cell = gmesh.cells[pr_ids[0]]
        if cell.dimension != gmesh.dimension:
            continue

        element = cellid_to_element[pr_ids[0]]

        # scattering dof
        dest = dof_map.destination_indices(cell.id)
        dest = dest[element.dof_ordering]
        alpha_l = alpha[dest]

        par_points = basix.geometry(element_type)

        target_node_id = gmesh.cells[id].node_tags[0]
        par_point_id = np.array(
            [i for i, node_id in enumerate(cell.node_tags) if node_id == target_node_id]
        )

        points = par_points[par_point_id]

        # evaluate mapping
        (x, jac, det_jac, inv_jac) = element.compute_mapping(points)
        phi_tab = element.evaluate_basis(points)
        u_e = fun(x[0, 0], x[0, 1], x[0, 2])
        u_h = np.dot(alpha_l, phi_tab[0, :, :])

        ue_data[target_node_id] = u_e
        uh_data[target_node_id] = u_h

    mesh_points = gmesh.points
    con_d = np.array(
        [
            cell.node_tags
            for cell in gmesh.cells
            if cell.dimension == gmesh.dimension and cell.id != None and cell.material_id != None
        ]
    )
    meshio_cell_types = { 0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
    cells_dict = {meshio_cell_types[gmesh.dimension]: con_d}
    u_data_dict = {"uh": uh_data, "ue": ue_data}

    mesh = meshio.Mesh(
        points=mesh_points,
        cells=cells_dict,
        # Optionally provide extra data on points, cells, etc.
        point_data=u_data_dict,
    )
    mesh.write("hdiv_projector.vtk")
    et = time.time()
    elapsed_time = et - st
    print("Post-processing time:", elapsed_time, "seconds")

def generate_mesh_1d():

    h_cell = 1.0 / (16.0)

    theta_x = 0.0 * (np.pi/180)
    theta_y = -45.0 * (np.pi/180)
    theta_z = -45.0 * (np.pi/180)
    rotation_x = np.array(
        [[1, 0, 0],[0, np.cos(theta_x), -np.sin(theta_x)],[0,np.sin(theta_x), np.cos(theta_x)]])
    rotation_y = np.array(
        [[np.cos(theta_y), 0 , -np.sin(theta_y)], [0, 1, 0],[np.sin(theta_y), 0, np.cos(theta_y)]])
    rotation_z = np.array([[np.cos(theta_z),-np.sin(theta_z),0],[np.sin(theta_z),np.cos(theta_z),0],[0,0,1]])

    # higher dimension domain geometry
    s = 1.0

    box_points = s * np.array([[0, 0, 0], [1, 0, 0]])
    box_points = box_points @ rotation_x @ rotation_y @ rotation_z
    g_builder = GeometryBuilder(dimension=1)
    g_builder.build_box_1D(box_points)
    g_builder.build_grahp()

    mesher = ConformalMesher(dimension=1)
    mesher.set_geometry_builder(g_builder)
    mesher.set_points()
    mesher.generate(h_cell)
    mesher.write_mesh("gmesh.msh")

    gmesh = Mesh(dimension=1, file_name="gmesh.msh")
    gmesh.set_conformal_mesher(mesher)
    gmesh.build_conformal_mesh_II()

    # gmesh.write_data()
    gmesh.write_vtk()
    print("h-size: ", h_cell)


    return gmesh

def generate_mesh_2d():

    h_cell = 1.0 / (8.0)
    # higher dimension domain geometry
    s = 1.0

    theta_x = 0.0 * (np.pi/180)
    theta_y = 0.0 * (np.pi/180)
    theta_z = 0.0 * (np.pi/180)
    rotation_x = np.array(
        [[1, 0, 0],[0, np.cos(theta_x), -np.sin(theta_x)],[0,np.sin(theta_x), np.cos(theta_x)]])
    rotation_y = np.array(
        [[np.cos(theta_y), 0 , -np.sin(theta_y)], [0, 1, 0],[np.sin(theta_y), 0, np.cos(theta_y)]])
    rotation_z = np.array([[np.cos(theta_z),-np.sin(theta_z),0],[np.sin(theta_z),np.cos(theta_z),0],[0,0,1]])

    box_points = s * np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    box_points = box_points @ rotation_x @ rotation_y @ rotation_z
    g_builder = GeometryBuilder(dimension=2)
    g_builder.build_box_2D(box_points)
    g_builder.build_grahp()

    gmesh = None
    fractures_q = False
    if fractures_q:
        # polygon_polygon_intersection()
        # h_cell = 1.0 / 4.0
        fracture_tags = [0, 1, 2, 3, 4, 5]
        fracture_1 = np.array([[0.5, 0.2], [0.5, 0.8]])
        fracture_1 = np.array([[0.5, 0.4], [0.5, 0.6]])
        fracture_2 = np.array([[0.25, 0.5], [0.75, 0.5]])
        fracture_3 = np.array([[0.2, 0.35], [0.85, 0.35]])
        fracture_4 = np.array([[0.15, 0.15], [0.85, 0.85]])
        fracture_5 = np.array([[0.15, 0.85], [0.85, 0.15]])
        fracture_6 = np.array([[0.22, 0.62], [0.92, 0.22]])
        disjoint_fractures = [
            fracture_1,
            fracture_2,
            fracture_3,
            fracture_4,
            fracture_5,
            fracture_6,
        ]

        mesher = ConformalMesher(dimension=2)
        mesher.set_geometry_builder(g_builder)
        fractures = []
        for tag in fracture_tags:
            fractures.append(disjoint_fractures[tag])
        fracture_network = fn.FractureNetwork(dimension=2, physical_tag_shift=10)
        fracture_network.intersect_1D_fractures(fractures, render_intersection_q=False)
        fracture_network.build_grahp(all_fixed_d_cells_q=True)
        # mesher.set_fracture_network(fracture_network)
        mesher.set_points()
        mesher.generate(h_cell)
        mesher.write_mesh("gmesh.msh")

        gmesh = Mesh(dimension=2, file_name="gmesh.msh")
        gmesh.set_conformal_mesher(mesher)
        gmesh.build_conformal_mesh_II()
        # map_fracs_edge = gmesh.cut_conformity_on_fractures_mds_ec()
        # factor = 0.025
        # gmesh.apply_visual_opening(map_fracs_edge, factor)

        gmesh.write_data()
        gmesh.write_vtk()
        # print("Skin boundary is closed Q:", gmesh.circulate_internal_bc())
        print("h-size: ", h_cell)
    else:
        # polygon_polygon_intersection()

        mesher = ConformalMesher(dimension=2)
        mesher.set_geometry_builder(g_builder)
        mesher.set_points()
        mesher.generate(h_cell)
        mesher.write_mesh("gmesh.msh")

        gmesh = Mesh(dimension=2, file_name="gmesh.msh")
        gmesh.set_conformal_mesher(mesher)
        gmesh.build_conformal_mesh_II()

        # gmesh.write_data()
        gmesh.write_vtk()
        print("h-size: ", h_cell)

    return gmesh

def generate_mesh_3d():

    h_cell = 1.0 / (32.0)

    theta_x = 0.0 * (np.pi/180)
    theta_y = 0.0 * (np.pi/180)
    theta_z = 0.0 * (np.pi/180)
    rotation_x = np.array(
        [[1, 0, 0],[0, np.cos(theta_x), -np.sin(theta_x)],[0,np.sin(theta_x), np.cos(theta_x)]])
    rotation_y = np.array(
        [[np.cos(theta_y), 0 , -np.sin(theta_y)], [0, 1, 0],[np.sin(theta_y), 0, np.cos(theta_y)]])
    rotation_z = np.array([[np.cos(theta_z),-np.sin(theta_z),0],[np.sin(theta_z),np.cos(theta_z),0],[0,0,1]])
    # higher dimension domain geometry
    s = 1.0

    box_points = s * np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )
    box_points = box_points @ rotation_x @ rotation_y @ rotation_z

    g_builder = GeometryBuilder(dimension=3)
    g_builder.build_box(box_points)
    g_builder.build_grahp()

    mesher = ConformalMesher(dimension=3)
    mesher.set_geometry_builder(g_builder)
    mesher.set_points()
    mesher.generate(h_cell)
    mesher.write_mesh("gmesh.msh")

    gmesh = Mesh(dimension=3, file_name="gmesh.msh")
    gmesh.set_conformal_mesher(mesher)
    gmesh.build_conformal_mesh_II()

    # gmesh.write_data()
    gmesh.write_vtk()
    print("h-size: ", h_cell)


    return gmesh

def main():

    gmesh_3d = generate_mesh_3d()
    # gmesh_2d = generate_mesh_2d()
    # gmesh_1d = generate_mesh_1d()

    # # pojectors

    h1_vec_projector(gmesh_3d)
    # hdiv_projector(gmesh_3d)


if __name__ == "__main__":
    main()
