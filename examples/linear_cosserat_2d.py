import numpy as np

from numpy import linalg as la

from shapely.geometry import LineString

import geometry.fracture_network as fn
import networkx as nx

import matplotlib.pyplot as plt

from geometry.geometry_cell import GeometryCell
from geometry.geometry_builder import GeometryBuilder
from geometry.mapping import store_mapping
from geometry.mapping import evaluate_linear_shapes
from geometry.mapping import evaluate_mapping

from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh
from topology.mesh_topology import MeshTopology
from basis.finite_element import FiniteElement
from basis.element_data import ElementData

from spaces.dof_map import DoFMap
from spaces.discrete_field import DiscreteField

import basix
import functools
from functools import partial
import copy
# from itertools import permutations
from functools import reduce


import scipy.sparse as sp
from scipy.sparse import coo_matrix
import pypardiso as sp_solver

import matplotlib.colors as mcolors

# import matplotlib.pyplot as plt
import matplotlib.pyplot as plot
import meshio

import time
import sys
import csv
import marshal

from geometry.vertex import Vertex
from geometry.edge import Edge
from geometry.domain_market import build_box_1D, build_box_2D, build_box_3D
from geometry.domain_market import build_box_2D_with_lines
from geometry.domain_market import read_fractures_file
from geometry.domain_market import build_disjoint_lines
from geometry.shape_manipulation import ShapeManipulation
from geometry.domain import Domain

def polygon_polygon_intersection():

    def read_fractures_file(n_points):
        file_name = "fracture_files/setting_3d_0.csv"
        fractures = np.empty((0, n_points, 3), float)
        with open(file_name, 'r') as file:
            loaded = csv.reader(file)
            for line in loaded:
                frac = [float(val) for val in line]
                fractures = np.append(fractures, np.array([np.split(np.array(frac),n_points)]), axis=0)
        return fractures

    # fracture_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    #
    # fracture_2 = np.array(
    #     [[0.5, 0.0, 0.5], [0.5, 0.0, -0.5], [0.5, 1.0, -0.5], [0.5, 1.0, 0.5]]
    # )
    # fracture_3 = np.array(
    #     [[0.0, 0.5, -0.5], [1.0, 0.5, -0.5], [1.0, 0.5, 0.5], [0.0, 0.5, 0.5]]
    # )
    #
    # fracture_2 = np.array(
    #     [[0.6, 0.0, 0.5], [0.6, 0.0, -0.5], [0.6, 1.0, -0.5], [0.6, 1.0, 0.5]]
    # )
    fracture_3 = np.array(
        [
            [0.25, 0.0, 0.5],
            [0.914463, 0.241845, -0.207107],
            [0.572443, 1.18154, -0.207107],
            [-0.0920201, 0.939693, 0.5],
        ]
    )
    # fractures = [fracture_1, fracture_2, fracture_3]

    fractures = list(read_fractures_file(4))
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

def h1_gen_projector(gmesh):

    # FESpace: data
    # polynomial order
    n_components = 3
    dim = gmesh.dimension
    discontinuous = True
    k_order = 3
    family = "Lagrange"

    u_field = DiscreteField(dim,n_components,family,k_order,gmesh)
    # u_field.make_discontinuous()
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
        (x, jac, det_jac, inv_jac, _) = element.mapping

        # destination indexes
        dest = u_field.dof_map.destination_indices(cell.id)

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
        alpha_l = alpha[dest]

        (x, jac, det_jac, inv_jac, _) = element.mapping
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
        (x, jac, det_jac, inv_jac, _) = element.compute_mapping(points)
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
    mesh.write("h1_gen_projector.vtk")
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


def hdiv_gen_projector(gmesh):

    # FESpace: data
    # polynomial order
    n_components = 1
    dim = gmesh.dimension
    discontinuous = True
    k_order = 2
    family = "BDM"

    u_field = DiscreteField(dim-1,n_components,family,k_order,gmesh)
    # u_field.make_discontinuous()
    u_field.build_dof_map()
    u_field.build_elements()

    # n-components tensor field
    # vectorization with numpy should be performed with care
    # this lambda x, y, z: np.array([[0.5, -0.5, -0.5]]) is not generating data for all
    # integration points
    # fun = lambda x, y, z: np.array([[0.5+0.0*y, -0.5+0.0*x, 0.5+0*z]])
    fun = lambda x, y, z: np.array([[y, -x, z]])
    # fun = lambda x, y, z: np.array([[y, -x, -z],[y, -x, -z],[y, -x, -z]])
    # fun = lambda x, y, z: np.array([[y * (1 - y), -x * (1 - x), -z * (1 - z)]])
    # fun = lambda x, y, z: np.array([[y * (1 - y), -x * (1 - x), -z * (1 - z)],
    #                                 [y * (1 - y), -x * (1 - x), -z * (1 - z)],
    #                                 [y * (1 - y), -x * (1 - x), -z * (1 - z)]])
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
        (x, jac, det_jac, inv_jac, axes) = element.mapping

        # destination indexes
        dest = u_field.dof_map.destination_indices(cell.id)

        n_phi = element.phi.shape[1]
        n_dof = n_phi * n_components
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        f_val_star = fun(x[:, 0], x[:, 1], x[:, 2])

        # local blocks
        jac_block = np.zeros((n_phi, n_phi))
        for i, omega in enumerate(weights):
            for d in range(3):
                phi_star = phi_tab[i, :, d]
                jac_block = jac_block + det_jac[i] * omega * np.outer(phi_star, phi_star)

        for c in range(n_components):
            b = c
            e = (c + 1) * n_phi * n_components + 1
            res_block = np.zeros(n_phi)
            for i, omega in enumerate(weights):
                f_val = fun(x[i, 0], x[i, 1], x[i, 2])[c]
                res_block = res_block + det_jac[i] * omega * phi_tab[i, :, :] @ f_val
            r_el[b:e:n_components] += res_block
            j_el[b:e:n_components, b:e:n_components] += jac_block

        # # linear_base
        # r_el_c = np.zeros(n_phi)
        # for i, omega in enumerate(weights):
        #     f_val = fun(x[i, 0], x[i, 1], x[i, 2])
        #     r_el_c = r_el_c + det_jac[i] * omega * phi_tab[i, :, :] @ f_val[0]

        if True:
            alpha_l = np.linalg.solve(j_el,r_el)
            u_e_s = fun(x[:, 0], x[:, 1], x[:, 2])
            alpha_star = np.array(np.split(alpha_l, n_phi))
            u_h_s = np.array([(phi_tab[:, :, d] @ alpha_star).T for d in range(3)])
            u_h_s = np.moveaxis(u_h_s, 0, 1)
            l2_error = np.sum(det_jac * weights * (u_e_s - u_h_s) * (u_e_s - u_h_s))
            aka = 0
        # scattering dof
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
        alpha_l = alpha[dest]

        (x, jac, det_jac, inv_jac, axes) = element.mapping
        points, weights = element.quadrature
        phi_tab = element.phi
        n_phi = phi_tab.shape[1]

        # vectorization
        # u_e_s = fun(x[:, 0], x[:, 1], x[:, 2])
        # alpha_star = np.array(np.split(alpha_l, n_phi))
        # u_h_s = np.array([(phi_tab[:, :, d] @ alpha_star).T for d in range(3)])
        # u_h_s = np.moveaxis(u_h_s,0,1)
        # l2_error = np.sum(det_jac * weights * (u_e_s - u_h_s) * (u_e_s - u_h_s))
        for c in range(n_components):
            for i, pt in enumerate(points):
                u_e = fun(x[i, 0], x[i, 1], x[i, 2])[c]
                u_e = (axes[0].T @ u_e) @ axes[0].T
                u_h = np.dot(alpha_l, phi_tab[i, :, :])
                l2_error += (
                        det_jac[i] * weights[i] * np.dot((u_h - u_e), (u_h - u_e))
                )
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
    # writing solution on mesh points

    st = time.time()
    cellid_to_element = dict(zip(u_field.element_ids, u_field.elements))
    uh_data = np.zeros((len(gmesh.points), 3, n_components))
    ue_data = np.zeros((len(gmesh.points), 3, n_components))

    vertices = u_field.mesh_topology.entities_by_dimension(0)
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
        (x, jac, det_jac, inv_jac, axes) = element.compute_mapping(points)
        phi_tab = element.evaluate_basis(points)
        n_phi = phi_tab.shape[1]
        u_e = fun(x[0, 0], x[0, 1], x[0, 2])[0]
        # alpha_star = np.array(np.split(alpha_l, n_phi))
        # u_h = np.array([(phi_tab[:, :, d] @ alpha_star).T for d in range(3)])
        # u_h = np.moveaxis(u_h,0,1)

        u_e = (axes[0].T @ u_e) @ axes[0].T
        u_h = np.dot(alpha_l, phi_tab[0, :, :])

        ue_data[target_node_id] = np.array([u_e]).T
        uh_data[target_node_id] = np.array([u_h]).T

    mesh_points = gmesh.points
    con_d = np.array(
        [
            element.cell.node_tags
            for element in u_field.elements
        ]
    )
    meshio_cell_types = { 0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
    cells_dict = {meshio_cell_types[u_field.dimension]: con_d}
    tensor_uh_data = [uh_data[:, :, i] for i in range(n_components)]
    tensor_uh_names = ["uh_" + str(i) for i in range(n_components)]
    tensor_ue_data = [ue_data[:, :, i] for i in range(n_components)]
    tensor_ue_names = ["ue_" + str(i) for i in range(n_components)]
    u_data_dict = dict(zip(tensor_uh_names + tensor_ue_names, tensor_uh_data + tensor_ue_data))

    mesh = meshio.Mesh(
        points=mesh_points,
        cells=cells_dict,
        # Optionally provide extra data on points, cells, etc.
        point_data=u_data_dict,
    )
    mesh.write("hdiv_gen_projector.vtk")
    et = time.time()
    elapsed_time = et - st
    print("Post-processing time:", elapsed_time, "seconds")

def generate_mesh_1d():

    h_cell = 1.0 / (1.0)

    theta_x = 0.0 * (np.pi/180)
    theta_y = 45.0 * (np.pi/180)
    theta_z = 30.0 * (np.pi/180)
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
    gmesh.build_conformal_mesh()

    # gmesh.write_data()
    gmesh.write_vtk()
    print("h-size: ", h_cell)


    return gmesh

def generate_mesh_2d():

    h_cell = 1.0 / (5.0)
    l = 0
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
    fractures_q = True
    if fractures_q:
        # polygon_polygon_intersection()
        # h_cell = 1.0 / 4.0
        fracture_tags = [0]
        fracture_1 = np.array([[0.5, 0.4, 0], [0.5, 0.6, 0]])
        fracture_1 = np.array([[0.5, 0.4, 0], [0.5, 0.6, 0]])
        fracture_2 = np.array([[0.4, 0.5, 0], [0.6, 0.5, 0]])
        fracture_3 = np.array([[0.2, 0.35, 0], [0.85, 0.35, 0]])
        fracture_4 = np.array([[0.15, 0.15, 0], [0.85, 0.85, 0]])
        fracture_5 = np.array([[0.15, 0.85, 0], [0.85, 0.15, 0]])
        fracture_6 = np.array([[0.22, 0.62, 0], [0.92, 0.22, 0]])
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

        mesher.set_fracture_network(fracture_network)
        mesher.set_points()
        mesher.generate(h_cell, l)
        mesher.write_mesh("gmesh.msh")

        gmesh = Mesh(dimension=2, file_name="gmesh.msh")
        gmesh.set_conformal_mesher(mesher)
        gmesh.build_conformal_mesh()
        map_fracs_edge = gmesh.cut_conformity_on_fractures_mds_ec()
        # factor = 0.05
        # gmesh.apply_visual_opening(map_fracs_edge, factor)

        # gmesh.write_data()
        gmesh.write_vtk()
        # print("Skin boundary is closed Q:", gmesh.circulate_internal_bc())
        print("h-size: ", h_cell)
        print("l-refi: ", l)

    else:
        # polygon_polygon_intersection()

        mesher = ConformalMesher(dimension=2)
        mesher.set_geometry_builder(g_builder)
        mesher.set_points()
        mesher.generate(h_cell, l)
        mesher.write_mesh("gmesh.msh")

        gmesh = Mesh(dimension=2, file_name="gmesh.msh")
        gmesh.set_conformal_mesher(mesher)
        gmesh.build_conformal_mesh()

        # gmesh.write_data()
        gmesh.write_vtk()
        print("h-size: ", h_cell)
        print("l-refi: ", l)

    return gmesh

def generate_mesh_3d():

    h_cell = 1.0 / (1.0)
    l = 3

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
    mesher.generate(h_cell,l)
    mesher.write_mesh("gmesh.msh")

    gmesh = Mesh(dimension=3, file_name="gmesh.msh")
    gmesh.set_conformal_mesher(mesher)
    gmesh.build_conformal_mesh()

    # gmesh.write_data()
    gmesh.write_vtk()
    print("h-size: ", h_cell)
    print("l-refi: ", l)


    return gmesh

def md_h1_laplace(gmesh):

    # FESpace: data
    n_components = 1
    dim = gmesh.dimension
    discontinuous = True
    k_order = 1
    family = "Lagrange"

    u_field = DiscreteField(dim, n_components, family, k_order, gmesh)
    u_field.build_structures([2, 3])
    # u_field.build_structures([2, 3, 4, 5])
    # u_field.build_structures([2, 3, 4, 5, 6, 7])

    st = time.time()
    # Assembler
    # Triplets data
    c_size = 0
    n_dof_g = 0
    cell_map = {}
    for element in u_field.elements:
        cell = element.data.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs) * n_components
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    for element in u_field.bc_elements:
        cell = element.data.cell
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
    print("Triplets creation time:", elapsed_time, "seconds")

    st = time.time()

    r_fun = lambda x, y, z: np.sqrt(x**2+y**2+z**2)
    f_exact = lambda x, y, z: np.array([r_fun(x,y,z) * (1 - r_fun(x,y,z))])
    f_rhs = lambda x, y, z: np.array([2 + 0.0*x])

    # fe_2d = lambda x, y, z: x*(1-x) * y*(1-y)
    # rhs_2d = lambda x, y, z: 2*(1 - x)*x + 2*(1 - y)*y
    # f_exact = lambda x, y, z: np.array([fe_2d(x,y,z),fe_2d(x,y,z),fe_2d(x,y,z)])
    # f_rhs = lambda x, y, z: np.array([rhs_2d(x,y,z),rhs_2d(x,y,z),rhs_2d(x,y,z)])

    # f_exact = lambda x, y, z: np.array([x*(1-x) * y*(1-y) * z*(1-z)])
    # f_rhs = lambda x, y, z: np.array([2*(1-x)*x*(1-y)*y+2*(1-x)*x*(1-z)*z+2*(1-y)*y*(1-z)*z])

    # fe_3d = lambda x, y, z: x*(1-x) * y*(1-y) * z*(1-z)
    # rhs_3d = lambda x, y, z: 2*(1-x)*x*(1-y)*y+2*(1-x)*x*(1-z)*z+2*(1-y)*y*(1-z)*z
    # # f_exact = lambda x, y, z: np.array([fe_3d(x,y,z),fe_3d(x,y,z),fe_3d(x,y,z)])
    # # f_rhs = lambda x, y, z: np.array([rhs_3d(x,y,z),rhs_3d(x,y,z),rhs_3d(x,y,z)])
    # f_exact = lambda x, y, z: np.array([fe_3d(x,y,z),fe_3d(x,y,z),fe_3d(x,y,z),fe_3d(x,y,z),fe_3d(x,y,z),fe_3d(x,y,z)])
    # f_rhs = lambda x, y, z: np.array([rhs_3d(x,y,z),rhs_3d(x,y,z),rhs_3d(x,y,z),rhs_3d(x,y,z),rhs_3d(x,y,z),rhs_3d(x,y,z)])

    def scatter_form_data(element, f_rhs, u_field, cell_map, row, col, data):

        n_components = u_field.n_comp
        el_data: ElementData = element.data

        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # destination indexes
        dest = u_field.dof_map.destination_indices(cell.id)

        n_phi = phi_tab.shape[2]
        n_dof = n_phi * n_components
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # Partial local vectorization
        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
        phi_s_star = (det_jac * weights * phi_tab[0, :, :, 0].T)

        # local blocks
        jac_block = np.zeros((n_phi, n_phi))
        for i, omega in enumerate(weights):
            grad_phi = inv_jac[i].T @ phi_tab[1:phi_tab.shape[0] + 1, i, :, 0]
            for d in range(3):
                jac_block += det_jac[i] * omega * np.outer(grad_phi[d], grad_phi[d])

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
        scatter_form_data(element, f_rhs, u_field, cell_map, row, col, data)
        for element in u_field.elements
    ]

    def scatter_bc_form_data(element, u_field, cell_map, row, col, data):

        n_components = u_field.n_comp
        el_data: ElementData = element.data

        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # find high-dimension neigh
        entity_map = u_field.dof_map.mesh_topology.entity_map_by_dimension(cell.dimension)
        neigh_list = list(entity_map.predecessors(cell.id))
        neigh_check_q = len(neigh_list) > 0
        assert neigh_check_q

        neigh_cell_id = neigh_list[0]
        neigh_cell_index = u_field.id_to_element[neigh_cell_id]
        neigh_cell = u_field.elements[neigh_cell_index].data.cell

        # destination indexes
        dest_neigh = u_field.dof_map.destination_indices(neigh_cell_id)
        dest = u_field.dof_map.bc_destination_indices(neigh_cell_id, cell.id)

        n_phi = phi_tab.shape[2]
        n_dof = n_phi * n_components
        js = (n_dof, n_dof)
        j_el = np.zeros(js)

        # local blocks
        beta = 1.0e12
        jac_block = np.zeros((n_phi, n_phi))
        for i, omega in enumerate(weights):
            phi = phi_tab[0, i, :, 0]
            jac_block += beta * det_jac[i] * omega * np.outer(phi, phi)

        for c in range(n_components):
            b = c
            e = (c + 1) * n_phi * n_components + 1
            j_el[b:e:n_components, b:e:n_components] += jac_block

        # scattering data
        c_sequ = cell_map[cell.id]

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

    [
        scatter_bc_form_data(element, u_field, cell_map, row, col, data)
        for element in u_field.bc_elements
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
        el_data = element.data
        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # scattering dof
        dest = u_field.dof_map.destination_indices(cell.id)
        alpha_l = alpha[dest]

        # vectorization
        n_phi = phi_tab.shape[2]
        p_e_s = f_exact(x[:, 0], x[:, 1], x[:, 2])
        alpha_star = np.array(np.split(alpha_l, n_phi))
        p_h_s = (phi_tab[0, :, :, 0] @ alpha_star).T
        l2_error = np.sum(det_jac * weights * (p_e_s - p_h_s) * (p_e_s - p_h_s))
        return l2_error

    st = time.time()
    error_vec = [compute_l2_error(element, u_field) for element in u_field.elements]
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    l2_error = functools.reduce(lambda x, y: x + y, error_vec)
    print("L2-error: ", np.sqrt(l2_error))

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
        phi_shapes = evaluate_linear_shapes(points, element.data)
        (x, jac, det_jac, inv_jac) = evaluate_mapping(cell.dimension, phi_shapes, gmesh.points[cell.node_tags])
        phi_tab = element.evaluate_basis(points)
        n_phi = phi_tab.shape[2]
        f_e = f_exact(x[:, 0], x[:, 1], x[:, 2])
        alpha_star = np.array(np.split(alpha_l, n_phi))
        f_h = (phi_tab[0, :, :, 0] @ alpha_star).T
        fh_data[target_node_id] = f_h.ravel()
        fe_data[target_node_id] = f_e.ravel()

    mesh_points = gmesh.points
    con_d = np.array(
        [
            element.data.cell.node_tags
            for element in u_field.elements
        ]
    )
    meshio_cell_types = { 0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
    cells_dict = {meshio_cell_types[u_field.dimension]: con_d}
    p_data_dict = {"u_h": fh_data, "u_exact": fe_data}

    mesh = meshio.Mesh(
        points=mesh_points,
        cells=cells_dict,
        # Optionally provide extra data on points, cells, etc.
        point_data=p_data_dict,
    )
    mesh.write("md_h1_laplace.vtk")
    et = time.time()
    elapsed_time = et - st
    print("Post-processing time:", elapsed_time, "seconds")


def md_h1_elasticity(gmesh):

    dim = gmesh.dimension
    # Material data

    m_lambda = 1.0
    m_mu = 1.0

    # FESpace: data
    n_components = 2
    if dim == 3:
        n_components = 3

    discontinuous = True
    k_order = 2
    family = "Lagrange"

    u_field = DiscreteField(dim, n_components, family, k_order, gmesh)
    # u_field.build_structures([2, 3])
    if dim == 2:
        u_field.build_structures([2, 3, 4, 5])
    elif dim == 3:
        u_field.build_structures([2, 3, 4, 5, 6, 7])

    st = time.time()
    # Assembler
    # Triplets data
    c_size = 0
    n_dof_g = 0
    cell_map = {}
    for element in u_field.elements:
        cell = element.data.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs) * n_components
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    for element in u_field.bc_elements:
        cell = element.data.cell
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
    print("Triplets creation time:", elapsed_time, "seconds")

    st = time.time()

    # facturated solution for
    f_exact = lambda x, y, z: np.array([np.sin(np.pi*x) * y*(1-y),np.sin(np.pi*y) * x*(1-x)])
    f_rhs_x = lambda x, y, z: -(np.pi*(-1 + 2*x)*(m_lambda + m_mu)*np.cos(np.pi*y)) + (-2*m_mu + (np.pi**2)*(-1 + y)*y*(m_lambda + 2*m_mu))*np.sin(np.pi*x)
    f_rhs_y = lambda x, y, z: -(np.pi*(-1 + 2*y)*(m_lambda + m_mu)*np.cos(np.pi*x)) + (-2*m_mu + (np.pi**2)*(-1 + x)*x*(m_lambda + 2*m_mu))*np.sin(np.pi*y)
    f_rhs = lambda x, y, z: np.array([-f_rhs_x(x,y,z), -f_rhs_y(x,y,z)])
    if dim == 3:
        f_exact = lambda x, y, z: np.array([np.sin(np.pi*x)*y*(1-y)*z*(1-z),np.sin(np.pi*y)*x*(1-x)*z*(1-z),np.sin(np.pi*z)*x*(1-x)*y*(1-y)])
        f_rhs_x = lambda x, y, z: -2*(np.pi**2)*(-1 + y)*y*(-1 + z)*z*m_mu*np.sin(np.pi*x) + (-1 + z)*z*m_mu*(np.pi*(-1 + 2*x)*np.cos(np.pi*y) + 2*np.sin(np.pi*x)) + (-1 + y)*y*m_mu*(np.pi*(-1 + 2*x)*np.cos(np.pi*z) + 2*np.sin(np.pi*x)) + np.pi*m_lambda*((-1 + 2*x)*(-1 + z)*z*np.cos(np.pi*y) + (-1 + y)*y*((-1 + 2*x)*np.cos(np.pi*z) - np.pi*(-1 + z)*z*np.sin(np.pi*x)))
        f_rhs_y = lambda x, y, z: -2*(np.pi**2)*(-1 + x)*x*(-1 + z)*z*m_mu*np.sin(np.pi*y) + (-1 + z)*z*m_mu*(np.pi*(-1 + 2*y)*np.cos(np.pi*x) + 2*np.sin(np.pi*y)) + (-1 + x)*x*m_mu*(np.pi*(-1 + 2*y)*np.cos(np.pi*z) + 2*np.sin(np.pi*y)) + np.pi*m_lambda*((-1 + 2*y)*(-1 + z)*z*np.cos(np.pi*x) + (-1 + x)*x*((-1 + 2*y)*np.cos(np.pi*z) - np.pi*(-1 + z)*z*np.sin(np.pi*y)))
        f_rhs_z = lambda x, y, z: -2*(np.pi**2)*(-1 + x)*x*(-1 + y)*y*m_mu*np.sin(np.pi*z) + (-1 + y)*y*m_mu*(np.pi*(-1 + 2*z)*np.cos(np.pi*x) + 2*np.sin(np.pi*z)) + (-1 + x)*x*m_mu*(np.pi*(-1 + 2*z)*np.cos(np.pi*y) + 2*np.sin(np.pi*z)) + np.pi*m_lambda*((-1 + y)*y*(-1 + 2*z)*np.cos(np.pi*x) + (-1 + x)*x*((-1 + 2*z)*np.cos(np.pi*y) - np.pi*(-1 + y)*y*np.sin(np.pi*z)))
        f_rhs = lambda x, y, z: np.array([-f_rhs_x(x,y,z), -f_rhs_y(x,y,z), -f_rhs_z(x,y,z)])

    def scatter_form_data(element, m_lambda, m_mu, f_rhs, u_field, cell_map, row, col, data):

        n_components = u_field.n_comp
        el_data: ElementData = element.data

        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # destination indexes
        dest = u_field.dof_map.destination_indices(cell.id)

        n_phi = phi_tab.shape[2]
        n_dof = n_phi * n_components
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # Partial local vectorization
        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
        phi_s_star = (det_jac * weights * phi_tab[0, :, :, 0].T)

        for c in range(n_components):
            b = c
            e = (c + 1) * n_phi * n_components + 1
            r_el[b:e:n_components] += phi_s_star @ f_val_star[c]

        # vectorized blocks
        phi_star_dirs = [[1,2],[0,2],[0,1]]
        for i, omega in enumerate(weights):
            grad_phi = inv_jac[i].T @ phi_tab[1:phi_tab.shape[0] + 1, i, :, 0]

            for i_d in range(n_components):
                for j_d in range(n_components):
                    phi_outer = np.outer(grad_phi[j_d], grad_phi[i_d])
                    stress_grad = m_mu * phi_outer
                    if i_d == j_d:
                        phi_outer_star = np.zeros((n_phi, n_phi))
                        for d in phi_star_dirs[i_d]:
                            phi_outer_star += np.outer(grad_phi[d], grad_phi[d])
                        stress_grad += (m_lambda + m_mu ) * phi_outer + m_mu * phi_outer_star
                    else:
                        stress_grad +=  m_lambda * np.outer(grad_phi[i_d], grad_phi[j_d])
                    j_el[i_d:n_dof+1:n_components,j_d:n_dof+1:n_components] += det_jac[i] * omega * stress_grad

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
        scatter_form_data(element, m_lambda, m_mu, f_rhs, u_field, cell_map, row, col, data)
        for element in u_field.elements
    ]

    def scatter_bc_form_data(element, u_field, cell_map, row, col, data):

        n_components = u_field.n_comp
        el_data: ElementData = element.data

        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # find high-dimension neigh
        entity_map = u_field.dof_map.mesh_topology.entity_map_by_dimension(cell.dimension)
        neigh_list = list(entity_map.predecessors(cell.id))
        neigh_check_q = len(neigh_list) > 0
        assert neigh_check_q

        neigh_cell_id = neigh_list[0]
        neigh_cell_index = u_field.id_to_element[neigh_cell_id]
        neigh_cell = u_field.elements[neigh_cell_index].data.cell

        # destination indexes
        dest_neigh = u_field.dof_map.destination_indices(neigh_cell_id)
        dest = u_field.dof_map.bc_destination_indices(neigh_cell_id, cell.id)

        n_phi = phi_tab.shape[2]
        n_dof = n_phi * n_components
        js = (n_dof, n_dof)
        j_el = np.zeros(js)

        # local blocks
        beta = 1.0e12
        jac_block = np.zeros((n_phi, n_phi))
        for i, omega in enumerate(weights):
            phi = phi_tab[0, i, :, 0]
            jac_block += beta * det_jac[i] * omega * np.outer(phi, phi)

        for c in range(n_components):
            b = c
            e = (c + 1) * n_phi * n_components + 1
            j_el[b:e:n_components, b:e:n_components] += jac_block

        # scattering data
        c_sequ = cell_map[cell.id]

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

    [
        scatter_bc_form_data(element, u_field, cell_map, row, col, data)
        for element in u_field.bc_elements
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
        el_data = element.data
        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # scattering dof
        dest = u_field.dof_map.destination_indices(cell.id)
        alpha_l = alpha[dest]

        # vectorization
        n_phi = phi_tab.shape[2]
        p_e_s = f_exact(x[:, 0], x[:, 1], x[:, 2])
        alpha_star = np.array(np.split(alpha_l, n_phi))
        p_h_s = (phi_tab[0, :, :, 0] @ alpha_star).T
        l2_error = np.sum(det_jac * weights * (p_e_s - p_h_s) * (p_e_s - p_h_s))
        return l2_error

    st = time.time()
    error_vec = [compute_l2_error(element, u_field) for element in u_field.elements]
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    l2_error = functools.reduce(lambda x, y: x + y, error_vec)
    print("L2-error: ", np.sqrt(l2_error))

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
        phi_shapes = evaluate_linear_shapes(points, element.data)
        (x, jac, det_jac, inv_jac) = evaluate_mapping(cell.dimension, phi_shapes, gmesh.points[cell.node_tags])
        phi_tab = element.evaluate_basis(points)
        n_phi = phi_tab.shape[2]
        f_e = f_exact(x[:, 0], x[:, 1], x[:, 2])
        alpha_star = np.array(np.split(alpha_l, n_phi))
        f_h = (phi_tab[0, :, :, 0] @ alpha_star).T
        fh_data[target_node_id] = f_h.ravel()
        fe_data[target_node_id] = f_e.ravel()

    mesh_points = gmesh.points
    con_d = np.array(
        [
            element.data.cell.node_tags
            for element in u_field.elements
        ]
    )
    meshio_cell_types = { 0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
    cells_dict = {meshio_cell_types[u_field.dimension]: con_d}
    p_data_dict = {"u_h": fh_data, "u_exact": fe_data}

    mesh = meshio.Mesh(
        points=mesh_points,
        cells=cells_dict,
        # Optionally provide extra data on points, cells, etc.
        point_data=p_data_dict,
    )
    mesh.write("md_h1_elasticity.vtk")
    et = time.time()
    elapsed_time = et - st
    print("Post-processing time:", elapsed_time, "seconds")

def md_h1_cosserat_elasticity(gmesh):

    dim = gmesh.dimension
    # Material data

    m_lambda = 1.0
    m_mu = 1.0
    m_kappa = 1.0
    m_gamma = 1.0

    # FESpace: data
    n_components = 3
    if dim == 3:
        n_components = 6

    discontinuous = True
    k_order = 3
    family = "Lagrange"

    u_field = DiscreteField(dim, n_components, family, k_order, gmesh)
    # u_field.build_structures([2, 3])
    if dim == 2:
        u_field.build_structures([2, 3, 4, 5])
    elif dim == 3:
        u_field.build_structures([2, 3, 4, 5, 6, 7])

    st = time.time()
    # Assembler
    # Triplets data
    c_size = 0
    n_dof_g = 0
    cell_map = {}
    for element in u_field.elements:
        cell = element.data.cell
        n_dof = 0
        for n_entity_dofs in element.basis_generator.num_entity_dofs:
            n_dof = n_dof + sum(n_entity_dofs) * n_components
        cell_map.__setitem__(cell.id, c_size)
        c_size = c_size + n_dof * n_dof

    for element in u_field.bc_elements:
        cell = element.data.cell
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
    print("Triplets creation time:", elapsed_time, "seconds")

    st = time.time()

    # smooth solution
    f_exact = lambda x, y, z: np.array([np.sin(np.pi*x) * y*(1-y),np.sin(np.pi*y) * x*(1-x), np.sin(np.pi*x)*np.sin(np.pi*y)])
    f_rhs_x = lambda x, y, z: -((2*m_kappa + 2*m_mu - np.pi**2*(-1 + y)*y*(m_lambda + 2*m_mu))*np.sin(np.pi*x)) + np.pi*np.cos(np.pi*y)*((-1 + 2*x)*(m_kappa - m_lambda - m_mu) + 2*m_kappa*np.sin(np.pi*x))
    f_rhs_y = lambda x, y, z: -((2*m_kappa + 2*m_mu - np.pi**2*(-1 + x)*x*(m_lambda + 2*m_mu))*np.sin(np.pi*y)) + np.pi*np.cos(np.pi*x)*((-1 + 2*y)*(m_kappa - m_lambda - m_mu) - 2*m_kappa*np.sin(np.pi*y))
    f_rhs_t = lambda x, y, z: -2*((-1 + 2*x)*m_kappa*np.sin(np.pi*y) + np.sin(np.pi*x)*(m_kappa - 2*y*m_kappa + ((np.pi**2)*m_gamma + 2*m_kappa)*np.sin(np.pi*y)))
    f_rhs = lambda x, y, z: np.array([-f_rhs_x(x,y,z), -f_rhs_y(x,y,z), -f_rhs_t(x,y,z)])
    if dim == 3:
        f_exact = lambda x, y, z: np.array([(1 - y)*y*(1 - z)*z*np.sin(np.pi*x),(1 - x)*x*(1 - z)*z*np.sin(np.pi*y),(1 - x)*x*(1 - y)*y*np.sin(np.pi*z),(1 - x)*x*np.sin(np.pi*y)*np.sin(np.pi*z),(1 - y)*y*np.sin(np.pi*x)*np.sin(np.pi*z),(1 - z)*z*np.sin(np.pi*x)*np.sin(np.pi*y)])
        f_rhs_x = lambda x, y, z: -2*(np.pi**2)*(1 - y)*y*(1 - z)*z*m_mu*np.sin(np.pi*x) + m_mu*(np.pi*(1 - x)*(1 - y)*y*np.cos(np.pi*z) - np.pi*x*(1 - y)*y*np.cos(np.pi*z) - 2*(1 - y)*y*np.sin(np.pi*x)) + m_mu*(np.pi*(1 - x)*(1 - z)*z*np.cos(np.pi*y) - np.pi*x*(1 - z)*z*np.cos(np.pi*y) - 2*(1 - z)*z*np.sin(np.pi*x)) + m_lambda*(np.pi*(1 - x)*(1 - z)*z*np.cos(np.pi*y) - np.pi*x*(1 - z)*z*np.cos(np.pi*y) + np.pi*(1 - x)*(1 - y)*y*np.cos(np.pi*z) - np.pi*x*(1 - y)*y*np.cos(np.pi*z) - (np.pi**2)*(1 - y)*y*(1 - z)*z*np.sin(np.pi*x)) + m_kappa*(-(np.pi*(1 - x)*(1 - z)*z*np.cos(np.pi*y)) + np.pi*x*(1 - z)*z*np.cos(np.pi*y) - 2*(1 - z)*z*np.sin(np.pi*x) + 2*np.pi*z*np.cos(np.pi*y)*np.sin(np.pi*x) - 2*np.pi*(z**2)*np.cos(np.pi*y)*np.sin(np.pi*x)) + m_kappa*(-(np.pi*(1 - x)*(1 - y)*y*np.cos(np.pi*z)) + np.pi*x*(1 - y)*y*np.cos(np.pi*z) - 2*(1 - y)*y*np.sin(np.pi*x) - 2*np.pi*y*np.cos(np.pi*z)*np.sin(np.pi*x) + 2*np.pi*(y**2)*np.cos(np.pi*z)*np.sin(np.pi*x))
        f_rhs_y = lambda x, y, z: -2*(np.pi**2)*(1 - x)*x*(1 - z)*z*m_mu*np.sin(np.pi*y) + m_mu*(np.pi*(1 - x)*x*(1 - y)*np.cos(np.pi*z) - np.pi*(1 - x)*x*y*np.cos(np.pi*z) - 2*(1 - x)*x*np.sin(np.pi*y)) + m_mu*(np.pi*(1 - y)*(1 - z)*z*np.cos(np.pi*x) - np.pi*y*(1 - z)*z*np.cos(np.pi*x) - 2*(1 - z)*z*np.sin(np.pi*y)) + m_lambda*(np.pi*(1 - y)*(1 - z)*z*np.cos(np.pi*x) - np.pi*y*(1 - z)*z*np.cos(np.pi*x) + np.pi*(1 - x)*x*(1 - y)*np.cos(np.pi*z) - np.pi*(1 - x)*x*y*np.cos(np.pi*z) - (np.pi**2)*(1 - x)*x*(1 - z)*z*np.sin(np.pi*y)) + m_kappa*(-(np.pi*(1 - y)*(1 - z)*z*np.cos(np.pi*x)) + np.pi*y*(1 - z)*z*np.cos(np.pi*x) - 2*(1 - z)*z*np.sin(np.pi*y) - 2*np.pi*z*np.cos(np.pi*x)*np.sin(np.pi*y) + 2*np.pi*(z**2)*np.cos(np.pi*x)*np.sin(np.pi*y)) + m_kappa*(-(np.pi*(1 - x)*x*(1 - y)*np.cos(np.pi*z)) + np.pi*(1 - x)*x*y*np.cos(np.pi*z) - 2*(1 - x)*x*np.sin(np.pi*y) + 2*np.pi*x*np.cos(np.pi*z)*np.sin(np.pi*y) - 2*np.pi*(x**2)*np.cos(np.pi*z)*np.sin(np.pi*y))
        f_rhs_z = lambda x, y, z: -2*(np.pi**2)*(1 - x)*x*(1 - y)*y*m_mu*np.sin(np.pi*z) + m_mu*(np.pi*(1 - x)*x*(1 - z)*np.cos(np.pi*y) - np.pi*(1 - x)*x*z*np.cos(np.pi*y) - 2*(1 - x)*x*np.sin(np.pi*z)) + m_mu*(np.pi*(1 - y)*y*(1 - z)*np.cos(np.pi*x) - np.pi*(1 - y)*y*z*np.cos(np.pi*x) - 2*(1 - y)*y*np.sin(np.pi*z)) + m_lambda*(np.pi*(1 - y)*y*(1 - z)*np.cos(np.pi*x) - np.pi*(1 - y)*y*z*np.cos(np.pi*x) + np.pi*(1 - x)*x*(1 - z)*np.cos(np.pi*y) - np.pi*(1 - x)*x*z*np.cos(np.pi*y) - (np.pi**2)*(1 - x)*x*(1 - y)*y*np.sin(np.pi*z)) + m_kappa*(-(np.pi*(1 - y)*y*(1 - z)*np.cos(np.pi*x)) + np.pi*(1 - y)*y*z*np.cos(np.pi*x) - 2*(1 - y)*y*np.sin(np.pi*z) + 2*np.pi*y*np.cos(np.pi*x)*np.sin(np.pi*z) - 2*np.pi*(y**2)*np.cos(np.pi*x)*np.sin(np.pi*z)) + m_kappa*(-(np.pi*(1 - x)*x*(1 - z)*np.cos(np.pi*y)) + np.pi*(1 - x)*x*z*np.cos(np.pi*y) - 2*(1 - x)*x*np.sin(np.pi*z) - 2*np.pi*x*np.cos(np.pi*y)*np.sin(np.pi*z) + 2*np.pi*(x**2)*np.cos(np.pi*y)*np.sin(np.pi*z))
        f_rhs_t_x = lambda x, y, z: -2*x*m_kappa*np.sin(np.pi*y) + 2*(x**2)*m_kappa*np.sin(np.pi*y) + 4*x*z*m_kappa*np.sin(np.pi*y) - 4*(x**2)*z*m_kappa*np.sin(np.pi*y) + 2*x*m_kappa*np.sin(np.pi*z) - 2*(x**2)*m_kappa*np.sin(np.pi*z) - 4*x*y*m_kappa*np.sin(np.pi*z) + 4*(x**2)*y*m_kappa*np.sin(np.pi*z) - 2*m_gamma*np.sin(np.pi*y)*np.sin(np.pi*z) - 2*(np.pi**2)*(1 - x)*x*m_gamma*np.sin(np.pi*y)*np.sin(np.pi*z) - 4*x*m_kappa*np.sin(np.pi*y)*np.sin(np.pi*z) + 4*(x**2)*m_kappa*np.sin(np.pi*y)*np.sin(np.pi*z)
        f_rhs_t_y = lambda x, y, z: 2*y*m_kappa*np.sin(np.pi*x) - 2*(y**2)*m_kappa*np.sin(np.pi*x) - 4*y*z*m_kappa*np.sin(np.pi*x) + 4*(y**2)*z*m_kappa*np.sin(np.pi*x) - 2*y*m_kappa*np.sin(np.pi*z) + 4*x*y*m_kappa*np.sin(np.pi*z) + 2*(y**2)*m_kappa*np.sin(np.pi*z) - 4*x*(y**2)*m_kappa*np.sin(np.pi*z) - 2*m_gamma*np.sin(np.pi*x)*np.sin(np.pi*z) - 2*(np.pi**2)*(1 - y)*y*m_gamma*np.sin(np.pi*x)*np.sin(np.pi*z) - 4*y*m_kappa*np.sin(np.pi*x)*np.sin(np.pi*z) + 4*(y**2)*m_kappa*np.sin(np.pi*x)*np.sin(np.pi*z)
        f_rhs_t_z = lambda x, y, z: -2*z*m_kappa*np.sin(np.pi*x) + 4*y*z*m_kappa*np.sin(np.pi*x) + 2*(z**2)*m_kappa*np.sin(np.pi*x) - 4*y*(z**2)*m_kappa*np.sin(np.pi*x) + 2*z*m_kappa*np.sin(np.pi*y) - 4*x*z*m_kappa*np.sin(np.pi*y) - 2*(z**2)*m_kappa*np.sin(np.pi*y) + 4*x*(z**2)*m_kappa*np.sin(np.pi*y) - 2*m_gamma*np.sin(np.pi*x)*np.sin(np.pi*y) - 2*(np.pi**2)*(1 - z)*z*m_gamma*np.sin(np.pi*x)*np.sin(np.pi*y) - 4*z*m_kappa*np.sin(np.pi*x)*np.sin(np.pi*y) + 4*(z**2)*m_kappa*np.sin(np.pi*x)*np.sin(np.pi*y)
        f_rhs = lambda x, y, z: np.array([-f_rhs_x(x,y,z), -f_rhs_y(x,y,z), -f_rhs_z(x,y,z), -f_rhs_t_x(x,y,z), -f_rhs_t_y(x,y,z), -f_rhs_t_z(x,y,z)])

    def scatter_form_data(element, m_lambda, m_mu, m_kappa, m_gamma, f_rhs, u_field, cell_map, row, col, data):

        n_components = u_field.n_comp
        el_data: ElementData = element.data

        n_comp_u = 2
        n_comp_t = 1
        if u_field.dimension == 3:
            n_comp_u = 3
            n_comp_t = 3

        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # destination indexes
        dest = u_field.dof_map.destination_indices(cell.id)

        n_phi = phi_tab.shape[2]
        n_dof = n_phi * n_components
        js = (n_dof, n_dof)
        rs = n_dof
        j_el = np.zeros(js)
        r_el = np.zeros(rs)

        # Partial local vectorization
        f_val_star = f_rhs(x[:, 0], x[:, 1], x[:, 2])
        phi_s_star = (det_jac * weights * phi_tab[0, :, :, 0].T)

        # rotation part
        rotation_block = np.zeros((n_phi, n_phi))
        assymetric_block = np.zeros((n_phi * n_comp_t, n_phi * n_comp_u))
        axial_pairs_idx = [[1, 0]]
        axial_dest_pairs_idx = [[0, 1]]
        if u_field.dimension== 3:
            axial_pairs_idx = [[2, 1],[0, 2],[1, 0]]
            axial_dest_pairs_idx = [[1, 2], [2, 0], [0, 1]]

        phi_star_dirs = [[1, 2], [0, 2], [0, 1]]
        for i, omega in enumerate(weights):
            grad_phi = inv_jac[i].T @ phi_tab[1:phi_tab.shape[0] + 1, i, :, 0]

            phi = phi_tab[0, i, :, 0]
            for i_c, pair in enumerate(axial_pairs_idx):
                adest = axial_dest_pairs_idx[i_c]
                sigma_rotation_0 = np.outer(phi, grad_phi[pair[0], :])
                sigma_rotation_1 = np.outer(phi, grad_phi[pair[1], :])
                assymetric_block[i_c*n_phi:n_phi+i_c*n_phi, adest[0]:n_dof + 1:n_comp_u] += det_jac[i] * omega * 2.0 * m_kappa * sigma_rotation_0
                assymetric_block[i_c*n_phi:n_phi+i_c*n_phi, adest[1]:n_dof + 1:n_comp_u] -= det_jac[i] * omega * 2.0 * m_kappa * sigma_rotation_1
            rotation_block += det_jac[i] * omega * 4.0 * m_kappa * np.outer(phi, phi)
            for d in range(3):
                rotation_block += det_jac[i] * omega * m_gamma * np.outer(grad_phi[d], grad_phi[d])

            for i_d in range(n_comp_u):
                for j_d in range(n_comp_u):
                    phi_outer = np.outer(grad_phi[i_d], grad_phi[j_d])
                    stress_grad = m_mu * phi_outer
                    if i_d == j_d:
                        phi_outer_star = np.zeros((n_phi, n_phi))
                        for d in phi_star_dirs[i_d]:
                            phi_outer_star += np.outer(grad_phi[d], grad_phi[d])
                        stress_grad += (m_lambda + m_mu) * phi_outer + (m_mu + m_kappa) * phi_outer_star
                    else:
                        stress_grad -= m_kappa * phi_outer
                        stress_grad += m_lambda * np.outer(grad_phi[j_d], grad_phi[i_d])
                    j_el[i_d:n_dof + 1:n_components, j_d:n_dof + 1:n_components] += \
                    det_jac[i] * omega * stress_grad

        for c in range(n_comp_t):
            b = c + n_comp_u
            j_el[b:n_dof + 1:n_components, b:n_dof + 1:n_components] += rotation_block

        for i_c, adest in enumerate(axial_dest_pairs_idx):
            j_el[n_comp_u+i_c:n_dof + 1:n_components,adest[0]:n_dof+1:n_components] += assymetric_block[i_c*n_phi:n_phi+i_c*n_phi,adest[0]:n_dof + 1:n_comp_u]
            j_el[adest[0]:n_dof+1:n_components,n_comp_u+i_c:n_dof + 1:n_components] += assymetric_block[i_c*n_phi:n_phi+i_c*n_phi,adest[0]:n_dof + 1:n_comp_u].T
            j_el[n_comp_u+i_c:n_dof + 1:n_components,adest[1]:n_dof+1:n_components] += assymetric_block[i_c*n_phi:n_phi+i_c*n_phi,adest[1]:n_dof + 1:n_comp_u]
            j_el[adest[1]:n_dof+1:n_components,n_comp_u+i_c:n_dof + 1:n_components] += assymetric_block[i_c*n_phi:n_phi+i_c*n_phi,adest[1]:n_dof + 1:n_comp_u].T

        for c in range(n_components):
            b = c
            e = (c + 1) * n_phi * n_components + 1
            r_el[b:e:n_components] += phi_s_star @ f_val_star[c]

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
        scatter_form_data(element, m_lambda, m_mu, m_kappa, m_gamma, f_rhs, u_field, cell_map, row, col, data)
        for element in u_field.elements
    ]

    def scatter_bc_form_data(element, u_field, cell_map, row, col, data):

        n_components = u_field.n_comp
        el_data: ElementData = element.data

        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # find high-dimension neigh
        entity_map = u_field.dof_map.mesh_topology.entity_map_by_dimension(cell.dimension)
        neigh_list = list(entity_map.predecessors(cell.id))
        neigh_check_q = len(neigh_list) > 0
        assert neigh_check_q

        neigh_cell_id = neigh_list[0]
        neigh_cell_index = u_field.id_to_element[neigh_cell_id]
        neigh_cell = u_field.elements[neigh_cell_index].data.cell

        # destination indexes
        dest_neigh = u_field.dof_map.destination_indices(neigh_cell_id)
        dest = u_field.dof_map.bc_destination_indices(neigh_cell_id, cell.id)

        n_phi = phi_tab.shape[2]
        n_dof = n_phi * n_components
        js = (n_dof, n_dof)
        j_el = np.zeros(js)

        # local blocks
        beta = 1.0e12
        jac_block = np.zeros((n_phi, n_phi))
        for i, omega in enumerate(weights):
            phi = phi_tab[0, i, :, 0]
            jac_block += beta * det_jac[i] * omega * np.outer(phi, phi)

        for c in range(n_components):
            b = c
            e = (c + 1) * n_phi * n_components + 1
            j_el[b:e:n_components, b:e:n_components] += jac_block

        # scattering data
        c_sequ = cell_map[cell.id]

        # contribute lhs
        block_sequ = np.array(range(0, len(dest) * len(dest))) + c_sequ
        row[block_sequ] += np.repeat(dest, len(dest))
        col[block_sequ] += np.tile(dest, len(dest))
        data[block_sequ] += j_el.ravel()

    [
        scatter_bc_form_data(element, u_field, cell_map, row, col, data)
        for element in u_field.bc_elements
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
        el_data = element.data
        cell = el_data.cell
        points = el_data.quadrature.points
        weights = el_data.quadrature.weights
        phi_tab = el_data.basis.phi

        x = el_data.mapping.x
        det_jac = el_data.mapping.det_jac
        inv_jac = el_data.mapping.inv_jac

        # scattering dof
        dest = u_field.dof_map.destination_indices(cell.id)
        alpha_l = alpha[dest]

        # vectorization
        n_phi = phi_tab.shape[2]
        p_e_s = f_exact(x[:, 0], x[:, 1], x[:, 2])
        alpha_star = np.array(np.split(alpha_l, n_phi))
        p_h_s = (phi_tab[0, :, :, 0] @ alpha_star).T
        l2_error = np.sum(det_jac * weights * (p_e_s - p_h_s) * (p_e_s - p_h_s))
        return l2_error

    st = time.time()
    error_vec = [compute_l2_error(element, u_field) for element in u_field.elements]
    et = time.time()
    elapsed_time = et - st
    print("L2-error time:", elapsed_time, "seconds")
    l2_error = functools.reduce(lambda x, y: x + y, error_vec)
    print("L2-error: ", np.sqrt(l2_error))

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
        phi_shapes = evaluate_linear_shapes(points, element.data)
        (x, jac, det_jac, inv_jac) = evaluate_mapping(cell.dimension, phi_shapes, gmesh.points[cell.node_tags])
        phi_tab = element.evaluate_basis(points)
        n_phi = phi_tab.shape[2]
        f_e = f_exact(x[:, 0], x[:, 1], x[:, 2])
        alpha_star = np.array(np.split(alpha_l, n_phi))
        f_h = (phi_tab[0, :, :, 0] @ alpha_star).T
        fh_data[target_node_id] = f_h.ravel()
        fe_data[target_node_id] = f_e.ravel()

    mesh_points = gmesh.points
    con_d = np.array(
        [
            element.data.cell.node_tags
            for element in u_field.elements
        ]
    )
    meshio_cell_types = { 0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
    cells_dict = {meshio_cell_types[u_field.dimension]: con_d}
    p_data_dict = {"u_h": fh_data, "u_exact": fe_data}

    mesh = meshio.Mesh(
        points=mesh_points,
        cells=cells_dict,
        # Optionally provide extra data on points, cells, etc.
        point_data=p_data_dict,
    )
    mesh.write("md_h1_cosserat_elasticity.vtk")
    et = time.time()
    elapsed_time = et - st
    print("Post-processing time:", elapsed_time, "seconds")


def Geometry():

    box_points = np.array(
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

    # domain = build_box_3D(box_points)
    # domain.build_grahp()

    # box_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    # domain = build_box_2D(box_points)
    # domain.build_grahp()

    box_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    box_points = 1.1 * np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    file = "fracture_files/setting_2d_0.csv"
    domain = build_box_2D_with_lines(box_points, file)
    domain.build_grahp()

    mesher = ConformalMesher(dimension=domain.dimension)
    mesher.domain = domain
    mesher.generate_from_domain(0.05)
    mesher.write_mesh("gmesh.msh")

    aka = 0

def main():
    # polygon_polygon_intersection()
    Geometry()

    # gmesh_3d = generate_mesh_3d()
    # gmesh_2d = generate_mesh_2d()
    # gmesh_1d = generate_mesh_1d()

    # laplace
    # md_h1_laplace(gmesh_2d)
    # md_h1_elasticity(gmesh_3d)
    # md_h1_cosserat_elasticity(gmesh_2d)
    return


if __name__ == "__main__":
    main()
