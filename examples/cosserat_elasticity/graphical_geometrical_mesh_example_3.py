import numpy as np
import pyvista

from geometry.domain import Domain
from geometry.domain_market import build_box_1D, build_box_2D, build_box_3D
from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh
from mesh.mesh_metrics import mesh_size


def create_domain(dimension):
    if dimension == 1:
        box_points = np.array([[0, 0, 0], [1, 0, 0]])
        domain = build_box_1D(box_points)
        return domain
    elif dimension == 2:
        box_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        domain = build_box_2D(box_points)
        return domain
    else:
        box_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
        domain = build_box_3D(box_points)
        return domain


def create_conformal_mesher_from_file(file_name, dim):
    mesher = ConformalMesher(dimension=dim)
    mesher.write_mesh(file_name)
    return mesher


def create_mesh_from_file(file_name, dim, write_vtk_q=False):
    gmesh = Mesh(dimension=dim, file_name=file_name)
    gmesh.build_conformal_mesh()
    if write_vtk_q:
        gmesh.write_vtk()
    return gmesh


def paint_on_canvas():
    dimension = 3
    n_ref = 4
    write_geometry_vtk_q = True
    view_dir = [-1.0, -1.0, -1.0]

    meshes = []
    mesh_sizes = []
    for lh in [0, 1, 2, 3]:
        mesh_file = "gmsh_files/example_2_" + str(dimension) + "d_l_" + str(lh) + ".msh"
        gmesh = create_mesh_from_file(mesh_file, dimension, write_geometry_vtk_q)
        _, _, h_max = mesh_size(gmesh)
        mesh = pyvista.read("geometric_mesh_3d.vtk")
        mesh_sizes.append(h_max)
        meshes.append(mesh)

    plotter = pyvista.Plotter(shape=(2, 2))

    plotter.subplot(0, 0)
    plotter.add_text(
        "h = " + "{:1.2e}".format(mesh_sizes[0]), font_size=14, font="courier"
    )
    plotter.add_mesh(meshes[0], show_edges=True)
    plotter.view_vector(view_dir)

    plotter.subplot(0, 1)
    plotter.add_text(
        "h = " + "{:1.2e}".format(mesh_sizes[1]), font_size=14, font="courier"
    )
    plotter.add_mesh(meshes[1], show_edges=True)
    plotter.view_vector(view_dir)

    plotter.subplot(1, 0)
    plotter.add_text(
        "h = " + "{:1.2e}".format(mesh_sizes[2]), font_size=14, font="courier"
    )
    plotter.add_mesh(meshes[2], show_edges=True)
    plotter.view_vector(view_dir)

    plotter.subplot(1, 1)
    plotter.add_text(
        "h = " + "{:1.2e}".format(mesh_sizes[3]), font_size=14, font="courier"
    )
    plotter.add_mesh(meshes[3], show_edges=True)
    plotter.view_vector(view_dir)

    return plotter


def paint_on_canvas_simple():
    dimension = 3
    n_ref = 4
    write_geometry_vtk_q = True
    view_dir = [-1.0, -1.0, -1.0]

    meshes = []
    mesh_sizes = []
    for lh in [1]:
        mesh_file = "gmsh_files/ex_3/example_3_" + str(dimension) + "d_l_" + str(lh) + ".msh"
        gmesh = create_mesh_from_file(mesh_file, dimension, write_geometry_vtk_q)
        _, _, h_max = mesh_size(gmesh)
        mesh = pyvista.read("geometric_mesh_3d.vtk")
        mesh_sizes.append(h_max)
        meshes.append(mesh)

    plotter = pyvista.Plotter(shape=(1, 1))
    plotter.subplot(0, 0)
    plotter.add_text(
        "h = " + "{:1.2e}".format(mesh_sizes[0]), font_size=14, font="courier"
    )
    plotter.add_mesh(meshes[0], show_edges=True, line_width=3.0)
    plotter.view_vector(view_dir)

    return plotter


# canvas = paint_on_canvas()
# canvas.save_graphic("figures/meshes_example_3_full.eps")
# canvas.save_graphic("figures/meshes_example_3_full.pdf")

canvas = paint_on_canvas_simple()
canvas.save_graphic("figures/meshes_example_3.eps")
canvas.save_graphic("figures/meshes_example_3.pdf")
