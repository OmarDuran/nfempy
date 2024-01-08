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


def create_conformal_mesher(domain: Domain, h, ref_l=0):
    mesher = ConformalMesher(dimension=domain.dimension)
    mesher.domain = domain
    mesher.generate_from_domain(h, ref_l)
    mesher.write_mesh("gmesh.msh")
    return mesher


def create_mesh(dimension, mesher: ConformalMesher, write_vtk_q=False):
    gmesh = Mesh(dimension=dimension, file_name="gmesh.msh")
    gmesh.set_conformal_mesher(mesher)
    gmesh.build_conformal_mesh()
    if write_vtk_q:
        gmesh.write_vtk()
    return gmesh


def paint_on_canvas():
    # The initial element size
    h = 1.0
    dimension = 3
    n_ref = 4
    write_geometry_vtk_q = True

    # Create a unit squared or a unit cube
    domain = create_domain(dimension)

    meshes = []
    mesh_sizes = []
    for lh in range(n_ref):
        h_val = h * (2**-lh)
        mesher = create_conformal_mesher(domain, h, lh)
        gmesh = create_mesh(dimension, mesher, write_geometry_vtk_q)
        _, _, h_max = mesh_size(gmesh)
        mesh = pyvista.read("geometric_mesh_3d.vtk")
        mesh_sizes.append(h_max)
        meshes.append(mesh)

    plotter = pyvista.Plotter(shape=(2, 2))

    plotter.subplot(0, 0)
    plotter.add_text("h = " + "{:1.2e}".format(mesh_sizes[0]), font_size=14, font="courier")
    plotter.add_mesh(meshes[0], show_edges=True)

    plotter.subplot(0, 1)
    plotter.add_text("h = " + "{:1.2e}".format(mesh_sizes[1]), font_size=14, font="courier")
    plotter.add_mesh(meshes[1], show_edges=True)

    plotter.subplot(1, 0)
    plotter.add_text("h = " + "{:1.2e}".format(mesh_sizes[2]), font_size=14, font="courier")
    plotter.add_mesh(meshes[2], show_edges=True)

    plotter.subplot(1, 1)
    plotter.add_text("h = " + "{:1.2e}".format(mesh_sizes[3]), font_size=14, font="courier")
    plotter.add_mesh(meshes[3], show_edges=True)

    return plotter


def paint_on_canvas_simple():
    # The initial element size
    h = 1.0
    dimension = 3
    n_ref = 4
    write_geometry_vtk_q = True

    # Create a unit squared or a unit cube
    domain = create_domain(dimension)

    meshes = []
    mesh_sizes = []
    for lh in [1, 2]:
        h_val = h * (2**-lh)
        mesher = create_conformal_mesher(domain, h, lh)
        gmesh = create_mesh(dimension, mesher, write_geometry_vtk_q)
        _, _, h_max = mesh_size(gmesh)
        mesh = pyvista.read("geometric_mesh_3d.vtk")
        mesh_sizes.append(h_max)
        meshes.append(mesh)

    plotter = pyvista.Plotter(shape=(1, 2))

    plotter.subplot(0, 0)
    plotter.add_text("h = " + "{:1.2e}".format(mesh_sizes[0]), font_size=14, font="courier")
    plotter.add_mesh(meshes[0], show_edges=True)

    plotter.subplot(0, 1)
    plotter.add_text("h = " + "{:1.2e}".format(mesh_sizes[1]), font_size=14, font="courier")
    plotter.add_mesh(meshes[1], show_edges=True)

    return plotter

canvas = paint_on_canvas()
canvas.save_graphic("images/meshes_example_1_full.eps")
canvas.save_graphic("images/meshes_example_1_full.pdf")

canvas = paint_on_canvas_simple()
canvas.save_graphic("images/meshes_example_1.eps")
canvas.save_graphic("images/meshes_example_1.pdf")
