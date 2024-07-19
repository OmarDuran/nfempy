from pathlib import Path
from sys import platform

import numpy as np
import pyvista
import matplotlib.pyplot as plt

if platform == "linux" or platform == "linux2":
    pyvista.start_xvfb()
pyvista.global_theme.colorbar_orientation = "horizontal"


def paint_on_canvas_plane():
    file_name_geo = "geometric_mesh_2d.vtk"
    file_name = "rates_hdiv_model_problem.vtk"
    # load data
    hdiv_solution = pyvista.read(file_name)

    # load data
    bc_data = pyvista.read(file_name_geo)

    plotter = pyvista.Plotter(shape=(1, 2))

    # qh sub canvas
    plotter.subplot(0, 0)
    qh_sargs = dict(
        title_font_size=20,
        label_font_size=20,
        shadow=False,
        n_labels=4,
        italic=True,
        fmt="%.1e",
        font_family="courier",
        position_x=0.2,
        position_y=0.91,
        title="Numeric flux norm",
    )
    q_h_data = hdiv_solution.point_data["q_h"]
    q_h_norm = np.array([np.linalg.norm(q_h) for q_h in q_h_data])

    plotter.add_mesh(
        hdiv_solution,
        scalars=q_h_norm,
        cmap="gist_earth",
        show_edges=False,
        scalar_bar_args=qh_sargs,
        copy_mesh=True,
    )
    plotter.view_xy()

    # qh sub canvas
    plotter.subplot(0, 1)
    qh_sargs = dict(
        title_font_size=20,
        label_font_size=20,
        shadow=False,
        n_labels=4,
        italic=True,
        fmt="%.1e",
        font_family="courier",
        position_x=0.2,
        position_y=0.91,
        title="Exact flux norm",
    )
    q_h_data = hdiv_solution.point_data["q_e"]
    q_h_norm = np.array([np.linalg.norm(q_h) for q_h in q_h_data])

    plotter.add_mesh(
        hdiv_solution,
        scalars=q_h_norm,
        cmap="gist_earth",
        show_edges=False,
        scalar_bar_args=qh_sargs,
        copy_mesh=True,
    )
    plotter.view_xy()
    # plotter.show()
    return plotter


def paint_on_canvas_warp_scalar():
    file_name_geo = "geometric_mesh_2d.vtk"
    file_name = "rates_hdiv_model_problem.vtk"
    # load data

    hdiv_solution = pyvista.read(file_name)

    # add norm of approximated flux
    q_h_data = hdiv_solution.point_data["q_e"]
    q_h_norm = np.array([np.linalg.norm(q_h) for q_h in q_h_data])
    hdiv_solution.point_data.set_scalars(q_h_norm, "q_norm")

    hdiv_solution_warped = hdiv_solution.warp_by_scalar("q_norm", factor=50.0)

    # load data
    bc_data = pyvista.read(file_name_geo)

    plotter = pyvista.Plotter(shape=(1, 1))

    # qh sub canvas
    plotter.subplot(0, 0)
    mh_sargs = dict(
        title_font_size=20,
        label_font_size=20,
        shadow=False,
        n_labels=5,
        italic=True,
        fmt="%.2e",
        font_family="courier",
        position_x=0.2,
        position_y=0.91,
        title="Flux norm",
    )

    # plotter.add_text("Couple Stress", font_size=14, font="courier")
    plotter.add_mesh(
        hdiv_solution_warped,
        scalars=q_h_norm,
        cmap="gist_earth",
        show_edges=False,
        scalar_bar_args=mh_sargs,
        copy_mesh=True,
    )
    plotter.add_mesh(
        bc_data,
        color="white",
        style="wireframe",
        line_width=2.0,
        show_edges=False,
    )

    plotter.view_isometric()
    return plotter


def plot_over_line(figure_file_name):
    # Make two points to construct the line between
    a = [0.25, 0.25, 0.0]
    b = [0.75, 0.75, 0.0]

    file_name = "rates_hdiv_model_problem.vtk"
    # load data
    hdiv_solution = pyvista.read(file_name)

    sampled = hdiv_solution.sample_over_line(a, b, resolution=1000)

    q_h_norm = np.array([np.linalg.norm(q_h) for q_h in sampled.point_data["q_h"]])
    q_e_norm = np.array([np.linalg.norm(q_e) for q_e in sampled.point_data["q_e"]])

    x = sampled["Distance"]
    q_e = q_e_norm
    q_data = np.vstack((q_e_norm, q_h_norm)).T
    lineObjects = plt.plot(x, q_data)
    styles = ["solid", "--"]
    linewidths = [2.0, 2.0]
    for i, line in enumerate(lineObjects):
        line.set_linestyle(styles[i])
        line.set_linewidth(linewidths[i])
    plt.legend(iter(lineObjects), (r"$|| \mathbf{q}_e ||$", r"$|| \mathbf{q}_h ||$"))
    plt.title("")
    plt.xlabel("Length")
    plt.ylabel("Flux")
    # plt.show()

    plt.savefig(figure_file_name)

    return


folder_name = "oden_figures"
import os

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

canvas = paint_on_canvas_plane()
canvas.save_graphic("oden_figures/qh_magnitude.eps")
canvas.save_graphic("oden_figures/qh_magnitude.pdf")

canvas = paint_on_canvas_warp_scalar()
canvas.save_graphic("oden_figures/qh_magnitude_warp.eps")
canvas.save_graphic("oden_figures/qh_magnitude_warp.pdf")

plot_over_line("oden_figures/plot_over_line_q.ps")
