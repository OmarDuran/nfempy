from pathlib import Path
from sys import platform

import numpy as np
import pyvista

if platform == "linux" or platform == "linux2":
    pyvista.start_xvfb()
pyvista.global_theme.colorbar_orientation = "horizontal"


def paint_on_canvas():
    view_dir = [1.0, 1.0, 0.0]

    file_name_geo = "geometric_mesh_2d.vtk"
    file_name = "rates_hdiv_model_problem.vtk"
    # load data
    hdiv_solution = pyvista.read(file_name)

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
        fmt="%.1f",
        font_family="courier",
        position_x=0.2,
        position_y=0.91,
        title="Flux norm",
    )
    q_h_data = hdiv_solution.point_data["q_h"]
    q_h_norm = np.array([np.linalg.norm(q_h) for q_h in q_h_data])

    # plotter.add_text("Couple Stress", font_size=14, font="courier")
    plotter.add_mesh(
        hdiv_solution,
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

    plotter.view_xy()

    return plotter


folder_name = "oden_figures"
import os
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

canvas = paint_on_canvas()
canvas.save_graphic("oden_figures/qh_magnitude.eps")
canvas.save_graphic("oden_figures/qh_magnitude.pdf")
