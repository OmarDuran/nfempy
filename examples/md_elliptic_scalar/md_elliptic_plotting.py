from sys import platform

import numpy as np
import pyvista
import matplotlib.pyplot as plt

if platform == "linux" or platform == "linux2":
    pyvista.start_xvfb()
pyvista.global_theme.colorbar_orientation = "horizontal"


# def paint_on_canvas_plane():
#     file_name_geo = "geometric_mesh_1d.vtk"
#     file_name = "mixed_rt_c_0_material_parameters_1.0_1.0_1.0_99999.99999999999_1.0_1e-05_mesh_size_0.00390625_md_elliptic_two_fields.vtk"
#     # load data
#     hdiv_solution = pyvista.read(file_name)
#
#     # load data
#     bc_data = pyvista.read(file_name_geo)
#
#     plotter = pyvista.Plotter(shape=(1, 2))
#
#     # qh sub canvas
#     plotter.subplot(0, 0)
#     qh_sargs = dict(
#         title_font_size=20,
#         label_font_size=20,
#         shadow=False,
#         n_labels=4,
#         italic=True,
#         fmt="%.1e",
#         font_family="courier",
#         position_x=0.2,
#         position_y=0.91,
#         title="Numeric flux norm",
#     )
#     u_h_data = hdiv_solution.point_data["u_h"]
#     u_h_norm = np.array([np.linalg.norm(u_h) for u_h in u_h_data])
#
#     plotter.add_mesh(
#         hdiv_solution,
#         scalars=u_h_norm,
#         cmap="terrain",
#         show_edges=False,
#         scalar_bar_args=qh_sargs,
#         copy_mesh=True,
#     )
#     plotter.view_xy()
#
#     # qh sub canvas
#     plotter.subplot(0, 1)
#     qh_sargs = dict(
#         title_font_size=20,
#         label_font_size=20,
#         shadow=False,
#         n_labels=4,
#         italic=True,
#         fmt="%.1e",
#         font_family="courier",
#         position_x=0.2,
#         position_y=0.91,
#         title="Exact flux norm",
#     )
#     u_h_data = hdiv_solution.point_data["u_e"]
#     u_h_norm = np.array([np.linalg.norm(u_h) for u_h in u_h_data])
#
#     plotter.add_mesh(
#         hdiv_solution,
#         scalars=u_h_norm,
#         cmap="terrain",
#         show_edges=False,
#         scalar_bar_args=qh_sargs,
#         copy_mesh=True,
#     )
#     plotter.view_xy()
#     # plotter.show()
#     return plotter

def paint_on_canvas_plane():
    file_name_geo = "geometric_mesh_2d.vtk"
    file_name = "mixed_rt_c_0_material_parameters_1.0_1.0_1.0_99999.99999999999_1.0_1e-05_mesh_size_0.00390625_md_elliptic_two_fields.vtk"
    # load data
    hdiv_solution = pyvista.read(file_name)

    # load data
    bc_data = pyvista.read(file_name_geo)

    plotter = pyvista.Plotter(shape=(1, 2))

    # qh sub canvas
    plotter.subplot(0, 0)
    ph_sargs = dict(
        title_font_size=20,
        label_font_size=20,
        shadow=False,
        n_labels=4,
        italic=True,
        fmt="%.1e",
        font_family="courier",
        position_x=0.2,
        position_y=0.91,
        title="Numeric potential norm",
    )
    p_h_data = hdiv_solution.point_data["p_h"]


    plotter.add_mesh(
        hdiv_solution,
        scalars=p_h_data,
        cmap="terrain",
        show_edges=False,
        scalar_bar_args=ph_sargs,
        copy_mesh=True,
    )
    plotter.view_xy()

    # qh sub canvas
    plotter.subplot(0, 1)
    ph_sargs = dict(
        title_font_size=20,
        label_font_size=20,
        shadow=False,
        n_labels=4,
        italic=True,
        fmt="%.1e",
        font_family="courier",
        position_x=0.2,
        position_y=0.91,
        title="Exact potential norm",
    )
    p_h_data = hdiv_solution.point_data["p_e"]


    plotter.add_mesh(
        hdiv_solution,
        scalars=p_h_data,
        cmap="terrain",
        show_edges=False,
        scalar_bar_args=ph_sargs,
        copy_mesh=True,
    )
    plotter.view_xy()
    # plotter.show()
    return plotter

k_order = 0
source_folder_name = "output"
figure_folder_name = "vincent_figures"
figure_format = "pdf"



def plot_over_line(figure_file_name):
    # Make two points to construct the line between
    a = [0.0, 0.5, 0.0]
    b = [1.0, 0.5, 0.0]
    file_name_geo = "geometric_mesh_1d.vtk"
    file_name = "mixed_rt_c_1_material_parameters_1.0_1.0_1.0_99999.99999999999_1.0_1e-05_mesh_size_0.00390625_md_elliptic_two_fields.vtk"
    # load data
    hdiv_solution = pyvista.read(file_name)

    sampled = hdiv_solution.sample_over_line(a, b, resolution=1000)

    # p_h_vals = sampled.point_data["p_h"]
    # p_e_vals = sampled.point_data["p_e"]
    u_h_norm = np.array([np.linalg.norm(u_h) for u_h in sampled.point_data["u_h"]])
    u_e_norm = np.array([np.linalg.norm(u_e) for u_e in sampled.point_data["u_e"]])

    x = sampled["Distance"]
    # p_e = p_e_vals
    # p_data = np.vstack((p_e_vals, p_h_vals)).T
    # lineObjects = plt.plot(x, p_data)
    u_e = u_e_norm
    u_data = np.vstack((u_e_norm, u_h_norm)).T
    lineObjects = plt.plot(x, u_data)

    styles = ["solid", "--"]
    linewidths = [2.0, 2.0]
    for i, line in enumerate(lineObjects):
        line.set_linestyle(styles[i])
        line.set_linewidth(linewidths[i])
    plt.legend(iter(lineObjects), (r"$|| \mathbf{u}_e ||$", r"$|| \mathbf{u}_h ||$"))
    plt.title("")
    plt.xlabel("Length")
    #plt.ylabel("potential")
    plt.ylabel("Flux")
    # plt.show()

    plt.savefig(figure_file_name)

    return


folder_name = "vincent_figures"
import os

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

canvas = paint_on_canvas_plane()
# canvas.save_graphic("vincent_figures/uh_magnitude.eps")
#canvas.save_graphic("vincent_figures/uh_magnitude.pdf")
canvas.save_graphic("vincent_figures/ph_magnitude.pdf")


plot_over_line("vincent_figures/plot_over_line_u.pdf")
#plot_over_line("vincent_figures/plot_over_line_p.pdf")
