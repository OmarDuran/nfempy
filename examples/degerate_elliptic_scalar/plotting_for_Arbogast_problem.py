from pathlib import Path
from sys import platform

import numpy as np
import pyvista
from degenerate_mixed_flow_model import compose_case_name, method_definition, material_data_definition, create_domain
import matplotlib.pyplot as plt

if platform == "linux" or platform == "linux2":
    pyvista.start_xvfb()
pyvista.global_theme.colorbar_orientation = "horizontal"




def paint_on_canvas_plane():
    file_name_geo = "geometric_mesh_2d.vtk"
    file_name_physical_two_field = "rates_arbogast_physical_two_fields.vtk"
    file_name_two_field = "rates_arbogast_two_fields.vtk"

    # Load data from the two VTK files
    physical_two_field_solution = pyvista.read(file_name_physical_two_field)
    two_field_solution = pyvista.read(file_name_two_field)

    plotter = pyvista.Plotter(shape=(1, 2))

    # First subplot for physical_solution
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
        title="Model Field",
    )
    q_h_data = two_field_solution.point_data["q_h"]  # Assuming 'm_h' is the relevant data key
    q_h_norm = np.array([np.linalg.norm(q_h) for q_h in q_h_data])

    plotter.add_mesh(
        two_field_solution,
        scalars=q_h_norm,
        cmap="gist_earth",
        show_edges=False,
        scalar_bar_args=qh_sargs,
        copy_mesh=True,
    )
    plotter.view_xy()

    # Second subplot for model_solution
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
        title="Physical Field",
    )
    p_h_data = physical_two_field_solution.point_data["p_h"]
    p_h_norm = np.array([np.linalg.norm(p_h) for p_h in p_h_data])

    plotter.add_mesh(
        physical_two_field_solution,
        scalars=p_h_norm,
        cmap="gist_earth",
        show_edges=False,
        scalar_bar_args=ph_sargs,
        copy_mesh=True,
    )
    plotter.view_xy()

    # Return the plotter to allow further manipulation or displaying elsewhere
    return plotter


k_order = 0
source_folder_name = "output"
dimensions = [2]
methods = method_definition(k_order)
for method in methods:
    for dimension in dimensions:
        # dimension dependent variants

        if dimension == 1 and method[0] == "mixed_bdm":
            continue

        fitted_domain = create_domain(dimension, make_fitted_q=True)
        unfitted_domain = create_domain(dimension, make_fitted_q=False)
        domains = {"fitted": fitted_domain, "unfitted": unfitted_domain}
        materials = material_data_definition(dimension)
        for domain in domains.items():
            for material in materials:
                case_name = compose_case_name(
                    method, dimension, domain, material, source_folder_name
                )
                aka=0

folder_name = "Arbogast_figures"
import os

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

canvas = paint_on_canvas_plane()
canvas.save_graphic("Arbogast_figures/ph_magnitude.eps")
canvas.save_graphic("Arbogast_figures/ph_magnitude.pdf")
canvas.save_graphic("Arbogast_figures/qh_magnitude.eps")
canvas.save_graphic("Arbogast_figures/qh_magnitude.pdf")

# def plot_over_line(figure_file_name):
#     # Make two points to construct the line between
#     a = [0.25, 0.25, 0.0]
#     b = [0.75, 0.75, 0.0]
#     file_name1 = "rates_arbogast_physical_two_fields.vtk"
#     file_name2 = "rates_arbogast_two_fields.vtk"
#     # Load data from both VTK files
#     arbogast_physical_two_fields_solution1 = pv.read(file_name1)
#     arbogast_two_fields_solution2 = pv.read(file_name2)
#
#     # Sample data over the line for both solutions
#     sampled1 = arbogast_physical_two_fields_solution1.sample_over_line(a, b, resolution=1000)
#     sampled2 = arbogast_two_fields_solution2.sample_over_line(a, b, resolution=1000)
#
#     # Compute norms for both solutions
#     u_h_norm1 = np.array([np.linalg.norm(u_h) for u_h in sampled1.point_data["u_h"]])
#     u_e_norm1 = np.array([np.linalg.norm(u_e) for u_e in sampled1.point_data["u_e"]])
#
#     v_h_norm2 = np.array([np.linalg.norm(v_h) for v_h in sampled2.point_data["v_h"]])
#     v_e_norm2 = np.array([np.linalg.norm(v_e) for v_e in sampled2.point_data["v_e"]])
#
#     x1 = sampled1["Distance"]
#     x2 = sampled2["Distance"]
#
#     # Plot data
#     fig, ax = plt.subplots()
#     ax.plot(x1, u_e_norm1, 'r-', label=r"$|| \mathbf{u}_e ||$ (file 1)", linestyle="solid", linewidth=2.0)
#     ax.plot(x1, u_h_norm1, 'r--', label=r"$|| \mathbf{u}_h ||$ (file 1)", linestyle="--", linewidth=2.0)
#
#     ax.plot(x2, v_e_norm2, 'b-', label=r"$|| \mathbf{v}_e ||$ (file 2)", linestyle="solid", linewidth=2.0)
#     ax.plot(x2, v_h_norm2, 'b--', label=r"$|| \mathbf{v}_h ||$ (file 2)", linestyle="--", linewidth=2.0)
#
#     ax.legend()
#     ax.set_title("")
#     ax.set_xlabel("Length")
#     ax.set_ylabel("Flux")
#
#     # Save the plot to a file
#     plt.savefig(figure_file_name)
#     plt.close()
#
#
#
# vtk_file_1d_1 = case_name + "rates_arbogast_physical_two_fields.vtk"
# vtk_file_1d_2 = case_name + "rates_arbogast_two_fields.vtk"
#
# plot_over_line("1d_comparison_plot.png", vtk_file_1d_1, vtk_file_1d_2)
#
#
#
# folder_name = "oden_figures"
# import os
#
# if not os.path.exists(folder_name):
#     os.makedirs(folder_name)
#
#
#
# plot_over_line("oden_figures/plot_over_line_q.ps")
