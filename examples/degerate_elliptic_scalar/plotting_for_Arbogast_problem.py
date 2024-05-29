from pathlib import Path
from sys import platform
import os
import numpy as np
import pyvista
from degenerate_mixed_flow_model import compose_case_name, method_definition, material_data_definition, create_domain
import matplotlib.pyplot as plt

if platform == "linux" or platform == "linux2":
    pyvista.start_xvfb()
pyvista.global_theme.colorbar_orientation = "horizontal"




def paint_scalar_on_canvas_plane(vtk_file_name, scalar_name, title_string):

    # Load data from the two VTK files
    pyvista_unstructure_mesh = pyvista.read(vtk_file_name)

    plotter = pyvista.Plotter(shape=(1, 1))

    # First subplot for physical_solution
    plotter.subplot(0, 0)
    args = dict(
        title_font_size=20,
        label_font_size=20,
        shadow=False,
        n_labels=4,
        italic=True,
        fmt="%.1e",
        font_family="courier",
        position_x=0.2,
        position_y=0.91,
        title=title_string,
    )
    quantity_data = pyvista_unstructure_mesh.point_data[scalar_name]
    quantity_norm = np.array([np.linalg.norm(value) for value in quantity_data])

    plotter.add_mesh(
        pyvista_unstructure_mesh,
        scalars=quantity_norm,
        cmap="terrain",
        show_edges=False,
        scalar_bar_args=args,
        copy_mesh=True,
    )
    plotter.view_xy()
    return plotter


k_order = 0
source_folder_name = "output"
figure_folder_name = "Arbogast_figures"
figure_format = 'eps'
dimensions = [2]
methods = method_definition(k_order)
titles_map = {'p_h': 'Physical pressure', 'q_h': 'Unphysical pressure', 'u_h': 'Physical velocity norm', 'p_error': 'l2-error in pressure'}

# to filter
filters = {'method': ['mixed_rt'], 'l': 3, 'suffix': '_physical_two_fields_l2_error.vtk', 'domain_type': ['fitted', 'unfitted'], 'parameter': [2.0],
           'scalar_name': 'p_error'}


for method in methods:
    if method[0] not in filters['method']:
        continue
    for dimension in dimensions:
        if dimension == 1 and method[0] == "mixed_bdm":
            continue
        fitted_domain = create_domain(dimension, make_fitted_q=True)
        unfitted_domain = create_domain(dimension, make_fitted_q=False)
        domains = {"fitted": fitted_domain, "unfitted": unfitted_domain}
        materials = material_data_definition(dimension)
        for domain in domains.items():
            if domain[0] not in filters['domain_type']:
                continue
            for material in materials:
                if material['m_par'] not in filters['parameter']:
                    continue
                case_name = compose_case_name(
                    method, dimension, domain, material, source_folder_name
                )
                vtk_suffix = 'l_' + str(filters['l']) + filters['suffix']
                vtk_file_name = case_name + vtk_suffix

                if not os.path.exists(figure_folder_name):
                    os.makedirs(figure_folder_name)

                title_string = titles_map[filters['scalar_name']]
                canvas = paint_scalar_on_canvas_plane(vtk_file_name, filters['scalar_name'], title_string)
                figure_case_name = compose_case_name(
                    method, dimension, domain, material, figure_folder_name
                )
                figure_suffix = filters['scalar_name'] + '_magnitude.' + figure_format
                figure_name = figure_case_name + figure_suffix
                canvas.save_graphic(figure_name)



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
