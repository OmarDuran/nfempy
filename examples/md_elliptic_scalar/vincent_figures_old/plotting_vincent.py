from sys import platform
import os
import numpy as np
import pyvista
from md_elliptic_scalar import (
    compose_case_name,
    method_definition,
)
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
        title_font_size=35,
        label_font_size=35,
        shadow=False,
        n_labels=4,
        italic=True,
        fmt="%.1e",
        font_family="courier",
        position_x=0.2,
        position_y=0.87,
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


def plot_over_line(vtk_file_name, scalar_names, title_string, figure_name):
    a = [0.0, 0.5, 0.0]
    b = [1.0, 0.5, 0.0]
    # Load data from the VTK file
    pyvista_unstructured_mesh = pyvista.read(vtk_file_name)
    sampled = pyvista_unstructured_mesh.sample_over_line(a, b, resolution=1000)

    quantities_norm = []
    for scalar_name in scalar_names:
        quantity_data = sampled.point_data[scalar_name]
        quantity_norm = np.array([np.linalg.norm(value) for value in quantity_data])
        quantities_norm.append(quantity_norm)

    x = sampled["Distance"] - 1.0

    # Plotting
    plt.figure(figsize=(12, 8))
    lineObjects = plt.plot(x, np.array(quantities_norm).T)
    linewidths = [4.0, 4.0]
    styles = ["solid", "--"]

    for i, line in enumerate(lineObjects):
        line.set_linestyle(styles[i])
        line.set_linewidth(linewidths[i])

    legends = []
    for scalar_name in scalar_names:
        legend = r"$|| " + scalar_name + " ||$"
        legends.append(legend)

    plt.legend(iter(lineObjects), legends, fontsize=20)
    plt.title(title_string, fontsize=25)
    plt.xlabel("Length", fontsize=25)
    plt.ylabel("", fontsize=25)
    plt.savefig(figure_name)
    return


def two_dimnesional_plots(
    k_order,
    source_folder_name,
    figure_folder_name,
    figure_format,
    dimensions,
    methods,
    titles_map,
    filters,
):
    for method in methods:
        if method[0] not in filters["method"]:
            continue
            case_name = compose_case_name(
                        method, dimension, domain, material, source_folder_name
            )
            vtk_suffix = "l_" + str(filters["l"]) + filters["suffix"]
            vtk_file_name = case_name + vtk_suffix

            if not os.path.exists(figure_folder_name):
                os.makedirs(figure_folder_name)

            title_string = titles_map[filters["scalar_name"]]
            canvas = paint_scalar_on_canvas_plane(
                vtk_file_name, filters["scalar_name"], title_string
            )
            figure_case_name = compose_case_name(
                method, dimension, domain, material, figure_folder_name
            )
            figure_suffix = (
                filters["scalar_name"] + "_magnitude." + figure_format
            )
            figure_name = figure_case_name + figure_suffix
            canvas.save_graphic(figure_name)


def one_dimnesional_plots(
    k_order,
    source_folder_name,
    figure_folder_name,
    figure_format,
    dimensions,
    methods,
    titles_map,
    filters,
):
    for method in methods:
        if method[0] not in filters["method"]:
            continue
            for domain in domains.items():
                if domain[0] not in filters["domain_type"]:
                    continue
                    case_name = compose_case_name(
                        method, dimension, domain, material, source_folder_name
                    )
                    vtk_suffix = "l_" + str(filters["l"]) + filters["suffix"]
                    vtk_file_name = case_name + vtk_suffix

                    if not os.path.exists(figure_folder_name):
                        os.makedirs(figure_folder_name)

                    title_string = titles_map[filters["scalar_names"][0]]
                    figure_case_name = compose_case_name(
                        method, dimension, domain, material, figure_folder_name
                    )

                    scalars_chunk = ""
                    for scalar_name in filters["scalar_names"]:
                        scalars_chunk += scalar_name
                        scalars_chunk += "_"

                    figure_suffix = scalars_chunk + "magnitude." + figure_format
                    figure_name = figure_case_name + figure_suffix
                    plot_over_line(
                        vtk_file_name,
                        filters["scalar_names"],
                        title_string,
                        figure_name,
                    )


k_order = 0
source_folder_name = "output"
figure_folder_name = "vincent_figures"
figure_format = "pdf"
dimensions = [2]
methods = method_definition(k_order)
titles_map = {
    "p_h": " pressure",
    "p_e": " pressure",
    "u_h": "velocity norm",
    "u_e": "velocity norm",

    "p_error": "l2-error in pressure",
    "u_error": "l2-error in velocity",

}

# Filters for 2d
filters_2d = {
    "method": ["mixed_rt"],
    "l": 4,
    "suffix": "_physical_two_fields.vtk",
    "scalar_name": "p_e",
}
two_dimnesional_plots(
    k_order,
    source_folder_name,
    figure_folder_name,
    figure_format,
    dimensions,
    methods,
    titles_map,
    filters_2d,
)

# # Filters for 1d
# figure_format_1d = "pdf"
# filters_1d = {
#     "method": ["mixed_rt"],
#     "l": 4,
#     "suffix": "_two_fields.vtk",
#     "domain_type": ["fitted", "unfitted"],
#     "parameter": [-1.5],
#     "scalar_names": ["v_e", "v_h"],
#}
# one_dimnesional_plots(
#      k_order,
#     source_folder_name,
#     figure_folder_name,
#     figure_format_1d,
#     dimensions,
#     methods,
#     titles_map,
#     filters_1d,
# )
