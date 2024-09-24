from pathlib import Path
from sys import platform
import os
import numpy as np
import pyvista
from md_degenerate_elliptic_scalar import (
    compose_case_name,
    method_definition,
    # material_data,
    # co_dim,
)

import matplotlib.pyplot as plt


if platform == "linux" or platform == "linux2":
    pyvista.start_xvfb()
pyvista.global_theme.colorbar_orientation = "horizontal"

deltas_frac = [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5]

for delta_frac in deltas_frac:
    config = {}
    # domain and discrete domain data
    config["min_xc"] = -1.0
    config["min_yc"] = -1.0
    config["max_xc"] = +1.0
    config["max_yc"] = +1.0
    config["degeneracy_q"] = False

    # Material data
    material_data = {
        "rho_1": 1.0 / 10.0,
        "rho_2": 1.0 / 50.0,
        "kappa_c0": 1.0,
        "kappa_c1": 1.0 / delta_frac,
        "mu": 1.0,
        "kappa_normal": 1.0,
        "delta": delta_frac,
        "xi": 1.0,
        "eta": 1.0,
        "chi": 1.0,
    }
    config["m_data"] = material_data

    # function space data
    config["n_ref"] = 0
    config["k_order"] = 0
    config["mesh_sizes"] = [
        0.5,
        0.25,
        0.125,
        0.0625,
        0.03125,
        0.015625,
        0.0078125,
    ]


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
    a = [-1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
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
        # for dimension in dimensions:
        #     if dimension == 1 and method[0] == "mixed_bdm":
        #         continue
            # fitted_domain = create_domain(dimension, make_fitted_q=True)
            # unfitted_domain = create_domain(dimension, make_fitted_q=False)
            # domains = {"fitted": fitted_domain, "unfitted": unfitted_domain}
        materials = material_data["m_data"]
        case_names_by_co_dim = compose_case_name(config)
        for co_dim in [0, 1]:
            if co_dim [0] not in filters["domain_type"]:
                continue
            for material in materials:
                if material["m_data"] not in filters["parameter"]:
                    continue
                case_name = case_names_by_co_dim[co_dim]
                vtk_suffix = "l_" + str(filters["l"]) + filters["suffix"]
                vtk_file_name = case_name + vtk_suffix

                if not os.path.exists(figure_folder_name):
                    os.makedirs(figure_folder_name)

                title_string = titles_map[filters["scalar_name"]]
                canvas = paint_scalar_on_canvas_plane(
                    vtk_file_name, filters["scalar_name"], title_string
                )
                figure_case_name = compose_case_name(
                    method, co_dim, material, figure_folder_name
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
        for dimension in dimensions:
            if dimension == 1 and method[0] == "mixed_bdm":
                continue
            fitted_domain = create_domain(dimension, make_fitted_q=True)
            unfitted_domain = create_domain(dimension, make_fitted_q=False)
            domains = {"fitted": fitted_domain, "unfitted": unfitted_domain}
            materials = material_data_definition(dimension)
            for domain in domains.items():
                if domain[0] not in filters["domain_type"]:
                    continue
                for material in materials:
                    if material["m_par"] not in filters["parameter"]:
                        continue
                    case_name = compose_case_name(config)
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

max_dim = 2
k_order = 0
k_order = config["k_order"]
config["var_names"] = ("v", "q")
config["physical_var_names"] = ("u", "p")

flux_name, potential_name = config["var_names"]
config["physical_var_names"] = ("u", "p")
m_data = config["m_data"]
source_folder_name = "output"
figure_folder_name = "md_degenerates_figures"
figure_format = "pdf"
co_dim = 2
d = max_dim - co_dim
methods = method_definition(d, k_order, flux_name, potential_name)
titles_map = {
    "p_h": "Physical pressure",
    "p_e": "Physical pressure",
    "q_h": "Unphysical pressure",
    "q_e": "Unphysical pressure",
    "u_h": "Physical velocity norm",
    "u_e": "Physical velocity norm",
    "v_h": "Unphysical velocity norm",
    "v_e": "Unphysical velocity norm",
    "p_error": "l2-error in physical pressure",
    "q_error": "l2-error in unphysical pressure",
    "u_error": "l2-error in physical velocity",
    "v_error": "l2-error in unphysical velocity",
}

# Filters for 2d
filters_2d = {
    "method": ["mixed_rt"],
    "l": 4,
    "suffix": "_physical_two_fields.vtk",
    "domain_type": ["fitted", "unfitted"],
    "parameter": [1],
    "scalar_name": "p_e",
}
two_dimnesional_plots(
    k_order,
    source_folder_name,
    figure_folder_name,
    figure_format,
    co_dim,
    methods,
    titles_map,
    filters_2d,
)

# Filters for 1d
figure_format_1d = "pdf"
filters_1d = {
    "method": ["mixed_rt"],
    "l": 4,
    "suffix": "_two_fields.vtk",
    "domain_type": ["fitted", "unfitted"],
    "parameter": [-1.5],
    "scalar_names": ["v_e", "v_h"],
}
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
