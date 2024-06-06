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


def plot_over_line(vtk_file_name, scalar_name,  title_string):
    a = [0.25, 0.25, 0.0]
    b = [0.75, 0.75, 0.0]
    # Load data from the VTK file
    pyvista_unstructured_mesh = pyvista.read(vtk_file_name)
    sampled = pyvista_unstructured_mesh.sample_over_line(a, b, resolution=1000)

    quantity_data = sampled.point_data[scalar_name]
    quantity_norm = np.array([np.linalg.norm(value) for value in quantity_data])


    x = sampled["Distance"]

    # Plotting
    plt.figure(figsize=(10, 6))
    lineObjects = plt.plot(x, quantity_norm)
    linewidths = [2.0, 2.0]
    styles = ["solid", "--"]

    for i, line in enumerate(lineObjects):
        line.set_linestyle(styles[i])
        line.set_linewidth(linewidths[i])
    plt.legend(iter(lineObjects), (r"$|| \mathbf{q}_e ||$", r"$|| \mathbf{q}_h ||$"))
    plt.title(title_string)
    plt.xlabel("Length")
    plt.ylabel(scalar_name)


    plt.show()
    return



def two_dimnesional_plots(k_order,source_folder_name,figure_folder_name,figure_format,dimensions,methods,titles_map,filters):
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



def one_dimnesional_plots(k_order,source_folder_name,figure_folder_name,figure_format,dimensions,methods,titles_map,filters):
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
                   # canvas = paint_scalar_on_canvas_plane(vtk_file_name, filters['scalar_name'], title_string)
                    figure_case_name = compose_case_name(
                        method, dimension, domain, material, figure_folder_name
                    )
                    figure_suffix = filters['scalar_name'] + '_magnitude.' + figure_format
                    figure_name = figure_case_name + figure_suffix
                    #canvas.save_graphic(figure_name)
                    # if dimension == 1:
                    plot_over_line(vtk_file_name, filters['scalar_name'], title_string)



k_order = 0
source_folder_name = "output"
figure_folder_name = "Arbogast_figures"
figure_format = 'eps'
dimensions = [1]
methods = method_definition(k_order)
titles_map = {'p_h': 'Physical pressure', 'q_h': 'Unphysical pressure', 'u_h': 'Physical velocity norm','v_h': 'Unphysical velocity norm', 'p_error': 'l2-error in physical pressure', 'q_error': 'l2-error in unphysical pressure','u_error':'l2-error in physical velocity','v_error':'l2-error in unphysical velocity'}

# to filter
filters = {'method': ['mixed_rt'], 'l': 5, 'suffix': '_physical_two_fields_l2_error.vtk', 'domain_type': ['fitted', 'unfitted'], 'parameter': [0.5],
           'scalar_name': 'u_error'}

one_dimnesional_plots(k_order,source_folder_name,figure_folder_name,figure_format,dimensions,methods,titles_map,filters)






# q_h_norm = np.array([np.linalg.norm(q_h) for q_h in sampled.point_data["q_h"]])
# q_e_norm = np.array([np.linalg.norm(q_e) for q_e in sampled.point_data["q_e"]])

# x = sampled["Distance"]
# q_e = quantity_norm
# q_data = np.vstack((q_e_norm, q_h_norm)).T
# lineObjects = plt.plot(x, q_data)
# styles = ["solid", "--"]
# linewidths = [2.0, 2.0]
# for i, line in enumerate(lineObjects):
#     line.set_linestyle(styles[i])
#     line.set_linewidth(linewidths[i])
# plt.legend(iter(lineObjects), (r"$|| \mathbf{q}_e ||$", r"$|| \mathbf{q}_h ||$"))
# plt.title("")
# plt.xlabel("Length")
# plt.ylabel("Flux")
#  plt.show()
#
# plt.savefig(figure_file_name)
#
# return
