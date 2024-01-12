import numpy as np
import pyvista
import vtk

pyvista.global_theme.colorbar_orientation = "horizontal"


def paint_on_canvas(crinkle_q):

    # (xmin, xmax, ymin, ymax, zmin, zmax)
    bounds = [0.3, 1.0, 0.3, 1.0, 0.3, 1.0]
    if crinkle_q:
        bounds = [0.2, 1.0, 0.2, 1.0, 0.2, 1.0]

    view_dir = [1.0, 1.0, 1.0]

    # load data
    hdiv_solution = pyvista.read("m2_dnc_k2_d3_four_fields_scaled_ex_3.vtk")
    bc_data = pyvista.read("geometric_mesh_2d.vtk")

    plotter = pyvista.Plotter(shape=(1, 1))

    # clip domain
    hdiv_solution_clipped = hdiv_solution.clip_box(bounds, crinkle=crinkle_q)
    bc_data_clipped = bc_data.clip_box(bounds, crinkle=crinkle_q)

    # couple stress sub canvas
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
        title=r"$\| \tilde{\boldsymbol{\omega}}_h \|$",
    )
    mh_data = [mh.reshape((3, 3)) for mh in hdiv_solution_clipped.point_data["m_h"]]
    m_h_norm = np.array([np.sqrt(np.trace(mh.T @ mh)) for mh in mh_data])

    # plotter.add_text("Couple Stress", font_size=14, font="courier")
    plotter.add_mesh(
        hdiv_solution_clipped,
        scalars=m_h_norm,
        cmap="balance",
        show_edges=False,
        scalar_bar_args=mh_sargs,
        copy_mesh=True,
    )
    plotter.add_mesh(
        bc_data_clipped,
        color="white",
        style="wireframe",
        line_width=2.0,
        show_edges=False,
    )

    plotter.view_vector(view_dir)
    plotter.view_isometric()

    return plotter


canvas = paint_on_canvas(False)
canvas.save_graphic("images/approximation_omega_example_3.eps")
canvas.save_graphic("images/approximation_omega_example_3.pdf")

canvas = paint_on_canvas(True)
canvas.save_graphic("images/approximation_omega_crinkle_example_3.eps")
canvas.save_graphic("images/approximation_omega_crinkle_example_3.pdf")
