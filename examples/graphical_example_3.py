import vtk
import numpy as np
import pyvista

pyvista.global_theme.colorbar_orientation = "vertical"


def paint_on_canvas():
    crinkle_q = True
    # (xmin, xmax, ymin, ymax, zmin, zmax)
    bounds = [0.3, 1.0, 0.3, 1.0, 0.3, 1.0]
    if crinkle_q:
        bounds = [0.2, 1.0, 0.2, 1.0, 0.2, 1.0]

    view_dir = [1.0, 1.0, 1.0]

    # load data
    hdiv_solution = pyvista.read("m2_dnc_k2_d3_four_fields_scaled_ex_3.vtk")
    bc_data = pyvista.read("geometric_mesh_2d_ex_3.vtk")

    plotter = pyvista.Plotter(shape=(2, 2))

    # clip domain
    hdiv_solution_clipped = hdiv_solution.clip_box(bounds, crinkle=crinkle_q)
    bc_data_clipped = bc_data.clip_box(bounds, crinkle=crinkle_q)

    # stress sub canvas
    plotter.subplot(0, 0)
    sh_sargs = dict(
        title_font_size=20,
        label_font_size=16,
        shadow=False,
        n_labels=5,
        italic=True,
        fmt="%.1f",
        font_family="courier",
        position_x=0.05,
        position_y=0.05,
        title=r"$\| \boldsymbol{\sigma} \|$",
    )

    sh_data = [sh.reshape((3, 3)) for sh in hdiv_solution_clipped.point_data["s_h"]]
    s_h_norm = np.array([np.sqrt(np.trace(sh.T @ sh)) for sh in sh_data])
    plotter.add_text("Stress", font_size=14, font="courier")
    plotter.add_mesh(
        hdiv_solution_clipped,
        scalars=s_h_norm,
        cmap="balance",
        show_edges=False,
        scalar_bar_args=sh_sargs,
        copy_mesh=True,
    )
    plotter.add_mesh(
        bc_data_clipped,
        color="white",
        style="wireframe",
        line_width=1.0,
        show_edges=False,
    )

    plotter.view_vector(view_dir)
    plotter.view_isometric()

    # couple stress sub canvas
    plotter.subplot(0, 1)
    mh_sargs = dict(
        title_font_size=20,
        label_font_size=16,
        shadow=False,
        n_labels=5,
        italic=True,
        fmt="%.1f",
        font_family="courier",
        position_x=0.05,
        position_y=0.05,
        title=r"$\| \tilde{\boldsymbol{\omega}} \|$",
    )
    mh_data = [mh.reshape((3, 3)) for mh in hdiv_solution_clipped.point_data["m_h"]]
    m_h_norm = np.array([np.sqrt(np.trace(mh.T @ mh)) for mh in mh_data])

    plotter.add_text("Couple Stress", font_size=14, font="courier")
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
        line_width=1.0,
        show_edges=False,
    )

    plotter.view_vector(view_dir)
    plotter.view_isometric()

    plotter.subplot(1, 0)
    uh_sargs = dict(
        title_font_size=20,
        label_font_size=16,
        shadow=False,
        n_labels=5,
        italic=True,
        fmt="%.1f",
        font_family="courier",
        position_x=0.05,
        position_y=0.05,
        title=r"$\| \mathbf{u} \|$",
    )
    plotter.add_text("Displacement", font_size=14, font="courier")
    u_h_norm = np.linalg.norm(hdiv_solution_clipped.point_data["u_h"], axis=1)
    plotter.add_mesh(
        hdiv_solution_clipped,
        scalars=u_h_norm,
        cmap="balance",
        show_edges=False,
        scalar_bar_args=uh_sargs,
        copy_mesh=True,
    )
    plotter.add_mesh(
        bc_data_clipped,
        color="white",
        style="wireframe",
        line_width=1.0,
        show_edges=False,
    )

    plotter.view_vector(view_dir)
    plotter.view_isometric()

    plotter.subplot(1, 1)
    th_sargs = dict(
        title_font_size=20,
        label_font_size=16,
        shadow=False,
        n_labels=5,
        italic=True,
        fmt="%.1f",
        font_family="courier",
        position_x=0.05,
        position_y=0.05,
        title=r"$\| \mathbf{r} \|$",
    )
    plotter.add_text("Rotation", font_size=14, font="courier")
    t_h_norm = np.linalg.norm(hdiv_solution_clipped.point_data["t_h"], axis=1)
    plotter.add_mesh(
        hdiv_solution_clipped,
        scalars=t_h_norm,
        cmap="balance",
        show_edges=False,
        scalar_bar_args=th_sargs,
        copy_mesh=True,
    )
    plotter.add_mesh(
        bc_data_clipped,
        color="white",
        style="wireframe",
        line_width=1.0,
        show_edges=False,
    )

    plotter.view_vector(view_dir)
    plotter.view_isometric()

    return plotter


canvas = paint_on_canvas()
canvas.save_graphic("images/approximations_crinkle_example_3.eps")
