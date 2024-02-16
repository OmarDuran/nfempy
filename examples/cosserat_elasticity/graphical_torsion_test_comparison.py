import numpy as np
import pyvista


def paint_on_canvas():
    plotter = pyvista.Plotter(shape=(2, 1))

    view_dir = [1.0, 1.0, 1.0]
    h1_solution = pyvista.read("torsion_h1_cosserat_elasticity.vtk")
    hdiv_solution = pyvista.read("torsion_hdiv_cosserat_elasticity.vtk")

    sargs = dict(
        title_font_size=20,
        label_font_size=16,
        shadow=True,
        n_labels=5,
        italic=True,
        fmt="%.1f",
        font_family="courier",
        position_x=0.05,
        position_y=0.05,
        title="u norm",
    )

    plotter.subplot(0, 0)
    plotter.add_text("Primal method", font_size=14, font="courier")
    u_h_norm = np.linalg.norm(h1_solution.point_data["u_h"], axis=1)
    arrows = h1_solution.glyph(scale="u_h", orient="u_h", factor=3.0)
    plotter.add_mesh(arrows, cmap="balance")
    plotter.remove_scalar_bar("GlyphScale")
    plotter.add_mesh(
        h1_solution,
        scalars=u_h_norm,
        cmap="balance",
        show_edges=True,
        scalar_bar_args=sargs,
    )

    plotter.view_vector(view_dir)
    plotter.camera_position = "xy"
    plotter.camera.azimuth = -45
    plotter.camera.elevation = 15
    plotter.camera.zoom(2.25)

    plotter.subplot(1, 0)
    plotter.add_text("Dual method", font_size=14, font="courier")
    u_h_norm = np.linalg.norm(hdiv_solution.point_data["u_h"], axis=1)
    arrows = hdiv_solution.glyph(scale="u_h", orient="u_h", factor=3.0)
    plotter.add_mesh(arrows, cmap="balance")
    plotter.remove_scalar_bar("GlyphScale")
    plotter.add_mesh(
        hdiv_solution,
        scalars=u_h_norm,
        cmap="balance",
        show_edges=True,
        scalar_bar_args=sargs,
    )

    plotter.view_vector(view_dir)
    plotter.camera_position = "xy"
    plotter.camera.azimuth = -45
    plotter.camera.elevation = 15
    plotter.camera.zoom(2.25)

    return plotter


canvas = paint_on_canvas()
canvas.save_graphic("images/approximations_example_2.eps")
