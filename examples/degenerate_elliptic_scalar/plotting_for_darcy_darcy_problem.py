from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from sys import platform
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pyvista
from degenerate_mixed_flow_model import (
    compose_case_name,
    create_domain,
    material_data_definition,
    method_definition,
)

if platform == "linux" or platform == "linux2":
    pyvista.start_xvfb()
pyvista.global_theme.colorbar_orientation = "horizontal"

SCRIPT_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


REPO_ROOT = find_repo_root(SCRIPT_DIR)


@dataclass(frozen=True)
class ScalarFieldPlot:
    name: str
    title: str
    cmap: str = "terrain"
    clim: tuple[float, float] | None = None
    use_norm: bool = False
    threshold: tuple[float, float] | None = None


@dataclass(frozen=True)
class FieldPair:
    name: str
    left: ScalarFieldPlot
    right: ScalarFieldPlot
    height_scale: tuple[float, float] | None = None


@dataclass(frozen=True)
class TriangleSpec:
    slope: float
    anchor_idx: int = -2
    scale: float = 1.0


@dataclass
class PlotConfig:
    figure_folder: Path
    figure_format: str
    vtks_folder: Path
    errors_folder: Path
    field_pairs: Sequence[FieldPair]
    methods: Sequence[tuple[str, dict]]
    material_params: Sequence[float]
    refinement_levels: Sequence[int]
    domain_type: str = "fitted"
    dimension: int = 2
    cmap: str = "viridis"
    loglog_markers: Sequence[str] = ("o", "s", "^", "D")
    triangle: TriangleSpec = TriangleSpec(slope=1.0)
    height_scale: float = 1.0
    camera_azimuth: float = 0.0
    camera_elevation: float = 30.0
    camera_zoom: float = 1.2
    field_resolution: tuple[int, int] = (1600, 900)
    color_bar_width: float = 1.0


def normalized_color_bar_width(config: PlotConfig) -> float:
    width = config.color_bar_width
    if width <= 0:
        return 0.02
    if width <= 1:
        return width
    window_width = max(config.field_resolution[0], 1)
    return min(0.4, width / window_width)


def ensure_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_mesh(vtk_path: Path) -> pyvista.DataSet:
    if not vtk_path.exists():
        raise FileNotFoundError(vtk_path)
    return pyvista.read(vtk_path)


def prepare_scalar_dataset(mesh: pyvista.DataSet, scalar: ScalarFieldPlot, height_scale: float) -> tuple[pyvista.DataSet, str]:
    if scalar.name not in mesh.point_data:
        raise KeyError(f"Scalar '{scalar.name}' not found in mesh point data")
    values = mesh.point_data[scalar.name]
    raw_values = np.asarray(values)
    is_vector = raw_values.ndim == 2 and raw_values.shape[1] > 1
    if scalar.use_norm or is_vector:
        scalars = np.linalg.norm(raw_values, axis=1)
    else:
        scalars = raw_values.reshape(-1)

    quantity_name = f"{scalar.name}_magnitude" if scalar.use_norm else scalar.name
    plot_mesh = mesh.copy(deep=True)
    plot_mesh.point_data[quantity_name] = scalars
    filtered = plot_mesh
    if scalar.threshold is not None:
        filtered_candidate = plot_mesh.threshold(
            value=scalar.threshold,
            scalars=quantity_name,
            preference="point",
        )
        if filtered_candidate.n_cells > 0:
            filtered = filtered_candidate
    warped = filtered.warp_by_scalar(quantity_name, factor=height_scale)
    warped.point_data[quantity_name] = filtered.point_data[quantity_name]
    return warped, quantity_name


def plot_field_pair(mesh: pyvista.DataSet, config: PlotConfig, pair: FieldPair, figure_path: Path) -> None:
    pair_scale = pair.height_scale if pair.height_scale is not None else (config.height_scale, config.height_scale)
    left_mesh, left_scalar_name = prepare_scalar_dataset(mesh, pair.left, pair_scale[0])
    right_mesh, right_scalar_name = prepare_scalar_dataset(mesh, pair.right, pair_scale[1])

    plotter = pyvista.Plotter(off_screen=True, window_size=config.field_resolution)
    bar_width = normalized_color_bar_width(config)
    left_bar = dict(
        title=pair.left.title,
        position_x=0.1,
        position_y=0.15,
        height=0.7,
        width=bar_width,
        vertical=True,
        title_font_size=32,
        label_font_size=32,
    )
    right_bar = dict(
        title=pair.right.title,
        position_x=0.9,
        position_y=0.15,
        height=0.7,
        width=bar_width,
        vertical=True,
        title_font_size=32,
        label_font_size=32,
    )

    plotter.add_mesh(
        left_mesh,
        scalars=left_scalar_name,
        cmap=pair.left.cmap,
        clim=pair.left.clim,
        show_edges=False,
        scalar_bar_args=left_bar,
        copy_mesh=True,
    )
    plotter.add_mesh(
        right_mesh,
        scalars=right_scalar_name,
        cmap=pair.right.cmap,
        clim=pair.right.clim,
        show_edges=False,
        scalar_bar_args=right_bar,
        copy_mesh=True,
    )
    plotter.enable_eye_dome_lighting()
    plotter.view_isometric()
    plotter.camera.azimuth = config.camera_azimuth
    plotter.camera.elevation = config.camera_elevation
    plotter.camera.zoom(config.camera_zoom)
    plotter.screenshot(str(figure_path), window_size=config.field_resolution)
    plotter.close()


def load_error_table(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.loadtxt(path, delimiter=",", skiprows=1)


def plot_loglog_convergence(
    error_table: np.ndarray,
    labels: Sequence[str],
    figure_path: Path,
    triangle: TriangleSpec,
    markers: Sequence[str],
) -> None:
    h_values = error_table[:, 0]
    error_values = error_table[:, 1::2]

    plt.figure(figsize=(8, 6))
    for idx, (errors, label) in enumerate(zip(error_values.T, labels)):
        plt.loglog(h_values, errors, marker=markers[idx % len(markers)], label=label)

    if triangle and len(h_values) >= 2:
        if error_values.shape[1] == 0:
            raise ValueError("No error series available for convergence triangle")
        anchor = triangle.anchor_idx
        anchor = anchor if anchor >= 0 else len(h_values) + anchor
        anchor = max(1, min(anchor, len(h_values) - 1))
        x0, x1 = h_values[anchor - 1], h_values[anchor]
        series_at_anchor = error_values[anchor - 1, :]
        best_idx = int(np.argmin(series_at_anchor))
        reference_series = error_values[:, best_idx]
        draw_data_triangle(plt.gca(), x0, x1, reference_series[anchor - 1], reference_series[anchor])

    plt.xlabel("Element size h")
    plt.ylabel("L2 error")
    plt.legend()
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.savefig(figure_path, bbox_inches="tight", dpi=300)
    plt.close()


def draw_data_triangle(ax: plt.Axes, x0: float, x1: float, y_prev: float, y_curr: float) -> None:
    if min(x0, x1, y_prev, y_curr) <= 0:
        return
    logx0, logx1 = np.log10([x0, x1])
    logy0 = np.log10(max(y_prev, y_curr))
    delta_x = logx1 - logx0
    delta_y = y_curr - y_prev
    log_shift = np.abs(delta_x) / 2.0
    if np.isclose(np.abs(delta_x), 0.0):
        return
    A = np.array([logx0, logy0 - log_shift])
    B = np.array([logx0, logy0 + delta_x - log_shift])
    C = np.array([logx1, logy0 + delta_x - log_shift])
    points_log = np.array([A, B, C])
    points = [(10 ** px, 10 ** py) for px, py in points_log]
    AB_xc = np.mean([np.array(points[0]), np.array(points[1])], axis=0)
    BC_xc = np.mean([np.array(points[1]), np.array(points[2])], axis=0)
    horizontal_len = abs(points[2][0] - points[1][0])
    vertical_len = abs(points[1][1] - points[0][1])
    label_x_shift = 0.15 * horizontal_len if horizontal_len > 0 else 0.0
    label_y_shift = 0.25 * vertical_len if vertical_len > 0 else 0.0
    triangle = Polygon(points, closed=True, fill=False, edgecolor="#444444", linewidth=2)
    ax.add_patch(triangle)
    ax.text(AB_xc[0] + label_x_shift, AB_xc[1], "1", ha="center", va="center", fontsize=8, color="#444444")
    ax.text(BC_xc[0], BC_xc[1] - label_y_shift, "1", ha="center", va="center", fontsize=8, color="#444444")


def resolve_cli_path(path_str: str, allow_missing: bool = False) -> Path:
    raw = Path(path_str).expanduser()
    if raw.is_absolute():
        if raw.exists() or allow_missing:
            return raw
        raise FileNotFoundError(raw)

    candidates = [Path.cwd() / raw, REPO_ROOT / raw, SCRIPT_DIR / raw]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    if allow_missing:
        return (REPO_ROOT / raw).resolve()

    raise FileNotFoundError(
        f"Could not resolve '{path_str}'. Tried cwd, repository root, and script-relative locations."
    )


def iter_cases(config: PlotConfig) -> Iterable[tuple[str, dict, tuple[str, object], dict, int]]:
    for method in config.methods:
        for material in material_data_definition(config.dimension):
            if material["m_par"] not in config.material_params:
                continue
            fitted = create_domain(config.dimension, make_fitted_q=True)
            domain = (config.domain_type, fitted)
            for level in config.refinement_levels:
                yield method, material, domain, config.dimension, level


def build_case_basename(method, dimension, domain, material, folder: Path | None = None) -> Path:
    prefix = compose_case_name(method, dimension, domain, material, str(folder) if folder else None)
    return Path(prefix)


def generate_field_plots(config: PlotConfig) -> None:
    ensure_folder(config.figure_folder)
    for method, material, domain, dimension, level in iter_cases(config):
        vtk_base = build_case_basename(method, dimension, domain, material, config.vtks_folder)
        vtk_file = vtk_base.with_name(vtk_base.name + f"l_{level}_two_fields.vtk")
        try:
            mesh = load_mesh(vtk_file)
        except FileNotFoundError:
            print(f"Skipping missing VTK file: {vtk_file}")
            continue
        case_prefix = build_case_basename(method, dimension, domain, material, config.figure_folder)
        for pair in config.field_pairs:
            figure_name = f"{case_prefix.name}l_{level}_{pair.name}_pair.{config.figure_format}"
            plot_field_pair(mesh, config, pair, config.figure_folder / figure_name)


def generate_convergence_plots(config: PlotConfig, table_kind: str) -> None:
    ensure_folder(config.figure_folder)
    labels = {
        "normal": ["q", "v", "p", "u"],
        "enhanced": ["proj q", "proj p"],
    }[table_kind]
    file_suffix = {
        "normal": "normal_conv_data.txt",
        "enhanced": "enhanced_conv_data.txt",
    }[table_kind]

    for method, material, domain, dimension, _ in iter_cases(config):
        case_prefix = build_case_basename(method, dimension, domain, material, config.errors_folder)
        table_path = case_prefix.with_name(case_prefix.name + file_suffix)
        try:
            table = load_error_table(table_path)
        except FileNotFoundError:
            continue
        figure_name = f"{case_prefix.name}{table_kind}_convergence.{config.figure_format}"
        plot_loglog_convergence(table, labels, config.figure_folder / figure_name, config.triangle, config.loglog_markers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Darcy degenerate plot generator")
    parser.add_argument("--vtks", default="examples/degenerate_elliptic_scalar/output_scenario_2", help="Folder containing VTK files")
    parser.add_argument("--errors", default="examples/degenerate_elliptic_scalar/output_scenario_2", help="Folder containing convergence tables")
    parser.add_argument("--figures", default="examples/degenerate_elliptic_scalar/figures_scenario_2", help="Destination folder for plots")
    parser.add_argument("--formats", default="png", help="Figure format")
    parser.add_argument("--materials", nargs="*", type=float, default=[2.0, 1.0, 0.25, 0.125])
    parser.add_argument("--levels", nargs="*", type=int, default=[6])
    parser.add_argument("--domain", default="fitted")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--plot-fields", action="store_true")
    parser.add_argument("--plot-normal", action="store_true")
    parser.add_argument("--plot-enhanced", action="store_true")
    parser.add_argument("--camera-azimuth", type=float, default=-150.0, help="Azimuth angle for the 3D camera (degrees)")
    parser.add_argument("--camera-elevation", type=float, default=10.0, help="Elevation angle for the 3D camera (degrees)")
    parser.add_argument("--camera-zoom", type=float, default=1.1, help="Zoom factor for the 3D camera")
    parser.add_argument(
        "--field-resolution",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(1600, 1600),
        help="Resolution of generated field figure PNGs",
    )
    parser.add_argument(
        "--color-bar-width",
        type=float,
        default=0.02,
        help="Width of the scalar bar as a fraction of the render window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # args.plot_fields = True
    args.plot_normal = True
    # args.plot_enhanced = True
    methods = list(method_definition(k_order=0))
    scalar_fields = [
        ScalarFieldPlot(name="p_h", title="Pressure", clim=(-1.5, 1.5), threshold=(-1.5, 1.5)),
        ScalarFieldPlot(name="p_e", title="Pressure", clim=(-1.5, 1.5), threshold=(-1.5, 1.5)),
        ScalarFieldPlot(name="q_h", title="Scaled pressure", clim=(-1.5, 1.5), threshold=(-1.5, 1.5)),
        ScalarFieldPlot(name="q_e", title="Scaled pressure", clim=(-1.5, 1.5), threshold=(-1.5, 1.5)),
        ScalarFieldPlot(name="u_h", title="Velocity norm", use_norm=True, clim=(0.0, 270.0)),
        ScalarFieldPlot(name="u_e", title="Velocity norm", use_norm=True, clim=(0.0, 270.0)),
        ScalarFieldPlot(name="v_h", title="Scaled velocity norm", use_norm=True, clim=(0.0, 4.0)),
        ScalarFieldPlot(name="v_e", title="Scaled velocity norm", use_norm=True, clim=(0.0, 4.0)),
    ]

    field_lookup = {field.name: field for field in scalar_fields}

    figures_path = resolve_cli_path(args.figures, allow_missing=True)
    vtks_path = resolve_cli_path(args.vtks)
    errors_path = resolve_cli_path(args.errors)

    config = PlotConfig(
        figure_folder=figures_path,
        figure_format=args.formats,
        vtks_folder=vtks_path,
        errors_folder=errors_path,
        field_pairs=[
            FieldPair("qh_ph", field_lookup["q_h"], field_lookup["p_h"], height_scale=(1.0, 1.0)),
            FieldPair("qe_pe", field_lookup["q_e"], field_lookup["p_e"], height_scale=(1.0, 1.0)),
            FieldPair("ve_ue", field_lookup["v_e"], field_lookup["u_e"], height_scale=(0.5, 1.0/5000.0)),
            FieldPair("vh_uh", field_lookup["v_h"], field_lookup["u_h"], height_scale=(0.5, 1.0/5000.0)),
        ],
        methods=methods,
        material_params=args.materials,
        refinement_levels=args.levels,
        domain_type=args.domain,
        dimension=args.dim,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        camera_zoom=args.camera_zoom,
        field_resolution=tuple(args.field_resolution),
        color_bar_width=args.color_bar_width,
    )

    if args.plot_fields:
        generate_field_plots(config)
    if args.plot_normal:
        generate_convergence_plots(config, "normal")
    if args.plot_enhanced:
        generate_convergence_plots(config, "enhanced")


if __name__ == "__main__":
    main()
