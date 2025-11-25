from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from sys import platform
from typing import Iterable, Sequence, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pyvista
import seaborn as sns

from darcy_elliptic_degenerate_scenario_1 import (
    compose_case_name,
    create_domain,
    material_data_definition,
    method_definition,
)

if platform == "linux" or platform == "linux2":
    pyvista.start_xvfb()
pyvista.global_theme.colorbar_orientation = "horizontal"

SCRIPT_DIR = Path(__file__).resolve().parent

CUSTOM_PALETTES = {
    "tokyo": [
        "#e76f51",
        "#f4a261",
        "#e0e1dd",
        "#778da9",
        "#415a77",
        "#1b263b",
        "#0d1b2a",
    ]
}

def get_palette(name: str, n_colors: int) -> list[str | tuple[float, float, float]]:
    base = CUSTOM_PALETTES.get(name)
    if base:
        if len(base) >= n_colors:
            return base[:n_colors]
        repeats = (n_colors + len(base) - 1) // len(base)
        return (base * repeats)[:n_colors]
    try:
        pal = sns.color_palette(name, n_colors=n_colors)
    except ValueError:
        pal = sns.color_palette("deep", n_colors=n_colors)
    return list(pal)

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


# Improved loader: differentiate missing file vs read/parse failure so caller prints accurate message.
def load_mesh(vtk_path: Path, verbose: bool = False) -> pyvista.DataSet:
    """Load a VTK file with clearer error semantics.

    Raises
    ------
    FileNotFoundError
        If the path itself does not exist on disk.
    RuntimeError
        If the file exists but pyvista fails to read/parse it for any reason.
    """
    if not vtk_path.exists():
        raise FileNotFoundError(vtk_path)
    if verbose:
        print(f"[load_mesh] Found file: {vtk_path} (size={vtk_path.stat().st_size} bytes)")
    try:
        return pyvista.read(vtk_path)
    except FileNotFoundError as e:
        # Path existed, so treat this as a read failure (e.g., corrupted file or unsupported format)
        raise RuntimeError(f"Read failure (FileNotFound inside reader) for existing VTK file '{vtk_path}': {e}") from e
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to read VTK file '{vtk_path}': {e}") from e


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


def plot_scalar_field(mesh: pyvista.DataSet, config: PlotConfig, field: ScalarFieldPlot, figure_path: Path) -> None:
    """Plot a single scalar field."""
    field_mesh, scalar_name = prepare_scalar_dataset(mesh, field, config.height_scale)

    plotter = pyvista.Plotter(off_screen=True, window_size=config.field_resolution)
    bar_width = normalized_color_bar_width(config)
    scalar_bar = dict(
        title=field.title,
        position_x=0.4,
        position_y=0.05,
        width=0.2,
        height=0.05,
        vertical=False,
        title_font_size=20,
        label_font_size=18,
    )

    plotter.add_mesh(
        field_mesh,
        scalars=scalar_name,
        cmap=field.cmap,
        clim=field.clim,
        show_edges=False,
        scalar_bar_args=scalar_bar,
        copy_mesh=True,
    )
    plotter.enable_eye_dome_lighting()
    plotter.view_isometric()
    plotter.camera.azimuth = config.camera_azimuth
    plotter.camera.elevation = config.camera_elevation
    plotter.camera.zoom(config.camera_zoom)
    plotter.screenshot(str(figure_path), window_size=config.field_resolution)
    plotter.close()


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
    conv_rate: int = 1,
    y_range: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot log-log convergence curves and save the figure.

    Parameters
    ----------
    error_table : np.ndarray
        Table with first column h values and subsequent pairs (err, order?) columns.
    labels : Sequence[str]
        Labels for each error series extracted from error_table.
    figure_path : Path
        Destination path (including filename) for the saved plot image.
    triangle : TriangleSpec
        Specification for decorative convergence-rate triangle; if None, skipped.
    markers : Sequence[str]
        Marker styles cycled across error series.
    conv_rate : int, default 1
        Rate annotation used for triangle slope label.
    """
    h_values = error_table[:, 0]
    error_values = error_table[:, 1::2]
    palette = get_palette("tokyo", n_colors=len(labels))

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    for idx, (errors, label) in enumerate(zip(error_values.T, labels)):
        ax.loglog(
            h_values,
            errors,
            marker=markers[idx % len(markers)],
            label=label,
            color=palette[idx % len(palette)],
        )

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
        draw_data_triangle(
            ax,
            float(x0),
            float(x1),
            float(reference_series[anchor - 1]),
            float(reference_series[anchor]),
            conv_rate=conv_rate,
        )

    # Axis decoration & saving
    ax.set_xlabel("h")
    ax.set_ylabel("Error")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=10)
    # Apply custom vertical axis range if requested
    if y_range is not None:
        try:
            ymin, ymax = float(y_range[0]), float(y_range[1])
            if ymin < ymax:
                ax.set_ylim(ymin, ymax)
        except Exception:
            # Ignore invalid ranges and proceed with automatic scaling
            pass
    plt.tight_layout()
    # Ensure parent directory exists (normally already created)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(figure_path), dpi=300)
    plt.close()


def draw_data_triangle(ax: plt.Axes, x0: float, x1: float, y_prev: float, y_curr: float, conv_rate: int = 1) -> None:
    if min(x0, x1, y_prev, y_curr) <= 0:
        return
    logx0, logx1 = np.log10([x0, x1])
    logy0 = np.log10(max(y_prev, y_curr))
    delta_x = logx1 - logx0
    log_shift = np.abs(delta_x) / 2.0
    if np.isclose(np.abs(delta_x), 0.0):
        return
    effective_delta = conv_rate * delta_x
    A = np.array([logx0, logy0 - log_shift])
    B = np.array([logx0, logy0 + effective_delta - log_shift])
    C = np.array([logx1, logy0 + effective_delta - log_shift])
    points_log = np.array([A, B, C])
    points = [(10 ** px, 10 ** py) for px, py in points_log]
    AB_xc = np.mean([np.array(points[0]), np.array(points[1])], axis=0)
    BC_xc = np.mean([np.array(points[1]), np.array(points[2])], axis=0)
    horizontal_len = abs(points[2][0] - points[1][0])
    vertical_len = abs(points[1][1] - points[0][1])
    label_x_shift = (0.15) * horizontal_len if horizontal_len > 0 else 0.0
    label_y_shift = (0.15/conv_rate) * vertical_len if vertical_len > 0 else 0.0
    triangle = Polygon(points, closed=True, fill=False, edgecolor="#444444", linewidth=2)
    ax.add_patch(triangle)
    label_text = str(conv_rate)
    ax.text(AB_xc[0] + label_x_shift, AB_xc[1], label_text, ha="center", va="center", fontsize=8, color="#444444")
    ax.text(BC_xc[0], BC_xc[1] - label_y_shift, str(1), ha="center", va="center", fontsize=8, color="#444444")


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


def generate_field_plots(config: PlotConfig, scalar_fields: Sequence[ScalarFieldPlot], verbose: bool = False) -> None:
    """Generate individual field plots for each scalar field."""
    ensure_folder(config.figure_folder)
    for method, material, domain, dimension, level in iter_cases(config):
        vtk_base = build_case_basename(method, dimension, domain, material, config.vtks_folder)
        vtk_file = vtk_base.with_name(vtk_base.name + f"l_{level}_two_fields.vtk")
        try:
            mesh = load_mesh(vtk_file, verbose=verbose)
        except FileNotFoundError:
            print(f"Skipping missing VTK file (not found): {vtk_file}")
            continue
        except RuntimeError as e:
            print(f"Skipping unreadable VTK file (read failure): {vtk_file} -> {e}")
            continue
        case_prefix = build_case_basename(method, dimension, domain, material, config.figure_folder)
        for field in scalar_fields:
            figure_name = f"{case_prefix.name}l_{level}_{field.name}.{config.figure_format}"
            try:
                plot_scalar_field(mesh, config, field, config.figure_folder / figure_name)
            except KeyError as e:
                print(f"Skipping field {field.name} (not found in mesh): {e}")
                continue


def generate_convergence_plots(config: PlotConfig, table_kind: str) -> None:
    """Generate convergence plots. This function will read convergence tables and call plot_loglog_convergence.

    The function will use custom y-range settings from config if present (via attributes added to main)."""
    ensure_folder(config.figure_folder)
    labels = {
        "normal": ["p", "u", "proj p"],
    }[table_kind]
    file_suffix = {
        "normal": "conv_data.txt",
    }[table_kind]

    for method, material, domain, dimension, _ in iter_cases(config):
        case_prefix = build_case_basename(method, dimension, domain, material, config.errors_folder)
        table_path = case_prefix.with_name(case_prefix.name + file_suffix)
        try:
            table = load_error_table(table_path)
        except FileNotFoundError:
            continue
        figure_name = f"{case_prefix.name}{table_kind}_convergence.{config.figure_format}"
        conv_rate = 1
        # Determine y-range to use: prefer attributes on config set by CLI (if any)
        y_range = None
        try:
            if table_kind == "normal":
                y_range = getattr(config, "y_range_normal", None)
        except Exception:
            y_range = None

        plot_loglog_convergence(
            table,
            labels,
            config.figure_folder / figure_name,
            config.triangle,
            config.loglog_markers,
            conv_rate=conv_rate,
            y_range=y_range,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Darcy degenerate plot generator")
    parser.add_argument("--vtks", default="examples/degenerate_elliptic_scalar/output_scenario_1", help="Folder containing VTK files")
    parser.add_argument("--errors", default="examples/degenerate_elliptic_scalar/output_scenario_1", help="Folder containing convergence tables")
    parser.add_argument("--figures", default="examples/degenerate_elliptic_scalar/figures_scenario_1", help="Destination folder for plots")
    parser.add_argument("--formats", default="png", help="Figure format")
    parser.add_argument("--materials", nargs="*", type=float, default=[2.0, 1.0, 0.25, 0.125])
    parser.add_argument("--levels", nargs="*", type=int, default=[6])
    parser.add_argument("--domain", default="fitted")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--plot-fields", action="store_true")
    parser.add_argument("--plot-normal", action="store_true")
    parser.add_argument(
        "--y-range-normal",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        default=(1.0e-3,1.0e+3),
        help="Custom vertical axis range for normal convergence plots (provide two floats: ymin ymax)",
    )
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
    parser.add_argument("--verbose", action="store_true", help="Enable verbose diagnostic output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.plot_fields = True
    args.plot_normal = True
    args.plot_enhanced = True
    methods = list(method_definition(k_order=0))
    scalar_fields = [
        ScalarFieldPlot(name="p_h", title="Pressure (Computed)", clim=(-1.5, 1.5), threshold=(-1.5, 1.5)),
        ScalarFieldPlot(name="p_e", title="Pressure (Exact)", clim=(-1.5, 1.5), threshold=(-1.5, 1.5)),
        ScalarFieldPlot(name="u_h", title="Velocity norm (Computed)", use_norm=True, clim=(0.0, 270.0)),
        ScalarFieldPlot(name="u_e", title="Velocity norm (Exact)", use_norm=True, clim=(0.0, 270.0)),
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
        field_pairs=[],
        methods=methods,
        material_params=args.materials,
        refinement_levels=args.levels,
        domain_type=args.domain,
        dimension=args.dim,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        camera_zoom=args.camera_zoom,
        field_resolution=(int(args.field_resolution[0]), int(args.field_resolution[1])),
        color_bar_width=args.color_bar_width,
    )

    # Attach CLI-provided y-ranges if present (convert lists to tuples)
    if args.y_range_normal is not None:
        setattr(config, "y_range_normal", (float(args.y_range_normal[0]), float(args.y_range_normal[1])))
    else:
        setattr(config, "y_range_normal", None)
    if args.y_range_enhanced is not None:
        setattr(config, "y_range_enhanced", (float(args.y_range_enhanced[0]), float(args.y_range_enhanced[1])))
    else:
        setattr(config, "y_range_enhanced", None)

    if args.plot_fields:
        generate_field_plots(config, verbose=args.verbose)
    if args.plot_normal:
        generate_convergence_plots(config, "normal")
    if args.plot_enhanced:
        generate_convergence_plots(config, "enhanced")


if __name__ == "__main__":
    main()
