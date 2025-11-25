from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from sys import platform
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
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
    cmap: str = "viridis"
    clim: tuple[float, float] | None = None


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
    scalar_fields: Sequence[ScalarFieldPlot]
    methods: Sequence[tuple[str, dict]]
    material_params: Sequence[float]
    refinement_levels: Sequence[int]
    domain_type: str = "fitted"
    dimension: int = 2
    cmap: str = "viridis"
    loglog_markers: Sequence[str] = ("o", "s", "^", "D")
    triangle: TriangleSpec = TriangleSpec(slope=1.0)


def ensure_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_mesh(vtk_path: Path) -> pyvista.DataSet:
    if not vtk_path.exists():
        raise FileNotFoundError(vtk_path)
    return pyvista.read(vtk_path)


def plot_scalar_field(mesh: pyvista.DataSet, config: PlotConfig, scalar: ScalarFieldPlot, figure_path: Path) -> None:
    if scalar.name not in mesh.point_data:
        raise KeyError(f"Scalar '{scalar.name}' not found in {figure_path}")
    values = mesh.point_data[scalar.name]
    norms = np.linalg.norm(values, axis=1) if values.ndim > 1 else values

    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_mesh(
        mesh,
        scalars=norms,
        cmap=scalar.cmap,
        clim=scalar.clim,
        show_edges=False,
        scalar_bar_args={"title": scalar.title},
        copy_mesh=True,
    )
    plotter.view_xy()
    plotter.camera.zoom(1.2)
    plotter.screenshot(str(figure_path))
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
        anchor = triangle.anchor_idx
        anchor = anchor if anchor >= 0 else len(h_values) + anchor
        anchor = max(1, min(anchor, len(h_values) - 1))
        x0, x1 = h_values[anchor - 1], h_values[anchor]
        y0 = error_values[0, anchor - 1] * triangle.scale
        slope_line = y0 * (np.array([x0, x1]) / x0) ** triangle.slope
        plt.loglog([x0, x1], slope_line, "k-", linewidth=2, label=f"rate {triangle.slope}")

    plt.xlabel("Element size h")
    plt.ylabel("L2 error")
    plt.legend()
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.savefig(figure_path, bbox_inches="tight", dpi=300)
    plt.close()


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
        for scalar in config.scalar_fields:
            figure_name = f"{case_prefix.name}l_{level}_{scalar.name}_map.{config.figure_format}"
            plot_scalar_field(mesh, config, scalar, config.figure_folder / figure_name)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.plot_fields = True
    # args.plot_normal = True
    # args.plot_enhanced = True
    methods = list(method_definition(k_order=0))
    scalar_fields = [
        ScalarFieldPlot(name="p_h", title="Physical pressure"),
        ScalarFieldPlot(name="p_e", title="Physical pressure"),
        ScalarFieldPlot(name="q_h", title="Unphysical pressure"),
        ScalarFieldPlot(name="q_e", title="Unphysical pressure"),
        ScalarFieldPlot(name="u_h", title="Physical velocity norm"),
        ScalarFieldPlot(name="u_e", title="Physical velocity norm"),
        ScalarFieldPlot(name="v_h", title="Unphysical velocity norm"),
        ScalarFieldPlot(name="v_e", title="Unphysical velocity norm"),
    ]

    figures_path = resolve_cli_path(args.figures, allow_missing=True)
    vtks_path = resolve_cli_path(args.vtks)
    errors_path = resolve_cli_path(args.errors)

    config = PlotConfig(
        figure_folder=figures_path,
        figure_format=args.formats,
        vtks_folder=vtks_path,
        errors_folder=errors_path,
        scalar_fields=scalar_fields,
        methods=methods,
        material_params=args.materials,
        refinement_levels=args.levels,
        domain_type=args.domain,
        dimension=args.dim,
    )

    if args.plot_fields:
        generate_field_plots(config)
    if args.plot_normal:
        generate_convergence_plots(config, "normal")
    if args.plot_enhanced:
        generate_convergence_plots(config, "enhanced")


if __name__ == "__main__":
    main()
