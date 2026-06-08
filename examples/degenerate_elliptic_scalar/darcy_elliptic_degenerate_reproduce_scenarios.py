"""Master reproducibility script for the degenerate elliptic scalar examples.

Execution order
---------------
1. darcy_elliptic_degenerate_scenario_1.py  →  output_scenario_1/
2. darcy_elliptic_degenerate_scenario_2.py  →  output_scenario_2/
3. plotting_for_darcy_darcy_scenario_1.py   →  figures_scenario_1/
4. plotting_for_darcy_darcy_scenario_2.py   →  figures_scenario_2/

All scripts are executed as subprocesses inside their own directory so that
local imports (domain_builder, LMWeakForm, strong_solutions_TArbogast, …)
resolve correctly regardless of where this master script is invoked from.

Full reproducibility – step by step
------------------------------------
Prerequisites
  • A working Python environment with all project dependencies installed
    (see requirements.txt or pyproject.toml at the repository root).
  • PETSc / petsc4py available and configured.
  • Run all commands from the **repository root** unless stated otherwise.

Step 1 – install the package in editable mode (once)
    pip install -e .

Step 2 – full run  (compute both scenarios, then generate all figures)
    python examples/degenerate_elliptic_scalar/darcy_elliptic_degenerate_reproduce_scenarios.py

    On success you will find:
        examples/degenerate_elliptic_scalar/output_scenario_1/   ← VTKs + convergence tables
        examples/degenerate_elliptic_scalar/output_scenario_2/   ← VTKs + convergence tables
        examples/degenerate_elliptic_scalar/figures_scenario_1/  ← PNG figures
        examples/degenerate_elliptic_scalar/figures_scenario_2/  ← PNG figures

Step 3 – re-generate figures only  (computations already done)
    python examples/degenerate_elliptic_scalar/darcy_elliptic_degenerate_reproduce_scenarios.py \\
        --plots-only

Step 4 – run computations only  (skip figure generation)
    python examples/degenerate_elliptic_scalar/darcy_elliptic_degenerate_reproduce_scenarios.py \\
        --compute-only

Notes
-----
  • Each stage reports [OK] / [FAIL] and the elapsed time.
  • If any stage fails the script exits with code 1 and prints a clear message.
  • The plotting stage is skipped automatically if the computation stage fails.
  • Individual scenario scripts can also be run directly from their directory:
        cd examples/degenerate_elliptic_scalar
        python darcy_elliptic_degenerate_scenario_1.py
        python darcy_elliptic_degenerate_scenario_2.py
        python plotting_for_darcy_darcy_scenario_1.py
        python plotting_for_darcy_darcy_scenario_2.py
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Detect uv and xvfb-run
# ---------------------------------------------------------------------------
_UV_PATH: str | None = shutil.which("uv")
_XVFB_PATH: str | None = shutil.which("xvfb-run")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hline(char: str = "-", width: int = 70) -> str:
    return char * width


def _banner(title: str) -> None:
    print(_hline("="))
    print(f"  {title}")
    print(_hline("="))


def _step(msg: str) -> None:
    print(f"\n{_hline()}")
    print(f"  STEP: {msg}")
    print(_hline())


def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}", file=sys.stderr)


_STDERR_NOISE = (
    "Context leak detected, CoreAnalytics returned false",
    "vtkCocoaRenderWindow",
    "vtkOpenGLRenderWindow",
    "Failed to get red color buffer",
    "Failed to get green color buffer",
    "Failed to get blue color buffer",
    "Failed to get alpha color buffer",
)


def _relay_stderr(stream) -> None:
    """Read *stream* line-by-line and forward to stderr, skipping known noise."""
    for raw in stream:
        line = raw.decode(errors="replace")
        if not any(noise in line for noise in _STDERR_NOISE):
            sys.stderr.write(line)
            sys.stderr.flush()


def run_script(
    script_name: str,
    extra_args: list[str] | None = None,
    use_xvfb: bool = False,
) -> bool:
    """Run *script_name* as a subprocess inside SCRIPT_DIR.

    Uses ``uv run <script>`` when *uv* is available on PATH; otherwise falls
    back to the current Python interpreter (``sys.executable``).

    When *use_xvfb* is True **and** ``xvfb-run`` is found on PATH, the command
    is wrapped with ``xvfb-run --auto-servernum`` so that off-screen VTK
    rendering works without a display.

    Returns True on success, False on non-zero exit code.
    stdout is streamed live; stderr is forwarded with macOS/VTK noise filtered out.
    """
    if _UV_PATH:
        cmd = [_UV_PATH, "run", script_name] + (extra_args or [])
    else:
        cmd = [sys.executable, script_name] + (extra_args or [])

    if use_xvfb and _XVFB_PATH:
        cmd = [_XVFB_PATH, "--auto-servernum"] + cmd

    print(f"  $ {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=SCRIPT_DIR, stderr=subprocess.PIPE)
    t = threading.Thread(target=_relay_stderr, args=(proc.stderr,), daemon=True)
    t.start()
    proc.wait()
    t.join()
    elapsed = time.perf_counter() - t0
    if proc.returncode == 0:
        _ok(f"{script_name} finished in {elapsed:.1f}s")
        return True
    else:
        _fail(f"{script_name} exited with code {proc.returncode} after {elapsed:.1f}s")
        return False


def check_folder(folder_name: str) -> bool:
    """Return True if *folder_name* exists inside SCRIPT_DIR and is non-empty."""
    path = SCRIPT_DIR / folder_name
    if not path.exists():
        _fail(f"Expected output folder not found: {path}")
        return False
    files = list(path.iterdir())
    if not files:
        _fail(f"Output folder exists but is empty: {path}")
        return False
    _ok(f"Output folder present with {len(files)} item(s): {path}")
    return True


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def run_computations() -> bool:
    """Execute both scenario computation scripts sequentially."""
    _step("Running scenario 1 computations  →  output_scenario_1/")
    ok1 = run_script("darcy_elliptic_degenerate_scenario_1.py")
    if ok1:
        ok1 = check_folder("output_scenario_1")

    _step("Running scenario 2 computations  →  output_scenario_2/")
    ok2 = run_script("darcy_elliptic_degenerate_scenario_2.py")
    if ok2:
        ok2 = check_folder("output_scenario_2")

    return ok1 and ok2


def run_plots() -> bool:
    """Execute both plotting scripts sequentially."""
    _step("Generating figures for scenario 1  →  figures_scenario_1/")
    ok1 = run_script(
        "plotting_for_darcy_darcy_scenario_1.py",
        [
            "--vtks",    "output_scenario_1",
            "--errors",  "output_scenario_1",
            "--figures", "figures_scenario_1",
            "--plot-fields",
            "--plot-fields-2d",
            "--plot-normal",
            "--levels", "2", "3", "4", "5", "6",
            "--field-resolution", "1600", "1600",
            "--color-bar-width", "0.04",
            "--camera-zoom", "1.1",
        ],
        use_xvfb=True,
    )
    if ok1:
        ok1 = check_folder("figures_scenario_1")

    _step("Generating figures for scenario 2  →  figures_scenario_2/")
    ok2 = run_script(
        "plotting_for_darcy_darcy_scenario_2.py",
        [
            "--vtks",    "output_scenario_2",
            "--errors",  "output_scenario_2",
            "--figures", "figures_scenario_2",
            "--plot-fields",
            "--plot-fields-2d",
            "--plot-normal",
            "--plot-enhanced",
            "--levels", "2", "3", "4", "5", "6",
            "--field-resolution", "1600", "1600",
            "--color-bar-width", "0.04",
            "--camera-zoom", "1.1",
        ],
        use_xvfb=True,
    )
    if ok2:
        ok2 = check_folder("figures_scenario_2")

    return ok1 and ok2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducibility master script for degenerate elliptic scalar scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip computation; only (re-)generate figures from existing output folders",
    )
    mode.add_argument(
        "--compute-only",
        action="store_true",
        help="Run computations only; skip figure generation",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()

    _banner("Darcy degenerate elliptic scalar – full reproducibility run")
    print(f"  Working directory : {SCRIPT_DIR}")
    print(f"  Python executable : {sys.executable}")
    if _UV_PATH:
        print(f"  Runner            : uv  ({_UV_PATH})  ← scripts run via 'uv run'")
    else:
        print(
            f"  Runner            : sys.executable  "
            f"(install 'uv' for faster dependency resolution)"
        )
    if _XVFB_PATH:
        print(f"  xvfb-run          : {_XVFB_PATH}  ← plotting wrapped with xvfb-run --auto-servernum")
    else:
        print("  xvfb-run          : not found  (plotting runs without virtual display)")

    success = True

    if not args.plots_only:
        ok = run_computations()
        if not ok:
            _fail("Computation stage failed — see errors above.")
            success = False

    if not args.compute_only and success:
        ok = run_plots()
        if not ok:
            _fail("Plotting stage failed — see errors above.")
            success = False

    total = time.perf_counter() - t_start
    print()
    _banner(f"{'DONE' if success else 'FAILED'}  –  total time {total:.1f}s")

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

