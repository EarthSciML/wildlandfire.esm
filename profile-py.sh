#!/usr/bin/env bash
# profile-py.sh — provision the venv and CPU-profile the Python simulate().
#
# The Python counterpart of profile-jl.jl / profile-rs.sh. Reuses the same local
# venv run-model-py.sh builds (numpy / scipy / sympy / jsonschema / matplotlib), and
# additionally installs `pyinstrument` — a statistical sampling profiler that, unlike
# cProfile, does NOT instrument every call (cProfile's per-call overhead multiplies
# this interpreter-heavy, many-tiny-step LSODA run ~30×), and unlike py-spy needs NO
# root on macOS (py-spy reads another process's memory, which SIP blocks without
# sudo — installing py-spy via pip *or* homebrew does not change that). So this runs
# at near-native speed on the full config with no elevated privileges. Outputs land
# in ./profiles/ (profile-python-top.txt, profile-python-tree.txt, and
# profile-python.speedscope.json — a compact flame graph for https://speedscope.app).
#
# Prefer a native-speed py-spy sampling profile instead? It is strictly a sampling
# profiler like this one, but on macOS it must run as root, e.g.:
#   sudo run-model-py/.venv/bin/py-spy record -o profiles/flamegraph-python.svg \
#        -- run-model-py/.venv/bin/python profile-py.py
# (pyinstrument gives the same sampling methodology here without the sudo.)
#
# Profiles the real three-loader simulate() (3DEP + LANDFIRE + ERA5, regridded
# in-model, at the runner's NX=18, NY=20); needs network + ~/.cdsapirc (ERA5 CDS).
# Usage: ./profile-py.sh [t_end]     (default t_end=30 s)
# The very short window is intentional: Python's per-cell interpreter + implicit
# LSODA is orders of magnitude slower than the compiled Julia/Rust explicit solvers,
# AND the one-time build (loader conservative regrid) already dominates a short run —
# it is t_end-independent, so a small window still captures it. Pass a larger t_end to
# sample more of the solve.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ESS_ROOT="${ESS_ROOT:-$(cd "$HERE/.." && pwd)/EarthSciAST}"
export ESS_ROOT
EIO_ROOT="${EIO_ROOT:-$(cd "$HERE/.." && pwd)/EarthSciIO}"
export EIO_ROOT

VENV="$HERE/run-model-py/.venv"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "creating venv at $VENV …"
  python3 -m venv "$VENV"
  "$PY" -m pip install --quiet --upgrade pip
  # Same stack the runner needs (profile-py.py drives the real three-loader
  # simulate): scientific core + the EarthSciIO provider deps (requests/certifi/
  # tifffile for the GeoTIFF loaders, xarray+netCDF4 for the ERA5 CDS netcdf).
  "$PY" -m pip install --quiet \
    "numpy>=1.20" "scipy>=1.7" "sympy>=1.9" "jsonschema>=4.0" "matplotlib>=3.5" \
    "requests>=2.20" "certifi>=2024.0" "tifffile>=2023.1" "xarray>=2023.1" "netCDF4>=1.6"
fi

# pyinstrument (statistical sampler) — the profiling backend. Install once if absent.
"$PY" -c 'import pyinstrument' 2>/dev/null || "$PY" -m pip install --quiet pyinstrument

exec "$PY" "$HERE/profile-py.py" "$@"
