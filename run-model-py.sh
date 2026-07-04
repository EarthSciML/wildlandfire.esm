#!/usr/bin/env bash
# run-model-py.sh — provision a minimal venv and run the Python single-model runner.
#
# The Python counterpart of run-model-rs.sh. run-model.py needs the
# earthsci_toolkit binding (used straight from the sibling EarthSciSerialization
# source tree, no install) plus numpy / scipy / sympy / jsonschema and matplotlib
# for the PNG. This wrapper materializes those into a local venv (run-model-py/)
# once, then reuses it — mirroring how run-model.jl bootstraps run-model-jl/ and
# run-model-rs.sh caches its cargo build. The heavy full-package extras
# (netcdf4/xarray/pint) are intentionally skipped: this runner only touches the
# core load + array/PDE simulate path.
#
# Usage: ./run-model-py.sh [model.esm] [t_end]
#   (defaults: ./wildlandfire.esm, t_end=2.0)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ESS_ROOT="${ESS_ROOT:-$(cd "$HERE/.." && pwd)/EarthSciSerialization}"
export ESS_ROOT

VENV="$HERE/run-model-py/.venv"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "creating venv at $VENV …"
  python3 -m venv "$VENV"
  "$PY" -m pip install --quiet --upgrade pip
  # Same scientific stack the toolkit's core + array/PDE simulate path needs,
  # plus matplotlib for the front PNG. Pinned only by lower bound (see the
  # toolkit's pyproject dependencies).
  "$PY" -m pip install --quiet \
    "numpy>=1.20" "scipy>=1.7" "sympy>=1.9" "jsonschema>=4.0" "matplotlib>=3.5"
fi

exec "$PY" "$HERE/run-model.py" "$@"
