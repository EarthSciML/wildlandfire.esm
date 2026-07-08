#!/usr/bin/env bash
# run-model-py.sh — provision a minimal venv and run the Python single-model runner.
#
# The Python counterpart of run-model-rs.sh. run-model.py needs the
# earthsci_toolkit binding (used straight from the sibling EarthSciSerialization
# source tree, no install) and the EarthSciIO Python binding (sibling EarthSciIO
# checkout) for the terrain loader, plus numpy / scipy / sympy / jsonschema and
# matplotlib for the PNG, and requests / tifffile for EarthSciIO's HTTP transport
# + pure-Python GeoTIFF reader (the 3DEP raster). This wrapper materializes those
# into a local venv (run-model-py/) once, then reuses it — mirroring how
# run-model.jl bootstraps run-model-jl/ and run-model-rs.sh caches its cargo
# build. The heavy full-package extras (netcdf4/xarray/pint, rasterio/GDAL) are
# intentionally skipped: this runner only touches the core load + array/PDE
# simulate path and reads the single-band GeoTIFF via tifffile.
#
# Requires a network connection — the model's terrain loader fetches live 3DEP.
#
# Usage: ./run-model-py.sh [model.esm] [t_end]
#   (defaults: ./wildlandfire.esm, t_end=57600 s = 16 h)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ESS_ROOT="${ESS_ROOT:-$(cd "$HERE/.." && pwd)/EarthSciSerialization}"
export ESS_ROOT
EIO_ROOT="${EIO_ROOT:-$(cd "$HERE/.." && pwd)/EarthSciIO}"
export EIO_ROOT

VENV="$HERE/run-model-py/.venv"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "creating venv at $VENV …"
  python3 -m venv "$VENV"
  "$PY" -m pip install --quiet --upgrade pip
  # Same scientific stack the toolkit's core + array/PDE simulate path needs,
  # plus matplotlib for the front PNG, and the EarthSciIO provider deps: requests
  # (HTTP + CDS transports; recent certifi for the ECMWF object-store TLS chain),
  # tifffile (pure-Python GeoTIFF reader — 3DEP elevation + LANDFIRE fuel), and
  # xarray + netCDF4 (the netcdf reader — ERA5 met via the CDS API). Pinned only
  # by lower bound (see the toolkit's / EarthSciIO's pyproject dependencies).
  "$PY" -m pip install --quiet \
    "numpy>=1.20" "scipy>=1.7" "sympy>=1.9" "jsonschema>=4.0" "matplotlib>=3.5" \
    "requests>=2.20" "certifi>=2024.0" "tifffile>=2023.1" "xarray>=2023.1" "netCDF4>=1.6"
fi

exec "$PY" "$HERE/run-model.py" "$@"
