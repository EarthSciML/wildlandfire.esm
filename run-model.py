#!/usr/bin/env python3
# run-model.py — minimal single-model runner for a 2-D EarthSci `.esm` (Python).
#
# The Python counterpart of run-model.jl / run-model-rs.sh. Same model, same
# `.esm`, same answer: a 2-D level-set Hamilton-Jacobi front  psi_t = -S(n̂)|grad psi|
# on the doubly-periodic Camp-Fire box, with |grad psi| discretized by the Godunov
# (Rouy-Tourin / Osher-Sethian) upwind scheme (imported by reference from the
# sibling earthscidiscretizations checkout on the model-ref mount edge — nothing
# is vendored here). psi starts as the signed distance to a circle, so the front
# {psi = 0} expands outward at a Rothermel + NFDRS spread rate S(n̂). The spread is
# driven by THREE real, per-cell data loaders, each conservatively regridded onto
# the fire grid INSIDE the .esm (esm-spec §8.6 — regridding is a coupling
# expression): (1) USGS 3DEP terrain elevation (live USGS ImageServer GeoTIFF, no
# auth) → per-cell dz/dx, dz/dy; (2) LANDFIRE FBFM13 fuel model (live USGS
# ImageServer GeoTIFF, no auth) → per-cell fuel code → the Anderson-13 fuel
# properties; (3) ERA5 single-level near-surface met (2 m t2m, 2 m dewpoint d2m,
# 10 m u10/v10) fetched from the Copernicus CDS API (submit/poll/download,
# ~/.cdsapirc auth) → per-cell temperature, 10 m wind and (Magnus-derived) humidity.
# This runner supplies them by binding EarthSciIO providers to the
# loader fields USGS3DEP.raw.elevation, LANDFIRE.raw.fuel_model and ERA5.sl.{t2m,u10,v10,d2m},
# and the ERA5 source geometry (its 0.25° cells placed in the fire metre-frame so the
# regrid generalizes to a moved/larger domain). Loads through the earthsci_ast
# Python binding, integrates with SciPy (RK45) while sampling several snapshots, and
# plots the front over the terrain with the ERA5 wind (PNG). No conformance / MMS.
#
# Usage:
#   python run-model.py [model.esm] [t_end]
#     model.esm   path to the .esm to run   (default: ./wildlandfire.esm)
#     t_end       final integration time    (default: 57600 s = 16 h)
#
# Requires a network connection (3DEP + LANDFIRE ImageServers, no auth) and a
# Copernicus CDS key in ~/.cdsapirc (the ERA5 loader fetches via the CDS API).
#
# Environment: needs the earthsci_ast binding (dev'd in the sibling
# EarthSciAST checkout — set ESS_ROOT to override) and the EarthSciIO
# Python binding (sibling EarthSciIO checkout — set EIO_ROOT to override) plus
# numpy / scipy / requests / certifi / tifffile (GeoTIFF) / xarray + netCDF4 (ERA5),
# and matplotlib for the PNG (the run still prints field ranges without it). The
# companion run-model-py.sh provisions a minimal venv and invokes this script.

import os
import re
import sys
import math
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ESS_ROOT = os.environ.get(
    "ESS_ROOT", os.path.normpath(os.path.join(HERE, "..", "EarthSciAST"))
)
if not os.path.isfile(os.path.join(ESS_ROOT, "esm-schema.json")):
    sys.exit(
        f"EarthSciAST not found at '{ESS_ROOT}'; set ESS_ROOT or clone "
        "it as a sibling checkout of this repo."
    )
EIO_ROOT = os.environ.get(
    "EIO_ROOT", os.path.normpath(os.path.join(HERE, "..", "EarthSciIO"))
)
if not os.path.isfile(os.path.join(EIO_ROOT, "earthsciio", "__init__.py")):
    sys.exit(
        f"EarthSciIO (Python binding) not found at '{EIO_ROOT}'; set EIO_ROOT or "
        "clone EarthSciIO as a sibling checkout (needed for the terrain loader)."
    )
# Use the dev'd Python bindings straight from source (no install step).
sys.path.insert(0, os.path.join(ESS_ROOT, "pkg", "earthsci-ast-py", "src"))
sys.path.insert(0, EIO_ROOT)

import numpy as np
import earthsci_ast as esm
import earthsciio as esio        # EarthSciIO provides the ESS provider seam via
                                 # the duck-typed adapter in _provider_sample_field
                                 # (the Python analog of the Julia weakdep ext);
                                 # tifffile backs EarthSciIO's geotiff reader.

# --- diagnostic solver knobs (env-gated; defaults reproduce the standard run) ---
# ESS_RTOL / ESS_ATOL override the RK45 tolerances. ESS_MAXSTEP caps the solver step
# (a CFL-style cap); esm.simulate does not expose max_step, so we inject it into every
# already-imported scipy solve_ivp call site. ESS_TAG suffixes the output PNG so
# stability experiments don't overwrite each other.
# rtol=1e-4 (was 1e-2): with the exact (uncapped) Rothermel spread rate the front
# dynamics are genuinely stiff on steep terrain, and 1e-2 under-resolves them (it
# over-spreads / drifts). 1e-4 is converged — rtol=1e-5 gives an identical front.
_RTOL = float(os.environ.get("ESS_RTOL", "1e-4"))
_ATOL = float(os.environ.get("ESS_ATOL", "1e-5"))
_TAG = os.environ.get("ESS_TAG", "")
_MAXSTEP = os.environ.get("ESS_MAXSTEP")
if _MAXSTEP:
    import scipy.integrate as _si
    _orig_solve_ivp = _si.solve_ivp
    def _solve_ivp_capped(*a, **k):
        k.setdefault("max_step", float(_MAXSTEP))
        return _orig_solve_ivp(*a, **k)
    _si.solve_ivp = _solve_ivp_capped
    for _n, _m in list(sys.modules.items()):
        if _n.startswith("earthsci_ast") and getattr(_m, "solve_ivp", None) is _orig_solve_ivp:
            _m.solve_ivp = _solve_ivp_capped
    print(f"[solver] max_step capped at {float(_MAXSTEP):g} s (CFL cap)")

NT = 6        # number of front snapshots (t = 0 … t_end)
NX = 72       # grid cells along x; dx = LX/NX = 500 m (4x finer than the archive
NY = 80       # grid cells along y; dy = LY/NY = 500 m   camp_fire_model.jl grid)
LX = 36000.0  # domain width  (m): Camp-Fire extent (half-width 18 km)
LY = 40000.0  # domain height (m): Camp-Fire extent (half-width 20 km)

# Camp-Fire domain (WGS84 lon/lat): the 36x40 km box centered on the Pulga-Paradise
# midpoint, matching archive/camp_fire_model.jl. All three loaders fetch over it.
BBOX_W, BBOX_S, BBOX_E, BBOX_N = -121.7400, 39.6049, -121.3192, 39.9651
BBOX = f"{BBOX_W},{BBOX_S},{BBOX_E},{BBOX_N}"
# Landmarks (WGS84 lon/lat) marked on the plot: the ignition at Pulga (the real
# Camp Fire origin) and the town of Paradise.
PULGA_LONLAT    = (-121.437222, 39.810278)   # 39°48′37″N 121°26′14″W (ignition)
PARADISE_LONLAT = (-121.621944, 39.759722)   # 39°45′35″N 121°37′19″W
# USGS 3DEP / LANDFIRE source rasters: GX×GY must match TerrainRegrid's
# conservative_overlap NSRC and its src cartesian_cell_rings GX/GY. The source grid
# is 2× finer than the fire grid (src_dx = LX/GX = tgt_dx/2), so it is exactly
# 2×-nested in and origin-aligned to the target — the precondition the gated overlap
# broad-phase relies on. Kept proportional to NX/NY so the regrid stays correct at
# any resolution (the model's src_x/src_y/src_cells index_sets are 2·NX/2·NY/4·NX·NY).
GX   = 2 * NX
GY   = 2 * NY
# ERA5 single-level (near-surface) CDS request: a small NxN 0.25° lon/lat
# block around the fire (ERA5_NX×ERA5_NY must match Era5Regrid's ov5 NSRC / e5s
# GX,GY = 5×5 = 25). DISCRETE (hourly, time-varying): the CDS fetch pulls both
# Camp-Fire days (8-9 Nov, all 24 h → 48 hourly `valid_time` records) and a
# DISCRETE Provider slices ONE hour's record per solver tick, so the met changes
# over the 16 h fire (the Julia binding does the same). The Provider's window
# opens at the ignition hour (2018-11-08 14:00 UTC ≈ 06 PST) — solver t=0 maps
# there — and runs to the next morning (2018-11-09 06:00 UTC).
ERA5_AREA = [40, -122, 39, -121]        # [N,W,S,E] — 1° box → 5×5 cells @ 0.25°
ERA5_NX, ERA5_NY = 5, 5
ERA5_YEAR, ERA5_MONTH = 2018, 11        # Camp Fire ignition month
ERA5_DAYS = [8, 9]                      # ignition day + next (48 hourly records)
ERA5_START  = datetime(2018, 11, 8, 0, 0)    # loader epoch (cadence anchor)
ERA5_IGNITE = datetime(2018, 11, 8, 14, 0)   # ignition hour = solver t=0 (UTC)
ERA5_END    = datetime(2018, 11, 9, 6, 0)    # run-window end (UTC)

model_path = sys.argv[1] if len(sys.argv) >= 2 else \
    os.path.join(HERE, "wildlandfire.esm")
t_end = float(sys.argv[2]) if len(sys.argv) >= 3 else 57600.0   # 16 h

esio.register_format_readers()   # idempotent; registers the geotiff/netcdf readers
cache = esio.Cache(auth={"cds": esio.cds_auth()})   # cds auth for the ERA5 fetch

# --- USGS 3DEP elevation (GeoTIFF, no auth) → USGS3DEP.raw.elevation --------
terrain_url = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/"
    f"exportImage?bbox={BBOX}&size={GX},{GY}&format=tiff&bboxSR=4326&imageSR=4326"
    "&pixelType=F32&interpolation=+RSP_BilinearInterpolation&f=image"
)
# --- LANDFIRE FBFM13 fuel model (GeoTIFF, no auth) → LANDFIRE.raw.fuel_model -
landfire_url = (
    "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2022/LF2022_FBFM13_CONUS/"
    f"ImageServer/exportImage?bbox={BBOX}&bboxSR=4326&imageSR=4326&size={GX},{GY}"
    "&format=tiff&pixelType=S16&interpolation=+RSP_NearestNeighbor&f=image"
)
print(f"fetching   USGS 3DEP + LANDFIRE ({GX}×{GY}) and ERA5 met ({ERA5_NX}×{ERA5_NY}) for {BBOX} …")
try:
    terr = esio.Provider(esio.DataLoader(
        name="USGS3DEP", format="geotiff", url=terrain_url, variables=["elevation"],
        temporal=None, reader_kwargs={"band_names": ["elevation"]}), cache)
    fuel = esio.Provider(esio.DataLoader(
        name="LANDFIRE", format="geotiff", url=landfire_url, variables=["fuel_model"],
        temporal=None, reader_kwargs={"band_names": ["fuel_model"]}), cache)
    # --- ERA5 surface met via CDS (DISCRETE hourly, single-level: t2m,d2m,u10,v10)
    # Fetch BOTH Camp-Fire days (8-9 Nov) at all 24 hourly steps → 48 hourly
    # `valid_time` records in one NetCDF; each ERA5 provider is DISCRETE
    # (LoaderTemporal, hourly cadence, time_dim="valid_time"), so it slices ONE
    # hour's [lat,lon] record per solver tick and the met varies in-sim (single-
    # level fields have no pressure_level dimension). The window (ignition hour →
    # next morning) bounds the refresh schedule and anchors solver t=0 at
    # 2018-11-08 14:00 UTC.
    from earthsciio.backends.cds import encode_cds_url as _enc
    # ERA5 SINGLE-LEVEL request (reanalysis-era5-single-levels): near-surface
    # 2 m T (t2m), 2 m dewpoint (d2m) and 10 m wind (u10,v10) — NO pressure_level.
    # Preferred over the 1000 hPa pressure level, which is at/below ground over
    # elevated terrain and extrapolated there. RH is derived in-model from t2m+d2m
    # (Magnus), since single-level ERA5 has no relative-humidity variable.
    ERA5_SL_DATASET = "reanalysis-era5-single-levels"
    _req = {
        "product_type": ["reanalysis"],
        "variable": sorted(["2m_temperature", "2m_dewpoint_temperature",
                             "10m_u_component_of_wind", "10m_v_component_of_wind"]),
        "year": [str(ERA5_YEAR)], "month": [f"{ERA5_MONTH:02d}"],
        "day": [f"{d:02d}" for d in sorted(ERA5_DAYS)],
        "time": [f"{h:02d}:00" for h in range(24)],   # all 24 hourly steps (48 records / 2 days)
        "data_format": "netcdf", "download_format": "unarchived",
        "area": [int(a) for a in ERA5_AREA],
    }
    era5_url = _enc(ERA5_SL_DATASET, _req)
    # records_per_sample=2: the provider returns the TWO records bracketing the
    # current time (not one floor slice); the model's ERA5.w_time then linearly
    # interpolates the met in time — a smooth ramp between hourly records instead
    # of the old piecewise-constant step.
    era5_temporal = esio.LoaderTemporal(
        start=ERA5_START, frequency=timedelta(hours=1),
        file_period=timedelta(hours=48), time_dim="valid_time",
        end=ERA5_START + timedelta(hours=48), records_per_sample=2)
    def era5_prov(short):
        return esio.Provider(esio.DataLoader(name="ERA5", format="netcdf", url=era5_url,
            variables=[short], temporal=era5_temporal, auth_realm="cds"),
            cache, window=(ERA5_IGNITE, ERA5_END))
    providers = {
        "USGS3DEP.raw.elevation": terr, "LANDFIRE.raw.fuel_model": fuel,
        "ERA5.sl.t2m": era5_prov("t2m"), "ERA5.sl.u10": era5_prov("u10"),
        "ERA5.sl.v10": era5_prov("v10"), "ERA5.sl.d2m": era5_prov("d2m"),
    }
    # Prove the met varies over the run AND is now time-interpolated: at the
    # ignition hour the provider returns the TWO bracketing hourly records; the
    # model blends them with a weight that ramps 0→1 over the hour, so the forcing
    # is continuous rather than a piecewise-constant hourly step.
    def _bracket_means(prov, when):
        nds = prov.refresh(when)  # interp mode: shape [valid_time=2, lat, lon]
        name = (nds.variable_names()[0] if hasattr(nds, "variable_names")
                else next(iter(nds.variables)))
        d = np.asarray(nds[name].data, dtype=float)
        return float(np.nanmean(d[0])), float(np.nanmean(d[1]))
    _t_prov = providers["ERA5.sl.t2m"]
    _r0, _r1 = _bracket_means(_t_prov, ERA5_IGNITE)
    _e0, _e1 = _bracket_means(_t_prov, ERA5_END)
    print(f"ERA5 met is time-INTERPOLATED (2-record bracket per tick): mean 2 m T "
          f"bracket [{_r0:.2f}, {_r1:.2f}] K @ ignition {ERA5_IGNITE:%H:%M}Z "
          f"→ [{_e0:.2f}, {_e1:.2f}] K @ end {ERA5_END:%H:%M}Z; the model blends each "
          f"bracket linearly in time (was piecewise-constant hourly)", file=sys.stderr)
except Exception as err:
    sys.exit(
        f"could not build the terrain/fuel/met providers for {BBOX}\n"
        "the model requires providers for USGS3DEP.raw.elevation, "
        "LANDFIRE.raw.fuel_model and ERA5.sl.{t2m,u10,v10,d2m} (network + ~/.cdsapirc needed).\n"
        f"underlying error: {err}"
    )

# ERA5 src geometry: place the ERA5 sub-grid at its TRUE position in the fire
# metre-frame (equirectangular map of the ERA5 lon/lat against the fire bbox) so
# the conservative overlap generalizes to a moved/larger fire domain — each fire
# cell picks up whichever ERA5 cell(s) actually cover it.
_mlon, _mlat = LX / (BBOX_E - BBOX_W), LY / (BBOX_N - BBOX_S)
era5_geom = {
    "Era5Regrid.src_x0": (ERA5_AREA[1] - 0.125 - BBOX_W) * _mlon,   # west edge, westmost cell
    "Era5Regrid.src_dx": 0.25 * _mlon,
    "Era5Regrid.src_y0": (ERA5_AREA[2] - 0.125 - BBOX_S) * _mlat,   # south edge, southmost cell
    "Era5Regrid.src_dy": 0.25 * _mlat,
}

# Time-interpolation phase: the provider loads the two ERA5 records bracketing the
# current time; ERA5.w_time = frac((t - t_interp_ref)/dt_interp) ramps the blend
# 0→1 across each hour. Solver t=0 is the ignition hour (an ERA5 cadence anchor),
# so t_interp_ref=0; dt_interp = the hourly ERA5 frequency.
era5_interp = {"ERA5.t_interp_ref": 0.0, "ERA5.dt_interp": 3600.0}

# Regrid cell geometry: the fire-grid spacing (tgt/dx/bin = LX/NX) and the 2×-finer
# source spacing (src = LX/GX). These MUST track NX/NY so the regrid target grid
# stays identical to the level-set grid and the source stays 2×-nested in it; the
# model's numeric defaults (tgt 2000 m / src 1000 m) are only correct at NX=18/NY=20.
_dxf, _dyf = LX / NX, LY / NY           # fire-grid (target) spacing
_dxs, _dys = LX / GX, LY / GY           # source spacing (2× finer than target)
regrid_geom = {
    "TerrainRegrid.tgt_dx": _dxf, "TerrainRegrid.tgt_dy": _dyf,
    "TerrainRegrid.dx": _dxf, "TerrainRegrid.dy": _dyf,
    "TerrainRegrid.bin_dx": _dxf, "TerrainRegrid.bin_dy": _dyf,
    "TerrainRegrid.src_dx": _dxs, "TerrainRegrid.src_dy": _dys,
    "Era5Regrid.tgt_dx": _dxf, "Era5Regrid.tgt_dy": _dyf,
}

print(f"loading    {model_path}  (NX={NX}, NY={NY})")
file = esm.load_path(model_path, metaparameters={"NX": NX, "NY": NY})

# RK45 (explicit Dormand-Prince 5(4)) — the SciPy analog of Julia's Tsit5 / Rust's
# Erk. The level-set Godunov Hamiltonian is hyperbolic (non-stiff), so an adaptive
# explicit RK takes a modest number of steps; the implicit LSODA instead drives its
# step count up on the non-smooth |∇ψ| and is orders of magnitude slower here.
print(f"simulating (0.0, {t_end}) with RK45, {NT} snapshots (rtol={_RTOL:g} atol={_ATOL:g}) …")
# `inspect` captures the build-once geometry so we can read the fields the model
# actually used (TerrainRegrid.elev_xy / fuel_xy, Era5Regrid.t_xy/…) — no re-fetch.
insp = esm.BuildInspection()
# EarthSciAST phase 4: build once with `esm_problem`, then `solve`. Everything
# the BUILD depends on (providers, inspect) moves to construction; only the
# solver knobs stay on `solve`, under SciML names — `alg` (was `method`),
# `reltol`/`abstol` (were `rtol`/`atol`). The tolerances stay explicit: the
# library defaults are now 1e-4/1e-6, and this run needs its own values.
sim = esm.solve(
    esm.esm_problem(
        file, (0.0, t_end),
        p={"LevelSetFireSpread.Lx": LX, "LevelSetFireSpread.Ly": LY,
           **era5_geom, **era5_interp, **regrid_geom},
        providers=providers, inspect=insp,
    ),
    alg="RK45", reltol=_RTOL, abstol=_ATOL,
)
if sim.retcode is not esm.ReturnCode.Success:
    sys.exit(f"solver failed ({sim.retcode.name}): {sim.message}")

# Assemble the primary 2-D field: element rows "<stem>[i,j]", grouped by stem,
# indices parsed as integers. sim.vars are the row labels of sim.y (n_rows × n_t).
groups = {}   # stem -> [(i, j, row)]
cell_re = re.compile(r"^(.*?)\[(\d+),\s*(\d+)\]$")
for row, name in enumerate(sim.vars):
    m = cell_re.match(name)
    if m is None:
        continue
    stem, i, j = m.group(1), int(m.group(2)), int(m.group(3))
    groups.setdefault(stem, []).append((i, j, row))
if not groups:
    sys.exit("no 2-D indexed state field found to plot")

# Prefer the level-set state field psi (the mounted component also exposes many
# [x,y] observeds); fall back to the widest 2-D field if no psi is present.
psi_stems = [k for k in groups if k.endswith("psi")]
stem = max(psi_stems or list(groups), key=lambda k: len(groups[k]))
cells = groups[stem]
nx = max(c[0] for c in cells)
ny = max(c[1] for c in cells)
dx = LX / nx
dy = LY / ny
xs = [(i - 0.5) * dx for i in range(1, nx + 1)]
ys = [(j - 0.5) * dy for j in range(1, ny + 1)]

# SciPy's solve() samples a dense trajectory rather than exact save times, so
# reconstruct psi at the NT requested snapshots by linear interpolation of each
# element row (the fixture-runner convention; matches the Julia saveat output to
# the plot's precision). psi[k] is the NX×NY field at snapshot time ts[k].
t = np.asarray(sim.t, dtype=float)
Y = np.asarray(sim.y, dtype=float)
ts = list(np.linspace(0.0, t_end, NT))
psi = [np.full((nx, ny), np.nan) for _ in ts]
for (i, j, row) in cells:
    yi = np.interp(ts, t, Y[row])
    for k in range(len(ts)):
        psi[k][i - 1, j - 1] = yi[k]


def zero_segments(xs, ys, Z):
    """Marching squares for the zero contour of a field sampled at (xs[i], ys[j]);
    returns line segments packed into (X, Y) with NaN breaks (one plot series)."""
    X, Y = [], []

    def interp(a, b):
        s = a[2] / (a[2] - b[2])
        return (a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]))

    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            c = ((xs[i],     ys[j],     Z[i, j]),
                 (xs[i + 1], ys[j],     Z[i + 1, j]),
                 (xs[i + 1], ys[j + 1], Z[i + 1, j + 1]),
                 (xs[i],     ys[j + 1], Z[i, j + 1]))     # CCW
            pts = []
            for e in range(4):
                a = c[e]
                b = c[(e + 1) % 4]
                if (a[2] < 0) != (b[2] < 0):
                    pts.append(interp(a, b))
            if len(pts) >= 2:
                X += [pts[0][0], pts[1][0], np.nan]
                Y += [pts[0][1], pts[1][1], np.nan]
    return X, Y


cellarea = dx * dy


# Every behavior input is now a real, per-cell loader field (terrain, fuel and
# met are all spatially varying), so summarise each regridded field's range +
# domain mean straight from the build inspection rather than a single scalar.
def _field_summary(key, label, unit, scale=1.0):
    a = insp.setup_arrays.get(key)
    if a is None:
        return None
    a = np.asarray(a, dtype=float) * scale
    return f"{label} {np.nanmin(a):.2f}–{np.nanmax(a):.2f} {unit} (mean {np.nanmean(a):.2f})"


print("\ncoupled loader forcing (per-cell, regridded onto the fire grid):")
for line in filter(None, [
    _field_summary("TerrainRegrid.elev_xy", "  3DEP elevation", "m"),
    _field_summary("TerrainRegrid.fuel_xy", "  LANDFIRE fuel code", ""),
    _field_summary("Era5Regrid.t_xy", "  ERA5 temperature", "K"),
    _field_summary("Era5Regrid.rh_xy", "  ERA5 rel. humidity", "", 100.0),
    _field_summary("Era5Regrid.u_xy", "  ERA5 u-wind", "m/s"),
    _field_summary("Era5Regrid.v_xy", "  ERA5 v-wind", "m/s"),
]):
    print(line)
print(f"solver: success; field '{stem}' is {nx}×{ny}")
# burning-area centroid: with wind the front is no longer centered on the ignition
for k, tt in enumerate(ts):
    burn = psi[k] < 0
    n = int(np.sum(burn))
    r = math.sqrt(n * cellarea / math.pi)
    cx = float((np.asarray(xs)[:, None] * burn).sum() / n) if n else float("nan")
    print(f"  t={tt:7.0f}s  r_eff={r:8.1f} m  x-centroid={cx:8.0f} m "
          f"(ignition x=25901, Pulga)")

# --- single figure: real 3DEP terrain + front (psi=0) at each time; PNG output -
# The heatmap is the terrain the model ACTUALLY USED, read from the run itself:
# TerrainRegrid.elev_xy (the loader elevation conservatively regridded onto the
# fire [x,y] grid), pulled from the build inspection — not a separate download.
# The contours are the in-model fire front (color = time) on the same grid.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    print("\n(matplotlib not available — skipping PNG; radii above are the check)")
    sys.exit(0)

elev_xy = np.asarray(insp.setup_arrays["TerrainRegrid.elev_xy"])   # [x,y] = (nx,ny)
cg = LinearSegmentedColormap.from_list(
    "warm", ["#ffe08a", "#ff9d3c", "#e5391c", "#8b0000"])   # warm early → late over terrain
fig, ax = plt.subplots(figsize=(6.8, 5.6))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
# terrain as an [y,x] image over the fire-grid extent (elev_xy is [x,y] → transpose)
im = ax.imshow(elev_xy.T, origin="lower", extent=(0, LX, 0, LY),
               cmap="terrain", alpha=0.85, aspect="equal", interpolation="nearest")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("elevation (m)")
# ERA5 regridded 10 m wind (the met loader) as a light quiver overlay, so the plot
# shows all three loaders: 3DEP terrain (heatmap), the front, and the ERA5 wind.
u_xy = np.asarray(insp.setup_arrays.get("Era5Regrid.u_xy"))
v_xy = np.asarray(insp.setup_arrays.get("Era5Regrid.v_xy"))
if u_xy is not None and u_xy.shape == (nx, ny):
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    st = 2   # thin the arrows
    ax.quiver(Xg[::st, ::st], Yg[::st, ::st], u_xy[::st, ::st], v_xy[::st, ::st],
              color="#333333", alpha=0.5, scale=140, width=0.004, pivot="mid")
# Landmark markers: the ignition at Pulga and the town of Paradise, placed from
# their real lon/lat within the domain bbox.
def _lonlat_xy(lon, lat):
    return ((lon - BBOX_W) / (BBOX_E - BBOX_W) * LX,
            (lat - BBOX_S) / (BBOX_N - BBOX_S) * LY)
for (lon, lat), name, mk in [(PULGA_LONLAT, "Pulga (ignition)", "*"),
                             (PARADISE_LONLAT, "Paradise", "s")]:
    mx, my = _lonlat_xy(lon, lat)
    ax.scatter([mx], [my], marker=mk, s=130 if mk == "*" else 60,
               facecolor="#111111", edgecolor="white", linewidth=0.8, zorder=6)
    ax.annotate(name, (mx, my), textcoords="offset points", xytext=(7, 5),
                fontsize=8, color="#111111", zorder=6,
                path_effects=None, fontweight="bold")
for k, tt in enumerate(ts):
    Xc, Yc = zero_segments(xs, ys, psi[k])
    frac = 1.0 if len(ts) == 1 else k / (len(ts) - 1)
    ax.plot(Xc, Yc, color=cg(frac), lw=2.5, label=f"{tt/3600:.1f}h")
ax.set_xlim(0, LX)
ax.set_ylim(0, LY)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(f"{os.path.basename(model_path)} — front {{{stem}=0}}: 3DEP terrain + "
             "LANDFIRE fuel + ERA5 wind, Python / RK45", fontsize=8.5)
leg = ax.legend(title="t", fontsize=8, loc="upper left",
                bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
leg.get_title().set_fontsize(9)
for spine in ax.spines.values():
    spine.set_color("#52514e")
fig.tight_layout()

plot_path = os.path.join(os.path.dirname(os.path.abspath(model_path)),
                         f"front_python{_TAG}.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"wrote plot: {plot_path}")
