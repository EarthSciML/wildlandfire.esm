#!/usr/bin/env python3
# profile-py.py — CPU sampling profile of the Python level-set simulate() (pyinstrument).
#
# The Python counterpart of profile-jl.jl and run-model-rs/src/bin/profile.rs. Loads
# and drives the SAME three-loader coupled model the runner uses (run-model.py): the
# Rothermel + NFDRS behavior stack feeding the level-set front, with every behavior
# input a real, per-cell data-loader field regridded onto the fire grid IN the .esm —
# 3DEP terrain, LANDFIRE fuel and ERA5 met. It fetches those loaders (cached after the
# first run) and binds them as providers, exactly like the runner — otherwise the
# profile is unrepresentative of the real coupled workload (the build-time conservative
# regrid is now the dominant cost, not the solve). Warms up the toolkit import + parse
# on a throwaway load (sheds one-time first-touch cost), then profiles a single
# simulate() at the runner's NX=18, NY=20 grid over a SHORT, BOUNDED t_end window
# (default 30 s; override via argv). Emits, under ./profiles/:
#   profile-python-top.txt   — self time by function (analog of the Rust/Julia -top)
#   profile-python-tree.txt  — pyinstrument call tree (self + total time per frame)
#   profile-python.speedscope.json — compact flame graph for https://speedscope.app
#
# Why pyinstrument. Statistical sampling profiler (like the Julia stdlib sampler and
# Rust's pprof-rs): it snapshots the Python stack via a signal timer FROM INSIDE the
# process, so unlike cProfile it does not instrument every call, and unlike py-spy it
# needs no root on macOS. The build-time regrid (s2 intersect_polygon over 1440×360
# source×target cells) shows as self-time of the earthsciio / interpreter frames; the
# per-cell RHS AST walk shows as the earthsci_toolkit interpreter frames — so the
# build-vs-solve and RHS-vs-solver splits are visible.
#
# Why a very short window. Python's per-cell AST interpreter plus implicit LSODA is
# orders of magnitude slower than the compiled Julia (Tsit5) / Rust (Erk) paths, AND
# the one-time build (loader regrid) already dominates a short run — so a small t_end
# is deliberate. The build is t_end-independent; a longer window only adds LSODA steps.
# Pass a larger t_end as argv[1] to sample more of the solve.
#
# Requires a network connection (3DEP + LANDFIRE, no auth) and a Copernicus CDS key in
# ~/.cdsapirc (ERA5) on the first run; subsequent runs read the warmed cache.
#
# Env: reuses run-model-py/.venv (numpy/scipy + requests/certifi/tifffile/xarray/
# netCDF4 + pyinstrument), provisioned by profile-py.sh. Set ESS_ROOT / EIO_ROOT to
# override the sibling checkouts.
#
# Usage: python profile-py.py [t_end]    (default t_end=30 s)

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ESS_ROOT = os.environ.get(
    "ESS_ROOT", os.path.normpath(os.path.join(HERE, "..", "EarthSciSerialization"))
)
if not os.path.isfile(os.path.join(ESS_ROOT, "esm-schema.json")):
    sys.exit(
        f"EarthSciSerialization not found at '{ESS_ROOT}'; set ESS_ROOT or clone "
        "it as a sibling checkout of this repo."
    )
EIO_ROOT = os.environ.get(
    "EIO_ROOT", os.path.normpath(os.path.join(HERE, "..", "EarthSciIO"))
)
if not os.path.isfile(os.path.join(EIO_ROOT, "earthsciio", "__init__.py")):
    sys.exit(f"EarthSciIO not found at '{EIO_ROOT}'; set EIO_ROOT or clone it as a sibling.")
sys.path.insert(0, os.path.join(ESS_ROOT, "packages", "earthsci_toolkit", "src"))
sys.path.insert(0, EIO_ROOT)

import earthsci_toolkit as esm  # noqa: E402
import earthsciio as esio  # noqa: E402
from earthsciio import era5 as _era5  # noqa: E402
from earthsciio.backends.cds import encode_cds_url as _enc  # noqa: E402
from pyinstrument import Profiler  # noqa: E402
from pyinstrument.renderers import SpeedscopeRenderer  # noqa: E402

t_end = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
outdir = os.path.join(HERE, "profiles")
os.makedirs(outdir, exist_ok=True)
model_path = os.path.join(HERE, "wildlandfire.esm")

# --- Grid + domain, identical to the runner (run-model.py). ------------------
NX, NY = 18, 20
LX, LY = 36000.0, 40000.0
BBOX_W, BBOX_S, BBOX_E, BBOX_N = -121.7400, 39.6049, -121.3192, 39.9651
BBOX = f"{BBOX_W},{BBOX_S},{BBOX_E},{BBOX_N}"
GX, GY = 36, 40
ERA5_AREA = [40, -122, 39, -121]
ERA5_DATE, ERA5_HOUR = (2018, 11, 8), "14:00"

# --- The three real data-loader providers (cached after the first run). ------
esio.register_format_readers()
cache = esio.Cache(auth={"cds": esio.cds_auth()})
terrain_url = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/"
    f"exportImage?bbox={BBOX}&size={GX},{GY}&format=tiff&bboxSR=4326&imageSR=4326"
    "&pixelType=F32&interpolation=+RSP_BilinearInterpolation&f=image"
)
landfire_url = (
    "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2022/LF2022_FBFM13_CONUS/"
    f"ImageServer/exportImage?bbox={BBOX}&bboxSR=4326&imageSR=4326&size={GX},{GY}"
    "&format=tiff&pixelType=S16&interpolation=+RSP_NearestNeighbor&f=image"
)
_req = _era5.era5_request(ERA5_DATE[0], ERA5_DATE[1], [ERA5_DATE[2]],
    ["temperature", "u_component_of_wind", "v_component_of_wind", "relative_humidity"],
    [1000], ERA5_AREA)
_req["time"] = [ERA5_HOUR]
era5_url = _enc(_era5.ERA5_DATASET, _req)


def era5_prov(short):
    return esio.Provider(esio.DataLoader(name="ERA5", format="netcdf", url=era5_url,
        variables=[short], temporal=None, auth_realm="cds"), cache)


PROVIDERS = {
    "USGS3DEP.raw.elevation": esio.Provider(esio.DataLoader(
        name="USGS3DEP", format="geotiff", url=terrain_url, variables=["elevation"],
        temporal=None, reader_kwargs={"band_names": ["elevation"]}), cache),
    "LANDFIRE.raw.fuel_model": esio.Provider(esio.DataLoader(
        name="LANDFIRE", format="geotiff", url=landfire_url, variables=["fuel_model"],
        temporal=None, reader_kwargs={"band_names": ["fuel_model"]}), cache),
    "ERA5.pl.t": era5_prov("t"), "ERA5.pl.u": era5_prov("u"),
    "ERA5.pl.v": era5_prov("v"), "ERA5.pl.r": era5_prov("r"),
}
_mlon, _mlat = LX / (BBOX_E - BBOX_W), LY / (BBOX_N - BBOX_S)
PARAMS = {
    "LevelSetFireSpread.Lx": LX, "LevelSetFireSpread.Ly": LY,
    "Era5Regrid.src_x0": (ERA5_AREA[1] - 0.125 - BBOX_W) * _mlon,
    "Era5Regrid.src_dx": 0.25 * _mlon,
    "Era5Regrid.src_y0": (ERA5_AREA[2] - 0.125 - BBOX_S) * _mlat,
    "Era5Regrid.src_dy": 0.25 * _mlat,
}


def simulate(file, tend):
    return esm.simulate(file, (0.0, tend), parameters=PARAMS, providers=PROVIDERS,
                        method="LSODA", rtol=1e-2, atol=1e-3)


print(f"loading    wildlandfire.esm  (NX={NX}, NY={NY})")

# Warm up the toolkit import + parse on a throwaway load (sheds one-time first-touch
# cost). We do NOT pre-run simulate() here — the build-time conservative regrid is the
# workload we want IN the profile, not shed by a warmup, and the loader fetches warm
# the cache on this first bind anyway.
print("warming up (import + parse + loader fetch) … ", end="", flush=True)
tw = time.perf_counter()
_ = esm.load(model_path, metaparameters={"NX": NX, "NY": NY})
for p in PROVIDERS.values():
    p.materialize()   # warm the cache (network) so the profiled run reads it locally
tw = time.perf_counter() - tw
print(f"done in {tw:.1f}s")

file = esm.load(model_path, metaparameters={"NX": NX, "NY": NY})

print(f"profiling simulate (0.0, {t_end}) with LSODA … ", end="", flush=True)
profiler = Profiler(interval=0.001)  # 1 ms sampling
t0 = time.perf_counter()
profiler.start()
sim = simulate(file, t_end)
profiler.stop()
tp = time.perf_counter() - t0
print(f"done in {tp:.2f}s (success={sim.success})")
if not sim.success:
    print(f"  WARNING: solver reported failure: {sim.message}", file=sys.stderr)

session = profiler.last_session
n_samples = getattr(session, "sample_count", "?")
header = (
    f"# NX={NX} NY={NY}  t_end={t_end}  wall={tp:.2f}s  solver=LSODA (scipy)\n"
    f"# three real loaders (3DEP + LANDFIRE + ERA5), regridded in-model\n"
    f"# pyinstrument statistical sampler @ 1 ms; {n_samples} samples; "
    f"CPU time {session.cpu_time:.2f}s\n"
)

# --- Call tree (self + total per frame) — the rich view. ---------------------
with open(os.path.join(outdir, "profile-python-tree.txt"), "w") as f:
    f.write("# Python simulate() call tree (pyinstrument; self + total time)\n")
    f.write(header + "\n")
    f.write(profiler.output_text(unicode=True, color=False, show_all=True))
print("wrote profiles/profile-python-tree.txt")

# --- Self-time by function — the analog of the Rust/Julia -top views. ---------
# pyinstrument models a frame's *self* time as a synthetic `[self]` child frame whose
# `.time` is that self time (a frame's own `.time` is total = self + children). So the
# per-function flat self-time is: attribute each `[self]` child's time to its PARENT
# function, keyed by function + file:line (summing `total_self_time` over all frames
# double-counts to ~2× CPU time — the `[self]`-to-parent attribution sums to CPU time).
self_by_func = {}
total = 0.0


def _walk(frame):
    global total
    if frame is None:
        return
    if frame.function == "[self]" and frame.parent is not None:
        p = frame.parent
        fpath = getattr(p, "file_path", None) or getattr(p, "file_path_short", None)
        where = f" ({os.path.basename(fpath)}:{p.line_no})" if fpath else ""
        key = f"{p.function}{where}"
        self_by_func[key] = self_by_func.get(key, 0.0) + frame.time
        total += frame.time
    for child in frame.children:
        _walk(child)


try:
    _walk(session.root_frame())
except Exception as e:  # pragma: no cover — defensive across pyinstrument versions
    print(f"  (frame-tree walk unavailable: {e}; tree/flame views still written)")

if self_by_func:
    ranked = sorted(self_by_func.items(), key=lambda kv: kv[1], reverse=True)
    with open(os.path.join(outdir, "profile-python-top.txt"), "w") as f:
        f.write("# Python simulate() self-time by function\n")
        f.write(header)
        f.write("#  self%   self(s)  function\n")
        for name, secs in ranked[:40]:
            pct = 100.0 * secs / total if total else 0.0
            f.write(f"{pct:7.2f}  {secs:8.3f}  {name}\n")
    print("wrote profiles/profile-python-top.txt")

# --- Flame graph: speedscope JSON (compact + interactive). -------------------
speedscope_path = os.path.join(outdir, "profile-python.speedscope.json")
with open(speedscope_path, "w") as f:
    f.write(profiler.output(SpeedscopeRenderer()))
print(f"wrote {speedscope_path}  (flame graph — open at https://speedscope.app)")

print("PROFILE-PY DONE")
