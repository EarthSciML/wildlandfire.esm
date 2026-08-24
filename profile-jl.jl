#!/usr/bin/env julia
# profile-jl.jl — CPU sampling profile of the Julia level-set build+solve.
#
# Uses PProf.jl (third-party; Google pprof) as the analysis tool on top of the
# stdlib sampling profiler. Loads and drives the SAME three-loader coupled model the
# runner uses (run-model.jl): the Rothermel + NFDRS behavior stack feeding the level-
# set front, with every behavior input a real, per-cell data-loader field regridded
# onto the fire grid IN the .esm — 3DEP terrain, LANDFIRE fuel and ERA5 met. It binds
# those loaders as providers exactly like the runner, so the profile reflects the real
# coupled workload (the build-time conservative regrid is now a major cost). Warms up
# once (so JIT compilation is NOT profiled), then samples a single solve at the
# 2×-refined NX=36, NY=40 fire grid (double the runner's 18×20). Emits, under ./profiles/:
#   flamegraph-julia.svg   — pprof flame graph (inclusive)
#   profile-julia-top.txt  — pprof -top (self time by function)
#   profile-julia-flat.txt — stdlib Profile flat list (cross-check, C frames on)
#   profile-julia-tree.txt — stdlib Profile call tree
#
# RUN SINGLE-THREADED:  julia --threads=1 profile-jl.jl [t_end]
# The level-set solve is serial, but Julia's sampling profiler snapshots EVERY thread
# each tick — so extra idle worker/GC threads park in `__psynch_cvwait` and swamp the
# one compute thread. One thread ⇒ the profile reflects actual solve compute.
#
# Requires a network connection (3DEP + LANDFIRE, no auth) and a Copernicus CDS key in
# ~/.cdsapirc (ERA5) on the first run; later runs read the warmed cache.
#
# Env: reuses a local project (profile-jl-env/) = dev'd ESS + EarthSciIO + solver +
# TiffImages + PProf. Set ESS_ROOT / EIO_ROOT to override the sibling checkouts.

import Pkg
const HERE = @__DIR__
const ESS_ROOT = get(ENV, "ESS_ROOT",
                     normpath(joinpath(HERE, "..", "EarthSciAST")))
isfile(joinpath(ESS_ROOT, "esm-schema.json")) ||
    error("EarthSciAST not found at '$ESS_ROOT'.")
const EIO_ROOT = get(ENV, "EIO_ROOT", normpath(joinpath(HERE, "..", "EarthSciIO", "julia")))
isfile(joinpath(EIO_ROOT, "Project.toml")) ||
    error("EarthSciIO (Julia) not found at '$EIO_ROOT'; needed for the data loaders.")

let env = joinpath(HERE, "profile-jl-env")
    mkpath(env)
    Pkg.activate(env; io=devnull)
    if !isfile(joinpath(env, "Manifest.toml"))   # build once; reuse thereafter (like run-model.jl)
        Pkg.develop(path=joinpath(ESS_ROOT, "pkg", "EarthSciAST.jl"); io=devnull)
        Pkg.develop(path=EIO_ROOT; io=devnull)
        # DiffEqCallbacks: triggers ESS's EarthSciASTDataRefreshExt so the DISCRETE
        # ERA5 provider's hourly refresh callback fires during the profiled solve
        # (matches run-model.jl; the old const-provider profile did not need it).
        # pprof_jll: the pprof CLI used below to render profile-julia-top.txt + the SVG.
        # SciMLBase explicit: phase 4 makes `solve` SciMLBase's own function.
        Pkg.add(["OrdinaryDiffEqTsit5", "TiffImages", "PProf", "DiffEqCallbacks",
                 "pprof_jll", "SciMLBase"]; io=devnull)
    end
    Pkg.instantiate(; io=devnull)
end

using EarthSciAST
import OrdinaryDiffEqTsit5
import SciMLBase
import DiffEqCallbacks           # triggers ESS's DataRefreshExt (discrete-loader refresh)
using EarthSciIO, TiffImages     # EarthSciIO provides the ESS provider seam + readers.
using Profile
import PProf
const ESS = EarthSciAST

if Threads.nthreads() > 1
    @warn "Running with $(Threads.nthreads()) threads; idle worker/GC threads park in " *
          "__psynch_cvwait and will dominate the sampling profile. Re-run with " *
          "`julia --threads=1 profile-jl.jl`."
end

const NX, NY = 36, 40    # 2×-refined fire grid (double the runner's 18×20)
t_end = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 3600.0   # 1 h (Julia is fast)
outdir = joinpath(HERE, "profiles"); mkpath(outdir)
alg = OrdinaryDiffEqTsit5.Tsit5()

# Camp-Fire domain + the three real data loaders, identical to run-model.jl. The
# build-time regrid is the workload; R_0 comes from Rothermel, driven by the loaders.
const LX, LY = 36000.0, 40000.0
const GX, GY = 36, 40
const BBOX_W, BBOX_S, BBOX_E, BBOX_N = -121.7400, 39.6049, -121.3192, 39.9651
const BBOX = "$BBOX_W,$BBOX_S,$BBOX_E,$BBOX_N"
const ERA5_AREA = [40, -122, 39, -121]   # [N,W,S,E] — 1° box → 5×5 cells @ 0.25°
const ERA5_YEAR, ERA5_MONTH = 2018, 11
const ERA5_DAYS = [8, 9]                 # ignition day + next (48 hourly records)
const ERA5_HOURS = 0:23                  # native HOURLY cadence → 48 records over 2 days
const ERA5_IGNITION_REC = 14             # 0-based valid_time index of 14:00 UTC day-8 (t=0)

cache = EarthSciIO.Cache()
terrain_url =
    "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/" *
    "exportImage?bbox=$BBOX&size=$GX,$GY&format=tiff&bboxSR=4326&imageSR=4326" *
    "&pixelType=F32&interpolation=+RSP_BilinearInterpolation&f=image"
landfire_url =
    "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2022/LF2022_FBFM13_CONUS/" *
    "ImageServer/exportImage?bbox=$BBOX&bboxSR=4326&imageSR=4326&size=$GX,$GY" *
    "&format=tiff&pixelType=S16&interpolation=+RSP_NearestNeighbor&f=image"
era5_url = EarthSciIO.era5_pressure_url(ERA5_YEAR, ERA5_MONTH;
    variables=["relative_humidity", "temperature", "u_component_of_wind", "v_component_of_wind"],
    pressure_levels=[1000], days=ERA5_DAYS, times=ERA5_HOURS, area=ERA5_AREA)
const ERA5_NREC  = length(ERA5_DAYS) * length(ERA5_HOURS)                 # 48
const era5_times = [(k - ERA5_IGNITION_REC) * 3600.0 for k in 0:(ERA5_NREC - 1)]
# DISCRETE, records_per_sample=2: the two records bracketing each solver time; the
# model blends them (ERA5.w_time). Identical to run-model.jl, so the profiled solve
# includes the per-tick refresh + time-blend (the .esm's era5_t axis is size 2).
era5p(v) = EarthSciIO.discrete_provider(cache, era5_url, era5_times;
    format="netcdf", variables=[v], auth_realm="cds",
    time_dim="valid_time", records_per_sample=2)
providers = Dict(
    "USGS3DEP.raw.elevation" => EarthSciIO.const_provider(cache, terrain_url;
        format="geotiff", reader_kwargs=(band_names=["elevation"],)),
    "LANDFIRE.raw.fuel_model" => EarthSciIO.const_provider(cache, landfire_url;
        format="geotiff", reader_kwargs=(band_names=["fuel_model"],)),
    "ERA5.pl.t" => era5p("t"), "ERA5.pl.u" => era5p("u"),
    "ERA5.pl.v" => era5p("v"), "ERA5.pl.r" => era5p("r"))
mlon = LX / (BBOX_E - BBOX_W); mlat = LY / (BBOX_N - BBOX_S)
params = Dict("LevelSetFireSpread.Lx" => LX, "LevelSetFireSpread.Ly" => LY,
    "Era5Regrid.src_x0" => (ERA5_AREA[2] - 0.125 - BBOX_W) * mlon,
    "Era5Regrid.src_dx" => 0.25 * mlon,
    "Era5Regrid.src_y0" => (ERA5_AREA[3] - 0.125 - BBOX_S) * mlat,
    "Era5Regrid.src_dy" => 0.25 * mlat,
    # time-interp phase: t=0 is an ERA5 cadence anchor (ignition hour); hourly dt.
    "ERA5.t_interp_ref" => 0.0, "ERA5.dt_interp" => 3600.0)

# phase 4: build + solve. Kept as ONE function on purpose — this script
# profiles the whole path (build-time regrid included), so splitting the two
# here would change what is being measured.
sim_once(tend) = SciMLBase.solve(
    ESS.esm_problem(file, (0.0, tend); providers=providers, p=params),
    alg; reltol=1e-2, abstol=1e-3)

println("loading    wildlandfire.esm  (NX=$NX, NY=$NY)")
file = ESS.load_path(joinpath(HERE, "wildlandfire.esm"); metaparameters=Dict("NX"=>NX, "NY"=>NY))

# Warm up: force JIT compilation of the whole build+solve path (incl. the build-time
# regrid) so the profile shows steady-state runtime, not one-shot compilation. This
# also warms the loader cache (network) so the profiled run reads it locally.
print("warming up (compiling + loader fetch) … "); flush(stdout)
tw = @elapsed sim_once(100.0)
println("done in $(round(tw; digits=1))s")

Profile.clear()
Profile.init(; n=10^8, delay=0.0005)   # 0.5 ms sampling, large buffer
print("profiling simulate (0.0, $t_end) … "); flush(stdout)
local sim
tp = @elapsed begin
    Profile.@profile sim = sim_once(t_end)
end
nsamp = length(Profile.fetch())
println("done in $(round(tp; digits=2))s  ($(sim.retcode); ~$(nsamp) raw buffer entries)")

# --- stdlib text views (guaranteed) ------------------------------------------
open(joinpath(outdir, "profile-julia-flat.txt"), "w") do io
    Profile.print(io; format=:flat, sortedby=:count, mincount=50, C=true)
end
open(joinpath(outdir, "profile-julia-tree.txt"), "w") do io
    Profile.print(io; format=:tree, maxdepth=40, mincount=50)
end
println("wrote stdlib text: profiles/profile-julia-flat.txt, profile-julia-tree.txt")

# --- PProf (third-party) flame graph + pprof -top ----------------------------
pb = joinpath(outdir, "profile-julia.pb.gz")
PProf.pprof(out=pb, web=false)
println("wrote pprof profile: $pb")
try
    import pprof_jll
    open(joinpath(outdir, "profile-julia-top.txt"), "w") do io
        run(pipeline(`$(pprof_jll.pprof()) -top -nodecount=45 $pb`; stdout=io, stderr=io))
    end
    run(`$(pprof_jll.pprof()) -svg -output=$(joinpath(outdir, "flamegraph-julia.svg")) $pb`)
    println("wrote pprof -top and flamegraph-julia.svg")
catch e
    @warn "pprof CLI step failed (pb.gz still usable via `pprof $pb`)" exception=e
end
println("PROFILE-JL DONE")
