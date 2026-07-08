#!/usr/bin/env julia
# profile-jl.jl — CPU sampling profile of the Julia level-set simulate().
#
# Uses PProf.jl (third-party; Google pprof) as the analysis tool on top of the
# stdlib sampling profiler. Loads and drives the SAME three-loader coupled model the
# runner uses (run-model.jl): the Rothermel + NFDRS behavior stack feeding the level-
# set front, with every behavior input a real, per-cell data-loader field regridded
# onto the fire grid IN the .esm — 3DEP terrain, LANDFIRE fuel and ERA5 met. It binds
# those loaders as providers exactly like the runner, so the profile reflects the real
# coupled workload (the build-time conservative regrid is now a major cost). Warms up
# once (so JIT compilation is NOT profiled), then samples a single simulate() at the
# runner's NX=18, NY=20 grid. Emits, under ./profiles/:
#   flamegraph-julia.svg   — pprof flame graph (inclusive)
#   profile-julia-top.txt  — pprof -top (self time by function)
#   profile-julia-flat.txt — stdlib Profile flat list (cross-check, C frames on)
#   profile-julia-tree.txt — stdlib Profile call tree
#
# RUN SINGLE-THREADED:  julia --threads=1 profile-jl.jl [t_end]
# The level-set solve is serial, but Julia's sampling profiler snapshots EVERY thread
# each tick — so extra idle worker/GC threads park in `__psynch_cvwait` and swamp the
# one compute thread. One thread ⇒ the profile reflects actual simulate() compute.
#
# Requires a network connection (3DEP + LANDFIRE, no auth) and a Copernicus CDS key in
# ~/.cdsapirc (ERA5) on the first run; later runs read the warmed cache.
#
# Env: reuses a local project (profile-jl-env/) = dev'd ESS + EarthSciIO + solver +
# TiffImages + PProf. Set ESS_ROOT / EIO_ROOT to override the sibling checkouts.

import Pkg
const HERE = @__DIR__
const ESS_ROOT = get(ENV, "ESS_ROOT",
                     normpath(joinpath(HERE, "..", "EarthSciSerialization")))
isfile(joinpath(ESS_ROOT, "esm-schema.json")) ||
    error("EarthSciSerialization not found at '$ESS_ROOT'.")
const EIO_ROOT = get(ENV, "EIO_ROOT", normpath(joinpath(HERE, "..", "EarthSciIO", "julia")))
isfile(joinpath(EIO_ROOT, "Project.toml")) ||
    error("EarthSciIO (Julia) not found at '$EIO_ROOT'; needed for the data loaders.")

let env = joinpath(HERE, "profile-jl-env")
    mkpath(env)
    Pkg.activate(env; io=devnull)
    Pkg.develop(path=joinpath(ESS_ROOT, "packages", "EarthSciSerialization.jl"); io=devnull)
    Pkg.develop(path=EIO_ROOT; io=devnull)
    Pkg.add(["OrdinaryDiffEqTsit5", "TiffImages", "PProf"]; io=devnull)
    Pkg.instantiate(; io=devnull)
end

using EarthSciSerialization
import OrdinaryDiffEqTsit5
using EarthSciIO, TiffImages     # EarthSciIO provides the ESS provider seam + readers.
using Profile
import PProf
const ESS = EarthSciSerialization

if Threads.nthreads() > 1
    @warn "Running with $(Threads.nthreads()) threads; idle worker/GC threads park in " *
          "__psynch_cvwait and will dominate the sampling profile. Re-run with " *
          "`julia --threads=1 profile-jl.jl`."
end

const NX, NY = 18, 20
t_end = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 3600.0   # 1 h (Julia is fast)
outdir = joinpath(HERE, "profiles"); mkpath(outdir)
alg = OrdinaryDiffEqTsit5.Tsit5()

# Camp-Fire domain + the three real data loaders, identical to run-model.jl. The
# build-time regrid is the workload; R_0 comes from Rothermel, driven by the loaders.
const LX, LY = 36000.0, 40000.0
const GX, GY = 36, 40
const BBOX_W, BBOX_S, BBOX_E, BBOX_N = -121.7400, 39.6049, -121.3192, 39.9651
const BBOX = "$BBOX_W,$BBOX_S,$BBOX_E,$BBOX_N"
const ERA5_AREA = [40, -122, 39, -121]
const ERA5_YEAR, ERA5_MONTH, ERA5_DAY, ERA5_HOUR = 2018, 11, 8, 14

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
    pressure_levels=[1000], days=[ERA5_DAY], times=[ERA5_HOUR], area=ERA5_AREA)
era5p(v) = EarthSciIO.const_provider(cache, era5_url; format="netcdf", variables=[v], auth_realm="cds")
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
    "Era5Regrid.src_dy" => 0.25 * mlat)

sim_once(tend) = ESS.simulate(file, (0.0, tend); alg=alg, providers=providers,
                              parameters=params, reltol=1e-2, abstol=1e-3)

println("loading    wildlandfire.esm  (NX=$NX, NY=$NY)")
file = ESS.load(joinpath(HERE, "wildlandfire.esm"); metaparameters=Dict("NX"=>NX, "NY"=>NY))

# Warm up: force JIT compilation of the whole simulate path (incl. the build-time
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
