#!/usr/bin/env julia
# run-model.jl — minimal single-model runner for a 2-D EarthSci `.esm` (Julia).
#
# Model: 2-D level-set Hamilton-Jacobi front propagation  psi_t = -S |grad psi|
# on the doubly-periodic Camp-Fire box, with |grad psi| discretized by the Godunov
# (Rouy-Tourin / Osher-Sethian) upwind scheme (imported by reference from the
# sibling earthscidiscretizations checkout — nothing is vendored here). psi is
# initialized as the signed distance to a circle, so the front {psi = 0} expands
# outward at a Rothermel + NFDRS spread rate S(n̂). The spread is driven by THREE
# real, per-cell data loaders, each conservatively regridded onto the fire grid
# INSIDE the .esm (esm-spec §8.6): USGS 3DEP terrain elevation (live ImageServer
# GeoTIFF, no auth) → per-cell dz/dx, dz/dy; LANDFIRE FBFM13 fuel model (live
# ImageServer GeoTIFF, no auth) → per-cell fuel code → the Anderson-13 fuel
# properties; and ERA5 1000 hPa surface met (t, u, v, r) from the Copernicus CDS
# API (~/.cdsapirc auth) → per-cell temperature, humidity and 10 m wind. This runner
# supplies them by binding EarthSciIO providers to USGS3DEP.raw.elevation,
# LANDFIRE.raw.fuel_model and ERA5.pl.{t,u,v,r} (plus the ERA5 source geometry, its
# 0.25° cells placed in the fire metre-frame so the regrid generalizes to a moved/
# larger domain). Loads through EarthSciAST.jl, integrates while saving
# several snapshots, and plots the front over the terrain with the ERA5 wind (PNG).
#
# Usage:
#   julia run-model.jl [model.esm] [t_end]
#     model.esm   path to the .esm to run   (default: ./wildlandfire.esm)
#     t_end       final integration time    (default: 57600 s = 16 h)
#
# Requires a network connection (3DEP + LANDFIRE ImageServers, no auth) and a
# Copernicus CDS key in ~/.cdsapirc (the ERA5 loader fetches via the CDS API).
#
# Environment: a local project (run-model-jl/) layering Plots.jl + EarthSciIO +
# TiffImages on top of the dev'd EarthSciAST + OrdinaryDiffEqTsit5. Set
# ESS_ROOT / EIO_ROOT to override the sibling-checkout locations.

import Pkg

const HERE = @__DIR__
const ESS_ROOT = get(ENV, "ESS_ROOT",
                     normpath(joinpath(HERE, "..", "EarthSciAST")))
isfile(joinpath(ESS_ROOT, "esm-schema.json")) ||
    error("EarthSciAST not found at '$ESS_ROOT'; set ESS_ROOT or " *
          "clone it as a sibling checkout of this repo.")
const EIO_ROOT = get(ENV, "EIO_ROOT",
                     normpath(joinpath(HERE, "..", "EarthSciIO", "julia")))
isfile(joinpath(EIO_ROOT, "Project.toml")) ||
    error("EarthSciIO (Julia package) not found at '$EIO_ROOT'; set EIO_ROOT or " *
          "clone EarthSciIO as a sibling checkout (needed for the terrain loader).")

let env = joinpath(HERE, "run-model-jl")
    mkpath(env)
    Pkg.activate(env; io=devnull)
    if !isfile(joinpath(env, "Manifest.toml"))
        Pkg.develop(path=joinpath(ESS_ROOT, "pkg", "EarthSciAST.jl"); io=devnull)
        Pkg.develop(path=EIO_ROOT; io=devnull)
        # DiffEqCallbacks: the DISCRETE-loader refresh callback (PresetTimeCallback that
        # re-samples the time-varying ERA5 met each hour) lives in ESS's
        # EarthSciASTDataRefreshExt, gated on DiffEqCallbacks + SciMLBase.
        Pkg.add(["OrdinaryDiffEqTsit5", "Plots", "TiffImages", "DiffEqCallbacks"]; io=devnull)
    end
    Pkg.instantiate(; io=devnull)
end

using EarthSciAST
import OrdinaryDiffEqTsit5
import DiffEqCallbacks             # triggers ESS's DataRefreshExt (discrete-loader refresh)
using EarthSciIO, TiffImages        # EarthSciIO provides the ESS provider seam via
using Printf                        # EarthSciASTEarthSciIOExt (loaded on `using`);
using Plots                         # TiffImages backs EarthSciIO's geotiff reader.
const ESS = EarthSciAST

const NT = 6        # number of front snapshots (t = 0 … t_end)
const NX = 18       # grid cells along x; dx = LX/NX = 2000 m (matches the archive
const NY = 20       # grid cells along y; dy = LY/NY = 2000 m   camp_fire_model.jl grid)
const LX = 36000.0  # domain width  (m): Camp-Fire extent (half-width 18 km)
const LY = 40000.0  # domain height (m): Camp-Fire extent (half-width 20 km)

# USGS 3DEP source raster (Paradise, CA / Camp Fire): a single-band F32 elevation
# GeoTIFF from the live no-auth USGS ImageServer. GX×GY must match TerrainRegrid's
# conservative_overlap NSRC and its `src` cartesian_cell_rings GX/GY (36×40 = 1440).
const GX   = 36          # 3DEP + LANDFIRE raster width  (must match TerrainRegrid src 36×40)
const GY   = 40          # 3DEP + LANDFIRE raster height
# Camp-Fire domain: the 36×40 km box centered on the Pulga–Paradise midpoint,
# matching archive/camp_fire_model.jl.
const BBOX_W, BBOX_S, BBOX_E, BBOX_N = -121.7400, 39.6049, -121.3192, 39.9651
const BBOX = "$BBOX_W,$BBOX_S,$BBOX_E,$BBOX_N"
# Landmarks marked on the plot (WGS84 lon/lat): the ignition at Pulga (the real
# Camp Fire origin) and the town of Paradise.
const PULGA_LONLAT    = (-121.437222, 39.810278)   # 39°48′37″N 121°26′14″W (ignition)
const PARADISE_LONLAT = (-121.621944, 39.759722)   # 39°45′35″N 121°37′19″W

# ERA5 pressure-level (1000 hPa surface) CDS request: a small 0.25° lon/lat block
# around the fire (ERA5_NX×ERA5_NY must match Era5Regrid's ov5 NSRC / e5s GX,GY =
# 5×5 = 25). The met is TIME-VARYING — a DISCRETE loader at ERA5's native HOURLY
# cadence — so it refreshes as the fire evolves (winds/humidity change hour to
# hour). No forcing scalars are set here any more — fuel comes from LANDFIRE,
# temperature / humidity / wind from ERA5, terrain from 3DEP, all per-cell through
# the in-model regridders.
const ERA5_AREA = [40, -122, 39, -121]   # [N,W,S,E] — 1° box → 5×5 cells @ 0.25°
const ERA5_NX, ERA5_NY = 5, 5
const ERA5_YEAR, ERA5_MONTH = 2018, 11
const ERA5_DAYS  = [8, 9]                # ignition day + next (the 16 h window crosses UTC midnight)
const ERA5_HOURS = 0:23                  # native HOURLY cadence → 48 records over the two days
const ERA5_IGNITION_REC = 14             # 0-based valid_time index of 14:00 UTC day-8 (≈06 PST ignition)

model_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(HERE, "wildlandfire.esm")
t_end      = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 57600.0   # 16 h

# Fetch the three loaders the model declares. CONST providers cache each source
# once; ESS folds them into the const arrays and the in-model TerrainRegrid /
# Era5Regrid conservatively regrid them onto the fire grid. The CDS ERA5 fetch
# authenticates via ~/.cdsapirc (the cds transport's fallback).
cache = EarthSciIO.Cache()
terrain_url =
    "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/" *
    "exportImage?bbox=$BBOX&size=$GX,$GY&format=tiff&bboxSR=4326&imageSR=4326" *
    "&pixelType=F32&interpolation=+RSP_BilinearInterpolation&f=image"
landfire_url =
    "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2022/LF2022_FBFM13_CONUS/" *
    "ImageServer/exportImage?bbox=$BBOX&bboxSR=4326&imageSR=4326&size=$GX,$GY" *
    "&format=tiff&pixelType=S16&interpolation=+RSP_NearestNeighbor&f=image"
# ERA5 surface met via CDS: one multi-hour file (1000 hPa t,u,v,r over the two
# days at hourly cadence → 48 valid_time records). `era5_times` maps each record to
# solver seconds with the ignition hour (record ERA5_IGNITION_REC) at t=0; earlier
# records get negative times (never reached), later ones run past t_end — the solver
# only fires the cadence ticks that land in [0, t_end].
era5_url = EarthSciIO.era5_pressure_url(ERA5_YEAR, ERA5_MONTH;
    variables = ["relative_humidity", "temperature",
                 "u_component_of_wind", "v_component_of_wind"],
    pressure_levels = [1000], days = ERA5_DAYS, times = ERA5_HOURS, area = ERA5_AREA)
const ERA5_NREC  = length(ERA5_DAYS) * length(ERA5_HOURS)                 # 48
const era5_times = [(k - ERA5_IGNITION_REC) * 3600.0 for k in 0:(ERA5_NREC - 1)]
println("fetching   3DEP + LANDFIRE ($(GX)×$(GY)) and ERA5 met ($(ERA5_NX)×$(ERA5_NY), " *
        "$(ERA5_NREC) hourly records) for $BBOX …")
providers = try
    # DISCRETE provider: one file, sliced to the valid_time record active at each
    # solver tick (time_dim="valid_time"). refresh_times() = era5_times drives the
    # solver's cadence callback, which refreshes the ERA5 forcing (and, via the
    # discrete materializer, the Era5Regrid→behavior stack over it) at each hour.
    # records_per_sample=2: the provider returns the TWO records bracketing each
    # solver time (a size-2 valid_time axis), and the model linearly interpolates
    # between them — a smooth ramp instead of the old piecewise-constant hourly step.
    era5p(v) = EarthSciIO.discrete_provider(cache, era5_url, era5_times;
                   format="netcdf", variables=[v], auth_realm="cds",
                   time_dim="valid_time", records_per_sample=2)
    Dict("USGS3DEP.raw.elevation" =>
             EarthSciIO.const_provider(cache, terrain_url;
                 format="geotiff", reader_kwargs=(band_names=["elevation"],)),
         "LANDFIRE.raw.fuel_model" =>
             EarthSciIO.const_provider(cache, landfire_url;
                 format="geotiff", reader_kwargs=(band_names=["fuel_model"],)),
         "ERA5.pl.t" => era5p("t"), "ERA5.pl.u" => era5p("u"),
         "ERA5.pl.v" => era5p("v"), "ERA5.pl.r" => era5p("r"))
catch err
    error("could not build the terrain/fuel/met providers for $BBOX\n" *
          "the model requires providers for USGS3DEP.raw.elevation, " *
          "LANDFIRE.raw.fuel_model and ERA5.pl.{t,u,v,r} (network + ~/.cdsapirc needed).\n" *
          "underlying error: $err")
end

# Confirm the ERA5 met is time-INTERPOLATED: at each solver time the provider now
# returns the TWO records bracketing it; report both bracket means at t=0 and
# t=t_end (mean over the 5×5 block). This also warms the CDS fetch before the solve.
let pt = providers["ERA5.pl.t"]
    brmeans(t) = begin
        f = EarthSciIO.refresh(pt, t).variables["t"]
        pos = findfirst(==("valid_time"), f.dims)
        r0 = selectdim(f.data, pos, 1); r1 = selectdim(f.data, pos, 2)
        (Float64(sum(r0)) / length(r0), Float64(sum(r1)) / length(r1))
    end
    b0 = brmeans(0.0); be = brmeans(t_end)
    @printf("ERA5 met is time-INTERPOLATED (2-record bracket): mean 1000 hPa T bracket [%.2f, %.2f] K at t=0 -> [%.2f, %.2f] K at t=%.0f s (model blends each bracket linearly in time)\n",
            b0[1], b0[2], be[1], be[2], t_end)
end

# ERA5 src geometry: place the ERA5 sub-grid at its TRUE position in the fire
# metre-frame (equirectangular map of the ERA5 lon/lat against the fire bbox) so
# the conservative overlap generalizes to a moved/larger fire domain.
mlon = LX / (BBOX_E - BBOX_W); mlat = LY / (BBOX_N - BBOX_S)
era5_geom = Dict(
    "Era5Regrid.src_x0" => (ERA5_AREA[2] - 0.125 - BBOX_W) * mlon,   # west edge, westmost cell
    "Era5Regrid.src_dx" => 0.25 * mlon,
    "Era5Regrid.src_y0" => (ERA5_AREA[3] - 0.125 - BBOX_S) * mlat,   # south edge, southmost cell
    "Era5Regrid.src_dy" => 0.25 * mlat)

println("loading    $model_path  (NX=$NX, NY=$NY)")
file = ESS.load(model_path; metaparameters = Dict("NX" => NX, "NY" => NY))

ts = collect(range(0.0, t_end; length = NT))
println("simulating (0.0, $t_end) with Tsit5, $NT snapshots …")
# `inspect` captures the build-once geometry so we can read the terrain the model
# actually used (TerrainRegrid.elev_xy, the loader elevation conservatively
# regridded onto the fire [x,y] grid) straight out of the run — no second fetch.
insp = ESS.BuildInspection()
sim = ESS.simulate(file, (0.0, t_end);
                   alg=OrdinaryDiffEqTsit5.Tsit5(),
                   providers = providers,
                   parameters = merge(Dict("LevelSetFireSpread.Lx" => LX,
                                           "LevelSetFireSpread.Ly" => LY,
                                           # time-interp phase: t=0 is an ERA5
                                           # cadence anchor (ignition hour), hourly dt
                                           "ERA5.t_interp_ref" => 0.0,
                                           "ERA5.dt_interp" => 3600.0),
                                      era5_geom),
                   reltol=1e-2, abstol=1e-3, saveat=ts, inspect=insp)
sim.success || error("solver failed: retcode=$(sim.retcode) — $(sim.message)")

# Assemble the primary 2-D field: state elements "<stem>[i,j]", grouped by stem,
# indices parsed as integers.
groups = Dict{String,Vector{NTuple{3,Int}}}()   # stem => [(i, j, slot)]
for (name, slot) in sim.var_map
    m = match(r"^(.*)\[(\d+),(\d+)\]$", name)
    m === nothing && continue
    push!(get!(groups, m.captures[1], NTuple{3,Int}[]),
          (parse(Int, m.captures[2]), parse(Int, m.captures[3]), slot))
end
isempty(groups) && error("no 2-D indexed state field found to plot")
# Prefer the level-set state field psi (the mounted component also exposes many
# [x,y] observeds); fall back to the widest 2-D field if no psi is present.
let psikeys = filter(k -> endswith(k, "psi") || occursin(r"\bpsi$", k), collect(keys(groups)))
    global stem = isempty(psikeys) ? argmax(k -> length(groups[k]), keys(groups)) :
                  argmax(k -> length(groups[k]), psikeys)
end
cells = groups[stem]
nx = maximum(c -> c[1], cells)
ny = maximum(c -> c[2], cells)
dx = LX / nx
dy = LY / ny
xs = [(i - 0.5) * dx for i in 1:nx]
ys = [(j - 0.5) * dy for j in 1:ny]
# psi[k] is the nx×ny field at saved time sim.t[k].
psi = [fill(NaN, nx, ny) for _ in eachindex(sim.t)]
for (i, j, slot) in cells, k in eachindex(sim.t)
    psi[k][i, j] = sim.u[k][slot]
end

# Marching squares for the zero contour of a field sampled at (xs[i], ys[j]);
# returns line segments packed into (X, Y) with NaN breaks (one plot series).
function zero_segments(xs, ys, Z)
    X = Float64[]; Y = Float64[]
    interp(a, b) = (t = a[3] / (a[3] - b[3]); (a[1] + t*(b[1]-a[1]), a[2] + t*(b[2]-a[2])))
    for i in 1:length(xs)-1, j in 1:length(ys)-1
        c = ((xs[i],   ys[j],   Z[i, j]),   (xs[i+1], ys[j],   Z[i+1, j]),
             (xs[i+1], ys[j+1], Z[i+1, j+1]), (xs[i],  ys[j+1], Z[i, j+1]))  # CCW
        pts = Tuple{Float64,Float64}[]
        for e in 1:4
            a = c[e]; b = c[e % 4 + 1]
            (a[3] < 0) != (b[3] < 0) && push!(pts, interp(a, b))
        end
        if length(pts) >= 2
            push!(X, pts[1][1], pts[2][1], NaN)
            push!(Y, pts[1][2], pts[2][2], NaN)
        end
    end
    X, Y
end

cellarea = dx * dy
# Every behavior input is now a real, per-cell loader field, so summarise each
# regridded field's range + domain mean straight from the build inspection.
function field_summary(key, label, unit; scale = 1.0)
    haskey(insp.setup_arrays, key) || return nothing
    a = insp.setup_arrays[key] .* scale
    @sprintf("  %-20s %.2f–%.2f %s (mean %.2f)", label, minimum(a), maximum(a), unit, sum(a) / length(a))
end
println("\ncoupled loader forcing (per-cell, regridded onto the fire grid):")
for line in filter(!isnothing, [
        field_summary("TerrainRegrid.elev_xy", "3DEP elevation", "m"),
        field_summary("TerrainRegrid.fuel_xy", "LANDFIRE fuel code", ""),
        field_summary("Era5Regrid.t_xy", "ERA5 temperature", "K"),
        field_summary("Era5Regrid.rh_xy", "ERA5 rel humidity", "%"; scale = 100.0),
        field_summary("Era5Regrid.u_xy", "ERA5 u-wind", "m/s"),
        field_summary("Era5Regrid.v_xy", "ERA5 v-wind", "m/s")])
    println(line)
end
println("solver: success ($(sim.retcode)); field '$stem' is $(nx)×$(ny)")
for (k, t) in enumerate(sim.t)
    n = sum(psi[k][i, j] < 0 ? 1 : 0 for i in 1:nx, j in 1:ny)
    r = sqrt(n * cellarea / pi)
    cx = n == 0 ? NaN : sum(psi[k][i, j] < 0 ? xs[i] : 0.0 for i in 1:nx, j in 1:ny) / n
    @printf("  t=%7.0fs  r_eff=%8.1f m  x-centroid=%8.0f m (ignition x=25901, Pulga)\n", t, r, cx)
end

# --- single figure: real 3DEP terrain + front (psi=0) at each time; PNG output -
# The heatmap is the terrain the model ACTUALLY USED, read from the run itself:
# TerrainRegrid.elev_xy (the loader elevation conservatively regridded onto the
# fire [x,y] grid), pulled from the build inspection — not a separate download.
# The contours are the in-model fire front (color = time) on the same grid.
gr()
elev_xy = insp.setup_arrays["TerrainRegrid.elev_xy"]   # [x,y] = (nx,ny), fire grid
p = heatmap(xs, ys, elev_xy'; c = :terrain, alpha = 0.85,
    size = (680, 560), aspect_ratio = 1, xlims = (0, LX), ylims = (0, LY),
    xlabel = "x (m)", ylabel = "y (m)", framestyle = :box, legend = :outertopright,
    colorbar_title = "elevation (m)", legendtitle = "t",
    title = "$(basename(model_path)) — front {$stem=0}: 3DEP terrain + LANDFIRE fuel + ERA5 wind, Julia / Tsit5",
    titlefontsize = 8, guidefontsize = 10, legendfontsize = 8)
# ERA5 regridded 10 m wind (the met loader) as a light quiver overlay, so the plot
# shows all three loaders: 3DEP terrain (heatmap), the front, and the ERA5 wind.
let u_xy = get(insp.setup_arrays, "Era5Regrid.u_xy", nothing),
    v_xy = get(insp.setup_arrays, "Era5Regrid.v_xy", nothing)
    if u_xy !== nothing && v_xy !== nothing
        st = 2; sc = 1000.0   # metres of arrow per (m/s) of wind
        xg = [xs[i] for i in 1:st:nx for j in 1:st:ny]
        yg = [ys[j] for i in 1:st:nx for j in 1:st:ny]
        uu = [u_xy[i, j] * sc for i in 1:st:nx for j in 1:st:ny]
        vv = [v_xy[i, j] * sc for i in 1:st:nx for j in 1:st:ny]
        quiver!(p, xg, yg; quiver = (uu, vv), color = :gray30, alpha = 0.5, lw = 1, label = "")
    end
end
cg = cgrad(["#ffe08a", "#ff9d3c", "#e5391c", "#8b0000"])   # warm early→late over terrain
for (k, t) in enumerate(sim.t)
    X, Y = zero_segments(xs, ys, psi[k])
    frac = length(sim.t) == 1 ? 1.0 : (k - 1) / (length(sim.t) - 1)
    plot!(p, X, Y; color = cg[frac], lw = 2.5, label = @sprintf("%.1fh", t / 3600))
end

# Landmark markers: the ignition at Pulga + the town of Paradise, from lon/lat.
lonlat_xy(lon, lat) = ((lon - BBOX_W) / (BBOX_E - BBOX_W) * LX,
                       (lat - BBOX_S) / (BBOX_N - BBOX_S) * LY)
for (lonlat, name, mk, ms) in ((PULGA_LONLAT, "Pulga (ignition)", :star5, 9),
                               (PARADISE_LONLAT, "Paradise", :rect, 6))
    mx, my = lonlat_xy(lonlat...)
    scatter!(p, [mx], [my]; marker = mk, markersize = ms, color = :black,
             markerstrokecolor = :white, markerstrokewidth = 1, label = "")
    annotate!(p, mx + 900, my + 900, text(name, 8, :black, :left))
end

plot_path = joinpath(dirname(abspath(model_path)), "front_julia.png")
savefig(p, plot_path)
println("wrote plot: $plot_path")
