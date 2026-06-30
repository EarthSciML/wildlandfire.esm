#!/usr/bin/env julia
#=
Phase 5 residual #5 — the FULL data flow: REAL ERA5 data drives the monolithic
camp_fire through the computed conservative regridder, refreshed at cadence.

    EarthSciIO.discrete_provider (corpus ERA5 netcdf, OFFLINE, 2 time records)
      --to_esio_loader/esio_provider--> ESS Provider protocol (provider_sample)
      --ComputedRegrid (the REAL conservative_regrid_overlap_join_computed weights)-->
        the live `ERA5tRegrid.F_tgt` forcing buffer the fire RHS reads
      --ESS.simulate(...; providers, regrid)--> build_refresh_callback + Tsit5 solve

So the fire's per-cell Rothermel rate of spread is computed from REAL ERA5 2-m
temperature, regridded onto the 19×21 fire grid by the same spherical-clip /
partition-of-unity weights the keystone uses, and that forcing REFRESHES in place
when the solver crosses the ERA5 cadence tick. The conformance-corpus ERA5 sub-tile
(lat 40.0/39.5/39.0, lon −122/−121.5/−121) is a real 0.5° grid OVER the Camp Fire
domain, so the regrid is geographic, not a repositioning. (The real monthly Camp
Fire ERA5 file rides the identical provider→regrid→refresh path; the corpus keeps
this check offline + fast.) The representative W→E wind ramp supplies the front
anisotropy; ERA5 temperature is the data-driven, cadence-refreshed field.

Run with plain `julia` (the support include activates the ESS pde_sim_adapter env;
this script augments it with the EarthSciIO + DiffEqCallbacks data-refresh stack).
=#
include(joinpath(dirname(@__DIR__), "run_camp_fire_e2e_pde.jl"))   # → camp_fire_support.jl (env + fire + ComputedRegrid)
const E = ESS
const WLROOT = dirname(dirname(@__DIR__))
const ESIO_ROOT = get(ENV, "EARTHSCIIO", normpath(joinpath(WLROOT, "..", "EarthSciIO")))
const ESIO_PKG = joinpath(ESIO_ROOT, "julia")
const CORPUS = get(ENV, "EARTHSCIIO_CORPUS", joinpath(ESIO_ROOT, "conformance", "corpus"))

# Augment the pde_sim_adapter env with the data-refresh stack (idempotent): EarthSciIO
# (the provider + readers) + DiffEqCallbacks/SciMLBase (the build_refresh_callback ext).
let deps = Pkg.project().dependencies
    haskey(deps, "EarthSciIO") || Pkg.develop(path = ESIO_PKG; io = devnull)
    adds = filter(p -> !haskey(deps, p), ["DiffEqCallbacks", "SciMLBase"])
    isempty(adds) || Pkg.add(adds; io = devnull)
    (haskey(deps, "EarthSciIO") && isempty(adds)) || Pkg.instantiate(; io = devnull)
end
import EarthSciIO
using DiffEqCallbacks            # triggers EarthSciSerializationDataRefreshExt (build_refresh_callback)
import SciMLBase
import OrdinaryDiffEqTsit5: Tsit5
include(joinpath(dirname(@__DIR__), "esio_ess_bridge.jl"))

"Source-cell polygons (lon/lat) + representative non-temp fields for the corpus ERA5
grid (`lats`×`lons` cell centers, `dll`° cells), indexed s=(loncol-1)*nlat+latrow to
match `vec` of a [lat,lon] provider sample. wu ramps W→E (anisotropy); rest constant."
function corpus_source(lats, lons; dll = 0.5)
    nlat = length(lats); nlon = length(lons); ns = nlat * nlon
    P = zeros(ns, 4, 2); slon = zeros(ns); slat = zeros(ns)
    u_col = collect(range(2.0, 10.0; length = nlon))
    field = Dict(k => zeros(ns) for k in ("temp", "rh", "wu", "wv", "sx", "sy"))
    for (lc, lo) in enumerate(lons), (lr, la) in enumerate(lats)
        s = (lc - 1) * nlat + lr
        for (k, (a, b)) in enumerate(((lo - dll/2, la - dll/2), (lo + dll/2, la - dll/2),
                                      (lo + dll/2, la + dll/2), (lo - dll/2, la + dll/2)))
            P[s, k, 1] = a; P[s, k, 2] = b
        end
        slon[s] = lo; slat[s] = la
        field["temp"][s] = 288.0; field["rh"][s] = 0.23
        field["wu"][s] = u_col[lc]; field["sx"][s] = 0.05
    end
    return P, slon, slat, field
end

function main()
    dom = camp_domain(CAMP_FIRE_ESM)
    nx, ny = dom.nx, dom.ny
    println("Phase 5 #5 — REAL ERA5 (offline corpus) → computed regrid → cadence-refreshed camp_fire\n")

    store = EarthSciIO.LocalStore(joinpath(CORPUS, "cache"))
    cache = EarthSciIO.Cache(store; offline = true, verify = true)
    era5 = "https://data.earthsci.dev/era5/2018/11/20181108.nc"
    full = EarthSciIO.materialize(esio_provider((; url = era5, format = "netcdf"); cache))
    lats = Float64.(full["latitude"].data); lons = Float64.(full["longitude"].data)
    times = Float64.(full["time"].data)
    println("  corpus ERA5 grid: lat ", lats, "  lon ", lons, "  cadence ", times)
    prov = esio_provider((; url = era5, format = "netcdf", times, time_dim = "time"); cache)

    # The REAL computed conservative regridder, weights for the corpus ERA5 → fire grid.
    src_poly, src_lon, src_lat, field = corpus_source(lats, lons)
    tgt_poly, tgt_lon, tgt_lat = fire_target_lonlat(dom)
    W = computed_regrid_weights(src_poly, src_lon, src_lat, tgt_poly, tgt_lon, tgt_lat)
    @assert all(rs -> isapprox(rs, 1.0; atol = 1e-9), vec(sum(W; dims = 2))) "W must be a partition of unity"

    # The 6 forcing buffers (param_arrays). Temp is data-driven; the rest representative.
    disc = fire_discretized(dom)
    forcing_of = Dict(_MONO_FORCING_OF)
    forcing = Dict{String,Any}(forcing_of[fn] => reshape(W * field[fn], nx, ny)
                               for fn in keys(forcing_of))
    tname = forcing_of["temp"]                       # "ERA5tRegrid.F_tgt"
    s0 = E.provider_sample(prov, times[1])
    forcing[tname] = reshape(W * vec(Float64.(s0["t2m"])), nx, ny)   # seed [0,tick0) from record 0

    regrid = ComputedRegrid(Dict(tname => W), Dict(tname => "t2m"), (nx, ny))
    t2m0 = reshape(W * vec(Float64.(s0["t2m"])), nx, ny)
    s1 = E.provider_sample(prov, times[2])
    finite1 = [isfinite(x) ? x : sum(filter(isfinite, vec(s1["t2m"]))) / count(isfinite, vec(s1["t2m"]))
               for x in vec(Float64.(s1["t2m"]))]
    t2m1 = reshape(W * finite1, nx, ny)
    println("  regridded ERA5 temp on the fire grid: record0 ∈ [", round(minimum(t2m0); digits = 3), ", ",
            round(maximum(t2m0); digits = 3), "]  record1 ∈ [", round(minimum(t2m1); digits = 3), ", ",
            round(maximum(t2m1); digits = 3), "]  (real, refreshing)")
    @assert 282.0 <= minimum(t2m0) && maximum(t2m0) <= 285.0 "regridded temp must lie in the real ERA5 range"
    @assert !isapprox(t2m0, t2m1) "the two ERA5 records must differ (a genuine refresh)"

    T = 25.0                                         # tick at t=1, then spread under the refreshed real-data forcing
    r = simulate(disc, (0.0, T); alg = Tsit5(),
                 param_arrays = forcing, seed_ic! = psi_seeder(dom),
                 providers = Dict(tname => prov), regrid = regrid,
                 parameters = Dict("LevelSetFireSpread.dx" => dom.dx),
                 reltol = 1e-4, abstol = 1e-6)
    @assert r.success "the data-driven fire must integrate (retcode=$(r.retcode))"

    psi(u, i, j) = (k = get(r.var_map, "LevelSetFireSpread.psi[$i,$j]", nothing); k === nothing ? NaN : u[k])
    burned(u) = count(((i, j),) -> (v = psi(u, i, j); !isnan(v) && v < 0),
                      [(i, j) for i in 1:nx for j in 1:ny])
    refreshed = forcing[tname]                       # simulate aliases the buffer; holds the last tick
    println("  solved: $(length(r.t)) steps over $(T)s, retcode=$(r.retcode)")
    println("  forcing buffer after solve == regridded ERA5 record1: ", isapprox(refreshed, t2m1; atol = 1e-6))
    println("  burned (psi<0) cells: ", burned(r.u[1]), " → ", burned(r.u[end]))
    @assert isapprox(refreshed, t2m1; atol = 1e-6) "the ERA5 forcing buffer must refresh in place to record1 at the cadence tick"
    @assert burned(r.u[end]) >= burned(r.u[1]) "the burned region must not shrink"

    println("\n  ✓ a real EarthSciIO ERA5 Provider drives the camp_fire forcing through the computed")
    println("    conservative regridder, refreshing in place at the ERA5 cadence — the fire's per-cell")
    println("    Rothermel rate is computed from real, cadence-refreshed data in ONE ESS.simulate call.")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
