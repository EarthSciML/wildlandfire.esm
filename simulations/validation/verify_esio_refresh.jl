#!/usr/bin/env julia
#=
Phase 4 e2e — a REAL EarthSciIO data Provider drives the ESS cadence-refresh
machinery through esio_ess_bridge.jl. Offline, over EarthSciIO's committed
conformance corpus (a 3×3 ERA5 sub-tile with two time records), so no network.

It proves the data binding end to end:
  EarthSciIO.discrete_provider (corpus netcdf, offline)
    --bridge adapter-->  ESS.provider_refresh_times / provider_is_const / provider_sample
    --ESS.apply_regrid!--> the live param_arrays buffer the RHS reads
    --ESS.build_refresh_callback + solve--> the buffer refreshes at the cadence anchor.

The forcing is piecewise-constant per segment (one ERA5 record per tick), so the
integral is closed form: a state D(y)=t2m[1] over [0,1.5] with tick0 on [0,1) and
tick1 on [1,1.5] integrates to t2m0[1]·1 + t2m1[1]·0.5.

Self-bootstraps a temp env (dev ESS + EarthSciIO; add the Tsit5 + callback stack).
Set EARTHSCISERIALIZATION / EARTHSCIIO / EARTHSCIIO_CORPUS to override paths.
=#
import Pkg
const REPO = dirname(@__DIR__)                       # simulations/
const WL = dirname(REPO)
const ESS_ROOT = get(ENV, "EARTHSCISERIALIZATION", normpath(joinpath(WL, "..", "EarthSciSerialization")))
const ESS_PKG = joinpath(ESS_ROOT, "packages", "EarthSciSerialization.jl")
const ESIO_ROOT = get(ENV, "EARTHSCIIO", normpath(joinpath(WL, "..", "EarthSciIO")))
const ESIO_PKG = joinpath(ESIO_ROOT, "julia")
const CORPUS = get(ENV, "EARTHSCIIO_CORPUS", joinpath(ESIO_ROOT, "conformance", "corpus"))

let env = get(ENV, "ESIO_REFRESH_ENV", joinpath(REPO, ".esio_refresh_env"))
    mkpath(env)
    Pkg.activate(env; io = devnull)
    # Ensure each dep is present (NOT gated on Manifest existence): this env is shared
    # with verify_to_esio_loader.jl, whose lighter bootstrap may have created it WITHOUT
    # the solver/callback stack — gating on the Manifest would then skip these and the
    # `using DiffEqCallbacks` below would fail depending on run order.
    deps = Pkg.project().dependencies
    haskey(deps, "EarthSciSerialization") || Pkg.develop(Pkg.PackageSpec(path = ESS_PKG); io = devnull)
    haskey(deps, "EarthSciIO") || Pkg.develop(Pkg.PackageSpec(path = ESIO_PKG); io = devnull)
    add = filter(p -> !haskey(deps, p), ["DiffEqCallbacks", "SciMLBase", "OrdinaryDiffEqTsit5"])
    isempty(add) || Pkg.add(add; io = devnull)
    Pkg.instantiate(; io = devnull)
end

using EarthSciSerialization
import EarthSciIO
using DiffEqCallbacks                 # loads EarthSciSerializationDataRefreshExt
using SciMLBase
import OrdinaryDiffEqTsit5 as ODE
const ESS = EarthSciSerialization
include(joinpath(REPO, "esio_ess_bridge.jl"))

# Inline model: D(y) = t2m[1] — one state reading flat cell 1 of the live buffer.
_v(s) = ESS.VarExpr(String(s)); _i(x) = ESS.IntExpr(Int64(x))
_op(o, a...; k...) = ESS.OpExpr(String(o), ESS.Expr[a...]; k...)
_idx(v, is...) = _op("index", _v(v), is...)
_D(v) = _op("D", _v(v); wrt = "t")

function main()
    store = EarthSciIO.LocalStore(joinpath(CORPUS, "cache"))
    cache = EarthSciIO.Cache(store; offline = true, verify = true)
    era5 = "https://data.earthsci.dev/era5/2018/11/20181108.nc"

    # Cadence grid = the file's own (raw) time axis, then a per-tick-sliced provider.
    full = EarthSciIO.materialize(esio_provider((; url = era5, format = "netcdf"); cache))
    times = Float64.(full["time"].data)
    prov = esio_provider((; url = era5, format = "netcdf", times, time_dim = "time"); cache)

    println("Phase 4 — EarthSciIO provider → ESS refresh, offline over the corpus\n")
    println("  cadence times (from the ERA5 time axis): ", times)

    # --- adapter contract (the bridge fills ESS's protocol generics) ---
    @assert ESS.provider_refresh_times(prov) == times
    @assert ESS.provider_is_const(prov) == false
    @assert ESS.provider_is_const(esio_provider((; url = era5, format = "netcdf"); cache)) == true
    s0 = ESS.provider_sample(prov, times[1])
    s1 = ESS.provider_sample(prov, times[2])
    @assert haskey(s0, "t2m") && size(s0["t2m"]) == (3, 3)
    t2m0 = s0["t2m"][1, 1]; t2m1 = s1["t2m"][1, 1]
    println("  provider_sample t2m[1,1]: tick0=", t2m0, "  tick1=", t2m1)
    @assert t2m0 != t2m1 "the two ERA5 records must differ (a real refresh)"

    # --- e2e refresh through build_refresh_callback + solve ---
    buf = zeros(9)                                   # 3×3 t2m, flattened
    ESS.apply_regrid!(ESS.IdentityRegrid(), buf, "t2m", s0)   # seed the [0,1) segment
    @assert buf[1] == t2m0

    model = ESS.Model(Dict("y" => ESS.ModelVariable(ESS.StateVariable)),
                      [ESS.Equation(_D("y"), _idx("t2m", _i(1)))])
    f!, u0, p, _ts, vm = build_evaluator(model;
        initial_conditions = Dict("y" => 0.0), param_arrays = Dict("t2m" => buf))
    cb, tstops = build_refresh_callback(model;
        providers = Dict("t2m" => prov),
        buffers = RefreshBuffers(Dict("t2m" => buf)),
        regrid = ESS.IdentityRegrid())
    println("  build_refresh_callback tstops: ", tstops)

    prob = ODE.ODEProblem(f!, u0, (0.0, 1.5), p)
    sol = ODE.solve(prob, ODE.Tsit5(); callback = cb, tstops = tstops)
    @assert sol.retcode == ODE.SciMLBase.ReturnCode.Success

    y = sol.u[end][vm["y"]]
    expected = t2m0 * 1.0 + t2m1 * 0.5               # tick0 on [0,1), tick1 on [1,1.5]
    println("  ∫ D(y)=t2m[1] over [0,1.5] = ", round(y; digits = 6),
            "  (expected ", round(expected; digits = 6), ")")
    @assert isapprox(y, expected; atol = 1e-5) "refreshed forcing integral must be closed-form"
    @assert buf[1] == t2m1 "the buffer must hold the last cadence tick (refreshed in place)"

    println("\n  ✓ a real EarthSciIO Provider auto-loads + refreshes ESS forcing at cadence ",
            "(netcdf via the bridge adapter; geotiff/csv ride the same path).")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
