#!/usr/bin/env julia
#=
Phase 5 — `to_esio_loader` unit check: a data-loader `.esm` block (source
url_template + temporal cadence + variables) maps to the esio_provider spec
(url / format / variables / time_dim / times), the Julia counterpart of the
Python `to_esio_loader` for the common netcdf URL-template case. Pure
dict→spec; no network.

Self-bootstraps a temp env (dev ESS + EarthSciIO). Set ESIO_REFRESH_ENV to
override; EARTHSCIMODELS to point at the real loader components.
=#
import Pkg
const REPO = dirname(@__DIR__)
const WL = dirname(REPO)
const ESS_ROOT = get(ENV, "EARTHSCISERIALIZATION", normpath(joinpath(WL, "..", "EarthSciSerialization")))
const ESS_PKG = joinpath(ESS_ROOT, "packages", "EarthSciSerialization.jl")
const ESIO_ROOT = get(ENV, "EARTHSCIIO", normpath(joinpath(WL, "..", "EarthSciIO")))
const ESIO_PKG = joinpath(ESIO_ROOT, "julia")
const ESM_MODELS = get(ENV, "EARTHSCIMODELS", normpath(joinpath(WL, "..", "EarthSciModels")))

let env = get(ENV, "ESIO_REFRESH_ENV", joinpath(REPO, ".esio_refresh_env"))
    mkpath(env)
    Pkg.activate(env; io = devnull)
    if !isfile(joinpath(env, "Manifest.toml"))
        Pkg.develop([Pkg.PackageSpec(path = ESS_PKG), Pkg.PackageSpec(path = ESIO_PKG)]; io = devnull)
    end
    haskey(Pkg.project().dependencies, "JSON3") || Pkg.add("JSON3"; io = devnull)
    Pkg.instantiate(; io = devnull)
end

import EarthSciSerialization, EarthSciIO
import Dates
import JSON3
include(joinpath(REPO, "esio_ess_bridge.jl"))

# An ERA5-pressure-level loader block, mirroring EarthSciModels
# components/earthsci_data/era5_loader.esm::data_loaders.ERA5_pl.
const ERA5_LOADER = Dict{String,Any}(
    "kind" => "pure_io",
    "source" => Dict{String,Any}("url_template" => "https://data.earthsci.dev/era5/era5_pl_{date:%Y}_{date:%m}.nc"),
    "temporal" => Dict{String,Any}("start" => "1940-01-01T00:00:00Z", "frequency" => "PT1H",
                                   "records_per_file" => 744, "time_variable" => "valid_time"),
    "variables" => Dict{String,Any}(
        "t" => Dict{String,Any}("file_variable" => "t", "units" => "K"),
        "u" => Dict{String,Any}("file_variable" => "u", "units" => "m/s"),
        "v" => Dict{String,Any}("file_variable" => "v", "units" => "m/s")))

function main()
    println("Phase 5 — to_esio_loader: data-loader .esm → esio_provider spec\n")

    # ERA5 netcdf, date-template URL resolved at an anchor + an explicit 2-tick cadence.
    spec = to_esio_loader(ERA5_LOADER; anchor = Dates.DateTime(2018, 11, 8), times = [0.0, 3600.0])
    println("  ERA5 spec: url=", spec.url)
    println("             format=", spec.format, "  variables=", spec.variables,
            "  time_dim=", spec.time_dim, "  times=", spec.times)
    @assert spec.url == "https://data.earthsci.dev/era5/era5_pl_2018_11.nc"
    @assert spec.format == "netcdf"
    @assert Set(spec.variables) == Set(["t", "u", "v"])
    @assert spec.time_dim == "valid_time"
    @assert spec.times == [0.0, 3600.0]
    # esio_provider must accept the spec (built offline; no fetch until materialize).
    prov = esio_provider(spec; cache = EarthSciIO.Cache(EarthSciIO.LocalStore(mktempdir()); offline = true))
    @assert EarthSciIO.is_const(prov) == false
    @assert EarthSciSerialization.provider_refresh_times(prov) == [0.0, 3600.0]
    println("  → esio_provider built a DISCRETE provider from the spec (refresh_times match)")

    # Cadence derived from temporal.frequency (PT1H) over a record count.
    spec2 = to_esio_loader(ERA5_LOADER; anchor = Dates.DateTime(2018, 11, 8), n_records = 3)
    @assert spec2.times == [0.0, 3600.0, 7200.0]   # PT1H × 3 records
    println("  cadence from temporal.frequency PT1H × 3 records: ", spec2.times)

    # GeoTIFF format inference (a LANDFIRE/USGS-style .tif URL).
    tif = Dict{String,Any}("source" => Dict{String,Any}("url_template" => "https://example.org/fuel.tif"),
                           "variables" => Dict{String,Any}("fuel" => Dict{String,Any}("file_variable" => "Band1")))
    gspec = to_esio_loader(tif)
    @assert gspec.format == "geotiff" && gspec.variables == ["Band1"] && gspec.times === nothing
    println("  GeoTIFF loader → format=", gspec.format, " (const: no temporal cadence)")

    # The real ERA5 loader .esm (if the EarthSciModels checkout is present).
    real = joinpath(ESM_MODELS, "components", "earthsci_data", "era5_loader.esm")
    if isfile(real)
        doc = JSON3.read(read(real, String), Dict{String,Any})
        dls = doc["data_loaders"]
        loader = first(values(dls))
        rspec = to_esio_loader(loader; anchor = Dates.DateTime(2018, 11, 8))
        @assert rspec.format == "netcdf" && rspec.time_dim == "valid_time"
        @assert occursin("era5_pl_2018_11.nc", rspec.url)
        println("  real era5_loader.esm → ", rspec.url, "  (", length(rspec.variables), " variables)")
    else
        println("  (EARTHSCIMODELS era5_loader.esm not found — skipped the real-file check)")
    end

    println("\n  ✓ to_esio_loader maps a loader .esm (url_template + temporal + variables) to a runnable",
            "\n    esio_provider spec; date-template expansion, ISO-8601 cadence, and format inference work.")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
