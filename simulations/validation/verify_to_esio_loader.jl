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

    # ArcGIS ImageServer (LANDFIRE / USGS GeoTIFF): the domain bbox + image size + static
    # version/product fills resolve the `exportImage` template to a concrete request URL.
    arcgis = Dict{String,Any}(
        "source" => Dict{String,Any}("url_template" =>
            "https://lfps.usgs.gov/arcgis/rest/services/Landfire_{version}/{version}_{product}_CONUS/ImageServer/" *
            "exportImage?bbox={bbox_west_deg},{bbox_south_deg},{bbox_east_deg},{bbox_north_deg}&bboxSR=4326" *
            "&imageSR=4326&size={image_width},{image_height}&format=tiff&f=image"),
        "metadata" => Dict{String,Any}("url_defaults" => Dict{String,Any}("version" => "LF2022", "product" => "FBFM13")),
        "variables" => Dict{String,Any}("fuel_model" => Dict{String,Any}("file_variable" => "Band1")))
    aspec = to_esio_loader(arcgis; target = (-121.89, 39.54, -121.31, 40.02), image_size = (290, 240))
    println("  ArcGIS LANDFIRE → ", aspec.url)
    @assert aspec.format == "geotiff"
    @assert occursin("Landfire_LF2022/LF2022_FBFM13_CONUS", aspec.url)
    @assert occursin("bbox=-121.89,39.54,-121.31,40.02", aspec.url)
    @assert occursin("size=290,240", aspec.url)
    @assert !occursin("{", aspec.url)                # every placeholder filled
    println("  → bbox/image-size + version/product fills resolved (no placeholder left)")

    # CDS (ERA5): a metadata.cds loader → a cds:// request URL whose `area` is the domain bbox.
    cdsl = Dict{String,Any}(
        "source" => Dict{String,Any}("url_template" => "cds://reanalysis-era5-pressure-levels"),
        "metadata" => Dict{String,Any}("cds" => Dict{String,Any}(
            "dataset" => "reanalysis-era5-pressure-levels", "pressure_levels" => Any[1000, 850])),
        "variables" => Dict{String,Any}("t" => Dict{String,Any}("file_variable" => "t"),
                                        "u" => Dict{String,Any}("file_variable" => "u")))
    cspec = to_esio_loader(cdsl; target = (-121.89, 39.54, -121.31, 40.02))
    println("  CDS ERA5 → ", cspec.url)
    @assert startswith(cspec.url, "cds://reanalysis-era5-pressure-levels?")
    @assert occursin("area=40.02/-121.89/39.54/-121.31", cspec.url)   # N/W/S/E from the bbox
    @assert occursin("pressure_level=1000,850", cspec.url)
    @assert Set(cspec.variables) == Set(["t", "u"])
    println("  → CDS area (N/W/S/E) + pressure levels + variables resolved from the domain")

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

    # The real LANDFIRE loader .esm (ArcGIS) end to end, if present.
    rlf = joinpath(ESM_MODELS, "components", "earthsci_data", "landfire_loader.esm")
    if isfile(rlf)
        loader = first(values(JSON3.read(read(rlf, String), Dict{String,Any})["data_loaders"]))
        lspec = to_esio_loader(loader; target = (-121.89, 39.54, -121.31, 40.02), image_size = (290, 240))
        @assert lspec.format == "geotiff" && !occursin("{", lspec.url) && occursin("bbox=-121.89", lspec.url)
        println("  real landfire_loader.esm → ", split(lspec.url, "?")[1], "?…(bbox+size filled)")
    else
        println("  (EARTHSCIMODELS landfire_loader.esm not found — skipped the real-file check)")
    end

    println("\n  ✓ to_esio_loader maps a loader .esm (url_template + temporal + variables) to a runnable",
            "\n    esio_provider spec; date-template expansion, ISO-8601 cadence, format inference, the ArcGIS",
            "\n    bbox/image-size fills, and the CDS `area` request all resolve from the domain.")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
