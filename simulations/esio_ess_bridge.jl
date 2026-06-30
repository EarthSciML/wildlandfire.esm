#=
esio_ess_bridge.jl — wire real EarthSciIO data providers into the ESS Julia
simulation stack (Phase 4: auto data loading).

EarthSciSerialization already ships the WHOLE consumer surface for cadence data
(src/data_refresh.jl): the `provider_refresh_times` / `provider_is_const` /
`provider_sample` Provider protocol, the `RegridApplier` seam (`IdentityRegrid`
passthrough + the ESD-rule `ESDRegrid` reproject/conservative applier), the
`RefreshBuffers` live-buffer registry, and `build_refresh_callback(model;
providers, buffers, regrid) -> (cb, tstops)` (a `PresetTimeCallback`, in a
`DiffEqCallbacks`+`SciMLBase` extension). ESS deliberately "never imports
EarthSciIO" — it calls those generics and lets the DATA BINDING fill them in.

This is that data binding for the Julia run env (the user's "depend on ESIO in
the run env"): a thin ADAPTER making an `EarthSciIO.Provider` satisfy the ESS
protocol, plus a minimal FACTORY from a loader spec to a `Provider`. With it, a
loader's native arrays (netcdf / csv / geotiff) flow:

    EarthSciIO.Provider  --adapter-->  ESS.provider_sample  -->  apply_regrid!
        --(in-place)-->  the live param_arrays buffer the RHS reads

so a discrete-cadence forcing refreshes at its anchors during a solve, and a
const forcing is materialized once into `const_arrays`.

Usage (after `using EarthSciSerialization, EarthSciIO`; `include` this file):

    prov = esio_provider((; url, format="netcdf", times=[0.0, 3600.0], time_dim="time"))
    cb, tstops = build_refresh_callback(model;
        providers = Dict("t2m" => prov),
        buffers   = RefreshBuffers(Dict("t2m" => buf)),
        regrid    = ESDRegrid(target_grid, src_lon, src_lat; methods=...))
=#
import EarthSciSerialization as ESS
import EarthSciIO
import Dates

# --- adapter: EarthSciIO.Provider satisfies the ESS Provider protocol -------
# These define methods on ESS's generics for EarthSciIO's type — the exact seam
# the ESS data_refresh.jl docstring sketches ("the data binding (EarthSciIO) ...
# must add a method"). Run-env glue, so the cross-package method is sanctioned.

ESS.provider_refresh_times(p::EarthSciIO.Provider) = EarthSciIO.refresh_times(p)
ESS.provider_is_const(p::EarthSciIO.Provider) = EarthSciIO.is_const(p)

# The native sample handed to `apply_regrid!`: a `var name => native array` dict
# (ESS's `_regrid_field` pulls the named field out). CONST providers materialize
# once (no tick); DISCRETE providers re-materialize at the cadence tick `t`.
function ESS.provider_sample(p::EarthSciIO.Provider, t::Real)
    nds = EarthSciIO.is_const(p) ? EarthSciIO.materialize(p) : EarthSciIO.refresh(p, t)
    return esio_native_dict(nds)
end

"A `NativeDataset` → `Dict(var name => native Float64 array)`. Coordinates are not
forcing fields, so only the data variables are surfaced; the array keeps its native
shape (the `RegridApplier` flattens / reprojects it)."
function esio_native_dict(nds::EarthSciIO.NativeDataset)
    out = Dict{String,Array{Float64}}()
    for name in EarthSciIO.variable_names(nds)
        out[name] = Array{Float64}(nds[name].data)
    end
    return out
end

# --- factory: a loader spec -> an EarthSciIO.Provider -----------------------
# Minimal mapping for the common case: a RESOLVED url (or `t -> url`) + format +
# either CONST (no `times`) or DISCRETE (`times=[...]`). Optional `variables`,
# `time_dim`, `reader_kwargs`, `source_loader`. The full data-loader `.esm` →
# spec mapping (URL-template / domain-bbox / CDS expansion, the Python
# `to_esio_loader`) layers on top of this and is the Phase-5 increment.
"""
    esio_provider(spec; cache=EarthSciIO.Cache()) -> EarthSciIO.Provider

Build a provider from `spec` (a NamedTuple/Dict): required `url` + `format`;
`times=[...]` makes it DISCRETE (else CONST). Optional `variables`, `time_dim`,
`reader_kwargs`, `source_loader`.
"""
function esio_provider(spec; cache::EarthSciIO.Cache = EarthSciIO.Cache())
    g(k, d = nothing) = spec isa AbstractDict ?
        get(spec, k, get(spec, String(k), d)) : get(spec, k, d)
    url = g(:url)
    url === nothing && error("esio_provider: loader spec needs a `url` (String or t->url)")
    fmt = String(g(:format, "netcdf"))
    times = g(:times, nothing)
    rk = g(:reader_kwargs, NamedTuple())
    common = (; variables = g(:variables, nothing),
                reader_kwargs = rk,
                source_loader = g(:source_loader, nothing))
    if times === nothing
        return EarthSciIO.const_provider(cache, url; format = fmt, common...)
    else
        return EarthSciIO.discrete_provider(cache, url, collect(Float64, times);
                                            format = fmt, time_dim = g(:time_dim, nothing),
                                            common...)
    end
end

# --- to_esio_loader: a data-loader `.esm` block → an esio_provider spec --------
# The Julia counterpart of the Python `earthsci_toolkit.data_loaders.esio_provider
# .to_esio_loader`, for the common netcdf URL-template case (the LANDFIRE / USGS
# ArcGIS `{bbox}`/image-size fills and the ERA5 CDS `area` request are the exotic
# tail — a documented gap). It maps the loader's declared `source.url_template`,
# `temporal` cadence, and `variables.<v>.file_variable` to the (url, format,
# variables, time_dim, times) spec `esio_provider` consumes, so a loader becomes
# a runnable Provider with no hand-written spec.

"Infer the EarthSciIO format key from a loader URL + metadata (mirrors Python `_esio_format`)."
function _esio_format(url::AbstractString, metadata::AbstractDict = Dict{String,Any}())
    haskey(metadata, "esio_format") && return String(metadata["esio_format"])
    u = lowercase(url)
    (endswith(u, ".tif") || endswith(u, ".tiff") || occursin("format=tiff", u)) && return "geotiff"
    (endswith(u, ".nc") || occursin("format=netcdf", u)) && return "netcdf"
    ff = lowercase(String(get(metadata, "file_format", "")))
    ff in ("geotiff", "tiff", "tif") && return "geotiff"
    ff in ("netcdf", "nc") && return "netcdf"
    error("to_esio_loader: cannot infer an EarthSciIO format for url '$url'; " *
          "set the loader metadata.esio_format (e.g. \"netcdf\")")
end

# ISO-8601 duration → seconds, for the common cadence units (PT…H/M/S, P…D).
# Calendar-length units (months/years) are returned as their nominal second count
# (30 d / 365 d) — adequate for a cadence grid, exact for the PT…H ERA5 case.
function _iso8601_seconds(s::AbstractString)
    m = match(r"^P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$", String(s))
    m === nothing && error("to_esio_loader: cannot parse ISO-8601 duration '$s'")
    n(i) = m.captures[i] === nothing ? 0 : parse(Int, m.captures[i])
    return n(1)*365*86400 + n(2)*30*86400 + n(3)*86400 + n(4)*3600 + n(5)*60 + n(6)
end

# Expand a `{date:%Y}`-style URL template at `dt` (the strftime fills the Python
# loader URLs use). Maps the common %-codes to Julia `Dates.format` codes.
function _expand_date_template(tpl::AbstractString, dt::Dates.DateTime)
    return replace(tpl, r"\{date:([^}]+)\}" => m -> begin
        fmt = match(r"\{date:([^}]+)\}", m).captures[1]
        for (pc, jc) in ("%Y" => "yyyy", "%m" => "mm", "%d" => "dd",
                         "%H" => "HH", "%M" => "MM", "%S" => "SS")
            fmt = replace(fmt, pc => jc)
        end
        Dates.format(dt, fmt)
    end)
end

# Domain-derived URL fills (the Python `to_esio_loader`'s `target`/CDS tail): the
# server-side-subsetting GeoTIFF loaders (LANDFIRE / USGS 3DEP ArcGIS ImageServer)
# carry `{bbox_*_deg}` + `{image_*}` placeholders the simulation domain fills, and a
# CDS loader (ERA5) carries `metadata.cds.dataset` whose request `area` comes from the
# domain bbox. `target = (min_lon, min_lat, max_lon, max_lat)` is the fire grid's lon/lat
# envelope (e.g. `_domain_lonlat_bbox(dom)`).

"Fill static `{key}` placeholders (e.g. `{version}`, `{product}`) from a defaults dict."
function _fill_consts(tpl::AbstractString, defaults::AbstractDict)
    out = String(tpl)
    for (k, v) in defaults
        out = replace(out, "{$(String(k))}" => string(v))
    end
    return out
end

"Fill the ArcGIS ImageServer `exportImage` domain placeholders — the WGS84 bbox
(`{bbox_west_deg}`,`{bbox_south_deg}`,`{bbox_east_deg}`,`{bbox_north_deg}`) and the
requested raster `{image_width}`,`{image_height}` — from the target bbox + image size."
function _fill_bbox(tpl::AbstractString, target, image_size)
    target === nothing && return String(tpl)
    minlon, minlat, maxlon, maxlat = target
    out = replace(String(tpl),
        "{bbox_west_deg}" => string(minlon), "{bbox_south_deg}" => string(minlat),
        "{bbox_east_deg}" => string(maxlon), "{bbox_north_deg}" => string(maxlat))
    if image_size !== nothing
        out = replace(out, "{image_width}" => string(image_size[1]),
                           "{image_height}" => string(image_size[2]))
    end
    return out
end

"ECMWF/CDS `area` = `[North, West, South, East]` from a `(min_lon,min_lat,max_lon,max_lat)` bbox."
era5_area_from_bbox(target) = [target[4], target[1], target[2], target[3]]

"Build a `cds://<dataset>?area=N/W/S/E&variable=…&…` request URL (the spec the ESIO `cds`
transport submits) from the domain bbox + the loader's file variables and CDS metadata."
function _cds_url(dataset, target, file_vars; levels = nothing, extra = Dict{String,Any}())
    n, w, s, e = era5_area_from_bbox(target)
    parts = ["dataset=$(dataset)", "area=$(n)/$(w)/$(s)/$(e)",
             "variable=$(join(file_vars, ","))"]
    levels === nothing || push!(parts, "pressure_level=$(join(levels, ","))")
    for (k, v) in extra; push!(parts, "$(String(k))=$(v)"); end
    return "cds://$(dataset)?" * join(parts, "&")
end

"""
    to_esio_loader(loader; anchor=nothing, times=nothing, target=nothing, image_size=nothing) -> NamedTuple spec

Map a single data-loader block (the `data_loaders.<name>` object from a loader
`.esm` — `source.url_template` + `temporal` + `variables`) to an
[`esio_provider`](@ref) spec. `anchor::Dates.DateTime` resolves a `{date:…}` URL
template to a concrete URL (else the raw template is returned, for a caller-built
`t -> url`). `times` overrides the cadence; otherwise it is built from the
loader's `temporal` (`start` + `frequency` over `records_per_file`, or a single
const sample when there is no cadence).

`target = (min_lon, min_lat, max_lon, max_lat)` (the simulation grid's lon/lat
envelope) fills the domain-derived URL parameters — the ArcGIS ImageServer
`{bbox_*_deg}`/`{image_*}` placeholders of the GeoTIFF loaders (LANDFIRE / USGS
3DEP), and the CDS `area` of a `metadata.cds` loader (ERA5). `image_size = (w, h)`
sets the requested ArcGIS raster size. Static `{version}`/`{product}` fills are
read from `metadata.url_defaults`. The returned spec has `url`, `format`,
`variables` (the file-variable names), `time_dim`, and `times`.
"""
function to_esio_loader(loader::AbstractDict; anchor::Union{Nothing,Dates.DateTime} = nothing,
                        times = nothing, n_records::Union{Nothing,Int} = nothing,
                        target = nothing, image_size = nothing)
    src = get(loader, "source", nothing)
    src === nothing && error("to_esio_loader: loader has no `source` block")
    tpl = String(get(src, "url_template", get(src, "url", "")))
    isempty(tpl) && error("to_esio_loader: loader source has no `url_template`/`url`")
    meta = get(loader, "metadata", Dict{String,Any}())

    vars = get(loader, "variables", Dict{String,Any}())
    file_vars = String[String(get(v, "file_variable", k)) for (k, v) in vars]

    temporal = get(loader, "temporal", nothing)
    time_dim = temporal === nothing ? nothing :
        (tv = get(temporal, "time_variable", nothing); tv === nothing ? nothing : String(tv))

    # Cadence: explicit `times`, else a uniform grid from temporal.frequency.
    cadence = times
    if cadence === nothing && temporal !== nothing && haskey(temporal, "frequency")
        step = _iso8601_seconds(String(temporal["frequency"]))
        nrec = n_records !== nothing ? n_records : Int(get(temporal, "records_per_file", 1))
        cadence = Float64[(i - 1) * step for i in 1:nrec]
    end

    # CDS loader: the source opts in with a `cds://<dataset>` url_template (a loader can
    # ALSO carry an http mirror template + metadata.cds — then the mirror is primary, as
    # for era5_loader.esm). Build the cds:// request URL with `area` from the domain bbox;
    # metadata.cds supplies pressure levels / format.
    if startswith(lowercase(tpl), "cds:")
        cds = get(meta, "cds", Dict{String,Any}())
        dataset = String(get(cds, "dataset", replace(tpl, r"^cds://"i => "")))
        target === nothing && error("to_esio_loader: CDS loader needs a `target` bbox for the request `area`")
        url = _cds_url(dataset, target, file_vars; levels = get(cds, "pressure_levels", nothing))
        return (; url = url, format = String(get(cds, "format", "netcdf")),
                variables = file_vars, time_dim = time_dim, times = cadence)
    end

    # Static fills (version/product) → domain bbox/image-size fills → date expansion.
    url = _fill_consts(tpl, get(meta, "url_defaults", Dict{String,Any}()))
    url = _fill_bbox(url, target, image_size)
    url = anchor === nothing ? url : (occursin("{date:", url) ? _expand_date_template(url, anchor) : url)
    return (; url = url, format = _esio_format(url, meta), variables = file_vars,
            time_dim = time_dim, times = cadence)
end
