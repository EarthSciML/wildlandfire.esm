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

"""
    to_esio_loader(loader; anchor=nothing, times=nothing) -> NamedTuple spec

Map a single data-loader block (the `data_loaders.<name>` object from a loader
`.esm` — `source.url_template` + `temporal` + `variables`) to an
[`esio_provider`](@ref) spec. `anchor::Dates.DateTime` resolves a `{date:…}` URL
template to a concrete URL (else the raw template is returned, for a caller-built
`t -> url`). `times` overrides the cadence; otherwise it is built from the
loader's `temporal` (`start` + `frequency` over `records_per_file`, or a single
const sample when there is no cadence). The returned spec has `url`, `format`,
`variables` (the file-variable names), `time_dim`, and `times`.
"""
function to_esio_loader(loader::AbstractDict; anchor::Union{Nothing,Dates.DateTime} = nothing,
                        times = nothing, n_records::Union{Nothing,Int} = nothing)
    src = get(loader, "source", nothing)
    src === nothing && error("to_esio_loader: loader has no `source` block")
    tpl = String(get(src, "url_template", get(src, "url", "")))
    isempty(tpl) && error("to_esio_loader: loader source has no `url_template`/`url`")
    meta = get(loader, "metadata", Dict{String,Any}())
    fmt = _esio_format(tpl, meta)

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

    url = anchor === nothing ? tpl :
          (occursin("{date:", tpl) ? _expand_date_template(tpl, anchor) : tpl)

    return (; url = url, format = fmt, variables = file_vars,
            time_dim = time_dim, times = cadence)
end
