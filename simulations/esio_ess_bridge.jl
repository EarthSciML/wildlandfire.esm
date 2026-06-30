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
