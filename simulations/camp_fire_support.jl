#!/usr/bin/env julia
#=
camp_fire_support.jl — the Camp Fire run-harness support library: the env
bootstrap plus every reusable helper for assembling and forcing the coupled
system (LCC reprojection, ERA5 source grids + live data, the conservative
regrid spec, the scalar Rothermel chain, and the MONOLITHIC fire assembly).

Extracted from the old multi-mode driver so the driver itself collapses to a
thin `simulate`-based shim (run_camp_fire_e2e_pde.jl) and the verify_*.jl
harnesses share one helper library. The 2-D centered |grad psi| stencils are
loaded BY REF from the EarthSciDiscretizations catalog (`gdd_to_rules`, Phase 2)
rather than hand-rolled; the run mechanics (build_evaluator + IC seed + solve +
cadence refresh) live in ESS `simulate` (Phase 5).

`include` this; it defines the helpers but runs nothing.
=#
import Pkg

const REPO = dirname(@__DIR__)
const CAMP_FIRE_ESM = joinpath(REPO, "simulations", "camp_fire.esm")
const ESS_ROOT = get(ENV, "EARTHSCISERIALIZATION",
                     normpath(joinpath(REPO, "..", "EarthSciSerialization")))
const ESS_PKG = joinpath(ESS_ROOT, "packages", "EarthSciSerialization.jl")

let env = joinpath(ESS_PKG, "scripts", "pde_sim_adapter")
    isdir(env) || error("pde_sim_adapter project not found at $env — set " *
                        "EARTHSCISERIALIZATION to your EarthSciSerialization checkout")
    Pkg.activate(env; io = devnull)
    isfile(joinpath(env, "Manifest.toml")) || Pkg.develop(path = ESS_PKG; io = devnull)
    haskey(Pkg.project().dependencies, "GeometryOps") ||
        Pkg.add(["GeometryOps", "GeoInterface"]; io = devnull)
    Pkg.instantiate(; io = devnull)
end
# Load the spherical-clip backend (GeometryOps + GeoInterface) at TOP LEVEL so the ESS
# GeometryOps extension's `intersect_polygon`/`polygon_area` spherical methods are visible
# in the world age every helper runs in. Importing it lazily INSIDE a function fails
# silently (world-age: build_evaluator, already running, can't see the just-added methods),
# yielding empty clips ⇒ a zero weight matrix. The computed conservative regridder is
# geographic, so this backend is mandatory for `computed_regrid_weights`.
import GeometryOps, GeoInterface

using EarthSciSerialization
using JSON3
using Logging
import OrdinaryDiffEqTsit5
const ODE = OrdinaryDiffEqTsit5
const ESS = EarthSciSerialization
const ESD_ROOT = get(ENV, "EARTHSCIDISCRETIZATIONS",
                     normpath(joinpath(REPO, "..", "earthscidiscretizations")))
const GRAD_2D = joinpath(ESD_ROOT, "discretizations", "finite_difference",
                         "centered_grad_2nd_uniform_cartesian_2d.json")
# The REAL conservative regridder camp_fire.esm couples in (ERA5t/r, Fuel, SlopeX/Y →
# F_tgt): A_ij COMPUTED from the spherical intersect_polygon clip + Van Oosterom-Strackee
# spherical-excess area, with the partition-of-unity-exact bin-skolem A_j_w denominator.
# The driver runs THIS component (not a hand-rolled planar shoelace) for its regrid — see
# `computed_regrid_weights`. (Phase-5 residual #4: retire the hand-rolled fields_regrid_spec.)
const REGRID_COMPUTED = joinpath(ESD_ROOT, "regridding",
                                 "conservative_regrid_overlap_join_computed.esm")
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"Resolve the LevelSetFireSpread component path from camp_fire.esm's own `{ref}`."
function levelset_ref_path(camp_path)
    raw = JSON3.read(read(camp_path, String), Dict{String, Any})
    ref = raw["models"]["LevelSetFireSpread"]["ref"]
    return abspath(joinpath(dirname(camp_path), String(ref)))
end

"Read the real `camp_fire_surface` domain (nx, ny, dx, signed-distance psi IC)."
function camp_domain(camp_path)
    camp = JSON3.read(read(camp_path, String), Dict{String, Any})
    surf = camp["domains"]["camp_fire_surface"]
    sx, sy = surf["spatial"]["x"], surf["spatial"]["y"]
    dx = Float64(sx["grid_spacing"])
    nx = Int(round((sx["max"] - sx["min"]) / sx["grid_spacing"])) + 1
    ny = Int(round((sy["max"] - sy["min"]) / sy["grid_spacing"])) + 1
    ic = parse_expression(surf["initial_conditions"]["values"]["psi"])
    (; nx, ny, dx, xmin = Float64(sx["min"]), ymin = Float64(sy["min"]),
       tstart = surf["temporal"]["start"], tend = surf["temporal"]["end"], ic)
end

"Build a discretize-ready ESM dict for the LevelSetFireSpread PDE core on the real
Camp Fire grid. The component's array-shaped observeds (psi_x, grad_mag, phi_W, …)
are folded into the single `D(psi,t)` RHS by single-pass topological substitution,
because the MTK-free evaluator only takes scalar observeds."
# The level-set inputs that --wind drives per cell from the conservative regrid:
# wind components (u_x, u_y → U_n → φ_W) and terrain slope (dzdx, dzdy → tan_phi_n
# → φ_S). All four feed S_n = R_0·(1+φ_W+φ_S) and so the front speed at each cell.
const COUPLED_FIELDS = ["u_x", "u_y", "dzdx", "dzdy"]

function levelset_pde_esm(ls_path, dom; wind::Bool = false)
    doc = JSON3.read(read(ls_path, String), Dict{String, Any})
    model = doc["models"]["LevelSetFireSpread"]
    vars = model["variables"]

    if wind
        # Each coupled field becomes a per-cell array fed by the regrid (attached
        # AFTER discretize). Declaring it grid-shaped makes discretize keep it as an
        # array INPUT — a bare reference inside the scalarized D(psi,t) arrayop that
        # the couple step rewrites to index(field,i,j); build_evaluator then inlines
        # the regrid per cell (the ess-14f.4 coupled-array bridge).
        for fn in COUPLED_FIELDS
            vars[fn]["shape"] = Any["x", "y"]
        end
    end

    obs_names = String[k for (k, v) in vars if get(v, "type", "") == "observed"]
    obs_expr = Dict{String, ESS.Expr}(nm => parse_expression(vars[nm]["expression"])
                                      for nm in obs_names)
    obs_set = Set(obs_names)
    deps = Dict(nm => intersect(free_variables(obs_expr[nm]), obs_set) for nm in obs_names)

    order = String[]
    done = Set{String}()
    while length(order) < length(obs_names)
        progressed = false
        for nm in obs_names
            nm in done && continue
            if all(d -> d in done, deps[nm])
                push!(order, nm)
                push!(done, nm)
                progressed = true
            end
        end
        progressed || error("cyclic observed dependency in LevelSetFireSpread")
    end
    resolved = Dict{String, ESS.Expr}()
    for nm in order
        resolved[nm] = ESS.substitute(obs_expr[nm], resolved)
    end

    stateq = model["equations"][1]
    stateq["rhs"] = ESS.serialize_expression(
        ESS.substitute(parse_expression(stateq["rhs"]), resolved))
    for nm in obs_names
        delete!(vars, nm)
    end
    vars["dx"] = Dict{String, Any}("type" => "parameter", "default" => dom.dx, "units" => "m")
    model["grid"] = "g"

    esm = Dict{String, Any}(
        "esm" => "0.5.0",
        "metadata" => Dict{String, Any}("name" => "CampFireLevelSetCore"),
        "grids" => Dict{String, Any}("g" => Dict{String, Any}(
            "family" => "cartesian",
            "dimensions" => Any[
                Dict{String, Any}("name" => "x", "size" => dom.nx,
                                  "periodic" => false, "spacing" => "uniform"),
                Dict{String, Any}("name" => "y", "size" => dom.ny,
                                  "periodic" => false, "spacing" => "uniform")])),
        "models" => Dict{String, Any}("LevelSetFireSpread" => model),
        "rules" => gdd_to_rules(GRAD_2D; spacing = "dx"))
    return esm
end

# ---------------------------------------------------------------------------
# Met→fire coupling — a declarative conservative regrid that drives the level-set's
# per-cell wind (u_x,u_y) and slope (dzdx,dzdy) from source fields, computed THROUGH
# the tree-walk evaluator (the overlap matrix A_ij — clip → polygon_area — and the
# apply A_ijᵀ·F_src/A_j are build-once at setup; each field is inlined per cell).
# The fire-grid target cells are CONSTRUCTED declaratively from the grid origin/
# spacing; the source cells come in as const polygons (`source_grid_*` below) — a
# coarse LCC band for the demo, the real ERA5 lat/lon grid reprojected to LCC for
# `--era5`. All fields share ONE geometry (clip/A_ij/A_j); only F_src differs.
# ---------------------------------------------------------------------------

_ixd(args...) = Dict{String, Any}("op" => "index", "args" => collect(Any, args))
_opd(o, args...) = Dict{String, Any}("op" => o, "args" => collect(Any, args))
_aggd(lv, set, body) = Dict{String, Any}("op" => "aggregate", "semiring" => "sum_product",
    "output_idx" => Any[], "ranges" => Dict{String, Any}(lv => Dict{String, Any}("from" => set)),
    "args" => Any[], "expr" => body)
_arrd(oidx, ranges, body) = Dict{String, Any}("op" => "arrayop", "output_idx" => collect(Any, oidx),
    "ranges" => Dict{String, Any}(k => Dict{String, Any}("from" => v) for (k, v) in ranges),
    "args" => Any[], "expr" => body)

"Host-build the source-grid cell polygons (const) as a coarse `nsx×nsy` grid over
the fire-domain extent, in the fire's OWN LCC frame (no reprojection — `--era5`
swaps in the real lat/lon grid). Returns (src_poly [ns,4,2], nsx, nsy); the flat
source index is s = (sy-1)*nsx + sx, column-major in (sx,sy)."
function source_grid_lcc(dom; nsx::Int = 3, nsy::Int = 3)
    nx, ny, dx = dom.nx, dom.ny, dom.dx
    xlo = dom.xmin - dx / 2          # fire-domain west/south edges (cells centered on nodes)
    ylo = dom.ymin - dx / 2
    dsx = nx * dx / nsx
    dsy = ny * dx / nsy
    ns = nsx * nsy
    sp = zeros(ns, 4, 2)
    for sy in 1:nsy, sx in 1:nsx
        s = (sy - 1) * nsx + sx
        x0 = xlo + (sx - 1) * dsx
        y0 = ylo + (sy - 1) * dsy
        sp[s, 1, :] = [x0, y0];            sp[s, 2, :] = [x0 + dsx, y0]
        sp[s, 3, :] = [x0 + dsx, y0 + dsy]; sp[s, 4, :] = [x0, y0 + dsy]
    end
    return sp, nsx, nsy
end

# --- Spherical secant Lambert Conformal Conic (the same projection as
#     reprojection/lambert_conformal.esm: lat_1=30, lat_2=60, lat_0=39, lon_0=-97),
#     used host-side to reproject a real ERA5 lat/lon source grid into the fire's
#     LCC frame. R is the authalic sphere radius; the round-trip is self-consistent
#     (same R both ways), so the reprojected ERA5 cells tile the fire domain exactly
#     regardless of the spherical-vs-ellipsoidal approximation in absolute position.
const _LCC_D = π / 180
_lcc_params() = (lat1 = 30.0, lat2 = 60.0, lat0 = 39.0, lon0 = -97.0, R = 6371000.0)
function _lcc_nF(p)
    φ1, φ2 = p.lat1 * _LCC_D, p.lat2 * _LCC_D
    t(q) = tan(π / 4 + q / 2)
    n = log(cos(φ1) / cos(φ2)) / log(t(φ2) / t(φ1))
    F = cos(φ1) * t(φ1)^n / n
    return n, F, t(p.lat0 * _LCC_D)
end
function lcc_forward(lon, lat, p = _lcc_params())
    n, F, t0 = _lcc_nF(p)
    t(q) = tan(π / 4 + q / 2)
    ρ0 = p.R * F / t0^n
    ρ = p.R * F / t(lat * _LCC_D)^n
    θ = n * (lon - p.lon0) * _LCC_D
    return (ρ * sin(θ), ρ0 - ρ * cos(θ))
end
function lcc_inverse(x, y, p = _lcc_params())
    n, F, t0 = _lcc_nF(p)
    ρ0 = p.R * F / t0^n
    ρ = sign(n) * sqrt(x^2 + (ρ0 - y)^2)
    θ = atan(x, ρ0 - y)
    lon = p.lon0 + θ / n / _LCC_D
    φ = 2 * atan((p.R * F / ρ)^(1 / n)) - π / 2
    return (lon, φ / _LCC_D)
end

"Real ERA5 lat/lon source grid (`dll`° cells, 0.25° = the ERA5 surface resolution)
covering the fire domain, each cell's lon/lat corners reprojected to the fire's LCC
frame. The domain's LCC extent is inverse-projected to lat/lon, snapped to the ERA5
grid, then each ERA5 cell's corners are forward-projected to LCC. Returns (src_poly
[ns,4,2] in LCC m, nlon, nlat, cell_lons, cell_lats); s = (ilat-1)*nlon + ilon."
function source_grid_era5(dom; dll::Float64 = 0.25)
    dx = dom.dx
    xlo, ylo = dom.xmin - dx / 2, dom.ymin - dx / 2
    xhi = dom.xmin + (dom.nx - 1) * dx + dx / 2
    yhi = dom.ymin + (dom.ny - 1) * dx + dx / 2
    # Sample the LCC boundary densely (not just the 4 corners): in lat/lon the domain
    # edges bow, so a 4-corner bbox can clip a sliver of an edge fire cell. Sampling
    # the perimeter captures the bowing, so the snapped ERA5 grid covers every cell.
    lons = Float64[]; lats = Float64[]
    for f in range(0.0, 1.0; length = 21)
        for (x, y) in ((xlo + f*(xhi-xlo), ylo), (xlo + f*(xhi-xlo), yhi),
                       (xlo, ylo + f*(yhi-ylo)), (xhi, ylo + f*(yhi-ylo)))
            lo, la = lcc_inverse(x, y); push!(lons, lo); push!(lats, la)
        end
    end
    lo0 = floor(minimum(lons) / dll) * dll; lo1 = ceil(maximum(lons) / dll) * dll
    la0 = floor(minimum(lats) / dll) * dll; la1 = ceil(maximum(lats) / dll) * dll
    nlon = Int(round((lo1 - lo0) / dll)); nlat = Int(round((la1 - la0) / dll))
    ns = nlon * nlat
    sp = zeros(ns, 4, 2); clons = zeros(ns); clats = zeros(ns)
    for ilat in 1:nlat, ilon in 1:nlon
        s = (ilat - 1) * nlon + ilon
        a, b = lo0 + (ilon - 1) * dll, lo0 + ilon * dll       # lon edges
        c, d = la0 + (ilat - 1) * dll, la0 + ilat * dll       # lat edges
        for (k, (lo, la)) in enumerate(((a, c), (b, c), (b, d), (a, d)))
            x, y = lcc_forward(lo, la); sp[s, k, 1] = x; sp[s, k, 2] = y
        end
        clons[s] = (a + b) / 2; clats[s] = (c + d) / 2
    end
    return sp, nlon, nlat, clons, clats
end

# --- Live ERA5 data (item 3). The Julia adapter env is NetCDF-free, so a small
#     netCDF4 Python shim (era5_extract.py) reads the real Camp Fire ERA5 file and
#     returns surface u/v on the native ERA5 grid; the values fill the live F_src
#     buffers, so the front is driven by — and refreshes from — real data. (This is
#     the data-access shim, not the formal EarthSciData loader binding.) ---
const ERA5_NC = get(ENV, "ERA5_NC",
    joinpath(get(ENV, "EARTHSCIDATADIR", joinpath(homedir(), "data", "earthsciml")),
             "era5_pressure_levels", "era5_pl_2018_11.nc"))
const ERA5_PY = get(ENV, "ERA5_PYTHON", joinpath(homedir(), "anaconda3", "bin", "python"))
const ERA5_EXTRACT = joinpath(@__DIR__, "era5_extract.py")

"Domain lat/lon bbox (padded) via inverse-LCC of the domain perimeter."
function _domain_lonlat_bbox(dom; pad = 0.3)
    dx = dom.dx
    xlo, ylo = dom.xmin - dx / 2, dom.ymin - dx / 2
    xhi = dom.xmin + (dom.nx - 1) * dx + dx / 2
    yhi = dom.ymin + (dom.ny - 1) * dx + dx / 2
    lons = Float64[]; lats = Float64[]
    for fr in range(0.0, 1.0; length = 21)
        for (x, y) in ((xlo + fr*(xhi-xlo), ylo), (xlo + fr*(xhi-xlo), yhi),
                       (xlo, ylo + fr*(yhi-ylo)), (xhi, ylo + fr*(yhi-ylo)))
            lo, la = lcc_inverse(x, y); push!(lons, lo); push!(lats, la)
        end
    end
    (minimum(lons) - pad, maximum(lons) + pad, minimum(lats) - pad, maximum(lats) + pad)
end

"Load real ERA5 1000-hPa u/v wind for the domain at `time_index` (hourly; 14 =
2018-11-08 14:00 UTC, the Camp Fire ignition hour) via the netCDF4 Python shim.
Returns (lons, lats, u, v, valid_time) over the native ERA5 grid points."
function load_era5_wind(dom; time_index::Int = 14)
    isfile(ERA5_NC) || error("ERA5 file not found: $ERA5_NC (set ERA5_NC)")
    isfile(ERA5_PY) || error("netCDF Python not found: $ERA5_PY (set ERA5_PYTHON)")
    lo0, lo1, la0, la1 = _domain_lonlat_bbox(dom)
    out = read(`$ERA5_PY $ERA5_EXTRACT $ERA5_NC $lo0 $lo1 $la0 $la1 $time_index`, String)
    j = JSON3.read(out)
    return (Float64.(j.lons), Float64.(j.lats), Float64.(j.u), Float64.(j.v), String(j.valid_time))
end

"Source-cell polygons centered on explicit lon/lat points (the ERA5 native grid),
each ±dll/2 reprojected to the fire's LCC frame. Returns src_poly [ns,4,2]."
function source_grid_from_points(lons, lats; dll::Float64 = 0.25)
    ns = length(lons); sp = zeros(ns, 4, 2)
    for s in 1:ns
        a, b = lons[s] - dll/2, lons[s] + dll/2
        c, e = lats[s] - dll/2, lats[s] + dll/2
        for (k, (lo, la)) in enumerate(((a, c), (b, c), (b, e), (a, e)))
            x, y = lcc_forward(lo, la); sp[s, k, 1] = x; sp[s, k, 2] = y
        end
    end
    return sp
end

"Index_sets + variables for the SHARED-geometry conservative regrid onto the fire
grid: the declaratively-constructed target cells, the const `src_poly` operand, the
ranged clip → A_ij → A_j, and one observed `fn[x,y] = Σ_s A_ij·F_src_<fn> / A_j`
per coupled field. `ns` = number of source cells. Returns (index_sets, variables)."
function fields_regrid_spec(dom, ns::Int, field_names)
    nx, ny, dx = dom.nx, dom.ny, dom.dx
    xlo = dom.xmin - dx / 2
    ylo = dom.ymin - dx / 2
    xstep(s) = _opd("*", s, _opd("+", _opd("==", "v", 2), _opd("==", "v", 3)))
    ystep(s) = _opd("*", s, _opd("+", _opd("==", "v", 3), _opd("==", "v", 4)))
    txc = _opd("+", xlo, _opd("*", _opd("-", "x", 1), dx), xstep(dx))
    tyc = _opd("+", ylo, _opd("*", _opd("-", "y", 1), dx), ystep(dx))
    tgt = _arrd(["x", "y", "v", "c"], ["x" => "gx", "y" => "gy", "v" => "verts", "c" => "coord"],
                _opd("ifelse", _opd("==", "c", 1), txc, tyc))
    ip = Dict{String, Any}("op" => "intersect_polygon", "id" => "regrid_clip", "manifold" => "planar",
        "args" => Any[_ixd("tgt_poly", "x", "y"), _ixd("src_poly", "s")])
    clip = _arrd(["x", "y", "s", "w", "c"],
        ["x" => "gx", "y" => "gy", "s" => "src_cells", "w" => "clip_ring", "c" => "coord"],
        _ixd(ip, "w", "c"))
    vp1 = _opd("+", "v", 1)
    shoe = _opd("*", 0.5, _opd("-",
        _opd("*", _ixd("clip", "x", "y", "s", "v", 1), _ixd("clip", "x", "y", "s", vp1, 2)),
        _opd("*", _ixd("clip", "x", "y", "s", vp1, 1), _ixd("clip", "x", "y", "s", "v", 2))))
    A_ij = _arrd(["x", "y", "s"], ["x" => "gx", "y" => "gy", "s" => "src_cells"],
                 _aggd("v", "clip_ring", shoe))
    A_j  = _arrd(["x", "y"], ["x" => "gx", "y" => "gy"], _aggd("s", "src_cells", _ixd("A_ij", "x", "y", "s")))
    O(e, shape...) = Dict{String, Any}("type" => "observed", "shape" => collect(Any, shape), "expression" => e)
    P(shape...) = Dict{String, Any}("type" => "parameter", "shape" => collect(Any, shape))
    isets = Dict{String, Any}(
        "coord" => Dict{String, Any}("kind" => "interval", "size" => 2),
        "verts" => Dict{String, Any}("kind" => "interval", "size" => 4),
        "gx" => Dict{String, Any}("kind" => "interval", "size" => nx),
        "gy" => Dict{String, Any}("kind" => "interval", "size" => ny),
        "src_cells" => Dict{String, Any}("kind" => "interval", "size" => ns),
        "clip_ring" => Dict{String, Any}("kind" => "derived", "from_faq" => "regrid_clip"))
    vars = Dict{String, Any}(
        "tgt_poly" => O(tgt, "gx", "gy", "verts", "coord"),
        "src_poly" => P("src_cells", "verts", "coord"),
        "clip" => O(clip, "gx", "gy", "src_cells", "clip_ring", "coord"),
        "A_ij" => O(A_ij, "gx", "gy", "src_cells"),
        "A_j" => O(A_j, "gx", "gy"))
    for fn in field_names
        src_fn = "F_src_$fn"
        num = _aggd("s", "src_cells", _opd("*", _ixd("A_ij", "x", "y", "s"), _ixd(src_fn, "s")))
        vars[src_fn] = P("src_cells")
        vars[fn] = O(_arrd(["x", "y"], ["x" => "gx", "y" => "gy"], _opd("/", num, _ixd("A_j", "x", "y"))),
                     "gx", "gy")
    end
    return isets, vars
end

"Recursively replace every bare operand equal to `name` (a string) in a JSON-AST
node with `repl`. Used to turn each discretized equation's bare field reference
(`u_x`, `dzdy`, …) into index(field, <loop vars>)."
function _replace_bare(node, name, repl)
    if node isa AbstractDict
        out = Dict{String, Any}()
        for (k, v) in node
            out[String(k)] = _replace_bare(v, name, repl)
        end
        return out
    elseif node isa AbstractVector
        return Any[x == name ? repl : _replace_bare(x, name, repl) for x in node]
    else
        return node == name ? repl : node
    end
end

"Attach the multi-field regrid to the discretized level-set esm: for each coupled
field rewrite its bare reference to index(field, <output loop vars>), merge the
regrid index_sets and variables, and drop the array-input parameters. Returns
(esm_dict, const_arrays, param_arrays) for build_evaluator: `const_arrays` carries
the source polygons, `param_arrays` the live per-field source buffers."
function couple_fields(disc, dom, src_poly, field_src::AbstractDict)
    esm = JSON3.read(JSON3.write(disc), Dict{String, Any})   # mutable copy
    model = first(values(esm["models"]))
    fields = collect(keys(field_src))
    for eq in model["equations"]
        rhs = eq["rhs"]
        oidx = rhs isa AbstractDict ? get(rhs, "output_idx", nothing) : nothing
        loop = oidx === nothing ? Any["i", "j"] : Any[String(s) for s in oidx]
        for fn in fields
            repl = Dict{String, Any}("op" => "index", "args" => Any[fn, loop...])
            eq["rhs"] = _replace_bare(eq["rhs"], fn, repl)
        end
    end
    ns = size(src_poly, 1)
    isets, rvars = fields_regrid_spec(dom, ns, fields)
    model["index_sets"] = Base.merge(get(model, "index_sets", Dict{String, Any}()), isets)
    for fn in fields
        delete!(model["variables"], fn)     # was the array-input parameter
    end
    for (k, v) in rvars
        model["variables"][k] = v
    end
    const_arrays = Dict{String, Any}("src_poly" => src_poly)
    param_arrays = Dict{String, Any}("F_src_$fn" => Float64[field_src[fn]...] for fn in fields)
    return esm, const_arrays, param_arrays
end

# ---------------------------------------------------------------------------
# WHOLE-SYSTEM stage: the Rothermel rate-of-spread PHYSICS chain (--full).
#
# `flatten(load(camp_fire.esm))` gives the full 21-component coupled system. The
# lossy-flatten blocker that used to prevent this lowering is FIXED in ESS: flatten
# is field-preserving (it no longer drops the regridders' `index_sets`, the
# FuelModelLookup `function_tables`, or intersect_polygon's `manifold`), and
# build_evaluator resolves DEEP observed chains on its own. `flattened_to_esm` even
# reconstitutes the whole system into one runnable document (verify_full_system §B).
#
# This stage extracts the SCALAR fire-physics chain — TerrainSlope, MidflameWind,
# EquilibriumMoistureContent, OneHourFuelMoisture, RothermelFireSpread — straight from
# the flattened equations, EMITS each as an observed, and runs it through
# build_evaluator (whose fixed-point observed resolver collapses the feed-forward
# chain — no hand topological pre-inline) to COMPUTE the Rothermel rate of spread R and
# the level-set coefficients (C/B/E_wind, beta_ratio, phi_s_coeff) from the regridded
# met/fuel/slope forcing — the values the driver otherwise holds at the constant --r0.
# FuelModelLookup's 5 outputs are supplied as the Anderson-13 FM1 (short grass) SI
# values its table emits for code 1 (the data-harness input for that component).
# ---------------------------------------------------------------------------

const PHYS_COMPONENTS = ["TerrainSlope", "MidflameWind", "EquilibriumMoistureContent",
                         "OneHourFuelMoisture", "RothermelFireSpread"]
# Anderson-13 FM1 (short grass) SI fuel-bed properties = FuelModelLookup outputs for code 1.
const FM1_FUEL = Dict{String,Float64}(
    "FuelModelLookup.sigma" => 11483.0, "FuelModelLookup.w_0" => 0.166,
    "FuelModelLookup.delta" => 0.3048, "FuelModelLookup.M_x" => 0.12,
    "FuelModelLookup.h" => 18608000.0)
# Each Rothermel read-out → the level-set parameter the camp_fire.esm coupling maps it to.
const ROTHERMEL_TO_LEVELSET = ["RothermelFireSpread.R" => "R_0",
    "RothermelFireSpread.C_coeff" => "C_wind", "RothermelFireSpread.B_coeff" => "B_wind",
    "RothermelFireSpread.E_coeff" => "E_wind", "RothermelFireSpread.beta_ratio" => "beta_ratio",
    "RothermelFireSpread.phi_s_coeff" => "phi_s_coeff"]

"Name an equation's LHS defines (a bare state/observed, or the integrand of D(x)/aggregate)."
function _eq_lhs_name(e)
    l = ESS.serialize_expression(e.lhs)
    l isa AbstractString && return String(l)
    if l isa AbstractDict && get(l, "op", "") in ("D", "index")
        a = l["args"][1]
        return a isa AbstractString ? String(a) : nothing
    end
    return nothing
end

"Collect every variable-name leaf in a serialized AST (skips op/structural keys)."
function _collect_refs!(acc, node)
    if node isa AbstractString
        push!(acc, String(node))
    elseif node isa AbstractDict
        for (k, v) in node
            k in ("op", "wrt", "id", "manifold", "semiring") || _collect_refs!(acc, v)
        end
    elseif node isa AbstractVector
        for x in node
            _collect_refs!(acc, x)
        end
    end
    return acc
end

"Compute the Rothermel rate of spread + level-set coefficients from the flattened
camp_fire system, given representative scalar forcing for the regridded coupling
inputs (`fieldvals`, keyed by the producer's flattened name, e.g. ERA5uRegrid.F_tgt).
Lifts the fire-physics algebraic equations OUT of the flattened system, emits each as
an OBSERVED, and lets build_evaluator's fixed-point observed resolver collapse the
feed-forward chain (ESS now resolves deep observed chains — no hand pre-inline).
Returns a Dict of the read-out values."
function rothermel_from_fields(flat, fieldvals::AbstractDict)
    pv = flat.parameters
    isphys(nm) = any(p -> startswith(nm, p * "."), PHYS_COMPONENTS)
    # Emit each fire-physics `x = expr` algebraic equation as an observed; the chain
    # is feed-forward, so build_evaluator inlines it transitively at build time.
    vars = Dict{String,Any}(); refs = Set{String}()
    for e in flat.equations
        nm = _eq_lhs_name(e)
        (nm === nothing || !isphys(nm)) && continue
        l = ESS.serialize_expression(e.lhs)
        l isa AbstractString || continue                   # bare `x = expr` algebraic def
        ex = ESS.serialize_expression(e.rhs)
        vars[nm] = Dict{String,Any}("type" => "observed", "expression" => ex)
        _collect_refs!(refs, ex)
    end

    # Probe states read each Rothermel read-out by name; build_evaluator inlines them.
    readouts = String.(first.(ROTHERMEL_TO_LEVELSET))
    eqs = Any[]; probe_of = Dict{String,String}()
    for (i, nm) in enumerate(readouts)
        haskey(vars, nm) || error("fire-physics read-out not found: $nm")
        pn = "__probe$i"; probe_of[nm] = pn
        vars[pn] = Dict{String,Any}("type" => "state")
        push!(eqs, Dict{String,Any}("lhs" => Dict{String,Any}("op" => "D", "args" => Any[pn], "wrt" => "t"),
                                    "rhs" => nm))
    end

    # Inputs the chain reads but doesn't define (regridder/lookup outputs, model
    # constants) become parameters: the FM1 fuel + regridded forcing where given,
    # else the flattened parameter's own default.
    forcing = Base.merge(Dict{String,Float64}(FM1_FUEL), Dict{String,Float64}(fieldvals))
    for r in refs
        (haskey(vars, r) || !occursin(".", r)) && continue
        d = Dict{String,Any}("type" => "parameter")
        if haskey(forcing, r)
            d["default"] = forcing[r]
        elseif haskey(pv, r) && pv[r].default !== nothing
            d["default"] = pv[r].default
        else
            d["default"] = 0.0
        end
        vars[r] = d
    end

    esm = Dict{String,Any}("esm" => "0.5.0",
        "metadata" => Dict{String,Any}("name" => "CampFireRothermelChain"),
        "models" => Dict{String,Any}("Phys" => Dict{String,Any}("variables" => vars, "equations" => eqs)))
    f!, u0, p, _t, vmap = build_evaluator(esm;
        initial_conditions = Dict{String,Float64}(pn => 0.0 for pn in values(probe_of)))
    du = similar(u0); f!(du, u0, p, 0.0)
    return Dict(nm => du[vmap[probe_of[nm]]] for nm in readouts)
end
# ---------------------------------------------------------------------------
# MONOLITHIC assembly — the whole flattened physics + level-set lowered and run
# in ONE evaluator (M2-5): keep the fire-physics + level-set components, seed the
# 6 regridder outputs as [x,y] parameter SEEDS, let ESS's promote=true lower the
# scalar Rothermel chain to per-cell, discretize with the ESD 2-D centered grad
# rule, then REPLACE the seed params with the conservative regridder so the regrid
# is computed inside the same ODE right-hand side. Shared by the driver + the
# verify_*.jl monolithic harnesses.
# ---------------------------------------------------------------------------
const _MONO_KEEP = ["TerrainSlope", "MidflameWind", "EquilibriumMoistureContent",
              "OneHourFuelMoisture", "RothermelFireSpread", "LevelSetFireSpread", "FuelConsumption"]
# Forcing name (clean, regridder-loop-var-safe) ← the camp_fire coupling field it produces.
const _MONO_FORCING_OF = ["temp" => "ERA5tRegrid.F_tgt", "rh" => "ERA5rRegrid.F_tgt",
                    "wu" => "ERA5uRegrid.F_tgt", "wv" => "ERA5vRegrid.F_tgt",
                    "sx" => "SlopeXRegrid.F_tgt", "sy" => "SlopeYRegrid.F_tgt"]
const _MONO_FORCING = [last(p) for p in _MONO_FORCING_OF]
fcomp(n) = String(split(n, ".")[1])

"Monolithic discretized doc on the real fire grid: keep the fire physics + level-set,
seed the 6 forcing as [x,y] params, and let ESS's promote=true lower the whole thing
with the ESD 2-D centered grad rule (renamed to the model's namespaced spacing)."
function fire_discretized(dom)
    flat = with_logger(ConsoleLogger(stderr, Logging.Error)) do
        ESS.flatten(ESS.load(CAMP_FIRE_ESM))
    end
    OD = ESS.OrderedDict; MV = ESS.ModelVariable
    keepeq = [eq for eq in flat.equations if (eq.lhs isa ESS.VarExpr ? fcomp(eq.lhs.name) in _MONO_KEEP :
              (eq.lhs isa ESS.OpExpr && eq.lhs.op == "D" && fcomp(eq.lhs.args[1].name) in _MONO_KEEP))]
    refs = Set{String}()
    collect_refs(e) = (e isa ESS.VarExpr ? push!(refs, e.name) :
        e isa ESS.OpExpr ? (foreach(collect_refs, e.args);
            for f in (e.expr_body, e.filter, e.lower, e.upper); f === nothing || collect_refs(f); end;
            e.values === nothing || foreach(collect_refs, e.values)) : nothing)
    for eq in keepeq; collect_refs(eq.lhs); collect_refs(eq.rhs); end
    keepvar(d) = OD{String,MV}(k => v for (k, v) in d if fcomp(k) in _MONO_KEEP && !(k in _MONO_FORCING))
    states = keepvar(flat.state_variables); params = keepvar(flat.parameters); obs = keepvar(flat.observed_variables)
    for f in _MONO_FORCING; params[f] = MV(ESS.ParameterVariable; shape = ["x", "y"]); end
    params["LevelSetFireSpread.dx"] = MV(ESS.ParameterVariable; default = dom.dx)
    defined = Set(keys(states)) ∪ Set(keys(params)) ∪ Set(keys(obs))
    for r in refs
        (r in defined || !occursin(".", r)) && continue
        params[r] = MV(ESS.ParameterVariable; default = get(FM1_FUEL, r, 0.0))
    end
    flat2 = ESS.FlattenedSystem(flat.independent_variables, states, params, obs, keepeq,
        ESS.ContinuousEvent[], ESS.DiscreteEvent[], flat.domain, flat.metadata,
        flat.index_sets, flat.function_tables)
    grids = Dict("g" => Dict{String,Any}("family" => "cartesian", "dimensions" => Any[
        Dict{String,Any}("name" => "x", "size" => dom.nx, "periodic" => false, "spacing" => "uniform"),
        Dict{String,Any}("name" => "y", "size" => dom.ny, "periodic" => false, "spacing" => "uniform")]))
    rules = gdd_to_rules(GRAD_2D; spacing = "LevelSetFireSpread.dx")
    return ESS.discretize(flat2; grids = grids, rules = rules,
                          strict_unrewritten = false, promote = true)
end

"RothermelFireSpread.R's declared shape in the discretized doc ([x,y] confirms the scalar
Rothermel chain was promoted to per-cell)."
function rothermel_shape(disc)
    for (_, m) in disc["models"]
        v = get(get(m, "variables", Dict{String,Any}()), "RothermelFireSpread.R", nothing)
        v === nothing && continue
        sh = get(v, "shape", nothing)
        return sh === nothing ? nothing : String[String(s) for s in sh]
    end
    return nothing
end

"Replace the 6 forcing PARAMETERS in the discretized monolithic doc with the conservative
regridder (shared geometry + one apply observed per field), so the regrid lowers in the
SAME evaluator. `field_src` maps each clean forcing name to its per-source-cell values.
Returns (const_arrays, param_arrays) for build_evaluator / simulate."
function merge_regridder!(disc, dom, src_poly, field_src)
    ns = size(src_poly, 1)
    fnames = collect(keys(field_src))
    isets, rvars = fields_regrid_spec(dom, ns, fnames)
    model = first(values(disc["models"]))
    model["index_sets"] = Base.merge(get(model, "index_sets", Dict{String,Any}()), isets)
    for fc in _MONO_FORCING; delete!(model["variables"], fc); end          # drop the seed params
    for k in ("tgt_poly", "src_poly", "clip", "A_ij", "A_j")         # shared geometry
        model["variables"][k] = rvars[k]
    end
    forcing_of = Dict(_MONO_FORCING_OF)
    for fn in fnames
        model["variables"]["F_src_$fn"] = rvars["F_src_$fn"]         # live source buffer
        model["variables"][forcing_of[fn]] = rvars[fn]               # apply observed (forcing name)
    end
    return Dict{String,Any}("src_poly" => src_poly),
           Dict{String,Any}("F_src_$fn" => Float64[field_src[fn]...] for fn in fnames)
end

"Seed the level-set signed-distance `psi` over the real fire grid from the domain's declared
IC expression (the generic ESS `seed_expression_ic!`). Returns a `(u0, var_map)->…` closure."
function psi_seeder(dom)
    X(i) = dom.xmin + (i - 1) * dom.dx
    Y(j) = dom.ymin + (j - 1) * dom.dx
    xs = [X(i) for i in 1:dom.nx]; ys = [Y(j) for j in 1:dom.ny]
    return (u0, vm) -> seed_expression_ic!(u0, vm, "LevelSetFireSpread.psi", dom.ic,
                                           ["x" => xs, "y" => ys])
end

"A coarse `nsx×nsy` representative source-field set over the fire domain: a W→E `wu` wind ramp
(drives the front anisotropy), constant temp/rh/slope. Returns (src_poly, field_src)."
function representative_forcing(dom; nsx = 3, nsy = 3, u_lo = 2.0, u_hi = 10.0)
    src_poly, ncol, nrow = source_grid_lcc(dom; nsx = nsx, nsy = nsy)
    ns = ncol * nrow
    col_of(s) = ((s - 1) % ncol) + 1
    u_col = collect(range(u_lo, u_hi; length = ncol))
    field_src = Dict{String,Vector{Float64}}(
        "temp" => fill(288.0, ns), "rh" => fill(0.23, ns),
        "wu" => [u_col[col_of(s)] for s in 1:ns], "wv" => zeros(ns),
        "sx" => fill(0.05, ns), "sy" => zeros(ns))
    return src_poly, field_src, ncol, nrow
end

# ---------------------------------------------------------------------------
# COMPUTED conservative regridder (Phase-5 residual #4) — drive the monolithic fire's
# 6 forcing fields through the REAL `conservative_regrid_overlap_join_computed.esm`
# (the component camp_fire.esm actually couples in), retiring the hand-rolled planar
# `fields_regrid_spec`. The component's overlap matrix A_ij is COMPUTED from the
# source/target polygon rings via the spherical intersect_polygon clip + the
# Van Oosterom-Strackee spherical-excess area, and the target denominator A_j_w is the
# partition-of-unity-exact bin-skolem row-sum (Phase 3b). We size it to the camp source→
# fire-grid target, extract its build-once WEIGHT MATRIX W (W·F_src = F_tgt; rows sum to
# 1), and apply W to each field — so the geometry is the real ESD engine, not reimplemented.
# ---------------------------------------------------------------------------

"Fire target cells in lon/lat: cell (i,j) center + its 4 corners (±dx/2 in the LCC frame)
inverse-projected to geographic. Flat target index t = (j-1)*nx + i (column-major in i).
Returns (tgt_poly[nt,4,2] CCW lon/lat, tgt_lon[nt], tgt_lat[nt])."
function fire_target_lonlat(dom)
    nx, ny, dx = dom.nx, dom.ny, dom.dx
    nt = nx * ny
    P = zeros(nt, 4, 2); clon = zeros(nt); clat = zeros(nt)
    for j in 1:ny, i in 1:nx
        t = (j - 1) * nx + i
        xc = dom.xmin + (i - 1) * dx; yc = dom.ymin + (j - 1) * dx
        corners = ((xc - dx/2, yc - dx/2), (xc + dx/2, yc - dx/2),
                   (xc + dx/2, yc + dx/2), (xc - dx/2, yc + dx/2))   # CCW
        for (k, (x, y)) in enumerate(corners)
            lo, la = lcc_inverse(x, y); P[t, k, 1] = lo; P[t, k, 2] = la
        end
        lo, la = lcc_inverse(xc, yc); clon[t] = lo; clat[t] = la
    end
    return P, clon, clat
end

"A coarse `nsx×nsy` representative source grid in lon/lat tiling the domain bbox (the
geographic counterpart of `representative_forcing`'s LCC grid), with the W→E `wu` wind
ramp + constant temp/rh/slope. Returns (src_poly[ns,4,2] lon/lat, src_lon, src_lat,
field_src, nsx, nsy); s = (sy-1)*nsx + sx."
function representative_forcing_lonlat(dom; nsx = 3, nsy = 3, u_lo = 2.0, u_hi = 10.0)
    lo0, lo1, la0, la1 = _domain_lonlat_bbox(dom; pad = 0.0)
    dlo = (lo1 - lo0) / nsx; dla = (la1 - la0) / nsy
    ns = nsx * nsy
    P = zeros(ns, 4, 2); slon = zeros(ns); slat = zeros(ns)
    u_col = collect(range(u_lo, u_hi; length = nsx))
    col_of(s) = ((s - 1) % nsx) + 1
    for sy in 1:nsy, sx in 1:nsx
        s = (sy - 1) * nsx + sx
        a = lo0 + (sx - 1) * dlo; b = lo0 + sx * dlo
        c = la0 + (sy - 1) * dla; d = la0 + sy * dla
        for (k, (lo, la)) in enumerate(((a, c), (b, c), (b, d), (a, d)))   # CCW
            P[s, k, 1] = lo; P[s, k, 2] = la
        end
        slon[s] = (a + b)/2; slat[s] = (c + d)/2
    end
    field_src = Dict{String,Vector{Float64}}(
        "temp" => fill(288.0, ns), "rh" => fill(0.23, ns),
        "wu" => [u_col[col_of(s)] for s in 1:ns], "wv" => zeros(ns),
        "sx" => fill(0.05, ns), "sy" => zeros(ns))
    return P, slon, slat, field_src, nsx, nsy
end

"Build-once WEIGHT MATRIX W[nt,ns] (W·F_src = F_tgt) of the REAL computed conservative
regridder, sized to this source(ns)→fire-grid(nt) geometry. Loads
`conservative_regrid_overlap_join_computed.esm`, resizes its src_cells/tgt_cells index
sets, widens the authored [1,3] ODE LHS ranges to [1,nt], and — because the geometry
(spherical clip → A_ij → A_j_w) is build-once SETUP — sweeps F_src over the unit basis
through ONE evaluator (mutating the F_src param buffer) to read each weight column.
The binning coords are shifted so the whole domain collapses to a single bin (the
broad-phase keeps every candidate pair; the sub-atol sliver filter drops non-overlaps),
which is exact for the small camp source. Rows of W sum to 1 (partition of unity)."
function computed_regrid_weights(src_poly, src_lon, src_lat, tgt_poly, tgt_lon, tgt_lat)
    ns = size(src_poly, 1); nt = size(tgt_poly, 1)
    raw = JSON3.read(read(REGRID_COMPUTED, String), Dict{String,Any})
    m = raw["models"]["ConservativeRegridOverlapJoinComputed"]
    m["index_sets"]["src_cells"]["size"] = ns
    m["index_sets"]["tgt_cells"]["size"] = nt
    for eq in m["equations"]                                   # widen authored [1,3] j-ranges
        lhs = eq["lhs"]
        lhs isa AbstractDict && get(lhs, "op", "") == "aggregate" &&
            haskey(lhs["ranges"], "j") && (lhs["ranges"]["j"] = Any[1, nt])
    end
    # Shift the binning coords so the whole source∪target extent collapses to a single bin
    # (no straddle): use the COMBINED min/extent, else a source cell west/south of the
    # target min floors to a negative bin and the equal-bin broad phase drops it.
    lo0 = min(minimum(src_lon), minimum(tgt_lon)); la0 = min(minimum(src_lat), minimum(tgt_lat))
    binw = max(maximum(src_lon), maximum(tgt_lon)) - lo0
    binw = max(binw, max(maximum(src_lat), maximum(tgt_lat)) - la0, 1.0) * 4
    ca = Dict{String,Any}("src_poly" => src_poly, "tgt_poly" => tgt_poly,
        "src_lon" => src_lon .- lo0, "src_lat" => src_lat .- la0,
        "tgt_lon" => tgt_lon .- lo0, "tgt_lat" => tgt_lat .- la0)
    fbuf = zeros(ns)
    ics = Base.merge(Dict("A_j[$j]" => 0.0 for j in 1:nt), Dict("F_tgt[$j]" => 0.0 for j in 1:nt))
    f!, u0, p, _t, vmap = build_evaluator(raw; model_name = "ConservativeRegridOverlapJoinComputed",
        initial_conditions = ics, const_arrays = ca, param_arrays = Dict("F_src" => fbuf),
        parameter_overrides = Dict("dx" => binw, "dy" => binw, "atol" => 1e-18))
    du = similar(u0)
    W = zeros(nt, ns)
    for s in 1:ns
        fill!(fbuf, 0.0); fbuf[s] = 1.0
        f!(du, u0, p, 0.0)
        for j in 1:nt; W[j, s] = du[vmap["F_tgt[$j]"]]; end
    end
    return W
end

"Apply the computed regrid weights to the representative source fields, producing the 6
monolithic forcing arrays keyed by their `_MONO_FORCING` names (`ERA5tRegrid.F_tgt`, …)
as `[nx,ny]` matrices — the param buffers `fire_discretized`'s seeded forcing reads."
function computed_forcing(dom, W, field_src)
    nx, ny = dom.nx, dom.ny
    forcing_of = Dict(_MONO_FORCING_OF)
    return Dict{String,Any}(forcing_of[fn] => reshape(W * field_src[fn], nx, ny)
                            for fn in keys(forcing_of))
end

"Convenience: the camp source geometry → computed weights → the 6 `[nx,ny]` forcing
param buffers, all from the REAL computed regridder. Returns (forcing::Dict, W, field_src)."
function computed_regrid_forcing(dom; nsx = 3, nsy = 3)
    tgt_poly, tgt_lon, tgt_lat = fire_target_lonlat(dom)
    src_poly, src_lon, src_lat, field_src, _, _ = representative_forcing_lonlat(dom; nsx = nsx, nsy = nsy)
    W = computed_regrid_weights(src_poly, src_lon, src_lat, tgt_poly, tgt_lon, tgt_lat)
    return computed_forcing(dom, W, field_src), W, field_src
end

# --- #5 data-refresh seam: a RegridApplier backed by the computed weights. Each forcing
#     buffer is refreshed in place as W·(native source field) at every cadence tick, so
#     real provider data (ERA5/LANDFIRE/USGS via to_esio_loader) drives the fire through
#     the SAME conservative weights. `weights[var]`/`source_var[var]` are per-forcing.
struct ComputedRegrid <: ESS.RegridApplier
    weights::Dict{String,Matrix{Float64}}     # forcing var → W[nt,ns]
    source_var::Dict{String,String}           # forcing var → native sample variable
    shape::Tuple{Int,Int}                     # (nx, ny) of each forcing buffer
end

function ESS.apply_regrid!(r::ComputedRegrid, buffer::Array{Float64},
                           var::AbstractString, sample)
    v = String(var)
    W = get(r.weights, v, nothing)
    W === nothing && throw(ESS.RefreshError("ComputedRegrid has no weights for '$v'"))
    src = vec(Float64.(ESS._regrid_field(sample, get(r.source_var, v, v))))
    # Missing-value fill: real provider data can carry NaN/Inf (e.g. an ERA5 fill cell);
    # since 0·NaN = NaN would poison even a zero-weight column, replace non-finite source
    # values with the finite mean (a neutral fill) before the conservative apply.
    if any(!isfinite, src)
        fin = filter(isfinite, src)
        m = isempty(fin) ? 0.0 : sum(fin) / length(fin)
        src = [isfinite(x) ? x : m for x in src]
    end
    buffer .= reshape(W * src, size(buffer))
    return buffer
end

