#!/usr/bin/env julia
"""
Camp Fire end-to-end (Julia) — a THIN driver over the real EarthSciSerialization.jl
pipeline. The Julia counterpart of `run_camp_fire_e2e_pde.py`: same three stages on
the same coupled `camp_fire.esm`:

    load        load(camp_fire.esm) + flatten          — assemble the 11 components
                                                          and resolve the coupling edges
    discretize  discretize(esm, centered |grad psi|)   — lower the 2-D level-set
                                                          Hamilton-Jacobi front PDE
    run         build_evaluator + solve (Tsit5)        — integrate the ArrayOp ODE

WHY JULIA. The README marks the full 2-D level-set PDE core as "the deferred Julia
campaign": the Python `simulate()` raises `UnsupportedDimensionalityError` on a
multi-D spatial system, so the Hamilton-Jacobi front is integrated *here*. This
driver lowers and integrates that PDE core at the REAL Camp Fire domain
(19x21 @ 2 km, LCC) seeded by the domain's declared signed-distance `psi` initial
condition — the part Python defers.

FAITHFUL TO THE PYTHON SIBLING's "thin driver, surface gaps" philosophy. The full
coupled system does not yet run end-to-end through the Julia binding; this driver
drives the real pipeline as far as it goes and prints the remaining Julia-binding
gaps — it does not pretend to reproduce the fire:

  * LOAD — top-level model `{ref}` stubs (now 21 component references) are
    resolved by `load()` itself (EarthSciSerialization.jl, schema §4.7
    `models.* = oneOf [Model, {ref}]`, incl. the `{"ref","model"}` selector for
    multi-model component libraries), so the LOAD stage is a plain
    `flatten(load(camp_fire.esm))` — no driver-side shim.
  * LOAD — reproject/regrid is no longer an `interfaces` block (which `flatten()`
    rejected for `bilinear`): `camp_fire.esm` now wires the EarthSciDiscretizations
    reprojection (`lambert_conformal`), lev=min reduction (`lev_min_surface_reduce`)
    and regridding (`conservative_regrid_overlap_join` / `bspline_regrid`)
    components into the coupling itself, so the coupled system flattens with no
    interface. (The regrid kernels' geometry — overlap matrix, B-spline offsets —
    is host-supplied, so this is a complete declarative spec; the coupled met→fire
    path is structural, not yet runnable through the tree-walk evaluator.)
  * DISCRETIZE/RUN — there is no Julia `flattened_to_esm`, so the DISCRETIZE/RUN
    stages drive the level-set PDE *core* (the slope/Rothermel coupling inputs
    held at representative constants, `--r0` the base spread rate). The remaining
    coupling inputs are scalar in the core; `--wind` couples ONE met field for
    real — see below.

  * `--wind` — MET→FIRE COUPLING (the first data-driven field wired end-to-end
    through the evaluator). A declarative conservative regrid (clip → A_ij → A_j
    → u_x, computed THROUGH `build_evaluator` with no host-supplied geometry; the
    cell polygons are CONSTRUCTED from the grid origin/spacing) maps a surface
    eastward-wind field onto the fire grid as a per-cell `u_x[x,y]`, which feeds
    `U_n → φ_W → S_n → D(psi,t)`. discretize() keeps `u_x` as a grid-shaped INPUT;
    the driver rewrites the bare reference to `index(u_x,i,j)` and attaches the
    regrid, then the evaluator inlines it per cell (the ess-14f.4 coupled-array
    bridge). The front is wind-elongated DOWNWIND (east, U_n>0) and stays at R_0
    UPWIND (west, U_n<0). The source wind here is a coarse W→E ramp in the fire's
    OWN LCC frame; swapping in ERA5's lat/lon grid + reprojection + live loader
    data (and routing through camp_fire.esm's ERA5uRegrid/MidflameWind components,
    rather than this driver-built regrid) is the remaining "add the rest" step —
    the regrid machinery is identical.

The integration window is SHORT by design (the Python `--seconds` analogue): the
goal is to drive the real pipeline and surface feasibility, not to reproduce the
16 h fire.

USAGE
    EARTHSCISERIALIZATION=.../EarthSciSerialization \\
        julia simulations/run_camp_fire_e2e_pde.jl [--seconds 30] [--r0 0.71] [--wind]

`EARTHSCISERIALIZATION` defaults to a sibling `../EarthSciSerialization` checkout.
The script activates that package's `scripts/pde_sim_adapter` project (which pins
EarthSciSerialization + OrdinaryDiffEqTsit5 + JSON3), so it runs with a plain
`julia` and no global package installs — mirroring the Python `PYTHONPATH` recipe.
"""

# ---------------------------------------------------------------------------
# Environment bootstrap (mirrors EarthSciSerialization.jl/scripts/pde_simulation_adapter.jl):
# activate the dedicated adapter project, dev the local ESS package on a fresh
# checkout, then instantiate. Warm runs (Manifest present) are a fast resolve.
# ---------------------------------------------------------------------------
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
    Pkg.instantiate(; io = devnull)
end

using EarthSciSerialization
using JSON3
using Logging
import OrdinaryDiffEqTsit5
const ODE = OrdinaryDiffEqTsit5
const ESS = EarthSciSerialization

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"Resolve the LevelSetFireSpread component path from camp_fire.esm's own `{ref}`."
function levelset_ref_path(camp_path)
    raw = JSON3.read(read(camp_path, String), Dict{String, Any})
    ref = raw["models"]["LevelSetFireSpread"]["ref"]
    return abspath(joinpath(dirname(camp_path), String(ref)))
end

"Centered 2nd-order finite-difference rules for `grad(psi, x|y)` on a 2-D Cartesian
grid (canonical loop names i, j; both dims indexed). The Godunov upwind |grad psi|
the Python uses needs an array-typed intermediate the tree-walk evaluator cannot
hold, so the front uses the centered form — adequate for the short window here."
function grad_rules_2d()
    idx(di, dj) = Dict{String, Any}("op" => "index", "args" => Any["\$u",
        di == 0 ? "i" : Dict{String, Any}("op" => "+", "args" => Any["i", di]),
        dj == 0 ? "j" : Dict{String, Any}("op" => "+", "args" => Any["j", dj])])
    centered(a, b, h) = Dict{String, Any}("op" => "/",
        "args" => Any[Dict{String, Any}("op" => "-", "args" => Any[a, b]),
                      Dict{String, Any}("op" => "*", "args" => Any[2, h])])
    [Dict{String, Any}("name" => "grad_x",
        "pattern" => Dict{String, Any}("op" => "grad", "args" => Any["\$u"], "dim" => "x"),
        "replacement" => centered(idx(1, 0), idx(-1, 0), "dx")),
     Dict{String, Any}("name" => "grad_y",
        "pattern" => Dict{String, Any}("op" => "grad", "args" => Any["\$u"], "dim" => "y"),
        "replacement" => centered(idx(0, 1), idx(0, -1), "dx"))]
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
        "rules" => grad_rules_2d())
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
# Driver
# ---------------------------------------------------------------------------

function parse_cli(args)
    seconds = 30.0
    r0 = 0.71            # representative Anderson FM1 grass spread, ~3 m/s midflame wind
    wind = false         # --wind: drive u_x from the declarative conservative regrid
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--seconds" && i < length(args)
            seconds = parse(Float64, args[i + 1]); i += 2
        elseif a == "--r0" && i < length(args)
            r0 = parse(Float64, args[i + 1]); i += 2
        elseif startswith(a, "--seconds=")
            seconds = parse(Float64, split(a, "=", limit = 2)[2]); i += 1
        elseif startswith(a, "--r0=")
            r0 = parse(Float64, split(a, "=", limit = 2)[2]); i += 1
        elseif a == "--wind"
            wind = true; i += 1
        else
            i += 1
        end
    end
    (seconds, r0, wind)
end

function main(args = ARGS)
    seconds, r0, wind = parse_cli(args)
    dom = camp_domain(CAMP_FIRE_ESM)
    # Coupled source fields on a coarse 3×3 source grid (the demo forcing): an
    # eastward midflame-wind ramp (W→E, drives φ_W) and an upslope terrain gradient
    # (S→N, drives φ_S). u_y / dzdx are wired to zero — proving all four fields plumb
    # through the same regrid, two with a visible, separable effect on the front.
    nsx = nsy = 3
    ux_col   = collect(range(2.0, 8.0; length = nsx))    # W→E midflame wind (m/s)
    dzdy_row = collect(range(0.0, 0.5; length = nsy))    # S→N terrain slope (rise/run)
    sx_of(s) = ((s - 1) % nsx) + 1
    sy_of(s) = ((s - 1) ÷ nsx) + 1
    ns_src = nsx * nsy
    field_src = Dict(
        "u_x"  => [ux_col[sx_of(s)] for s in 1:ns_src],
        "u_y"  => zeros(ns_src),
        "dzdx" => zeros(ns_src),
        "dzdy" => [dzdy_row[sy_of(s)] for s in 1:ns_src])
    phi_s_coeff = 5.275                                   # slope-factor coeff (β=1); 0 disables φ_S

    println("Camp Fire — thin Julia driver " *
            "(load camp_fire.esm -> discretize via ESS.jl -> run via tree-walk + Tsit5)\n")

    # 1. LOAD — assemble the real coupled system and resolve its coupling edges.
    #    load() resolves all component {ref} stubs (incl. the ESD regrid/reproject
    #    components wired into the coupling); no driver-side shim — flatten passes
    #    with no interface (see docstring).
    println("== load ==")
    # Quiet the v0.2.0 domain-BC deprecation notes during load (the .esm carries
    # domain-level boundary_conditions) — mirrors the Python sibling's
    # warnings.filterwarnings("ignore"); errors still surface.
    f = with_logger(ConsoleLogger(stderr, Logging.Error)) do
        load(CAMP_FIRE_ESM)
    end
    flat = flatten(f)
    defined = Set{String}()
    for eq in flat.equations
        lhs = eq.lhs
        lhs isa ESS.VarExpr && push!(defined, lhs.name)
        if lhs isa ESS.OpExpr && lhs.op == "D" && !isempty(lhs.args) &&
           lhs.args[1] isa ESS.VarExpr
            push!(defined, lhs.args[1].name)
        end
    end
    loader_fed = count(n -> !(n in defined), keys(flat.observed_variables))
    println("   $(length(f.models)) components, $(length(flat.equations)) equations, " *
            "$(length(flat.state_variables)) states, $(length(flat.parameters)) parameters, " *
            "$(length(flat.observed_variables)) observeds ($loader_fed loader-fed)")
    println("   domain $(dom.nx)x$(dom.ny) @ $(round(Int, dom.dx)) m  " *
            "($(dom.tstart) -> $(dom.tend))")
    println("   [load() resolved $(length(f.models)) component {ref} stubs " *
            "(incl. ESD reproject/reduce/regrid wired into the coupling); no interface]")

    # 2. DISCRETIZE — lower the 2-D level-set Hamilton-Jacobi front PDE (the part
    #    Python defers) with centered |grad psi| stencils, on the real grid.
    println("== discretize (centered 2nd-order |grad psi| stencils) ==")
    ls_path = levelset_ref_path(CAMP_FIRE_ESM)
    esm = levelset_pde_esm(ls_path, dom; wind = wind)
    disc = discretize(esm)
    sysclass = get(disc["metadata"], "system_class", "?")
    local wind_ca, wind_pa
    if wind
        # Attach the declarative conservative regrid (shared geometry, one field per
        # coupled input) to the discretized core: each field[x,y] = regrid(F_src_field)
        # is built through build_evaluator (clip → A_ij → A_j → field, inlined per
        # cell) and feeds U_n/tan_phi_n → φ_W/φ_S → S_n → D(psi,t).
        src_poly, _, _ = source_grid_lcc(dom; nsx = nsx, nsy = nsy)
        disc, wind_ca, wind_pa = couple_fields(disc, dom, src_poly, field_src)
        println("   level-set PDE -> $sysclass; $(dom.nx * dom.ny) ODE states + declarative " *
                "$(nsx)×$(nsy)-cell conservative regrid of $(length(COUPLED_FIELDS)) fields " *
                "(u_x $(round(first(ux_col);digits=1))→$(round(last(ux_col);digits=1)) m/s W→E, " *
                "dzdy 0→$(round(last(dzdy_row);digits=2)) S→N), R_0=$(r0) m/s, φ_s=$(phi_s_coeff)")
    else
        println("   level-set PDE -> $sysclass; $(dom.nx * dom.ny) ODE states " *
                "(Rothermel/wind/slope coupling held at constants, R_0=$(r0) m/s)")
    end

    # 3. RUN — integrate the ArrayOp ODE; seed psi from the declared signed-distance
    #    IC, evaluated per cell. Loaders/coupling held at representative constants.
    println("== run (ESS build_evaluator; Tsit5); tspan=(0, $(round(seconds; digits = 0)) s) ==")
    f!, u0, p, _tspan, var_map = wind ?
        build_evaluator(disc; parameter_overrides = Dict("R_0" => r0, "phi_s_coeff" => phi_s_coeff),
                        const_arrays = wind_ca, param_arrays = wind_pa) :
        build_evaluator(disc; parameter_overrides = Dict("R_0" => r0))
    X(i) = dom.xmin + (i - 1) * dom.dx
    Y(j) = dom.ymin + (j - 1) * dom.dx
    for i in 1:dom.nx, j in 1:dom.ny
        k = get(var_map, "psi[$i,$j]", nothing)
        k === nothing && continue
        u0[k] = ESS.evaluate_expr(dom.ic, Dict("x" => X(i), "y" => Y(j)))
    end
    burn0 = count(<(0), u0)
    # Burned area as a smooth Heaviside cell-fraction H(-psi) = 1/(1+exp(2 psi/eps_h)),
    # eps_h = dx — the same smooth gate camp_fire.esm uses to couple psi to fuel
    # consumption. It grows continuously even when the front advance is sub-grid,
    # so it tracks the front where an integer psi<0 cell count cannot.
    burned(u) = sum(ui -> 1.0 / (1.0 + exp(2.0 * ui / dom.dx)), u)

    t0 = time()
    prob = ODE.ODEProblem(f!, u0, (0.0, seconds), p)
    sol = ODE.solve(prob, ODE.Tsit5(); reltol = 1e-4, abstol = 1e-6)
    dt = time() - t0
    if sol.retcode != ODE.SciMLBase.ReturnCode.Success
        println("   simulate did not complete ($(round(dt; digits = 0))s): " *
                "retcode=$(sol.retcode)")
        return 1
    end
    # Instantaneous spread rate at the psi=0 contour: D(psi,t) = -S_n*|grad psi|^2,
    # so -du at the front (|grad psi|~1 for a signed-distance field) is the front
    # speed ~ R_0. Measured at the contour, not the cone tip — centered |grad psi|
    # vanishes at the tip (the Godunov upwind the Python uses fixes that; here the
    # tip stalls, the front itself still advances at the Rothermel rate).
    du0 = similar(u0)
    f!(du0, u0, p, 0.0)
    band = [k for k in eachindex(u0) if abs(u0[k]) <= dom.dx]
    rate = isempty(band) ? NaN : maximum(-du0[k] for k in band)
    burn1 = count(<(0), sol.u[end])
    println("   solved in $(round(dt; digits = 0))s: $(length(sol.t)) steps over " *
            "$(round(seconds; digits = 0)) s, $(length(u0)) state elements")
    println("   fire front: spreads ~$(round(rate; digits = 2)) m/s at the psi=0 contour " *
            "(Rothermel R_0=$(r0) m/s); burned area " *
            "$(round(burned(u0); digits = 2)) -> $(round(burned(sol.u[end]); digits = 2)) cells " *
            "(smooth), psi<0 cells $burn0 -> $burn1")
    if wind
        # Each regridded field must drive its own front anisotropy: the eastward
        # wind enhances the DOWNWIND (east) front (U_n>0) and the upslope gradient
        # enhances the UPHILL (north) front (tan_phi_n>0); the opposite sides stay
        # at ≈R_0 (max(0,·)=0). Split the front band about the ignition cell on each
        # axis to show the regrid-driven anisotropy is real and per-field separable.
        # The eastward wind (the dominant field) enhances the DOWNWIND/E front
        # (U_n>0) and leaves the UPWIND/W front at ≈R_0 (max(0,U_n)=0) — the clean,
        # field-driven anisotropy. (Slope is ~100× weaker than wind here and the
        # ignition circle is only ~1.5 cells across, so the two fields can't be
        # separated spatially on one run; verify_coupling.jl ablates each in turn.)
        i0 = round(Int, (-2000072.1 - dom.xmin) / dom.dx) + 1
        spd(i, j) = -du0[var_map["psi[$i,$j]"]]
        infront(i, j) = haskey(var_map, "psi[$i,$j]") && abs(u0[var_map["psi[$i,$j]"]]) <= dom.dx
        half(sel) = (v = [spd(i, j) for i in 1:dom.nx, j in 1:dom.ny if infront(i, j) && sel(i, j)];
                     isempty(v) ? NaN : sum(v) / length(v))
        eE = half((i, j) -> i > i0); eW = half((i, j) -> i < i0)
        fieldlist = join(COUPLED_FIELDS, ",")
        println("   $(length(COUPLED_FIELDS)) fields coupled through the regrid " *
                "($fieldlist). wind anisotropy (u_x W→E): downwind/E " *
                "~$(round(eE;digits=2)) m/s vs upwind/W ~$(round(eW;digits=2)) m/s (≈R_0) — " *
                "the regridded field drives the front per cell (per-field ablation: verify_coupling.jl)")
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
