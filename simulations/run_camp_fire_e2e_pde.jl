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
function levelset_pde_esm(ls_path, dom; wind::Bool = false)
    doc = JSON3.read(read(ls_path, String), Dict{String, Any})
    model = doc["models"]["LevelSetFireSpread"]
    vars = model["variables"]

    if wind
        # u_x becomes a per-cell field fed by the wind regrid (attached AFTER
        # discretize). Declaring it grid-shaped makes discretize keep it as an array
        # INPUT — a bare reference inside the scalarized D(psi,t) arrayop that the
        # couple step rewrites to index(u_x,i,j); build_evaluator then inlines the
        # regrid per cell. (u_y stays the scalar-0 parameter, so U_n = u_x·ψ_x/|∇ψ|.)
        vars["u_x"]["shape"] = Any["x", "y"]
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
# Wind coupling — a declarative conservative regrid that drives the level-set's
# per-cell eastward midflame wind u_x from a source wind field, computed THROUGH
# the tree-walk evaluator (no host-supplied geometry). This is the first met→fire
# field wired end-to-end: ERA5 surface wind → regrid → u_x[x,y] → U_n → φ_W → S_n
# → D(psi,t). The source grid here is a coarse band in the fire's own LCC frame
# (a west→east wind ramp); swapping in ERA5's lat/lon grid + reprojection +
# live loader data is the remaining "add the rest" step — the regrid machinery
# (clip → A_ij → A_j → u_x, inlined per cell) is identical.
# ---------------------------------------------------------------------------

_ixd(args...) = Dict{String, Any}("op" => "index", "args" => collect(Any, args))
_opd(o, args...) = Dict{String, Any}("op" => o, "args" => collect(Any, args))
_aggd(lv, set, body) = Dict{String, Any}("op" => "aggregate", "semiring" => "sum_product",
    "output_idx" => Any[], "ranges" => Dict{String, Any}(lv => Dict{String, Any}("from" => set)),
    "args" => Any[], "expr" => body)
_arrd(oidx, ranges, body) = Dict{String, Any}("op" => "arrayop", "output_idx" => collect(Any, oidx),
    "ranges" => Dict{String, Any}(k => Dict{String, Any}("from" => v) for (k, v) in ranges),
    "args" => Any[], "expr" => body)

"Index_sets + variables for the conservative regrid of an eastward-wind source
field (`ns` cells banded along x, spanning the full y extent) onto the fire grid,
producing `u_x[x,y]`. Cell polygons are CONSTRUCTED from the grid origin/spacing
(cell-centered on the nodes X(i),Y(j)); A_ij = polygon_area(intersect_polygon(...))
ranges over (x,y,s). Returns (index_sets, variables, F_src)."
function wind_regrid_spec(dom; ns::Int = 3, uvals = collect(range(2.0, 8.0; length = ns)))
    nx, ny, dx = dom.nx, dom.ny, dom.dx
    xlo = dom.xmin - dx / 2          # fire-domain west/south edges (cells centered on nodes)
    ylo = dom.ymin - dx / 2
    dsx = nx * dx / ns               # source cell width in x
    dys = ny * dx                    # source spans the full y extent
    xstep(s) = _opd("*", s, _opd("+", _opd("==", "v", 2), _opd("==", "v", 3)))
    ystep(s) = _opd("*", s, _opd("+", _opd("==", "v", 3), _opd("==", "v", 4)))
    txc = _opd("+", xlo, _opd("*", _opd("-", "x", 1), dx), xstep(dx))
    tyc = _opd("+", ylo, _opd("*", _opd("-", "y", 1), dx), ystep(dx))
    tgt = _arrd(["x", "y", "v", "c"], ["x" => "gx", "y" => "gy", "v" => "verts", "c" => "coord"],
                _opd("ifelse", _opd("==", "c", 1), txc, tyc))
    sxc = _opd("+", xlo, _opd("*", _opd("-", "s", 1), dsx), xstep(dsx))
    syc = _opd("+", ylo, ystep(dys))
    src = _arrd(["s", "v", "c"], ["s" => "src_cells", "v" => "verts", "c" => "coord"],
                _opd("ifelse", _opd("==", "c", 1), sxc, syc))
    ip = Dict{String, Any}("op" => "intersect_polygon", "id" => "wind_clip", "manifold" => "planar",
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
    num  = _aggd("s", "src_cells", _opd("*", _ixd("A_ij", "x", "y", "s"), _ixd("F_src", "s")))
    u_x  = _arrd(["x", "y"], ["x" => "gx", "y" => "gy"], _opd("/", num, _ixd("A_j", "x", "y")))
    O(e, shape...) = Dict{String, Any}("type" => "observed", "shape" => collect(Any, shape), "expression" => e)
    isets = Dict{String, Any}(
        "coord" => Dict{String, Any}("kind" => "interval", "size" => 2),
        "verts" => Dict{String, Any}("kind" => "interval", "size" => 4),
        "gx" => Dict{String, Any}("kind" => "interval", "size" => nx),
        "gy" => Dict{String, Any}("kind" => "interval", "size" => ny),
        "src_cells" => Dict{String, Any}("kind" => "interval", "size" => ns),
        "clip_ring" => Dict{String, Any}("kind" => "derived", "from_faq" => "wind_clip"))
    vars = Dict{String, Any}(
        "F_src" => Dict{String, Any}("type" => "parameter", "shape" => Any["src_cells"]),
        "tgt_poly" => O(tgt, "gx", "gy", "verts", "coord"),
        "src_poly" => O(src, "src_cells", "verts", "coord"),
        "clip" => O(clip, "gx", "gy", "src_cells", "clip_ring", "coord"),
        "A_ij" => O(A_ij, "gx", "gy", "src_cells"),
        "A_j" => O(A_j, "gx", "gy"),
        "u_x" => O(u_x, "gx", "gy"))
    return isets, vars, Float64[uvals...]
end

"Recursively replace every bare operand equal to `name` (a string) in a JSON-AST
node with `repl`. Used to turn the discretized equation's bare `u_x` reference
into index(u_x, <loop vars>)."
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

"Attach the wind regrid to the discretized level-set esm: rewrite the bare `u_x`
in each equation to index(u_x, <output loop vars>), merge the regrid index_sets
and variables, and replace the array-input `u_x` with the regrid observed. Returns
(esm_dict, F_src) ready for build_evaluator(...; param_arrays=Dict(\"F_src\"=>F_src))."
function couple_wind(disc, dom; ns::Int = 3, uvals = collect(range(2.0, 8.0; length = ns)))
    esm = JSON3.read(JSON3.write(disc), Dict{String, Any})   # mutable copy
    model = first(values(esm["models"]))
    for eq in model["equations"]
        rhs = eq["rhs"]
        oidx = rhs isa AbstractDict ? get(rhs, "output_idx", nothing) : nothing
        loop = oidx === nothing ? Any["i", "j"] : Any[String(s) for s in oidx]
        repl = Dict{String, Any}("op" => "index", "args" => Any["u_x", loop...])
        eq["rhs"] = _replace_bare(rhs, "u_x", repl)
    end
    isets, rvars, F_src = wind_regrid_spec(dom; ns = ns, uvals = uvals)
    model["index_sets"] = Base.merge(get(model, "index_sets", Dict{String, Any}()), isets)
    delete!(model["variables"], "u_x")     # was the array-input parameter
    for (k, v) in rvars
        model["variables"][k] = v
    end
    return esm, F_src
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
    wind_ns = 3
    wind_uvals = collect(range(2.0, 8.0; length = wind_ns))   # west→east midflame-wind ramp (m/s)

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
    if wind
        # Attach the declarative conservative wind regrid to the discretized core:
        # u_x[x,y] = regrid(F_src) is built through build_evaluator (clip → A_ij →
        # A_j → u_x, inlined per cell) and feeds U_n → φ_W → S_n → D(psi,t).
        disc, wind_Fsrc = couple_wind(disc, dom; ns = wind_ns, uvals = wind_uvals)
        println("   level-set PDE -> $sysclass; $(dom.nx * dom.ny) ODE states + declarative " *
                "wind regrid ($(wind_ns) src cells, u_x ramp $(round(first(wind_uvals); digits=1))" *
                "→$(round(last(wind_uvals); digits=1)) m/s W→E), R_0=$(r0) m/s")
    else
        println("   level-set PDE -> $sysclass; $(dom.nx * dom.ny) ODE states " *
                "(Rothermel/wind/slope coupling held at constants, R_0=$(r0) m/s)")
    end

    # 3. RUN — integrate the ArrayOp ODE; seed psi from the declared signed-distance
    #    IC, evaluated per cell. Loaders/coupling held at representative constants.
    println("== run (ESS build_evaluator; Tsit5); tspan=(0, $(round(seconds; digits = 0)) s) ==")
    f!, u0, p, _tspan, var_map = wind ?
        build_evaluator(disc; parameter_overrides = Dict("R_0" => r0),
                        param_arrays = Dict("F_src" => wind_Fsrc)) :
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
        # The regridded eastward wind must enhance the DOWNWIND (east) front
        # (U_n = u_x·ψ_x/|∇ψ| > 0) and leave the UPWIND (west) front at R_0
        # (U_n < 0 ⇒ max(0,U_n)=0). Split the front band at the ignition column to
        # show the regrid-driven anisotropy is real, not a uniform speed-up.
        i0 = round(Int, (-2000072.1 - dom.xmin) / dom.dx) + 1
        ew(side) = [(-du0[var_map["psi[$i,$j]"]]) for i in 1:dom.nx, j in 1:dom.ny
                    if haskey(var_map, "psi[$i,$j]") &&
                       abs(u0[var_map["psi[$i,$j]"]]) <= dom.dx && side(i, i0)]
        e = ew((i, i0) -> i > i0); w = ew((i, i0) -> i < i0)
        em = isempty(e) ? NaN : sum(e) / length(e)
        wm = isempty(w) ? NaN : sum(w) / length(w)
        println("   wind anisotropy (regridded u_x $(round(first(wind_uvals);digits=1))→" *
                "$(round(last(wind_uvals);digits=1)) m/s W→E): downwind/E front " *
                "~$(round(em; digits = 2)) m/s vs upwind/W front ~$(round(wm; digits = 2)) m/s " *
                "(≈R_0) — the regridded met field drives the front per cell")
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
