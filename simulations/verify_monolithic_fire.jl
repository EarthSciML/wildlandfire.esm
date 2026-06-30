#!/usr/bin/env julia
#=
SINGLE-build_evaluator monolithic fire (M2-5) — regrid + per-cell Rothermel physics +
level-set front, all lowered and integrated in ONE build_evaluator.

verify_regridded_fire.jl (M2-4) proved the real conservative regridder can FORCE the
monolithic promoted fire, but in TWO stages: one build_evaluator computed the regridded
fields, a second integrated the fire. This driver fuses them: the conservative regridder
(clip → A_ij → A_j → F_tgt = Σ_s A_ij·F_src/A_j) lives in the SAME evaluator as the
shape-promoted physics and the level-set PDE. No intermediate readout — the per-cell
forcing is computed and consumed inside one ODE right-hand side.

The fusion exploits a property of `promote_downstream_shapes`: it seeds shape inference
from a variable's DECLARED shape, regardless of whether that variable is a parameter or
an observed. So the pipeline is verify_spatial_fire's verbatim — keep the 6 forcing as
`[x,y]` parameter SEEDS, fold the level-set observeds, algebraic_states_to_observeds,
promote, discretize, index_promoted_refs! — which leaves the physics reading the forcing
as `index(F_tgt, i, j)` per cell. THEN, post-discretize, the 6 forcing parameters are
REPLACED by the regridder: its shared geometry (one tgt_poly/clip/A_ij/A_j) plus one
apply observed per forcing field, merged into the discretized document exactly as the
driver's proven `couple_fields` merges a regrid into the discretized level-set. The
forcing references — already `index(F_tgt, …)` — now inline the regridder's per-cell
apply. The conservative weights A_ij/A_j materialize once at setup; F_tgt stays a runtime
observed because it reads the live F_src field. One build_evaluator, one ODE.

Asserts: Rothermel R promotes to [x,y]; the front speed varies per cell; the downwind/E
front is faster than upwind/W under the regridded W→E wind ramp — and ALL of it from a
single build_evaluator whose const/param inputs are only the source polygons and the
source field buffers (no precomputed forcing arrays).

Run with plain `julia` (the included run_camp_fire_e2e_pde.jl activates the ESS
pde_sim_adapter env; set EARTHSCISERIALIZATION to override).
=#
include(joinpath(@__DIR__, "run_camp_fire_e2e_pde.jl"))  # env bootstrap + regridder helpers + ESS
import GeometryOps, GeoInterface
const E = ESS

KEEP = ["TerrainSlope", "MidflameWind", "EquilibriumMoistureContent", "OneHourFuelMoisture",
        "RothermelFireSpread", "LevelSetFireSpread", "FuelConsumption"]
# Forcing name (camp_fire coupling) ← the clean regridder field that produces it. Clean
# names avoid the regridder's single-char loop vars (x,y,s,w,c,v); a field named "v"
# would alias the shoelace's clip_ring agg var and empty the geometry-setup closure.
FORCING_OF = ["temp" => "ERA5tRegrid.F_tgt", "rh" => "ERA5rRegrid.F_tgt",
              "wu" => "ERA5uRegrid.F_tgt", "wv" => "ERA5vRegrid.F_tgt",
              "sx" => "SlopeXRegrid.F_tgt", "sy" => "SlopeYRegrid.F_tgt"]
FORCING = [last(p) for p in FORCING_OF]
FM1 = Dict("FuelModelLookup.sigma" => 11483.0, "FuelModelLookup.w_0" => 0.166,
           "FuelModelLookup.delta" => 0.3048, "FuelModelLookup.M_x" => 0.12,
           "FuelModelLookup.h" => 18608000.0)
fcomp(n) = String(split(n, ".")[1])
fop(o, a...) = Dict{String,Any}("op" => o, "args" => collect(Any, a))
fidxd(di, dj) = Dict{String,Any}("op" => "index", "args" => Any["\$u", di == 0 ? "i" : fop("+", "i", di),
                                                                dj == 0 ? "j" : fop("+", "j", dj)])
fcentered(a, b, h) = fop("/", fop("-", a, b), fop("*", 2, h))
fgrad_rules(dxn) = [
    Dict{String,Any}("name" => "gx", "pattern" => Dict{String,Any}("op" => "grad", "args" => Any["\$u"], "dim" => "x"),
        "replacement" => fcentered(fidxd(1, 0), fidxd(-1, 0), dxn)),
    Dict{String,Any}("name" => "gy", "pattern" => Dict{String,Any}("op" => "grad", "args" => Any["\$u"], "dim" => "y"),
        "replacement" => fcentered(fidxd(0, 1), fidxd(0, -1), dxn))]

"RothermelFireSpread.R's declared shape in the discretized doc — [x,y] confirms the scalar
Rothermel chain was promoted to per-cell by the monolithic lowering."
function rshape(disc)
    for (_, m) in disc["models"]
        v = get(get(m, "variables", Dict{String,Any}()), "RothermelFireSpread.R", nothing)
        v === nothing && continue
        sh = get(v, "shape", nothing)
        return sh === nothing ? nothing : String[String(s) for s in sh]
    end
    return nothing
end

"Discretize the spatially-varying monolithic fire, keeping the 6 forcing as [x,y] parameter
SEEDS; ESS's `discretize(...; promote=true)` does the monolithic lowering. Returns (disc, rsh)."
function fire_discretized(dom)
    flat = with_logger(ConsoleLogger(stderr, Logging.Error)) do
        E.flatten(E.load(CAMP_FIRE_ESM))
    end
    OD = ESS.OrderedDict; MV = ESS.ModelVariable

    keepeq = [eq for eq in flat.equations if (eq.lhs isa E.VarExpr ? fcomp(eq.lhs.name) in KEEP :
              (eq.lhs isa E.OpExpr && eq.lhs.op == "D" && fcomp(eq.lhs.args[1].name) in KEEP))]
    refs = Set{String}()
    collect_refs(e) = (e isa E.VarExpr ? push!(refs, e.name) :
        e isa E.OpExpr ? (foreach(collect_refs, e.args);
            for f in (e.expr_body, e.filter, e.lower, e.upper); f === nothing || collect_refs(f); end;
            e.values === nothing || foreach(collect_refs, e.values)) : nothing)
    for eq in keepeq; collect_refs(eq.lhs); collect_refs(eq.rhs); end
    keepvar(d) = OD{String,MV}(k => v for (k, v) in d if fcomp(k) in KEEP && !(k in FORCING))
    states = keepvar(flat.state_variables); params = keepvar(flat.parameters); obs = keepvar(flat.observed_variables)
    for f in FORCING; params[f] = MV(ESS.ParameterVariable; shape = ["x", "y"]); end
    params["LevelSetFireSpread.dx"] = MV(ESS.ParameterVariable; default = dom.dx)
    defined = Set(keys(states)) ∪ Set(keys(params)) ∪ Set(keys(obs))
    for r in refs
        (r in defined || !occursin(".", r)) && continue
        params[r] = MV(ESS.ParameterVariable; default = get(FM1, r, 0.0))
    end

    flat2 = E.FlattenedSystem(flat.independent_variables, states, params, obs, keepeq,
        E.ContinuousEvent[], E.DiscreteEvent[], flat.domain, flat.metadata, flat.index_sets, flat.function_tables)

    grids = Dict("g" => Dict{String,Any}("family" => "cartesian", "dimensions" => Any[
        Dict{String,Any}("name" => "x", "size" => dom.nx, "periodic" => false, "spacing" => "uniform"),
        Dict{String,Any}("name" => "y", "size" => dom.ny, "periodic" => false, "spacing" => "uniform")]))
    # ESS applies the WHOLE monolithic lowering via promote=true: inline the level-set's
    # elementwise array observeds (psi_x/grad_mag/U_n/S_n…) into D(psi,t), reclassify the
    # algebraic-state physics, promote the scalar Rothermel chain to [x,y], discretize the
    # grad stencils, and index the per-cell refs — no driver-side fold / stage / index pass.
    disc = E.discretize(flat2; grids = grids, rules = fgrad_rules("LevelSetFireSpread.dx"),
                        strict_unrewritten = false, promote = true)
    return disc, rshape(disc)
end

"Replace the 6 forcing PARAMETERS in the discretized doc with the conservative regridder
(shared geometry + one apply observed per forcing field), so the whole regrid lowers in
the SAME evaluator. Mirrors run_camp_fire_e2e_pde.jl::couple_fields, generalized to the
6 camp_fire forcing names and the promoted physics. Returns (const_arrays, param_arrays)."
function merge_regridder!(disc, dom, src_poly, field_src)
    ns = size(src_poly, 1)
    fnames = collect(keys(field_src))
    isets, rvars = fields_regrid_spec(dom, ns, fnames)
    model = first(values(disc["models"]))
    model["index_sets"] = Base.merge(get(model, "index_sets", Dict{String,Any}()), isets)
    for fc in FORCING; delete!(model["variables"], fc); end          # drop the seed params
    for k in ("tgt_poly", "src_poly", "clip", "A_ij", "A_j")          # shared geometry
        model["variables"][k] = rvars[k]
    end
    forcing_of = Dict(FORCING_OF)
    for fn in fnames
        model["variables"]["F_src_$fn"] = rvars["F_src_$fn"]          # live source buffer
        model["variables"][forcing_of[fn]] = rvars[fn]                # apply observed under the forcing name
    end
    const_arrays = Dict{String,Any}("src_poly" => src_poly)
    param_arrays = Dict{String,Any}("F_src_$fn" => Float64[field_src[fn]...] for fn in fnames)
    return const_arrays, param_arrays
end

function main()
    dom = camp_domain(CAMP_FIRE_ESM)
    nx, ny = dom.nx, dom.ny

    # Coarse source grid + representative source fields (the W→E `wu` ramp drives the
    # anisotropy). The regrid spreads each source value over the fire cells it overlaps.
    src_poly, ncol, nrow = source_grid_lcc(dom; nsx = 3, nsy = 3)
    ns = ncol * nrow
    col_of(s) = ((s - 1) % ncol) + 1; row_of(s) = ((s - 1) ÷ ncol) + 1
    u_col  = collect(range(2.0, 10.0; length = ncol))
    sy_row = collect(range(0.0, 0.3;  length = nrow))
    field_src = Dict{String,Vector{Float64}}(
        "temp" => fill(288.0, ns), "rh" => fill(0.23, ns),
        "wu"   => [u_col[col_of(s)] for s in 1:ns], "wv" => zeros(ns),
        "sx"   => fill(0.05, ns),  "sy" => [sy_row[row_of(s)] for s in 1:ns])

    disc, rsh = fire_discretized(dom)
    const_arrays, param_arrays = merge_regridder!(disc, dom, src_poly, field_src)

    # ONE build_evaluator: regrid + promoted physics + level-set.
    f!, u0, p, _t, vmap = E.build_evaluator(disc; const_arrays = const_arrays, param_arrays = param_arrays,
        parameter_overrides = Dict("LevelSetFireSpread.dx" => dom.dx))
    for i in 1:nx, j in 1:ny
        k = get(vmap, "LevelSetFireSpread.psi[$i,$j]", nothing); k === nothing && continue
        u0[k] = sqrt((i - nx / 2)^2 + (j - ny / 2)^2) - 2.0
    end
    du = similar(u0); f!(du, u0, p, 0.0)
    band = [(i, j) for i in 1:nx, j in 1:ny if (k = get(vmap, "LevelSetFireSpread.psi[$i,$j]", nothing); k !== nothing && abs(u0[k]) <= 1.5)]
    spd(i, j) = -du[vmap["LevelSetFireSpread.psi[$i,$j]"]]
    sp = [spd(i, j) for (i, j) in band]
    e = [spd(i, j) for (i, j) in band if i > nx / 2]; w = [spd(i, j) for (i, j) in band if i < nx / 2]
    em = isempty(e) ? 0.0 : sum(e) / length(e); wm = isempty(w) ? 0.0 : sum(w) / length(w)

    println("Single-evaluator monolithic camp_fire (M2-5) — regrid + physics + level-set in ONE build_evaluator\n")
    println("  inputs to build_evaluator: src_poly (const) + ", length(field_src), " F_src buffers (param) — NO precomputed forcing arrays")
    println("  source grid: $(ncol)×$(nrow) LCC-frame cells → $(nx)×$(ny) = $(nx*ny) fire cells (clip → A_ij → A_j → apply, in-evaluator)")
    println("  RothermelFireSpread.R shape after promotion: ", rsh === nothing ? "absent" : rsh, "  (per-cell)")
    println("  front cells: ", length(band))
    println("  front speed: min ", round(minimum(sp); digits = 4), " max ", round(maximum(sp); digits = 4),
            "  — varies per cell: ", maximum(sp) - minimum(sp) > 1e-4)
    println("  wind anisotropy (regridded W→E ramp): downwind/E ", round(em; digits = 4),
            " > upwind/W ", round(wm; digits = 4))

    @assert rsh == ["x", "y"]                    "Rothermel R must promote to [x,y]"
    @assert !isempty(band)                       "expected a psi=0 front band"
    @assert maximum(sp) - minimum(sp) > 1e-4     "front speed must vary per cell"
    @assert em > wm                              "downwind front must be faster (regridded wind anisotropy)"
    println("\n  ✓ the conservative regrid, the per-cell Rothermel physics, and the level-set front")
    println("    are lowered and integrated in ONE build_evaluator — the regridded per-cell forcing is")
    println("    computed and consumed inside a single ODE right-hand side, no intermediate readout.")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
