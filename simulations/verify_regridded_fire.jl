#!/usr/bin/env julia
#=
M2-4 — REAL conservative-regridder forcing for the spatially-varying monolithic fire.

verify_spatial_fire.jl proves the per-cell physics: the scalar Rothermel chain is
lifted to per-cell [x,y] by the shape-promotion AST transform and the whole physics +
level-set integrates in ONE build_evaluator. There the 6 forcing fields were a
representative spatial DEMO (`arr()/ramp_x()/ramp_y()` hand-written arrays).

This driver replaces those hand-written arrays with the output of the REAL conservative
regridder, sized to the 399-cell (19×21) fire grid:

  Stage 1 — REGRID.  The declarative conservative regrid that camp_fire.esm couples in
    (clip → A_ij → A_j → F_tgt = Σ_s A_ij·F_src/A_j, the `fields_regrid_spec` geometry
    from run_camp_fire_e2e_pde.jl) maps coarse source fields onto the fire grid THROUGH
    build_evaluator — real intersect_polygon clips, real shoelace areas, real
    area-weighted apply at tgt_cells = 399. Mass-conservative: A_j ≈ the fire cell area
    on every cell. This is the same regridder verify_coupling.jl §A/§C exercises; here
    it produces all 6 camp_fire forcing fields (temp/RH/u/v/slope-x/slope-y) at once.

  Stage 2 — FIRE.  The 6 regridded 399-cell fields are fed as the forcing of the
    spatially-varying monolithic fire (verify_spatial_fire.jl's pipeline verbatim:
    keep physics+level-set → fold level-set observeds → algebraic_states_to_observeds →
    promote_downstream_shapes → discretize → index_promoted_refs! → build_evaluator).
    The ONLY change from verify_spatial_fire is the provenance of the forcing arrays:
    the real regridder instead of the demo ramps. The regrid weights are build-once and
    the source field is a parameter, so the regridder output IS the per-cell forcing —
    feeding it as const_arrays is exact, not an approximation.

Asserts: the regrid conserves mass (A_j ≈ cell area); the regridded u field is
monotone W→E (so the forcing genuinely comes from the regridder, not a constant);
Rothermel R promotes to [x,y]; the front speed varies per cell; and the downwind/E
front is faster than upwind/W under the regridded W→E wind ramp.

Run with plain `julia` (the included run_camp_fire_e2e_pde.jl activates the ESS
pde_sim_adapter env; set EARTHSCISERIALIZATION to override).
=#
include(joinpath(@__DIR__, "run_camp_fire_e2e_pde.jl"))  # env bootstrap + regridder helpers + ESS
import GeometryOps, GeoInterface
const E = ESS

# ── Stage 1: real conservative regrid of the 6 forcing fields onto the fire grid ──
# Coarse source grid (LCC-frame, tiling the fire domain) carrying representative source
# values; the W→E `u` ramp and S→N `sy` ramp are the anisotropy drivers. The conservative
# regrid spreads each source cell's value over the fire cells it overlaps (area-weighted).
"Run `fields_regrid_spec` for all 6 camp_fire forcing fields and read out each F_tgt[x,y]
plus the conservation denominator A_j[x,y]. Returns (fields::Dict, A_j::Matrix, ncol, nrow)."
function regrid_forcing(dom; nsx::Int = 3, nsy::Int = 3)
    src_poly, ncol, nrow = source_grid_lcc(dom; nsx = nsx, nsy = nsy)
    ns = ncol * nrow
    col_of(s) = ((s - 1) % ncol) + 1
    row_of(s) = ((s - 1) ÷ ncol) + 1
    u_col  = collect(range(2.0, 10.0; length = ncol))   # W→E midflame wind (m/s)
    sy_row = collect(range(0.0, 0.3;  length = nrow))    # S→N terrain slope (rise/run)
    # NB: field names must avoid the regridder's single-char loop vars (x,y,s,w,c,v) —
    # `fields_regrid_spec`'s shoelace binds `v` over clip_ring, so a field named "v"
    # would alias that loop var and break the geometry-setup closure. Use `wu`/`wv`.
    field_src = Dict{String,Vector{Float64}}(
        "temp" => fill(288.0, ns), "rh" => fill(0.23, ns),
        "wu"   => [u_col[col_of(s)] for s in 1:ns], "wv" => zeros(ns),
        "sx"   => fill(0.05, ns),  "sy" => [sy_row[row_of(s)] for s in 1:ns])
    fnames = collect(keys(field_src))

    isets, rvars = fields_regrid_spec(dom, ns, fnames)
    vars = Dict{String,Any}(rvars)
    eqs = Any[]
    # Tiny consumer per field: D(w_fn[x,y],t) = fn[x,y]; du reads the regridded field.
    readout(name, body) = (vars["w_$name"] = Dict{String,Any}("type" => "state", "shape" => Any["gx", "gy"]);
        push!(eqs, Dict{String,Any}(
            "lhs" => Dict{String,Any}("op" => "aggregate", "output_idx" => Any["x", "y"],
                "ranges" => Dict{String,Any}("x" => Dict{String,Any}("from" => "gx"),
                                             "y" => Dict{String,Any}("from" => "gy")),
                "expr" => Dict{String,Any}("op" => "D", "args" => Any[_ixd("w_$name", "x", "y")], "wrt" => "t")),
            "rhs" => _arrd(["x", "y"], ["x" => "gx", "y" => "gy"], body))))
    for fn in fnames
        readout(fn, _ixd(fn, "x", "y"))
    end
    readout("Aj", _ixd("A_j", "x", "y"))   # conservation: A_j per fire cell

    esm = Dict{String,Any}("esm" => "0.6.0", "metadata" => Dict{String,Any}("name" => "regrid_forcing"),
        "models" => Dict{String,Any}("M" => Dict{String,Any}(
            "index_sets" => isets, "variables" => vars, "equations" => eqs)))
    ic = Dict{String,Float64}()
    for name in vcat(fnames, ["Aj"]), i in 1:dom.nx, j in 1:dom.ny
        ic["w_$name[$i,$j]"] = 0.0
    end
    f!, u0, p, _t, vm = build_evaluator(esm; const_arrays = Dict("src_poly" => src_poly),
        param_arrays = Dict("F_src_$fn" => field_src[fn] for fn in fnames),
        initial_conditions = ic)
    du = similar(u0); f!(du, u0, p, 0.0)
    grid(name) = [du[vm["w_$name[$i,$j]"]] for i in 1:dom.nx, j in 1:dom.ny]
    fields = Dict{String,Matrix{Float64}}(fn => grid(fn) for fn in fnames)
    return fields, grid("Aj"), ncol, nrow
end

# ── Stage 2: the spatially-varying monolithic fire (verify_spatial_fire pipeline) ──
KEEP = ["TerrainSlope", "MidflameWind", "EquilibriumMoistureContent", "OneHourFuelMoisture",
        "RothermelFireSpread", "LevelSetFireSpread", "FuelConsumption"]
FORCING = ["ERA5tRegrid.F_tgt", "ERA5rRegrid.F_tgt", "ERA5uRegrid.F_tgt", "ERA5vRegrid.F_tgt",
           "SlopeXRegrid.F_tgt", "SlopeYRegrid.F_tgt"]
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

"Build the spatially-varying monolithic fire evaluator forced by `ca` (the 6 dotted
forcing names → regridded [nx,ny] arrays). Mirrors verify_spatial_fire.jl exactly."
function build_fire(dom, ca)
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

    # fold the level-set's spatial observeds into D(psi,t)
    lsdef = Dict{String,E.Expr}()
    for eq in keepeq; eq.lhs isa E.VarExpr && fcomp(eq.lhs.name) == "LevelSetFireSpread" && (lsdef[eq.lhs.name] = eq.rhs); end
    order = String[]; done = Set{String}()
    while length(order) < length(lsdef)
        prog = false
        for nm in keys(lsdef)
            nm in done && continue
            all(d -> d in done, intersect(E.free_variables(lsdef[nm]), Set(keys(lsdef)))) && (push!(order, nm); push!(done, nm); prog = true)
        end
        prog || error("cyclic level-set observed")
    end
    resolved = Dict{String,E.Expr}(); for nm in order; resolved[nm] = E.substitute(lsdef[nm], resolved); end
    keepeq2 = E.Equation[]
    for eq in keepeq
        if eq.lhs isa E.VarExpr && haskey(lsdef, eq.lhs.name); continue
        elseif eq.lhs isa E.OpExpr && eq.lhs.op == "D" && fcomp(eq.lhs.args[1].name) == "LevelSetFireSpread"
            push!(keepeq2, E.Equation(eq.lhs, E.substitute(eq.rhs, resolved); _comment = eq._comment, region = eq.region))
        else; push!(keepeq2, eq); end
    end
    for nm in keys(lsdef); delete!(obs, nm); end

    flat2 = E.FlattenedSystem(flat.independent_variables, states, params, obs, keepeq2,
        E.ContinuousEvent[], E.DiscreteEvent[], flat.domain, flat.metadata, flat.index_sets, flat.function_tables)
    flat2 = E.algebraic_states_to_observeds(flat2)
    prom = E.promote_downstream_shapes(flat2)
    rsh = get(prom.observed_variables, "RothermelFireSpread.R", nothing)

    grids = Dict("g" => Dict{String,Any}("family" => "cartesian", "dimensions" => Any[
        Dict{String,Any}("name" => "x", "size" => dom.nx, "periodic" => false, "spacing" => "uniform"),
        Dict{String,Any}("name" => "y", "size" => dom.ny, "periodic" => false, "spacing" => "uniform")]))
    disc = E.discretize(prom; grids = grids, rules = fgrad_rules("LevelSetFireSpread.dx"), strict_unrewritten = false)
    arrnames = Set{String}(k for (k, v) in Base.merge(Dict(prom.parameters), Dict(prom.observed_variables))
                           if v.shape !== nothing && !isempty(v.shape))
    E.index_promoted_refs!(disc, arrnames)

    f!, u0, p, _t, vmap = E.build_evaluator(disc; const_arrays = ca,
        parameter_overrides = Dict("LevelSetFireSpread.dx" => dom.dx))
    return f!, u0, p, vmap, rsh
end

function main()
    dom = camp_domain(CAMP_FIRE_ESM)
    nx, ny = dom.nx, dom.ny

    # Stage 1 — the real conservative regrid.
    fields, aj, ncol, nrow = regrid_forcing(dom)
    cell_area = dom.dx^2
    ajmin, ajmax = minimum(aj) / cell_area, maximum(aj) / cell_area
    ucol = [fields["wu"][i, ny ÷ 2] for i in 1:nx]   # regridded u along a mid row, W→E

    ca = Dict{String,Any}(
        "ERA5tRegrid.F_tgt" => fields["temp"], "ERA5rRegrid.F_tgt" => fields["rh"],
        "ERA5uRegrid.F_tgt" => fields["wu"],   "ERA5vRegrid.F_tgt" => fields["wv"],
        "SlopeXRegrid.F_tgt" => fields["sx"],  "SlopeYRegrid.F_tgt" => fields["sy"])

    # Stage 2 — the spatially-varying monolithic fire, forced by the regridder output.
    f!, u0, p, vmap, rsh = build_fire(dom, ca)
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

    println("Regridder-forced spatially-varying monolithic camp_fire (M2-4)\n")
    println("  Stage 1 — REAL conservative regrid → fire grid ($(nx)×$(ny) = $(nx*ny) cells)")
    println("    source grid: $(ncol)×$(nrow) LCC-frame cells (area-weighted clip → A_ij → A_j → apply)")
    println("    mass conservation: A_j / cell_area  min ", round(ajmin; digits = 4),
            "  max ", round(ajmax; digits = 4), "  (≈1.0 — regrid covers every fire cell)")
    println("    regridded u (W→E across the grid): ", round.(ucol[[1, nx ÷ 2, nx]]; digits = 3),
            "  monotone: ", issorted(ucol))
    println("    regridded fields fed as forcing: ", join(sort(collect(keys(ca))), ", "))
    println("\n  Stage 2 — spatially-varying monolithic fire (one build_evaluator)")
    println("    RothermelFireSpread.R shape after promotion: ", rsh === nothing ? "absent" : rsh.shape, "  (per-cell)")
    println("    front cells: ", length(band))
    println("    front speed: min ", round(minimum(sp); digits = 4), " max ", round(maximum(sp); digits = 4),
            "  — varies per cell: ", maximum(sp) - minimum(sp) > 1e-4)
    println("    wind anisotropy (regridded W→E ramp): downwind/E ", round(em; digits = 4),
            " > upwind/W ", round(wm; digits = 4))

    @assert 0.98 <= ajmin && ajmax <= 1.02      "conservative regrid must cover every fire cell (A_j ≈ cell area)"
    @assert issorted(ucol)                       "regridded u must increase W→E (forcing genuinely from the regridder)"
    @assert rsh !== nothing && rsh.shape == ["x", "y"]  "Rothermel R must promote to [x,y]"
    @assert !isempty(band)                       "expected a psi=0 front band"
    @assert maximum(sp) - minimum(sp) > 1e-4     "front speed must vary per cell"
    @assert em > wm                              "downwind front must be faster (regridded wind anisotropy)"
    println("\n  ✓ the 6 forcing fields are produced by the REAL conservative regridder at the")
    println("    399-cell fire-grid target (mass-conservative), then drive the per-cell Rothermel")
    println("    physics + level-set in ONE build_evaluator — no hand-written demo forcing arrays.")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
