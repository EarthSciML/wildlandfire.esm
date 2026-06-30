#!/usr/bin/env julia
#=
Spatially-varying MONOLITHIC camp_fire fire (M2) — the whole flattened physics +
level-set integrated in ONE build_evaluator, with R_0 (and the front speed) varying
CELL-BY-CELL from the per-cell forcing. The scalar Rothermel chain is lifted to
per-cell automatically by ESS's `promote_downstream_shapes` AST transform (any scalar
variable downstream of an array variable is promoted to the array's shape and its
equation rewritten to an `arrayop`), so NO component is re-authored per-cell and NO
per-cell logic lives in the runner.

Pipeline (all ESS, MTK-free):
  flatten(load(camp_fire.esm))
    → keep the fire-physics + level-set; feed the regridder outputs as per-cell
      [x,y] DEMO forcing (the value-invention regridders' fire-grid sizing is the
      remaining follow-on — here the forcing is a representative spatial field)
    → algebraic_states_to_observeds   (Rothermel R / midflame U / EMC … are authored
                                        as algebraic states; reclassify so they promote)
    → fold the level-set's spatial observeds into D(psi,t)  (the driver's trick)
    → promote_downstream_shapes        (the physics chain → per-cell [x,y])
    → discretize(grad → centered stencils on the 19×21 fire grid)
    → index_promoted_refs!             (index the per-cell fields in the level-set body)
    → build_evaluator + evaluate

Asserts: R promotes to [x,y]; the front speed VARIES per cell; the downwind (east)
front is faster than upwind (west) under a W→E wind ramp. Run with plain `julia`
(activates the ESS pde_sim_adapter env; set EARTHSCISERIALIZATION to override).
=#
import Pkg
const REPO = dirname(@__DIR__)
const CAMP = joinpath(REPO, "simulations", "camp_fire.esm")
const ESS_ROOT = get(ENV, "EARTHSCISERIALIZATION", normpath(joinpath(REPO, "..", "EarthSciSerialization")))
const ESS_PKG = joinpath(ESS_ROOT, "packages", "EarthSciSerialization.jl")
let env = joinpath(ESS_PKG, "scripts", "pde_sim_adapter")
    isdir(env) || error("pde_sim_adapter project not found at $env — set EARTHSCISERIALIZATION")
    Pkg.activate(env; io = devnull)
    isfile(joinpath(env, "Manifest.toml")) || Pkg.develop(path = ESS_PKG; io = devnull)
    Pkg.instantiate(; io = devnull)
end
using Logging
import EarthSciSerialization as ESS
import GeometryOps, GeoInterface
const E = ESS

KEEP = ["TerrainSlope","MidflameWind","EquilibriumMoistureContent","OneHourFuelMoisture",
        "RothermelFireSpread","LevelSetFireSpread","FuelConsumption"]
FORCING = ["ERA5tRegrid.F_tgt","ERA5rRegrid.F_tgt","ERA5uRegrid.F_tgt","ERA5vRegrid.F_tgt",
           "SlopeXRegrid.F_tgt","SlopeYRegrid.F_tgt"]
FM1 = Dict("FuelModelLookup.sigma"=>11483.0,"FuelModelLookup.w_0"=>0.166,
           "FuelModelLookup.delta"=>0.3048,"FuelModelLookup.M_x"=>0.12,"FuelModelLookup.h"=>18608000.0)
comp(n) = String(split(n,".")[1])
op(o,a...) = Dict{String,Any}("op"=>o,"args"=>collect(Any,a))
idxd(di,dj) = Dict{String,Any}("op"=>"index","args"=>Any["\$u", di==0 ? "i" : op("+","i",di), dj==0 ? "j" : op("+","j",dj)])
centered(a,b,h) = op("/",op("-",a,b),op("*",2,h))
grad_rules(dxn) = [
    Dict{String,Any}("name"=>"gx","pattern"=>Dict{String,Any}("op"=>"grad","args"=>Any["\$u"],"dim"=>"x"),"replacement"=>centered(idxd(1,0),idxd(-1,0),dxn)),
    Dict{String,Any}("name"=>"gy","pattern"=>Dict{String,Any}("op"=>"grad","args"=>Any["\$u"],"dim"=>"y"),"replacement"=>centered(idxd(0,1),idxd(0,-1),dxn))]

function main()
    flat = with_logger(ConsoleLogger(stderr, Logging.Error)) do; E.flatten(E.load(CAMP)); end
    OD = ESS.OrderedDict; MV = ESS.ModelVariable

    keepeq = [eq for eq in flat.equations if (eq.lhs isa E.VarExpr ? comp(eq.lhs.name) in KEEP :
              (eq.lhs isa E.OpExpr && eq.lhs.op=="D" && comp(eq.lhs.args[1].name) in KEEP))]
    refs = Set{String}()
    collect_refs(e) = (e isa E.VarExpr ? push!(refs,e.name) :
        e isa E.OpExpr ? (foreach(collect_refs,e.args); for f in (e.expr_body,e.filter,e.lower,e.upper); f===nothing||collect_refs(f); end; e.values===nothing||foreach(collect_refs,e.values)) : nothing)
    for eq in keepeq; collect_refs(eq.lhs); collect_refs(eq.rhs); end
    keepvar(d) = OD{String,MV}(k=>v for (k,v) in d if comp(k) in KEEP && !(k in FORCING))
    states = keepvar(flat.state_variables); params = keepvar(flat.parameters); obs = keepvar(flat.observed_variables)
    for f in FORCING; params[f] = MV(ESS.ParameterVariable; shape=["x","y"]); end
    params["LevelSetFireSpread.dx"] = MV(ESS.ParameterVariable; default=2000.0)
    defined = Set(keys(states)) ∪ Set(keys(params)) ∪ Set(keys(obs))
    for r in refs
        (r in defined || !occursin(".",r)) && continue
        params[r] = MV(ESS.ParameterVariable; default=get(FM1,r,0.0))
    end

    # fold the level-set's spatial observeds into D(psi,t)
    lsdef = Dict{String,E.Expr}()
    for eq in keepeq; eq.lhs isa E.VarExpr && comp(eq.lhs.name)=="LevelSetFireSpread" && (lsdef[eq.lhs.name]=eq.rhs); end
    order=String[]; done=Set{String}()
    while length(order)<length(lsdef)
        prog=false
        for nm in keys(lsdef)
            nm in done && continue
            all(d->d in done, intersect(E.free_variables(lsdef[nm]),Set(keys(lsdef)))) && (push!(order,nm);push!(done,nm);prog=true)
        end
        prog || error("cyclic level-set observed")
    end
    resolved=Dict{String,E.Expr}(); for nm in order; resolved[nm]=E.substitute(lsdef[nm],resolved); end
    keepeq2=E.Equation[]
    for eq in keepeq
        if eq.lhs isa E.VarExpr && haskey(lsdef,eq.lhs.name); continue
        elseif eq.lhs isa E.OpExpr && eq.lhs.op=="D" && comp(eq.lhs.args[1].name)=="LevelSetFireSpread"
            push!(keepeq2, E.Equation(eq.lhs, E.substitute(eq.rhs,resolved); _comment=eq._comment, region=eq.region))
        else; push!(keepeq2, eq); end
    end
    for nm in keys(lsdef); delete!(obs,nm); end

    flat2 = E.FlattenedSystem(flat.independent_variables, states, params, obs, keepeq2,
        E.ContinuousEvent[], E.DiscreteEvent[], flat.domain, flat.metadata, flat.index_sets, flat.function_tables)
    flat2 = E.algebraic_states_to_observeds(flat2)
    prom = E.promote_downstream_shapes(flat2)
    rsh = get(prom.observed_variables,"RothermelFireSpread.R",nothing)

    nx,ny = 19,21
    grids = Dict("g"=>Dict{String,Any}("family"=>"cartesian","dimensions"=>Any[
        Dict{String,Any}("name"=>"x","size"=>nx,"periodic"=>false,"spacing"=>"uniform"),
        Dict{String,Any}("name"=>"y","size"=>ny,"periodic"=>false,"spacing"=>"uniform")]))
    disc = E.discretize(prom; grids=grids, rules=grad_rules("LevelSetFireSpread.dx"), strict_unrewritten=false)
    arrnames = Set{String}(k for (k,v) in merge(Dict(prom.parameters),Dict(prom.observed_variables))
                           if v.shape !== nothing && !isempty(v.shape))
    E.index_promoted_refs!(disc, arrnames)

    arr(v) = [v for _ in 1:nx, _ in 1:ny]
    ramp_x(lo,hi) = [lo+(hi-lo)*(i-1)/(nx-1) for i in 1:nx, _ in 1:ny]
    ramp_y(lo,hi) = [lo+(hi-lo)*(j-1)/(ny-1) for _ in 1:nx, j in 1:ny]
    ca = Dict{String,Any}("ERA5tRegrid.F_tgt"=>arr(288.0),"ERA5rRegrid.F_tgt"=>arr(0.23),
        "ERA5uRegrid.F_tgt"=>ramp_x(2.0,10.0),"ERA5vRegrid.F_tgt"=>arr(0.0),
        "SlopeXRegrid.F_tgt"=>arr(0.05),"SlopeYRegrid.F_tgt"=>ramp_y(0.0,0.3))

    f!, u0, p, _t, vmap = E.build_evaluator(disc; const_arrays=ca,
        parameter_overrides=Dict("LevelSetFireSpread.dx"=>2000.0))
    for i in 1:nx, j in 1:ny
        k=get(vmap,"LevelSetFireSpread.psi[$i,$j]",nothing); k===nothing && continue
        u0[k]=sqrt((i-nx/2)^2+(j-ny/2)^2)-2.0
    end
    du=similar(u0); f!(du,u0,p,0.0)
    band=[(i,j) for i in 1:nx, j in 1:ny if (k=get(vmap,"LevelSetFireSpread.psi[$i,$j]",nothing); k!==nothing && abs(u0[k])<=1.5)]
    spd(i,j)=-du[vmap["LevelSetFireSpread.psi[$i,$j]"]]
    @assert rsh !== nothing && rsh.shape==["x","y"]  "Rothermel R must promote to [x,y]"
    @assert !isempty(band)                            "expected a psi=0 front band"
    sp=[spd(i,j) for (i,j) in band]
    e=[spd(i,j) for (i,j) in band if i>nx/2]; w=[spd(i,j) for (i,j) in band if i<nx/2]
    em=isempty(e) ? 0.0 : sum(e)/length(e); wm=isempty(w) ? 0.0 : sum(w)/length(w)
    println("Spatially-varying monolithic camp_fire (M2) — one build_evaluator\n")
    println("  RothermelFireSpread.R shape after promotion: ", rsh.shape, "  (per-cell)")
    println("  front cells: ", length(band))
    println("  front speed: min ", round(minimum(sp);digits=4), " max ", round(maximum(sp);digits=4),
            "  — varies per cell: ", maximum(sp)-minimum(sp) > 1e-4)
    println("  wind anisotropy (W→E ramp 2→10 m/s): downwind/E ", round(em;digits=4),
            " > upwind/W ", round(wm;digits=4))
    @assert maximum(sp)-minimum(sp) > 1e-4            "front speed must vary per cell"
    @assert em > wm                                   "downwind front must be faster (wind anisotropy)"
    println("\n  ✓ per-cell Rothermel R from per-cell forcing via the shape-promotion AST transform;")
    println("    the whole physics + level-set lowers in ONE build_evaluator, no per-cell runner logic.")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
