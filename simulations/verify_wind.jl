#!/usr/bin/env julia
# Verify the wind regrid actually drives the front, spatially: eastward wind must
# enhance the DOWNWIND (east) front (U_n>0) and leave the UPWIND (west) front at R_0
# (U_n<0 ⇒ max(0,U_n)=0). Also confirm the regridded u_x ramps W→E.
include(joinpath(@__DIR__, "run_camp_fire_e2e_pde.jl"))

dom = camp_domain(CAMP_FIRE_ESM)
ls_path = levelset_ref_path(CAMP_FIRE_ESM)
r0 = 0.71
uvals = [2.0, 5.0, 8.0]

# --- regridded u_x field (isolate it: D(w[x,y],t) = u_x[x,y]) ---
isets, rvars, Fsrc = wind_regrid_spec(dom; ns = 3, uvals = uvals)
ux_lhs = Dict{String,Any}("op"=>"aggregate","output_idx"=>Any["x","y"],
    "ranges"=>Dict{String,Any}("x"=>Dict{String,Any}("from"=>"gx"),"y"=>Dict{String,Any}("from"=>"gy")),
    "expr"=>Dict{String,Any}("op"=>"D","args"=>Any[_ixd("w","x","y")],"wrt"=>"t"))
ux_rhs = _arrd(["x","y"], ["x"=>"gx","y"=>"gy"], _ixd("u_x","x","y"))
uxvars = Base.merge(rvars, Dict{String,Any}("w"=>Dict{String,Any}("type"=>"state","shape"=>Any["gx","gy"])))
uxesm = Dict{String,Any}("esm"=>"0.6.0","metadata"=>Dict{String,Any}("name"=>"uxcheck"),
    "models"=>Dict{String,Any}("M"=>Dict{String,Any}("index_sets"=>isets,"variables"=>uxvars,
        "equations"=>Any[Dict{String,Any}("lhs"=>ux_lhs,"rhs"=>ux_rhs)])))
ic = Dict("w[$i,$j]"=>0.0 for i in 1:dom.nx, j in 1:dom.ny)
fu!, u0u, pu, _, vmu = build_evaluator(uxesm; param_arrays=Dict("F_src"=>Fsrc), initial_conditions=ic)
duu = similar(u0u); fu!(duu, u0u, pu, 0.0)
ux(i,j) = duu[vmu["w[$i,$j]"]]
println("regridded u_x across the row (j=11):")
println("  i=1(W)=", round(ux(1,11);digits=3), "  i=10(C)=", round(ux(10,11);digits=3),
        "  i=19(E)=", round(ux(19,11);digits=3), "  (source ramp ", uvals, ")")

# --- coupled front: east vs west spread asymmetry ---
esm = levelset_pde_esm(ls_path, dom; wind = true)
disc = discretize(esm)
disc2, Fsrc2 = couple_wind(disc, dom; ns = 3, uvals = uvals)
f!, u0, p, _t, vmap = build_evaluator(disc2; parameter_overrides=Dict("R_0"=>r0),
    param_arrays=Dict("F_src"=>Fsrc2))
X(i)=dom.xmin+(i-1)*dom.dx; Y(j)=dom.ymin+(j-1)*dom.dx
for i in 1:dom.nx, j in 1:dom.ny
    k=get(vmap,"psi[$i,$j]",nothing); k===nothing && continue
    u0[k]=ESS.evaluate_expr(dom.ic, Dict("x"=>X(i),"y"=>Y(j)))
end
du=similar(u0); f!(du,u0,p,0.0)
# ignition column (where psi IC is centered): x0=-2000072.1
i0 = round(Int, (-2000072.1 - dom.xmin)/dom.dx) + 1
east=Float64[]; west=Float64[]
for i in 1:dom.nx, j in 1:dom.ny
    k=get(vmap,"psi[$i,$j]",nothing); k===nothing && continue
    abs(u0[k]) <= dom.dx || continue          # front band
    spd = -du[k]                               # outward front speed
    if i > i0; push!(east, spd); elseif i < i0; push!(west, spd); end
end
println("ignition column i0=", i0, " (nx=", dom.nx, ")")
println("front spread speed (m/s) in the |psi|<=dx band:")
println("  EAST (downwind, U_n>0): max=", round(maximum(east);digits=2), " mean=", round(sum(east)/length(east);digits=2), " n=", length(east))
println("  WEST (upwind,  U_n<0): max=", round(maximum(west);digits=2), " mean=", round(sum(west)/length(west);digits=2), " n=", length(west))
println("  expect EAST >> WEST≈R_0(", r0, ") — the regridded eastward wind elongates the front downwind")
