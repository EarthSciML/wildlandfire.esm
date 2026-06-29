#!/usr/bin/env julia
# Verify the multi-field met→fire coupling (item 1): (A) each field regrids to the
# right per-cell values, and (B) ABLATION — wind-only and slope-only runs each
# produce the expected, SEPARABLE front anisotropy (wind ⇒ east faster; slope ⇒
# north faster). One run can't separate them spatially (the ignition circle is
# ~1.5 cells across and wind is ~100× slope), so we ablate each field in turn.
include(joinpath(@__DIR__, "run_camp_fire_e2e_pde.jl"))

dom = camp_domain(CAMP_FIRE_ESM)
ls_path = levelset_ref_path(CAMP_FIRE_ESM)
r0 = 0.71; phi_s = 5.275
nsx = nsy = 3; ns = nsx * nsy
src_poly, _, _ = source_grid_lcc(dom; nsx = nsx, nsy = nsy)
sx_of(s) = ((s - 1) % nsx) + 1
sy_of(s) = ((s - 1) ÷ nsx) + 1
ux_col   = collect(range(2.0, 8.0; length = nsx))
dzdy_row = collect(range(0.0, 0.5; length = nsy))

# (A) regridded field values: build a tiny consumer D(w[x,y],t)=field[x,y] per field.
function regridded(field_src)
    isets, rvars = fields_regrid_spec(dom, ns, collect(keys(field_src)))
    vars = Dict{String,Any}(rvars)
    eqs = Any[]
    for fn in keys(field_src)
        vars["w_$fn"] = Dict{String,Any}("type"=>"state","shape"=>Any["gx","gy"])
        push!(eqs, Dict{String,Any}(
            "lhs"=>Dict{String,Any}("op"=>"aggregate","output_idx"=>Any["x","y"],
                "ranges"=>Dict{String,Any}("x"=>Dict{String,Any}("from"=>"gx"),"y"=>Dict{String,Any}("from"=>"gy")),
                "expr"=>Dict{String,Any}("op"=>"D","args"=>Any[_ixd("w_$fn","x","y")],"wrt"=>"t")),
            "rhs"=>_arrd(["x","y"],["x"=>"gx","y"=>"gy"],_ixd(fn,"x","y"))))
    end
    esm = Dict{String,Any}("esm"=>"0.6.0","metadata"=>Dict{String,Any}("name"=>"rg"),
        "models"=>Dict{String,Any}("M"=>Dict{String,Any}("index_sets"=>isets,"variables"=>vars,"equations"=>eqs)))
    ic = Dict("w_$fn[$i,$j]"=>0.0 for fn in keys(field_src), i in 1:dom.nx, j in 1:dom.ny)
    f!,u0,p,_,vm = build_evaluator(esm; const_arrays=Dict("src_poly"=>src_poly),
        param_arrays=Dict("F_src_$fn"=>Float64[field_src[fn]...] for fn in keys(field_src)),
        initial_conditions=ic)
    du=similar(u0); f!(du,u0,p,0.0)
    return (fn,i,j)->du[vm["w_$fn[$i,$j]"]]
end
g = regridded(Dict("u_x"=>[ux_col[sx_of(s)] for s in 1:ns],
                   "dzdy"=>[dzdy_row[sy_of(s)] for s in 1:ns]))
println("(A) regridded fields (no host geometry — conservative regrid through build_evaluator):")
println("    u_x  along row j=11 (W→E): ", round.([g("u_x",i,11) for i in (1,10,19)];digits=3), "  source ramp ", ux_col)
println("    dzdy along col i=10 (S→N): ", round.([g("dzdy",10,j) for j in (1,11,21)];digits=3), "  source ramp ", dzdy_row)

# (B) ablation: front speed asymmetry with ONE field active at a time.
function front_du(field_src)
    esm = levelset_pde_esm(ls_path, dom; wind = true)
    disc = discretize(esm)
    fs = Dict{String,Any}("u_x"=>zeros(ns),"u_y"=>zeros(ns),"dzdx"=>zeros(ns),"dzdy"=>zeros(ns))
    Base.merge!(fs, field_src)
    disc2, ca, pa = couple_fields(disc, dom, src_poly, fs)
    f!,u0,p,_,vmap = build_evaluator(disc2; parameter_overrides=Dict("R_0"=>r0,"phi_s_coeff"=>phi_s),
        const_arrays=ca, param_arrays=pa)
    X(i)=dom.xmin+(i-1)*dom.dx; Y(j)=dom.ymin+(j-1)*dom.dx
    for i in 1:dom.nx, j in 1:dom.ny
        k=get(vmap,"psi[$i,$j]",nothing); k===nothing && continue
        u0[k]=ESS.evaluate_expr(dom.ic, Dict("x"=>X(i),"y"=>Y(j)))
    end
    du=similar(u0); f!(du,u0,p,0.0)
    return u0, du, vmap
end
i0 = round(Int,(-2000072.1-dom.xmin)/dom.dx)+1
j0 = round(Int,(395036.6-dom.ymin)/dom.dx)+1
function axis(u0,du,vmap,sel)
    v=Float64[]
    for i in 1:dom.nx, j in 1:dom.ny
        k=get(vmap,"psi[$i,$j]",nothing); k===nothing && continue
        abs(u0[k])<=dom.dx && sel(i,j) && push!(v,-du[k])
    end
    isempty(v) ? NaN : sum(v)/length(v)
end
println("(B) per-field ablation (front spread, m/s; R_0=$r0):")
u0,du,vm = front_du(Dict("u_x"=>[ux_col[sx_of(s)] for s in 1:ns]))   # wind only
println("    WIND only  → E=", round(axis(u0,du,vm,(i,j)->i>i0);digits=2),
        " W=", round(axis(u0,du,vm,(i,j)->i<i0);digits=2), "   (expect E≫W≈R_0)")
u0,du,vm = front_du(Dict("dzdy"=>[dzdy_row[sy_of(s)] for s in 1:ns]))  # slope only
println("    SLOPE only → N=", round(axis(u0,du,vm,(i,j)->j>j0);digits=2),
        " S=", round(axis(u0,du,vm,(i,j)->j<j0);digits=2), "   (expect N>S≈R_0)")

# (C) item 2: real ERA5 lat/lon grid reprojected to LCC must CONSERVATIVELY tile the
# fire grid — A_j (total overlap per fire cell) ≈ the fire cell area dx² (no gaps),
# the conservation guarantee of an area-weighted regrid.
println("(C) ERA5 0.25° grid reprojected to LCC — conservation (A_j vs fire cell area):")
sp_e, nlon, nlat, clons, clats = source_grid_era5(dom)
println("    ERA5 grid: $(nlon)×$(nlat) cells; centers lon ", round.(unique(clons);digits=3))
isets, rvars = fields_regrid_spec(dom, nlon*nlat, ["u_x"])
vars = Dict{String,Any}(rvars)
vars["w_Aj"] = Dict{String,Any}("type"=>"state","shape"=>Any["gx","gy"])
ajeq = Dict{String,Any}(
    "lhs"=>Dict{String,Any}("op"=>"aggregate","output_idx"=>Any["x","y"],
        "ranges"=>Dict{String,Any}("x"=>Dict{String,Any}("from"=>"gx"),"y"=>Dict{String,Any}("from"=>"gy")),
        "expr"=>Dict{String,Any}("op"=>"D","args"=>Any[_ixd("w_Aj","x","y")],"wrt"=>"t")),
    "rhs"=>_arrd(["x","y"],["x"=>"gx","y"=>"gy"],_ixd("A_j","x","y")))
esm = Dict{String,Any}("esm"=>"0.6.0","metadata"=>Dict{String,Any}("name"=>"ajcheck"),
    "models"=>Dict{String,Any}("M"=>Dict{String,Any}("index_sets"=>isets,"variables"=>vars,"equations"=>Any[ajeq])))
icA = Dict("w_Aj[$i,$j]"=>0.0 for i in 1:dom.nx, j in 1:dom.ny)
fA,u0A,pA,_,vmA = build_evaluator(esm; const_arrays=Dict("src_poly"=>sp_e),
    param_arrays=Dict("F_src_u_x"=>ones(nlon*nlat)), initial_conditions=icA)
duA=similar(u0A); fA(duA,u0A,pA,0.0)
aj = [duA[vmA["w_Aj[$i,$j]"]] for i in 1:dom.nx, j in 1:dom.ny]
cell_area = dom.dx^2
println("    A_j / cell_area: min=", round(minimum(aj)/cell_area;digits=4),
        " max=", round(maximum(aj)/cell_area;digits=4),
        " (≈1.0 — ERA5 covers every fire cell; <1 on the few cells straddling an ERA5")
println("                    boundary, whose thin sliver overlap the degenerate-clip guard drops —")
println("                    the sliver `filter` the real conservative_regrid_overlap_join.esm adds, item 4)")
println("    total Σ A_j = ", round(sum(aj)/1e6;digits=2), " km² vs fire-domain area ",
        round(dom.nx*dom.ny*cell_area/1e6;digits=2), " km² (mass-conservative to ",
        round(100*(1-abs(sum(aj)-dom.nx*dom.ny*cell_area)/(dom.nx*dom.ny*cell_area));digits=2), "%)")

# (D) item 3: LIVE ERA5 data into F_src + loader refresh (guarded on data + python).
if isfile(ERA5_NC) && isfile(ERA5_PY)
    println("(D) LIVE ERA5 Camp Fire data → F_src + loader refresh (no rebuild):")
    lons,lats,u,v,vt = load_era5_wind(dom; time_index=14)
    sp = source_grid_from_points(lons, lats)
    esm = levelset_pde_esm(ls_path, dom; wind=true); disc = discretize(esm)
    fsd = Dict{String,Any}("u_x"=>copy(u),"u_y"=>copy(v),"dzdx"=>zeros(length(u)),"dzdy"=>zeros(length(u)))
    disc2, ca, pa = couple_fields(disc, dom, sp, fsd)
    f!,u0,p,_,vmap = build_evaluator(disc2; parameter_overrides=Dict("R_0"=>r0,"phi_s_coeff"=>phi_s),
        const_arrays=ca, param_arrays=pa)
    X(i)=dom.xmin+(i-1)*dom.dx; Y(j)=dom.ymin+(j-1)*dom.dx
    for i in 1:dom.nx, j in 1:dom.ny
        k=get(vmap,"psi[$i,$j]",nothing); k===nothing && continue
        u0[k]=ESS.evaluate_expr(dom.ic, Dict("x"=>X(i),"y"=>Y(j)))
    end
    band=[k for k in eachindex(u0) if abs(u0[k])<=dom.dx]
    du=similar(u0); f!(du,u0,p,0.0); r14=maximum(-du[k] for k in band)
    _,_,u18,v18,vt18 = load_era5_wind(dom; time_index=18)   # later hour — different wind
    pa["F_src_u_x"] .= u18; pa["F_src_u_y"] .= v18           # update IN PLACE
    f!(du,u0,p,0.0); r18=maximum(-du[k] for k in band)        # SAME f!, no rebuild
    println("    ", length(u), " ERA5 cells; front max spread ", round(r14;digits=2), " m/s @", vt,
            " → refresh in place → ", round(r18;digits=2), " m/s @", vt18,
            "  (Δ=", round(r18-r14;digits=2), "; real data drives + refreshes the front)")
else
    println("(D) live ERA5 data: skipped (ERA5_NC=$(ERA5_NC) or ERA5_PY not found)")
end
