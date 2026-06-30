#!/usr/bin/env julia
#=
Phase 3a — validate the declarative ESD target-cell geometry component
(earthscidiscretizations/regridding/cartesian_target_cells.esm). It emits, from the
uniform-cartesian grid spec (origin x0/y0, spacing dx/dy, sizes gx/gy), the per-cell
centres `tgt_lon`/`tgt_lat` and the four-corner rings `tgt_poly[x,y,cell_verts,coord]`
that the conservative regridder otherwise takes as HOST-SUPPLIED parameters — replacing
the driver's hand-built `source_grid_lcc`/`fields_regrid_spec` target construction.

This builds the component through ESS, reads the geometry back via probe states, and
checks it against the analytic affine formula on the Camp Fire 19x21 @ 2 km LCC grid:
centre = xmin + (x-1)*dx; corners = the CCW ring (W,S)->(E,S)->(E,N)->(W,N).

Run with plain `julia` (activates the ESS pde_sim_adapter env; set EARTHSCISERIALIZATION
and EARTHSCIDISCRETIZATIONS to override).
=#
import Pkg
const REPO = dirname(@__DIR__)
const ESS_ROOT = get(ENV, "EARTHSCISERIALIZATION", normpath(joinpath(REPO, "..", "EarthSciSerialization")))
const ESS_PKG = joinpath(ESS_ROOT, "packages", "EarthSciSerialization.jl")
const ESD_ROOT = get(ENV, "EARTHSCIDISCRETIZATIONS", normpath(joinpath(REPO, "..", "earthscidiscretizations")))
const COMP = joinpath(ESD_ROOT, "regridding", "cartesian_target_cells.esm")
let env = joinpath(ESS_PKG, "scripts", "pde_sim_adapter")
    isdir(env) || error("pde_sim_adapter project not found at $env — set EARTHSCISERIALIZATION")
    Pkg.activate(env; io = devnull)
    isfile(joinpath(env, "Manifest.toml")) || Pkg.develop(path = ESS_PKG; io = devnull)
    Pkg.instantiate(; io = devnull)
end
using EarthSciSerialization
using JSON3
const E = EarthSciSerialization

# Camp Fire grid spec (matches the component's defaults and camp_fire.esm's domain).
const NX, NY, DX, DY = 19, 21, 2000.0, 2000.0
const X0, Y0 = -2027020.2, 373725.0          # SW-most cell edges = (xmin,ymin) - (dx,dy)/2
agg(name) = Dict{String,Any}("op"=>"aggregate","output_idx"=>Any["x","y"],
    "ranges"=>Dict{String,Any}("x"=>Dict{String,Any}("from"=>"gx"),"y"=>Dict{String,Any}("from"=>"gy")),
    "expr"=>Dict{String,Any}("op"=>"D","args"=>Any[Dict{String,Any}("op"=>"index","args"=>Any[name,"x","y"])],"wrt"=>"t"))
arr(body) = Dict{String,Any}("op"=>"arrayop","args"=>Any[],"output_idx"=>Any["x","y"],
    "ranges"=>Dict{String,Any}("x"=>Dict{String,Any}("from"=>"gx"),"y"=>Dict{String,Any}("from"=>"gy")),"expr"=>body)
ix(a...) = Dict{String,Any}("op"=>"index","args"=>collect(Any,a))

function main()
    doc = JSON3.read(read(COMP, String), Dict{String,Any})
    model = doc["models"]["CartesianTargetCells"]
    # Probe each geometry output into a state we can read from du.
    probes = Dict("lon"=>ix("tgt_lon","x","y"), "lat"=>ix("tgt_lat","x","y"),
        "pWx"=>ix("tgt_poly","x","y",1,1), "pEx"=>ix("tgt_poly","x","y",2,1),
        "pSy"=>ix("tgt_poly","x","y",1,2), "pNy"=>ix("tgt_poly","x","y",3,2))
    for (nm, body) in probes
        model["variables"]["w_$nm"] = Dict{String,Any}("type"=>"state","shape"=>Any["gx","gy"])
        push!(get!(model, "equations", Any[]),
              Dict{String,Any}("lhs"=>agg("w_$nm"), "rhs"=>arr(body)))
    end
    ic = Dict{String,Float64}("w_$nm[$i,$j]"=>0.0 for nm in keys(probes), i in 1:NX, j in 1:NY)
    f!, u0, p, _t, vmap = E.build_evaluator(doc; initial_conditions=ic)
    du = similar(u0); f!(du, u0, p, 0.0)
    g(nm,i,j) = du[vmap["w_$nm[$i,$j]"]]

    # Analytic affine reference.
    cx(i) = X0 + (i-1)*DX + DX/2
    cy(j) = Y0 + (j-1)*DY + DY/2
    wx(i) = X0 + (i-1)*DX;  ex(i) = X0 + i*DX
    sy(j) = Y0 + (j-1)*DY;  ny_(j) = Y0 + j*DY

    maxerr = 0.0
    for i in 1:NX, j in 1:NY
        maxerr = max(maxerr,
            abs(g("lon",i,j)-cx(i)), abs(g("lat",i,j)-cy(j)),
            abs(g("pWx",i,j)-wx(i)), abs(g("pEx",i,j)-ex(i)),
            abs(g("pSy",i,j)-sy(j)), abs(g("pNy",i,j)-ny_(j)))
    end

    println("Declarative ESD target-cell geometry (cartesian_target_cells.esm) — Phase 3a\n")
    println("  grid: $(NX)×$(NY) @ $(round(Int,DX)) m, x0=$(X0) y0=$(Y0)")
    println("  cell (1,1)  centre=(", round(g("lon",1,1);digits=1), ",", round(g("lat",1,1);digits=1),
            ")  ring x∈[", round(g("pWx",1,1);digits=1), ",", round(g("pEx",1,1);digits=1),
            "] y∈[", round(g("pSy",1,1);digits=1), ",", round(g("pNy",1,1);digits=1), "]")
    println("  cell ($NX,$NY) centre=(", round(g("lon",NX,NY);digits=1), ",", round(g("lat",NX,NY);digits=1), ")")
    println("  max |declarative − analytic affine| over all $(NX*NY) cells: ", maxerr)

    @assert g("lon",1,1) ≈ -2026020.2   "cell-1 centre x must equal domain xmin"
    @assert g("lat",1,1) ≈ 374725.0     "cell-1 centre y must equal domain ymin"
    @assert g("lon",NX,NY) ≈ -1990020.2 "cell-NX centre x must equal domain xmax"
    @assert maxerr < 1e-6               "declarative geometry must match the analytic affine formula"
    println("\n  ✓ the conservative regridder's TARGET geometry (tgt_poly / tgt_lon / tgt_lat) is")
    println("    produced declaratively from the grid spec by an ESD component — no host-built polygons.")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
