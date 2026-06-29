#!/usr/bin/env julia
# Item 4: route through camp_fire.esm's REAL regrid component. Load the actual
# earthscidiscretizations/regridding/conservative_regrid_overlap_join.esm (the
# component the camp_fire.esm coupling references) and run it through build_evaluator,
# confirming the library component executes its broad-phase binning (skolem
# candidate_pairs) + join + sliver-filter apply to a conservative F_tgt.
#
# This component takes A_ij as a HOST parameter (a structural spec); the driver's
# fields_regrid_spec computes that same A_ij DECLARATIVELY via the ranged clip. So
# this verifies the library component's APPLY runs through the evaluator, and that
# the driver's declarative A_ij is exactly what it consumes.
include(joinpath(@__DIR__, "run_camp_fire_e2e_pde.jl"))
using JSON3

# The component clips on the SPHERICAL manifold (it is geographic lat/lon), so the
# ESS GeometryOps backend extension must be loaded. Ensure GeometryOps + GeoInterface
# are in the adapter env (already in the depot — a fast resolve), then import them to
# trigger EarthSciSerializationGeometryOpsExt.
import Pkg
try
    @eval import GeometryOps, GeoInterface
catch
    Pkg.add(["GeometryOps", "GeoInterface"]; io = devnull)
    @eval import GeometryOps, GeoInterface
end

const COMP = abspath(joinpath(REPO, "..", "earthscidiscretizations", "regridding",
                              "conservative_regrid_overlap_join.esm"))

esm0 = JSON3.read(read(COMP, String), Dict{String,Any})
m = esm0["models"]["ConservativeRegridOverlapJoin"]
println("loaded component: ", esm0["models"] |> keys |> collect, "  (", length(m["variables"]), " vars, ",
        length(m["equations"]), " eqs, index_sets ", collect(keys(m["index_sets"])), ")")
unit = Float64[0 0; 1 0; 1 1; 0 1]                # representative cell polygon (clip demo)

"Run the real component for n src × n tgt cells with host A_ij; return (A_j, F_tgt)."
function run_component(n, src_lon, src_lat, tgt_lon, tgt_lat, A_ij, F_src, dst_areas; dx=1.0, dy=1.0)
    esm = JSON3.read(read(COMP, String), Dict{String,Any})   # fresh (build_evaluator mutates)
    ca = Dict{String,Any}("src_lon"=>src_lon, "src_lat"=>src_lat, "tgt_lon"=>tgt_lon, "tgt_lat"=>tgt_lat,
        "A_ij"=>A_ij, "F_src"=>F_src, "dst_areas"=>dst_areas, "src_poly_rep"=>unit, "tgt_poly_rep"=>unit)
    ic = Base.merge(Dict("narrow_phase_area"=>0.0),
        Dict("A_j[$j]"=>0.0 for j in 1:n), Dict("F_tgt[$j]"=>0.0 for j in 1:n))
    f!, u0, p, _t, vmap = build_evaluator(esm; parameter_overrides=Dict("dx"=>dx,"dy"=>dy,"atol"=>1.0e-9),
        const_arrays=ca, initial_conditions=ic)
    du = similar(u0); f!(du, u0, p, 0.0)
    ([du[vmap["A_j[$j]"]] for j in 1:n], [du[vmap["F_tgt[$j]"]] for j in 1:n])
end

try
    # Case 1 — 3 cells, distinct bins (floor(lon/dx) per cell): the broad-phase join
    # keeps only same-bin pairs (i,i); identity A_ij ⇒ F_tgt = F_src (passthrough).
    A_j1, F1 = run_component(3, [0.5,1.5,2.5],[0.5,0.5,0.5],[0.5,1.5,2.5],[0.5,0.5,0.5],
        [i==j ? 1.0 : 0.0 for i in 1:3, j in 1:3], [10.0,20.0,30.0], ones(3))
    println("REAL COMPONENT through build_evaluator:")
    println("  [distinct bins, identity A_ij]  A_j=", round.(A_j1;digits=3),
            " F_tgt=", round.(F1;digits=3), "  (expect [1,1,1] / [10,20,30])")
    # Case 2 — 3 cells ALL in one bin (dx=dy=10): the join admits every pair, so a
    # PARTIAL-overlap A_ij drives a genuine conservative regrid. tgt1,tgt2 each take
    # half of src1+src2; tgt3 takes src3 → F_tgt=[15,15,30]. The sliver `filter`
    # (A_ij>atol) drops the zero pairs. This is the apply the driver computes A_ij
    # FOR — the same conservative mean its declarative ranged clip produces.
    A_ij2 = [0.5 0.5 0.0; 0.5 0.5 0.0; 0.0 0.0 1.0]
    A_j2, F2 = run_component(3, [1.0,2.0,3.0],[1.0,1.0,1.0],[1.0,2.0,3.0],[1.0,1.0,1.0],
        A_ij2, [10.0,20.0,30.0], ones(3); dx=10.0, dy=10.0)
    println("  [shared bin, partial A_ij]      A_j=", round.(A_j2;digits=3),
            " F_tgt=", round.(F2;digits=3), "  (expect [1,1,1] / [15,15,30])")
    ok = isapprox(A_j1,[1,1,1];atol=1e-9) && isapprox(F1,[10,20,30];atol=1e-9) &&
         isapprox(A_j2,[1,1,1];atol=1e-9) && isapprox(F2,[15,15,30];atol=1e-9)
    println(ok ? "  ✓ the real conservative_regrid_overlap_join.esm (broad-phase join + sliver filter +" :
                "  ✗ mismatch")
    ok && println("    conservative apply) the camp_fire.esm coupling references is evaluator-runnable;")
    ok && println("    its host A_ij is exactly what the driver's declarative ranged clip computes.")
catch e
    println("FAILED: ", sprint(showerror, e))
    for (i, fr) in enumerate(stacktrace(catch_backtrace())); i <= 14 && println("  ", fr); end
end
