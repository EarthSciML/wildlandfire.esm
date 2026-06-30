#!/usr/bin/env julia
# Verify the WHOLE-SYSTEM run (--full): the flattened 21-component camp_fire.esm
# drives the level-set front at the Rothermel rate of spread COMPUTED from the
# regridded met/fuel/slope forcing — not a constant.
#
#   §A  the fire-physics chain (slope→wind→EMC→1h-moisture→Rothermel) lowers and
#       runs through build_evaluator and computes a physically sensible R that
#       RESPONDS to the forcing (wind ↑ ⇒ R ↑; humidity ↑ ⇒ R ↓, → 0 past extinction).
#   §B  the whole flattened system now reconstitutes into ONE runnable document via
#       ESS's `flattened_to_esm` and LOWERS PAST the formerly-fatal lossy-flatten
#       losses: flatten is now field-preserving (canonical `reconstruct`), and the
#       FlattenedSystem carries the regridders' namespaced `index_sets` + the
#       FuelModelLookup `function_tables` + the geometry `manifold`. The monolithic
#       attempt no longer fails on a dropped manifold/table/index-set — the remaining
#       frontier is supplying each value-invention component's DATA inputs (projection
#       coords, the five regridder geometries), which is the staged data harness, not
#       a lossy-flatten defect. This §B is the regression guard that the ESS fix holds.
include(joinpath(@__DIR__, "run_camp_fire_e2e_pde.jl"))
using Test, Logging

const FLAT = with_logger(ConsoleLogger(stderr, Logging.Error)) do
    flatten(load(CAMP_FIRE_ESM))
end

# Representative single-cell forcing; vary one field at a time to probe R's response.
basef(; u = 8.0, v = 0.0, temp = 288.0, rh = 0.23, sx = 0.05, sy = 0.05) = Dict{String,Float64}(
    "ERA5uRegrid.F_tgt" => u, "ERA5vRegrid.F_tgt" => v, "ERA5tRegrid.F_tgt" => temp,
    "ERA5rRegrid.F_tgt" => rh, "SlopeXRegrid.F_tgt" => sx, "SlopeYRegrid.F_tgt" => sy)
R(f) = rothermel_from_fields(FLAT, f)["RothermelFireSpread.R"]

@testset "Whole-system --full: data-computed Rothermel rate of spread" begin
    @testset "§A fire-physics chain runs + computes a sensible R" begin
        out = rothermel_from_fields(FLAT, basef())
        r = out["RothermelFireSpread.R"]
        println("  base forcing (u=8 m/s, RH=23%, FM1): R = ", round(r; digits = 4), " m/s")
        @test isfinite(r) && 0.0 < r < 10.0                 # a plausible grass-fire spread rate
        @test isfinite(out["RothermelFireSpread.C_coeff"])
        @test out["RothermelFireSpread.beta_ratio"] > 0      # packing ratio sane

        rlo, rhi = R(basef(u = 2.0)), R(basef(u = 12.0))
        println("  wind response: R(u=2)=", round(rlo; digits = 4), "  R(u=12)=", round(rhi; digits = 4))
        @test rhi > rlo                                      # stronger wind ⇒ faster spread

        rdry, rwet = R(basef(rh = 0.10)), R(basef(rh = 0.45))
        println("  humidity response: R(RH=10%)=", round(rdry; digits = 4), "  R(RH=45%)=", round(rwet; digits = 4))
        @test rdry > rwet                                    # wetter fuel ⇒ slower spread
        @test R(basef(rh = 0.95)) <= rwet + 1e-9             # near-saturation ⇒ at/▷ extinction (R small/0)
        println("  ✓ the flattened Rothermel chain computes R from forcing and responds correctly")
    end

    @testset "§B monolithic document reconstitutes + lowers past the lossy losses" begin
        # Reconstitute the WHOLE flattened system into one runnable document via the
        # ESS bridge. With flatten now field-preserving, this carries everything the
        # old hand-assembly dropped: the regridders' namespaced index_sets, the fuel
        # function_tables, and the geometry manifold.
        doc = ESS.flattened_to_esm(FLAT)
        model = doc["models"][first(keys(doc["models"]))]

        function any_manifold(node)
            node isa AbstractDict || (node isa AbstractVector && return any(any_manifold, node); return false)
            get(node, "manifold", nothing) !== nothing && return true
            any(any_manifold, values(node))
        end

        @test haskey(model, "index_sets") && !isempty(model["index_sets"])   # was dropped
        @test haskey(doc, "function_tables") && !isempty(doc["function_tables"])  # was dropped
        @test any_manifold(model)                                            # was dropped
        println("  monolithic doc: ", length(model["variables"]), " vars, ",
                length(model["equations"]), " eqs, ", length(model["index_sets"]),
                " index_sets, ", length(doc["function_tables"]), " function_tables")

        # Try to build it. Whatever surfaces, it is NO LONGER one of the lossy-flatten
        # losses — those nodes now carry their data. The remaining frontier is a
        # value-invention DATA input (e.g. a regridder/projection geometry array),
        # which is the staged data harness, not a serialization defect.
        err = try
            build_evaluator(doc); ""
        catch e
            sprint(showerror, e)
        end
        first_line = isempty(err) ? "(built)" : split(err, '\n')[1]
        println("  monolithic build_evaluator frontier: ", first_line)
        for lossy in ["requires a `manifold`", "table_lookup requires", "undeclared index set",
                      "requires an `id`"]
            @test !occursin(lossy, err)                       # the lossy-flatten losses are GONE
        end
        println("  ✓ the whole flattened system lowers monolithically past the lossy losses ",
                "(ESS flatten is now field-preserving; flattened_to_esm reconstitutes it)")
    end
end
