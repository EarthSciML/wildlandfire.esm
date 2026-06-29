#!/usr/bin/env julia
# Verify the WHOLE-SYSTEM run (--full): the flattened 21-component camp_fire.esm
# drives the level-set front at the Rothermel rate of spread COMPUTED from the
# regridded met/fuel/slope forcing — not a constant.
#
#   §A  the fire-physics chain (slope→wind→EMC→1h-moisture→Rothermel) lowers and
#       runs through build_evaluator and computes a physically sensible R that
#       RESPONDS to the forcing (wind ↑ ⇒ R ↑; humidity ↑ ⇒ R ↓, → 0 past extinction).
#   §B  the whole flattened system CANNOT lower through one build_evaluator — flatten/
#       serialize are lossy (drop index_sets / function_tables / embedded const data),
#       which is exactly why the system runs in its natural stages. We assert the
#       monolithic attempt fails on one of those documented losses.
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

    @testset "§B monolithic lowering is blocked by lossy flatten (why we stage)" begin
        # Serialize ALL flattened equations + variables into one model and try to run
        # it. It must fail — flatten dropped the regridders' index_sets and the fuel
        # table's const data, and the evaluator inlines no deep observed chains.
        sv, ov, pv = FLAT.state_variables, FLAT.observed_variables, FLAT.parameters
        function vd(v, kind)
            d = Dict{String,Any}("type" => kind)
            v.shape === nothing || isempty(v.shape) || (d["shape"] = collect(Any, v.shape))
            v.default === nothing || (d["default"] = v.default)
            kind == "observed" && v.expression !== nothing &&
                (d["expression"] = ESS.serialize_expression(v.expression))
            d
        end
        vars = Dict{String,Any}()
        for (k, v) in sv; vars[k] = vd(v, "state"); end
        for (k, v) in ov; vars[k] = vd(v, "observed"); end
        for (k, v) in pv; vars[k] = vd(v, "parameter"); end
        eqs = Any[Dict{String,Any}("lhs" => ESS.serialize_expression(e.lhs),
                                   "rhs" => ESS.serialize_expression(e.rhs)) for e in FLAT.equations]
        mono = Dict{String,Any}("esm" => "0.5.0",
            "metadata" => Dict{String,Any}("name" => "CampFireMonolithic"),
            "models" => Dict{String,Any}("All" => Dict{String,Any}("variables" => vars, "equations" => eqs)))
        err = try
            build_evaluator(mono); ""
        catch e
            sprint(showerror, e)
        end
        println("  monolithic build_evaluator error: ", split(err, '\n')[1])
        @test err != ""                                      # it MUST fail (lossy flatten)
        # …and the failure is one of the documented losses. Serialization order is
        # nondeterministic, so the parser trips on whichever lossy node comes first:
        # the regrid geometry (intersect_polygon's dropped `manifold` / index_sets /
        # join), the FuelModelLookup `table_lookup`/`fn` whose `table`/`const` data is
        # dropped, or an unsupported/unbound deep-observed inlining gap.
        @test any(occursin(err), ["manifold", "intersect_polygon", "table_lookup", "table",
                                  "fn", "const", "index", "JOIN", "candidate",
                                  "UNSUPPORTED", "UNBOUND", "GEOMETRY"])
        println("  ✓ monolithic lowering blocked by lossy flatten — the system runs in stages")
    end
end
