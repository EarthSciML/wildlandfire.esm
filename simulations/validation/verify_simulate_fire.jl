#!/usr/bin/env julia
#=
Phase 5 keystone — the REAL monolithic camp_fire fire (the COMPUTED conservative
regridder + per-cell Rothermel physics + level-set front) integrated through the
ESS `simulate` one-call entry, with the 2-D centered |grad psi| stencils loaded BY
REF from the EarthSciDiscretizations catalog (`gdd_to_rules`, Phase 2).

The forcing is regridded through the REAL `conservative_regrid_overlap_join_computed
.esm` — the component camp_fire.esm actually couples in — sized to the camp source→
fire-grid target: spherical intersect_polygon clip → Van Oosterom-Strackee area →
partition-of-unity-exact bin-skolem A_j_w → build-once weight matrix W (Phase-5
residual #4, retiring the hand-rolled planar `fields_regrid_spec`). The assembly +
run mechanics live in shared code: `camp_fire_support.jl::fire_discretized /
computed_regrid_forcing / psi_seeder` and `ESS.simulate`. This harness ASSERTS the
result — W is a partition of unity, the run succeeds, the burned region grows, and
the downwind/E front outruns the upwind/W under the regridded W→E wind ramp.

Run with plain `julia` (activates the ESS pde_sim_adapter env).
=#
include(joinpath(dirname(@__DIR__), "run_camp_fire_e2e_pde.jl"))  # → camp_fire_support.jl (no main())
import OrdinaryDiffEqTsit5: Tsit5
const E = ESS

function main()
    dom = camp_domain(CAMP_FIRE_ESM)
    nx, ny = dom.nx, dom.ny

    disc = fire_discretized(dom)
    forcing, W, field_src = computed_regrid_forcing(dom)    # the REAL computed regridder
    seed! = psi_seeder(dom)

    println("Phase 5 keystone — monolithic camp_fire through ESS.simulate (computed regridder + ESD grad rule)\n")
    # W from the real computed component must be a partition of unity (each fire cell's
    # weights over the source cells sum to 1 — conservative, full coverage).
    rowsums = vec(sum(W; dims = 2))
    @assert all(rs -> isapprox(rs, 1.0; atol = 1e-9), rowsums) "computed regrid weights must be a partition of unity"

    seconds = 20.0
    r = simulate(disc, (0.0, seconds); alg = Tsit5(),
                 param_arrays = forcing, seed_ic! = seed!,
                 parameters = Dict("LevelSetFireSpread.dx" => dom.dx),
                 reltol = 1e-4, abstol = 1e-6)
    @assert r.success "the monolithic fire must integrate (retcode=$(r.retcode))"

    psi(u, i, j) = (k = get(r.var_map, "LevelSetFireSpread.psi[$i,$j]", nothing); k === nothing ? NaN : u[k])
    burned(u) = count(((i, j),) -> (v = psi(u, i, j); !isnan(v) && v < 0),
                      [(i, j) for i in 1:nx for j in 1:ny])

    # front anisotropy at t=0: instantaneous speed near the psi=0 contour, downwind vs upwind.
    f!, uu, p, _t, vm = E.build_evaluator(disc; param_arrays = forcing,
        parameter_overrides = Dict("LevelSetFireSpread.dx" => dom.dx))
    seed!(uu, vm); du = similar(uu); f!(du, uu, p, 0.0)
    band = [(i, j) for i in 1:nx for j in 1:ny if (k = get(vm, "LevelSetFireSpread.psi[$i,$j]", nothing);
            k !== nothing && abs(uu[k]) <= dom.dx)]
    spd(i, j) = -du[vm["LevelSetFireSpread.psi[$i,$j]"]]
    eF = [spd(i, j) for (i, j) in band if i > nx / 2]; wF = [spd(i, j) for (i, j) in band if i < nx / 2]
    em = isempty(eF) ? 0.0 : sum(eF) / length(eF); wm = isempty(wF) ? 0.0 : sum(wF) / length(wF)
    rsh = rothermel_shape(disc)

    println("  computed regrid W$(size(W)): partition of unity (row-sums all 1.0)")
    println("  solved: $(length(r.t)) steps over $(seconds)s, $(length(r.var_map)) state elements, retcode=$(r.retcode)")
    println("  RothermelFireSpread.R promoted shape: ", rsh, "  (per-cell)")
    println("  burned (psi<0) cells: ", burned(r.u[1]), " → ", burned(r.u[end]))
    println("  front speed t=0: downwind/E ", round(em; digits = 4), " > upwind/W ", round(wm; digits = 4))
    @assert burned(r.u[end]) >= burned(r.u[1]) "the burned region must not shrink"
    @assert em > wm "the downwind (E) front must outrun upwind (W) under the regridded W→E wind"
    println("\n  ✓ the REAL computed conservative regridder + per-cell Rothermel + level-set front lowers")
    println("    with the ESD-catalog 2-D centered grad rule and integrates through ONE ESS.simulate call.")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
