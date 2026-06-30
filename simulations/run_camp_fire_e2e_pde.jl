#!/usr/bin/env julia
#=
Camp Fire end-to-end (Julia) — a THIN driver over the real ESS / ESD pipeline,
the Julia counterpart of `run_camp_fire_e2e_pde.py`. It does exactly three
things on the coupled `camp_fire.esm`, with NO hand-wired build/seed/solve:

    load        flatten(load(camp_fire.esm))                 — assemble the coupled system
    discretize  fire_discretized + merge_regridder!          — lower the MONOLITHIC fire
                                                                (conservative regrid + per-cell
                                                                Rothermel physics + level-set),
                                                                ESD 2-D centered |grad psi| rule
    run         ESS.simulate(disc, tspan; …)                 — build_evaluator + seed psi from
                                                                the domain IC + Tsit5 solve

Everything reusable (the env bootstrap, the LCC reprojection, the conservative
regrid spec, the Rothermel chain, and the monolithic assembly) lives in
`camp_fire_support.jl`; this file is just the orchestration + reporting. The
front spreads at the data-computed per-cell Rothermel rate and is wind-elongated
downwind under the regridded W→E wind ramp (the per-field ablation lives in
verify_coupling.jl / verify_simulate_fire.jl). The window is SHORT by design —
the goal is to drive the real pipeline through `simulate`, not reproduce the fire.

USAGE
    EARTHSCISERIALIZATION=.../EarthSciSerialization \
        julia simulations/run_camp_fire_e2e_pde.jl [--seconds 30]
=#
include(joinpath(@__DIR__, "camp_fire_support.jl"))
import OrdinaryDiffEqTsit5: Tsit5

function parse_seconds(args)
    for (i, a) in enumerate(args)
        a == "--seconds" && i < length(args) && return parse(Float64, args[i + 1])
        startswith(a, "--seconds=") && return parse(Float64, split(a, "=", limit = 2)[2])
    end
    return 30.0
end

function main(args = ARGS)
    seconds = parse_seconds(args)
    println("Camp Fire — thin Julia driver (load camp_fire.esm → discretize via ESD → run via ESS.simulate)\n")

    # 1+2. LOAD + DISCRETIZE — the MONOLITHIC coupled system (conservative regrid +
    #      per-cell Rothermel physics + level-set front) lowered with the ESD 2-D
    #      centered |grad psi| rule; promote=true does the whole monolithic lowering.
    dom = camp_domain(CAMP_FIRE_ESM)
    src_poly, field_src, ncol, nrow = representative_forcing(dom)
    println("== load + discretize (ESD centered |grad psi|; promote=true monolithic lowering) ==")
    disc = fire_discretized(dom)
    ca, pa = merge_regridder!(disc, dom, src_poly, field_src)
    println("   domain $(dom.nx)×$(dom.ny) @ $(round(Int, dom.dx)) m  ($(dom.tstart) → $(dom.tend)); ",
            "source grid $(ncol)×$(nrow); RothermelFireSpread.R promoted to $(rothermel_shape(disc)) (per-cell)")

    # 3. RUN — one ESS.simulate call: build_evaluator + seed psi from the domain IC
    #    + Tsit5 solve. The conservative regrid is computed inside the ODE RHS.
    println("== run (ESS.simulate; Tsit5); tspan=(0, $(round(seconds)) s) ==")
    t0 = time()
    r = simulate(disc, (0.0, seconds); alg = Tsit5(),
                 const_arrays = ca, param_arrays = pa, seed_ic! = psi_seeder(dom),
                 parameters = Dict("LevelSetFireSpread.dx" => dom.dx),
                 reltol = 1e-4, abstol = 1e-6)
    dt = time() - t0
    if !r.success
        println("   simulate did not complete ($(round(dt; digits = 0))s): retcode=$(r.retcode)")
        return 1
    end

    psi(u, i, j) = (k = get(r.var_map, "LevelSetFireSpread.psi[$i,$j]", nothing); k === nothing ? NaN : u[k])
    burned(u) = count(((i, j),) -> (v = psi(u, i, j); !isnan(v) && v < 0),
                      [(i, j) for i in 1:dom.nx for j in 1:dom.ny])
    println("   solved in $(round(dt; digits = 0))s: $(length(r.t)) steps over $(round(seconds)) s, ",
            "$(length(r.var_map)) state elements")
    println("   fire front (level-set psi<0 burned cells): $(burned(r.u[1])) → $(burned(r.u[end])) ",
            "— the conservative-regrid + per-cell Rothermel + level-set front ran in ONE simulate call")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
