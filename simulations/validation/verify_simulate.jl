#!/usr/bin/env julia
#=
Phase 5 — exercise the new ESS `simulate` one-call run entry (the Julia
counterpart of the Python `simulate`). Proves three things end to end through
the SciMLBase solver extension, with no hand-wired build/seed/solve block:

  A. a scalar ODE   D(y)=1, y(0)=0 over [0,2]      → y(2)=2 (the ext loads,
                                                      SimulationResult plumbing)
  B. a parameter override                            → D(y)=k, y over [0,3]=3k
  C. an array state with a `seed_ic!` closure + an
     element initial_condition override              → seeding paths both fire

Run with plain `julia` (activates the ESS pde_sim_adapter env, which pins ESS +
OrdinaryDiffEqTsit5; set EARTHSCISERIALIZATION to override).
=#
import Pkg
const REPO = dirname(@__DIR__)
const WL = dirname(REPO)
const ESS_ROOT = get(ENV, "EARTHSCISERIALIZATION", normpath(joinpath(WL, "..", "EarthSciSerialization")))
const ESS_PKG = joinpath(ESS_ROOT, "packages", "EarthSciSerialization.jl")
let env = joinpath(ESS_PKG, "scripts", "pde_sim_adapter")
    isdir(env) || error("pde_sim_adapter project not found at $env — set EARTHSCISERIALIZATION")
    Pkg.activate(env; io = devnull)
    isfile(joinpath(env, "Manifest.toml")) || Pkg.develop(path = ESS_PKG; io = devnull)
    Pkg.instantiate(; io = devnull)
end
import EarthSciSerialization as ESS
import OrdinaryDiffEqTsit5: Tsit5

# tiny AST helpers
_D(v) = Dict{String,Any}("op" => "D", "args" => Any[v], "wrt" => "t")
_idx(v, i) = Dict{String,Any}("op" => "index", "args" => Any[v, i])

scalar_esm(rhs) = Dict{String,Any}(
    "esm" => "0.5.0", "metadata" => Dict{String,Any}("name" => "Scalar"),
    "models" => Dict{String,Any}("M" => Dict{String,Any}(
        "variables" => Dict{String,Any}("y" => Dict{String,Any}("type" => "state")),
        "equations" => Any[Dict{String,Any}("lhs" => _D("y"), "rhs" => rhs)])))

function main()
    println("Phase 5 — ESS.simulate one-call run entry\n")

    # A. scalar ODE D(y)=1 over [0,2]  → y(2)=2
    rA = ESS.simulate(scalar_esm(1.0), (0.0, 2.0); alg = Tsit5(),
                      initial_conditions = Dict("y" => 0.0))
    yA = rA["y"][end]
    println("  A scalar  D(y)=1,  y(0)=0,  [0,2]  → y = ", round(yA; digits = 6),
            "  (expected 2.0)  retcode=", rA.retcode)
    @assert rA.success && isapprox(yA, 2.0; atol = 1e-6)

    # B. parameter override:  D(y)=k,  k=2.5,  [0,3]  → 7.5
    esmB = scalar_esm("k")
    esmB["models"]["M"]["variables"]["k"] = Dict{String,Any}("type" => "parameter", "default" => 1.0)
    rB = ESS.simulate(esmB, (0.0, 3.0); alg = Tsit5(),
                      parameters = Dict("k" => 2.5), initial_conditions = Dict("y" => 0.0))
    yB = rB["y"][end]
    println("  B param   D(y)=k,  k=2.5,  [0,3]  → y = ", round(yB; digits = 6), "  (expected 7.5)")
    @assert rB.success && isapprox(yB, 7.5; atol = 1e-5)

    # C. array state u[1..3], D(u[i]) = u[i] (decoupled growth), seeded by a
    #    closure for u[2],u[3] and an explicit element IC for u[1].
    esmC = Dict{String,Any}(
        "esm" => "0.5.0", "metadata" => Dict{String,Any}("name" => "Arr"),
        "models" => Dict{String,Any}("M" => Dict{String,Any}(
            "index_sets" => Dict{String,Any}("n" => Dict{String,Any}("kind" => "interval", "size" => 3)),
            "variables" => Dict{String,Any}("u" => Dict{String,Any}("type" => "state", "shape" => Any["n"])),
            "equations" => Any[Dict{String,Any}(
                "lhs" => Dict{String,Any}("op" => "arrayop", "output_idx" => Any["i"],
                    "ranges" => Dict{String,Any}("i" => Dict{String,Any}("from" => "n")),
                    "args" => Any[], "expr" => _D(_idx("u", "i"))),
                "rhs" => Dict{String,Any}("op" => "arrayop", "output_idx" => Any["i"],
                    "ranges" => Dict{String,Any}("i" => Dict{String,Any}("from" => "n")),
                    "args" => Any[], "expr" => _idx("u", "i")))])))
    seed! = (u0, vm) -> begin
        haskey(vm, "u[2]") && (u0[vm["u[2]"]] = 2.0)
        haskey(vm, "u[3]") && (u0[vm["u[3]"]] = 3.0)
    end
    rC = ESS.simulate(esmC, (0.0, 1.0); alg = Tsit5(),
                      initial_conditions = Dict("u[1]" => 1.0), seed_ic! = seed!)
    # closed form: u_i(1) = u_i(0)·e
    got = [rC["u[1]"][end], rC["u[2]"][end], rC["u[3]"][end]]
    want = [1.0, 2.0, 3.0] .* exp(1)
    println("  C array   D(u)=u,  u0=[1,2,3],  [0,1]  → u(1) = ", round.(got; digits = 4),
            "  (expected ", round.(want; digits = 4), ")")
    @assert rC.success && all(isapprox.(got, want; rtol = 1e-3))
    println("    (u[1] from initial_conditions override, u[2]/u[3] from the seed_ic! closure)")

    println("\n  ✓ ESS.simulate runs scalar + parametric + array states through the SciMLBase ",
            "extension; SimulationResult indexing, parameter overrides, and both IC-seeding paths work.")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
