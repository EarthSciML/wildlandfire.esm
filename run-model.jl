#!/usr/bin/env julia
# run-model.jl — minimal single-model runner for a 2-D EarthSci `.esm` (Julia).
#
# Model: 2-D level-set Hamilton-Jacobi front propagation  psi_t = -S |grad psi|
# on the doubly-periodic unit box, with |grad psi| discretized by the Godunov
# (Rouy-Tourin / Osher-Sethian) upwind scheme (imported by reference from the
# sibling earthscidiscretizations checkout — nothing is vendored here). psi is
# initialized as the signed distance to a circle, so the front {psi = 0} expands
# outward at speed S. Loads through EarthSciSerialization.jl, integrates while
# saving several snapshots, and plots the front location at each time as a single
# figure with time encoded by color (PNG). No conformance / MMS checks.
#
# Usage:
#   julia run-model.jl [model.esm] [t_end]
#     model.esm   path to the .esm to run   (default: ./wildlandfire.esm)
#     t_end       final integration time    (default: 2.0)
#
# Environment: a local project (run-model-jl/) layering Plots.jl on top of the
# dev'd EarthSciSerialization + OrdinaryDiffEqTsit5. Set ESS_ROOT to override.

import Pkg

const HERE = @__DIR__
const ESS_ROOT = get(ENV, "ESS_ROOT",
                     normpath(joinpath(HERE, "..", "EarthSciSerialization")))
isfile(joinpath(ESS_ROOT, "esm-schema.json")) ||
    error("EarthSciSerialization not found at '$ESS_ROOT'; set ESS_ROOT or " *
          "clone it as a sibling checkout of this repo.")

let env = joinpath(HERE, "run-model-jl")
    mkpath(env)
    Pkg.activate(env; io=devnull)
    if !isfile(joinpath(env, "Manifest.toml"))
        Pkg.develop(path=joinpath(ESS_ROOT, "packages", "EarthSciSerialization.jl"); io=devnull)
        Pkg.add(["OrdinaryDiffEqTsit5", "Plots"]; io=devnull)
    end
    Pkg.instantiate(; io=devnull)
end

using EarthSciSerialization
import OrdinaryDiffEqTsit5
using Printf
using Plots
const ESS = EarthSciSerialization

const NT = 6   # number of front snapshots (t = 0 … t_end)
const N  = 32  # grid resolution (metaparameters NX = NY = N); matches the Rust
               # runner, which is capped here by the non-smooth Godunov RHS cost.
const R0 = 0.1 # LevelSetFireSpread.R_0 override (m/s): the mounted component's
               # default R_0=1.0 was calibrated for its old 500 m domain; on the
               # unit box that overshoots, so we set the isotropic speed to 0.1
               # (front reaches 0.15 + 0.1·t_end, staying clear of the seam).

model_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(HERE, "wildlandfire.esm")
t_end      = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 2.0

println("loading    $model_path  (NX=NY=$N)")
file = ESS.load(model_path; metaparameters = Dict("NX" => N, "NY" => N))

ts = collect(range(0.0, t_end; length = NT))
println("simulating (0.0, $t_end) with Tsit5, $NT snapshots …")
sim = ESS.simulate(file, (0.0, t_end);
                   alg=OrdinaryDiffEqTsit5.Tsit5(),
                   parameters = Dict("LevelSetFireSpread.R_0" => R0),
                   reltol=1e-2, abstol=1e-3, saveat=ts)
sim.success || error("solver failed: retcode=$(sim.retcode) — $(sim.message)")

# Assemble the primary 2-D field: state elements "<stem>[i,j]", grouped by stem,
# indices parsed as integers.
groups = Dict{String,Vector{NTuple{3,Int}}}()   # stem => [(i, j, slot)]
for (name, slot) in sim.var_map
    m = match(r"^(.*)\[(\d+),(\d+)\]$", name)
    m === nothing && continue
    push!(get!(groups, m.captures[1], NTuple{3,Int}[]),
          (parse(Int, m.captures[2]), parse(Int, m.captures[3]), slot))
end
isempty(groups) && error("no 2-D indexed state field found to plot")
# Prefer the level-set state field psi (the mounted component also exposes many
# [x,y] observeds); fall back to the widest 2-D field if no psi is present.
let psikeys = filter(k -> endswith(k, "psi") || occursin(r"\bpsi$", k), collect(keys(groups)))
    global stem = isempty(psikeys) ? argmax(k -> length(groups[k]), keys(groups)) :
                  argmax(k -> length(groups[k]), psikeys)
end
cells = groups[stem]
NX = maximum(c -> c[1], cells)
NY = maximum(c -> c[2], cells)
xs = [(i - 0.5) / NX for i in 1:NX]
ys = [(j - 0.5) / NY for j in 1:NY]
# psi[k] is the NX×NY field at saved time sim.t[k].
psi = [fill(NaN, NX, NY) for _ in eachindex(sim.t)]
for (i, j, slot) in cells, k in eachindex(sim.t)
    psi[k][i, j] = sim.u[k][slot]
end

# Marching squares for the zero contour of a field sampled at (xs[i], ys[j]);
# returns line segments packed into (X, Y) with NaN breaks (one plot series).
function zero_segments(xs, ys, Z)
    X = Float64[]; Y = Float64[]
    interp(a, b) = (t = a[3] / (a[3] - b[3]); (a[1] + t*(b[1]-a[1]), a[2] + t*(b[2]-a[2])))
    for i in 1:length(xs)-1, j in 1:length(ys)-1
        c = ((xs[i],   ys[j],   Z[i, j]),   (xs[i+1], ys[j],   Z[i+1, j]),
             (xs[i+1], ys[j+1], Z[i+1, j+1]), (xs[i],  ys[j+1], Z[i, j+1]))  # CCW
        pts = Tuple{Float64,Float64}[]
        for e in 1:4
            a = c[e]; b = c[e % 4 + 1]
            (a[3] < 0) != (b[3] < 0) && push!(pts, interp(a, b))
        end
        if length(pts) >= 2
            push!(X, pts[1][1], pts[2][1], NaN)
            push!(Y, pts[1][2], pts[2][2], NaN)
        end
    end
    X, Y
end

cellarea = (1.0 / NX) * (1.0 / NY)
println("\nsolver: success ($(sim.retcode)); field '$stem' is $(NX)×$(NY)")
for (k, t) in enumerate(sim.t)
    r = sqrt(sum(psi[k][i, j] < 0 ? cellarea : 0.0 for i in 1:NX, j in 1:NY) / pi)
    @printf("  t=%.2f  front r_eff=%.4f\n", t, r)
end

# --- single figure: front (psi=0) at each time, color = time; PNG output -------
gr()
cg = cgrad(["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])   # sequential blue (early→late)
p = plot(size = (620, 540), aspect_ratio = 1, xlims = (0, 1), ylims = (0, 1),
    xlabel = "x", ylabel = "y", framestyle = :box, legend = :outertopright,
    background_color = "#fcfcfb", foreground_color = "#52514e",
    title = "$(basename(model_path)) — front {$stem=0}, Julia / Tsit5",
    titlefontsize = 11, guidefontsize = 10, legendfontsize = 8, legendtitle = "t")
for (k, t) in enumerate(sim.t)
    X, Y = zero_segments(xs, ys, psi[k])
    frac = length(sim.t) == 1 ? 1.0 : (k - 1) / (length(sim.t) - 1)
    plot!(p, X, Y; color = cg[frac], lw = 2, label = @sprintf("%.2f", t))
end

plot_path = joinpath(dirname(abspath(model_path)), "front_julia.png")
savefig(p, plot_path)
println("wrote plot: $plot_path")
