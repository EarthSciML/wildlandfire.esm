# Running the 2-D level-set model (`wildlandfire.esm`)

A minimal, three-language harness for running one **2-D** EarthSci `.esm` model
here — the **level-set Hamilton-Jacobi** fire-front-propagation equation
`ψ_t = -S(n̂) |∇ψ|` on the doubly-periodic unit box `[0,1]²`, with the gradient
magnitude `|∇ψ| = √((∂ψ/∂x)² + (∂ψ/∂y)²)` discretized by the **Godunov**
(Rouy-Tourin / Osher-Sethian) first-order upwind scheme. `ψ` is initialized as the
signed distance to a circle of radius 0.15 at `(0.5, 0.5)`, so with calm/flat
conditions (`S = R_0`) the front `{ψ = 0}` advances outward at speed `R_0` — a
circle of radius `0.15 + R_0·t`.

**Assembled by reference — nothing is hand-rolled.** `wildlandfire.esm` is a
thin wrapper that imports the pre-built **`LevelSetFireSpread`** model component
(the Mandel 2011 / Muñoz-Esparza 2018 fire-spread PDE) from the sibling
`../earthscimodels` checkout by `ref`. That component in turn connects its spatial
discretization by reference, through its own `expression_template_imports`: the
Godunov rule for the `√(D(ψ,x)² + D(ψ,y)²)` gradient-magnitude compound, plus the
central `D1x`/`D1y` rules for the `∂ψ/∂x, ∂ψ/∂y` components in the wind/slope
normal projections (all from `../earthscidiscretizations/grids/cartesian_uniform_2d`).
Both the model and the discretization are imported by path, not vendored. This is a
plain "does it run" setup: it loads, integrates while saving snapshots, and plots
the front at each time. No conformance gates or MMS error checks.

## Files

| Path | Role |
|------|------|
| `wildlandfire.esm` | Thin **assembler**: declares the grid metaparameters `NX,NY` (bound to 32×32 by the runners) and imports the `LevelSetFireSpread` model component by `ref`. No physics of its own. |
| `../earthscimodels/components/wildland_fire/level_set_fire_spread.esm` | The **pre-built model component** (not in this repo). Migrated here to v0.8.0 + bare-`D`; it connects its discretization by reference through its own `expression_template_imports` (Godunov + central `D1x/D1y`). Referenced, not vendored. |
| `run-model.jl` | Julia runner (`EarthSciSerialization.jl` + `OrdinaryDiffEqTsit5`; front plot via `Plots.jl`). Overrides `LevelSetFireSpread.R_0 = 0.1`. |
| `run-model-jl/` | Auto-generated local Julia project (develops ESS + adds the solver + `Plots.jl`). Gitignored; rebuilt on first run. |
| `run-model-rs/` | Rust runner (a cargo bin depending on the `earthsci-toolkit` crate by path; front plot via `plotters`). |
| `run-model-rs.sh` | Wrapper that builds+runs the Rust bin with the OpenSSL env it needs. |
| `run-model.py` | Python runner (the `earthsci_toolkit` binding + SciPy `LSODA`; front plot via `matplotlib`). Overrides `LevelSetFireSpread.R_0 = 0.1`. |
| `run-model-py.sh` | Wrapper that provisions a minimal venv (numpy/scipy/sympy/jsonschema/matplotlib) and runs `run-model.py`. |
| `run-model-py/` | Auto-generated local Python venv. Gitignored; rebuilt on first run. |

All three runners resolve `EarthSciSerialization` as a sibling checkout
(`../EarthSciSerialization`); set `ESS_ROOT` to override (Julia + Python).

## Julia

```bash
julia run-model.jl                 # defaults: ./wildlandfire.esm, t_end=2.0
julia run-model.jl <model.esm> <t_end>
```

## Rust

```bash
./run-model-rs.sh                  # defaults: ./wildlandfire.esm, t_end=2.0
./run-model-rs.sh <model.esm> <t_end>
```

The first Rust build compiles the whole toolkit + vendored `s2geometry` C++
kernel + `plotters` (~a few minutes). That kernel `#include`s `<openssl/bn.h>`;
on macOS the Homebrew OpenSSL headers/libs aren't on the default search path, so
`run-model-rs.sh` adds them via `CPATH` / `LIBRARY_PATH` from
`brew --prefix openssl@3`. To run `cargo` directly, export those yourself.

## Python

```bash
./run-model-py.sh                  # defaults: ./wildlandfire.esm, t_end=2.0
./run-model-py.sh <model.esm> <t_end>
```

The first run creates `run-model-py/.venv` and pip-installs the minimal
scientific stack (`numpy`/`scipy`/`sympy`/`jsonschema`/`matplotlib`); later runs
reuse it. The `earthsci_toolkit` binding is used straight from the sibling
`EarthSciSerialization` source (no install). To run `run-model.py` with your own
interpreter, put that stack on the path and ensure `earthsci_toolkit` is
importable (or let the script add `$ESS_ROOT/packages/earthsci_toolkit/src`).

## Plots

Each runner writes a PNG next to the model — `front_julia.png` (Julia `Plots.jl`
/ GR backend), `front_rust.png` (Rust `plotters`, bitmap backend), and
`front_python.png` (Python `matplotlib`, Agg backend). Each is a
**single figure** showing the front `{ψ = 0}` at `NT = 6` snapshot times
(`t = 0 … t_end`) as nested contours, with **time encoded by color** (the
validated dataviz sequential blue ramp, light = early → dark = late; a legend
maps color to `t`). The front is extracted from the `ψ` field by marching
squares in all three languages.

## Notes

- The front advances at the expected speed: the effective radius from the burned
  area (`ψ < 0`), `r ≈ √(area/π)`, tracks `0.15 + S·t`
  (`0.1537, 0.1866, 0.2258, 0.2639, 0.3054, 0.3419` at `t = 0, 0.4, …, 2.0`,
  `S=0.1`; the Godunov scheme runs slightly below the analytic line at late
  times on the coarse 32×32 grid).
- **All three backends — Julia, Rust, and Python — agree to four decimals on the
  front radius at every time**, from the SAME `.esm` (the whole point: one
  discretization-agnostic library leaf, mounted with the assembler's grid, runs
  identically across toolkits). Each runner plots the `ψ` state field; scalar /
  observed rows (e.g. `dx`, `dy`, the `[x,y]` observeds) are skipped.
- **Where the time goes (measured — see `profiles/` and "Profiling" below).**
  The two backends are bottlenecked in *opposite* places, and neither is limited
  by the Godunov math or by the ODE solver itself:
  - **Julia** pays at **build time.** ~99% of `simulate()` (~1.8 s at 32×32,
    after a one-time ~24 s JIT warm-up) is `build_evaluator` symbolically
    expanding the discretized stencil **per grid cell** — recursive
    `_sub_preserving` tree substitution — dominated by GC churn and type
    instantiation. The compiled RHS then integrates in a handful of profiler
    samples (negligible).
  - **Rust** pays at **solve time.** ~66% of `simulate()` (~14 s at 32×32,
    release-optimized) is allocation/clone/drop churn inside the array-expression
    *interpreter* (`ExpressionNode::clone` 14%, `Vec::to_vec` 13%,
    `drop_in_place` 9%, `alloc`/`push`/`grow` …), which rebuilds its working
    `Value` buffers on **every** RHS evaluation; ~8% more is HashMap name lookups
    and ~8% ndarray shape bookkeeping. The diffsol solver itself is ~0.5% and the
    float math (`eval_arith`/`powf`) ~1%.

  So Julia is fast at solving *because* it is slow at building (compile once,
  integrate for free); Rust never compiles the RHS, so it re-interprets — and
  re-allocates — the expression tree every step. The non-smooth Godunov
  Hamiltonian forces the adaptive solver into many small steps, which *multiplies*
  Rust's per-eval allocation cost — that, not the solver, is why finer grids /
  tighter tolerances blow up in Rust. At this config Rust's `simulate()` is ~8×
  Julia's (not the ~500× guessed earlier).
- **Build the Rust runner optimized.** `run-model-rs.sh` passes `--release`; the
  full runner then finishes in ~12 s vs ~58 s in debug (`cargo run` with no
  `--release`) — a ~4.7× penalty *on top of* the interpreter cost. The trade is a
  one-time ~2–4 min optimized compile of the toolkit; later runs reuse it.
- **Resolution `NX=NY=32`, `reltol 1e-2`** (bound to the grid metaparameters by
  both runners) keeps the Rust run tractable; marching-squares interpolation
  keeps the contours smooth despite the coarse grid.
- `S` is a model parameter (default `0.1`); `t_end` is a runner CLI arg. Keep
  `0.15 + S·t_end < 0.5` so the front stays clear of the periodic seam.

## Profiling

Two harnesses CPU-profile `simulate()` with third-party sampling profilers,
writing artifacts to `profiles/`:

| Command | Tool | Outputs (`profiles/`) |
|---------|------|-----------------------|
| `julia profile-jl.jl [t_end]` | [PProf.jl](https://github.com/JuliaPerf/PProf.jl) (Google `pprof`) over the stdlib sampler | `flamegraph-julia.svg`, `profile-julia-top.txt`, `profile-julia-{flat,tree}.txt` |
| `./profile-rs.sh [t_end]` | [pprof-rs](https://github.com/tikv/pprof-rs) in-process SIGPROF sampler | `flamegraph-rust.svg`, `profile-rust-top.txt` |

The Julia harness warms up once so JIT **compilation is not profiled** — only
steady-state runtime. The Rust harness builds a dedicated `profile` bin under a
`profiling` cargo profile (release optimization **+** debug symbols, so
`earthsci-toolkit`'s internal frames get named) enabled by `--features profile`;
the normal runner never compiles `pprof`. Both sample the same 32×32, `t_end=2.0`
config the runners use. See the findings above for what they reveal.
