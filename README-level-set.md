# Running the coupled 2-D wildland-fire model (`wildlandfire.esm`)

A minimal, three-language harness for running a **coupled** 2-D EarthSci wildland-
fire-spread model, assembled **entirely by reference**. The spatial core is the
**level-set Hamilton-Jacobi** fire front `ψ_t = -S(n̂) |∇ψ|` (Mandel 2011 /
Muñoz-Esparza 2018) on a doubly-periodic **36 km × 40 km** box (the Camp-Fire domain
from `archive/camp_fire_model.jl`: `dx = dy = 2000 m`, so the runners bind
`NX = 18`, `NY = 20`), with `|∇ψ| = √((∂ψ/∂x)² + (∂ψ/∂y)²)` discretized by the
**Godunov** (Rouy-Tourin / Osher-Sethian) first-order upwind scheme. `ψ` is
initialized as the signed distance to a **3000 m** ignition circle at the domain
center `(18000, 20000) m`; the runners integrate **16 h** (`t_end = 57600 s`).

The front's direction-dependent spread rate `S(n̂) = R_0·(1 + φ_W + φ_S)` is **not
hand-set** — it is supplied by a **Rothermel (1972) + NFDRS behavior stack** wired
in by a top-level `coupling` block:

- **`FuelModelLookup`** (Anderson 13 fuel models → σ, w₀, δ, Mₓ, h) and
  **`EquilibriumMoistureContent` → `OneHourFuelMoisture`** (T, RH → dead-fuel
  moisture `Mf`) feed **`RothermelFireSpread`**, which yields the base rate `R_0`
  and the fuel-specific wind/slope coefficients `C_wind, B_wind, E_wind,
  beta_ratio, phi_s_coeff` for the leaf;
- **`MidflameWind`** (10 m wind → midflame `U`, `u_mf_x/y`) and **`TerrainSlope`**
  (dz/dx, dz/dy → tan φ, aspect) feed both Rothermel and the leaf's normal-
  projection wind/slope terms; **`FuelConsumption`** (Mandel 2011 Eq. 3) is mounted
  and receives the initial fuel load.

**Uniform forcing for now.** The behavior components are scalar/pointwise, so fuel,
wind, moisture and slope are **constant over the grid** — the level set itself is
fully spatial, but the forcing is uniform. Per-cell spatial forcing (the LANDFIRE /
USGS3DEP / ERA5 `[x,y]` data loaders) is a planned next step. See
`audit-coupling-expansion.md` for the coupling audit and the upstream 0.7.0→0.8.0
migration this required. This is a "does it run" setup — it loads, integrates while
saving snapshots, and plots the front at each time; no conformance / MMS checks.

## Files

| Path | Role |
|------|------|
| `wildlandfire.esm` | The **coupled assembler**: declares `NX,NY` (bound to 18×20), imports 8 components by `ref` (`LevelSetFireSpread` + Rothermel/NFDRS behavior stack), carries the `InitialPerimeter` IC model, and the top-level `coupling` block (21 `variable_map`/`param_to_var` edges) wiring fuel/moisture/wind/slope → Rothermel → the level-set leaf. No physics of its own. |
| `../earthscimodels/components/wildland_fire/…` | The **referenced components** (not in this repo; migrated to esm 0.8.0): `level_set_fire_spread` (the PDE leaf, with its own Godunov + central `D1x/D1y` discretization imports), `rothermel/fire_spread`, `fuel_model_lookup`, `nfdrs/emc`, `nfdrs/moisture_1h`, `midflame_wind`, `terrain_slope`, `level_set/fuel_consumption`. Referenced, not vendored. |
| `run-model.jl` | Julia runner (`EarthSciAST.jl` + `OrdinaryDiffEqTsit5`; front plot via `Plots.jl`). Supplies the extent `Lx,Ly` + scalar forcing (fuel code, T, RH, 10 m wind, terrain gradient); `R_0` comes from Rothermel. |
| `run-model-jl/` | Auto-generated local Julia project (develops ESS + adds the solver + `Plots.jl`). Gitignored; rebuilt on first run. |
| `run-model-rs/` | Rust runner (a cargo bin depending on the `earthsci-ast` crate by path; front plot via `plotters`). |
| `run-model-rs.sh` | Wrapper that builds+runs the Rust bin with the OpenSSL env it needs. |
| `run-model.py` | Python runner (the `earthsci_ast` binding + SciPy `LSODA`; front plot via `matplotlib`). Supplies the extent `Lx,Ly` + scalar forcing (fuel code, T, RH, 10 m wind, terrain gradient); `R_0` comes from Rothermel. |
| `run-model-py.sh` | Wrapper that provisions a minimal venv (numpy/scipy/sympy/jsonschema/matplotlib) and runs `run-model.py`. |
| `run-model-py/` | Auto-generated local Python venv. Gitignored; rebuilt on first run. |

All three runners resolve `EarthSciAST` as a sibling checkout
(`../EarthSciAST`); set `ESS_ROOT` to override (Julia + Python).

## Julia

```bash
julia run-model.jl                 # defaults: ./wildlandfire.esm, t_end=57600 (16 h)
julia run-model.jl <model.esm> <t_end>
```

## Rust

```bash
./run-model-rs.sh                  # defaults: ./wildlandfire.esm, t_end=57600 (16 h)
./run-model-rs.sh <model.esm> <t_end>
```

The first Rust build compiles the whole toolkit + vendored `s2geometry` C++
kernel + `plotters` (~a few minutes). That kernel `#include`s `<openssl/bn.h>`;
on macOS the Homebrew OpenSSL headers/libs aren't on the default search path, so
`run-model-rs.sh` adds them via `CPATH` / `LIBRARY_PATH` from
`brew --prefix openssl@3`. To run `cargo` directly, export those yourself.

> **Note — the Rust runner integrates the full coupled model** (`Erk` solver,
> Tsitouras 5(4)). `run-model-rs/src/main.rs` carries the coupled wiring (Camp-Fire
> domain + scalar forcing, `R_0` from Rothermel) and runs the whole 16 h, agreeing
> with the Julia `Tsit5` run to the digit (`r_eff` → ~8292 m, x-centroid → ~21741 m
> at 16 h — same tableau). This once errored at `t = 0` because the Rust simulate
> interpreters had no **`fn` operator** arm, so `FuelModelLookup`'s `interp.linear`
> fuel-table lookups returned `NaN` and poisoned Rothermel → `R_0` → `D(psi)`. That
> gap is now fixed upstream (`earthsci-ast-rs`: `fn` routes to
> `registered_functions` in both interpreters). See `audit-coupling-expansion.md` §6.

## Python

```bash
./run-model-py.sh                  # defaults: ./wildlandfire.esm, t_end=57600 (16 h)
./run-model-py.sh <model.esm> <t_end>
```

The first run creates `run-model-py/.venv` and pip-installs the minimal
scientific stack (`numpy`/`scipy`/`sympy`/`jsonschema`/`matplotlib`); later runs
reuse it. The `earthsci_ast` binding is used straight from the sibling
`EarthSciAST` source (no install). To run `run-model.py` with your own
interpreter, put that stack on the path and ensure `earthsci_ast` is
importable (or let the script add `$ESS_ROOT/pkg/earthsci-ast-py/src`).

## Plots

Each runner writes a PNG next to the model — `front_julia.png` (Julia `Plots.jl`
/ GR backend), `front_python.png` (Python `matplotlib`, Agg backend), and
`front_rust.png` (Rust `plotters`). Each is a **single figure** showing the front `{ψ = 0}` at
`NT = 6` snapshot times (`t = 0 … t_end`) as nested contours, with **time encoded by
color** (the validated dataviz sequential blue ramp, light = early → dark = late; a
legend maps color to `t`). The front is extracted from the `ψ` field by marching
squares.

## Notes

- **Coupled behavior.** With the default forcing (Anderson **FM4** chaparral,
  `RH = 0.2`, `T = 290 K`, `U = 1 m/s` east wind) `RothermelFireSpread` produces a
  no-wind base rate `R_0 = 0.0684 m/s` and a with-wind rate `R = 0.22 m/s`; the
  dead-fuel moisture is a dry `MC1 = 0.046`. The spatial front spreads and is
  clearly biased **downwind (east)** — the burned-area x-centroid drifts from the
  ignition `x = 18000` to ≈ 22000 m over 16 h. Change `FUEL_CODE`, `RH`, `U_WIND`,
  `DZDX/DZDY` (etc.) at the top of the runners to drive it. Keep the wind modest so
  the wind-amplified downwind front stays clear of the **periodic** seam over 16 h
  (a stronger wind wraps the domain — an artifact of the periodic BCs, not the
  physics).
- **Coarse grid.** The **18 × 20** grid (`dx = dy = 2000 m`, the archive's
  operational resolution) is very coarse relative to the front, so the burned-area
  radius `r ≈ √(area/π)` is heavily discretized (a 3 km ignition circle is only
  ~1.5 cells in radius). Marching-squares interpolation keeps the plotted contours
  smooth.
- **Both backends run the SAME coupled `.esm`.** Julia (Tsit5) and Python (LSODA)
  each load the 9-model / 21-edge assembly and integrate it. On the *smooth* calm
  base case they agree to four decimals; with the *non-smooth* wind term at the
  loose `reltol = 1e-2` the two solvers diverge modestly (e.g. r_eff 8.3 km vs
  9.0 km at 16 h) — expected for different integrators on a non-smooth Hamiltonian,
  not a discrepancy in the model. Each runner plots the `ψ` state field; scalar /
  observed rows (`dx`, Rothermel outputs, the `[x,y]` observeds) are skipped.
- **Where the time goes (measured — see `profiles/` and "Profiling" below;
  re-baselined to the coupled model with the runner's Camp-Fire forcing).** In all
  three bindings the cost is the **RHS / observed-expression evaluation**, never the
  Godunov math and never the ODE solver (the solver is a rounding error in every
  profile). They split into *compile-once* vs *re-interpret-every-step*:
  - **Julia — compile once, at build time.** After a one-time ~20 s JIT warm-up,
    `simulate()` (~5 s at 32×32 / 16 h) is dominated by allocation + GC
    (`jl_gc_small_alloc_inner`, `gc_mark_*`/`gc_sweep_*` ≈ 25 % cumulative) and
    type/symbol lookups (`typekeyvalue_hash`, `symtab_lookup`,
    `lookup_type_setvalue`) — `build_evaluator` symbolically expanding the
    discretized stencil per grid cell. The compiled RHS then integrates almost for
    free. *(Profile single-threaded — `julia --threads=1` — or idle worker/GC
    threads park in `__psynch_cvwait` and swamp the sampler; the script warns.)*
  - **Rust — re-interpret, in compiled code.** `simulate()` (~2 s at 32×32 / 16 h,
    release) never compiles the RHS; it walks the array-expression tree every step.
    Time goes to the interpreter itself: `eval_broadcast` (≈ 24 %), HashMap
    name-hashing (`Hasher::write_u8`/`write_str`, `make_hash` ≈ 30 % combined), and
    `Vec`/`Value` alloc-drop churn rebuilding working buffers per eval. diffsol's
    solve is ≈ 0.5 %.
  - **Python — re-interpret, in pure Python.** `simulate()` (LSODA) is orders of
    magnitude slower; over a bounded 120 s window (32×32) it is ≈ 98 % the recursive
    numpy interpreter — `eval_expr` dispatch (≈ 37 %), `_eval_index` stencil gathers
    (≈ 14 %), and per-node `isinstance` type checks (≈ 14 %). The SciPy solve is
    negligible.

  So Julia is fast at solving *because* it is slow at building (compile once,
  integrate for free); Rust and Python never compile the RHS, so they re-interpret
  the expression tree every step — Rust in optimized code (hashing + alloc bound),
  Python in pure-Python recursion (dispatch + `isinstance` bound). The non-smooth
  Godunov Hamiltonian forces the adaptive solver into many small steps, which
  *multiplies* that per-eval cost for the re-interpreters. Wall times are **not**
  cross-comparable here — each binding profiles its own solver (Tsit5 / Erk / LSODA)
  and Python samples a shorter window — so read each profile for *where its own time
  goes*, not as a language speed race.
- **Build the Rust runner optimized.** `run-model-rs.sh` passes `--release`; the
  full runner then finishes in ~12 s vs ~58 s in debug (`cargo run` with no
  `--release`) — a ~4.7× penalty *on top of* the interpreter cost. The trade is a
  one-time ~2–4 min optimized compile of the toolkit; later runs reuse it.
- **Resolution `NX=18, NY=20` (dx = dy = 2000 m), `reltol 1e-2`** (bound to the
  grid metaparameters by both runners) matches the archive's operational grid and
  keeps the Rust run tractable; marching-squares interpolation keeps the contours
  smooth despite the coarse grid.
- `t_end` is a runner CLI arg (default 16 h). The spread rate is no longer a hand-
  set constant — it comes from the coupled Rothermel stack — so to keep the front
  clear of the periodic seam at the 18 km half-width, moderate the forcing (fuel
  model, dryness, wind) rather than a single `R_0`. The default FM4 / RH 0.2 /
  1 m/s config reaches ≈ 9 km at 16 h.

## Profiling

Three harnesses CPU-profile `simulate()` with **statistical sampling** profilers,
each loading the coupled model and applying the **same Camp-Fire forcing the runners
use** (so the profile reflects the real coupled workload, not the default unit box),
writing artifacts to `profiles/`:

| Command | Tool | Outputs (`profiles/`) |
|---------|------|-----------------------|
| `julia --threads=1 profile-jl.jl [t_end]` | [PProf.jl](https://github.com/JuliaPerf/PProf.jl) (Google `pprof`) over the stdlib sampler | `flamegraph-julia.svg`, `profile-julia-top.txt`, `profile-julia-{flat,tree}.txt` |
| `./profile-rs.sh [t_end]` | [pprof-rs](https://github.com/tikv/pprof-rs) in-process SIGPROF sampler | `flamegraph-rust.svg`, `profile-rust-top.txt` |
| `./profile-py.sh [t_end]` | [pyinstrument](https://github.com/joerick/pyinstrument) in-process signal-timer sampler | `profile-python-top.txt`, `profile-python-tree.txt`, `profile-python.speedscope.json` (flame graph — open at [speedscope.app](https://speedscope.app)) |

- **Julia** warms up once so JIT **compilation is not profiled** — only steady-state
  runtime — then samples `simulate()` at 32×32, `t_end=57600`. **Run single-threaded**
  (`--threads=1`): the sampler snapshots every thread, so idle worker/GC threads
  otherwise park in `__psynch_cvwait` and dominate the profile (the script warns if
  `nthreads > 1`).
- **Rust** builds a dedicated `profile` bin under a `profiling` cargo profile (release
  optimization **+** debug symbols, so `earthsci-ast`'s internal frames get named)
  enabled by `--features profile`; the normal runner never compiles `pprof`. Samples
  32×32, `t_end=57600`.
- **Python** uses `pyinstrument` — a sampling profiler (like the other two), **not**
  `cProfile`: cProfile's per-call instrumentation multiplies this interpreter-heavy,
  many-tiny-step LSODA run ~30× (the full 16 h took > 30 min under it). pyinstrument
  needs **no root** (unlike `py-spy`, which on macOS requires `sudo` regardless of how
  it is installed — pip or homebrew — because it reads another process's memory). The
  Python run samples 32×32 over a **bounded `t_end=120 s`** window: Python's per-cell
  interpreter + implicit LSODA is orders of magnitude slower than the compiled
  explicit solvers, so the full 16 h is impractical under any profiler; the sampled
  function distribution is window-independent (every step exercises the same code).
  Prefer a native-speed `py-spy` profile instead? Run, with your password:
  `sudo run-model-py/.venv/bin/py-spy record -o profiles/flamegraph-python.svg -- run-model-py/.venv/bin/python profile-py.py`.

See the findings above for what they reveal.
