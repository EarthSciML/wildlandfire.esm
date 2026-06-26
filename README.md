# wildlandfire.esm

Wildland-fire **simulations** expressed in the [EarthSciML Serialization
Format](https://github.com/EarthSciML/EarthSciSerialization) (`.esm`).

Where [EarthSciModels](https://github.com/EarthSciML/EarthSciModels) holds the
authoritative, runtime-agnostic `.esm` definitions of individual model
*components* (Rothermel spread, the level-set front PDE, fuel-model lookup,
data loaders, …), this repo holds the `.esm` files that **couple those
components into runnable simulations** of specific fires — the domain, the
ignition, the meteorology drivers, and the run window.

## Simulations

### `simulations/camp_fire.esm` — Camp Fire (2018)

A re-expression of the legacy `WildlandFire.jl` Camp Fire spread script
(Rothermel + level-set front + LANDFIRE fuels + USGS 3DEP terrain + ERA5 wind)
in the ESM format. The old script wired the components together imperatively in
Julia and never fully ran end-to-end; this file captures the same coupled
system declaratively, so it can be driven by any canonical ESS runner.

**Coupled components** (authoritative definitions live in EarthSciModels):

| Role | Component | EarthSciModels path |
|---|---|---|
| Surface fire spread (Rothermel 1972) | `RothermelFireSpread` | `components/wildland_fire/rothermel/fire_spread.esm` |
| Fire-front PDE (level set, 2-D) | `LevelSetFireSpread` | `components/wildland_fire/level_set_fire_spread.esm` |
| Fuel-bed properties from fuel code | `FuelModelLookup` | `components/wildland_fire/fuel_model_lookup.esm` |
| Slope steepness & aspect | `TerrainSlope` | `components/wildland_fire/terrain_slope.esm` |
| 10 m → midflame wind | `MidflameWind` | `components/wildland_fire/midflame_wind.esm` |
| Fuel burn-down feedback | `FuelConsumption` | `components/wildland_fire/level_set/fuel_consumption.esm` |
| Equilibrium moisture content | `EquilibriumMoistureContent` | `components/wildland_fire/nfdrs/emc.esm` |
| 1-hour dead fuel moisture | `OneHourFuelMoisture` | `components/wildland_fire/nfdrs/moisture_1h.esm` |
| Fuel model codes | `LANDFIRE` | `components/earthsci_data/landfire.esm` |
| Elevation + slope | `USGS3DEP` | `components/earthsci_data/usgs3dep.esm` |
| Wind / temperature / humidity | `ERA5` | `components/earthsci_data/era5.esm` |

**Coupling topology** (all routing is `param_to_var`; the level-set field gates
fuel consumption through a smooth Heaviside `couple` connector):

```
LANDFIRE.fuel_model ─► FuelModelLookup ─► (σ, w₀, δ, Mx, h) ─┐
USGS3DEP.dzdx/dzdy ─► TerrainSlope ─► tan_phi ───────────────┤
                                   └─► slope_aspect ─► MidflameWind
ERA5.u/v ───────────────────────────────► MidflameWind ─► U ─┤
ERA5.t/r ─► EquilibriumMoistureContent ─► OneHourFuelMoisture ─► Mf ─┤
                                                                    ▼
                                                        RothermelFireSpread
                                                           │ R, C/B/E, β_ratio, φ_s
                                                           ▼  (lifted pointwise)
USGS3DEP.dzdx/dzdy, MidflameWind.u_mf ───────►  LevelSetFireSpread (ψ, 2-D PDE)
                                                           │ ψ
                                                           ▼  0.5·(1−tanh(ψ/εₕ))
                                                        FuelConsumption
```

**Simulation configuration** (carried over verbatim from the original script):

- **Projection** — Lambert Conformal Conic (`lat_1=30, lat_2=60, lat_0=39,
  lon_0=-97`), units metres. Stored as each domain's `spatial_ref`.
- **Domain** — `camp_fire_surface`, a 19 × 21 cell grid at `dx = 2000 m`,
  centred between Pulga (ignition, −121.44, 39.81) and Paradise (−121.62,
  39.76). ERA5 meteorology is served on the 3-D pressure-level domain
  `camp_fire_3d` (`lev = 1:5`) and reduced to the surface through the
  `ground_surface` interface (`lev = min`, bilinear).
- **Ignition** — a signed-distance level set of radius 3000 m about the Pulga
  point, encoded as the `expression`-typed initial condition for `psi` on
  `camp_fire_surface`.
- **Run window** — `2018-11-08T14:30Z → 2018-11-09T06:30Z` (16 h). The original
  integrated the discretized Hamilton–Jacobi PDE with `SSPRK33` at
  `dt = dx/400 s`, saving every 600 s.

#### Resolution / fidelity notes

- Each `models` entry here is a **coupling-interface stub** — it declares only
  the variables that participate in the wiring, following the house convention
  of `EarthSciModels/couplings/*.esm`. The full equations stay single-sourced in
  the referenced component files; a runner resolves them when assembling the
  coupled system. This keeps the simulation spec a thin, reviewable diff and
  avoids duplicating authoritative model content.
- `tanh` is not an ESM AST op, so the level-set → fuel-consumption smooth
  Heaviside `0.5·(1 − tanh(ψ/εₕ))` is written in its algebraically identical
  `exp` form `1 / (1 + exp(2ψ/εₕ))`, with `εₕ = dx`.

## Validation & running

`simulations/camp_fire.esm` validates against `esm-schema.json` and all of its
coupling / domain / interface references resolve.

**0-D fire-behavior chain — verified runnable.** The point parameterizations
(`FuelModelLookup`, `TerrainSlope`, `MidflameWind`, `EquilibriumMoistureContent`,
`OneHourFuelMoisture`, `RothermelFireSpread`, `FuelConsumption`) run through the
canonical Python ESS runner (`earthsci_toolkit.simulation.simulate`) and produce
correct physics — e.g. for an Anderson FM1 grass bed under 3.4 m/s midflame wind
and a 22 % slope, Rothermel gives a rate of spread `R ≈ 1.08 m/s (65 m/min)`;
`FuelConsumption` burns a cell down to ~0 fuel over the run window.

**2-D level-set PDE core — now runs in Python too.** `LevelSetFireSpread` is a
spatial Hamilton–Jacobi PDE (`system_kind: "pde"`, independent variables
`t, x, y`). It runs end-to-end through the EarthSciSerialization Python pipeline:

```
flatten → flattened_to_esm → spatial_discretize(GDD) → simulate
```

where `spatial_discretize` (the generic, GDD-driven method-of-lines pass) lowers
the spatial operators to ArrayOp stencils via the EarthSciDiscretizations catalog
rules — including the **Godunov upwind `|∇ψ|`** rule needed for a stable level-set
front. No new AST op or spec change; the scheme is selected by the GDD.

Run it with [`simulations/run_camp_fire.py`](simulations/run_camp_fire.py), the
runnable successor to the legacy Julia script.

**Required EarthSciSerialization commit.** The runner imports the
`spatial_discretize` pipeline, and the live data-loader path (below) reads
through the `EARTHSCIDATADIR` cache seam — both are present as of
EarthSciSerialization commit
[`86b09d86`](https://github.com/EarthSciML/EarthSciSerialization/commit/86b09d86ec894bfb6661893b3d28af04c2e56df4)
(`feat(python): EARTHSCIDATADIR content-addressed cached opener/fetcher for the
loader seam`, on `main`). Check the toolkit out at that commit so the run is
reproducible:

```bash
# two checkouts: the pinned ESS toolkit and the EarthSciModels components
EARTHSCISERIALIZATION=…/EarthSciSerialization
EARTHSCIMODELS=…/EarthSciModels
git -C "$EARTHSCISERIALIZATION" checkout 86b09d86ec894bfb6661893b3d28af04c2e56df4

PYTHONPATH="$EARTHSCISERIALIZATION/packages/earthsci_toolkit/src" \
EARTHSCIMODELS="$EARTHSCIMODELS" \
python simulations/run_camp_fire.py
```

`EARTHSCIMODELS` defaults to a sibling `../EarthSciModels` checkout when unset;
`PYTHONPATH` must point at the pinned toolkit's `src`. The runner itself needs
no live network or data — the LANDFIRE / USGS 3DEP / ERA5 inputs are held at
constants (the full live path is the [data cache](#data-cache-for-the-live-data-loader-path-earthscidatadir)
below).

It drives the **real** EarthSciModels components and reports the fire-front
radius over time, for three configurations:

- *standalone* `LevelSetFireSpread` — an eikonal front advancing at `R_0` (≈1 m/s);
- *coupled* `RothermelFireSpread → LevelSetFireSpread` — the front driven by the
  Rothermel-computed rate of spread (≈0.71 m/s for Anderson FM1 grass under 3 m/s
  midflame wind and a mild slope), confirming the fire-behavior chain is wired in;
- *full fuel chain* `FuelModelLookup → RothermelFireSpread → LevelSetFireSpread` —
  a LANDFIRE fuel code (1 = Anderson FM1) is looked up in `FuelModelLookup`'s
  Anderson (1982) tables (an `ifelse` over `fn: interp.linear` on the integer
  fuel-code axis — *not* a registered handler) to produce the fuel-bed
  properties (σ, w₀, δ, Mx, h) that Rothermel needs. The looked-up bed
  reproduces the coupled run's front exactly, confirming the real fuel lookup
  drives the spread.

Data-driven inputs (LANDFIRE fuel codes, USGS 3DEP terrain, ERA5 wind) are held
at constants in the runner; the **full** data-loader path still needs live
LANDFIRE / USGS 3DEP / ERA5 data at runtime (and the Julia toolchain remains the
reference for high-resolution / curvilinear runs).

### Data cache for the live data-loader path (`EARTHSCIDATADIR`)

That live data-loader path reads its inputs through EarthSciSerialization's
content-addressed disk cache (added in the pinned commit). A populated cache
lets the loaders run offline and keeps large NetCDF / GeoTIFF pulls off
inode-quota'd home filesystems. It is configured entirely by environment:

| Env var | Effect |
|---|---|
| `EARTHSCIDATADIR` | Cache root. Resolution order is explicit `data_dir=` arg → `$EARTHSCIDATADIR` → a temp dir (`$TMPDIR/earthsci-cache`). Point it at scratch — e.g. `EARTHSCIDATADIR=/scratch.local/$USER/earthsci-cache` — for Camp Fire-sized data. |
| `EARTHSCI_OFFLINE` | Set truthy (`1`/`true`/`yes`/`on`) to force cache-only mode: a miss raises `CacheMiss` instead of hitting the network. Leave unset to fetch-and-cache on miss. |

Cache files are keyed on `sha256(resolved_url)` (laid out
`<root>/<aa>/<sha256><suffix>`), so a cache populated once is reused across
mirrors and runs. The mechanism wraps the loaders' existing dependency-injection
seam — `GridLoader` / `StaticLoader` take `opener(url) -> Dataset`, `PointsLoader`
takes `fetcher(url) -> bytes` — via `earthsci_toolkit.cached_opener` /
`cached_fetcher`; wiring is additive, so loaders default to the uncached path
unless a caller opts in.

Acquiring the Camp Fire window into the cache (the real ERA5 / LANDFIRE / USGS
3DEP pulls over the run window) is wf data-engineering work tracked separately;
**ERA5 additionally needs a Copernicus CDS key in `~/.cdsapirc`**.

This Python PDE capability was added across EarthSciSerialization (the rule
engine, `spatial_discretize`, coupling-flatten, canonicalization, and the single
`discretize(esm, gdd=…)` entry) and EarthSciDiscretizations (the Godunov catalog
rule).

## Validating against the observed fire (`simulations/validation/`)

`run_camp_fire.py` produces the *simulated* fire progression — a signed
level-set field `psi(t, y, x)` whose `psi ≤ 0` region is the burned area and
whose `psi = 0` contour is the fire front. The
[`simulations.validation`](simulations/validation) harness scores that
progression against the **observed** 2018 Camp Fire and emits a report. It is a
*soft oracle*: observed-versus-simulated wildfire agreement is a research
comparison (the model "never fully ran" end-to-end), so its thresholds colour
each metric `OK`/`WARN` to guide the eye rather than asserting a bit-exact pass —
the numbers themselves are always reported verbatim.

Three metric families (campaign bead **E2**):

| Family | Simulated quantity | Observed reference (campaign **E1**) | Headline metrics |
|---|---|---|---|
| **Burned-area agreement** | final `psi ≤ 0` footprint | MTBS severity + last perimeter | area ratio, IoU, Dice, precision / recall |
| **Perimeter over time / spread rate** | `psi ≤ 0` area at each observed time | NIFC / GeoMAC daily perimeters | area-vs-time RMSE, mean spread rate, per-time IoU |
| **Ignition match** | burned-region centroid + window start | VIIRS / MODIS first detection | ignition distance (km), time offset (h) |

Try it on built-in synthetic data (an expanding-disk fire — no external files):

```bash
python -m simulations.validation --demo --markdown report.md --json report.json
```

Wire a real run + observed reference (the end-to-end **E3** path):

```python
from simulations.validation import LevelSetRun, ObservedReference, validate

# `result` is the earthsci_toolkit simulate() output from run_camp_fire.py
run = LevelSetRun.from_simulate_result(result, dx=2000.0, t0=window_start)
ref = ObservedReference.from_netcdf("camp_fire_observed.nc")  # rasterised E1 loaders
validate(run, ref).write(json_path="report.json", markdown_path="report.md")
```

The observed reference is whatever the E1 observed-fire loaders rasterise to a
`burned_fraction(time, y, x)` grid (perimeters / MTBS / active-fire are
rasterised offline so each reads as an ordinary `kind:grid` loader). The harness
nearest-neighbour-resamples that grid onto the run grid, so the two need not
share a resolution. The data model is pure numpy; `netCDF4` is imported only when
reading a reference from disk.

Run the harness tests with `pytest` (from the repo root).

## End-to-end DRAIN run (`simulations/run_camp_fire_e2e.py`, campaign bead **E3**)

The campaign capstone: one driver that assembles the **full 11-component** model
over its **real** domain/window, drives the fire physics with the **data
loaders**, integrates a fire progression, and scores it with the **E2** harness —
emitting a validation report.

```bash
PYTHONPATH=.../earthsci_toolkit/src EARTHSCIMODELS=.../EarthSciModels \
    python simulations/run_camp_fire_e2e.py --outdir campfire_e2e_out
# writes camp_fire_run.npz + camp_fire_validation.{md,json}; --observed REF.nc
# scores against a real rasterised E1 reference instead of the documented-facts one.
```

**Scope — what runs in Python, what is deferred.** Per the campaign decision, the
Python end-to-end path is **0-D fire-behavior chains + assembly + the
loader/regrid path**; the full 2-D level-set Hamilton–Jacobi PDE *core* is the
deferred **Julia** campaign (Python `simulate()` raises
`UnsupportedDimensionalityError` on a multi-D spatial system). The driver's five
stages:

1. **Assemble** — `et.load` resolves the 11 `{ref}` components (by-name model-ref
   resolver) and `et.flatten` couples them into one non-empty system over the real
   Camp Fire domain (19×21 @ 2 km, LCC) and window (62 equations, 40 states, 20
   loader fields, all 11 source systems).
2. **Loader/regrid path** — `build_target_grid` from the flattened domain, then
   each data source (LANDFIRE fuel, USGS 3DEP slope, ERA5 wind/T/RH) is regridded
   onto the real grid (reproject + bspline + `lev=min`). Cached real data is used
   when `EARTHSCIDATADIR` holds it; otherwise physically representative Camp Fire
   fields stand in (the live pull is a documented blocker — below).
3. **0-D fire-behavior chain** — per cell, the *real* components
   (`FuelModelLookup → TerrainSlope → MidflameWind → EMC → 1-h moisture →
   Rothermel`) run on the regridded inputs to give the rate of spread `R(x, y)`.
   This is the loader-driven physics integration.
4. **Fire progression** — a kinematic anisotropic minimum-travel-time front
   (Dijkstra with the wind-driven elliptical fire shape; Anderson 1983 +
   Finney 2002 MTT) from the Pulga ignition gives `psi(t, y, x)` — a post-process
   of the 0-D `R` field, *not* the deferred PDE core.
5. **E2 validation** — the progression is scored against the observed reference
   and a Markdown + JSON report is written.

**Result (representative run).** The simulated front spreads from the documented
Pulga ignition (ignition distance ≈ 0.3 km) SW toward Paradise, reaching ~30 km
and ~240 km² over the 16 h window — consistent with the documented Camp Fire
first-day extent (~280 km², ~25–30 km run), with heading rates up to ~300 m/min.
All physical-sanity checks pass; the soft-oracle overlap metrics are moderate
against the coarse documented-facts reference (a `WARN` is advisory, not a
failure).

**Blockers / deferrals** (the driver prints these):

1. **Full 2-D level-set PDE core → Julia.** Python `simulate()` rejects the
   multi-D spatial system, so the coupled Hamilton–Jacobi front (curvature +
   fuel-consumption feedback) is integrated in Julia; Python runs the chains +
   assembly + loader/regrid + the kinematic MTT progression.
2. **Live ERA5 / LANDFIRE / USGS 3DEP → not acquired (user-gated).** The loader
   seam (cache, regrid, injection) is exercised on the real grid with
   representative fields: `EARTHSCIDATADIR` is empty, `cdsapi` is not installed,
   and the default loader URLs are placeholders. Populating the cache over the run
   window is data-engineering work (ERA5 needs a CDS key).
3. **Observed reference (E1) → not acquired.** The MTBS/NIFC/VIIRS loaders landed,
   but the rasterised Camp Fire reference needs a FIRMS key + perimeter download;
   until then the harness scores against a documented-facts stand-in. Pass
   `--observed REF.nc` to use the real rasterisation.
