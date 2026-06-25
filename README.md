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

## Validation

`simulations/camp_fire.esm` validates against `esm-schema.json` and all of its
coupling / domain / interface references resolve.
