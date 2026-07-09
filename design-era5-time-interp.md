# ERA5 time interpolation — design & implementation contract

> **STATUS: IMPLEMENTED & VERIFIED (Python, Rust, Julia).** The provider loads the 2
> bracketing records; the `.esm` interpolates via `ERA5.w_time = frac((t - t_interp_ref)/
> dt_interp)` (fixed cadence params, see DECISION UPDATE below); `Era5Regrid` regrids both
> records and blends on the fire grid. All three runners produce the identical t=0 bracket
> `[286.13, 285.76] K` and `r_eff = 3191.5 m`; Python numerically proves
> `t_xy == (1-w)·t_xy0 + w·t_xy1` to 4 dp, continuous across hour boundaries.


## Goal

Replace the ERA5 met's **piecewise-constant, hourly-step** forcing with **linear-in-time
interpolation** between the two ERA5 records that bracket the current simulation time.
Per the user's decisions:

- **(1b)** The provider *injects the two bracket timestamps* (general, no cadence-param
  syncing). The model computes `w = clamp((t - t0)/(t1 - t0), 0, 1)`.
- **(2)** **Regrid-both-then-blend**: regrid the two bracket records at discrete (hourly)
  cadence, blend the two *fire-grid* fields with `w`. Valid because conservative regrid is
  linear, so `regrid(blend) == blend(regrid)`; keeps the expensive overlap regrid hourly and
  makes only a cheap per-cell AXPY continuous.
- **(3)** Handle the **cross-file successor** (the "after" record may live in the next file).
- **(4)** **End/start clamping** (degenerate bracket at the data edges).
- **(6)** **Promote into the shared components** (`era5.esm` / `era5_loader.esm`) as the
  general mechanism for *any* discrete loader — "we will usually want time interpolation for
  discrete data loaders."
- **(7)** All three languages: **Julia, Rust, Python**.

## The one place the decisions compose non-trivially

(1b/6) put the interpolation *in the shared component*; (2) needs the blend *after* the
regrid, which lives in wildlandfire-local `Era5Regrid`. Resolution:

- The shared **`era5.esm`** owns the interpolation *brain*: it computes the weight
  `ERA5.w_time` from `t` and the injected bracket timestamps, and provides **ready blended
  outputs** (`ERA5.t`, `ERA5.u`, …) for simple consumers (and MeanWind).
- It also **passes the two raw bracket records through** (`ERA5.pl.<var>`, now a size-2
  time axis) so a consumer behind a linear operator can defer the blend.
- **`Era5Regrid`** (wildlandfire) takes the raw 2-record fields **+ `ERA5.w_time`**, regrids
  each record (discrete/cached), and blends on the fire grid (continuous/cheap).

So the interpolation *logic* is shared; wildlandfire only relocates *where the linear blend
is evaluated*, for speed.

---

## Layer 1 — EarthSciIO provider: "bracket" mode

**Trigger:** a DISCRETE loader whose `temporal` carries `records_per_sample: 2` (new,
optional key; absence ⇒ today's single-record floor behavior, unchanged).

**Contract change** (`refresh(t)` in all three impls):
- Return, per variable, the two bracketing records `[rec, rec+1]` as a **size-2 leading
  time axis** (do *not* drop `time_dim`). This mirrors the shape CONST already returns.
- **Retain the 2-element `time` coordinate** (the bracket timestamps, native units/attrs).
- `rec` = floor (at-or-before), as today; `rec+1` = successor.
- **Cross-file (3):** if `rec+1` is past the current file's last record, open the successor
  file and take its record 0. Providers must be able to hold/join two files at the seam.
- **End clamp (4):** at the last available record (no successor) return a **degenerate
  bracket** `[last, last]` with equal timestamps; the model's `w` clamps and the blend is a
  no-op (holds the endpoint). Before the first record, clamp to `[first, first]`.

Change loci:
- Python `earthsciio/provider.py`: `_record_index` (422), `_slice_record`/`_slice_field`
  (342–370), `refresh` (224). Add a `bracket` path; keep single-record path intact.
- Rust `rust/src/provider.rs`: rec/slice loop (279–298); a "select two leading records"
  helper beside `select_leading` (`format/mod.rs:134`).
- Julia `julia/src/provider.jl`: `_tick_index` (127) → bracket helper; `_slice_dim` (140)
  → keep `idx:idx+1`.

Tests to update/add: `tests/test_provider.py`, `rust/tests/provider_cadence.rs`,
`julia/test/test_provider.jl` (they currently pin "time axis dropped, shape [3,3]").

## DECISION UPDATE (supersedes the 1b timestamp-injection in Layers 2–4 below)

Julia and Rust build a **fixed parameter vector once** and only live-mutate a *forcing
buffer* between cadence segments (they do NOT rebuild the RHS per segment like Python), so
per-segment *provider-injected timestamps* (1b) don't port cleanly. Per the user's decision we
use **fixed cadence params** instead:

- The provider still **loads the two bracketing records** (Layer 1, unchanged).
- The shared `era5.esm` derives the weight from a fixed cadence, not injected timestamps:
  **`w_time = frac((t - t_interp_ref) / dt_interp)`** where `t_interp_ref` (a cadence anchor in
  sim-clock seconds) and `dt_interp` (the loader frequency, 3600 s) are **runner-set scalar
  params** (like the src-geometry params). No timestamps, no per-segment injection.
- **The ESS adapter needs NO changes** in any language: the size-2 loader field binds through
  the existing loader-field → forcing-buffer path by element count. (The Python per-segment
  injection I first wrote was reverted.)
- Byte-identical to injected-timestamps for ERA5's uniform hourly cadence. Verified in Python:
  `t_xy(model) == (1-w)·t_xy0 + w·t_xy1` to 4 dp at every step, continuous across hour
  boundaries.
- Caveat: `frac` wraps to 0 at an exact anchor, so if `t_end` lands precisely on a cadence
  anchor the terminal instant uses the earlier bracket record (one derivative eval; physically
  negligible; the front trajectory is unaffected).

The three per-language sub-sections below are the ORIGINAL 1b plan, kept for context but
NOT implemented — read them as "what we would have done for injected timestamps."

## Layer 2 (NOT USED) — ESS/EarthSciAST adapter: expose 2-record field + inject bracket timestamps

For a loader in interpolate mode, the runtime, at each discrete refresh, must:
1. Bind each variable's **size-2** native array into the consumer parameter (element-count
   match; consumer declares a leading size-2 time index set).
2. Expose the loader's **`time_variable` as a consumable coordinate field** holding the two
   bracket timestamps **converted to solver seconds** (native-units → seconds via the run
   epoch, which the adapter already owns). Reference name: `<Loader>.<time_variable>`
   (e.g. `ERA5.pl.valid_time`), a length-2 vector `[t0, t1]`.

This is the general "provider-injected timestamps" machinery. Loci:
- Python `data_loaders/esio_provider.py` + `simulation_loaders.py` (`_provider_refresh_field`
  645, `_refresh_discrete` 770).
- Julia `data_refresh.jl` (`_write_forcing!` 135, `build_refresh_callback` 198).
- Rust `provider.rs` `RefreshExecutor` (330–490).

Note: `ERA5.w_time` references `t` ⇒ classified **continuous** by every partitioner (Rust
`cadence.rs:112`, Julia `cadence.jl:148`/`build.jl:901`, Python `simulation_array.py:607`).
The two per-record regrids do **not** reference `t` (only the size-2 loader field, which is
DISCRETE) ⇒ they stay cached hourly. Only the final blend goes per-step. This split falls out
of the cadence analysis automatically.

## Layer 3 — shared `era5.esm` / `era5_loader.esm`

`era5_loader.esm`: add `"interpolation": "linear"` to `data_loaders.ERA5_pl.temporal`.

`era5.esm` (the ERA5 model component):
- New parameter-ish inputs: the bracket timestamps `t0`,`t1` come from the loader subsystem
  (`ERA5.pl.valid_time`, length 2).
- New observed **`w_time = clamp((t - t0)/(t1 - t0), 0, 1)`** with a guard for `t1==t0`
  (degenerate bracket ⇒ `w_time = 0`).
- New convenience interpolated observeds `t`,`u`,`v`,`w`(omega),`r`,… (top level) =
  `(1-w_time)*pl.<var>[1] + w_time*pl.<var>[2]` for simple consumers.
- **Repoint MeanWind coupling** `v_lon ~ ERA5.u`, `v_lat ~ ERA5.v`, `v_lev ~ ERA5.w`
  (interpolated) instead of `ERA5.pl.u/v/w` (now 2-record). Update the coupling in
  `couplings/earthscidata_spatial_params.esm` if it references the raw fields.

## Layer 4 — wildlandfire `Era5Regrid` (regrid-both-then-blend)

- `index_sets.era5_t`: size **2** (was 1, vestigial).
- `F_t/F_u/F_v/F_r`: shape → `["era5_t","era5_lev","era5_y","era5_x"]`.
- New param `w_time` (coupled from `ERA5.w_time`).
- Split each reindex/regrid into record-0 and record-1 streams:
  `F_t_h0/F_t_h1` (index time axis 1 / 2) → `t_regridded0/1` → `t_xy0/1` (all DISCRETE).
- Final `t_xy = (1-w_time)*t_xy0 + w_time*t_xy1` (CONTINUOUS AXPY over `[x,y]`), likewise
  `u_xy/v_xy/rh_xy`. Downstream physics unchanged.

## Layer 5 — runners (jl / py / rs)

- Build the ERA5 providers in **bracket/interpolate mode**.
- Couple `ERA5.w_time → Era5Regrid.w_time` (or rely on the shared coupling once promoted).
- Update the "mean T at t=0 vs t_end" sanity print to also sample mid-hour and show the
  value now lands *between* the two records (continuity check).

## Verification

- Per-language provider tests (bracket shape, timestamps, cross-file seam, end clamp).
- End-to-end: sample `Era5Regrid.t_xy` mid-hour; assert it is strictly between the two hourly
  records (was equal to the floor record before). Cross-check Julia↔Rust front `r_eff` still
  agrees to the digit.
