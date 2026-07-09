//! run-model — runner for the COUPLED 2-D EarthSci wildland-fire `.esm` (Rust).
//!
//! The Rust counterpart of run-model.jl / run-model.py. Same model, same `.esm`,
//! same answer: a 2-D level-set Hamilton-Jacobi front  psi_t = -S(n̂)|grad psi|
//! on the doubly-periodic 36 km × 40 km Camp-Fire domain (dx = dy = 2000 m;
//! NX=18, NY=20), |grad psi| by the Godunov (Rouy-Tourin / Osher-Sethian) upwind
//! scheme. The spread rate S is supplied by a coupled Rothermel + NFDRS behavior
//! stack (imported by reference; wired by the assembler's `coupling` block). The
//! spread is driven by REAL terrain: the mounted USGS3DEP data loader fetches a
//! Camp-Fire elevation raster at runtime (live USGS ImageServer, no auth), which
//! the in-model TerrainRegrid conservatively regrids onto the fire grid and
//! differentiates into per-cell dz/dx, dz/dy — all inside the .esm. This runner
//! supplies that terrain by binding an EarthSciIO (Rust) provider to the loader
//! field USGS3DEP.raw.elevation, exactly as run-model.jl binds the Julia
//! EarthSciIO provider and run-model.py the Python one. Loads through
//! earthsci-ast, integrates while saving snapshots, and plots the front at
//! each time over the terrain the run actually used (PNG). No conformance / MMS.
//!
//! Usage:
//!   cargo run --release -- [model.esm] [t_end]
//!     model.esm   path to the .esm to run   (default: ../wildlandfire.esm)
//!     t_end       final integration time    (default: 57600 = 16 h)
//!
//! Requires a network connection — the model's terrain loader fetches live 3DEP.
//!
//! History: this runner once ran a flat-terrain stand-in (uniform dz/dx=dz/dy=0).
//! It now runs the terrain-coupled model — which surfaced (and this branch fixes)
//! the Rust array runtime's static-observed hoist: the conservative-regrid
//! geometry + Rothermel coefficients are STATE-FREE, so they are materialized
//! once at setup instead of on every RHS step. The front then agrees with the
//! Julia (Tsit5) and Python (LSODA) runs over the same live 3DEP terrain.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use earthsci_ast::parse::load_path_with_options;
use earthsci_ast::provider::{CadenceProvider, NativeField, ProviderError};
use earthsci_ast::simulate::{simulate_with_providers_inspect, SimulateOptions, SolverChoice};
use earthsci_ast::simulate_array::BuildInspection;
use earthsciio::{ArrayData, Cache, DataLoader, Provider};
use ndarray::{ArrayD, Axis, IxDyn};
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

const NT: usize = 6; // number of front snapshots (t = 0 … t_end)
// Grid: NX × NY cells over the LX × LY domain (dx = dy = 2000 m, matching the
// archive camp_fire_model.jl grid).
const NX: i64 = 18;
const NY: i64 = 20;
const LX: f64 = 36000.0; // domain width  (m): Camp-Fire extent (half-width 18 km)
const LY: f64 = 40000.0; // domain height (m): Camp-Fire extent (half-width 20 km)

// Camp-Fire domain (WGS84 lon/lat). 3DEP + LANDFIRE fetch over this bbox at GX×GY
// (must match TerrainRegrid's conservative_overlap NSRC / src cell-rings = 36×40).
const GX: usize = 36;
const GY: usize = 40;
// Camp-Fire domain: the archive Pulga–Paradise-midpoint box (matches the
// current wildlandfire.esm ignition at Pulga (25901, 22807) and run-model.py/.jl).
const BBOX_W: f64 = -121.7400;
const BBOX_S: f64 = 39.6049;
const BBOX_E: f64 = -121.3192;
const BBOX_N: f64 = 39.9651;
// Landmark lon/lat (mapped to the fire metre-frame via the bbox in `plot`): the
// Pulga ignition origin and Paradise, marked on the plot like the Julia/Python runners.
const PULGA_LONLAT: (f64, f64) = (-121.437222, 39.810278);
const PARADISE_LONLAT: (f64, f64) = (-121.621944, 39.759722);
// ERA5 pressure-level (1000 hPa surface) CDS request: a small 0.25° lon/lat block
// around the fire (ERA5_NX×ERA5_NY must match Era5Regrid's ov5 NSRC / e5s GX,GY =
// 5×5 = 25), a single ignition-hour snapshot (CONST). No forcing scalars are set
// here any more — fuel comes from LANDFIRE, temperature/humidity/wind from ERA5,
// terrain from 3DEP, all per-cell through the in-model regridders.
const ERA5_AREA: [i32; 4] = [40, -122, 39, -121]; // [N,W,S,E] — 1° box → 5×5 @ 0.25°
const ERA5_NX: usize = 5;
const ERA5_NY: usize = 5;
const ERA5_HOUR: u32 = 14; // ≈06 PST ignition window (UTC), 2018-11-08

const SURFACE: RGBColor = RGBColor(0xfc, 0xfc, 0xfb);
const INK: RGBColor = RGBColor(0x52, 0x51, 0x4e);
const AXIS: RGBColor = RGBColor(0xc3, 0xc2, 0xb7);
const GRID: RGBColor = RGBColor(0xe1, 0xe0, 0xd9);
// Warm early→late ramp for the time encoding (over the terrain), matching the
// Julia / Python runners' cgrad(["#ffe08a","#ff9d3c","#e5391c","#8b0000"]).
const RAMP: [(u8, u8, u8); 4] = [
    (0xff, 0xe0, 0x8a),
    (0xff, 0x9d, 0x3c),
    (0xe5, 0x39, 0x1c),
    (0x8b, 0x00, 0x00),
];

/// A CONST EarthSciIO provider adapted to the ESS [`CadenceProvider`] seam — the
/// Rust analog of the Julia `EarthSciASTEarthSciIOExt` weakdep
/// extension and the Python duck-typed `_provider_sample_field` adapter. It
/// delegates `materialize` to the real EarthSciIO `Provider` (which fetches +
/// decodes the GeoTIFF) and converts each native field to the ESS `NativeField`.
/// The raster is static, so `refresh`/`refresh_times` report CONST.
struct EioConstProvider(Provider);

impl CadenceProvider for EioConstProvider {
    fn materialize(&mut self) -> Result<HashMap<String, NativeField>, ProviderError> {
        let fields = self.0.materialize().map_err(|e| ProviderError(e.to_string()))?;
        let mut out = HashMap::with_capacity(fields.len());
        for (name, f) in fields {
            let arr = match f.data {
                ArrayData::F64(v) => ArrayD::from_shape_vec(IxDyn(&f.shape), v)
                    .map_err(|e| ProviderError(format!("{name}: {e}")))?,
                other => {
                    return Err(ProviderError(format!(
                        "{name}: expected an f64 raster band, got {:?}",
                        other.dtype()
                    )))
                }
            };
            out.insert(name, NativeField::new(arr));
        }
        Ok(out)
    }

    fn refresh(&mut self, _t: f64) -> Result<Option<HashMap<String, NativeField>>, ProviderError> {
        Ok(None) // CONST: never refreshes
    }

    fn refresh_times(&self) -> Vec<f64> {
        Vec::new() // CONST: contributes no cadence anchor
    }
}

/// A DISCRETE (hourly, time-varying) EarthSciIO provider adapted to the ESS
/// [`CadenceProvider`] seam — the Rust analog of the Julia EIO
/// `Provider(cadence=DISCRETE, time_dim="valid_time")`. It materializes the whole
/// multi-hour ERA5 file ONCE (a 4-D `[valid_time, pressure_level, lat, lon]`
/// field) and, at each solver-second refresh anchor, slices the `valid_time`
/// record active at that time to a 3-D `[pressure_level, lat, lon]` slice — the
/// shape the model's now-3-D `Era5Regrid.F_*` reads. `refresh_times` is the file's
/// hourly grid in SOLVER seconds (record `record0` = ignition hour = t=0).
struct EioDiscreteProvider {
    /// The bound forcing-variable name (e.g. `"ERA5.pl.t"`).
    var: String,
    /// The full field, `[valid_time, pressure_level, lat, lon]`, read once.
    full: ArrayD<f64>,
    /// The `valid_time` record index that maps to solver-second `t = 0`.
    record0: i64,
    /// Number of `valid_time` records.
    n_records: usize,
    /// Cadence step (s) between records (hourly = 3600).
    freq_s: f64,
    /// Last record written (None-skip: refresh returns `None` when unchanged).
    last: Option<usize>,
}

impl EioDiscreteProvider {
    /// Materialize the whole multi-hour file once; `record0` is the record index
    /// at solver time 0 (the ignition hour), `freq_s` the cadence step.
    fn new(mut provider: Provider, var: String, record0: i64, freq_s: f64) -> Result<Self, String> {
        let fields = provider.materialize().map_err(|e| e.to_string())?;
        if fields.len() != 1 {
            return Err(format!("ERA5 provider fed {} fields; expected 1", fields.len()));
        }
        let (_name, f) = fields.into_iter().next().expect("checked len == 1");
        let full = match f.data {
            ArrayData::F64(v) => ArrayD::from_shape_vec(IxDyn(&f.shape), v)
                .map_err(|e| format!("{var}: {e}"))?,
            other => return Err(format!("{var}: expected an f64 field, got {:?}", other.dtype())),
        };
        let n_records = full.shape().first().copied().unwrap_or(0);
        Ok(Self { var, full, record0, n_records, freq_s, last: None })
    }

    /// The FLOOR `valid_time` record (at or before solver-second `t`) — the lower
    /// of the two bracketing records the model linearly interpolates between.
    fn record_at(&self, t: f64) -> usize {
        let k = (t / self.freq_s).floor() as i64 + self.record0;
        k.clamp(0, self.n_records as i64 - 1) as usize
    }

    /// The 4-D `[valid_time=2, pressure_level, lat, lon]` BRACKET at floor record
    /// `rec`: records `rec` and `rec+1` stacked on a leading size-2 axis — the
    /// shape the model's 4-D `Era5Regrid.F_*` reads. At the last record there is no
    /// successor, so the bracket degenerates to `[rec, rec]` (the model's weight
    /// then holds the endpoint). Mirrors the EarthSciIO provider's bracket mode.
    fn bracket(&self, rec: usize) -> NativeField {
        let rec1 = (rec + 1).min(self.n_records - 1); // end-clamp: hold the last record
        let r0 = self.full.index_axis(Axis(0), rec);
        let r1 = self.full.index_axis(Axis(0), rec1);
        let mut shape = vec![2usize];
        shape.extend_from_slice(r0.shape());
        let mut data = Vec::with_capacity(2 * r0.len());
        data.extend(r0.iter().copied());
        data.extend(r1.iter().copied());
        NativeField::new(
            ArrayD::from_shape_vec(IxDyn(&shape), data).expect("bracket shape"),
        )
    }

    /// Domain-means of the two bracket records at solver-second `t` (for the print).
    fn bracket_means(&self, t: f64) -> (f64, f64) {
        let rec = self.record_at(t);
        let rec1 = (rec + 1).min(self.n_records - 1);
        let mean = |r: usize| {
            let s = self.full.index_axis(Axis(0), r);
            let n = s.len().max(1);
            s.iter().copied().filter(|v| !v.is_nan()).sum::<f64>() / n as f64
        };
        (mean(rec), mean(rec1))
    }
}

impl CadenceProvider for EioDiscreteProvider {
    fn materialize(&mut self) -> Result<HashMap<String, NativeField>, ProviderError> {
        let rec = self.record_at(0.0);
        self.last = Some(rec);
        Ok(HashMap::from([(self.var.clone(), self.bracket(rec))]))
    }

    fn refresh(&mut self, t: f64) -> Result<Option<HashMap<String, NativeField>>, ProviderError> {
        let rec = self.record_at(t);
        if self.last == Some(rec) {
            return Ok(None); // floor record (hence bracket) unchanged — None-skip
        }
        self.last = Some(rec);
        Ok(Some(HashMap::from([(self.var.clone(), self.bracket(rec))])))
    }

    fn refresh_times(&self) -> Vec<f64> {
        // Hourly grid in solver seconds: record `record0` sits at t = 0.
        (0..self.n_records)
            .map(|k| (k as i64 - self.record0) as f64 * self.freq_s)
            .collect()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../wildlandfire.esm").to_string()
    });
    let t_end: f64 = match args.next() {
        Some(s) => s.parse().map_err(|e| format!("t_end: {e}"))?,
        None => 57600.0, // 16 h
    };

    // Fetch the terrain the model's USGS3DEP loader declares. A CONST provider
    // caches the raster once; ESS folds it into the const/forcing arrays and
    // TerrainRegrid regrids it. The GeoTIFF is decoded by the EarthSciIO `geotiff`
    // reader (pure-Rust `tiff` crate), the same field the Julia/Python readers
    // return.
    let bbox = format!("{BBOX_W},{BBOX_S},{BBOX_E},{BBOX_N}");
    let terrain_url = format!(
        "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/\
         exportImage?bbox={bbox}&size={GX},{GY}&format=tiff&bboxSR=4326&imageSR=4326\
         &pixelType=F32&interpolation=+RSP_BilinearInterpolation&f=image"
    );
    let landfire_url = format!(
        "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2022/LF2022_FBFM13_CONUS/\
         ImageServer/exportImage?bbox={bbox}&bboxSR=4326&imageSR=4326&size={GX},{GY}\
         &format=tiff&pixelType=S16&interpolation=+RSP_NearestNeighbor&f=image"
    );
    // ERA5 surface met via CDS: a DISCRETE (hourly) `cds://` request spanning two
    // days (Nov 8-9 2018) × all 24 hours = 48 `valid_time` records, 1000 hPa,
    // t/u/v/r over ERA5_AREA. The ignition hour (record ERA5_HOUR = 14, i.e.
    // 2018-11-08 14:00 UTC) is solver time t = 0; an EioDiscreteProvider slices
    // the record active at each solver hour. The cds transport authenticates from
    // ~/.cdsapirc (auth_realm "cds").
    let [n5, w5, s5, e5] = ERA5_AREA;
    let all_hours = (0..24)
        .map(|h| format!("\"{h:02}:00\""))
        .collect::<Vec<_>>()
        .join(",");
    let era5_req = format!(
        "{{\"area\":[{n5},{w5},{s5},{e5}],\"data_format\":\"netcdf\",\"day\":[\"08\",\"09\"],\
         \"download_format\":\"unarchived\",\"month\":[\"11\"],\"pressure_level\":[\"1000\"],\
         \"product_type\":[\"reanalysis\"],\"time\":[{all_hours}],\
         \"variable\":[\"relative_humidity\",\"temperature\",\"u_component_of_wind\",\
         \"v_component_of_wind\"],\"year\":[\"2018\"]}}"
    );
    let era5_url = format!("cds://reanalysis-era5-pressure-levels?{era5_req}");
    eprintln!(
        "fetching   3DEP + LANDFIRE ({GX}×{GY}) and ERA5 met ({ERA5_NX}×{ERA5_NY}) for {bbox} …"
    );
    let cache_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/.esio-cache"));
    // Register the `cds` auth resolver (PRIVATE-TOKEN from ~/.cdsapirc) so the CDS
    // transport can authenticate the ERA5 fetch. Unlike the Julia/Python caches,
    // the Rust cache has no implicit ~/.cdsapirc fallback — the realm is explicit.
    let cds_auth = std::sync::Arc::new(
        earthsciio::transport::cds_auth()
            .map_err(|e| format!("CDS auth (needs ~/.cdsapirc or $CDSAPI_KEY): {e}"))?,
    );
    let cache = Arc::new(
        Cache::builder()
            .data_dir(cache_dir)
            .offline(false)
            .register_auth(cds_auth)
            .build()
            .map_err(|e| format!("build EarthSciIO cache: {e}"))?,
    );
    let geotiff_provider = |name: &str, url: &str| -> Result<EioConstProvider, String> {
        let loader = DataLoader::new(name, "geotiff", url);
        Provider::new(loader, cache.clone(), None)
            .map(EioConstProvider)
            .map_err(|e| format!("build the {name} provider (network needed for {url}): {e}"))
    };
    // ERA5 is DISCRETE: build an EioDiscreteProvider per var over the 48-record
    // file. record0 = ERA5_HOUR (ignition hour → t=0); hourly cadence.
    let era5_provider = |short: &str| -> Result<EioDiscreteProvider, String> {
        let loader = DataLoader::new("ERA5", "netcdf", &era5_url)
            .variables([short])
            .auth_realm("cds");
        let provider = Provider::new(loader, cache.clone(), None).map_err(|e| {
            format!("build the ERA5.pl.{short} CDS provider (~/.cdsapirc needed): {e}")
        })?;
        EioDiscreteProvider::new(provider, format!("ERA5.pl.{short}"), ERA5_HOUR as i64, 3600.0)
    };
    let mut providers: HashMap<String, Box<dyn CadenceProvider>> = HashMap::new();
    providers.insert("USGS3DEP.raw.elevation".to_string(),
        Box::new(geotiff_provider("USGS3DEP", &terrain_url)?));
    providers.insert("LANDFIRE.raw.fuel_model".to_string(),
        Box::new(geotiff_provider("LANDFIRE", &landfire_url)?));
    // Evidence the met is time-INTERPOLATED: each solver time now gets the TWO
    // records bracketing it; print both bracket domain-means at t=0 and t=t_end.
    // The model blends each bracket linearly (was one piecewise-constant record).
    eprintln!("ERA5 met is time-INTERPOLATED (2-record bracket): bracket domain-means at t=0 vs t={t_end}s");
    for short in ["t", "u", "v", "r"] {
        let p = era5_provider(short)?;
        let (a0, b0) = p.bracket_means(0.0);
        let (ae, be) = p.bracket_means(t_end);
        eprintln!(
            "  ERA5.pl.{short:<2} record0={} of {} records: bracket [{:.3}, {:.3}] -> [{:.3}, {:.3}]",
            p.record0, p.n_records, a0, b0, ae, be
        );
        providers.insert(format!("ERA5.pl.{short}"), Box::new(p));
    }

    // ERA5 src geometry: place the ERA5 sub-grid at its TRUE position in the fire
    // metre-frame (equirectangular map of the ERA5 lon/lat against the fire bbox)
    // so the conservative overlap generalizes to a moved/larger fire domain.
    let mlon = LX / (BBOX_E - BBOX_W);
    let mlat = LY / (BBOX_N - BBOX_S);
    let era5_geom = [
        ("Era5Regrid.src_x0", (ERA5_AREA[1] as f64 - 0.125 - BBOX_W) * mlon),
        ("Era5Regrid.src_dx", 0.25 * mlon),
        ("Era5Regrid.src_y0", (ERA5_AREA[2] as f64 - 0.125 - BBOX_S) * mlat),
        ("Era5Regrid.src_dy", 0.25 * mlat),
    ];

    eprintln!("loading    {model_path}  (NX={NX}, NY={NY})");
    let bindings: std::collections::BTreeMap<String, i64> =
        [("NX".to_string(), NX), ("NY".to_string(), NY)].into_iter().collect();
    let file = load_path_with_options(&model_path, &bindings)
        .map_err(|e| format!("{model_path}: {e}"))?;

    let times: Vec<f64> = (0..NT).map(|k| t_end * k as f64 / (NT - 1) as f64).collect();
    // Erk (explicit Tsitouras 5(4)) — the SAME method as Julia's Tsit5. The
    // level-set Godunov Hamiltonian is hyperbolic, not stiff, so an adaptive
    // explicit RK is the natural choice.
    let opts = SimulateOptions {
        solver: SolverChoice::Erk,
        reltol: 1e-2,
        abstol: 1e-3,
        output_times: Some(times.clone()),
        ..Default::default()
    };
    // Extent + scalar forcing for the coupled behavior stack. R_0 comes from
    // Rothermel (driven by these), so it is NOT set here. The terrain slope is
    // per-cell now, supplied by the USGS3DEP provider through TerrainRegrid — no
    // TerrainSlope.dzdx/dzdy override.
    // Fuel, moisture and wind are no longer set here — they come per-cell from
    // LANDFIRE + ERA5 through the in-model regridders; only the domain extent and
    // the ERA5 source geometry are supplied.
    let mut params: HashMap<String, f64> = [
        ("LevelSetFireSpread.Lx".to_string(), LX),
        ("LevelSetFireSpread.Ly".to_string(), LY),
    ]
    .into_iter()
    .collect();
    for (k, v) in era5_geom {
        params.insert(k.to_string(), v);
    }
    // Time-interpolation phase: solver t=0 is the ignition hour (an ERA5 cadence
    // anchor), so t_interp_ref=0; dt_interp is the hourly ERA5 frequency. The model
    // computes w = frac((t - t_interp_ref)/dt_interp) to blend each bracket.
    params.insert("ERA5.t_interp_ref".to_string(), 0.0);
    params.insert("ERA5.dt_interp".to_string(), 3600.0);
    eprintln!("simulating (0.0, {t_end}) with Erk, {NT} snapshots …");
    // `inspect` captures the build-once geometry so we can read the terrain the
    // model actually used (TerrainRegrid.elev_xy) straight out of the run.
    let mut insp = BuildInspection::default();
    let sol = simulate_with_providers_inspect(
        &file,
        (0.0, t_end),
        &params,
        &HashMap::new(),
        &opts,
        providers,
        Some(&mut insp),
    )?;

    // Assemble the primary 2-D field: elements "<stem>[i,j]", grouped by stem.
    let mut groups: BTreeMapAlias = BTreeMapAlias::new();
    for (row, name) in sol.state_variable_names.iter().enumerate() {
        if let Some((stem, i, j)) = parse_cell2(name) {
            groups.entry(stem).or_default().push((i, j, row));
        }
    }
    // Prefer the level-set state field psi (the mounted component exposes many
    // [x,y] observeds too); fall back to the widest 2-D field if no psi present.
    let stem = {
        let psi_keys: Vec<&String> = groups.keys().filter(|k| k.ends_with("psi")).collect();
        let pool: Vec<&String> = if psi_keys.is_empty() {
            groups.keys().collect()
        } else {
            psi_keys
        };
        pool.into_iter()
            .max_by_key(|k| groups[*k].len())
            .cloned()
            .ok_or("no 2-D indexed state field found to plot")?
    };
    let cells = &groups[&stem];
    let nx = cells.iter().map(|c| c.0).max().unwrap();
    let ny = cells.iter().map(|c| c.1).max().unwrap();
    let dx = LX / nx as f64;
    let dy = LY / ny as f64;
    let xs: Vec<f64> = (0..nx).map(|i| (i as f64 + 0.5) * dx).collect();
    let ys: Vec<f64> = (0..ny).map(|j| (j as f64 + 0.5) * dy).collect();
    let nt = sol.time.len();
    let mut psi = vec![vec![vec![f64::NAN; ny]; nx]; nt];
    for &(i, j, row) in cells {
        for k in 0..nt {
            psi[k][i - 1][j - 1] = sol.state[row][k];
        }
    }

    let cell_area = dx * dy;
    println!("\nsolver: {} ; field '{stem}' is {nx}×{ny}", sol.metadata.solver);
    for k in 0..nt {
        let mut n = 0usize;
        let mut sx = 0.0;
        for i in 0..nx {
            for j in 0..ny {
                if psi[k][i][j] < 0.0 {
                    n += 1;
                    sx += xs[i];
                }
            }
        }
        let r = ((n as f64 * cell_area) / std::f64::consts::PI).sqrt();
        let cx = if n > 0 { sx / n as f64 } else { f64::NAN };
        println!(
            "  t={:7.0}s  r_eff={:8.1} m  x-centroid={:8.0} m (ignition x=25901, Pulga)",
            sol.time[k], r, cx
        );
    }

    // The terrain the model ACTUALLY USED, read from the run itself:
    // TerrainRegrid.elev_xy (the loader elevation conservatively regridded onto
    // the fire [x,y] grid), pulled from the build inspection — not a second fetch.
    let elev = insp
        .setup_arrays
        .get("TerrainRegrid.elev_xy")
        .ok_or("TerrainRegrid.elev_xy not captured in the build inspection")?;
    let (emin, emax) = elev
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    // Every behavior input is now a real, per-cell loader field — summarise each
    // regridded field's range + domain mean straight from the build inspection.
    println!("\ncoupled loader forcing (per-cell, regridded onto the fire grid):");
    for (key, label, unit, scale) in [
        ("TerrainRegrid.elev_xy", "3DEP elevation", "m", 1.0),
        ("TerrainRegrid.fuel_xy", "LANDFIRE fuel code", "", 1.0),
        ("Era5Regrid.t_xy", "ERA5 temperature", "K", 1.0),
        ("Era5Regrid.rh_xy", "ERA5 rel humidity", "%", 100.0),
        ("Era5Regrid.u_xy", "ERA5 u-wind", "m/s", 1.0),
        ("Era5Regrid.v_xy", "ERA5 v-wind", "m/s", 1.0),
    ] {
        if let Some(a) = insp.setup_arrays.get(key) {
            let (mut lo, mut hi, mut sum) = (f64::INFINITY, f64::NEG_INFINITY, 0.0);
            for &v in a.iter() {
                let v = v * scale;
                lo = lo.min(v);
                hi = hi.max(v);
                sum += v;
            }
            println!("  {label:<20} {lo:.2}–{hi:.2} {unit} (mean {:.2})", sum / a.len() as f64);
        }
    }

    let model_name = Path::new(&model_path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| model_path.clone());
    let plot_path = Path::new(&model_path)
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("front_rust.png");
    plot(
        &plot_path,
        &format!("{model_name} — front {{{stem}=0}}: 3DEP terrain + LANDFIRE fuel + ERA5 wind, Rust / {}", sol.metadata.solver),
        &xs,
        &ys,
        &psi,
        &sol.time,
        elev,
        (emin, emax),
    )?;
    println!("wrote plot: {}", plot_path.display());
    Ok(())
}

type BTreeMapAlias = std::collections::BTreeMap<String, Vec<(usize, usize, usize)>>;

/// Parse `"<stem>[i,j]"` into `(stem, i, j)`; `None` otherwise.
fn parse_cell2(name: &str) -> Option<(String, usize, usize)> {
    let inner = name.strip_suffix(']')?;
    let (stem, idx) = inner.rsplit_once('[')?;
    let (i, j) = idx.split_once(',')?;
    Some((stem.to_string(), i.trim().parse().ok()?, j.trim().parse().ok()?))
}

/// Warm (early→late) color for `t` in [0,1] (piecewise-linear over `RAMP`).
fn warm_color(t: f64) -> RGBColor {
    let t = t.clamp(0.0, 1.0) * (RAMP.len() - 1) as f64;
    let k = t.floor() as usize;
    if k >= RAMP.len() - 1 {
        let s = RAMP[RAMP.len() - 1];
        return RGBColor(s.0, s.1, s.2);
    }
    let f = t - k as f64;
    let (a, b) = (RAMP[k], RAMP[k + 1]);
    let l = |x: u8, y: u8| (x as f64 + (y as f64 - x as f64) * f).round() as u8;
    RGBColor(l(a.0, b.0), l(a.1, b.1), l(a.2, b.2))
}

/// A matplotlib-`terrain`-like color for a normalized elevation `t` in [0,1]
/// (blue lowlands → green → yellow → brown → white peaks), lightened toward white
/// to mimic the Julia/Python heatmap's alpha=0.85 over a light background.
fn terrain_color(t: f64) -> RGBColor {
    // Stops matching matplotlib's `terrain` colormap.
    const STOPS: [(f64, (f64, f64, f64)); 6] = [
        (0.00, (0.20, 0.20, 0.60)),
        (0.15, (0.00, 0.60, 1.00)),
        (0.25, (0.00, 0.80, 0.40)),
        (0.50, (1.00, 1.00, 0.60)),
        (0.75, (0.50, 0.36, 0.33)),
        (1.00, (1.00, 1.00, 1.00)),
    ];
    let t = t.clamp(0.0, 1.0);
    let mut c = STOPS[STOPS.len() - 1].1;
    for w in STOPS.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            c = (
                c0.0 + (c1.0 - c0.0) * f,
                c0.1 + (c1.1 - c0.1) * f,
                c0.2 + (c1.2 - c0.2) * f,
            );
            break;
        }
    }
    // alpha=0.85 over white: mix 0.85*color + 0.15*white.
    let mix = |v: f64| ((v * 0.85 + 0.15) * 255.0).round().clamp(0.0, 255.0) as u8;
    RGBColor(mix(c.0), mix(c.1), mix(c.2))
}

/// Marching squares: line segments of the zero contour of a field sampled at
/// (xs[i], ys[j]). Each boundary cell contributes one segment.
fn zero_segments(xs: &[f64], ys: &[f64], z: &[Vec<f64>]) -> Vec<[(f64, f64); 2]> {
    let interp = |a: (f64, f64, f64), b: (f64, f64, f64)| {
        let t = a.2 / (a.2 - b.2);
        (a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1))
    };
    let mut segs = Vec::new();
    for i in 0..xs.len() - 1 {
        for j in 0..ys.len() - 1 {
            let c = [
                (xs[i], ys[j], z[i][j]),
                (xs[i + 1], ys[j], z[i + 1][j]),
                (xs[i + 1], ys[j + 1], z[i + 1][j + 1]),
                (xs[i], ys[j + 1], z[i][j + 1]),
            ];
            let mut pts = Vec::new();
            for e in 0..4 {
                let (a, b) = (c[e], c[(e + 1) % 4]);
                if (a.2 < 0.0) != (b.2 < 0.0) {
                    pts.push(interp(a, b));
                }
            }
            if pts.len() >= 2 {
                segs.push([pts[0], pts[1]]);
            }
        }
    }
    segs
}

/// Single figure: the 3DEP terrain heatmap + the front (zero contour) at each
/// saved time (color = time).
#[allow(clippy::too_many_arguments)]
fn plot(
    path: &Path,
    title: &str,
    xs: &[f64],
    ys: &[f64],
    psi: &[Vec<Vec<f64>>],
    times: &[f64],
    elev: &ArrayD<f64>,
    erange: (f64, f64),
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (680, 560)).into_drawing_area();
    root.fill(&SURFACE)?;

    let (title_area, body) = root.split_vertically(30);
    let centered = TextStyle::from(("sans-serif", 14, &INK).into_text_style(&title_area))
        .pos(Pos::new(HPos::Center, VPos::Center));
    title_area.draw_text(title, &centered, (340, 16))?;

    let mut chart = ChartBuilder::on(&body)
        .margin_top(9)
        .margin_bottom(9)
        .margin_left(40)
        .margin_right(52)
        .x_label_area_size(36)
        .y_label_area_size(56)
        .build_cartesian_2d(0f64..LX, 0f64..LY)?;

    // Terrain heatmap: one filled rectangle per fire cell, colored by elev_xy.
    // elev is [x,y] = (nx, ny); cell (i,j) spans [i*dx,(i+1)*dx] × [j*dy,(j+1)*dy].
    let (nx, ny) = (elev.shape()[0], elev.shape()[1]);
    let dx = LX / nx as f64;
    let dy = LY / ny as f64;
    let (emin, emax) = erange;
    let span = (emax - emin).max(1e-9);
    chart.draw_series((0..nx).flat_map(|i| {
        (0..ny).map(move |j| {
            let t = (elev[[i, j]] - emin) / span;
            let x0 = i as f64 * dx;
            let y0 = j as f64 * dy;
            Rectangle::new(
                [(x0, y0), (x0 + dx, y0 + dy)],
                terrain_color(t).filled(),
            )
        })
    }))?;

    chart
        .configure_mesh()
        .x_desc("x (m)")
        .y_desc("y (m)")
        .x_labels(6)
        .y_labels(6)
        .axis_style(AXIS)
        .bold_line_style(GRID.mix(0.35))
        .light_line_style(TRANSPARENT)
        .label_style(("sans-serif", 10, &INK))
        .draw()?;

    let n = times.len();
    for (k, &t) in times.iter().enumerate() {
        let frac = if n == 1 { 1.0 } else { k as f64 / (n - 1) as f64 };
        let color = warm_color(frac);
        let segs = zero_segments(xs, ys, &psi[k]);
        chart
            .draw_series(
                segs.iter()
                    .map(|s| PathElement::new(vec![s[0], s[1]], color.stroke_width(3))),
            )?
            .label(format!("{:.1}h", t / 3600.0))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], color.stroke_width(3)));
    }

    // Landmark markers — Pulga (ignition origin) and Paradise — mapped from their
    // real lon/lat into the fire metre-frame via the domain bbox, matching the
    // town markers the Julia / Python runners draw.
    let to_xy = |(lon, lat): (f64, f64)| {
        (
            (lon - BBOX_W) / (BBOX_E - BBOX_W) * LX,
            (lat - BBOX_S) / (BBOX_N - BBOX_S) * LY,
        )
    };
    for (lonlat, name) in [
        (PULGA_LONLAT, "Pulga (ignition)"),
        (PARADISE_LONLAT, "Paradise"),
    ] {
        let p = to_xy(lonlat);
        chart.draw_series(std::iter::once(Cross::new(p, 7, INK.stroke_width(2))))?;
        chart.draw_series(std::iter::once(Circle::new(p, 3, INK.filled())))?;
        chart.draw_series(std::iter::once(Text::new(
            name.to_string(),
            (p.0 + 900.0, p.1),
            ("sans-serif", 11).into_font().color(&INK),
        )))?;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(SURFACE.mix(0.85))
        .border_style(AXIS)
        .label_font(("sans-serif", 10, &INK))
        .draw()?;

    root.present()?;
    Ok(())
}
