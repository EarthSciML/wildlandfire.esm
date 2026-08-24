//! profile — CPU sampling profile of the Rust level-set build+solve (pprof-rs).
//!
//! Loads and drives the SAME three-loader coupled model the runner uses
//! (run-model-rs): the Rothermel + NFDRS behavior stack feeding the level-set
//! front, with every behavior input a real, per-cell data-loader field regridded
//! onto the fire grid IN the .esm — 3DEP terrain, LANDFIRE fuel and ERA5 met. It
//! binds those loaders as providers exactly like the runner (so the profile
//! reflects the real coupled workload — the build-time conservative regrid is now a
//! major cost, not the solve). Samples a single build+solve at the runner's
//! 2×-refined NX=36, NY=40 grid (double the runner's 18×20); t_end via env PROFILE_TEND (default 3600 s — the build
//! dominates, so a shorter horizon still captures it). Emits, under ../profiles/:
//!   flamegraph-rust.svg     — inclusive flame graph
//!   profile-rust-top.txt    — self time by leaf function (what the CPU sat in)
//!
//! Requires a network connection (3DEP + LANDFIRE, no auth) + a Copernicus CDS key
//! in ~/.cdsapirc (ERA5) on the first run; later runs read the warmed cache. Build
//! with the `profile` feature + the `profiling` profile (see ../profile-rs.sh).

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use earthsci_ast::parse::load_path_with_options;
use earthsci_ast::provider::{CadenceProvider, NativeField, ProviderError};
use earthsci_ast::problem::{esm_problem, solve, ProblemOptions};
use earthsci_ast::simulate::{Alg, SolveOptions};
use earthsciio::{ArrayData, Cache, DataLoader, Provider};
use ndarray::{ArrayD, Axis, IxDyn};

// Grid + domain, identical to the runner (run-model-rs/src/main.rs).
const NX: i64 = 36; // 2×-refined fire grid (double the runner's 18×20)
const NY: i64 = 40;
const LX: f64 = 36000.0;
const LY: f64 = 40000.0;
const GX: usize = 36;
const GY: usize = 40;
const BBOX_W: f64 = -121.7400;
const BBOX_S: f64 = 39.6049;
const BBOX_E: f64 = -121.3192;
const BBOX_N: f64 = 39.9651;
const ERA5_AREA: [i32; 4] = [40, -122, 39, -121];
const ERA5_NX: usize = 5;
const ERA5_NY: usize = 5;
const ERA5_HOUR: u32 = 14;

/// A CONST EarthSciIO provider adapted to the ESS `CadenceProvider` seam (the same
/// adapter run-model-rs uses): delegates `materialize` to the real EarthSciIO
/// `Provider` and converts each native f64 field to an ESS `NativeField`.
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
        Ok(None)
    }
    fn refresh_times(&self) -> Vec<f64> {
        Vec::new()
    }
}

/// A DISCRETE (hourly, time-varying) EarthSciIO provider adapted to the ESS
/// [`CadenceProvider`] seam — identical to the one run-model-rs uses. It materializes
/// the whole multi-hour ERA5 file ONCE (`[valid_time, pressure_level, lat, lon]`) and
/// at each solver-second refresh anchor returns the 2-record `[valid_time=2, …]`
/// bracket the model's 4-D `Era5Regrid.F_*` reads and time-interpolates.
struct EioDiscreteProvider {
    var: String,
    full: ArrayD<f64>,
    record0: i64,
    n_records: usize,
    freq_s: f64,
    last: Option<usize>,
}

impl EioDiscreteProvider {
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

    /// The FLOOR `valid_time` record (at or before solver-second `t`).
    fn record_at(&self, t: f64) -> usize {
        let k = (t / self.freq_s).floor() as i64 + self.record0;
        k.clamp(0, self.n_records as i64 - 1) as usize
    }

    /// The 4-D `[valid_time=2, pressure_level, lat, lon]` bracket at floor record
    /// `rec` (records `rec` and `rec+1`, end-clamped to hold the last record).
    fn bracket(&self, rec: usize) -> NativeField {
        let rec1 = (rec + 1).min(self.n_records - 1);
        let r0 = self.full.index_axis(Axis(0), rec);
        let r1 = self.full.index_axis(Axis(0), rec1);
        let mut shape = vec![2usize];
        shape.extend_from_slice(r0.shape());
        let mut data = Vec::with_capacity(2 * r0.len());
        data.extend(r0.iter().copied());
        data.extend(r1.iter().copied());
        NativeField::new(ArrayD::from_shape_vec(IxDyn(&shape), data).expect("bracket shape"))
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
        (0..self.n_records)
            .map(|k| (k as i64 - self.record0) as f64 * self.freq_s)
            .collect()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = concat!(env!("CARGO_MANIFEST_DIR"), "/../wildlandfire.esm");
    let outdir = concat!(env!("CARGO_MANIFEST_DIR"), "/../profiles");
    std::fs::create_dir_all(outdir)?;
    let t_end: f64 = std::env::var("PROFILE_TEND").ok().and_then(|s| s.parse().ok()).unwrap_or(3600.0);

    // The three real data-loader providers, identical to run-model-rs.
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
    let [n5, w5, s5, e5] = ERA5_AREA;
    // DISCRETE ERA5: two days (Nov 8-9) × all 24 hours = 48 valid_time records, so
    // the provider can return the 2-record bracket the .esm's era5_t axis expects.
    let all_hours = (0..24).map(|h| format!("\"{h:02}:00\"")).collect::<Vec<_>>().join(",");
    let era5_req = format!(
        "{{\"area\":[{n5},{w5},{s5},{e5}],\"data_format\":\"netcdf\",\"day\":[\"08\",\"09\"],\
         \"download_format\":\"unarchived\",\"month\":[\"11\"],\"pressure_level\":[\"1000\"],\
         \"product_type\":[\"reanalysis\"],\"time\":[{all_hours}],\
         \"variable\":[\"relative_humidity\",\"temperature\",\"u_component_of_wind\",\
         \"v_component_of_wind\"],\"year\":[\"2018\"]}}"
    );
    let era5_url = format!("cds://reanalysis-era5-pressure-levels?{era5_req}");
    eprintln!("fetching   3DEP + LANDFIRE ({GX}×{GY}) and ERA5 met ({ERA5_NX}×{ERA5_NY}) for {bbox} …");
    let cache_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/.esio-cache"));
    let cds_auth = std::sync::Arc::new(
        earthsciio::transport::cds_auth()
            .map_err(|e| format!("CDS auth (needs ~/.cdsapirc or $CDSAPI_KEY): {e}"))?,
    );
    let cache = Arc::new(
        Cache::builder().data_dir(cache_dir).offline(false).register_auth(cds_auth).build()
            .map_err(|e| format!("build EarthSciIO cache: {e}"))?,
    );
    let geotiff_provider = |name: &str, url: &str| -> Result<EioConstProvider, String> {
        Provider::new(DataLoader::new(name, "geotiff", url), cache.clone(), None)
            .map(EioConstProvider).map_err(|e| format!("build {name} provider: {e}"))
    };
    // ERA5 is DISCRETE: an EioDiscreteProvider per var over the 48-record file.
    // record0 = ERA5_HOUR (ignition hour → t=0); hourly (3600 s) cadence; returns the
    // 2-record bracket the model time-interpolates (identical to run-model-rs).
    let era5_provider = |short: &str| -> Result<EioDiscreteProvider, String> {
        let loader = DataLoader::new("ERA5", "netcdf", &era5_url).variables([short]).auth_realm("cds");
        let provider = Provider::new(loader, cache.clone(), None)
            .map_err(|e| format!("build ERA5.pl.{short}: {e}"))?;
        EioDiscreteProvider::new(provider, format!("ERA5.pl.{short}"), ERA5_HOUR as i64, 3600.0)
    };
    let mut providers: HashMap<String, Box<dyn CadenceProvider>> = HashMap::new();
    providers.insert("USGS3DEP.raw.elevation".to_string(), Box::new(geotiff_provider("USGS3DEP", &terrain_url)?));
    providers.insert("LANDFIRE.raw.fuel_model".to_string(), Box::new(geotiff_provider("LANDFIRE", &landfire_url)?));
    for short in ["t", "u", "v", "r"] {
        providers.insert(format!("ERA5.pl.{short}"), Box::new(era5_provider(short)?));
    }
    let mlon = LX / (BBOX_E - BBOX_W);
    let mlat = LY / (BBOX_N - BBOX_S);
    let params: HashMap<String, f64> = [
        ("LevelSetFireSpread.Lx".to_string(), LX),
        ("LevelSetFireSpread.Ly".to_string(), LY),
        ("Era5Regrid.src_x0".to_string(), (ERA5_AREA[1] as f64 - 0.125 - BBOX_W) * mlon),
        ("Era5Regrid.src_dx".to_string(), 0.25 * mlon),
        ("Era5Regrid.src_y0".to_string(), (ERA5_AREA[2] as f64 - 0.125 - BBOX_S) * mlat),
        ("Era5Regrid.src_dy".to_string(), 0.25 * mlat),
        // time-interp phase: t=0 is an ERA5 cadence anchor (ignition hour); hourly dt.
        ("ERA5.t_interp_ref".to_string(), 0.0),
        ("ERA5.dt_interp".to_string(), 3600.0),
    ].into_iter().collect();

    eprintln!("loading    {model}  (NX={NX}, NY={NY})");
    let bindings: std::collections::BTreeMap<String, i64> =
        [("NX".to_string(), NX), ("NY".to_string(), NY)].into_iter().collect();
    let file = load_path_with_options(model, &bindings).map_err(|e| format!("{model}: {e}"))?;

    // phase 4: SciML option names — `alg` (was `solver`), `saveat` (was
    // `output_times`). Kept deliberately loose (1e-2/1e-3): this is a profile.
    let opts = SolveOptions {
        alg: Alg::Erk,
        reltol: 1e-2,
        abstol: 1e-3,
        saveat: Some(vec![0.0, t_end]),
        ..Default::default()
    };

    // Install the sampling profiler around the whole build + integration.
    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(997) // Hz; prime to avoid aliasing with any periodic work
        .blocklist(&["libsystem", "libdyld", "libc"])
        .build()?;

    eprintln!("profiling  build+solve (0.0, {t_end}) with Erk …");
    let t0 = std::time::Instant::now();
    // Build AND solve inside the profiled region on purpose: this script
    // profiles the whole path, and the build-time conservative regrid is a
    // major part of what is being measured.
    let prob = esm_problem(
        &file,
        (0.0, t_end),
        ProblemOptions { p: params.clone(), providers, ..Default::default() },
    )?;
    let sol = solve(&prob, &opts)?;
    let wall = t0.elapsed().as_secs_f64();
    eprintln!("build+solve done in {wall:.1}s ({} snapshots)", sol.time.len());

    let report = guard.report().build()?;
    let svg_path = format!("{outdir}/flamegraph-rust.svg");
    report.flamegraph(std::fs::File::create(&svg_path)?)?;
    eprintln!("wrote {svg_path}");

    // Self time by leaf frame: which function the CPU was executing when sampled.
    let mut self_counts: HashMap<String, isize> = HashMap::new();
    let mut total: isize = 0;
    for (frames, count) in report.data.iter() {
        total += *count;
        let leaf = frames.frames.first().and_then(|inlined| inlined.first())
            .map(|sym| format!("{sym}")).unwrap_or_else(|| "<unknown>".to_string());
        *self_counts.entry(leaf).or_insert(0) += *count;
    }
    let mut ranked: Vec<(String, isize)> = self_counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));

    let top_path = format!("{outdir}/profile-rust-top.txt");
    let mut f = std::fs::File::create(&top_path)?;
    writeln!(
        f,
        "# Rust build+solve self-time by leaf function\n\
         # NX={NX} NY={NY}  t_end={t_end}  wall={wall:.1}s  solver=Erk (diffsol)\n\
         # three real loaders (3DEP + LANDFIRE + ERA5), regridded in-model\n\
         # {total} samples @ 997 Hz\n\
         #  self%   samples  function"
    )?;
    for (name, c) in ranked.iter().take(40) {
        writeln!(f, "{:7.2}  {:8}  {}", 100.0 * *c as f64 / total as f64, c, name)?;
    }
    eprintln!("wrote {top_path}  ({total} samples)");
    eprintln!("PROFILE-RS DONE");
    Ok(())
}
