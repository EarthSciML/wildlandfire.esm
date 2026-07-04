//! run-model — minimal single-model runner for a 2-D EarthSci `.esm` (Rust).
//!
//! Model: 2-D level-set Hamilton-Jacobi front propagation  psi_t = -S |grad psi|
//! on the doubly-periodic unit box, with |grad psi| discretized by the Godunov
//! (Rouy-Tourin / Osher-Sethian) upwind scheme (imported by reference from the
//! sibling earthscidiscretizations checkout — nothing is vendored here). psi is
//! initialized as the signed distance to a circle, so the front {psi = 0}
//! expands outward at speed S. Loads through earthsci-toolkit, integrates while
//! saving several snapshots, and plots the front location at each time as a
//! single figure with time encoded by color (PNG). No conformance / MMS checks.
//!
//! Usage:
//!   cargo run --quiet -- [model.esm] [t_end]
//!     model.esm   path to the .esm to run   (default: ../wildlandfire.esm)
//!     t_end       final integration time    (default: 2.0)

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::Path;

use earthsci_toolkit::parse::load_path_with_options;
use earthsci_toolkit::simulate::{simulate, SimulateOptions, SolverChoice};
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

const NT: usize = 6; // number of front snapshots (t = 0 … t_end)
// Grid resolution (metaparameters NX = NY = N). The Godunov Hamiltonian is
// non-smooth, so diffsol's adaptive Erk takes many tiny steps and the array
// backend interprets the RHS — both scale with cell count, so this Rust path
// runs a coarser grid than the (compiled-RHS) Julia one to stay tractable.
const N: i64 = 32;
// LevelSetFireSpread.R_0 override (m/s): the referenced component's default
// R_0=1.0 was calibrated for its old 500 m domain; on the unit box that
// overshoots, so we set the isotropic speed to 0.1 (front reaches 0.15 + 0.1·t).
const R0: f64 = 0.1;

const SURFACE: RGBColor = RGBColor(0xfc, 0xfc, 0xfb);
const INK: RGBColor = RGBColor(0x52, 0x51, 0x4e);
const AXIS: RGBColor = RGBColor(0xc3, 0xc2, 0xb7);
const GRID: RGBColor = RGBColor(0xe1, 0xe0, 0xd9);
// Validated dataviz sequential blue ramp (medium→dark) for the time encoding.
const RAMP: [(u8, u8, u8); 4] = [
    (0x86, 0xb6, 0xef),
    (0x39, 0x87, 0xe5),
    (0x1c, 0x5c, 0xab),
    (0x0d, 0x36, 0x6b),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../wildlandfire.esm").to_string()
    });
    let t_end: f64 = match args.next() {
        Some(s) => s.parse().map_err(|e| format!("t_end: {e}"))?,
        None => 2.0,
    };

    eprintln!("loading    {model_path}  (NX=NY={N})");
    let bindings: BTreeMap<String, i64> =
        [("NX".to_string(), N), ("NY".to_string(), N)].into_iter().collect();
    let file = load_path_with_options(&model_path, &bindings)
        .map_err(|e| format!("{model_path}: {e}"))?;

    let times: Vec<f64> = (0..NT).map(|k| t_end * k as f64 / (NT - 1) as f64).collect();
    let opts = SimulateOptions {
        solver: SolverChoice::Erk,
        reltol: 1e-2,
        abstol: 1e-3,
        output_times: Some(times.clone()),
        ..Default::default()
    };
    let params: HashMap<String, f64> =
        [("LevelSetFireSpread.R_0".to_string(), R0)].into_iter().collect();
    eprintln!("simulating (0.0, {t_end}) with Erk, {NT} snapshots (R_0={R0}) …");
    let sol = simulate(&file, (0.0, t_end), &params, &HashMap::new(), &opts)?;

    // Assemble the primary 2-D field: elements "<stem>[i,j]", grouped by stem.
    let mut groups: BTreeMap<String, Vec<(usize, usize, usize)>> = BTreeMap::new(); // stem => (i,j,row)
    for (row, name) in sol.state_variable_names.iter().enumerate() {
        if let Some((stem, i, j)) = parse_cell2(name) {
            groups.entry(stem).or_default().push((i, j, row));
        }
    }
    let stem = groups
        .keys()
        .max_by_key(|k| groups[*k].len())
        .cloned()
        .ok_or("no 2-D indexed state field found to plot")?;
    let cells = &groups[&stem];
    let nx = cells.iter().map(|c| c.0).max().unwrap();
    let ny = cells.iter().map(|c| c.1).max().unwrap();
    let xs: Vec<f64> = (0..nx).map(|i| (i as f64 + 0.5) / nx as f64).collect();
    let ys: Vec<f64> = (0..ny).map(|j| (j as f64 + 0.5) / ny as f64).collect();
    // psi[k] is the nx×ny field at saved time sol.time[k].
    let nt = sol.time.len();
    let mut psi = vec![vec![vec![f64::NAN; ny]; nx]; nt];
    for &(i, j, row) in cells {
        for k in 0..nt {
            psi[k][i - 1][j - 1] = sol.state[row][k];
        }
    }

    let cell_area = (1.0 / nx as f64) * (1.0 / ny as f64);
    println!("\nsolver: {} ; field '{stem}' is {nx}×{ny}", sol.metadata.solver);
    for k in 0..nt {
        let burned: f64 = psi[k].iter().flatten().filter(|&&v| v < 0.0).count() as f64 * cell_area;
        println!("  t={:.2}  front r_eff={:.4}", sol.time[k], (burned / std::f64::consts::PI).sqrt());
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
        &format!("{model_name} — front {{{stem}=0}}, Rust / Erk"),
        &xs,
        &ys,
        &psi,
        &sol.time,
    )?;
    println!("wrote plot: {}", plot_path.display());
    Ok(())
}

/// Parse `"<stem>[i,j]"` into `(stem, i, j)`; `None` otherwise.
fn parse_cell2(name: &str) -> Option<(String, usize, usize)> {
    let inner = name.strip_suffix(']')?;
    let (stem, idx) = inner.rsplit_once('[')?;
    let (i, j) = idx.split_once(',')?;
    Some((stem.to_string(), i.trim().parse().ok()?, j.trim().parse().ok()?))
}

/// Sequential blue color for `t` in [0,1] (piecewise-linear over `RAMP`).
fn seq_color(t: f64) -> RGBColor {
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

/// Marching squares: line segments of the zero contour of a field sampled at
/// (xs[i], ys[j]). Each boundary cell contributes one segment (its first two
/// edge crossings) — clean for smooth contours like an expanding circle.
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

/// Single figure: the front (zero contour) at each saved time, color = time.
fn plot(
    path: &Path,
    title: &str,
    xs: &[f64],
    ys: &[f64],
    psi: &[Vec<Vec<f64>>],
    times: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (620, 560)).into_drawing_area();
    root.fill(&SURFACE)?;

    let (title_area, body) = root.split_vertically(30);
    let centered = TextStyle::from(("sans-serif", 15, &INK).into_text_style(&title_area))
        .pos(Pos::new(HPos::Center, VPos::Center));
    title_area.draw_text(title, &centered, (310, 16))?;

    // Margins chosen so the data rect is square (round circles): 476×476.
    let mut chart = ChartBuilder::on(&body)
        .margin_top(9)
        .margin_bottom(9)
        .margin_left(52)
        .margin_right(52)
        .x_label_area_size(36)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..1f64, 0f64..1f64)?;
    chart
        .configure_mesh()
        .x_desc("x")
        .y_desc("y")
        .x_labels(6)
        .y_labels(6)
        .axis_style(AXIS)
        .bold_line_style(GRID)
        .light_line_style(TRANSPARENT)
        .label_style(("sans-serif", 10, &INK))
        .draw()?;

    let n = times.len();
    for (k, &t) in times.iter().enumerate() {
        let frac = if n == 1 { 1.0 } else { k as f64 / (n - 1) as f64 };
        let color = seq_color(frac);
        let segs = zero_segments(xs, ys, &psi[k]);
        chart
            .draw_series(
                segs.iter()
                    .map(|s| PathElement::new(vec![s[0], s[1]], color.stroke_width(2))),
            )?
            .label(format!("{t:.2}"))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], color.stroke_width(3)));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(SURFACE)
        .border_style(AXIS)
        .label_font(("sans-serif", 10, &INK))
        .draw()?;

    root.present()?;
    Ok(())
}
