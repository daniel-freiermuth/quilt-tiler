//! Convert decrypted OESU chart files to a `PMTiles` vector tile archive,
//! a `MapLibre` GL style JSON, and a Signal K chart metadata file.
//!
//! Parses all input `.oesu` files in parallel, then writes MVT tiles directly
//! into a `PMTiles` v3 archive alongside `<stem>.style.json` and
//! `<stem>.metadata.json`.

mod style;
mod tiles;
mod zoom;
use std::time::{SystemTime, UNIX_EPOCH};

/// Current UTC time as `YYYY-MM-DDTHH:MM:SSZ` without pulling in chrono.
fn chrono_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let (s, m, h) = (secs % 60, (secs / 60) % 60, (secs / 3600) % 24);
    let (year, month, day) = days_to_ymd(secs / 86_400);
    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    let mut year = 1970u64;
    loop {
        let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
        let in_year = if leap { 366 } else { 365 };
        if days < in_year {
            break;
        }
        days -= in_year;
        year += 1;
    }
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let month_days = [31u64, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 1u64;
    for &md in &month_days {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    (year, month, days + 1)
}

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use tracing::{info, warn};

use zoom::zoom_from_scale;

/// Convert decrypted OESU chart files to a `PMTiles` vector tile archive and a `MapLibre` GL style.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Decrypted `.oesu` files to convert (glob or explicit paths).
    #[arg(required = true)]
    input: Vec<PathBuf>,

    /// Output `PMTiles` file.
    #[arg(short, long, default_value = "chart.pmtiles")]
    output: PathBuf,

    /// Where to write the generated `MapLibre` GL style JSON.
    /// Defaults to `<output-stem>.style.json` next to the `PMTiles` file.
    #[arg(long)]
    style_output: Option<PathBuf>,

    /// MVT tile URL template embedded in the generated style.
    /// Defaults to `http://localhost:3000/<output-stem>/{z}/{x}/{y}`.
    #[arg(long)]
    tile_url: Option<String>,

    /// Depth (metres) at or above which water is considered dangerous.
    /// DEPARE areas shallower than this get the darkest fill; the DEPCNT
    /// contour at exactly this depth is drawn as a prominent red line.
    #[arg(long, default_value_t = 3.0)]
    safety_depth: f64,

    /// Upper boundary of the "shallow but navigable" zone (metres).
    /// DEPARE areas between `safety_depth` and this value get a medium fill;
    /// deeper areas get the lightest fill.
    #[arg(long, default_value_t = 10.0)]
    shoal_depth: f64,

    /// Human-readable chart name used in the Signal K metadata.
    /// Defaults to the output file stem.
    #[arg(long)]
    name: Option<String>,

    /// Cap the output at this zoom level.  Charts natively rendered at higher
    /// zooms still fill down correctly to `--max-zoom`.  Useful to cut build
    /// time and output size during development.
    #[arg(long)]
    max_zoom: Option<u8>,
}

// Rayon pool setup + style/metadata output push main past 100 lines. Accept.
#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    // Tracy must start before rayon spins up its workers so every thread
    // is registered with the profiler from the moment it first runs.
    #[cfg(feature = "tracy")]
    let _tracy = tracy_client::Client::start();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    // Register the main thread with the profiler, then build the global rayon
    // pool so every worker calls register_thread!() before doing any work.
    profiling::register_thread!("main");
    ThreadPoolBuilder::new()
        .spawn_handler(|thread| {
            let mut b = std::thread::Builder::new();
            if let Some(name) = thread.name() { b = b.name(name.to_owned()); }
            if let Some(sz)   = thread.stack_size() { b = b.stack_size(sz); }
            b.spawn(move || {
                profiling::register_thread!();
                thread.run();
            })?;
            Ok(())
        })
        .build_global()
        .context("building rayon thread pool")?;

    let args = Args::parse();

    // Parse all input files in parallel; skip files that fail to parse.
    let cells: Vec<s57::S57Cell> = args
        .input
        .par_iter()
        .filter_map(|path| {
            let data = match std::fs::read(path) {
                Ok(d) => d,
                Err(e) => {
                    warn!(file = %path.display(), error = %e, "cannot read");
                    return None;
                }
            };
            match oesu::parse_file(&data) {
                Ok(cell) => {
                    let z = zoom_from_scale(cell.native_scale);
                    info!(
                        name = %cell.name,
                        scale = cell.native_scale,
                        zoom = z,
                        features = cell.features.len(),
                        "parsed"
                    );
                    Some(cell)
                }
                Err(e) => {
                    warn!(file = %path.display(), error = %e, "skipping");
                    None
                }
            }
        })
        .collect();

    info!(parsed = cells.len(), "charts parsed, writing tiles");

    // Aggregate geographic bounds from parsed cells for metadata.
    let bounds = cells.iter().fold(
        [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
        |mut acc, c| {
            acc[0] = acc[0].min(c.bounds[0]);
            acc[1] = acc[1].min(c.bounds[1]);
            acc[2] = acc[2].max(c.bounds[2]);
            acc[3] = acc[3].max(c.bounds[3]);
            acc
        },
    );

    let (min_zoom, out_max_zoom) = tiles::write_pmtiles(&cells, &args.output, args.max_zoom)?;

    // Derive output siblings from the PMTiles path unless overridden.
    let source_id = args
        .output
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("chart")
        .to_owned();
    let chart_name = args.name.unwrap_or_else(|| source_id.clone());
    let tile_url = args.tile_url.unwrap_or_else(|| {
        format!("http://localhost:3000/{source_id}/{{z}}/{{x}}/{{y}}")
    });

    // Write style.json
    let style_path = args
        .style_output
        .unwrap_or_else(|| args.output.with_extension("style.json"));
    let style_filename = style_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("style.json")
        .to_owned();
    let style_json =
        style::build_style(args.safety_depth, args.shoal_depth, &tile_url, min_zoom, out_max_zoom);
    std::fs::write(&style_path, &style_json)
        .with_context(|| format!("writing style to {}", style_path.display()))?;
    info!(path = %style_path.display(), "style written");

    // Write metadata.json (Signal K charts plugin format)
    let meta_path = args.output.with_extension("metadata.json");
    let metadata = serde_json::json!({
        "id":          source_id,
        "name":        chart_name,
        "description": "OESU chart converted by oesu2geojson",
        "type":        "mapstyleJSON",
        "format":      "pbf",
        "created":     chrono_now(),
        "minZoom":     min_zoom,
        "maxZoom":     out_max_zoom,
        "bounds":      bounds,
        "tilemapUrl":  tile_url,
        "styleUrl":    style_filename,
    });
    std::fs::write(&meta_path, serde_json::to_string_pretty(&metadata)?)
        .with_context(|| format!("writing metadata to {}", meta_path.display()))?;
    info!(path = %meta_path.display(), "metadata written");

    Ok(())
}
