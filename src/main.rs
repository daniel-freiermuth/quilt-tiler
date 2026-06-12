mod convert;
mod s57;
mod style;

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use tracing::{debug, info, warn};

/// Returns the current UTC time as an ISO 8601 string (seconds precision).
fn chrono_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Format: YYYY-MM-DDTHH:MM:SSZ
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let days = secs / 86400; // days since 1970-01-01
    // Compute calendar date from days
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    // Gregorian calendar calculation
    let mut year = 1970u64;
    loop {
        let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
        let days_in_year = if leap { 366 } else { 365 };
        if days < days_in_year { break; }
        days -= days_in_year;
        year += 1;
    }
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let month_days = [31u64, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 1u64;
    for &md in &month_days {
        if days < md { break; }
        days -= md;
        month += 1;
    }
    (year, month, days + 1)
}

/// Compute the ideal tile zoom level for a chart at a given native scale (1:N).
///
/// Formula: `Z = floor(log2(earth_circumference / (256 * 0.00028 * N)))`
/// where `0.00028` m/px is a standard screen pixel size at 90 DPI.
/// Combined constant: `40_075_016 / (256 * 0.00028) ≈ 559_082_264`.
///
/// Result is clamped to `[0, 22]`.
const ZOOM_K: f64 = 559_082_264.0;
fn zoom_from_scale(native_scale: u32) -> u8 {
    if native_scale == 0 {
        return 14;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    // Safety: value is clamped to [0.0, 22.0] before cast
    let z = (ZOOM_K / f64::from(native_scale)).log2().floor().clamp(0.0, 22.0) as u8;
    z
}


/// Convert decrypted OSENC (.oesu) chart files to `GeoJSON` + `MapLibre` style JSON.
///
/// Outputs one .geojson file per S-57 object class into <outdir>,
/// plus a style.json suitable for use as a mapstyleJSON chart in Signal K.
///
/// Feed the `GeoJSON` files to tippecanoe to produce vector tiles:
///   cd <outdir> && tippecanoe -o chart.mbtiles --no-tile-compression \
///     --drop-densest-as-needed --minimum-zoom=N --maximum-zoom=N *.geojson
///
/// tippecanoe automatically uses the filename (without .geojson) as the
/// layer name, so no `-l` flags are needed.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Input .oesu file or directory containing .oesu files
    input: PathBuf,

    /// Output directory (created if it doesn't exist)
    #[arg(short, long, default_value = "out")]
    outdir: PathBuf,

    /// Human-readable chart name (defaults to output directory name)
    #[arg(long)]
    name: Option<String>,

    /// Safety depth in metres: the DEPCNT contour at exactly this depth is
    /// highlighted red, and DEPARE areas shallower than this are drawn in
    /// dark blue. Set this to your vessel's maximum draft plus a safety margin.
    #[arg(long, default_value_t = 3.0)]
    safety_depth: f64,

    /// Upper boundary of the shallow-but-navigable zone in metres. DEPARE
    /// areas between safety-depth and shoal-depth get a medium blue; deeper
    /// areas get a very light blue.
    #[arg(long, default_value_t = 10.0)]
    shoal_depth: f64,

    /// Full tile URL template that the style source should point to,
    /// e.g. `http://localhost:3000/chart/{z}/{x}/{y}`.
    /// This is embedded in the `sources.enc` block of `style.json` so `MapLibre`
    /// can find the tiles.  Use `{z}`, `{x}`, `{y}` as placeholders.
    /// If omitted the source block is written with a placeholder so you can
    /// edit it manually before serving.
    #[arg(long, default_value = "http://localhost:3000/chart/{z}/{x}/{y}")]
    tile_url: String,

}

type LayerMap = std::collections::HashMap<String, Vec<geojson::Feature>>;

#[allow(clippy::too_many_lines)] // CLI entry point: parsing + I/O + hints
fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();

    let inputs: Vec<PathBuf> = if args.input.is_dir() {
        fs::read_dir(&args.input)?
            .filter_map(std::result::Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "oesu"))
            .collect()
    } else {
        vec![args.input.clone()]
    };

    if inputs.is_empty() {
        anyhow::bail!("No .oesu files found in {}", args.input.display());
    }

    fs::create_dir_all(&args.outdir)
        .with_context(|| format!("creating output dir {}", args.outdir.display()))?;

    // Parse and convert all files in parallel, then merge sequentially.
    let parsed: Vec<([f64; 4], u32, LayerMap)> = inputs
        .par_iter()
        .filter_map(|path| {
            let data = match fs::read(path) {
                Ok(d) => d,
                Err(e) => {
                    warn!(path = %path.display(), "skipping: {e}");
                    return None;
                }
            };
            let cell = match oesu::parse_file(&data) {
                Ok(c) => c,
                Err(e) => {
                    warn!(path = %path.display(), "skipping: {e}");
                    return None;
                }
            };
            info!(
                file = %path.file_name().unwrap_or_default().display(),
                features = cell.features.len(),
                scale = cell.native_scale,
                ref_lat = format_args!("{:.4}", cell.ref_lat),
                ref_lon = format_args!("{:.4}", cell.ref_lon),
                "parsed",
            );
            let bounds = cell.bounds;
            let native_scale = cell.native_scale;
            let layer_map: LayerMap = convert::cell_to_geojson(&cell)
                .into_iter()
                .map(|(k, fc)| (k, fc.features))
                .collect();
            Some((bounds, native_scale, layer_map))
        })
        .collect();

    // Sequential merge
    let mut all_layers: LayerMap = std::collections::HashMap::new();
    let mut combined_bounds = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    let mut min_native_scale = u32::MAX; // most detailed cell
    let mut max_native_scale = 0u32;     // coarsest cell
    for (bounds, native_scale, layers) in parsed {
        min_native_scale = min_native_scale.min(native_scale);
        max_native_scale = max_native_scale.max(native_scale);
        combined_bounds[0] = combined_bounds[0].min(bounds[0]);
        combined_bounds[1] = combined_bounds[1].min(bounds[1]);
        combined_bounds[2] = combined_bounds[2].max(bounds[2]);
        combined_bounds[3] = combined_bounds[3].max(bounds[3]);
        for (acronym, features) in layers {
            all_layers.entry(acronym).or_default().extend(features);
        }
    }

    // Serialize and write one GeoJSON file per layer, in parallel.
    let mut written_layers: Vec<String> = all_layers
        .par_iter()
        .map(|(acronym, features)| {
            let fc = geojson::FeatureCollection {
                bbox: None,
                features: features.clone(),
                foreign_members: None,
            };
            let outpath = args.outdir.join(format!("{acronym}.geojson"));
            let json = serde_json::to_string(&fc)?;
            debug!(path = %outpath.display(), "writing layer");
            fs::write(&outpath, &json)
                .with_context(|| format!("writing {}", outpath.display()))?;
            Ok(acronym.clone())
        })
        .collect::<Result<Vec<_>>>()?;
    written_layers.sort();

    // Compute zoom levels from the actual native scales observed across all cells.
    // max_zoom: ideal zoom for the most detailed cell (smallest native_scale).
    // min_zoom: two levels below the coarsest cell's ideal zoom, floored at 0.
    let max_zoom = zoom_from_scale(if min_native_scale == u32::MAX { 22_000 } else { min_native_scale });
    let min_zoom = zoom_from_scale(max_native_scale).saturating_sub(2);
    info!(min_zoom, max_zoom, min_scale = min_native_scale, max_scale = max_native_scale, "zoom range computed from native scales");

    info!(layers = written_layers.len(), dir = %args.outdir.display(), "wrote GeoJSON layers");

    // Write style.json
    let style_path = args.outdir.join("style.json");
    fs::write(
        &style_path,
        style::build_style(args.safety_depth, args.shoal_depth, &args.tile_url, min_zoom, max_zoom),
    )
        .with_context(|| format!("writing {}", style_path.display()))?;
    info!(path = %style_path.display(), "wrote style.json");

    // Derive chart id / name from the output directory
    let chart_id = args
        .outdir
        .file_name().map_or_else(|| "enc-chart".to_owned(), |n| n.to_string_lossy().into_owned());
    let chart_name = args.name.as_deref().unwrap_or(&chart_id).to_owned();

    // Write metadata.json (Signal K charts plugin format)
    let metadata = serde_json::json!({
        "id":          chart_id,
        "name":        chart_name,
        "description": "OESU chart converted by oesu2geojson",
        "type":        "mapstyleJSON",
        "format":      "pbf",
        "created":     chrono_now(),
        "minZoom":     min_zoom,
        "maxZoom":     max_zoom,
        "bounds":      combined_bounds,
        "tilemapUrl":  "{z}/{x}/{y}.pbf",
        "styleUrl":    "style.json"
    });
    let meta_path = args.outdir.join("metadata.json");
    fs::write(&meta_path, serde_json::to_string_pretty(&metadata)?)
        .with_context(|| format!("writing {}", meta_path.display()))?;
    info!(path = %meta_path.display(), "wrote metadata.json");

    // Print tippecanoe command hint
    info!("next step: generate tiles with tippecanoe");
    let out = args.outdir.display();
    info!(
        "\n\
── Step 1: generate tiles ──────────────────────────────────────────\n\
  cd {out} && tippecanoe -o chart.mbtiles --no-tile-compression --drop-densest-as-needed \\\n\
  --minimum-zoom={min_zoom} --maximum-zoom={max_zoom} *.geojson\n\
  # tippecanoe uses the filename as the layer name automatically\n\
\n\
── Step 2: serve the tiles (pick one) ──────────────────────────────\n\
\n\
  Option A — martin tile server (serves .mbtiles directly):\n\
    cargo install martin\n\
    martin {out}/chart.mbtiles\n\
    # tiles at  http://localhost:3000/chart/{{z}}/{{x}}/{{y}}\n\
\n\
  Option B — tileserver-gl (serves .mbtiles directly):\n\
    docker run --rm -v {out}:/data -p 8080:8080 maptiler/tileserver-gl\n\
\n\
  Option C — .pmtiles (single file, HTTP range-request servable):\n\
    pmtiles convert {out}/chart.mbtiles {out}/chart.pmtiles\n\
    # serve statically via any HTTP server with range support\n\
\n\
  Option D — flat {{}}/{{z}}/{{x}}/{{y}}.pbf directory tree:\n\
    mb-util --image-format=pbf {out}/chart.mbtiles {out}/tiles\n\
\n\
── Step 3: Signal K integration ────────────────────────────────────\n\
  The output dir already contains metadata.json and style.json.\n\
  Add the output directory as a chart path in the Signal K charts plugin.\n\
  Set the tile URL in the plugin to point at whichever server you chose.\n\
  The style.json is served from the same directory and referenced\n\
  by metadata.json as \"styleUrl\": \"style.json\"."
    );

    Ok(())
}
