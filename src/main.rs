mod convert;
mod georef;
mod osenc;
mod s57;
mod style;

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

/// Convert decrypted OSENC (.oesu) chart files to `GeoJSON` + `MapLibre` style JSON.
///
/// Outputs one .geojson file per S-57 object class into <outdir>,
/// plus a style.json suitable for use as a mapstyleJSON chart in Signal K.
///
/// Feed the `GeoJSON` files to tippecanoe to produce vector tiles:
///   tippecanoe -o chart.mbtiles --no-tile-compression \
///     -l DEPARE depare.geojson -l LNDARE lndare.geojson ...
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Input .oesu file or directory containing .oesu files
    input: PathBuf,

    /// Output directory (created if it doesn't exist)
    #[arg(short, long, default_value = "out")]
    outdir: PathBuf,

    /// Tile URL template for the style.json source
    /// (use {z}/{x}/{y} placeholders)
    #[arg(long, default_value = "{z}/{x}/{y}.pbf")]
    tile_url: String,

    /// Minimum zoom level for the style source
    #[arg(long, default_value_t = 7)]
    min_zoom: u8,

    /// Maximum zoom level for the style source
    #[arg(long, default_value_t = 14)]
    max_zoom: u8,
}

fn main() -> Result<()> {
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

    // Accumulate all layers across files into a single map
    let mut all_layers: std::collections::HashMap<String, Vec<geojson::Feature>> =
        std::collections::HashMap::new();

    let mut combined_bounds = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];

    for path in &inputs {
        let data =
            fs::read(path).with_context(|| format!("reading {}", path.display()))?;

        let cell = match osenc::parse_file(&data) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Warning: skipping {}: {e}", path.display());
                continue;
            }
        };

        eprintln!(
            "Parsed {}: {} features, scale 1:{}, ref ({:.4}, {:.4})",
            path.file_name().unwrap_or_default().display(),
            cell.features.len(),
            cell.native_scale,
            cell.ref_lat,
            cell.ref_lon,
        );

        // Expand combined bounds
        combined_bounds[0] = combined_bounds[0].min(cell.bounds[0]);
        combined_bounds[1] = combined_bounds[1].min(cell.bounds[1]);
        combined_bounds[2] = combined_bounds[2].max(cell.bounds[2]);
        combined_bounds[3] = combined_bounds[3].max(cell.bounds[3]);

        let layers = convert::cell_to_geojson(&cell);
        for (acronym, fc) in layers {
            all_layers.entry(acronym).or_default().extend(fc.features);
        }
    }

    // Write one GeoJSON file per layer
    let mut written_layers: Vec<String> = Vec::new();
    for (acronym, features) in &all_layers {
        let fc = geojson::FeatureCollection {
            bbox: None,
            features: features.clone(),
            foreign_members: None,
        };
        let outpath = args.outdir.join(format!("{acronym}.geojson"));
        let json = serde_json::to_string(&fc)?;
        fs::write(&outpath, json)
            .with_context(|| format!("writing {}", outpath.display()))?;
        written_layers.push(acronym.clone());
    }
    written_layers.sort();

    eprintln!(
        "Wrote {} layer files to {}",
        written_layers.len(),
        args.outdir.display()
    );

    // Write style.json
    let style =
        style::generate_style(&args.tile_url, combined_bounds, args.min_zoom, args.max_zoom);
    let style_path = args.outdir.join("style.json");
    fs::write(&style_path, serde_json::to_string_pretty(&style)?)
        .with_context(|| format!("writing {}", style_path.display()))?;
    eprintln!("Wrote style.json");

    // Print tippecanoe command hint
    eprintln!("\nNext step — generate tiles with tippecanoe:");
    let layer_args: Vec<String> = written_layers
        .iter()
        .map(|l| format!("  -l {l} {l}.geojson"))
        .collect();
    eprintln!(
        "cd {} && tippecanoe -o chart.mbtiles --no-tile-compression \\",
        args.outdir.display()
    );
    for arg in &layer_args {
        eprintln!("{arg} \\");
    }
    eprintln!("  --minimum-zoom={} --maximum-zoom={}", args.min_zoom, args.max_zoom);

    Ok(())
}
