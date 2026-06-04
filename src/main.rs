mod convert;
mod georef;
mod osenc;
mod s57;
mod style;

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::{debug, info, warn};

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

}

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
                warn!(path = %path.display(), "skipping: {e}");
                continue;
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
        debug!(path = %outpath.display(), "writing layer");
        fs::write(&outpath, json)
            .with_context(|| format!("writing {}", outpath.display()))?;
        written_layers.push(acronym.clone());
    }
    written_layers.sort();

    info!(layers = written_layers.len(), dir = %args.outdir.display(), "wrote GeoJSON layers");

    // Write style.json
    let style_path = args.outdir.join("style.json");
    fs::write(&style_path, style::STYLE_JSON)
        .with_context(|| format!("writing {}", style_path.display()))?;
    info!(path = %style_path.display(), "wrote style.json");

    // Print tippecanoe command hint
    info!("next step: generate tiles with tippecanoe");
    let layer_args = written_layers
        .iter()
        .map(|l| format!("-l {l} {l}.geojson"))
        .collect::<Vec<_>>()
        .join(" \\\n  ");
    info!(
        "tippecanoe command:\n  cd {} && tippecanoe -o chart.mbtiles --no-tile-compression \\\n  {layer_args} \\\n  --minimum-zoom=7 --maximum-zoom=14",
        args.outdir.display(),
    );

    Ok(())
}
