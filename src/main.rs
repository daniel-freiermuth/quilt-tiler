//! Convert decrypted OESU chart files to a `PMTiles` vector tile archive.
//!
//! Parses all input `.oesu` files in parallel, then writes MVT tiles directly
//! into a `PMTiles` v3 archive.

mod s57;
mod tiles;
mod zoom;

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use rayon::prelude::*;
use tracing::{info, warn};

use zoom::zoom_from_scale;

/// Convert decrypted OESU chart files to a `PMTiles` vector tile archive.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Decrypted .oesu files to convert (glob or explicit paths).
    #[arg(required = true)]
    input: Vec<PathBuf>,

    /// Output `PMTiles` file.
    #[arg(short, long, default_value = "chart.pmtiles")]
    output: PathBuf,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();

    // Parse all input files in parallel; skip files that fail to parse.
    let cells: Vec<oesu::OesuCell> = args
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
    tiles::write_pmtiles(&cells, &args.output)?;

    Ok(())
}
