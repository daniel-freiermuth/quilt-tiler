//! Parallel chart-cell loading shared by the batch CLI (`src/main.rs`) and
//! the live tile server (`src/bin/tileserver.rs`).
//!
//! Both loaders read+parse every input file in parallel and skip — log and
//! drop, not fail — any file that can't be read or parsed, so one bad cell
//! in a directory of charts never aborts the whole run.

use std::path::Path;

use rayon::prelude::*;
use tracing::{debug, warn};

use crate::rnc::RncCell;
use crate::zoom::zoom_from_scale;

/// `true` if `path`'s extension is `.rnc` (a raster cell), case-insensitive.
///
/// Used to dispatch input files between [`load_s57_cells`] and
/// [`load_rnc_cells`] — vector and raster cells are never mixed in one run.
#[must_use]
pub fn is_rnc(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("rnc"))
}

/// Parse all `.oesu`/`.osenc` vector cell files in `paths` in parallel.
pub fn load_s57_cells(paths: &[impl AsRef<Path> + Sync], zoom_offset: f64) -> Vec<s57::S57Cell> {
    paths
        .par_iter()
        .filter_map(|path| {
            let path = path.as_ref();
            profiling::scope!("parse");
            #[cfg(feature = "profiling")]
            let _frame = tracy_client::non_continuous_frame!("parse");
            let data = match std::fs::read(path) {
                Ok(d) => d,
                Err(e) => {
                    warn!(file = %path.display(), error = %e, "cannot read");
                    return None;
                }
            };
            match oesu::parse_file(path.to_str().unwrap_or_default().to_owned(), &data) {
                Ok(cell) => {
                    let z = zoom_from_scale(cell.native_scale, zoom_offset);
                    debug!(
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
        .collect()
}

/// Parse all `.rnc` raster cell files in `paths` in parallel.
pub fn load_rnc_cells(paths: &[impl AsRef<Path> + Sync], zoom_offset: f64) -> Vec<RncCell> {
    paths
        .par_iter()
        .filter_map(|path| {
            let path = path.as_ref();
            profiling::scope!("parse");
            #[cfg(feature = "profiling")]
            let _frame = tracy_client::non_continuous_frame!("parse");
            let data = match std::fs::read(path) {
                Ok(d) => d,
                Err(e) => {
                    warn!(file = %path.display(), error = %e, "cannot read");
                    return None;
                }
            };
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("cell")
                .to_owned();
            match RncCell::parse(name.clone(), data) {
                Ok(cell) => {
                    let z = zoom_from_scale(RncCell::native_scale(&cell), zoom_offset);
                    debug!(
                        name = %name,
                        scale = RncCell::native_scale(&cell),
                        zoom = z,
                        cols = cell.cols(),
                        rows = cell.rows(),
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
        .collect()
}
