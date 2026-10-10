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

/// Read a cell file from disk, logging a warning and returning `None` on I/O
/// failure so the caller can skip the file without aborting the batch.
fn read_cell_data(path: &Path) -> Option<Vec<u8>> {
    std::fs::read(path)
        .inspect_err(|e| warn!(file = %path.display(), error = %e, "cannot read"))
        .ok()
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
            let data = read_cell_data(path)?;
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
            let data = read_cell_data(path)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A per-test scratch directory under the system temp dir, removed on
    /// drop so a failing assertion does not leak files.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(test: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("quilt-tiler-loader-{}-{test}", std::process::id()));
            // A leftover from an earlier crashed run with the same pid.
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("creating scratch dir");
            Self(dir)
        }

        fn write(&self, name: &str, data: &[u8]) -> PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, data).expect("writing scratch file");
            path
        }

        fn missing(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn record(rec_type: u16, payload: &[u8]) -> Vec<u8> {
        let len = u32::try_from(payload.len() + 6).expect("tiny test record");
        let mut out = rec_type.to_le_bytes().to_vec();
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// Minimal decrypted SENC stream `oesu::parse_file` accepts: server
    /// status (decrypted, licensed), version 201, cell name, native scale.
    fn minimal_oesu(name: &str, scale: u32) -> Vec<u8> {
        let status: Vec<u8> = [1u16, 1, 1, 30, 0, 0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut cstr = name.as_bytes().to_vec();
        cstr.push(0);
        let mut out = record(200, &status);
        out.extend(record(1, &201u16.to_le_bytes()));
        out.extend(record(2, &cstr));
        out.extend(record(7, &scale.to_le_bytes()));
        out
    }

    /// Minimal `.rnc` `RncCell::parse` accepts: a 1×1 grid whose offset
    /// table points at zero-length tiles, followed by the JSON footer.
    fn minimal_rnc(scale: f64) -> Vec<u8> {
        let mut out = vec![0u8; 8];
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        // 1 tile + 2 trailing slots, all pointing at the end of the table.
        for _ in 0..3 {
            out.extend_from_slice(&28u32.to_le_bytes());
        }
        let footer = serde_json::json!({
            "cover": [],
            "lat0": 57.0, "lat1": 58.0, "lon0": 11.0, "lon1": 12.0,
            "edate": "01/01/2026", "name": "TEST", "scale": scale,
        });
        out.extend_from_slice(footer.to_string().as_bytes());
        out
    }

    #[test]
    fn read_cell_data_returns_exact_file_contents() {
        let dir = Scratch::new("read-ok");
        let path = dir.write("cell.bin", b"cell bytes");
        assert_eq!(read_cell_data(&path).as_deref(), Some(&b"cell bytes"[..]));
    }

    #[test]
    fn read_cell_data_returns_none_for_unreadable_path() {
        let dir = Scratch::new("read-missing");
        assert_eq!(read_cell_data(&dir.missing("absent.bin")), None);
    }

    #[test]
    fn load_s57_cells_keeps_valid_cells_and_skips_bad_files() {
        let dir = Scratch::new("s57");
        let paths = [
            dir.write("good.oesu", &minimal_oesu("GOODCELL", 50_000)),
            dir.missing("absent.oesu"),
            dir.write("garbage.oesu", b"not a senc stream"),
        ];
        let cells = load_s57_cells(&paths, 0.0);
        assert_eq!(cells.len(), 1, "only the valid cell survives");
        assert_eq!(cells[0].name, "GOODCELL");
        assert_eq!(cells[0].native_scale, 50_000);
    }

    #[test]
    fn load_rnc_cells_keeps_valid_cells_and_skips_bad_files() {
        let dir = Scratch::new("rnc");
        let paths = [
            dir.write("good.rnc", &minimal_rnc(250_000.0)),
            dir.missing("absent.rnc"),
            dir.write("garbage.rnc", b"not an rnc file"),
        ];
        let cells = load_rnc_cells(&paths, 0.0);
        assert_eq!(cells.len(), 1, "only the valid cell survives");
        assert_eq!(cells[0].name(), "good", "name comes from the file stem");
        assert_eq!(cells[0].native_scale(), 250_000);
    }
}
