//! Parallel chart-cell loading shared by the batch CLI (`src/main.rs`) and
//! the live tile server (`src/bin/tileserver.rs`).
//!
//! [`load_cells`] reads every input exactly once — so pipes and process
//! substitutions work — recognises its format by content
//! ([`Format::detect`]), not its name, and parses it. Any file that can't be
//! read, recognised or parsed is skipped — logged and dropped, not fatal —
//! so one bad cell in a directory of charts never aborts the whole run.

use std::path::Path;

use anyhow::{Result, bail};
use rayon::prelude::*;
use tracing::{debug, warn};

use crate::rnc::RncCell;
use crate::zoom::zoom_from_scale;

/// A chart cell file format, recognised by content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Decrypted SENC vector cells (`.oesu`/`.osenc`).
    Senc,
    /// `.djson` JSON vector cells.
    Djson,
    /// `.rnc` raster cells.
    Rnc,
}

impl Format {
    /// The format `data` is in; `None` when no supported format claims it.
    #[must_use]
    pub fn detect(data: &[u8]) -> Option<Self> {
        if rnc_format::detect(data) {
            Some(Self::Rnc)
        } else if djson_format::detect(data) {
            Some(Self::Djson)
        } else if oesu::detect(data) {
            Some(Self::Senc)
        } else {
            None
        }
    }
}

/// A run's loaded cells. One `PMTiles` archive (and one server route) holds
/// one tile type, so a run is all vector or all raster.
pub enum Cells {
    Vector(Vec<s57::S57Cell>),
    Raster(Vec<RncCell>),
}

/// One loaded input.
enum Cell {
    Vector(s57::S57Cell),
    Raster(RncCell),
}

/// Read, recognise and parse every input in parallel.
///
/// # Errors
/// When no input yields a cell, or raster and vector cells are mixed. A mix
/// is only noticed once every input is parsed: telling them apart needs
/// their bytes, and a pipe can't be read twice.
pub fn load_cells(paths: &[impl AsRef<Path> + Sync], zoom_offset: f64) -> Result<Cells> {
    let loaded: Vec<Cell> = paths
        .par_iter()
        .filter_map(|path| load_cell(path.as_ref(), zoom_offset))
        .collect();
    let (mut vector, mut raster) = (Vec::new(), Vec::new());
    for cell in loaded {
        match cell {
            Cell::Vector(c) => vector.push(c),
            Cell::Raster(c) => raster.push(c),
        }
    }
    match (vector.is_empty(), raster.is_empty()) {
        (true, true) => bail!("no input could be loaded as a chart cell"),
        (false, true) => Ok(Cells::Vector(vector)),
        (true, false) => Ok(Cells::Raster(raster)),
        (false, false) => bail!(
            "cannot mix raster and vector inputs in one run ({} raster, {} vector); \
             run them separately",
            raster.len(),
            vector.len()
        ),
    }
}

/// Read the cell at `path` once and parse it by its detected [`Format`];
/// logs and returns `None` when it can't be read, recognised or parsed.
fn load_cell(path: &Path, zoom_offset: f64) -> Option<Cell> {
    profiling::scope!("parse");
    #[cfg(feature = "profiling")]
    let _frame = tracy_client::non_continuous_frame!("parse");
    let data = read_cell_data(path)?;
    match Format::detect(&data) {
        Some(Format::Senc) => {
            let source = path.to_string_lossy().into_owned();
            let cell = oesu::parse_file(source, &data)
                .inspect_err(|e| warn_skipped_s57(path, e))
                .ok()?;
            Some(Cell::Vector(parsed_s57(cell, zoom_offset)))
        }
        Some(Format::Djson) => {
            let source = path.to_string_lossy().into_owned();
            let cell = djson_format::parse(source, &data)
                .and_then(s57::S57Cell::try_from)
                .inspect_err(|e| warn!(file = %path.display(), error = %e, "skipping"))
                .ok()?;
            Some(Cell::Vector(parsed_s57(cell, zoom_offset)))
        }
        Some(Format::Rnc) => {
            let cell = RncCell::parse(path.to_string_lossy().into_owned(), data)
                .inspect_err(|e| warn!(file = %path.display(), error = %e, "skipping"))
                .ok()?;
            debug!(
                source = %cell.source(),
                scale = cell.native_scale(),
                zoom = zoom_from_scale(cell.native_scale(), zoom_offset),
                cols = cell.cols(),
                rows = cell.rows(),
                "parsed"
            );
            Some(Cell::Raster(cell))
        }
        None => {
            warn_unrecognised(path, &data);
            None
        }
    }
}

/// Log why `path` was not recognised. A file *named* like a SENC cell gets
/// the SENC parser's own diagnosis — typically that it is still encrypted,
/// the usual mistake — rather than a generic skip. The name only picks the
/// message; content alone decides how a file is parsed.
fn warn_unrecognised(path: &Path, data: &[u8]) {
    let senc_named = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("oesu") || e.eq_ignore_ascii_case("osenc"));
    if senc_named && let Err(e) = oesu::parse_file(path.display().to_string(), data) {
        warn_skipped_s57(path, &e);
    } else {
        warn!(file = %path.display(), "not a recognised chart cell, skipping");
    }
}

/// Log a parsed vector cell and pass it on.
fn parsed_s57(cell: s57::S57Cell, zoom_offset: f64) -> s57::S57Cell {
    debug!(
        name = %cell.name,
        scale = cell.native_scale,
        zoom = zoom_from_scale(cell.native_scale, zoom_offset),
        features = cell.features.len(),
        "parsed"
    );
    cell
}

/// Read a cell file, logging a warning and returning `None` on I/O failure
/// so the caller can skip the file without aborting the batch.
fn read_cell_data(path: &Path) -> Option<Vec<u8>> {
    std::fs::read(path)
        .inspect_err(|e| warn!(file = %path.display(), error = %e, "cannot read"))
        .ok()
}

/// Log why a vector cell was skipped, separating refusals (an intact chart
/// the user must act on: expired licence, bad key, raster, unsupported
/// version) from corrupt or truncated data.
fn warn_skipped_s57(path: &Path, e: &oesu::OesuError) {
    if e.is_refusal() {
        warn!(file = %path.display(), error = %e, "chart refused, skipping");
    } else {
        warn!(file = %path.display(), error = %e, "skipping");
    }
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
    /// status (decrypted, licensed), version 201, cell name, native scale,
    /// cell extent.
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
        // (lat, lon) corners in SW, NW, NE, SE order.
        let extent: Vec<u8> = [57.0, 11.0, 58.0, 11.0, 58.0, 12.0, 57.0, 12.0]
            .iter()
            .flat_map(|v: &f64| v.to_le_bytes())
            .collect();
        out.extend(record(100, &extent));
        out
    }

    /// Minimal `.rnc` `RncCell::parse` accepts: a 1×1 grid whose offset
    /// table points at zero-length tiles, followed by the JSON footer.
    fn minimal_rnc(scale: f64) -> Vec<u8> {
        let mut out = rnc_format::SIGNATURE.to_vec();
        out.push(4); // format version
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

    /// Minimal `.djson` cell with a name, bounds, scale and one depth area.
    fn minimal_djson(name: &str, scale: u32) -> Vec<u8> {
        let m = |lon, lat| <[f64; 2]>::from(rnc_format::wgs84_to_rnc_merc(lon, lat));
        let ([xmin, y_north], [xmax, y_south]) = (m(11.0, 58.0), m(12.0, 57.0));
        let ring: Vec<f64> = [(11.0, 57.0), (12.0, 57.0), (12.0, 58.0), (11.0, 58.0)]
            .into_iter()
            .flat_map(|(lon, lat)| m(lon, lat))
            .collect();
        serde_json::to_vec(&serde_json::json!({
            "__system__": { "mima": [xmin, y_north, xmax, y_south, scale, scale], "name": name },
            "A_DEPARE": [{ "_XY": ring, "DRVAL1": 0, "DRVAL2": 5 }],
        }))
        .unwrap()
    }

    #[test]
    fn format_is_detected_from_content_not_name() {
        assert_eq!(
            Format::detect(&minimal_oesu("C", 50_000)),
            Some(Format::Senc)
        );
        assert_eq!(
            Format::detect(&minimal_djson("C", 50_000)),
            Some(Format::Djson)
        );
        assert_eq!(Format::detect(&minimal_rnc(50_000.0)), Some(Format::Rnc));
        assert_eq!(Format::detect(b"not a chart"), None);

        let dir = Scratch::new("load-names");
        let paths = [
            dir.write("a.rnc", &minimal_djson("A", 50_000)),
            dir.write("b.dat", &minimal_oesu("B", 50_000)),
            dir.write("notes.txt", b"not a chart"),
            dir.missing("absent.djson"),
        ];
        let names: Vec<_> = vector(load_cells(&paths, 0.0))
            .into_iter()
            .map(|c| c.name)
            .collect();
        assert_eq!(
            names,
            ["A", "B"],
            "unreadable and unknown files are left out"
        );
    }

    fn vector(cells: Result<Cells>) -> Vec<s57::S57Cell> {
        match cells.expect("cells load") {
            Cells::Vector(cells) => cells,
            Cells::Raster(_) => panic!("expected vector cells"),
        }
    }

    fn raster(cells: Result<Cells>) -> Vec<RncCell> {
        match cells.expect("cells load") {
            Cells::Raster(cells) => cells,
            Cells::Vector(_) => panic!("expected raster cells"),
        }
    }

    fn load_error(paths: &[PathBuf]) -> String {
        load_cells(paths, 0.0)
            .err()
            .expect("loading fails")
            .to_string()
    }

    #[test]
    fn load_cells_refuses_mixed_runs_and_runs_with_nothing_loaded() {
        let dir = Scratch::new("load-errors");
        let raster = dir.write("r", &minimal_rnc(50_000.0));
        let vector = dir.write("v", &minimal_djson("V", 50_000));
        let mixed = load_error(&[raster, vector]);
        assert!(mixed.contains("cannot mix raster and vector"), "{mixed}");
        let none = load_error(&[
            dir.write("x", b"not a chart"),
            dir.write("y.oesu", &minimal_oesu("Y", 50_000)[..20]),
        ]);
        assert!(
            none.contains("no input could be loaded as a chart cell"),
            "{none}"
        );
    }

    /// A pipe can be read only once: detecting the format must not consume
    /// bytes the parser then misses.
    #[cfg(unix)]
    #[test]
    fn load_cells_reads_a_pipe_once() {
        let dir = Scratch::new("fifo");
        let fifo = dir.missing("cell");
        let made = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .expect("running mkfifo");
        assert!(made.success());
        let writer = {
            let fifo = fifo.clone();
            std::thread::spawn(move || std::fs::write(fifo, minimal_oesu("PIPED", 50_000)))
        };
        // A second open of the pipe would block forever; bound the wait.
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let names: Vec<_> = vector(load_cells(&[fifo], 0.0))
                .into_iter()
                .map(|c| c.name)
                .collect();
            tx.send(names).expect("sending result");
        });
        let names = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("the pipe is read once and loading finishes");
        assert_eq!(names, ["PIPED"]);
        writer.join().unwrap().expect("writing the pipe");
    }

    #[test]
    fn load_cells_parses_djson_whatever_its_name() {
        let dir = Scratch::new("djson");
        let cells = vector(load_cells(
            &[dir.write("cell.dat", &minimal_djson("JCELL", 25_000))],
            0.0,
        ));
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].name, "JCELL");
        assert_eq!(cells[0].native_scale, 25_000);
        assert_eq!(
            s57::object_acronym(cells[0].features[0].type_code),
            Some("DEPARE")
        );
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
    fn load_cells_keeps_valid_vector_cells_and_skips_bad_files() {
        let dir = Scratch::new("s57");
        let paths = [
            dir.write("good.oesu", &minimal_oesu("GOODCELL", 50_000)),
            dir.missing("absent.oesu"),
            dir.write("garbage.oesu", b"not a senc stream"),
        ];
        let cells = vector(load_cells(&paths, 0.0));
        assert_eq!(cells.len(), 1, "only the valid cell survives");
        assert_eq!(cells[0].name, "GOODCELL");
        assert_eq!(cells[0].native_scale, 50_000);
    }

    /// Captures formatted log output written on the current thread.
    #[derive(Clone, Default)]
    struct LogBuf(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for LogBuf {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .expect("log buffer lock")
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogBuf {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    fn log_of(f: impl FnOnce()) -> String {
        let buf = LogBuf::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(buf.clone())
            .with_ansi(false)
            .finish();
        tracing::subscriber::with_default(subscriber, f);
        let bytes = buf.0.lock().expect("log buffer lock").clone();
        String::from_utf8(bytes).expect("utf-8 log output")
    }

    fn skip_log(e: &oesu::OesuError) -> String {
        log_of(|| warn_skipped_s57(Path::new("cell.oesu"), e))
    }

    #[test]
    fn skipped_s57_cells_log_refusals_apart_from_corruption() {
        let refused = skip_log(&oesu::OesuError::UnsupportedRaster);
        assert!(refused.contains("chart refused, skipping"), "{refused}");

        let corrupt = skip_log(&oesu::OesuError::MissingNativeScale);
        assert!(corrupt.contains("skipping"), "{corrupt}");
        assert!(!corrupt.contains("chart refused"), "{corrupt}");
    }

    /// Still-encrypted SENC is the usual mistake: a file named like one keeps
    /// the parser's hint instead of a generic skip.
    #[test]
    fn unrecognised_senc_named_files_get_the_senc_diagnosis() {
        let encrypted = [0x5a_u8, 0x13, 0xc7, 0x02, 0x99, 0x41, 0x00, 0x7e];
        for name in ["cell.oesu", "CELL.OSENC"] {
            let log = log_of(|| warn_unrecognised(Path::new(name), &encrypted));
            assert!(log.contains("still encrypted"), "{name}: {log}");
        }
        let other = log_of(|| warn_unrecognised(Path::new("notes.txt"), &encrypted));
        assert!(other.contains("not a recognised chart cell"), "{other}");
        assert!(!other.contains("encrypted"), "{other}");
    }

    #[test]
    fn load_cells_keeps_valid_raster_cells_and_skips_bad_files() {
        let dir = Scratch::new("rnc");
        let paths = [
            dir.write("good.rnc", &minimal_rnc(250_000.0)),
            dir.missing("absent.rnc"),
            dir.write("garbage.rnc", b"not an rnc file"),
        ];
        let cells = raster(load_cells(&paths, 0.0));
        assert_eq!(cells.len(), 1, "only the valid cell survives");
        assert_eq!(
            cells[0].source(),
            paths[0].to_string_lossy(),
            "the path identifies it"
        );
        assert_eq!(cells[0].native_scale(), 250_000);
    }
}
