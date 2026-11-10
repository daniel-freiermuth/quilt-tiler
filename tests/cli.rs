//! End-to-end tests of the `quilt-tiler` batch binary: run it on fixture
//! charts and check the `PMTiles` archive, style and metadata it writes.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::{Box4, COARSE_BOX, COARSE_ZOOM, FINE_BOX, FINE_ZOOM, Scratch};
use pmtiles::{Header, TileType};
use serde_json::Value;

/// Run `quilt-tiler [--sparse] -o <out> <inputs>` and require success.
fn run_quilt_tiler(out: &Path, inputs: &[PathBuf], sparse: bool) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_quilt-tiler"));
    if sparse {
        cmd.arg("--sparse");
    }
    let output = cmd
        .arg("-o")
        .arg(out)
        .args(inputs)
        .output()
        .expect("spawning quilt-tiler");
    assert!(
        output.status.success(),
        "quilt-tiler failed ({}):\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

fn pmtiles_header(path: &Path) -> Header {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    Header::try_from_bytes(bytes::Bytes::from(data))
        .unwrap_or_else(|e| panic!("{} has no valid PMTiles v3 header: {e}", path.display()))
}

fn read_json(path: &Path) -> Value {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()))
}

fn assert_bounds(metadata: &Value, expected: Box4) {
    let bounds: Vec<f64> = metadata["bounds"]
        .as_array()
        .unwrap_or_else(|| panic!("metadata bounds is not an array: {metadata}"))
        .iter()
        .map(|v| {
            v.as_f64()
                .unwrap_or_else(|| panic!("metadata bounds entry {v} is not a number"))
        })
        .collect();
    assert_eq!(bounds.len(), 4, "bounds {bounds:?}");
    for (got, want) in bounds.iter().zip(expected) {
        assert!(
            (got - want).abs() < 1e-9,
            "metadata bounds {bounds:?} != {expected:?}"
        );
    }
}

#[test]
fn raster_run_writes_png_archive_style_and_metadata() {
    let dir = Scratch::new("cli-raster");
    let inputs = common::write_raster_fixtures(&dir);
    let out = dir.join("r.pmtiles");

    run_quilt_tiler(&out, &inputs, true);

    assert_eq!(pmtiles_header(&out).tile_type, TileType::Png);

    let style = read_json(&dir.join("r.style.json"));
    let source = &style["sources"]["raster"];
    assert_eq!(source["type"], "raster", "style: {style}");
    assert!(
        source.get("sparse").is_none(),
        "raster sources fall back natively and must not carry `sparse`: {source}"
    );

    let metadata = read_json(&dir.join("r.metadata.json"));
    let union = [
        COARSE_BOX[0].min(FINE_BOX[0]),
        COARSE_BOX[1].min(FINE_BOX[1]),
        COARSE_BOX[2].max(FINE_BOX[2]),
        COARSE_BOX[3].max(FINE_BOX[3]),
    ];
    assert_bounds(&metadata, union);
    assert_eq!(metadata["minZoom"], COARSE_ZOOM - 2, "metadata: {metadata}");
    assert_eq!(metadata["maxZoom"], FINE_ZOOM, "metadata: {metadata}");
}

/// Vector cell box: the `CELL_EXTENT` corners and the `COVR` ring.
/// Every value is exact in `f32`, the `COVR` wire precision.
const VECTOR_BOX: Box4 = [10.0, 54.0, 11.0, 54.5];

fn record(rec_type: u16, payload: &[u8]) -> Vec<u8> {
    let len = u32::try_from(payload.len() + 6).expect("tiny test record");
    let mut out = rec_type.to_le_bytes().to_vec();
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Minimal decrypted SENC stream `oesu::parse_file` accepts, covering
/// [`VECTOR_BOX`]: server status (decrypted, licensed), version 201, cell
/// name, native scale, `CELL_EXTENT` (100) and one `COVR` (98) ring.
fn minimal_oesu() -> Vec<u8> {
    let [west, south, east, north] = VECTOR_BOX;
    let status: Vec<u8> = [1u16, 1, 1, 30, 0, 0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    // (lat, lon) corners in SW, NW, NE, SE order.
    let extent: Vec<u8> = [south, west, north, west, north, east, south, east]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let ring = [
        (south, west),
        (south, east),
        (north, east),
        (north, west),
        (south, west),
    ];
    let mut covr = u32::try_from(ring.len())
        .expect("tiny ring")
        .to_le_bytes()
        .to_vec();
    #[allow(clippy::cast_possible_truncation)] // values are exact in f32
    for (lat, lon) in ring {
        covr.extend_from_slice(&(lat as f32).to_le_bytes());
        covr.extend_from_slice(&(lon as f32).to_le_bytes());
    }

    let mut out = record(200, &status);
    out.extend(record(1, &201u16.to_le_bytes()));
    out.extend(record(2, b"TESTCELL\0"));
    out.extend(record(7, &1_000_000u32.to_le_bytes()));
    out.extend(record(100, &extent));
    out.extend(record(98, &covr));
    out
}

/// Run the vector pipeline once; returns `(style, metadata)`.
fn run_vector(dir: &Scratch, stem: &str, sparse: bool) -> (Value, Value) {
    let input = dir.write("cell.oesu", &minimal_oesu());
    let out = dir.join(&format!("{stem}.pmtiles"));

    run_quilt_tiler(&out, &[input], sparse);

    let header = pmtiles_header(&out);
    assert_eq!(header.tile_type, TileType::Mlt);
    let style = read_json(&dir.join(&format!("{stem}.style.json")));
    let metadata = read_json(&dir.join(&format!("{stem}.metadata.json")));
    assert_bounds(&metadata, VECTOR_BOX);
    assert!(
        metadata["minZoom"].as_u64() <= metadata["maxZoom"].as_u64()
            && metadata["maxZoom"].is_u64(),
        "metadata zoom range: {metadata}"
    );
    (style, metadata)
}

#[test]
fn vector_run_with_sparse_marks_style_source_sparse() {
    let dir = Scratch::new("cli-vector-sparse");
    let (style, _) = run_vector(&dir, "v", true);
    assert_eq!(style["sources"]["enc"]["sparse"], true, "style: {style}");
}

#[test]
fn vector_run_without_sparse_leaves_style_source_dense() {
    let dir = Scratch::new("cli-vector-dense");
    let (style, _) = run_vector(&dir, "v", false);
    let source = &style["sources"]["enc"];
    assert!(source.is_object(), "style has no `enc` source: {style}");
    assert!(
        source.get("sparse").is_none(),
        "dense style source: {source}"
    );
}
