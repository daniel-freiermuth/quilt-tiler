//! End-to-end: both binaries must refuse a `--safety-depth` / `--shoal-depth`
//! pair `MapLibre`'s DEPARE `step` expression can't accept, and do so before
//! loading any cells — `quilt-tiler` instead of after the slow tile
//! conversion, `tileserver` instead of starting and serving 500s from
//! `/style.json`.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

use std::path::PathBuf;
use std::process::{Command, Output};

/// A vector input that does not exist: a run that got past the depth check
/// would fail on it with a different error, so the expected message proves
/// the depth check is what stopped the run.
fn missing_vector_input() -> PathBuf {
    std::env::temp_dir().join("quilt-tiler-cli-test-missing-input.oesu")
}

fn run(bin: &str, extra_args: &[&str]) -> Output {
    Command::new(bin)
        .arg(missing_vector_input())
        .args(extra_args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {bin}: {e}"))
}

fn assert_rejected(bin: &str, extra_args: &[&str], expected: &str) {
    let out = run(bin, extra_args);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "{bin} {extra_args:?} must exit with an error, got {}; stderr:\n{stderr}",
        out.status
    );
    assert!(
        stderr.contains(expected),
        "{bin} {extra_args:?} must report {expected:?}; stderr:\n{stderr}"
    );
}

/// `(flags, depths as the error message prints them)`; the omitted flag
/// takes its default (safety 3, shoal 10).
const INVALID: &[(&[&str], &str)] = &[
    (
        &["--safety-depth", "12"],
        "safety depth (12) and shoal depth (10)",
    ),
    (
        &["--safety-depth", "10", "--shoal-depth", "10"],
        "safety depth (10) and shoal depth (10)",
    ),
    (
        &["--safety-depth", "NaN"],
        "safety depth (NaN) and shoal depth (10)",
    ),
];

#[test]
fn quilt_tiler_rejects_invalid_depths_before_converting() {
    let output = std::env::temp_dir().join("quilt-tiler-cli-test-invalid-depths.pmtiles");
    let output = output.to_str().expect("temp dir path is UTF-8");
    for (flags, expected) in INVALID {
        let mut args = vec!["-o", output];
        args.extend_from_slice(flags);
        assert_rejected(env!("CARGO_BIN_EXE_quilt-tiler"), &args, expected);
    }
}

#[test]
fn tileserver_rejects_invalid_depths_at_startup() {
    for (flags, expected) in INVALID {
        assert_rejected(env!("CARGO_BIN_EXE_tileserver"), flags, expected);
    }
}
