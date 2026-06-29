//! Scans every .oesu file's raw COVR/NOCOVR records and classifies every
//! intersecting (COVR exterior, NOCOVR ring) pair by:
//!   - overlap_fraction = intersection_area(ext, nocovr) / ext_area
//!   - is_rect = nocovr's own area / its bbox area (1.0 = perfect rectangle)
//!   - shared_vertex_frac = fraction of nocovr's vertices that coincide
//!     (within 1e-4 deg) with an ext vertex (high = traces ext's boundary)
//! to characterize the actual distribution of COVR/NOCOVR relationships in
//! this corpus, rather than generalizing from 2 examples. Also flags any
//! cell/exterior that sees a *mix* of the 0%/100% patterns.
use geo::{Area, BooleanOps, Coord, Intersects, LineString, MultiPolygon, Polygon};
use std::collections::{BTreeMap, BTreeSet};

struct RawCell {
    covr: Vec<Polygon>,
    nocovr: Vec<Polygon>,
}

fn raw_rings(data: &[u8]) -> Option<RawCell> {
    let u16_at = |o: usize| u16::from_le_bytes([data[o], data[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
    let f32_at = |o: usize| f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
    if data.len() < 6 {
        return None;
    }
    let mut off = u32_at(2) as usize;
    let mut covr = Vec::new();
    let mut nocovr = Vec::new();
    while off + 6 <= data.len() {
        let rt = u16_at(off);
        let rl = u32_at(off + 2) as usize;
        if rl < 6 || off + rl > data.len() {
            break;
        }
        if rt == 98 || rt == 99 {
            let mut p = off + 6;
            if p + 4 > data.len() {
                break;
            }
            let count = u32_at(p) as usize;
            p += 4;
            let mut coords = Vec::with_capacity(count);
            for _ in 0..count {
                if p + 8 > data.len() {
                    break;
                }
                let lat = f32_at(p);
                let lon = f32_at(p + 4);
                p += 8;
                coords.push(Coord {
                    x: f64::from(lon),
                    y: f64::from(lat),
                });
            }
            if coords.len() >= 3 {
                let poly = Polygon::new(LineString::new(coords), vec![]);
                if rt == 98 {
                    covr.push(poly);
                } else {
                    nocovr.push(poly);
                }
            }
        }
        off += rl;
    }
    Some(RawCell { covr, nocovr })
}

fn bbox_area(ring: &LineString) -> f64 {
    let xs: Vec<f64> = ring.coords().map(|c| c.x).collect();
    let ys: Vec<f64> = ring.coords().map(|c| c.y).collect();
    let (xmin, xmax) = (
        xs.iter().cloned().fold(f64::INFINITY, f64::min),
        xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    let (ymin, ymax) = (
        ys.iter().cloned().fold(f64::INFINITY, f64::min),
        ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    (xmax - xmin) * (ymax - ymin)
}

fn shared_vertex_frac(a: &LineString, b: &LineString) -> f64 {
    if a.0.is_empty() {
        return 0.0;
    }
    let mut shared = 0;
    for pa in a.coords() {
        for pb in b.coords() {
            if (pa.x - pb.x).abs() < 1e-4 && (pa.y - pb.y).abs() < 1e-4 {
                shared += 1;
                break;
            }
        }
    }
    f64::from(shared) / a.0.len() as f64
}

fn stats(label: &str, v: &[(f64, f64, f64)]) {
    if v.is_empty() {
        println!("{label}: (none)");
        return;
    }
    let n = v.len() as f64;
    let (mut frac_min, mut frac_max, mut rect_min, mut rect_max, mut svf_min, mut svf_max) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    let (mut rect_sum, mut svf_sum) = (0.0, 0.0);
    let mut rect_low_count = 0;
    let mut svf_low_count = 0;
    for &(frac, rect, svf) in v {
        frac_min = frac_min.min(frac);
        frac_max = frac_max.max(frac);
        rect_min = rect_min.min(rect);
        rect_max = rect_max.max(rect);
        svf_min = svf_min.min(svf);
        svf_max = svf_max.max(svf);
        rect_sum += rect;
        svf_sum += svf;
        if rect < 0.99 {
            rect_low_count += 1;
        }
        if svf < 0.1 {
            svf_low_count += 1;
        }
    }
    println!(
        "{label}: n={} frac=[{:.6},{:.6}] is_rect=[{:.4},{:.4}] avg_rect={:.4} not_rect_count={} shared_vertex_frac=[{:.4},{:.4}] avg_svf={:.4} low_svf_count={}",
        v.len(),
        frac_min,
        frac_max,
        rect_min,
        rect_max,
        rect_sum / n,
        rect_low_count,
        svf_min,
        svf_max,
        svf_sum / n,
        svf_low_count
    );
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/mnt/fedora/home/daniel/segeln/oesenc-export/exported".into());

    let mut n_cells = 0u32;
    let mut n_pairs = 0u32;
    let mut n_no_nocovr = 0u32;
    let mut n_no_intersecting_pairs = 0u32;
    let mut overlap_buckets: BTreeMap<&str, u32> = BTreeMap::new();
    let mut interesting_partial: Vec<(String, usize, usize, f64, f64, f64)> = Vec::new();
    let mut zero_stats: Vec<(f64, f64, f64)> = Vec::new();
    let mut full_stats: Vec<(f64, f64, f64)> = Vec::new();
    let mut nocovr_count_hist: BTreeMap<usize, u32> = BTreeMap::new();
    let mut covr_count_hist: BTreeMap<usize, u32> = BTreeMap::new();
    let mut mixed_within_ext = 0u32;
    let mut mixed_across_exts = 0u32;

    for entry in std::fs::read_dir(&dir).expect("read_dir") {
        let entry = entry.expect("entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("oesu") {
            continue;
        }
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let Some(raw) = raw_rings(&data) else {
            continue;
        };
        n_cells += 1;
        *covr_count_hist.entry(raw.covr.len()).or_insert(0) += 1;
        *nocovr_count_hist.entry(raw.nocovr.len()).or_insert(0) += 1;

        if raw.nocovr.is_empty() {
            n_no_nocovr += 1;
            continue;
        }

        let mut any_pair = false;
        let mut per_ext_buckets: BTreeMap<usize, Vec<&str>> = BTreeMap::new();
        for (ei, ext) in raw.covr.iter().enumerate() {
            for (ni, int) in raw.nocovr.iter().enumerate() {
                if !ext.intersects(int) {
                    continue;
                }
                any_pair = true;
                n_pairs += 1;
                let ext_mp = MultiPolygon::new(vec![ext.clone()]);
                let int_mp = MultiPolygon::new(vec![int.clone()]);
                let ext_area = ext.unsigned_area();
                let overlap = ext_mp.intersection(&int_mp).unsigned_area();
                let frac = if ext_area > 0.0 {
                    overlap / ext_area
                } else {
                    0.0
                };
                let nocovr_rect = {
                    let a = int.unsigned_area();
                    let b = bbox_area(int.exterior());
                    if b > 0.0 { a / b } else { 0.0 }
                };
                let svf = shared_vertex_frac(int.exterior(), ext.exterior());

                let bucket = if frac < 0.001 {
                    "0%"
                } else if frac > 0.999 {
                    "100%"
                } else {
                    "PARTIAL"
                };
                *overlap_buckets.entry(bucket).or_insert(0) += 1;
                per_ext_buckets.entry(ei).or_default().push(bucket);

                if bucket == "0%" {
                    zero_stats.push((frac, nocovr_rect, svf));
                } else if bucket == "100%" {
                    full_stats.push((frac, nocovr_rect, svf));
                } else {
                    interesting_partial.push((
                        path.to_string_lossy().to_string(),
                        ei,
                        ni,
                        frac,
                        nocovr_rect,
                        svf,
                    ));
                }
            }
        }
        for (ei, buckets) in &per_ext_buckets {
            let unique: BTreeSet<&&str> = buckets.iter().collect();
            if unique.len() > 1 {
                mixed_within_ext += 1;
                println!(
                    "MIXED within one exterior: {} COVR[{ei}] sees {:?}",
                    path.display(),
                    buckets
                );
            }
        }
        if per_ext_buckets.len() > 1 {
            let all: BTreeSet<&str> = per_ext_buckets.values().flatten().copied().collect();
            if all.len() > 1 {
                mixed_across_exts += 1;
                println!(
                    "MIXED across exteriors in one cell: {} sees {:?}",
                    path.display(),
                    all
                );
            }
        }
        if !any_pair {
            n_no_intersecting_pairs += 1;
        }
    }

    println!("\ncells scanned: {n_cells}");
    println!("cells with zero NOCOVR records: {n_no_nocovr}");
    println!("cells with NOCOVR but no intersecting COVR/NOCOVR pair: {n_no_intersecting_pairs}");
    println!("total intersecting (COVR,NOCOVR) pairs: {n_pairs}");
    println!(
        "cells with mixed pattern within one exterior (multiple NOCOVR, different buckets): {mixed_within_ext}"
    );
    println!(
        "cells with mixed pattern across exteriors (same cell, different buckets): {mixed_across_exts}"
    );
    println!("\ncovr_count histogram: {covr_count_hist:?}");
    println!("nocovr_count histogram: {nocovr_count_hist:?}");
    println!("\noverlap_fraction buckets:");
    for (k, v) in &overlap_buckets {
        println!("  {k}: {v}");
    }
    println!();
    stats("0% bucket  (adjacent/complement)", &zero_stats);
    stats("100% bucket (full erasure)      ", &full_stats);
    println!(
        "\npartial-overlap pairs (neither 0% nor 100%) — candidates for genuine 'true interior hole': {}",
        interesting_partial.len()
    );
    for (path, ei, ni, frac, rect, svf) in &interesting_partial {
        println!(
            "  {path}  COVR[{ei}]/NOCOVR[{ni}]  overlap_frac={frac:.6}  nocovr_is_rect={rect:.4}  shared_vertex_frac={svf:.4}"
        );
    }
}
