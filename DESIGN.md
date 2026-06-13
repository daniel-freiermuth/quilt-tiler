# Tile Pipeline Design

Living document — update as decisions are made.

---

## The Problem

OESU charts are multi-scale: the same geographic area is covered by several overlapping
charts at different native scales (e.g. 1:1 500 000 for the whole Baltic, 1:22 000 for
a harbour). A tile renderer must show the right scale at each zoom level — not four
coastlines stacked on top of each other.

The charts are already *cartographically authored* at their native scale. The 1:1.5M
coastline isn't a simplified version of the 1:22K coastline; it's a different drawing,
generalised differently by a human cartographer. This is the key invariant the pipeline
must respect.

---

## Current Pipeline (working, but fighting itself)

```
582 × .oesu  →  parse  →  quilt  →  110 × .geojson  →  tippecanoe  →  chart.mbtiles
```

**Quilting** (in `quilt.rs`) assigns per-feature `minzoom`/`maxzoom` based on
`native_scale` and coverage zones, so tippecanoe produces zoom-gated tiles.  
Fixed: `"tippecanoe"` must be a top-level GeoJSON Feature key (in `foreign_members`),
not inside `properties` — otherwise tippecanoe ignores it entirely.

**What it gets right:** each zoom level shows only the features from the appropriate
chart scale. Verified at z=10–14 near Hinsholmen.

**What it fights:** tippecanoe was not designed for multi-scale nautical charts.
Every hint has to be smuggled in as GeoJSON metadata and tippecanoe reserves the right
to ignore or override it. Intermediate GeoJSON files total several GB. The quilting
algorithm (BooleanOps per polygon, zone iteration) is the most complex part of the
codebase for what should be a simple invariant.

---

## Idea 1 — Finest-Only + Tippecanoe Simplification

**Principle:** for any geographic area, output only the features from the finest
available chart. Drop coarser chart features that are fully covered by a finer one.
Keep `minzoom` hints (from `native_scale`), drop `maxzoom` hints. Let tippecanoe
simplify the fine features across zoom levels.

**Where simplification helps here:** the finest chart features are authored for
high-zoom viewing. At low zoom, tippecanoe's Douglas-Peucker simplification produces
*something reasonable* — not cartographically authored, but acceptable for a viewer
whose primary use is at high zoom. tippecanoe's point-feature thinning (e.g. SOUNDG
at z=8) is also valuable and hard to replicate.

**Quilting simplifies to:** for each feature, test if it is fully inside a finer
chart's coverage zone → drop it. No BooleanOps needed (no polygon splitting at seams;
seams become tippecanoe's problem). Partial-coverage features stay in.

**Remaining weaknesses:**
- Simplified 1:22K harbour coastline at z=8 looks different from (worse than) an
  authored 1:1.5M coastline at z=8. Seams between chart coverage areas may be visible.
- Still needs tippecanoe in the pipeline. Still produces large intermediate GeoJSON.
- Cartographically wrong at low zoom for areas that have fine coverage everywhere
  (e.g. Swedish west coast with dense 1:50K charts).

**Best fit for:** a viewer where users mostly navigate at high zoom; low-zoom overview
is secondary; fast iteration matters more than cartographic perfection.

---

## Idea 2 — Direct Tile Writing (no tippecanoe)

**Principle:** each chart *owns* the zoom levels that its `native_scale` maps to.
Features go directly into those tiles. No intermediate GeoJSON. No tippecanoe.

```
582 × .oesu  →  parse  →  spatial index  →  write tiles  →  chart.pmtiles
```

**Why no simplification is needed:** the charts are already simplified by the
cartographer at their native scale. A 1:1.5M chart has exactly the right vertex
density for zoom 6–8. A 1:22K chart has the right density for zoom 14–15. The
cartographer did the simplification work upfront; we just route each chart to
its zoom band and the geometry arrives appropriate by construction.

For charts that span several zoom levels (e.g. a 1:100K chart at z=11–13), serve
identical geometry across the whole band. In practice vertex counts are already
low enough that tile sizes stay small — no further simplification needed there either.

**No tile clipping required (first pass):** MVT allows feature coordinates outside
the 0–4096 tile extent. Renderers (MapLibre GL) clip client-side. Skipping
server-side clipping removes the only algorithmically hard step from the pipeline.
Can be added later if tile sizes become a concern.

### Crate Stack

| Role | Crate | Version | Notes |
|---|---|---|---|
| MVT encoding | `fast-mvt` | 0.3.1 | Fuzz-tested, round-trip tested, spec-fixture tested; integer-first API matches the MVT spec exactly; by nyurik (maintainer of `martin-tile-utils`); `MvtTileBuilder → layer → feature`; takes `i32` tile-space coords. **MLT alternative: see below.** |
| Tile container | `pmtiles` | 0.23.0 | 187K downloads; Stadia Maps; `PmTilesWriter::new(Mvt).create(file) → add_tile → finalize`; built-in tile deduplication; XYZ y-axis (no inversion needed) |
| Tile math | `martin-tile-utils` | 0.7.3 | 41K downloads; from Martin itself; `tile_index`, `bbox_to_xyz`, `wgs84_to_webmercator` |
| Spatial index | `rstar` | 0.13.0 | **29M downloads**; georust; de-facto standard; MSRV 1.85 |
| TileJSON metadata | `tilejson` | 0.4.3 | 205K downloads; georust; simple format serialization |

**Not used now / parked for future:**
- `mvt` (0.13.0) — conservative fallback if `fast-mvt` proves problematic; f64 coordinate API adds an implicit rounding step that the spec doesn't have
- `mlt-core` (0.10.0) — MLT encoder/decoder; see dedicated section; full stack already in place
- `osmic-tiles` (0.1.1) — correct problem domain but "APIs will change" is a correctness-risk signal at this stage, not just an API-churn concern
- `versatiles_*` — full tile toolbox; useful to know about, not what we need

**Rejected:**
- `geozero` — docs build broken on 0.15.x; large dependency surface for a write-only path
- `rusqlite` / MBTiles — see format comparison below; PMTiles preferred
- `gpq-tiles-core` — 205 total downloads, LLM-generated
- `tile-grid` — redundant with `martin-tile-utils`; dormant since Oct 2024

### Coordinate Transform

`fast-mvt` takes `i32` tile-space coordinates. Projection path using `martin-tile-utils`:

```rust
const EXTENT: i32 = 4096;

fn to_tile_px(lon: f64, lat: f64, tile_bbox_mercator: [f64; 4]) -> (i32, i32) {
    let (x_m, y_m) = wgs84_to_webmercator(lon, lat);
    let w = tile_bbox_mercator[2] - tile_bbox_mercator[0];
    let h = tile_bbox_mercator[3] - tile_bbox_mercator[1];
    let px = ((x_m - tile_bbox_mercator[0]) / w * EXTENT as f64) as i32;
    let py = ((tile_bbox_mercator[3] - y_m) / h * EXTENT as f64) as i32;  // y=0 at north
    (px, py)
}
```

`xyz_to_bbox` returns WGS84 bounds; convert all four corners with `wgs84_to_webmercator`
to get the Mercator bbox. Out-of-range values (coordinates outside 0–4096) are allowed
by the MVT spec and clipped client-side by MapLibre GL.

### Pipeline Sketch

```
for each zoom z:
  for each chart where zoom_from_scale(native_scale) == z:
    (min_x, min_y, max_x, max_y) = bbox_to_xyz(chart.bbox, z)
    for each tile (x, y) in that range:
      features = spatial_index.query(tile_bbox)
      if features.is_empty(): continue
      tile_bytes = encode_mvt(features, tile_mercator_bbox)  // or encode_mlt
      pmtiles_writer.add_tile(z, x, y, tile_bytes)
```

No quilting. No BooleanOps. No GeoJSON. The zoom-scale assignment IS the quilting —
charts don't overlap in zoom space by definition (each scale owns its zoom band).

### SOUNDG Point Thinning

Without tippecanoe's density-based dropping, low-zoom tiles would have thousands of
depth soundings. Fix: at z < (chart_minzoom + 2), keep only the shallowest sounding
per 32×32 pixel grid cell. Priority queue by depth ascending; O(N) per tile.

### What Disappears

`quilt.rs` entirely, BooleanOps zone accumulation, GeoJSON serialisation (~several GB
of intermediate files), tippecanoe invocation.

### Remaining Trade-offs

- Label collision avoidance (tippecanoe does this for point layers) is gone.
  Probably not critical for nautical use; point density is low except SOUNDG.
- Antimeridian and polar edge cases untested. Coastal Sweden does not trigger these.

**Best fit for:** correct cartographic output, clean codebase, no external tooling
dependency, long-term maintainable.

---

## MBTiles vs PMTiles

Both are served by Martin. The choice affects write complexity and output size.

| | MBTiles 1.3 | PMTiles v3 |
|---|---|---|
| Container | SQLite 3 | Custom binary |
| Spec stability | Frozen since ~2017 | Stable since ~2022 |
| Tile deduplication | No | Yes — content-addressed |
| Y-axis | TMS (inverted: `y = (1<<z)-1-y_xyz`) | XYZ (y=0 at top, same as our loop) |
| Inspection | Any SQLite tool | Specialised tools only |
| Primary use case | Local file serving | Direct CDN/S3 range requests |
| Write crate | `rusqlite` direct (skip `mbtiles` crate — adds `sqlx`) | `pmtiles` 0.23.0 |

**Tile deduplication matters here:** at low zoom levels, empty-sea tiles from
different charts are byte-identical. PMTiles stores each unique blob once.
MBTiles stores duplicates verbatim (the `mbtiles` crate's `NormalizedSchema`
adds deduplication but also adds `sqlx` as a heavy async dependency — not worth it).

**The y-axis trap in MBTiles:** tiles must be stored with TMS y (`y_tms = (1<<z)-1-y_xyz`).
The mistake is silent — tiles appear at wrong positions and the rendering just looks
slightly wrong rather than crashing.

**Decision:** PMTiles. The CDN advantage doesn't apply locally, but deduplication
and the simpler y-axis convention both matter. Martin already serves it. The project
already has `chart.pmtiles`. If MBTiles is ever needed (e.g. for a tool that doesn't
support PMTiles), `rusqlite` + 3 SQL statements is the path — not the `mbtiles` crate.

## MVT vs MLT

Both are supported by our entire stack right now. This is a real choice, not a future one.

| | MVT (Mapbox Vector Tile) | MLT (MapLibre Tile) |
|---|---|---|
| Spec origin | Mapbox (2014) | MapLibre (stable Oct 2025) |
| Layout | Per-feature tag/value (row-oriented) | Column-oriented |
| Compression | gzip of protobuf | Custom lightweight codecs (FastPFor, varint, byte-RLE, delta, Morton/Hilbert) |
| Tile size | baseline | up to 6× smaller |
| SIMD decoding | no | yes (designed for it) |
| Typed properties | no — any key can have any type | yes — column type fixed per layer |
| Martin support | always | since 1.3.0 — we run **1.10.1** ✓ |
| MapLibre GL JS | always | since 5.12.0 — we run **5.24.0** ✓ |
| PMTiles | `TileType::Mvt` | `TileType::Mlt` — same crate, one word change |
| Encoder crate | `fast-mvt` | `mlt-core` |
| Write API | `MvtTileBuilder → layer → feature → finish()` | `TileLayer { Vec<TileFeature { Geometry<i32>, Vec<PropValue> } }` → `StagedLayer::encode()` |
| Geometry type | `i32` tile-space | `geo_types::Geometry<i32>` — same as our `geo` pipeline |
| Decoder testing | fuzz + spec fixtures | WASM + native integration tests |
| Reference encoder | Rust (`fast-mvt`) | Java; Rust (`mlt-core`) is secondary |
| Inspection tooling | any tile inspector | `mlt` CLI (TUI visualiser) |

**MLT property column model and S-57 data:** MLT requires one type per property per
layer — a feature that lacks a property gets a typed null. S-57 data already has
well-defined schemas per object class. This maps cleanly. It is stricter than MVT
(where the same key can have different types on different features), which is actually
a correctness benefit for typed nautical data.

**The `mlt` CLI tools** use `mlt-core` + `pmtiles` + `martin-tile-utils` + `rstar`
in its own dependency list — literally our planned stack. That's an independent
confidence signal that the combination works.

**The one honest gap:** the Rust encoder in `mlt-core` is secondary to the Java
reference encoder. The spec itself still has a few "Caution: unclear" annotations in
the metadata sections. For the 0x01 feature table tag (MVT-compatible mode, stable
spec section, no experimental markers), these don't apply — but it's worth noting.

**Decision:** start with MVT (`fast-mvt`) to get the pipeline working with a
battle-hardened encoder, then switch to MLT (`mlt-core`) once the tile output is
verified correct. The switch is mechanical: `TileType::Mvt` → `TileType::Mlt`,
`MvtTileBuilder` → `TileLayer/StagedLayer`, nothing else in the pipeline changes.

---


## Decision Guidance

| Criterion | Idea 1 | Idea 2 |
|---|---|---|
| Cartographic correctness at low zoom | ✗ simplified fine features | ✓ authored coarse features |
| Cartographic correctness at high zoom | ✓ | ✓ |
| Simplification needed | tippecanoe handles it (benefit) | not needed — already authored |
| Codebase complexity | lower than current | moderate; `quilt.rs` replaced by tile writer |
| Pipeline speed | still slow (GeoJSON + tippecanoe) | fast (direct) |
| External dependencies | tippecanoe required | none after build |
| Implementation risk | low | medium |

Idea 1 is a useful stepping stone. Idea 2 is the correct long-term architecture.

---

## Format Landscape

Three independent layers. Confusion comes from mixing them.

### Layer 1 — Tile Encoding (what is inside one tile blob)

| Format | Structure | Wire format | Notes |
|---|---|---|---|
| **MVT** | Row-oriented: each feature carries its own properties | Protobuf (PBF) | De-facto standard; spec owned by Mapbox |
| **MLT** | Column-oriented: all geometries together, all depths together | Custom binary (FastPFor, delta, Morton/Hilbert, byte-RLE) | MapLibre; up to 6× smaller; SIMD-decodable |

**PBF vs MVT:** Protobuf Binary Format (PBF) is Google's serialization format. MVT is a
*schema* that uses protobuf as its wire format. In tile URLs (`/z/x/y.pbf`), `.pbf` means
"MVT tile." Unrelated: OSM planet files (`.osm.pbf`) use the same serialization with a
completely different schema.

**Two compression layers in MVT:** (1) MVT geometry → protobuf binary — always. (2) Optional
gzip of those protobuf bytes inside the container — default on in tippecanoe; our previous
pipeline used `--no-tile-compression` to disable step 2. These are independent: the encoding
is always protobuf; the container compression is optional.

### Layer 2 — Tile Container (how a set of tiles is stored)


| Format | Storage | Y-axis | Deduplication | Use case |
|---|---|---|---|---|
| **MBTiles** | SQLite: `(zoom, x, y_tms, blob)` | TMS (inverted: `y = (1<<z)-1-y_xyz`) | No | Local serving, broad tooling |
| **PMTiles** | Single binary file with content-addressed directory | XYZ (y=0 at north) | Yes — by content hash | CDN/S3 range requests; also fine locally |
| **XYZ directory** | Plain files `z/x/y.pbf` | XYZ | No | Static hosting |

Container and encoding are independent. MVT or MLT blobs go into any container.
PMTiles has an explicit `TileType` field (`Mvt`, `Mlt`, `Png`, …).

### Layer 3 — Metadata and Serving

**TileJSON** — small JSON spec: name, bounds, center, min/maxzoom, `vector_layers`
(which layers exist, which properties they have, what types). Embedded in MBTiles
`metadata` table and PMTiles header. Martin serves it at `/{source}` as JSON.

**Martin** — HTTP tile server. Reads MBTiles or PMTiles, serves individual tile
blobs at `/{source}/{z}/{x}/{y}`, serves TileJSON at `/{source}`. Detects and
serves MLT tiles since v1.3.0 (we run v1.10.1).

### How it fits together

```
OESU charts
    │
    ▼
tile encoding
    ├── MVT blob  (fast-mvt crate → protobuf bytes)
    └── MLT blob  (mlt-core crate → custom binary bytes)
         │
         ▼
    PMTiles container  (pmtiles crate, TileType::Mvt or ::Mlt)
         │
         ▼
    Martin 1.10.1  (HTTP tile server)
         │
         ▼
    MapLibre GL JS 5.24.0  (renderer)
```

**GeoJSON** is a feature interchange format, not a tile format. It sits outside this stack.

**tippecanoe** is a bridge: GeoJSON/FlatGeobuf → MVT → MBTiles or PMTiles or XYZ directory.
It outputs MVT only — no MLT support. PMTiles output works (`-o file.pmtiles`), but the tile
blobs inside are still MVT. In Idea 2, tippecanoe is gone entirely.

---

## Zoom Scale Mapping

```
zoom = log2(ZOOM_K / native_scale)   where ZOOM_K = 559_082_264
```

| native_scale | zoom |
|---|---|
| 1 500 000 | ~8 |
| 300 000 | ~10 |
| 100 000 | ~12 |
| 50 000 | ~13 |
| 22 000 | ~14 |
| 5 000 | ~16 |
| 2 000 | ~18 |

A chart "owns" zoom levels from `zoom_from_scale(native_scale)` up to the zoom where
the next finer chart takes over (i.e., the next finer chart's `minzoom - 1`).

---

## Chart Format Notes

- Decrypted files: `oesenc-export/exported/*.oesu` (582 files, all SENC version 201)
- RNC raster charts use `.oernc` extension — completely different protocol, must be
  detected and rejected verbosely
- EXT records (84/85/86) not present in current files but must be handled
- SERVER_STATUS payload: 6 × u16 (serverStatus, decryptStatus, expireStatus,
  expireDaysRemaining, graceDaysAllowed, graceDaysRemaining)
- Authoritative reference: `o-charts_pi/src/Osenc.cpp` + `Osenc.h`
