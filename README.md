# Quilt-tiler

Quilts cell-based charts like traditional seacharts with a few meaningful mixed zoom levels into a single tile layer.

Supported input formats: S57-ish (.000 (planned), decrypted oesu, osenc, GeoJSON-representations),
and `.rnc` raster cells.

Supported output formats: pmtiles carrying mvt (vector charts) or png (raster charts; mbtiles/mlt planned)
and an accompanying style.json + Signal K metadata.json.

## Usage

```
quilt-tiler -o chart.pmtiles <input-charts>          # vector: .oesu/.osenc cells
quilt-tiler -o chart.pmtiles <cells>/*.rnc            # raster: rnc cells
```

### Sparse output

`--sparse` (batch and `tileserver`) leaves out every tile whose charts already show all their detail at a
coarser zoom — most fine-zoom tiles over areas only coarse charts cover. The client rebuilds those from
the nearest stored ancestor, so it must treat a missing tile as "use the parent":

- Vector: `MapLibre` GL JS ≥ 6.17, whose experimental `sparse` source flag the generated style sets.
  Older clients and other renderers show nothing there.
- Raster: `MapLibre` falls back natively.
- Missing tiles must arrive as **404**: `tileserver` does; tileserver-gl does via the archive's
  `sparse` metadata; a `pmtiles://` protocol handler must reject them with a 404 status. martin answers
  204 (per the MapLibre PR discussion), which disables the fallback.

## Development

```
just check                 # fmt, clippy -D warnings, nextest, cargo-deny, cargo-machete — same as CI
just mutants src/bbox.rs   # mutation-test one file
just mutants-diff          # mutants introduced since github/master, incl. uncommitted changes
```

Pull requests must pass the "Mutation coverage of the diff" check: every mutant the diff
introduces has to be caught by a test. See `.cargo/mutants.toml` for what a surviving mutant
means and how to run the audit locally.
