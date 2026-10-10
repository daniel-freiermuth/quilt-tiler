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

## Development

```
just check                 # fmt, clippy -D warnings, nextest, cargo-deny, cargo-machete — same as CI
just mutants src/bbox.rs   # mutation-test one file
just mutants-diff          # mutants introduced since github/master, incl. uncommitted changes
```

Pull requests must pass the "Mutation coverage of the diff" check: every mutant the diff
introduces has to be caught by a test. See `.cargo/mutants.toml` for what a surviving mutant
means and how to run the audit locally.
